import os
import re
import logging
import time
import asyncio
from dataclasses import dataclass, field
from typing import AsyncIterator, Dict, Any
from typing import Tuple, List

from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from azure.ai.projects.aio import AIProjectClient as AioAIProjectClient
from azure.identity.aio import DefaultAzureCredential as AioDefaultAzureCredential
from app.config import settings
# from dotenv import load_dotenv

# 環境変数読み込み
# load_dotenv()
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

@dataclass
class _AsyncSession:
    thread_id: str
    injected_count: int = 0
    last_used: float = field(default_factory=lambda: time.time())

class FoundryAgentService:
    def __init__(self):
        # 環境変数から設定取得
        self.endpoint = settings.azure_ai_foundry_endpoint
        self.agent_id = settings.azure_ai_foundry_agent_id

        if not self.endpoint or not self.agent_id:
            raise ValueError("AZURE_AI_FOUNDRY_ENDPOINT または AZURE_AI_FOUNDRY_AGENT_ID が未設定です。")

        # 認証（DefaultAzureCredentialでAAD認証 or マネージドID）
        # 同期クライアント（非ストリーム・フォールバック用）
        self.client = AIProjectClient(endpoint=self.endpoint, credential=DefaultAzureCredential())
        self.agent = self.client.agents.get_agent(agent_id=self.agent_id)
        # self.thread = None  # 文脈を保持するスレッドID
        self.max_history = 5
        
        # 非同期クライアント（ストリーム用）
        self._aclient = None
        self._acred = None
        # セッション管理（session_id -> _AsyncSession）
        self._sessions: dict[str, _AsyncSession] = {}
        self._session_ttl_sec = 60 * 60  # 1時間
        
        logger.info(f"[FoundryAgentService] Agent {self.agent.name} ({self.agent.id}) loaded")
    
    def query_agent(self, user_input: str) -> Tuple[str, List[str]]:
        """
        単発クエリ用: スレッドを作成して1回だけ問い合わせ
        """
        thread = self.client.agents.threads.create()
        self.client.agents.messages.create(thread_id=thread.id, role="user", content=user_input)

        run = self.client.agents.runs.create(thread_id=thread.id, agent_id=self.agent.id)
        while run.status in ["queued", "in_progress", "requires_action"]:
            time.sleep(1)
            run = self.client.agents.runs.get(thread_id=thread.id, run_id=run.id)

        return self._extract_answer_and_sources(thread.id)

    def search_with_bing(self, session_id: str, chat_history: list[dict]) -> Dict[str, Any]:
        """
        session_id ごとに同じ thread を使い回し、差分だけを投入してから Run 実行 → 回答+引用を返す。
        """
        # セッション（thread）取得/作成
        sess = self._get_or_create_session_sync(session_id)
        
        # フル履歴を渡して OK（未投入分だけ追加される）
        self._inject_delta_sync(sess, chat_history)

        run = self.client.agents.runs.create(thread_id=sess.thread_id, agent_id=self.agent.id)
        while run.status in ["queued", "in_progress", "requires_action"]:
            time.sleep(0.8)
            run = self.client.agents.runs.get(thread_id=sess.thread_id, run_id=run.id)

        # 回答と引用元を抽出
        answer, idx_to_url = self._extract_answer_and_sources(sess.thread_id)

        # 引用元を本文末尾に追加
        
        # 引用を [docN] の順で配列化
        citations = [{"title": u, "content": "", "filePath": u, "url": u}
                    for _, u in sorted(idx_to_url.items(), key=lambda x: x[0])]
        # sources_text = ""
        # if sources:
        #     sources_text = "\n\n---\n**引用元:**\n"
        #     sources_text += "\n".join([f"- [{i+1}] {url}" for i, url in enumerate(sources)])
        content = answer

        # FastAPI用のレスポンス形式で返却
        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": content,
                    "context": {"citations": citations}
                }
            }]
        }
    
    def _format_answer_with_citations(self, text_data: dict) -> Tuple[str, dict]:
        """
        本文中の Azure 形式の引用タグ（例: , ）を [docN] に正規化し、
        N -> URL のマップを返す。
        """
        raw_text = text_data["value"]
        annotations = text_data.get("annotations", [])

        # まず本文からタグを検出して出現順に doc 番号を採番
        tag_re = re.compile(r"【(\d+(?::\d+)?)\s*[^】]*?source】")  # †有無に依存しないように後半はゆるめに
        tag_order: list[str] = []
        tag_to_idx: dict[str, int] = {}

        def _sub(m: re.Match) -> str:
            full = m.group(0)  # 例: 
            if full not in tag_to_idx:
                tag_to_idx[full] = len(tag_order) + 1
                tag_order.append(full)
            return f"[doc{tag_to_idx[full]}]"

        formatted_text = tag_re.sub(_sub, raw_text)

        # annotations から tag -> URL を作る（text が無ければ start/end から切り出す）
        tag_to_url: dict[str, str] = {}
        for ann in annotations:
            if ann.get("type") != "url_citation":
                continue
            url = (ann.get("url_citation") or {}).get("url")
            if not url:
                continue
            tag_text = ann.get("text")
            if not tag_text:
                try:
                    s = ann.get("start_index"); e = ann.get("end_index")
                    if isinstance(s, int) and isinstance(e, int):
                        tag_text = raw_text[s:e]
                except Exception:
                    pass
            if tag_text:
                tag_to_url[tag_text] = url

        # [docN] の順に URL を並べた idx->url マップを作る
        idx_to_url: dict[int, str] = {}
        for tag in tag_order:
            if tag in tag_to_url:
                idx = tag_to_idx[tag]
                idx_to_url[idx] = tag_to_url[tag]

        return formatted_text, idx_to_url

    def _extract_answer_and_sources(self, thread_id: str) -> Tuple[str, Dict[int, str]]:
        """
        最新のエージェントの回答だけを返す
        """
        # 全メッセージを取得して逆順にループ
        messages = list(self.client.agents.messages.list(thread_id=thread_id))

        latest_answer = ""
        idx_to_url: Dict[int, str] = {}

        for message in messages:  # 最新から走査
            if message.role.name.lower() == "agent":
                for part in message.content:
                    if part["type"] == "text":
                        formatted_answer, idx_to_url = self._format_answer_with_citations(part["text"])
                        latest_answer = formatted_answer
                break  # 最新の agent メッセージを取得したらループ終了

        return latest_answer.strip(), idx_to_url
    
    # ---------- 非同期クライアント（ストリーミング） ----------
    async def _ensure_async(self):
        if self._aclient is None:
            self._acred = AioDefaultAzureCredential()
            self._aclient = AioAIProjectClient(
                endpoint=self.endpoint,
                credential=self._acred
            )
            try:
                import azure.ai.projects as _p, azure.ai.agents as _a
                logger.info(f"[versions] projects={getattr(_p,'__version__','?')} agents={getattr(_a,'__version__','?')}")
                logger.info(f"[runs ops] has create_stream={hasattr(self._aclient.agents.runs,'create_stream')} "
                            f"stream={hasattr(self._aclient.agents.runs,'stream')}")
            except Exception:
                pass
    
    async def aclose(self):
        try:
            if self._aclient:
                await self._aclient.close()
            if self._acred:
                await self._acred.close()
        except Exception:
            pass
    
    async def search_with_bing_stream(self, session_id: str, chat_history: list[dict]) -> AsyncIterator[Dict[str, Any]]:
        """
        現行 SDK では runs.stream は同期API。
        → スレッドで回し、async 側へ asyncio.Queue でブリッジする。
        yield 形式:
          {"type":"token","delta":"..."}
          {"type":"citations","items":[{title,content,filePath,url}...]}
          {"type":"done"} / {"type":"error","message":"..."}
        """
        # 1) 同期クライアント側のセッションを使う（差分だけ投入）
        sess = self._get_or_create_session_sync(session_id)
        self._inject_delta_sync(sess, chat_history)

        loop = asyncio.get_running_loop()
        q: asyncio.Queue[dict] = asyncio.Queue()

        def _put(msg: dict):
            asyncio.run_coroutine_threadsafe(q.put(msg), loop)

        def _worker():
            try:
                try:
                    from azure.ai.agents.models import MessageDeltaChunk  # あれば使う
                except Exception:
                    MessageDeltaChunk = None  # type: ignore
                
                # ★ ここから追加：タグ採番のローカル状態と置換関数
                import re
                tag_re = re.compile(r"【(\d+(?::\d+)?)\s*[^】]*?source】")
                tag_order: list[str] = []          # 出現順のタグ原文（例: ""）
                tag_to_idx: dict[str, int] = {}    # タグ原文 -> doc番号（1..）
                ann_tag_to_url: dict[str, str] = {}  # タグ原文 -> URL（annotationsから）

                def norm_text_to_doc(text: str) -> str:
                    def _sub(m: re.Match) -> str:
                        full = m.group(0)
                        if full not in tag_to_idx:
                            tag_to_idx[full] = len(tag_order) + 1
                            tag_order.append(full)
                        return f"[doc{tag_to_idx[full]}]"
                    return tag_re.sub(_sub, text)

                seen_types = {}           # デバッグ用: 受信した event 名カウント
                emitted_any = False       # 1 文字でも出したか
                # citations_buf = []        # 途中で見つけた citation URL 一時バッファ

                with self.client.agents.runs.stream(
                    thread_id=sess.thread_id,
                    agent_id=self.agent.id
                ) as stream:
                    for ev in stream:
                        # ev はタプル or オブジェクトのどちらか
                        if isinstance(ev, tuple) and len(ev) >= 2:
                            event_type, event_data = ev[0], ev[1]
                        else:
                            event_type = getattr(ev, "event", None) or getattr(ev, "type", None)
                            event_data = getattr(ev, "data", None) or ev

                        etype = (str(event_type) if event_type is not None else "").lower()
                        seen_types[etype] = seen_types.get(etype, 0) + 1

                        # ---- 1) トークン増分 ----
                        if MessageDeltaChunk and isinstance(event_data, MessageDeltaChunk):
                            text = getattr(event_data, "text", None) or ""
                            if text:
                                _put({"type": "token", "delta": norm_text_to_doc(text)})
                                emitted_any = True
                            continue

                        # 形で判定（data.delta.content[*].text.value）
                        delta = getattr(event_data, "delta", None)
                        if delta is None and isinstance(event_data, dict):
                            delta = event_data.get("delta")
                        if delta is not None:
                            try:
                                contents = getattr(delta, "content", None) or delta.get("content", [])
                            except Exception:
                                contents = []
                            for part in contents:
                                ptype = part.get("type")
                                if ptype in ("output_text_delta", "text_delta", "text", "response.output_text.delta"):
                                    val = (part.get("text", {}) or {}).get("value") or part.get("value") or ""
                                    if val:
                                        _put({"type": "token", "delta": norm_text_to_doc(val)})
                                        emitted_any = True

                        # 一部の SDK は 'message.delta' という名前
                        if etype in ("message.delta",):
                            val = getattr(event_data, "text", None) or ""
                            if not val and isinstance(event_data, dict):
                                val = (event_data.get("text", {}) or {}).get("value", "")
                            if val:
                                _put({"type": "token", "delta": norm_text_to_doc(val)})
                                emitted_any = True

                        # ---- 2) message 完了で最終本文を拾う（breakしない）----
                        if etype.endswith("message.completed"):
                            try:
                                content = getattr(event_data, "content", None)
                                if content is None and isinstance(event_data, dict):
                                    content = event_data.get("content", [])
                                text_buf = []
                                for part in content or []:
                                    if part.get("type") in ("text", "output_text"):
                                        tv = (part.get("text", {}) or {}).get("value") or part.get("value") or ""
                                        if tv:
                                            text_buf.append(tv)
                                        for a in (part.get("text", {}) or {}).get("annotations", []):
                                            if a.get("type") == "url_citation":
                                                url = a["url_citation"].get("url", "")
                                                if not url: 
                                                    continue
                                                tag_text = a.get("text")
                                                if not tag_text:
                                                    try:
                                                        s = a.get("start_index"); e = a.get("end_index")
                                                        tv = (part.get("text", {}) or {}).get("value", "")
                                                        if isinstance(s, int) and isinstance(e, int):
                                                            tag_text = tv[s:e]
                                                    except Exception:
                                                        pass
                                                if tag_text:
                                                    ann_tag_to_url[tag_text] = url
                                # すでに delta で出している場合は、ここで全文を再送しない。
                                if text_buf and not emitted_any:
                                    _put({"type": "token", "delta": norm_text_to_doc("".join(text_buf))})
                                    emitted_any = True
                            except Exception:
                                pass
                            # run.completed までは続行
                            continue

                        # ---- 3) エラー/終了判定 ----
                        if "error" in etype:
                            _put({"type": "error", "message": str(event_data)})
                            return

                        if etype in ("done", "run.completed"):
                            break
                        # run.step.completed などは無視して続行

                # デバッグ: 受け取ったイベント種別を 1 行ログ
                try:
                    logger.info(f"[foundry stream events] {seen_types}")
                except Exception:
                    pass

                # ストリーム後：引用を送る（まずは途中で拾った URL、無ければ最終メッセージから抽出）
                try:
                    items = []
                    for tag in tag_order:
                        url = ann_tag_to_url.get(tag)
                        if url:
                            items.append({"title": url, "content": "", "filePath": url, "url": url})
                    # annotations で取得できなかった場合は、最終メッセージ解析の番号順で補完
                    if not items:
                        try:
                            _, idx_to_url = self._extract_answer_and_sources(sess.thread_id)
                            for _, url in sorted(idx_to_url.items(), key=lambda x: x[0]):
                                items.append({"title": url, "content": "", "filePath": url, "url": url})
                        except Exception:
                            pass
                    if items:
                        _put({"type": "citations", "items": items})
                except Exception:
                    pass

                # もし 1 文字も出せていない場合は、同期 API で最終回答を取って疑似ストリーム
                if not emitted_any:
                    try:
                        ans, _ = self._extract_answer_and_sources(sess.thread_id)
                        for ch in self._chunk_text(ans, 120):
                            _put({"type": "token", "delta": ch})
                    except Exception:
                        pass

            except Exception as e:
                _put({"type": "error", "message": str(e)})
            finally:
                _put({"type": "done"})

        # 2) バックグラウンドで同期ストリームを実行
        loop.run_in_executor(None, _worker)

        # 3) async 側はキューから受けてそのまま yield
        while True:
            msg = await q.get()
            yield msg
            if msg.get("type") in ("done", "error"):
                break
    
    def _gc_sessions(self):
        now = time.time()
        gone = [sid for sid, s in self._sessions.items() if now - s.last_used > self._session_ttl_sec]
        for sid in gone:
            self._sessions.pop(sid, None)
    
    def _get_or_create_session_sync(self, session_id: str) -> _AsyncSession:
        """同期クライアント用: セッション取得/作成"""
        self._gc_sessions()
        s = self._sessions.get(session_id)
        if s:
            s.last_used = time.time()
            return s
        th = self.client.agents.threads.create()
        s = _AsyncSession(thread_id=th.id, injected_count=0, last_used=time.time())
        self._sessions[session_id] = s
        return s

    def _inject_delta_sync(self, sess: _AsyncSession, chat_history: list[dict]):
        """
        同期クライアントで“未投入の分だけ”を thread に追加。
        ※ フル履歴を渡してOK。差分は injected_count で管理。
        """
        start = sess.injected_count
        for msg in chat_history[start:]:
            role = msg.get("role", "user")
            if role not in ("user", "assistant"):
                role = "user"
            self.client.agents.messages.create(
                thread_id=sess.thread_id,
                role=role,
                content=msg.get("content", "")
            )
        sess.injected_count = len(chat_history)
        sess.last_used = time.time()

    async def _get_or_create_session(self, session_id: str) -> _AsyncSession:
        await self._ensure_async()
        self._gc_sessions()
        s = self._sessions.get(session_id)
        if s:
            s.last_used = time.time()
            return s
        # 新しい thread を作成
        th = await self._aclient.agents.threads.create()
        s = _AsyncSession(thread_id=th.id, injected_count=0, last_used=time.time())
        self._sessions[session_id] = s
        return s

    async def _inject_delta(self, sess: _AsyncSession, chat_history: list[dict]):
        """セッションに未投入の履歴だけを Agent thread に追加"""
        start = sess.injected_count
        for msg in chat_history[start:]:
            role = msg.get("role", "user")
            if role not in ("user", "assistant"):
                role = "user"  # system 等は user 扱いに寄せる
            await self._aclient.agents.messages.create(
                thread_id=sess.thread_id,
                role=role,
                content=msg.get("content", "")
            )
        sess.injected_count = len(chat_history)
        sess.last_used = time.time()

    @staticmethod
    def _chunk_text(text: str, n: int) -> list[str]:
        """n文字程度で安全に分割（語の途中で切れにくい簡易実装）"""
        out, buf = [], []
        count = 0
        for ch in text:
            buf.append(ch)
            count += 1
            if ch in "。．.!?\n" or count >= n:
                out.append("".join(buf))
                buf, count = [], 0
        if buf:
            out.append("".join(buf))
        return out

# Create singleton instance
foundry_agent_service = FoundryAgentService()