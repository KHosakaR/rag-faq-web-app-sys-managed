"""
FastAPI RAG application using Azure OpenAI and Azure AI Search

This sample app demonstrates how to build a RAG (Retrieval Augmented Generation) application
that combines the power of Azure OpenAI with Azure AI Search to create an AI assistant
that answers questions based on your own data.

Key components:
1. FastAPI web framework for the backend API
2. Azure OpenAI for AI chat completions
3. Azure AI Search for document retrieval
4. Pydantic for configuration and data validation
5. Bootstrap and JavaScript for the frontend UI
"""
import os
import logging
import uvicorn
import json
from uuid import uuid4
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse
from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, StreamingResponse
import asyncio

from app.models.chat_models import ChatRequest, ChatMessage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Import the RAG chat service after logging to capture any initialization logs
from app.services.rag_chat_service import rag_chat_service
from app.services.foundry_agent_service import foundry_agent_service
from app.services.chat_filter import chat_filter
from app.services.blob_sas_service import blob_sas_service, SasResult
from app.config import settings

# クリックで生成するだけなので短めで十分（例: 10分）
DEFAULT_TTL_MIN = settings.blob_sas_ttl_min

# 生成頻度を下げるための超シンプルなメモリキャッシュ
# キー: (path, ttl_min) -> 値: {"result": SasResult, "expires": datetime}
_sas_cache: dict[tuple[str, int], dict] = {}

# セキュリティ: 許可するコンテナのホワイトリスト（必要に応じて環境変数化）
ALLOWED_CONTAINERS = {"faq", "faq-outpput"}  # ここは実環境に合わせて

ACCOUNT_URL = settings.blob_account_url

def is_allowed_host(path_or_url: str) -> bool:
    if not ACCOUNT_URL:
        return False
    try:
        if path_or_url.startswith("http"):
            return urlparse(path_or_url).netloc == urlparse(ACCOUNT_URL).netloc
        # 'container/blob' 形式は自アカウント扱い
        return True
    except Exception:
        return False

def is_allowed_container(path_or_url: str) -> bool:
    # URLでも 'container/blob' でも先頭のコンテナ名を取り出して判定
    try:
        if path_or_url.startswith("http"):
            u = urlparse(path_or_url)
            parts = [p for p in u.path.split("/") if p]
            container = parts[0] if parts else ""
        else:
            container = path_or_url.split("/", 1)[0]
        return container in ALLOWED_CONTAINERS
    except Exception:
        return False

# Create FastAPI app
app = FastAPI(
    title="FastAPI RAG with Azure OpenAI and Azure AI Search",
    description="A FastAPI application that demonstrates retrieval augmented generation using Azure OpenAI and Azure AI Search.",
    version="1.0.0",
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up template directory
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    """
    Serve the main chat interface
    """
    acc_host = urlparse(settings.blob_account_url).netloc if settings.blob_account_url else ""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "blob_account_host": acc_host
    })

@app.get("/api/blob/sas", response_model=SasResult)
async def get_blob_sas(
    path: str = Query(..., description="Blob のフルURL または 'container/blob' 形式のパス"),
    ttl_min: int | None = Query(None, ge=1, le=60, description="SAS の有効期限（分）。未指定は既定値。"),
):
    """
    Search の source_path（プライベートURL）から、短命の Read-only SAS URL を発行して返す。
    - 非同期実装（内部の同期SDK呼び出しはスレッドプールへ）
    - 簡易キャッシュで生成回数を低減
    - ホワイトリストで意図しないコンテナをブロック
    """
    try:
        if not is_allowed_host(path):
            raise HTTPException(status_code=403, detail="許可されていないストレージアカウントです。")
        if not is_allowed_container(path):
            raise HTTPException(status_code=403, detail="このコンテナは許可されていません。")

        ttl = ttl_min or DEFAULT_TTL_MIN
        key = (path, ttl)

        # キャッシュヒット
        now = datetime.now(timezone.utc)
        if key in _sas_cache:
            item = _sas_cache[key]
            if item["expires"] > now:
                return item["result"]

        # 生成（同期SDKをスレッドで）
        res: SasResult = await run_in_threadpool(blob_sas_service.get_sas_url, path, ttl)

        # キャッシュ（SAS有効期限の手前で失効させる: 例 90%）
        expire_at = now + timedelta(minutes=int(ttl * 0.9))
        _sas_cache[key] = {"result": res, "expires": expire_at}

        return res

    except HTTPException:
        raise
    except Exception as e:
        # 具体的な理由はログに残しつつ、表には出しすぎない
        logger.exception("Failed to issue SAS")
        raise HTTPException(status_code=400, detail=f"SAS発行に失敗: {e}")


@app.post("/api/chat/completion")
async def chat_completion(request: Request, chat_request: ChatRequest):
    """
    Process a chat completion request with RAG capabilities
    
    This endpoint:
    1. Receives the chat history from the client
    2. Passes it to the RAG service for processing
    3. Returns AI-generated responses with citations
    4. Handles errors gracefully with user-friendly messages
    """
    try:
        if not chat_request.messages:
            raise HTTPException(status_code=400, detail="Messages cannot be empty")
        
        ## NG質問ロジック処理
        # 最新のユーザーの質問を抽出（最後のuserロールのメッセージ）
        user_messages = [msg.content for msg in chat_request.messages if msg.role == "user"]
        latest_user_question = user_messages[-1] if user_messages else ""

        # NG質問フィルタリング（講義外・悪用系の質問を遮断）
        is_blocked, reason = chat_filter.is_blocked_input(latest_user_question)
        if is_blocked:
            return {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": f"申し訳ありません。この質問は不適切な内容であると見做されたため、お答えすることができません。\nほかの内容でお願いいたします。\n理由: {reason}"
                        # "content": "申し訳ありません。この質問は講義とは関係のない内容（設定ワードと類似）と判断されたため、お答えできません。"
                    }
                }]
            }
        
        # Get chat completion from RAG service
        response, is_low_confidence = await rag_chat_service.get_chat_completion(chat_request.messages)
        
        if is_low_confidence:
            logger.info("信頼度判定 → Bing Grounding Agent にフォールバック")
            
            # セッションIDの決め方（例）: Cookie / Header / IP など
            session_id = request.headers.get("x-session-id") or request.cookies.get("sid")
            if not session_id:
                session_id = str(uuid4())
            history_data = [{"role": m.role, "content": m.content} for m in chat_request.messages]
            response = foundry_agent_service.search_with_bing(session_id, history_data)
            
            # 案内文を付与(太字)
            notice_message = (
                "**※以下はWeb検索による回答です。**\n\n"
            )

            if "choices" in response and len(response["choices"]) > 0:
                original_content = response["choices"][0]["message"]["content"]
                response["choices"][0]["message"]["content"] = notice_message + original_content
            
            response.setdefault("citations", [])
        
        return response
        
    except Exception as e:
        error_str = str(e).lower()
        logger.error(f"Error in chat completion: {str(e)}")
        
        # Handle specific error types with friendly messages
        if "rate limit" in error_str or "capacity" in error_str or "quota" in error_str:
            return {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "The AI service is currently experiencing high demand. Please wait a moment and try again."
                    }
                }]
            }
        else:
            # Return a standard error response for all other errors
            return {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": f"An error occurred: {str(e)}"
                    }
                }]
            }

@app.post("/api/chat/stream")
async def chat_stream(request: Request):
    """
    SSE ストリーミングで RAG 応答を逐次返す。
    - event: token     { delta, cursor }
    - event: citations { items }
    - event: done      { content_len, is_low }
    - event: error     { message }
    """
    try:
        print("stream now")
        payload = await request.json()
        # フロントは {history:[...]} を送る想定。互換のため {messages:[...]} も受ける
        raw_history = payload.get("history") or payload.get("messages") or []
        history: list[ChatMessage] = []
        for m in raw_history:
            if isinstance(m, dict):
                # role / content が欠けていても落ちないようにフォールバック
                history.append(ChatMessage(role=m.get("role", "user"), content=str(m.get("content", ""))))
            else:
                # すでに ChatMessage の形ならそのまま
                history.append(m)
                

        # NG質問フィルタ（最後の user 発話を対象）
        latest_user = ""
        for m in reversed(history):
            if m.role == "user":
                latest_user = (m.content or "")
                break

        is_blocked, reason = chat_filter.is_blocked_input(latest_user)

        async def gen():
            # ブロック時はSSEで即返して終了
            if is_blocked:
                denial = f"申し訳ありません。この質問は不適切または講義外のため回答できません。理由: {reason}"
                yield rag_chat_service._sse_event("token", {"delta": denial, "cursor": len(denial)})
                yield rag_chat_service._sse_event("citations", {"items": []})
                yield rag_chat_service._sse_event("done", {"content_len": len(denial), "is_low": True})
                return
            
            # セッションID（completion と同じ規約）
            session_id = request.headers.get("x-session-id") or request.cookies.get("sid") or str(uuid4())
            
            # --- RAG ストリームを取り込みつつ判断 ---
            rag_citations = []
            rag_low = False
            
            # 通常：RAGストリームをそのまま中継
            async for ev in rag_chat_service.stream_chat_completion(history):
                if await request.is_disconnected():
                    return
                
                # ここで SSE の 1 イベントを解析
                try:
                    text = ev.decode("utf-8")
                    event = None
                    data = None
                    for line in text.splitlines():
                        if line.startswith("event:"):
                            event = line[6:].strip()
                        elif line.startswith("data:"):
                            data = json.loads(line[5:].strip())
                except Exception:
                    # パース失敗は素通し
                    yield ev
                    continue

                if event == "token":
                    # 本文はそのまま中継
                    yield ev

                elif event == "citations":
                    # RAG の citations は保持のみ（送らない）
                    rag_citations = data.get("items", [])

                elif event == "done":
                    rag_low = bool(data.get("is_low"))
                    if not rag_low:
                        # 通常完了：ここで RAG の citations を送り、RAG の done を中継
                        yield rag_chat_service._sse_event("citations", {"items": rag_citations})
                        yield ev
                        return
                    else:
                        # フロントに「低信頼だったよ」を通知（確定せずにUIをクリアさせる）
                        yield rag_chat_service._sse_event("done", {"content_len": data.get("content_len", 0), "is_low": True})
                        # 低信頼 → Bing に切替（RAG の done は流さない）
                        notice = "該当する資料が見当たらなかったので、Web検索による回答を生成しています。\n\n"
                        yield rag_chat_service._sse_event("token", {"delta": notice, "cursor": 0})

                        # フロント履歴形式に合わせて辞書化
                        history_dicts = [{"role": m.role, "content": m.content} for m in history]
                        async for w in foundry_agent_service.search_with_bing_stream(session_id, history_dicts):
                            if await request.is_disconnected():
                                return
                            t = w.get("type")
                            if t == "token":
                                yield rag_chat_service._sse_event("token", {"delta": w.get("delta", ""), "cursor": 0})
                            elif t == "citations":
                                yield rag_chat_service._sse_event("citations", {"items": w.get("items", [])})
                            elif t == "done":
                                # ここで初めて done を１回だけ返す
                                yield rag_chat_service._sse_event("done", {"content_len": 0, "is_low": False})
                                return
                            elif t == "error":
                                # 失敗時も会話を閉じてバブルを確定させる
                                yield rag_chat_service._sse_event("token", {"delta": "\n\n(検索でエラーが発生しました)", "cursor": 0})
                                yield rag_chat_service._sse_event("done", {"content_len": 0, "is_low": False})
                                return
                        # 念のため
                        yield rag_chat_service._sse_event("done", {"content_len": 0, "is_low": False})
                        return

                else:
                    # 未知イベントは中継
                    yield ev

                # 小刻みにflush（環境によってはなくてもOK）
                await asyncio.sleep(0)
            
            # RAG 側が自然終了した場合のフォールバック（通常ここには来ない）
            yield rag_chat_service._sse_event("done", {"content_len": 0, "is_low": False})
        
        headers = {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Nginx系のバッファ抑止
        }
        return StreamingResponse(gen(), media_type="text/event-stream", headers=headers)

    except asyncio.CancelledError:
        # クライアント側で中断された
        return
    except Exception as e:
        # SSEではHTTPエラーよりも error イベントで返した方がフロント処理が楽
        async def gen_err():
            yield rag_chat_service._sse_event("error", {"message": str(e)})
        return StreamingResponse(gen_err(), media_type="text/event-stream")

@app.get("/api/health")
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "ok"}


if __name__ == "__main__":
    # This lets you test the application locally with Uvicorn
    # For production deployment, use a proper ASGI server like Gunicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("application:app", host="0.0.0.0", port=port, reload=True)
