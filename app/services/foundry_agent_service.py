import os
import logging
import time
from typing import Tuple, List

from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from app.config import settings
# from dotenv import load_dotenv

# 環境変数読み込み
# load_dotenv()
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

class FoundryAgentService:
    def __init__(self):
        # 環境変数から設定取得
        self.endpoint = settings.azure_ai_foundry_endpoint
        self.agent_id = settings.azure_ai_foundry_agent_id
        

        if not self.endpoint or not self.agent_id:
            raise ValueError("AZURE_AI_FOUNDRY_ENDPOINT または AZURE_AI_FOUNDRY_AGENT_ID が未設定です。")

        # 認証（DefaultAzureCredentialでAAD認証 or マネージドID）
        self.client = AIProjectClient(endpoint=self.endpoint, credential=DefaultAzureCredential())
        
        self.agent = self.client.agents.get_agent(agent_id=self.agent_id)
        self.thread = None  # 文脈を保持するスレッドID
        self.max_history = 5
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

    def search_with_bing(self, chat_history: list[dict]) -> Tuple[str, List[str]]:
        """
        履歴対応版: thread を維持して文脈を引き継ぐ
        """
        # thread = self.client.agents.threads.create()
        if not self.thread:
            self.thread = self.client.agents.threads.create()

        for msg in chat_history:
            self.client.agents.messages.create(
                thread_id=self.thread.id,
                role=msg["role"],
                content=msg["content"]
            )

        run = self.client.agents.runs.create(thread_id=self.thread.id, agent_id=self.agent.id)
        while run.status in ["queued", "in_progress", "requires_action"]:
            time.sleep(1)
            run = self.client.agents.runs.get(thread_id=self.thread.id, run_id=run.id)

        # 回答と引用元を抽出
        answer, sources = self._extract_answer_and_sources(self.thread.id)

        # 引用元を本文末尾に追加
        sources_text = ""
        if sources:
            sources_text = "\n\n---\n**引用元:**\n"
            sources_text += "\n".join([f"- [{i+1}] {url}" for i, url in enumerate(sources)])

        # FastAPI用のレスポンス形式で返却
        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": f"{answer}{sources_text}"
                }
            }]
        }
    
    def _format_answer_with_citations(self, text_data: dict) -> Tuple[str, dict]:
        """
        本文を番号に置換しつつ、番号→URL のマッピングを返す
        """
        raw_text = text_data["value"]
        annotations = text_data.get("annotations", [])

        citation_map = {}
        url_map = {}
        counter = 1

        # annotationsのtextごとに番号を振る
        for ann in annotations:
            ann_text = ann["text"]
            if ann_text not in citation_map:
                citation_map[ann_text] = counter
                # URLマッピングも作成
                if ann["type"] == "url_citation":
                    url_map[counter] = ann["url_citation"]["url"]
                counter += 1

        # 本文内の引用箇所を番号に置換
        formatted_text = raw_text
        for ann_text, idx in citation_map.items():
            formatted_text = formatted_text.replace(ann_text, f"[{idx}]")

        return formatted_text, url_map

    def _extract_answer_and_sources(self, thread_id: str) -> Tuple[str, List[str]]:
        """
        最新のエージェントの回答だけを返す
        """
        # 全メッセージを取得して逆順にループ
        messages = list(self.client.agents.messages.list(thread_id=thread_id))

        latest_answer = ""
        sources = []

        for message in messages:  # 最新から走査
            if message.role.name.lower() == "agent":
                for part in message.content:
                    if part["type"] == "text":
                        formatted_answer, url_map = self._format_answer_with_citations(part["text"])
                        latest_answer = formatted_answer
                        sources = list(url_map.values())
                break  # 最新の agent メッセージを取得したらループ終了

        return latest_answer.strip(), list(set(sources))

# Create singleton instance
foundry_agent_service = FoundryAgentService()