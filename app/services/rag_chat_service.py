"""
RAG Chat Service using Azure OpenAI and Azure AI Search

This module implements a Retrieval Augmented Generation (RAG) service that connects
Azure OpenAI with Azure AI Search. RAG enhances LLM responses by grounding them in
your enterprise data stored in Azure AI Search.
"""
import logging
import json
import asyncio
from typing import List
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AsyncAzureOpenAI
from app.models.chat_models import ChatMessage
from app.config import settings

logger = logging.getLogger(__name__)


class RagChatService:
    """
    Service that provides Retrieval Augmented Generation (RAG) capabilities
    by connecting Azure OpenAI with Azure AI Search for grounded responses.
    
    This service:
    1. Handles authentication to Azure services using Managed Identity
    2. Implements the "On Your Data" pattern using Azure AI Search as a data source
    3. Processes user queries and returns AI-generated responses grounded in your data
    """
    
    def __init__(
            self,
            max_history_messages: int = 12,
            citation_max: int = 5
        ):
        """Initialize the RAG chat service using settings from app config"""
        # Store settings for easy access
        self.openai_endpoint = settings.azure_openai_endpoint
        self.gpt_deployment = settings.azure_openai_gpt_deployment
        self.embedding_deployment = settings.azure_openai_embedding_deployment
        self.openai_api_version = settings.azure_openai_api_version
        self.search_url = settings.azure_search_service_url
        self.search_index_name = settings.azure_search_index_name
        self.system_prompt = settings.system_prompt
        self.max_history_messages = max_history_messages
        self.citation_max = citation_max
        
        self.LOW_CONFIDENCE_PHRASES = [
            "requested information is not available",
            "please try another query",
            "no relevant information found",
            "could not find relevant information",
            "data is unavailable",
            "情報が見つかりません",
            "別のクエリやトピックを試してください",
            "別の質問やトピックでお試しください",
            "該当する情報が見つかりません",
            "情報を取得できません",
            "関連する情報がありません",
            "情報は含まれていません",
            "情報が含まれていません"
        ]
        
        # Create Azure credentials for managed identity
        # This allows secure, passwordless authentication to Azure services
        self.credential = DefaultAzureCredential()
        token_provider = get_bearer_token_provider(
            self.credential,
            "https://cognitiveservices.azure.com/.default"
        )
        

        # Create Azure OpenAI client
        # We use the latest Azure OpenAI Python SDK with async support
        self.openai_client = AsyncAzureOpenAI(
            azure_endpoint=self.openai_endpoint,
            azure_ad_token_provider=token_provider,
            api_version=self.openai_api_version,
        )
        
        logger.info("RagChatService initialized with environment variables")
    
    def _extract_citations(self, message_or_ctx) -> list[dict]:
        """
        Azure OpenAI OYD の citations を UI 用に正規化
        -> [{title, content, filePath, url}]
        """
        cites = []
        # message でも context(dict) でも受けられるようにする
        if isinstance(message_or_ctx, dict):
            ctx = message_or_ctx
        else:
            ctx = getattr(message_or_ctx, "context", None) or {}
        count = 0
        for c in ctx.get("citations", []):
            url = (
                c.get("url")
                or c.get("filepath")
                or c.get("filePath")
                or c.get("source")
                or ""
            )
            title = (
                c.get("title")
                or c.get("filename")
                or (url.split("/")[-1] if url else "source")
            )
            cites.append({
                "title": title,
                "content": c.get("content", ""),
                "filePath": url,   # UI は filePath or url を参照
                "url": url,
            })
            count += 1
            if count >= self.citation_max:
                break
        return cites
    
    def _normalize_response(self, content: str, citations: list[dict]) -> dict:
        """
        UI がそのまま使える共通フォーマットに整形
        """
        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": content,
                    "context": {"citations": citations}
                }
            }]
        }
    
    async def get_chat_completion(self, history: List[ChatMessage]):
        """
        Process a chat completion request with RAG capabilities by integrating with Azure AI Search
        
        This method:
        1. Formats the conversation history for Azure OpenAI
        2. Configures Azure AI Search as a data source using the "On Your Data" pattern
        3. Sends the request to Azure OpenAI with data_sources parameter
        4. Returns the response with citations to source documents
        
        Args:
            history: List of chat messages from the conversation history
            
        Returns:
            Raw response from the OpenAI API with citations from Azure AI Search
        """
        try:
            # Limit chat history to the paran[max_history_messages] most recent messages to prevent token limit issues
            recent_history = history[-self.max_history_messages:] if len(history) > self.max_history_messages else history
            
            # Convert to Azure OpenAI compatible message format
            messages = []
            
            # Add system message
            messages.append({
                "role": "system", 
                "content": self.system_prompt
            })
            
            # Add conversation history
            for msg in recent_history:
                messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
            
            # Configure Azure AI Search data source according to the "On Your Data" pattern
            # This connects Azure OpenAI directly to your search index without needing to
            # manually implement vector search, chunking, or semantic rankers
            data_source = {
                "type": "azure_search",
                "parameters": {
                    "endpoint": self.search_url,
                    "index_name": self.search_index_name,
                    "authentication": {
                        "type": "system_assigned_managed_identity"
                    },
                    # Combines vector and traditional search
                    "query_type": "semantic",
                    # "query_type": "vector_semantic_hybrid",
                    # The naming pattern for semantic configuration is generated by Azure AI Search 
                    # during integrated vectorization and cannot be customized
                    "semantic_configuration": f"{self.search_index_name}-semantic-configuration",
                    # "embedding_dependency": {
                    #     "type": "deployment_name",
                    #     "deployment_name": self.embedding_deployment
                    # },
                    # Search インデックスのフィールドと OYD の対応
                    "fields_mapping": {
                        "content_fields": ["content_text"],
                        # "vector_fields":  ["content_embedding"],
                        "title_field":    "document_title",
                        "filepath_field": "source_path"
                    },
                    # （任意）取り込み件数や厳密度の調整
                    "top_n_documents": 5,
                    "strictness": 3,
                    "in_scope": True,
                    # （任意）モデルへ渡す役割説明（プロンプト強化）
                    # "role_information": "回答は簡潔明瞭に。必要十分な説明のみ。冗長な前置きや繰り返しは避ける。出典は重複を除いて最大5件。教材の用語・表記を優先し、コードは最小限に留める。情報が特定できない場合は「手元の資料では特定できません」と明示し、確認すべきポイントを短く示す。"
                }
            }
            
            # Call Azure OpenAI for completion with the data_sources parameter directly
            # The data_sources parameter enables the "On Your Data" pattern, where
            # Azure OpenAI automatically retrieves relevant documents from your search index
            response = await self.openai_client.chat.completions.create(
                model=self.gpt_deployment,
                messages=messages,
                extra_body={
                    "data_sources": [data_source]
                },
                temperature=0.2,
                max_tokens=1024,
                stream=False
            )
            
            ai_choice = response.choices[0]
            ai_msg  = ai_choice.message
            finish = getattr(ai_choice, "finish_reason", None)
            usage = getattr(response, "usage", None)
            
            logger.info(f"[rag] finish_reason={finish} usage={getattr(usage, 'total_tokens', None)}")
            
            content_raw = ai_msg .content or ""
            citations_norm  = self._extract_citations(ai_msg)
            
            # NGワード判定（文言/出典の少なさ）
            content_lc = content_raw.lower()
            is_low = (
                any(p in content_lc for p in self.LOW_CONFIDENCE_PHRASES)
                or len(citations_norm) == 0
            )
            
            normalized = self._normalize_response(content_raw, citations_norm if not is_low else [])
            
            # Return the raw response
            return normalized, is_low
            
        except Exception as e:
            logger.error(f"Error in get_chat_completion: {str(e)}")
            # Propagate all errors to the controller layer
            raise
    
    # ===== ここからストリーミング（SSE）用 ==================================
    def _sse_event(self, event: str, data: dict) -> bytes:
        """SSE 1イベントのエンコード"""
        return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8")

    async def _citations_min_call(self, messages, data_source) -> list[dict]:
        """
        ストリームで citations が得られない場合の保険：
        max_tokens=1 の最小呼び出しで citations のみ取得
        """
        try:
            resp = await self.openai_client.chat.completions.create(
                model=self.gpt_deployment,
                messages=messages,
                extra_body={"data_sources": [data_source]},
                temperature=0.0,
                max_tokens=1,
                stream=False
            )
            return self._extract_citations(resp.choices[0].message)
        except Exception as e:
            logger.warning(f"[stream] citations min-call failed: {e}")
            return []

    async def stream_chat_completion(self, history: List[ChatMessage]):
        """
        SSE 用ジェネレータ：
          - event: token     { delta: "…", cursor: int }
          - event: citations { items: [...] }
          - event: done      { content_len: int, is_low: bool }
          - event: error     { message: str }
        """
        # 1) messages / data_source 準備
        recent_history = history[-self.max_history_messages:] if len(history) > self.max_history_messages else history
        messages = [{"role": "system", "content": self.system_prompt}]
        for msg in recent_history:
            messages.append({"role": msg.role, "content": msg.content})

        data_source = {
            "type": "azure_search",
            "parameters": {
                "endpoint": self.search_url,
                "index_name": self.search_index_name,
                "authentication": {"type": "system_assigned_managed_identity"},
                "query_type": "semantic",
                "semantic_configuration": f"{self.search_index_name}-semantic-configuration",
                "fields_mapping": {
                    "content_fields": ["content_text"],
                    "title_field":    "document_title",
                    "filepath_field": "source_path"
                },
                "top_n_documents": 5,
                "strictness": 3,
                "in_scope": True
            }
        }

        full_parts: list[str] = []
        cursor = 0
        citations: list[dict] = []

        try:
            # 2) OpenAI ストリーム開始
            stream = await self.openai_client.chat.completions.create(
                model=self.gpt_deployment,
                messages=messages,
                extra_body={"data_sources": [data_source]},
                temperature=0.2,
                max_tokens=1024,
                stream=True
            )

            async for chunk in stream:
                # ChatCompletionChunk 形式を想定
                for choice in getattr(chunk, "choices", []):
                    delta = getattr(choice, "delta", None)
                    if not delta:
                        continue
                    piece = getattr(delta, "content", None)
                    if piece:
                        full_parts.append(piece)
                        cursor += len(piece)
                        # 本文差分を即送信
                        yield self._sse_event("token", {"delta": piece, "cursor": cursor})
                    # モデル/SDKによっては delta.context に citations が混ざることがあるため拾っておく
                    ctx = getattr(delta, "context", None)
                    if ctx and not citations:
                        try:
                            citations = self._extract_citations(ctx)
                        except Exception:
                            pass

            content = "".join(full_parts).strip()

            # 3) 引用が未取得なら最小呼び出しで取得
            if not citations:
                citations = await self._citations_min_call(messages + [{"role": "assistant", "content": content}], data_source)

            # 低信頼判定（既存ロジックに合わせる）
            content_lc = content.lower()
            is_low = (len(citations) == 0) or any(p in content_lc for p in self.LOW_CONFIDENCE_PHRASES)

            # 4) citations → done
            yield self._sse_event("citations", {"items": citations})
            yield self._sse_event("done", {"content_len": len(content), "is_low": is_low})

        except Exception as e:
            logger.error(f"[stream] error: {e}")
            yield self._sse_event("error", {"message": str(e)})


# Create singleton instance
rag_chat_service = RagChatService()
