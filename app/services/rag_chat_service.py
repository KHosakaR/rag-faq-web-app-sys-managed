"""
RAG Chat Service using Azure OpenAI and Azure AI Search

This module implements a Retrieval Augmented Generation (RAG) service that connects
Azure OpenAI with Azure AI Search. RAG enhances LLM responses by grounding them in
your enterprise data stored in Azure AI Search.
"""
import logging
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
            "追加の情報は含まれていません"
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
    
    def _extract_citations(self, message) -> list[dict]:
        """
        Azure OpenAI OYD の citations を UI 用に正規化
        -> [{title, content, filePath, url}]
        """
        cites = []
        ctx = getattr(message, "context", None) or {}
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


# Create singleton instance
rag_chat_service = RagChatService()
