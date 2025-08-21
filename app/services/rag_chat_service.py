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
    
    def __init__(self):
        """Initialize the RAG chat service using settings from app config"""
        # Store settings for easy access
        self.openai_endpoint = settings.azure_openai_endpoint
        self.gpt_deployment = settings.azure_openai_gpt_deployment
        self.embedding_deployment = settings.azure_openai_embedding_deployment
        self.openai_api_version = settings.azure_openai_api_version
        self.search_url = settings.azure_search_service_url
        self.search_index_name = settings.azure_search_index_name
        self.system_prompt = settings.system_prompt
        
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
        Azure OpenAI 'on your data' の citations から
        [{title: str, url: str}] の配列を作る
        """
        cites = []
        ctx = getattr(message, "context", None) or {}
        for c in ctx.get("citations", []):
            # Azure AI Search 統合の citation で取り得るキーを広めにカバー
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
            if url:
                cites.append({"title": title, "url": url})
        return cites
    
    def _normalize_response(self, message, citations: list[dict]) -> dict:
        """
        UI がそのまま使える共通フォーマットに整形
        """
        return {
            "choices": [{
                "message": {
                    "role": message.role,
                    "content": message.content
                }
            }],
            "citations": citations
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
            # Limit chat history to the 20 most recent messages to prevent token limit issues
            recent_history = history[-20:] if len(history) > 20 else history
            
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
                    "query_type": "vector_semantic_hybrid",
                    # The naming pattern for semantic configuration is generated by Azure AI Search 
                    # during integrated vectorization and cannot be customized
                    "semantic_configuration": f"{self.search_index_name}-semantic-configuration",
                    "embedding_dependency": {
                        "type": "deployment_name",
                        "deployment_name": self.embedding_deployment
                    },
                    # Search インデックスのフィールドと OYD の対応
                    "fieldsMapping": {
                        "contentFields": ["content_text"],
                        "vectorFields":  ["content_embedding"],
                        "titleField":    "document_title",
                        "filepathField": "source_path"
                    },
                    # （任意）取り込み件数や厳密度の調整
                    "top_n_documents": 5,
                    "strictness": 3,
                    # （任意）モデルへ渡す役割説明（プロンプト強化）
                    # "role_information": self.system_prompt or ""
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
                stream=False
            )
            
            message = response.choices[0].message
            content_lc = (message.content or "").lower()
            citations = self._extract_citations(message)
            
            # NGワード判定（文言/出典の少なさ）
            is_low = any(p in content_lc for p in self.LOW_CONFIDENCE_PHRASES) or (len(citations) <= 1)
            
            normalized = self._normalize_response(msg, citations if not is_low else [])
            
            # Return the raw response
            return normalized, is_low
            
        except Exception as e:
            logger.error(f"Error in get_chat_completion: {str(e)}")
            # Propagate all errors to the controller layer
            raise


# Create singleton instance
rag_chat_service = RagChatService()
