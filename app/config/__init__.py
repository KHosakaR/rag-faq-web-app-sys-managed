"""
Configuration settings for the FastAPI RAG application

This module uses Pydantic's settings management to:
1. Define configuration models for the application
2. Load environment variables from .env file or environment
3. Validate the configuration values
4. Provide strongly-typed access to settings throughout the app
"""
from typing import Optional
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
import logging

logger = logging.getLogger(__name__)


class OpenAISettings(BaseModel):
    """Azure OpenAI settings"""
    endpoint: str
    gpt_deployment: str
    embedding_deployment: Optional[str] = ""


class SearchSettings(BaseModel):
    """Azure AI Search settings"""
    url: str
    index_name: str


class AppSettings(BaseSettings):
    """Application settings with environment variable loading capabilities"""
    # Azure OpenAI Settings
    azure_openai_endpoint: str = Field(..., env="AZURE_OPENAI_ENDPOINT")
    azure_openai_gpt_deployment: str = Field(..., env="AZURE_OPENAI_GPT_DEPLOYMENT")
    azure_openai_embedding_deployment: str = Field("", env="AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    azure_openai_api_version: str = Field(..., env="AZURE_OPENAI_API_VERSION")
    
    # Azure AI Search Settings
    azure_search_service_url: str = Field(..., env="AZURE_SEARCH_SERVICE_URL")
    azure_search_index_name: str = Field(..., env="AZURE_SEARCH_INDEX_NAME")
    
    # Azure Content Safety Settings
    azure_content_safety_endpoint: str = Field(..., env="AZURE_CONTENT_SAFETY_ENDPOINT")
    
    # Azure AI Foundry Agent Settings
    azure_ai_foundry_endpoint: str = Field(..., env="AZURE_AI_FOUNDRY_ENDPOINT")
    azure_ai_foundry_agent_id: str = Field(..., env="AZURE_AI_FOUNDRY_AGENT_ID")
    
    # Other settings
    system_prompt: str = Field(
        "You are an AI assistant that helps people find information from their documents. Always cite your sources using the document title.",
        env="SYSTEM_PROMPT"
    )
    
    # Optional port setting
    port: int = Field(8080, env="PORT")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        env_nested_delimiter = "__"
        # Setting env_priority to True prioritizes environment variables over .env file
        env_priority = True
    
    @property
    def openai(self) -> OpenAISettings:
        """Return OpenAI settings in the format used by the application"""
        return OpenAISettings(
            endpoint=self.azure_openai_endpoint,
            gpt_deployment=self.azure_openai_gpt_deployment,
            embedding_deployment=self.azure_openai_embedding_deployment
        )
    
    @property
    def search(self) -> SearchSettings:
        """Return Search settings in the format used by the application"""
        return SearchSettings(
            url=self.azure_search_service_url,
            index_name=self.azure_search_index_name
        )


# Create settings instance - environment variables will be loaded automatically
# This creates a singleton instance that can be imported throughout the app
settings = AppSettings()
