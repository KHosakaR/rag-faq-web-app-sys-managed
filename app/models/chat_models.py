"""
Chat models for FastAPI RAG application

This module defines Pydantic models that:
1. Validate the structure of API requests
2. Provide type hints for better development experience
3. Generate automatic API documentation with FastAPI
4. Support JSON serialization/deserialization

The models focus on the core message structures needed for the chat interface.
"""
from typing import List
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """
    Base chat message model representing a single message in the conversation
    
    Compatible with Azure OpenAI's message format, where:
    - role: can be 'system', 'user', or 'assistant'
    - content: contains the actual message text
    """
    role: str
    content: str


class ChatRequest(BaseModel):
    """Chat completion request model for the API endpoint"""
    messages: List[ChatMessage] = Field(..., description="List of chat messages")
