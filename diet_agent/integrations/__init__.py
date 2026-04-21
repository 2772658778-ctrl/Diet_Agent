"""External integration adapters for Diet Agent."""

from .database import PostgreSQLClient, get_postgres_client
from .llm import QwenLLMClient, get_llm, get_llm_client
from .observability import RequestLogger, get_langfuse_callback
from .vectorstore import (
    get_kg_enhanced_vectorstore,
    get_or_connect_vectorstore,
    get_vectorstore,
    init_kg_enhanced_vectorstore,
    init_vectorstore,
)

__all__ = [
    "PostgreSQLClient",
    "get_postgres_client",
    "QwenLLMClient",
    "get_llm",
    "get_llm_client",
    "RequestLogger",
    "get_langfuse_callback",
    "get_kg_enhanced_vectorstore",
    "get_or_connect_vectorstore",
    "get_vectorstore",
    "init_kg_enhanced_vectorstore",
    "init_vectorstore",
]
