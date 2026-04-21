"""
LLM 客户端模块

提供 Qwen LLM 的初始化和管理功能
"""

from .qwen_client import (
    QwenLLMClient,
    get_llm_client,
    get_llm,
    test_llm_connection
)

__all__ = [
    "QwenLLMClient",
    "get_llm_client",
    "get_llm",
    "test_llm_connection"
]
