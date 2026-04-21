"""
RAG 深度升级模块 (Phase 1)

提供 Adaptive RAG、Query Transformation、Self-RAG 和语义分块四大能力：
- AdaptiveRAG: 根据查询复杂度自动选择检索策略
- QueryTransformer: Multi-Query / HyDE / Step-Back 三种查询变换
- SelfRAGJudge: 四层质量门控（检索必要性、相关性、幻觉、有用性）
- SemanticChunkerWrapper: 基于 embedding 的语义分块

复用模块:
- src/graph/llm.py::get_graph_llm()
- src/graph/schemas.py — Structured Output 模型
- src/retriever/enhanced_retriever_v3.py::EnhancedRetrieverV3
- src/reranker/cross_encoder_reranker.py::CrossEncoderReranker
- src/config.py::get_settings()
- src/utils/logger.py::get_logger()
"""

from .adaptive_rag import AdaptiveRAG
from .query_transform import QueryTransformer
from .self_rag import SelfRAGJudge
from .chunking import SemanticChunkerWrapper

__all__ = [
    "AdaptiveRAG",
    "QueryTransformer",
    "SelfRAGJudge",
    "SemanticChunkerWrapper",
]
