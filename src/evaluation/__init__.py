"""
评测体系模块 (Phase 2)

提供 RAGAS 评测、检索指标、生成指标和端到端 Benchmark 四个子模块：
- RAGASEvaluator: 封装 RAGAS 四大核心指标
- RetrievalMetrics: Recall@K, MRR, nDCG
- GenerationMetrics: Faithfulness, Answer Relevancy, Completeness
- E2EBenchmark: 端到端评测 + A/B 对比报告

复用模块:
- src/graph/diet_graph.py::build_diet_graph()
- src/rag/adaptive_rag.py::AdaptiveRAG
- src/retriever/enhanced_retriever_v3.py::EnhancedRetrieverV3
- src/reranker/cross_encoder_reranker.py::CrossEncoderReranker
- src/config.py::get_settings()
- src/utils/logger.py::get_logger()
"""

from .ragas_eval import RAGASEvaluator
from .retrieval_metrics import RetrievalMetrics
from .generation_metrics import GenerationMetrics
from .e2e_benchmark import E2EBenchmark
from .quant_benchmark import QuantBenchmarkCase, QuantBenchmarkResult, QuantitativeBenchmark

__all__ = [
    "RAGASEvaluator",
    "RetrievalMetrics",
    "GenerationMetrics",
    "E2EBenchmark",
    "QuantBenchmarkCase",
    "QuantBenchmarkResult",
    "QuantitativeBenchmark",
]
