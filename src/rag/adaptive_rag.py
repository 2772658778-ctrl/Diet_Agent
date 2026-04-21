"""
自适应 RAG 策略选择模块

根据查询复杂度动态选择检索策略：
- simple    → 直接调用 EnhancedRetrieverV3
- complex   → Multi-Query + 多路召回 + CrossEncoder 精排
- ambiguous → HyDE / Step-Back 改写 + 扩展检索

复用模块:
- src/retriever/enhanced_retriever_v3.py::EnhancedRetrieverV3
- src/reranker/cross_encoder_reranker.py::CrossEncoderReranker
- src/rag/query_transform.py::QueryTransformer
- src/graph/llm.py::get_graph_llm()
- src/graph/schemas.py::QueryComplexity
- src/config.py::get_settings()
- src/utils/logger.py::get_logger()
"""

from typing import Any, Optional, TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage

from ..graph.llm import get_graph_llm
from ..graph.schemas import QueryComplexity
from ..config import get_settings
from .query_features import extract_query_features
from ..utils.logger import get_logger
from .query_transform import QueryTransformer

if TYPE_CHECKING:
    from ..retriever.enhanced_retriever_v3 import EnhancedRetrieverV3
    from ..reranker.cross_encoder_reranker import CrossEncoderReranker


logger = get_logger(__name__)

_CLASSIFY_PROMPT = """你是一个查询复杂度分类专家，专注于食谱和饮食查询。

判断规则：
- simple: 直接、具体的食谱查询（如"番茄炒蛋怎么做"）
- complex: 包含多个约束条件（如食材 + 健康目标 + 时间限制）
- ambiguous: 模糊、不明确、需要改写才能有效检索（如"今天不知道吃什么"）"""


def _extract_docs(results: Any) -> list[dict[str, Any]]:
    if isinstance(results, tuple):
        docs = results[0]
    else:
        docs = results

    return docs if isinstance(docs, list) else []


class AdaptiveRAG:
    """自适应 RAG 策略选择器

    根据查询复杂度动态选择检索策略：
    - simple    → 直接向量检索（复用 EnhancedRetrieverV3）
    - complex   → Multi-Query + 多路召回 + 重排
    - ambiguous → HyDE / Step-Back + 扩展检索

    复用模块:
    - src/retriever/enhanced_retriever_v3.py::EnhancedRetrieverV3
    - src/reranker/cross_encoder_reranker.py::CrossEncoderReranker
    - src/rag/query_transform.py::QueryTransformer
    - src/graph/llm.py::get_graph_llm()
    - src/config.py::get_settings()
    - src/utils/logger.py::get_logger()
    """

    def __init__(
        self,
        retriever: "EnhancedRetrieverV3",
        reranker: Optional["CrossEncoderReranker"] = None,
        query_transformer: Optional[QueryTransformer] = None,
    ) -> None:
        """初始化自适应 RAG。

        Args:
            retriever: EnhancedRetrieverV3 基础检索器（必需）
            reranker: CrossEncoderReranker 精排器（可选，complex 策略使用）
            query_transformer: QueryTransformer 查询变换器（可选，自动创建）
        """
        self.retriever = retriever
        self.reranker = reranker
        self.query_transformer = query_transformer or QueryTransformer()
        self._settings = get_settings()
        self.last_query_features: dict[str, Any] = {}
        logger.info(
            f"AdaptiveRAG 初始化完成: "
            f"reranker={'enabled' if reranker else 'disabled'}"
        )

    def classify_query(self, query: str) -> QueryComplexity:
        """使用 LLM 判断查询复杂度。

        Args:
            query: 用户查询文本

        Returns:
            QueryComplexity，异常时降级返回 level='simple'
        """
        logger.info(f"查询复杂度分类: '{query[:50]}'")

        try:
            llm = get_graph_llm()
            structured_llm = llm.with_structured_output(QueryComplexity)
            result: QueryComplexity = structured_llm.invoke([
                SystemMessage(content=_CLASSIFY_PROMPT),
                HumanMessage(content=f"查询：{query}"),
            ])
            logger.info(
                f"查询复杂度: level={result.level}, "
                f"strategy={result.suggested_strategy}"
            )
            return result
        except Exception as e:
            logger.error(f"查询复杂度分类失败，降级为 simple: {e}", exc_info=True)
            return QueryComplexity(
                level="simple",
                reasoning="分类失败，默认降级",
                suggested_strategy="standard",
            )

    def retrieve(
        self,
        query: str,
        user_id: str = "",
        user_context: Optional[dict[str, Any]] = None,
        top_k: int = 10,
    ) -> tuple[list[dict[str, Any]], str]:
        """根据查询复杂度选择策略并执行检索。

        Args:
            query: 用户查询文本
            user_id: 用户 ID
            user_context: 用户上下文（食材、健康目标等）
            top_k: 返回结果数量

        Returns:
            (检索结果列表, 实际使用的策略名称)
        """
        logger.info(f"AdaptiveRAG 检索: query='{query[:50]}', top_k={top_k}")
        self.last_query_features = extract_query_features(query, user_context)
        logger.info(f"AdaptiveRAG query_features={self.last_query_features}")

        complexity = self.classify_query(query)
        level = complexity.level

        try:
            if level == "simple":
                results, strategy = self._retrieve_simple(query, user_id, user_context, top_k)
            elif level == "complex":
                results, strategy = self._retrieve_complex(query, user_id, user_context, top_k)
            else:  # ambiguous
                results, strategy = self._retrieve_ambiguous(query, user_id, user_context, top_k)
        except Exception as e:
            logger.error(
                f"AdaptiveRAG 策略执行失败，降级为 simple: {e}", exc_info=True
            )
            results, strategy = self._retrieve_simple(query, user_id, user_context, top_k)

        logger.info(f"AdaptiveRAG 检索完成: strategy={strategy}, results={len(results)}")
        return results, strategy

    # ── 内部策略实现 ──────────────────────────────────────────────────────────

    def _retrieve_simple(
        self,
        query: str,
        user_id: str,
        user_context: Optional[dict],
        top_k: int,
    ) -> tuple[list[dict], str]:
        """simple 策略: 直接调用 EnhancedRetrieverV3。"""
        logger.info("使用 simple 策略（直接检索）")
        results = self.retriever.retrieve(
            query=query,
            user_id=user_id,
            user_context=user_context or {},
            top_k=top_k,
        )
        return results, "simple"

    def _retrieve_complex(
        self,
        query: str,
        user_id: str,
        user_context: Optional[dict],
        top_k: int,
    ) -> tuple[list[dict], str]:
        """complex 策略: Multi-Query + 多路召回 + CrossEncoder 精排。"""
        logger.info("使用 complex 策略（Multi-Query + 多路召回 + 精排）")

        sub_queries = self.query_transformer.multi_query(query)
        logger.info(f"Multi-Query 生成 {len(sub_queries)} 条子查询")

        # 多路召回并合并去重
        seen_ids: set[str] = set()
        all_docs: list[dict] = []
        per_query_k = max(top_k, 5)

        for sub_q in sub_queries:
            try:
                sub_results = self.retriever.retrieve(
                    query=sub_q,
                    user_id=user_id,
                    user_context=user_context or {},
                    top_k=per_query_k,
                )
                for doc in _extract_docs(sub_results):
                    doc_id = doc.get("id") or doc.get("name") or str(doc)
                    if doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        all_docs.append(doc)
            except Exception as e:
                logger.warning(f"子查询 '{sub_q[:30]}' 检索失败: {e}")

        if not all_docs:
            logger.warning("Multi-Query 无结果，降级为原始查询")
            return self._retrieve_simple(query, user_id, user_context, top_k)

        # CrossEncoder 精排
        if self.reranker and all_docs:
            try:
                all_docs = self.reranker.rerank(
                    query=query,
                    documents=all_docs,
                    top_k=top_k,
                )
                logger.info(f"精排后保留 {len(all_docs)} 条结果")
            except Exception as e:
                logger.warning(f"精排失败，使用未精排结果: {e}")
                all_docs = all_docs[:top_k]
        else:
            all_docs = all_docs[:top_k]

        return all_docs, "complex_multi_query"

    def _retrieve_ambiguous(
        self,
        query: str,
        user_id: str,
        user_context: Optional[dict],
        top_k: int,
    ) -> tuple[list[dict], str]:
        """ambiguous 策略: HyDE + Step-Back 改写后检索，取并集精排。"""
        logger.info("使用 ambiguous 策略（HyDE + Step-Back）")

        seen_ids: set[str] = set()
        all_docs: list[dict] = []

        # HyDE 检索
        try:
            hypothesis = self.query_transformer.hyde(query)
            hyde_results = self.retriever.retrieve(
                query=hypothesis,
                user_id=user_id,
                user_context=user_context or {},
                top_k=top_k,
            )
            hyde_docs = _extract_docs(hyde_results)
            for doc in hyde_docs:
                doc_id = doc.get("id") or doc.get("name") or str(doc)
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    all_docs.append(doc)
            logger.info(f"HyDE 召回 {len(hyde_docs)} 条")
        except Exception as e:
            logger.warning(f"HyDE 检索失败: {e}")

        # Step-Back 检索
        try:
            abstract_query = self.query_transformer.step_back(query)
            if abstract_query != query:
                sb_results = self.retriever.retrieve(
                    query=abstract_query,
                    user_id=user_id,
                    user_context=user_context or {},
                    top_k=top_k,
                )
                sb_docs = _extract_docs(sb_results)
                for doc in sb_docs:
                    doc_id = doc.get("id") or doc.get("name") or str(doc)
                    if doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        all_docs.append(doc)
                logger.info(f"Step-Back 额外召回 {len(sb_docs)} 条")
        except Exception as e:
            logger.warning(f"Step-Back 检索失败: {e}")

        if not all_docs:
            logger.warning("ambiguous 策略无结果，降级为原始查询")
            return self._retrieve_simple(query, user_id, user_context, top_k)

        # 精排
        if self.reranker and all_docs:
            try:
                all_docs = self.reranker.rerank(
                    query=query,
                    documents=all_docs,
                    top_k=top_k,
                )
            except Exception as e:
                logger.warning(f"精排失败: {e}")
                all_docs = all_docs[:top_k]
        else:
            all_docs = all_docs[:top_k]

        return all_docs, "ambiguous_hyde_stepback"


# 模块级单例
_adaptive_rag: Optional[AdaptiveRAG] = None


def get_adaptive_rag(
    retriever: "EnhancedRetrieverV3",
    reranker: Optional["CrossEncoderReranker"] = None,
    query_transformer: Optional[QueryTransformer] = None,
) -> AdaptiveRAG:
    """获取 AdaptiveRAG 单例。

    Args:
        retriever: EnhancedRetrieverV3 实例
        reranker: CrossEncoderReranker 实例（可选）
        query_transformer: QueryTransformer 实例（可选）

    Returns:
        AdaptiveRAG 实例
    """
    global _adaptive_rag
    if _adaptive_rag is None:
        _adaptive_rag = AdaptiveRAG(
            retriever=retriever,
            reranker=reranker,
            query_transformer=query_transformer,
        )
        logger.info("AdaptiveRAG 单例初始化完成")
    return _adaptive_rag
