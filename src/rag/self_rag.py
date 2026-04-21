"""
Self-RAG 四层质量门控模块

实现 Self-RAG / Corrective RAG 的核心判断逻辑：
1. judge_need_retrieval: 判断是否需要外部检索
2. judge_relevance: 过滤不相关文档
3. judge_hallucination: 检测回复幻觉
4. judge_usefulness: 判断回复有用性

复用模块:
- src/graph/llm.py::get_graph_llm()
- src/graph/schemas.py — RetrievalJudgement, RelevanceJudgement,
                          HallucinationJudgement, UsefulnessJudgement
- src/config.py::get_settings()
- src/utils/logger.py::get_logger()
"""

from typing import Optional

import numpy as np
from langchain_core.messages import HumanMessage, SystemMessage

from ..graph.llm import get_graph_llm
from ..graph.schemas import (
    RetrievalJudgement,
    RelevanceJudgement,
    HallucinationJudgement,
    UsefulnessJudgement,
    SelfRAGLiteJudgement,
)
from ..config import get_settings
from ..utils.logger import get_logger


logger = get_logger(__name__)

# ── 系统提示 ──────────────────────────────────────────────────────────────────

_NEED_RETRIEVAL_PROMPT = """你是一个饮食助手的检索决策专家。
判断以下用户查询是否需要检索外部食谱/营养知识库。

规则：
- 事实性查询（食谱、营养、食材搭配）→ need_retrieval=true
- 纯闲聊、打招呼、简单确认 → need_retrieval=false"""

_RELEVANCE_PROMPT = """你是一个文档相关性评估专家。
判断给定的食谱文档是否与用户查询相关。
相关性分数：0.0（完全无关）到 1.0（高度相关）"""

_HALLUCINATION_PROMPT = """你是一个回复质量审核专家。
检测以下饮食助手的回复中是否包含没有被参考文档支持的声明（幻觉）。
只有明确与文档矛盾或完全不在文档中的内容才算幻觉，合理推断不算。"""

_USEFULNESS_PROMPT = """你是一个回复有用性评估专家。
判断以下饮食助手的回复是否真正解决了用户的问题。
如果回复直接、完整地回答了用户需求则视为有用。"""

_SELF_RAG_LITE_PROMPT = """你是一个饮食助手回复质量审核专家。
请基于参考文档和用户问题，同时完成两件事：
1. 判断助手回复是否包含不被参考文档支持的声明（幻觉）
2. 判断助手回复是否真正解决了用户问题（有用性）

判定规则：
- 只有明确与文档矛盾或完全不在文档中的内容才算幻觉，合理推断不算
- 如果回复直接、完整地回答了用户需求则视为有用
- 请严格输出结构化结果，不要额外解释"""


def _coerce_docs(docs: list) -> list[dict]:
    normalized_docs: list[dict] = []
    for item in docs:
        if isinstance(item, dict):
            normalized_docs.append(item)
        elif isinstance(item, list):
            normalized_docs.extend(_coerce_docs(item))
        elif item is not None:
            normalized_docs.append({"text": str(item)})
    return normalized_docs


def _normalize_docs_for_relevance(items: list) -> list[dict]:
    normalized: list[dict] = []

    def _visit(value):
        if isinstance(value, dict):
            normalized.append(value)
            return
        if isinstance(value, (list, tuple)):
            for child in value:
                _visit(child)
            return
        normalized.append({"text": str(value)})

    for item in items or []:
        _visit(item)
    return normalized


class SelfRAGJudge:
    """Self-RAG 四层质量门控

    实现 Self-RAG / Corrective RAG 的核心判断逻辑：
    1. judge_need_retrieval: 判断是否需要检索
    2. judge_relevance: 过滤不相关文档
    3. judge_hallucination: 检测回复幻觉
    4. judge_usefulness: 判断回复有用性

    复用模块:
    - src/graph/llm.py::get_graph_llm()
    - src/graph/schemas.py — Structured Output 模型
    - src/config.py::get_settings()
    - src/utils/logger.py::get_logger()
    """

    def __init__(self) -> None:
        """初始化 SelfRAGJudge，从配置读取阈值参数。"""
        self._settings = get_settings()

    def judge_need_retrieval(
        self,
        query: str,
        history: Optional[list] = None,
    ) -> RetrievalJudgement:
        """判断当前查询是否需要外部检索。

        Args:
            query: 用户查询文本
            history: 对话历史（可选）

        Returns:
            RetrievalJudgement，异常时降级返回 need_retrieval=True

        Examples:
            >>> judge.judge_need_retrieval("你好")
            RetrievalJudgement(need_retrieval=False, reason="闲聊")
            >>> judge.judge_need_retrieval("番茄炒蛋怎么做")
            RetrievalJudgement(need_retrieval=True, reason="食谱查询")
        """
        logger.info(f"Self-RAG 检索必要性判断: query='{query[:50]}'")

        history_text = ""
        if history:
            for msg in history[-3:]:
                content = msg.content if hasattr(msg, "content") else str(msg)
                role = getattr(msg, "type", "user")
                history_text += f"{role}: {content}\n"

        human_content = f"用户查询：{query}"
        if history_text:
            human_content = f"对话历史：\n{history_text}\n{human_content}"

        try:
            llm = get_graph_llm()
            structured_llm = llm.with_structured_output(RetrievalJudgement)
            result: RetrievalJudgement = structured_llm.invoke([
                SystemMessage(content=_NEED_RETRIEVAL_PROMPT),
                HumanMessage(content=human_content),
            ])
            logger.info(
                f"检索必要性判断结果: need_retrieval={result.need_retrieval}, "
                f"reason='{result.reason[:60]}'"
            )
            return result
        except Exception as e:
            logger.error(f"检索必要性判断失败，降级为 True: {e}", exc_info=True)
            return RetrievalJudgement(need_retrieval=True, reason="判断失败，默认需要检索")

    def judge_relevance(
        self,
        query: str,
        docs: list[dict],
    ) -> list[dict]:
        """逐个判断文档与查询的相关性，过滤低分文档。

        Args:
            query: 用户查询文本
            docs: 文档列表，每个文档需包含 'text' 或 'name' 字段

        Returns:
            过滤后的相关文档列表（附带 relevance_score 字段），
            异常时降级返回原始文档列表
        """
        if not docs:
            return []

        docs = _coerce_docs(docs)

        logger.info(f"Self-RAG 相关性过滤: query='{query[:50]}', docs={len(docs)}")
        threshold = self._settings.rag_relevance_threshold
        relevant_docs: list[dict] = []

        try:
            llm = get_graph_llm()
            structured_llm = llm.with_structured_output(RelevanceJudgement)

            for doc in docs:
                doc_text = doc.get("text") or doc.get("name", str(doc))[:200]
                human_content = (
                    f"用户查询：{query}\n\n"
                    f"文档内容：{doc_text}"
                )
                try:
                    result: RelevanceJudgement = structured_llm.invoke([
                        SystemMessage(content=_RELEVANCE_PROMPT),
                        HumanMessage(content=human_content),
                    ])
                    doc_with_score = dict(doc)
                    doc_with_score["relevance_score"] = result.relevance_score
                    if result.relevance_score >= threshold:
                        relevant_docs.append(doc_with_score)
                except Exception as e:
                    logger.warning(f"单文档相关性判断失败，保留该文档: {e}")
                    doc_fallback = dict(doc)
                    doc_fallback["relevance_score"] = 1.0
                    relevant_docs.append(doc_fallback)

            logger.info(
                f"相关性过滤完成: {len(relevant_docs)}/{len(docs)} 文档通过 "
                f"(threshold={threshold})"
            )
            return relevant_docs

        except Exception as e:
            logger.error(f"相关性过滤失败，降级返回全部文档: {e}", exc_info=True)
            return docs

    # ── Phase 5: embedding 快速相关性过滤 ──────────────────────────────────────

    def _get_embedding_function(self):
        """获取 embedding 函数实例（复用 vectorstore 的 DashScopeEmbeddings）。

        Returns:
            DashScopeEmbeddings 实例
        """
        from langchain_community.embeddings import DashScopeEmbeddings

        return DashScopeEmbeddings(
            model=self._settings.embedding_model,
            dashscope_api_key=self._settings.dashscope_api_key,
        )

    @staticmethod
    def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """计算两个向量的余弦相似度。

        Args:
            vec_a: 向量 A
            vec_b: 向量 B

        Returns:
            余弦相似度值 [-1, 1]
        """
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))

    def judge_relevance_fast(
        self,
        query: str,
        docs: list[dict],
        threshold: float = 0.3,
    ) -> list[dict]:
        """基于 embedding 余弦相似度的快速相关性过滤。

        相比 judge_relevance() 的 O(N) 次 LLM 调用，
        本方法通过 embedding 向量计算实现 O(1) 批量打分。

        Args:
            query: 用户查询文本
            docs: 文档列表，每个文档需包含 'text' 或 'name' 字段
            threshold: 余弦相似度阈值

        Returns:
            过滤后的文档列表（附带 relevance_score 字段），
            异常时降级返回原始文档列表
        """
        if not docs:
            return []

        normalized_docs = _normalize_docs_for_relevance(docs)
        if not normalized_docs:
            return []

        logger.info(
            f"Self-RAG 快速相关性过滤: query='{query[:50]}', "
            f"docs={len(normalized_docs)}, threshold={threshold}"
        )

        try:
            embeddings = self._get_embedding_function()

            # 1. 获取 query embedding
            query_vec = np.array(embeddings.embed_query(query))

            # 2. 获取每篇 doc 的文本，批量 embedding
            doc_texts = []
            for doc in normalized_docs:
                text = doc.get("text") or doc.get("name", str(doc))[:200]
                doc_texts.append(text)

            doc_vecs = np.array(embeddings.embed_documents(doc_texts))

            # 3. 批量计算余弦相似度
            relevant_docs: list[dict] = []
            for i, doc in enumerate(normalized_docs):
                score = self._cosine_similarity(query_vec, doc_vecs[i])
                doc_with_score = dict(doc)
                doc_with_score["relevance_score"] = round(score, 4)
                if score >= threshold:
                    relevant_docs.append(doc_with_score)

            # 4. 按分数降序排序
            relevant_docs.sort(key=lambda d: d["relevance_score"], reverse=True)

            logger.info(
                f"快速相关性过滤完成: {len(relevant_docs)}/{len(normalized_docs)} 文档通过 "
                f"(threshold={threshold})"
            )
            return relevant_docs

        except Exception as e:
            logger.error(
                f"快速相关性过滤失败，降级返回全部文档: {e}", exc_info=True
            )
            return normalized_docs

    def judge_hallucination(
        self,
        response: str,
        docs: list[dict],
    ) -> HallucinationJudgement:
        """检测回复中是否存在幻觉（不被检索文档支持的声明）。

        Args:
            response: 助手生成的回复文本
            docs: 参考文档列表

        Returns:
            HallucinationJudgement，异常时降级返回 has_hallucination=False
        """
        logger.info(f"Self-RAG 幻觉检测: response_len={len(response)}, docs={len(docs)}")

        docs_text = ""
        for i, doc in enumerate(docs[:5], 1):
            if doc.get("text"):
                doc_text = doc["text"][:300]
            else:
                # 用所有元数据字段拼成可读文本（与 generator 保持一致）
                parts = []
                for key in ("name", "description", "cuisine", "time", "difficulty",
                            "calories", "tags", "health_goals", "protein", "carbs", "fat"):
                    val = doc.get(key)
                    if val:
                        parts.append(f"{key}={val}")
                doc_text = ", ".join(parts) or doc.get("name", "未知")
            docs_text += f"[文档{i}] {doc_text}\n"

        if not docs_text:
            docs_text = "（无参考文档）"

        human_content = (
            f"参考文档：\n{docs_text}\n\n"
            f"助手回复：\n{response}"
        )

        try:
            llm = get_graph_llm()
            structured_llm = llm.with_structured_output(HallucinationJudgement)
            result: HallucinationJudgement = structured_llm.invoke([
                SystemMessage(content=_HALLUCINATION_PROMPT),
                HumanMessage(content=human_content),
            ])
            logger.info(
                f"幻觉检测结果: has_hallucination={result.has_hallucination}, "
                f"claims={len(result.hallucinated_claims)}"
            )
            return result
        except Exception as e:
            logger.error(f"幻觉检测失败，降级为无幻觉: {e}", exc_info=True)
            return HallucinationJudgement(has_hallucination=False, hallucinated_claims=[])

    def judge_usefulness(
        self,
        query: str,
        response: str,
    ) -> UsefulnessJudgement:
        """判断回复是否真正解决了用户问题。

        Args:
            query: 用户查询文本
            response: 助手生成的回复文本

        Returns:
            UsefulnessJudgement，异常时降级返回 is_useful=True
        """
        logger.info(f"Self-RAG 有用性判断: query='{query[:50]}'")

        human_content = (
            f"用户查询：{query}\n\n"
            f"助手回复：\n{response}"
        )

        try:
            llm = get_graph_llm()
            structured_llm = llm.with_structured_output(UsefulnessJudgement)
            result: UsefulnessJudgement = structured_llm.invoke([
                SystemMessage(content=_USEFULNESS_PROMPT),
                HumanMessage(content=human_content),
            ])
            logger.info(
                f"有用性判断结果: is_useful={result.is_useful}, "
                f"missing={result.missing_info}"
            )
            return result
        except Exception as e:
            logger.error(f"有用性判断失败，降级为有用: {e}", exc_info=True)
            return UsefulnessJudgement(is_useful=True, missing_info=[])

    def judge_quality_lite(
        self,
        query: str,
        response: str,
        docs: list[dict],
    ) -> SelfRAGLiteJudgement:
        """一次调用同时完成幻觉检测与有用性判断。"""
        logger.info(
            f"Self-RAG Lite 联合判断: query='{query[:50]}', response_len={len(response)}, docs={len(docs)}"
        )

        docs_text = ""
        for i, doc in enumerate(docs[:5], 1):
            if doc.get("text"):
                doc_text = doc["text"][:300]
            else:
                parts = []
                for key in (
                    "name", "description", "cuisine", "time", "difficulty",
                    "calories", "tags", "health_goals", "protein", "carbs", "fat"
                ):
                    val = doc.get(key)
                    if val:
                        parts.append(f"{key}={val}")
                doc_text = ", ".join(parts) or doc.get("name", "未知")
            docs_text += f"[文档{i}] {doc_text}\n"

        if not docs_text:
            docs_text = "（无参考文档）"

        human_content = (
            f"用户查询：{query}\n\n"
            f"参考文档：\n{docs_text}\n"
            f"助手回复：\n{response}"
        )

        try:
            llm = get_graph_llm()
            structured_llm = llm.with_structured_output(SelfRAGLiteJudgement)
            result: SelfRAGLiteJudgement = structured_llm.invoke([
                SystemMessage(content=_SELF_RAG_LITE_PROMPT),
                HumanMessage(content=human_content),
            ])
            logger.info(
                "Self-RAG Lite 结果: has_hallucination=%s, claims=%s, is_useful=%s, missing=%s",
                result.has_hallucination,
                len(result.hallucinated_claims),
                result.is_useful,
                result.missing_info,
            )
            return result
        except Exception as e:
            logger.error(f"Self-RAG Lite 判断失败，降级为安全默认值: {e}", exc_info=True)
            return SelfRAGLiteJudgement(
                has_hallucination=False,
                hallucinated_claims=[],
                is_useful=True,
                missing_info=[],
            )


# 模块级单例
_self_rag_judge: Optional[SelfRAGJudge] = None


def get_self_rag_judge() -> SelfRAGJudge:
    """获取 SelfRAGJudge 单例。

    Returns:
        SelfRAGJudge 实例
    """
    global _self_rag_judge
    if _self_rag_judge is None:
        _self_rag_judge = SelfRAGJudge()
        logger.info("SelfRAGJudge 初始化完成")
    return _self_rag_judge
