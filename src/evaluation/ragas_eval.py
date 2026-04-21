"""
RAGAS 框架评测模块 (Phase 2)

封装 RAGAS 四大核心指标（自研 LLM-as-Judge 实现，不强依赖 ragas 库）：
- context_precision: 检索的文档中有多少真正相关
- context_recall:    相关文档有多少被检索到
- faithfulness:      回复是否忠于检索文档
- answer_relevancy:  回复是否真正回答了问题

复用模块:
- src/graph/llm.py::get_graph_llm()
- src/config.py::get_settings()
- src/utils/logger.py::get_logger()
"""

from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from ..graph.llm import get_graph_llm
from ..config import get_settings
from ..utils.logger import get_logger


logger = get_logger(__name__)


# ── Pydantic Schema（LLM Structured Output 用） ────────────────────────────────

class ContextPrecisionScore(BaseModel):
    """上下文精确度评分

    Attributes:
        score: 精确度分数 (0.0~1.0)
        reasoning: 评分理由
    """
    score: float = Field(ge=0.0, le=1.0, description="上下文精确度分数，0.0 到 1.0")
    reasoning: str = Field(default="", description="评分理由")


class ContextRecallScore(BaseModel):
    """上下文召回率评分

    Attributes:
        score: 召回率分数 (0.0~1.0)
        reasoning: 评分理由
    """
    score: float = Field(ge=0.0, le=1.0, description="上下文召回率分数，0.0 到 1.0")
    reasoning: str = Field(default="", description="评分理由")


class FaithfulnessScore(BaseModel):
    """忠实度评分

    Attributes:
        score: 忠实度分数 (0.0~1.0)
        reasoning: 评分理由
    """
    score: float = Field(ge=0.0, le=1.0, description="忠实度分数，0.0 到 1.0")
    reasoning: str = Field(default="", description="评分理由")


class AnswerRelevancyScore(BaseModel):
    """回复相关性评分

    Attributes:
        score: 相关性分数 (0.0~1.0)
        reasoning: 评分理由
    """
    score: float = Field(ge=0.0, le=1.0, description="回复相关性分数，0.0 到 1.0")
    reasoning: str = Field(default="", description="评分理由")


# ── 系统提示 ────────────────────────────────────────────────────────────────────

_CONTEXT_PRECISION_PROMPT = """你是一个检索质量评估专家。
请评估检索到的文档对于回答问题的精确程度。

评分标准（0.0 到 1.0）：
- 1.0: 全部检索文档都与问题高度相关
- 0.7~0.9: 大部分文档相关，少数不太相关
- 0.4~0.6: 约一半文档相关
- 0.0~0.3: 大部分文档与问题无关"""

_CONTEXT_RECALL_PROMPT = """你是一个检索覆盖度评估专家。
给定标准答案，请评估检索到的文档是否覆盖了回答问题所需的全部关键信息。

评分标准（0.0 到 1.0）：
- 1.0: 检索文档完全覆盖了标准答案中的所有关键信息
- 0.7~0.9: 覆盖了大部分关键信息
- 0.4~0.6: 覆盖了约一半关键信息
- 0.0~0.3: 几乎没有覆盖标准答案的关键信息"""

_FAITHFULNESS_PROMPT = """你是一个回复忠实度评估专家。
请评估助手的回复是否忠实于检索到的参考文档。

评分标准（0.0 到 1.0）：
- 1.0: 回复中所有声明都被参考文档支持
- 0.7~0.9: 大部分声明有文档支持，少数为合理推断
- 0.4~0.6: 部分声明缺乏文档支持
- 0.0~0.3: 回复大量内容没有文档依据"""

_ANSWER_RELEVANCY_PROMPT = """你是一个回复相关性评估专家。
请评估助手的回复是否直接回答了用户的问题。

评分标准（0.0 到 1.0）：
- 1.0: 完全直接地回答了用户问题
- 0.7~0.9: 回答了核心问题，但有少量偏题
- 0.4~0.6: 部分回答了问题，但有明显遗漏
- 0.0~0.3: 完全没有回答用户的问题"""


class RAGASEvaluator:
    """RAGAS 评测器

    封装 RAGAS 四大核心指标：
    - context_precision: 检索的文档中有多少真正相关
    - context_recall:    相关文档有多少被检索到
    - faithfulness:      回复是否忠于检索文档
    - answer_relevancy:  回复是否真正回答了问题

    Attributes:
        _settings: Settings 配置实例
    """

    def __init__(self) -> None:
        """初始化 RAGASEvaluator，加载配置。"""
        self._settings = get_settings()

    def _compute_context_precision(
        self,
        contexts: list[str],
        ground_truth: str,
    ) -> float:
        """使用 LLM 判断检索文档的精确度

        Args:
            contexts: 检索到的文档列表
            ground_truth: 标准答案

        Returns:
            精确度分数 ∈ [0.0, 1.0]，LLM 失败时降级返回 0.0
        """
        docs_text = "\n".join(
            f"[文档{i}] {ctx[:300]}" for i, ctx in enumerate(contexts, 1)
        ) or "（无检索文档）"

        human_content = (
            f"检索到的文档：\n{docs_text}\n\n"
            f"标准答案：{ground_truth}"
        )

        try:
            llm = get_graph_llm()
            structured_llm = llm.with_structured_output(ContextPrecisionScore)
            result: ContextPrecisionScore = structured_llm.invoke([
                SystemMessage(content=_CONTEXT_PRECISION_PROMPT),
                HumanMessage(content=human_content),
            ])
            logger.info(f"context_precision 评分: {result.score:.2f}")
            return result.score
        except Exception as e:
            logger.error(f"context_precision 评分失败，降级返回 0.0: {e}", exc_info=True)
            return 0.0

    def _compute_context_recall(
        self,
        contexts: list[str],
        ground_truth: str,
    ) -> float:
        """使用 LLM 判断相关文档的召回率

        Args:
            contexts: 检索到的文档列表
            ground_truth: 标准答案

        Returns:
            召回率分数 ∈ [0.0, 1.0]，LLM 失败时降级返回 0.0
        """
        docs_text = "\n".join(
            f"[文档{i}] {ctx[:300]}" for i, ctx in enumerate(contexts, 1)
        ) or "（无检索文档）"

        human_content = (
            f"检索到的文档：\n{docs_text}\n\n"
            f"标准答案：{ground_truth}"
        )

        try:
            llm = get_graph_llm()
            structured_llm = llm.with_structured_output(ContextRecallScore)
            result: ContextRecallScore = structured_llm.invoke([
                SystemMessage(content=_CONTEXT_RECALL_PROMPT),
                HumanMessage(content=human_content),
            ])
            logger.info(f"context_recall 评分: {result.score:.2f}")
            return result.score
        except Exception as e:
            logger.error(f"context_recall 评分失败，降级返回 0.0: {e}", exc_info=True)
            return 0.0

    def _compute_faithfulness(
        self,
        answer: str,
        contexts: list[str],
    ) -> float:
        """使用 LLM 检测回复的忠实度

        Args:
            answer: 助手生成的回复文本
            contexts: 参考文档列表

        Returns:
            忠实度分数 ∈ [0.0, 1.0]，LLM 失败时降级返回 0.0
        """
        docs_text = "\n".join(
            f"[文档{i}] {ctx[:300]}" for i, ctx in enumerate(contexts, 1)
        ) or "（无参考文档）"

        human_content = (
            f"参考文档：\n{docs_text}\n\n"
            f"助手回复：\n{answer}"
        )

        try:
            llm = get_graph_llm()
            structured_llm = llm.with_structured_output(FaithfulnessScore)
            result: FaithfulnessScore = structured_llm.invoke([
                SystemMessage(content=_FAITHFULNESS_PROMPT),
                HumanMessage(content=human_content),
            ])
            logger.info(f"faithfulness 评分: {result.score:.2f}")
            return result.score
        except Exception as e:
            logger.error(f"faithfulness 评分失败，降级返回 0.0: {e}", exc_info=True)
            return 0.0

    def _compute_answer_relevancy(
        self,
        query: str,
        answer: str,
    ) -> float:
        """使用 LLM 判断回复的相关性

        Args:
            query: 用户查询文本
            answer: 助手生成的回复文本

        Returns:
            相关性分数 ∈ [0.0, 1.0]，LLM 失败时降级返回 0.0
        """
        human_content = (
            f"用户问题：{query}\n\n"
            f"助手回复：\n{answer}"
        )

        try:
            llm = get_graph_llm()
            structured_llm = llm.with_structured_output(AnswerRelevancyScore)
            result: AnswerRelevancyScore = structured_llm.invoke([
                SystemMessage(content=_ANSWER_RELEVANCY_PROMPT),
                HumanMessage(content=human_content),
            ])
            logger.info(f"answer_relevancy 评分: {result.score:.2f}")
            return result.score
        except Exception as e:
            logger.error(f"answer_relevancy 评分失败，降级返回 0.0: {e}", exc_info=True)
            return 0.0

    def evaluate_single(
        self,
        query: str,
        answer: str,
        contexts: list[str],
        ground_truth: str,
    ) -> dict[str, float]:
        """单条评测，返回四个指标分数

        Args:
            query: 用户查询文本
            answer: 助手生成的回复文本
            contexts: 检索到的文档列表
            ground_truth: 标准答案

        Returns:
            包含 context_precision, context_recall, faithfulness, answer_relevancy 的 dict
        """
        logger.info(f"RAGAS 单条评测: query='{query[:50]}'")
        return {
            "context_precision": self._compute_context_precision(contexts, ground_truth),
            "context_recall": self._compute_context_recall(contexts, ground_truth),
            "faithfulness": self._compute_faithfulness(answer, contexts),
            "answer_relevancy": self._compute_answer_relevancy(query, answer),
        }

    def evaluate_batch(self, dataset: list[dict]) -> dict[str, float]:
        """批量评测，返回平均分数

        Args:
            dataset: 列表，每个元素包含 query, answer, contexts, ground_truth

        Returns:
            四个指标的平均值 dict
        """
        if not dataset:
            return {
                "context_precision": 0.0,
                "context_recall": 0.0,
                "faithfulness": 0.0,
                "answer_relevancy": 0.0,
            }

        totals: dict[str, float] = {
            "context_precision": 0.0,
            "context_recall": 0.0,
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
        }

        for item in dataset:
            metrics = self.evaluate_single(
                query=item.get("query", ""),
                answer=item.get("answer", ""),
                contexts=item.get("contexts", []),
                ground_truth=item.get("ground_truth", ""),
            )
            for key in totals:
                totals[key] += metrics[key]

        n = len(dataset)
        return {key: val / n for key, val in totals.items()}


# 模块级单例
_ragas_evaluator: Optional[RAGASEvaluator] = None


def get_ragas_evaluator() -> RAGASEvaluator:
    """获取 RAGASEvaluator 单例。

    Returns:
        RAGASEvaluator 实例
    """
    global _ragas_evaluator
    if _ragas_evaluator is None:
        _ragas_evaluator = RAGASEvaluator()
        logger.info("RAGASEvaluator 初始化完成")
    return _ragas_evaluator
