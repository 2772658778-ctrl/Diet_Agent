"""
生成质量指标评估模块 (Phase 2)

使用 LLM-as-Judge 模式评估生成回复质量：
- faithfulness:     回复是否忠于参考文档（与 SelfRAGJudge.judge_hallucination 互补）
- answer_relevancy: 回复是否直接回答了用户问题
- completeness:     回复是否覆盖了关键信息点

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
    score: float = Field(ge=0.0, le=1.0, description="相关性分数，0.0 到 1.0")
    reasoning: str = Field(default="", description="评分理由")


# ── 系统提示 ────────────────────────────────────────────────────────────────────

_FAITHFULNESS_PROMPT = """你是一个回复忠实度评估专家。
请评估以下饮食助手的回复是否忠实于参考文档。

评分标准（0.0 到 1.0）：
- 1.0: 回复中所有声明都被参考文档支持
- 0.7~0.9: 大部分声明有文档支持，少数为合理推断
- 0.4~0.6: 部分声明缺乏文档支持
- 0.0~0.3: 回复大量内容没有文档依据，存在严重幻觉

请返回一个连续分数（float），而非 True/False。"""

_ANSWER_RELEVANCY_PROMPT = """你是一个回复相关性评估专家。
请评估以下饮食助手的回复是否直接回答了用户的问题。

评分标准（0.0 到 1.0）：
- 1.0: 完全直接地回答了用户问题
- 0.7~0.9: 回答了核心问题，但有少量偏题
- 0.4~0.6: 部分回答了问题，但有明显遗漏或偏题
- 0.0~0.3: 完全没有回答用户的问题

请返回一个连续分数（float），而非 True/False。"""


class GenerationMetrics:
    """生成质量指标评估器

    使用 LLM-as-Judge 模式评估生成回复质量：
    - faithfulness:     回复是否忠于参考文档（与 SelfRAGJudge.judge_hallucination 互补）
    - answer_relevancy: 回复是否直接回答了用户问题
    - completeness:     回复是否覆盖了关键信息点

    复用模块:
    - src/graph/llm.py::get_graph_llm()
    - src/utils/logger.py::get_logger()
    """

    def __init__(self) -> None:
        """初始化 GenerationMetrics，加载配置。"""
        self._settings = get_settings()

    def score_faithfulness(self, answer: str, contexts: list[str]) -> float:
        """评估回复忠实度

        使用 LLM-as-Judge 判断回复是否忠于参考文档，返回 0.0~1.0 连续分数。

        Args:
            answer: 助手生成的回复文本
            contexts: 参考文档列表

        Returns:
            忠实度分数 ∈ [0.0, 1.0]，LLM 失败时降级返回 0.0
        """
        logger.info(f"评估 faithfulness: answer_len={len(answer)}, contexts={len(contexts)}")

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

    def score_answer_relevancy(self, query: str, answer: str) -> float:
        """评估回复相关性

        使用 LLM-as-Judge 判断回复是否直接回答了用户问题。

        Args:
            query: 用户查询文本
            answer: 助手生成的回复文本

        Returns:
            相关性分数 ∈ [0.0, 1.0]，LLM 失败时降级返回 0.0
        """
        logger.info(f"评估 answer_relevancy: query='{query[:50]}'")

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

    def score_completeness(
        self,
        query: str,
        answer: str,
        ground_truth_keywords: list[str],
    ) -> float:
        """评估回复完整性（关键词覆盖率）

        纯关键词匹配实现，不依赖 LLM，简洁可靠。

        Args:
            query: 用户查询文本（未使用，保留接口一致性）
            answer: 助手生成的回复文本
            ground_truth_keywords: 标注的关键词列表

        Returns:
            关键词覆盖率 ∈ [0.0, 1.0]

        Examples:
            >>> gm = GenerationMetrics()
            >>> gm.score_completeness("", "番茄炒蛋15分钟", ["番茄", "鸡蛋", "分钟"])
            0.666...
        """
        if not ground_truth_keywords:
            return 1.0

        answer_lower = answer.lower()
        hits = sum(1 for kw in ground_truth_keywords if kw.lower() in answer_lower)
        score = hits / len(ground_truth_keywords)
        logger.info(
            f"completeness 评分: {score:.2f} ({hits}/{len(ground_truth_keywords)} 关键词命中)"
        )
        return score

    def evaluate_single(
        self,
        query: str,
        answer: str,
        contexts: list[str],
        ground_truth_keywords: Optional[list[str]] = None,
    ) -> dict[str, float]:
        """一次性评估全部生成指标

        Args:
            query: 用户查询文本
            answer: 助手生成的回复文本
            contexts: 参考文档列表
            ground_truth_keywords: 标注关键词列表（可选）

        Returns:
            包含 faithfulness, answer_relevancy, completeness 三个指标的 dict
        """
        return {
            "faithfulness": self.score_faithfulness(answer, contexts),
            "answer_relevancy": self.score_answer_relevancy(query, answer),
            "completeness": self.score_completeness(
                query, answer, ground_truth_keywords or []
            ),
        }

    def evaluate_batch(self, dataset: list[dict]) -> dict[str, float]:
        """批量评估生成指标并取平均

        Args:
            dataset: 列表，每个元素包含 query, answer, contexts, ground_truth_keywords

        Returns:
            全部指标的平均值 dict
        """
        if not dataset:
            return {"faithfulness": 0.0, "answer_relevancy": 0.0, "completeness": 0.0}

        totals: dict[str, float] = {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "completeness": 0.0,
        }

        for item in dataset:
            metrics = self.evaluate_single(
                query=item.get("query", ""),
                answer=item.get("answer", ""),
                contexts=item.get("contexts", []),
                ground_truth_keywords=item.get("ground_truth_keywords", []),
            )
            for key in totals:
                totals[key] += metrics[key]

        n = len(dataset)
        return {key: val / n for key, val in totals.items()}


# 模块级单例
_generation_metrics: Optional[GenerationMetrics] = None


def get_generation_metrics() -> GenerationMetrics:
    """获取 GenerationMetrics 单例。

    Returns:
        GenerationMetrics 实例
    """
    global _generation_metrics
    if _generation_metrics is None:
        _generation_metrics = GenerationMetrics()
        logger.info("GenerationMetrics 初始化完成")
    return _generation_metrics
