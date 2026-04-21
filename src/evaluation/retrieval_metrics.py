"""
检索质量指标计算模块 (Phase 2)

实现三个经典 IR 指标（纯数学计算，无 LLM 依赖）：
- Recall@K: 前 K 个结果中包含多少相关文档
- MRR (Mean Reciprocal Rank): 第一个相关文档的位置倒数的均值
- nDCG (normalized Discounted Cumulative Gain): 排序质量

复用模块:
- src/utils/logger.py::get_logger()
"""

import math
from typing import Optional

from ..utils.logger import get_logger


logger = get_logger(__name__)


class RetrievalMetrics:
    """检索质量指标计算器

    实现三个经典 IR 指标（纯数学计算，无 LLM 依赖）：
    - Recall@K: 前 K 个结果中包含多少相关文档
    - MRR (Mean Reciprocal Rank): 第一个相关文档的位置倒数的均值
    - nDCG (normalized Discounted Cumulative Gain): 排序质量
    """

    @staticmethod
    def recall_at_k(
        retrieved_ids: list[str],
        relevant_ids: list[str],
        k: int = 5,
    ) -> float:
        """计算 Recall@K

        前 K 个检索结果中，包含的相关文档占全部相关文档的比例。

        Args:
            retrieved_ids: 检索返回的文档 ID 列表（按排序）
            relevant_ids: 标注的相关文档 ID 列表
            k: 取前 K 个结果

        Returns:
            Recall@K 分数，∈ [0.0, 1.0]

        Examples:
            >>> RetrievalMetrics.recall_at_k(["r1","r2","r3"], ["r1","r4"], k=3)
            0.5
        """
        if not retrieved_ids or not relevant_ids:
            return 0.0
        top_k = set(retrieved_ids[:k])
        relevant_set = set(relevant_ids)
        hits = len(top_k & relevant_set)
        return hits / len(relevant_set)

    @staticmethod
    def mrr(
        retrieved_ids: list[str],
        relevant_ids: list[str],
    ) -> float:
        """计算 MRR (Mean Reciprocal Rank)

        第一个相关文档位置的倒数。对于单条查询即 Reciprocal Rank。

        Args:
            retrieved_ids: 检索返回的文档 ID 列表（按排序）
            relevant_ids: 标注的相关文档 ID 列表

        Returns:
            MRR 分数，∈ [0.0, 1.0]

        Examples:
            >>> RetrievalMetrics.mrr(["r1","r2","r3"], ["r1"])
            1.0
            >>> RetrievalMetrics.mrr(["r1","r2","r3"], ["r2"])
            0.5
        """
        if not retrieved_ids or not relevant_ids:
            return 0.0
        relevant_set = set(relevant_ids)
        for i, doc_id in enumerate(retrieved_ids, 1):
            if doc_id in relevant_set:
                return 1.0 / i
        return 0.0

    @staticmethod
    def ndcg(
        retrieved_ids: list[str],
        relevant_ids: list[str],
        k: int = 5,
    ) -> float:
        """计算 nDCG@K (normalized Discounted Cumulative Gain)

        标准公式：DCG_k / IDCG_k，相关文档 relevance = 1，非相关 = 0。

        Args:
            retrieved_ids: 检索返回的文档 ID 列表（按排序）
            relevant_ids: 标注的相关文档 ID 列表
            k: 取前 K 个结果

        Returns:
            nDCG@K 分数，∈ [0.0, 1.0]

        Examples:
            >>> RetrievalMetrics.ndcg(["r1","r2"], ["r1","r2"], k=2)
            1.0
        """
        if not retrieved_ids or not relevant_ids:
            return 0.0

        relevant_set = set(relevant_ids)

        # 计算 DCG@K
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_ids[:k], 1):
            rel = 1.0 if doc_id in relevant_set else 0.0
            dcg += rel / math.log2(i + 1)

        # 计算 IDCG@K（理想排序：全部相关文档排在最前面）
        ideal_count = min(len(relevant_ids), k)
        idcg = 0.0
        for i in range(1, ideal_count + 1):
            idcg += 1.0 / math.log2(i + 1)

        if idcg == 0.0:
            return 0.0

        return dcg / idcg

    @classmethod
    def compute_all(
        cls,
        retrieved_ids: list[str],
        relevant_ids: list[str],
        k: int = 5,
    ) -> dict[str, float]:
        """一次性计算全部检索指标

        Args:
            retrieved_ids: 检索返回的文档 ID 列表（按排序）
            relevant_ids: 标注的相关文档 ID 列表
            k: K 值

        Returns:
            包含 recall_at_k, mrr, ndcg 三个指标的 dict
        """
        return {
            "recall_at_k": cls.recall_at_k(retrieved_ids, relevant_ids, k),
            "mrr": cls.mrr(retrieved_ids, relevant_ids),
            "ndcg": cls.ndcg(retrieved_ids, relevant_ids, k),
        }

    @classmethod
    def compute_batch(
        cls,
        results: list[dict],
        k: int = 5,
    ) -> dict[str, float]:
        """批量计算检索指标并取平均

        Args:
            results: 列表，每个元素为 {"retrieved_ids": [...], "relevant_ids": [...]}
            k: K 值

        Returns:
            全部指标的平均值 dict
        """
        if not results:
            return {"recall_at_k": 0.0, "mrr": 0.0, "ndcg": 0.0}

        totals: dict[str, float] = {"recall_at_k": 0.0, "mrr": 0.0, "ndcg": 0.0}
        for item in results:
            metrics = cls.compute_all(
                retrieved_ids=item.get("retrieved_ids", []),
                relevant_ids=item.get("relevant_ids", []),
                k=k,
            )
            for key in totals:
                totals[key] += metrics[key]

        n = len(results)
        return {key: val / n for key, val in totals.items()}
