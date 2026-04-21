"""
端到端 Benchmark 模块 (Phase 2)

运行完整 Diet Agent 管线，收集各环节指标，生成 A/B 对比报告。

复用模块:
- src/graph/diet_graph.py::build_diet_graph()
- src/evaluation/ragas_eval.py::RAGASEvaluator
- src/evaluation/retrieval_metrics.py::RetrievalMetrics
- src/evaluation/generation_metrics.py::GenerationMetrics
- src/config.py::get_settings()
- src/utils/logger.py::get_logger()
"""

import json
import os
import statistics
import time
from typing import Optional

from diet_agent.runtime import build_diet_graph

from langchain_core.messages import HumanMessage

from ..config import get_settings
from ..utils.logger import get_logger
from .ragas_eval import RAGASEvaluator
from .retrieval_metrics import RetrievalMetrics
from .generation_metrics import GenerationMetrics


logger = get_logger(__name__)


class E2EBenchmark:
    """端到端评测 Benchmark

    加载测试数据集，运行完整 Diet Agent 管线，
    收集检索/生成/端到端指标，并支持多版本 A/B 对比。

    复用模块:
    - src/graph/diet_graph.py::build_diet_graph()
    - src/evaluation/ 三个子模块
    """

    def __init__(
        self,
        dataset_path: str = "src/evaluation/test_dataset.json",
    ) -> None:
        """初始化 E2EBenchmark

        Args:
            dataset_path: 测试数据集 JSON 文件路径
        """
        self._settings = get_settings()
        self._dataset_path = dataset_path
        self._dataset: list[dict] = []
        self._ragas = RAGASEvaluator()
        self._gen_metrics = GenerationMetrics()

        # 尝试加载数据集
        if os.path.exists(dataset_path):
            self._dataset = self.load_dataset(dataset_path)
            logger.info(f"E2EBenchmark 初始化完成，数据集 {len(self._dataset)} 条")
        else:
            logger.warning(f"数据集文件不存在: {dataset_path}，请稍后手动加载")

    def load_dataset(self, path: str) -> list[dict]:
        """读取 JSON 测试数据集

        Args:
            path: JSON 文件路径

        Returns:
            测试数据列表
        """
        logger.info(f"加载评测数据集: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"数据集加载完成: {len(data)} 条")
        return data

    def run_single(self, test_case: dict) -> dict:
        """运行单条测试并收集指标

        调用 build_diet_graph().invoke(initial_state) 获得最终 state，
        从 state 提取 retrieved_docs, reranked_docs, response，
        然后分别计算检索指标、生成指标和 RAGAS 指标。

        Args:
            test_case: 单条测试数据，需包含 query, ground_truth_doc_ids,
                       ground_truth_answer_keywords, ground_truth_answer

        Returns:
            包含 retrieval_metrics, generation_metrics, ragas_metrics, latency 的 dict
        """
        query = test_case.get("query", "")
        ground_truth_doc_ids = test_case.get("ground_truth_doc_ids", [])
        ground_truth_keywords = test_case.get("ground_truth_answer_keywords", [])
        ground_truth_answer = test_case.get("ground_truth_answer", "")

        logger.info(f"E2E 单条测试: query='{query[:50]}'")

        # 构建初始 state 并运行图
        start_time = time.perf_counter()

        try:
            graph = build_diet_graph()
            initial_state = {
                "messages": [HumanMessage(content=query)],
                "intent": "",
                "confidence": 0.0,
                "extracted_params": {},
                "plan": [],
                "retrieved_docs": [],
                "reranked_docs": [],
                "response": "",
                "evaluation": {},
                "current_step": 0,
                "retry_count": 0,
                "query_complexity": "",
                "transformed_queries": [],
                "self_rag_judgements": {},
                "retrieval_strategy": "standard",
                "eval_metrics": {},
            }
            result_state = graph.invoke(initial_state)
        except Exception as e:
            logger.error(f"E2E 图运行失败: {e}", exc_info=True)
            return {
                "query": query,
                "retrieval_metrics": {"recall_at_k": 0.0, "mrr": 0.0, "ndcg": 0.0},
                "generation_metrics": {
                    "faithfulness": 0.0,
                    "answer_relevancy": 0.0,
                    "completeness": 0.0,
                },
                "ragas_metrics": {
                    "context_precision": 0.0,
                    "context_recall": 0.0,
                    "faithfulness": 0.0,
                    "answer_relevancy": 0.0,
                },
                "latency": time.perf_counter() - start_time,
                "error": str(e),
            }

        latency = time.perf_counter() - start_time

        # 提取结果
        retrieved_docs = result_state.get("retrieved_docs", [])
        reranked_docs = result_state.get("reranked_docs", [])
        response = result_state.get("response", "")

        # 使用精排后的文档计算检索指标；如果没有精排则用原始检索
        eval_docs = reranked_docs if reranked_docs else retrieved_docs
        retrieved_ids = [doc.get("id", "") for doc in eval_docs]

        # 检索指标（纯计算）
        k = self._settings.eval_retrieval_k
        retrieval_metrics = RetrievalMetrics.compute_all(
            retrieved_ids=retrieved_ids,
            relevant_ids=ground_truth_doc_ids,
            k=k,
        )

        # 生成指标（LLM-as-Judge + 关键词匹配）
        contexts = []
        for doc in eval_docs[:5]:
            text = doc.get("text") or doc.get("name", str(doc))[:300]
            contexts.append(text)

        generation_metrics = self._gen_metrics.evaluate_single(
            query=query,
            answer=response,
            contexts=contexts,
            ground_truth_keywords=ground_truth_keywords,
        )

        # RAGAS 四大指标
        ragas_metrics = self._ragas.evaluate_single(
            query=query,
            answer=response,
            contexts=contexts,
            ground_truth=ground_truth_answer,
        )

        return {
            "query": query,
            "response": response,
            "retrieval_metrics": retrieval_metrics,
            "generation_metrics": generation_metrics,
            "ragas_metrics": ragas_metrics,
            "latency": latency,
        }

    def run_all(self) -> dict:
        """运行全部测试集，返回汇总指标

        Returns:
            包含 avg_retrieval, avg_generation, avg_ragas, avg_latency 和各条 details 的 dict
        """
        if not self._dataset:
            logger.warning("数据集为空，请先加载数据集")
            return {}

        logger.info(f"开始 E2E 全量评测: {len(self._dataset)} 条")
        details: list[dict] = []

        for i, case in enumerate(self._dataset, 1):
            logger.info(f"进度: {i}/{len(self._dataset)}")
            result = self.run_single(case)
            details.append(result)

        # 汇总
        retrieval_keys = ["recall_at_k", "mrr", "ndcg"]
        generation_keys = ["faithfulness", "answer_relevancy", "completeness"]
        ragas_keys = ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]
        latencies = [d["latency"] for d in details]

        avg_retrieval = {
            key: statistics.mean([d["retrieval_metrics"].get(key, 0.0) for d in details])
            for key in retrieval_keys
        }
        avg_generation = {
            key: statistics.mean([d["generation_metrics"].get(key, 0.0) for d in details])
            for key in generation_keys
        }
        avg_ragas = {
            key: statistics.mean([d["ragas_metrics"].get(key, 0.0) for d in details])
            for key in ragas_keys
        }

        summary = {
            "total_cases": len(self._dataset),
            "avg_retrieval": avg_retrieval,
            "avg_generation": avg_generation,
            "avg_ragas": avg_ragas,
            "avg_latency": statistics.mean(latencies),
            "std_latency": statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
            "details": details,
        }

        logger.info(
            f"E2E 全量评测完成: avg_recall@k={avg_retrieval['recall_at_k']:.3f}, "
            f"avg_faithfulness={avg_ragas['faithfulness']:.3f}, "
            f"avg_latency={summary['avg_latency']:.2f}s"
        )
        return summary

    def generate_report(
        self,
        results: dict,
        version_label: str = "V4",
    ) -> str:
        """生成 Markdown 格式评测报告

        Args:
            results: run_all() 返回的汇总结果
            version_label: 版本标签

        Returns:
            Markdown 格式的报告字符串
        """
        if not results:
            return "# 评测报告\n\n无评测数据。"

        avg_r = results.get("avg_retrieval", {})
        avg_g = results.get("avg_generation", {})
        avg_ragas = results.get("avg_ragas", {})

        report = f"""# {version_label} 评测报告

## 概览
- **测试用例数**: {results.get('total_cases', 0)}
- **平均延迟**: {results.get('avg_latency', 0):.2f}s (±{results.get('std_latency', 0):.2f}s)

## 检索指标

| 指标 | 分数 |
|------|------|
| Recall@K | {avg_r.get('recall_at_k', 0):.4f} |
| MRR | {avg_r.get('mrr', 0):.4f} |
| nDCG | {avg_r.get('ndcg', 0):.4f} |

## 生成指标

| 指标 | 分数 |
|------|------|
| Faithfulness | {avg_g.get('faithfulness', 0):.4f} |
| Answer Relevancy | {avg_g.get('answer_relevancy', 0):.4f} |
| Completeness | {avg_g.get('completeness', 0):.4f} |

## RAGAS 指标

| 指标 | 分数 |
|------|------|
| Context Precision | {avg_ragas.get('context_precision', 0):.4f} |
| Context Recall | {avg_ragas.get('context_recall', 0):.4f} |
| Faithfulness | {avg_ragas.get('faithfulness', 0):.4f} |
| Answer Relevancy | {avg_ragas.get('answer_relevancy', 0):.4f} |
"""
        return report

    @staticmethod
    def compare_versions(
        results_a: dict,
        results_b: dict,
        label_a: str = "V3",
        label_b: str = "V4",
    ) -> str:
        """生成 A/B 版本对比报告

        Args:
            results_a: 版本 A 的 run_all() 结果
            results_b: 版本 B 的 run_all() 结果
            label_a: 版本 A 标签
            label_b: 版本 B 标签

        Returns:
            Markdown 格式的对比表格字符串
        """

        def _get(results: dict, section: str, key: str) -> float:
            return results.get(section, {}).get(key, 0.0)

        def _delta(a: float, b: float) -> str:
            if a == 0:
                return "N/A"
            pct = (b - a) / a * 100
            sign = "+" if pct >= 0 else ""
            return f"{sign}{pct:.1f}%"

        rows = [
            ("Recall@K", "avg_retrieval", "recall_at_k"),
            ("MRR", "avg_retrieval", "mrr"),
            ("nDCG", "avg_retrieval", "ndcg"),
            ("Faithfulness (Gen)", "avg_generation", "faithfulness"),
            ("Answer Relevancy (Gen)", "avg_generation", "answer_relevancy"),
            ("Completeness", "avg_generation", "completeness"),
            ("Context Precision", "avg_ragas", "context_precision"),
            ("Context Recall", "avg_ragas", "context_recall"),
            ("Faithfulness (RAGAS)", "avg_ragas", "faithfulness"),
            ("Answer Relevancy (RAGAS)", "avg_ragas", "answer_relevancy"),
        ]

        table = f"| 指标 | {label_a} | {label_b} | 提升 |\n"
        table += "|------|-------|-------|------|\n"

        for name, section, key in rows:
            val_a = _get(results_a, section, key)
            val_b = _get(results_b, section, key)
            delta = _delta(val_a, val_b)
            table += f"| {name} | {val_a:.4f} | {val_b:.4f} | {delta} |\n"

        # 延迟对比
        lat_a = results_a.get("avg_latency", 0.0)
        lat_b = results_b.get("avg_latency", 0.0)
        lat_delta = _delta(lat_a, lat_b) if lat_a > 0 else "N/A"
        table += f"| 平均延迟 | {lat_a:.2f}s | {lat_b:.2f}s | {lat_delta} |\n"

        report = f"# {label_a} vs {label_b} 对比报告\n\n{table}"
        return report


if __name__ == "__main__":
    settings = get_settings()
    benchmark = E2EBenchmark(dataset_path=settings.eval_dataset_path)
    results = benchmark.run_all()
    report = benchmark.generate_report(results)
    print(report)
