"""
Embedding 模型对比评测模块

在中文食谱领域对比多个 Embedding 模型的检索效果：
- 检索准确率（Recall@K, MRR, nDCG）
- 推理延迟
- embedding 维度

复用模块:
- src/vectorstore/chroma_client.py — ChromaDB 操作
- src/config.py::get_settings()
- src/utils/logger.py::get_logger()
"""

import time
from typing import Any, Optional

from ..config import get_settings
from ..utils.logger import get_logger


logger = get_logger(__name__)


MODELS_TO_COMPARE: list[dict[str, str]] = [
    {
        "name": "text-embedding-v1",
        "provider": "dashscope",
        "description": "阿里当前版本",
    },
    {
        "name": "text-embedding-v3",
        "provider": "dashscope",
        "description": "阿里最新版",
    },
    {
        "name": "BAAI/bge-large-zh-v1.5",
        "provider": "huggingface",
        "description": "中文 SOTA 开源",
    },
    {
        "name": "BAAI/bge-m3",
        "provider": "huggingface",
        "description": "多语言稠密+稀疏",
    },
]

# 内置评测数据集（20 条食谱搜索查询）
_BUILTIN_QUERIES: list[dict[str, Any]] = [
    {"query": "番茄炒蛋怎么做", "relevant_keywords": ["番茄", "鸡蛋", "炒"]},
    {"query": "减肥低卡路里食谱", "relevant_keywords": ["低卡", "减脂", "健康"]},
    {"query": "快手20分钟早餐", "relevant_keywords": ["早餐", "快手", "简单"]},
    {"query": "高蛋白增肌餐", "relevant_keywords": ["蛋白质", "增肌", "健身"]},
    {"query": "素食蔬菜料理", "relevant_keywords": ["素食", "蔬菜", "素菜"]},
    {"query": "酸辣开胃菜", "relevant_keywords": ["酸", "辣", "开胃"]},
    {"query": "儿童营养便当", "relevant_keywords": ["儿童", "营养", "便当"]},
    {"query": "冬季暖身汤", "relevant_keywords": ["汤", "暖身", "冬天"]},
    {"query": "无麸质饮食食谱", "relevant_keywords": ["无麸质", "过敏"]},
    {"query": "豆腐料理方法", "relevant_keywords": ["豆腐", "大豆", "豆制品"]},
    {"query": "鸡胸肉健康做法", "relevant_keywords": ["鸡胸", "鸡肉", "低脂"]},
    {"query": "宵夜简单小吃", "relevant_keywords": ["宵夜", "简单", "小吃"]},
    {"query": "家常红烧肉", "relevant_keywords": ["红烧肉", "猪肉", "红烧"]},
    {"query": "清淡老人食谱", "relevant_keywords": ["清淡", "老人", "少油少盐"]},
    {"query": "夏天清凉解暑菜", "relevant_keywords": ["清凉", "夏天", "解暑"]},
    {"query": "糖尿病患者饮食", "relevant_keywords": ["糖尿病", "低糖", "控糖"]},
    {"query": "下饭香辣菜", "relevant_keywords": ["下饭", "香辣", "辣"]},
    {"query": "炒饭创意食谱", "relevant_keywords": ["炒饭", "米饭", "蛋炒饭"]},
    {"query": "海鲜料理推荐", "relevant_keywords": ["海鲜", "鱼", "虾", "螃蟹"]},
    {"query": "蛋糕烘焙入门", "relevant_keywords": ["蛋糕", "烘焙", "甜点"]},
]


class EmbeddingBenchmark:
    """Embedding 模型对比评测

    在中文食谱领域评测多个 Embedding 模型的检索效果：
    - 检索准确率（Recall@K, MRR）
    - 推理延迟
    - embedding 维度

    复用模块:
    - src/vectorstore/chroma_client.py — ChromaDB 操作
    - src/config.py::get_settings()
    - src/utils/logger.py::get_logger()
    """

    def __init__(self, recipes: Optional[list[dict]] = None) -> None:
        """初始化评测器。

        Args:
            recipes: 食谱列表，用于构建评测 ground truth；
                     为 None 时从 ChromaDB 查询。
        """
        self._settings = get_settings()
        self._recipes = recipes or []

    def build_test_dataset(self) -> list[dict[str, Any]]:
        """构建评测数据集。

        使用内置的 20 条查询 + 关键词匹配生成 ground truth。
        如果提供了食谱数据，则基于关键词匹配构建相关/不相关食谱列表。

        Returns:
            评测数据集列表，每条格式为：
            {
                'query': str,
                'relevant_keywords': list[str],
                'relevant_recipe_ids': list[str],   # 仅当有食谱数据时填充
                'irrelevant_recipe_ids': list[str],
            }
        """
        dataset: list[dict[str, Any]] = []

        for item in _BUILTIN_QUERIES:
            entry: dict[str, Any] = {
                "query": item["query"],
                "relevant_keywords": item["relevant_keywords"],
                "relevant_recipe_ids": [],
                "irrelevant_recipe_ids": [],
            }

            if self._recipes:
                keywords = [kw.lower() for kw in item["relevant_keywords"]]
                for recipe in self._recipes:
                    recipe_text = (
                        recipe.get("name", "") + " "
                        + recipe.get("description", "") + " "
                        + " ".join(recipe.get("tags", []))
                    ).lower()
                    recipe_id = recipe.get("id", recipe.get("name", ""))
                    if any(kw in recipe_text for kw in keywords):
                        entry["relevant_recipe_ids"].append(recipe_id)
                    else:
                        entry["irrelevant_recipe_ids"].append(recipe_id)

            dataset.append(entry)

        logger.info(f"评测数据集构建完成: {len(dataset)} 条查询")
        return dataset

    def _get_embedding_model(self, model_name: str, provider: str):
        """根据 provider 创建 embedding 模型实例。

        Args:
            model_name: 模型名称
            provider: 'dashscope' 或 'huggingface'

        Returns:
            embedding 模型实例
        """
        if provider == "dashscope":
            from langchain_community.embeddings import DashScopeEmbeddings
            return DashScopeEmbeddings(
                model=model_name,
                dashscope_api_key=self._settings.dashscope_api_key,
            )
        elif provider == "huggingface":
            from langchain_community.embeddings import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(model_name=model_name)
        else:
            raise ValueError(f"不支持的 provider: {provider}")

    def evaluate_model(
        self,
        model_name: str,
        provider: str,
        test_data: list[dict[str, Any]],
        top_k: int = 5,
    ) -> dict[str, Any]:
        """对单个模型计算评测指标。

        Args:
            model_name: 模型名称
            provider: 模型来源 ('dashscope' / 'huggingface')
            test_data: 评测数据集（来自 build_test_dataset()）
            top_k: 检索返回的文档数量

        Returns:
            评测指标字典：
            {
                'model_name': str,
                'provider': str,
                'recall_at_5': float,
                'recall_at_10': float,
                'mrr': float,
                'avg_latency_ms': float,
                'error': str or None,
            }
        """
        logger.info(f"开始评测模型: {model_name} ({provider})")

        result: dict[str, Any] = {
            "model_name": model_name,
            "provider": provider,
            "recall_at_5": 0.0,
            "recall_at_10": 0.0,
            "mrr": 0.0,
            "avg_latency_ms": 0.0,
            "error": None,
        }

        if not test_data or not self._recipes:
            logger.warning("评测数据或食谱数据为空，跳过评测")
            return result

        try:
            embedding_model = self._get_embedding_model(model_name, provider)

            latencies: list[float] = []
            recall5_scores: list[float] = []
            recall10_scores: list[float] = []
            mrr_scores: list[float] = []

            # 构建食谱向量索引
            recipe_texts = [
                (r.get("name", "") + " " + r.get("description", ""))
                for r in self._recipes
            ]
            recipe_ids = [r.get("id", r.get("name", "")) for r in self._recipes]

            recipe_embeddings = embedding_model.embed_documents(recipe_texts)

            import numpy as np

            for item in test_data:
                query = item["query"]
                relevant_ids = set(item.get("relevant_recipe_ids", []))
                if not relevant_ids:
                    continue

                # 计算查询向量并排序
                t0 = time.time()
                query_emb = embedding_model.embed_query(query)
                latencies.append((time.time() - t0) * 1000)

                query_vec = np.array(query_emb)
                doc_vecs = np.array(recipe_embeddings)
                scores = np.dot(doc_vecs, query_vec) / (
                    np.linalg.norm(doc_vecs, axis=1) * np.linalg.norm(query_vec) + 1e-9
                )

                ranked_ids = [
                    recipe_ids[i]
                    for i in np.argsort(scores)[::-1]
                ]

                # Recall@5
                top5 = set(ranked_ids[:5])
                recall5_scores.append(
                    len(top5 & relevant_ids) / max(len(relevant_ids), 1)
                )

                # Recall@10
                top10 = set(ranked_ids[:10])
                recall10_scores.append(
                    len(top10 & relevant_ids) / max(len(relevant_ids), 1)
                )

                # MRR
                mrr = 0.0
                for rank, rid in enumerate(ranked_ids[:10], 1):
                    if rid in relevant_ids:
                        mrr = 1.0 / rank
                        break
                mrr_scores.append(mrr)

            def _avg(lst: list[float]) -> float:
                return round(sum(lst) / len(lst), 4) if lst else 0.0

            result["recall_at_5"] = _avg(recall5_scores)
            result["recall_at_10"] = _avg(recall10_scores)
            result["mrr"] = _avg(mrr_scores)
            result["avg_latency_ms"] = _avg(latencies)

            logger.info(
                f"模型 {model_name} 评测完成: "
                f"Recall@5={result['recall_at_5']}, "
                f"MRR={result['mrr']}, "
                f"latency={result['avg_latency_ms']}ms"
            )

        except Exception as e:
            logger.error(f"评测模型 {model_name} 失败: {e}", exc_info=True)
            result["error"] = str(e)

        return result

    def run_benchmark(self, top_k: int = 5) -> list[dict[str, Any]]:
        """遍历所有模型，执行完整评测。

        Args:
            top_k: 检索返回文档数量

        Returns:
            所有模型的评测结果列表
        """
        logger.info(f"开始运行 Embedding Benchmark，共 {len(MODELS_TO_COMPARE)} 个模型")
        test_data = self.build_test_dataset()
        results: list[dict[str, Any]] = []

        for model_info in MODELS_TO_COMPARE:
            result = self.evaluate_model(
                model_name=model_info["name"],
                provider=model_info["provider"],
                test_data=test_data,
                top_k=top_k,
            )
            result["description"] = model_info.get("description", "")
            results.append(result)

        logger.info("Embedding Benchmark 运行完成")
        return results

    def generate_report(self, results: list[dict[str, Any]]) -> str:
        """生成 Markdown 格式的评测报告。

        Args:
            results: run_benchmark() 返回的结果列表

        Returns:
            Markdown 格式报告字符串
        """
        if not results:
            return "## Embedding Benchmark 报告\n\n暂无评测结果。\n"

        lines = [
            "## Embedding 模型对比评测报告",
            "",
            "### 评测场景",
            "- 领域：中文食谱检索",
            f"- 查询数量：{len(_BUILTIN_QUERIES)} 条",
            "- 指标：Recall@5, Recall@10, MRR, 平均延迟",
            "",
            "### 模型对比",
            "",
            "| 模型 | 提供商 | 说明 | Recall@5 | Recall@10 | MRR | 延迟(ms) | 状态 |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]

        for r in results:
            status = "❌ " + r.get("error", "")[:30] if r.get("error") else "✅"
            lines.append(
                f"| {r['model_name']} "
                f"| {r['provider']} "
                f"| {r.get('description', '')} "
                f"| {r['recall_at_5']:.4f} "
                f"| {r['recall_at_10']:.4f} "
                f"| {r['mrr']:.4f} "
                f"| {r['avg_latency_ms']:.1f} "
                f"| {status} |"
            )

        # 找出最佳模型
        valid = [r for r in results if not r.get("error")]
        if valid:
            best = max(valid, key=lambda x: x["recall_at_5"])
            lines += [
                "",
                "### 推荐",
                "",
                f"**最佳 Recall@5 模型**: `{best['model_name']}` "
                f"(Recall@5={best['recall_at_5']:.4f}，"
                f"MRR={best['mrr']:.4f})",
            ]

        lines += ["", "---", "_报告由 EmbeddingBenchmark 自动生成_"]
        return "\n".join(lines)
