"""
混合检索器 (V2)

实现向量检索 + 关系推理的混合检索策略
使用 RRF 融合和 MMR 多样性算法优化检索结果
"""

import json
from typing import List, Dict, Any, Optional, Tuple
from langchain_community.vectorstores import Chroma
from ..config import get_settings
from ..utils.logger import get_logger
from .relation_query import RelationQuery

logger = get_logger(__name__)

def _build_stable_recipe_id(item: Dict[str, Any]) -> str:
    explicit_id = str(item.get("id") or "").strip()
    if explicit_id:
        return explicit_id

    name = str(item.get("name") or "").strip()
    if not name:
        return ""

    normalized_name = name.replace(" ", "_").replace("/", "_").replace("\\", "_")
    return f"recipe_{normalized_name}"

class HybridRetriever:
    """混合检索器：向量检索 + 关系推理
    
    结合语义相似度检索和知识图谱关系推理
    使用 RRF 融合多个结果列表，使用 MMR 保证结果多样性
    """
    
    def __init__(self, vectorstore: Chroma):
        """
        初始化混合检索器
        
        Args:
            vectorstore: ChromaDB 向量存储实例
        """
        self.vectorstore = vectorstore
        self.relation_query = RelationQuery(vectorstore)
        self.settings = get_settings()
        self.last_fusion_metadata: Dict[str, Any] = {}
    
    def retrieve(
        self,
        query: str,
        user_context: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
        lambda_param: float = 0.7,
        rrf_k: int = 60
    ) -> List[Dict[str, Any]]:
        """
        混合检索主函数
        
        执行流程：
        1. 向量检索（语义相似）
        2. 关系推理（图遍历）
        3. RRF 融合
        4. MMR 多样性
        
        Args:
            query: 搜索查询文本
            user_context: 用户上下文，包含:
                - time_limit: 时间限制（分钟）
                - difficulty: 难度要求
                - health_goal: 健康目标
                - available_ingredients: 已有食材列表
                - meal_type: 餐次类型
            top_k: 返回结果数量
            lambda_param: MMR 参数（0-1，越大越注重相关性，越小越注重多样性）
            rrf_k: RRF 参数（默认60）
        
        Returns:
            检索结果列表，每项包含完整的元数据
        
        Example:
            >>> retriever = HybridRetriever(vectorstore)
            >>> context = {"health_goal": "减肥", "time_limit": 30}
            >>> results = retriever.retrieve("低热量食谱", context, top_k=5)
            >>> print(results[0]['name'])
            '清蒸鱼'
        """
        if user_context is None:
            user_context = {}
        
        logger.info(f"开始混合检索: query='{query}', context={user_context}, top_k={top_k}")
        
        try:
            query_features = user_context.get("query_features", {}) or {}
            fusion_mode = str(
                user_context.get("fusion_mode")
                or getattr(self.settings, "fusion_mode", "fixed_fusion")
                or "fixed_fusion"
            ).strip() or "fixed_fusion"
            # 1. 向量检索（语义相似）
            logger.info("  步骤 1/4: 向量检索...")
            vector_results = self._vector_search(query, user_context, top_k * 2)
            logger.info(f"    向量检索返回 {len(vector_results)} 条结果")
            
            # 2. 关系推理（图遍历）
            logger.info("  步骤 2/4: 关系推理...")
            relation_results = self._relation_search(user_context, top_k * 2)
            logger.info(f"    关系推理返回 {len(relation_results)} 条结果")
            
            # 3. RRF 融合
            logger.info("  步骤 3/4: RRF 融合...")
            fusion_weights = self._resolve_fusion_weights(query_features, fusion_mode)
            fused_results = self._weighted_reciprocal_rank_fusion(
                vector_results=vector_results,
                relation_results=relation_results,
                weights=fusion_weights,
                k=rrf_k,
                fusion_mode=fusion_mode,
            )
            self.last_fusion_metadata = {
                "fusion_mode": fusion_mode,
                **fusion_weights,
                "vector_candidate_count": len(vector_results),
                "relation_candidate_count": len(relation_results),
            }
            logger.info(
                f"    融合权重: mode={fusion_mode}, "
                f"w_sem={fusion_weights['w_sem']}, w_struct={fusion_weights['w_struct']}"
            )
            logger.info(f"    融合后得到 {len(fused_results)} 条结果")
            
            # 4. MMR 多样性
            logger.info("  步骤 4/4: MMR 多样性...")
            final_results = self._maximal_marginal_relevance(
                fused_results[:top_k * 2],
                lambda_param=lambda_param,
                top_k=top_k
            )
            logger.info(f"    最终返回 {len(final_results)} 条结果")
            
            return final_results
        
        except Exception as e:
            logger.error(f"混合检索失败: {e}", exc_info=True)
            self.last_fusion_metadata = {}
            return []
    
    def _resolve_fusion_weights(
        self,
        query_features: Dict[str, Any],
        fusion_mode: str,
    ) -> Dict[str, float]:
        if fusion_mode != "query_aware_fusion":
            return {"w_struct": 0.5, "w_sem": 0.5}

        constraint_density = float(query_features.get("entity_constraint_density", 0.0) or 0.0)
        semantic_abstraction = float(query_features.get("semantic_abstraction_score", 0.0) or 0.0)
        inventory_signal = float(query_features.get("inventory_signal", 0.0) or 0.0)

        w_struct = 0.35 + 0.35 * constraint_density + 0.20 * inventory_signal - 0.25 * semantic_abstraction
        w_struct = max(0.15, min(0.85, round(w_struct, 4)))
        w_sem = round(1.0 - w_struct, 4)
        return {"w_struct": w_struct, "w_sem": w_sem}
    
    def _vector_search(
        self,
        query: str,
        context: Dict[str, Any],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        向量检索
        
        使用语义相似度检索，支持元数据过滤
        
        Args:
            query: 搜索查询
            context: 用户上下文
            top_k: 返回结果数量
        
        Returns:
            检索结果列表
        """
        try:
            # 构建过滤条件列表（仅添加有效的业务字段，collection 本身只存食谱，无需 entity_type 过滤）
            filter_conditions = []

            # 添加时间限制
            if context.get("max_cooking_time"):
                filter_conditions.append({"time": {"$lte": context["max_cooking_time"]}})

            # 添加难度限制
            if context.get("difficulty"):
                filter_conditions.append({"difficulty": {"$eq": context["difficulty"]}})

            # 组合过滤条件
            if len(filter_conditions) == 0:
                filters = None
            elif len(filter_conditions) == 1:
                filters = filter_conditions[0]
            else:
                filters = {"$and": filter_conditions}
            
            # 执行检索
            results = self.vectorstore.similarity_search(
                query=query,
                k=top_k,
                filter=filters
            )
            
            # 转换为元数据列表（保留 page_content 以供 fast_relevance 使用全文相似度）
            normalized_results: List[Dict[str, Any]] = []
            for result in results:
                metadata = dict(result.metadata or {})
                stable_id = _build_stable_recipe_id(metadata)
                if stable_id and not metadata.get("id"):
                    metadata["id"] = stable_id
                normalized_results.append({**metadata, "text": result.page_content})
            return normalized_results
        
        except Exception as e:
            logger.error(f"向量检索失败: {e}", exc_info=True)
            return []
    
    def _relation_search(
        self,
        context: Dict[str, Any],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        关系推理检索
        
        基于关系的检索，支持多种关系类型
        
        Args:
            context: 用户上下文
            top_k: 返回结果数量
        
        Returns:
            检索结果列表
        """
        results = []
        
        try:
            # 1. 如果有已有食材，通过关系查找
            if context.get("available_ingredients"):
                logger.info(f"    根据食材查找: {context['available_ingredients']}")
                recipes = self.relation_query.get_recipe_by_ingredients(
                    context["available_ingredients"],
                    exact_match=False  # 部分匹配，提高召回率
                )
                results.extend(recipes)
            
            # 2. 如果有健康目标，通过关系查找
            if context.get("health_goal"):
                logger.info(f"    根据健康目标查找: {context['health_goal']}")
                recipes = self.relation_query.get_recipes_for_health_goal(
                    context["health_goal"],
                    top_k=top_k
                )
                results.extend(recipes)
            
            # 3. 去重
            seen = set()
            unique_results = []
            for r in results:
                if r["id"] not in seen:
                    seen.add(r["id"])
                    unique_results.append(r)
            
            return unique_results[:top_k]
        
        except Exception as e:
            logger.error(f"关系推理检索失败: {e}", exc_info=True)
            return []
    
    def _reciprocal_rank_fusion(
        self,
        result_lists: List[List[Dict[str, Any]]],
        k: int = 60
    ) -> List[Dict[str, Any]]:
        scores = {}
        item_map = {}

        for result_list in result_lists:
            for rank, item in enumerate(result_list, 1):
                item_id = item.get("id") or item.get("name")
                if not item_id:
                    continue

                if item_id not in scores:
                    scores[item_id] = 0.0
                    item_map[item_id] = item

                scores[item_id] += 1 / (k + rank)

        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [item_map[item_id] for item_id, score in sorted_items]
    
    def _weighted_reciprocal_rank_fusion(
        self,
        vector_results: List[Dict[str, Any]],
        relation_results: List[Dict[str, Any]],
        weights: Dict[str, float],
        k: int,
        fusion_mode: str,
    ) -> List[Dict[str, Any]]:
        scores: Dict[str, float] = {}
        item_map: Dict[str, Dict[str, Any]] = {}
        rank_debug: Dict[str, Dict[str, int]] = {}

        weighted_lists = [
            ("vector", vector_results, float(weights.get("w_sem", 0.5))),
            ("relation", relation_results, float(weights.get("w_struct", 0.5))),
        ]

        for source_name, results, weight in weighted_lists:
            for rank, item in enumerate(results, 1):
                item_id = item.get("id") or item.get("name")
                if not item_id:
                    continue

                if item_id not in scores:
                    scores[item_id] = 0.0
                    item_map[item_id] = dict(item)
                    rank_debug[item_id] = {}

                scores[item_id] += weight / (k + rank)
                rank_debug[item_id][f"{source_name}_rank"] = rank

        sorted_items = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
        fused_items: List[Dict[str, Any]] = []
        for item_id, fusion_score in sorted_items:
            enriched_item = dict(item_map[item_id])
            enriched_item["fusion_score"] = round(fusion_score, 6)
            enriched_item["w_struct"] = round(float(weights.get("w_struct", 0.5)), 4)
            enriched_item["w_sem"] = round(float(weights.get("w_sem", 0.5)), 4)
            enriched_item["fusion_mode"] = fusion_mode
            enriched_item.update(rank_debug.get(item_id, {}))
            fused_items.append(enriched_item)

        return fused_items
    
    def _maximal_marginal_relevance(
        self,
        items: List[Dict[str, Any]],
        lambda_param: float = 0.7,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        MMR 多样性算法
        
        使用 Maximal Marginal Relevance 算法保证结果多样性
        MMR 公式: MMR = λ * Relevance - (1-λ) * max(Similarity)
        
        Args:
            items: 候选结果列表
            lambda_param: 平衡参数（0-1）
                - 接近1：更注重相关性
                - 接近0：更注重多样性
            top_k: 返回结果数量
        
        Returns:
            多样化的结果列表
        
        Example:
            >>> items = [{"id": "A", "tags": "快手,家常"}, 
            ...          {"id": "B", "tags": "快手,家常"},
            ...          {"id": "C", "tags": "养生,清淡"}]
            >>> diverse = self._maximal_marginal_relevance(items, lambda_param=0.5)
            >>> # 会选择 A 和 C，因为它们更不相似
        """
        if not items:
            return []
        
        # 如果结果数量不超过 top_k，直接返回
        if len(items) <= top_k:
            return items
        
        # 初始化：选择第一个（假设已按相关性排序）
        selected = [items[0]]
        remaining = items[1:]
        
        while remaining and len(selected) < top_k:
            mmr_scores = []
            
            for item in remaining:
                # 相关性分数（基于排名位置）
                # 排名越靠前，相关性越高
                relevance = 1.0 / (len(selected) + len(remaining))
                
                # 与已选择项的最大相似度
                max_sim = max([
                    self._calculate_similarity(item, s)
                    for s in selected
                ])
                
                # MMR 分数
                mmr = lambda_param * relevance - (1 - lambda_param) * max_sim
                mmr_scores.append((item, mmr))
            
            # 选择 MMR 最高的
            best_item = max(mmr_scores, key=lambda x: x[1])[0]
            selected.append(best_item)
            remaining.remove(best_item)
        
        return selected
    
    def _calculate_similarity(
        self,
        item1: Dict[str, Any],
        item2: Dict[str, Any]
    ) -> float:
        """
        计算两个食谱的相似度
        
        基于标签重叠度计算 Jaccard 相似度
        
        Args:
            item1: 第一个食谱
            item2: 第二个食谱
        
        Returns:
            相似度分数（0-1）
        
        Example:
            >>> item1 = {"tags": "快手,家常,下饭"}
            >>> item2 = {"tags": "快手,家常,酸甜"}
            >>> sim = self._calculate_similarity(item1, item2)
            >>> # sim = 2/4 = 0.5 (两个共同标签 / 四个不同标签)
        """
        # 提取标签
        tags1_str = item1.get("tags", "")
        tags2_str = item2.get("tags", "")
        
        if not tags1_str or not tags2_str:
            return 0.0
        
        tags1 = set(tags1_str.split(","))
        tags2 = set(tags2_str.split(","))
        
        # Jaccard 相似度
        intersection = len(tags1 & tags2)
        union = len(tags1 | tags2)
        
        return intersection / union if union > 0 else 0.0
