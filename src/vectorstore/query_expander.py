"""
查询扩展器 (V2)

利用知识图谱关系扩展查询，提升检索效果
根据食材、健康目标、场景等上下文信息智能扩展查询
"""

from typing import Dict, Any, List, Optional
from langchain_community.vectorstores import Chroma
from ..utils.logger import get_logger
from .relation_query import RelationQuery

logger = get_logger(__name__)


class QueryExpander:
    """查询扩展器：利用关系扩展查询
    
    通过分析用户上下文（食材、健康目标、场景等），
    利用知识图谱关系信息扩展原始查询，提升检索召回率和准确性
    """
    
    def __init__(self, vectorstore: Chroma):
        """
        初始化查询扩展器
        
        Args:
            vectorstore: ChromaDB 向量存储实例
        """
        self.vectorstore = vectorstore
        self.relation_query = RelationQuery(vectorstore)
    
    def expand_query(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        扩展查询，加入关系信息
        
        根据上下文信息，利用知识图谱关系扩展原始查询：
        1. 如果提到食材，加入搭配推荐
        2. 如果提到健康目标，加入相关标签
        3. 如果提到场景，加入场景特征
        
        Args:
            query: 原始查询文本
            context: 用户上下文，包含:
                - available_ingredients: 已有食材列表
                - health_goal: 健康目标
                - meal_type: 餐次类型
        
        Returns:
            扩展后的查询文本
        
        Example:
            >>> expander = QueryExpander(vectorstore)
            >>> context = {
            ...     "available_ingredients": ["鸡蛋"],
            ...     "health_goal": "减肥",
            ...     "meal_type": "早餐"
            ... }
            >>> expanded = expander.expand_query("快手食谱", context)
            >>> print(expanded)
            '快手食谱 搭配番茄 搭配韭菜 低热量 高蛋白 低脂 清淡 快手 营养 易消化'
        """
        if context is None:
            context = {}
        
        logger.info(f"开始查询扩展: query='{query}', context={context}")
        
        expanded_parts = [query]
        
        try:
            # 1. 如果提到食材，加入搭配推荐
            if context.get("available_ingredients"):
                ingredient_expansions = self._expand_by_ingredients(
                    context["available_ingredients"]
                )
                if ingredient_expansions:
                    expanded_parts.extend(ingredient_expansions)
                    logger.info(f"  添加食材搭配扩展: {ingredient_expansions}")
            
            # 2. 如果提到健康目标，加入相关标签
            if context.get("health_goal"):
                goal_tags = self._get_goal_related_tags(context["health_goal"])
                if goal_tags:
                    expanded_parts.extend(goal_tags)
                    logger.info(f"  添加健康目标标签: {goal_tags}")
            
            # 3. 如果提到场景，加入场景特征
            if context.get("meal_type"):
                scenario_features = self._get_scenario_features(context["meal_type"])
                if scenario_features:
                    expanded_parts.extend(scenario_features)
                    logger.info(f"  添加场景特征: {scenario_features}")
            
            expanded_query = " ".join(expanded_parts)
            logger.info(f"查询扩展完成: '{query}' -> '{expanded_query}'")
            
            return expanded_query
        
        except Exception as e:
            logger.error(f"查询扩展失败: {e}", exc_info=True)
            # 失败时返回原始查询
            return query
    
    def _expand_by_ingredients(
        self,
        ingredients: List[str],
        max_pairings_per_ingredient: int = 2
    ) -> List[str]:
        """
        根据食材扩展查询
        
        查询每个食材的搭配推荐，加入到查询中
        
        Args:
            ingredients: 食材列表
            max_pairings_per_ingredient: 每个食材最多添加的搭配数量
        
        Returns:
            扩展词列表
        
        Example:
            >>> expansions = self._expand_by_ingredients(["鸡蛋"])
            >>> print(expansions)
            ['搭配番茄', '搭配韭菜']
        """
        expansions = []
        
        try:
            for ingredient in ingredients:
                # 获取搭配推荐
                pairings = self.relation_query.get_ingredient_pairings(ingredient)
                
                if pairings:
                    # 只取前 N 个搭配
                    top_pairings = pairings[:max_pairings_per_ingredient]
                    
                    for pairing in top_pairings:
                        pair_name = pairing.get("name", "")
                        if pair_name:
                            expansions.append(f"搭配{pair_name}")
            
            return expansions
        
        except Exception as e:
            logger.error(f"根据食材扩展查询失败: {e}", exc_info=True)
            return []
    
    def _get_goal_related_tags(self, health_goal: str) -> List[str]:
        """
        获取健康目标相关的标签
        
        根据健康目标返回相关的食谱特征标签
        
        Args:
            health_goal: 健康目标（如：减肥、增肌、养生）
        
        Returns:
            相关标签列表
        
        Example:
            >>> tags = self._get_goal_related_tags("减肥")
            >>> print(tags)
            ['低热量', '高蛋白', '低脂', '清淡']
        """
        # 健康目标到标签的映射
        goal_tag_map = {
            "减肥": ["低热量", "高蛋白", "低脂", "清淡"],
            "增肌": ["高蛋白", "适量碳水", "营养丰富"],
            "养生": ["清淡", "营养均衡", "易消化"],
            "快速补充能量": ["高碳水", "快手", "易消化"],
            "健身": ["高蛋白", "低脂", "营养丰富"],
            "素食": ["素食", "蔬菜", "豆制品"],
            "低糖": ["低糖", "低碳水", "控糖"],
            "高纤维": ["高纤维", "蔬菜", "粗粮"],
        }
        
        # 返回对应的标签，如果没有匹配则返回空列表
        tags = goal_tag_map.get(health_goal, [])
        
        # 如果没有精确匹配，尝试模糊匹配
        if not tags:
            for goal_key, goal_tags in goal_tag_map.items():
                if goal_key in health_goal or health_goal in goal_key:
                    tags = goal_tags
                    break
        
        return tags
    
    def _get_scenario_features(self, meal_type: str) -> List[str]:
        """
        获取场景特征
        
        根据餐次类型返回相关的食谱特征
        
        Args:
            meal_type: 餐次类型（如：早餐、午餐、晚餐、加餐）
        
        Returns:
            场景特征列表
        
        Example:
            >>> features = self._get_scenario_features("早餐")
            >>> print(features)
            ['快手', '营养', '易消化']
        """
        # 场景到特征的映射
        scenario_feature_map = {
            "早餐": ["快手", "营养", "易消化"],
            "午餐": ["营养丰富", "饱腹感强"],
            "晚餐": ["清淡", "易消化", "不油腻"],
            "加餐": ["快手", "低热量", "方便"],
            "夜宵": ["清淡", "易消化", "低热量"],
            "健身后": ["高蛋白", "快速补充能量"],
            "加班后": ["快手", "营养", "下饭"],
            "工作日": ["快手", "简单"],
            "周末": ["营养丰富", "家常"],
        }
        
        # 返回对应的特征，如果没有匹配则返回空列表
        features = scenario_feature_map.get(meal_type, [])
        
        # 如果没有精确匹配，尝试模糊匹配
        if not features:
            for scenario_key, scenario_features in scenario_feature_map.items():
                if scenario_key in meal_type or meal_type in scenario_key:
                    features = scenario_features
                    break
        
        return features
    
    def expand_with_synonyms(
        self,
        query: str,
        max_synonyms: int = 3
    ) -> str:
        """
        使用同义词扩展查询（可选功能）
        
        查找查询中食材的别名，加入到查询中以提升召回率
        
        Args:
            query: 原始查询
            max_synonyms: 每个词最多添加的同义词数量
        
        Returns:
            扩展后的查询
        
        Example:
            >>> expanded = self.expand_with_synonyms("鸡蛋食谱")
            >>> print(expanded)
            '鸡蛋食谱 蛋 鸡子'
        """
        try:
            # 尝试在查询中找到食材
            # 这里简化实现，实际可以使用 NLP 分词
            expanded_parts = [query]
            
            # 获取所有食材
            all_ingredients = self.vectorstore.get(
                where={"entity_type": {"$eq": "ingredient"}}
            )
            
            if not all_ingredients or not all_ingredients.get("metadatas"):
                return query
            
            # 查找查询中提到的食材
            for ingredient_meta in all_ingredients["metadatas"]:
                ing_name = ingredient_meta.get("name", "")
                
                # 如果查询中包含这个食材名
                if ing_name and ing_name in query:
                    # 获取别名
                    aliases_str = ingredient_meta.get("aliases", "")
                    if aliases_str:
                        aliases = aliases_str.split(",")
                        # 添加前 N 个别名
                        expanded_parts.extend(aliases[:max_synonyms])
            
            return " ".join(expanded_parts)
        
        except Exception as e:
            logger.error(f"同义词扩展失败: {e}", exc_info=True)
            return query
