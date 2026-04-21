"""
带缓存的关系查询工具 (V2 性能优化)

使用 LRU 缓存优化热门查询，减少数据库调用次数
"""

from functools import lru_cache
import json
from typing import List, Dict, Any
from langchain_community.vectorstores import Chroma
from .relation_query import RelationQuery
from ..utils.logger import get_logger

logger = get_logger(__name__)


class CachedRelationQuery(RelationQuery):
    """带缓存的关系查询工具类
    
    继承 RelationQuery，为热门查询添加 LRU 缓存
    缓存策略：
    - 食材搭配查询：缓存 200 个最常用食材
    - 食材禁忌查询：缓存 100 个最常用食材组合
    """
    
    def __init__(self, vectorstore: Chroma):
        """
        初始化带缓存的关系查询工具
        
        Args:
            vectorstore: ChromaDB 向量存储实例
        """
        super().__init__(vectorstore)
        logger.info("初始化带缓存的关系查询工具")
    
    @lru_cache(maxsize=200)
    def get_ingredient_pairings_cached(self, ingredient_name: str) -> str:
        """
        缓存版本的食材搭配查询
        
        使用 LRU 缓存存储最近 200 个查询结果
        返回 JSON 字符串以便缓存（不可变类型）
        
        Args:
            ingredient_name: 食材名称
        
        Returns:
            JSON 字符串，包含搭配推荐列表
        """
        logger.debug(f"缓存查询: 食材搭配 - {ingredient_name}")
        pairings = super().get_ingredient_pairings(ingredient_name)
        return json.dumps(pairings, ensure_ascii=False)
    
    @lru_cache(maxsize=100)
    def check_ingredient_conflicts_cached(self, ingredients_tuple: tuple) -> str:
        """
        缓存版本的食材禁忌查询
        
        使用 LRU 缓存存储最近 100 个查询结果
        使用 tuple 作为参数以便缓存（不可变类型）
        
        Args:
            ingredients_tuple: 食材名称元组（已排序）
        
        Returns:
            JSON 字符串，包含冲突列表
        """
        logger.debug(f"缓存查询: 食材禁忌 - {ingredients_tuple}")
        conflicts = super().check_ingredient_conflicts(list(ingredients_tuple))
        return json.dumps(conflicts, ensure_ascii=False)
    
    def get_ingredient_pairings(self, ingredient_name: str) -> List[Dict[str, Any]]:
        """
        获取食材搭配推荐（使用缓存）
        
        覆盖父类方法，使用缓存版本
        
        Args:
            ingredient_name: 食材名称
        
        Returns:
            搭配推荐列表
        """
        cached_result = self.get_ingredient_pairings_cached(ingredient_name)
        return json.loads(cached_result)
    
    def check_ingredient_conflicts(
        self,
        ingredients: List[str]
    ) -> List[Dict[str, Any]]:
        """
        检查食材禁忌（使用缓存）
        
        覆盖父类方法，使用缓存版本
        将列表转换为排序后的元组以便缓存
        
        Args:
            ingredients: 食材名称列表
        
        Returns:
            冲突列表
        """
        # 转成排序后的 tuple 以便缓存
        # 排序确保 ["鸡蛋", "豆浆"] 和 ["豆浆", "鸡蛋"] 使用同一缓存
        ingredients_tuple = tuple(sorted(ingredients))
        cached_result = self.check_ingredient_conflicts_cached(ingredients_tuple)
        return json.loads(cached_result)
    
    def clear_cache(self):
        """
        清除所有缓存
        
        在数据更新后调用，确保缓存一致性
        """
        logger.info("清除关系查询缓存")
        self.get_ingredient_pairings_cached.cache_clear()
        self.check_ingredient_conflicts_cached.cache_clear()
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        获取缓存统计信息
        
        Returns:
            包含缓存命中率等统计信息的字典
        """
        pairings_info = self.get_ingredient_pairings_cached.cache_info()
        conflicts_info = self.check_ingredient_conflicts_cached.cache_info()
        
        return {
            "ingredient_pairings": {
                "hits": pairings_info.hits,
                "misses": pairings_info.misses,
                "maxsize": pairings_info.maxsize,
                "currsize": pairings_info.currsize,
                "hit_rate": (
                    pairings_info.hits / (pairings_info.hits + pairings_info.misses)
                    if (pairings_info.hits + pairings_info.misses) > 0
                    else 0.0
                )
            },
            "ingredient_conflicts": {
                "hits": conflicts_info.hits,
                "misses": conflicts_info.misses,
                "maxsize": conflicts_info.maxsize,
                "currsize": conflicts_info.currsize,
                "hit_rate": (
                    conflicts_info.hits / (conflicts_info.hits + conflicts_info.misses)
                    if (conflicts_info.hits + conflicts_info.misses) > 0
                    else 0.0
                )
            }
        }
    
    def batch_get_ingredient_pairings(
        self,
        ingredient_names: List[str]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        批量获取食材搭配推荐
        
        一次性检索多个食材，减少数据库调用次数
        优化策略：
        1. 先检查缓存，获取已缓存的结果
        2. 对未缓存的食材进行批量查询
        3. 更新缓存
        
        Args:
            ingredient_names: 食材名称列表
        
        Returns:
            字典，键为食材名称，值为搭配推荐列表
        
        Example:
            >>> pairings = query.batch_get_ingredient_pairings(["鸡蛋", "番茄", "土豆"])
            >>> print(pairings["鸡蛋"])
            [{'id': 'ing_002', 'name': '番茄', 'reason': '酸甜口味互补', 'score': 0.95}]
        """
        logger.info(f"批量查询食材搭配: {len(ingredient_names)} 个食材")
        
        pairings_map = {}
        uncached_ingredients = []
        
        # 1. 先检查缓存
        for ing_name in ingredient_names:
            try:
                # 尝试从缓存获取
                cached_result = self.get_ingredient_pairings_cached(ing_name)
                pairings_map[ing_name] = json.loads(cached_result)
            except Exception:
                # 缓存未命中，加入待查询列表
                uncached_ingredients.append(ing_name)
        
        # 2. 批量查询未缓存的食材
        if uncached_ingredients:
            logger.debug(f"批量查询未缓存的 {len(uncached_ingredients)} 个食材")
            
            try:
                # 一次性检索所有未缓存的食材
                results = self.vectorstore.get(
                    where={
                        "entity_type": {"$eq": "ingredient"}
                    }
                )
                
                if results and results.get("metadatas"):
                    # 构建名称到元数据的映射
                    for metadata in results["metadatas"]:
                        name = metadata.get("name")
                        if name in uncached_ingredients:
                            relations_str = metadata.get("relations", "{}")
                            try:
                                relations = json.loads(relations_str)
                                pairings = relations.get("pairs_well_with", [])
                                pairings_map[name] = pairings
                                
                                # 更新缓存
                                self.get_ingredient_pairings_cached(name)
                            except json.JSONDecodeError as e:
                                logger.error(f"解析关系数据失败: {name} - {e}")
                                pairings_map[name] = []
            
            except Exception as e:
                logger.error(f"批量查询食材失败: {e}", exc_info=True)
                # 为未查询到的食材设置空列表
                for ing_name in uncached_ingredients:
                    if ing_name not in pairings_map:
                        pairings_map[ing_name] = []
        
        logger.info(f"批量查询完成: 返回 {len(pairings_map)} 个结果")
        return pairings_map
    
    def batch_check_ingredient_conflicts(
        self,
        ingredient_groups: List[List[str]]
    ) -> List[List[Dict[str, Any]]]:
        """
        批量检查多组食材的禁忌
        
        一次性检查多组食材组合，减少数据库调用
        
        Args:
            ingredient_groups: 食材组列表，每组是一个食材名称列表
        
        Returns:
            冲突列表的列表，每个元素对应一组食材的冲突
        
        Example:
            >>> groups = [["鸡蛋", "豆浆"], ["番茄", "黄瓜"]]
            >>> conflicts = query.batch_check_ingredient_conflicts(groups)
            >>> print(len(conflicts))
            2
        """
        logger.info(f"批量检查食材禁忌: {len(ingredient_groups)} 组")
        
        results = []
        
        # 收集所有需要查询的食材（去重）
        all_ingredients = set()
        for group in ingredient_groups:
            all_ingredients.update(group)
        
        # 批量获取所有食材的元数据
        ingredient_data_map = {}
        try:
            all_results = self.vectorstore.get(
                where={"entity_type": {"$eq": "ingredient"}}
            )
            
            if all_results and all_results.get("metadatas"):
                for metadata in all_results["metadatas"]:
                    name = metadata.get("name")
                    if name in all_ingredients:
                        ingredient_data_map[name] = metadata
        
        except Exception as e:
            logger.error(f"批量获取食材数据失败: {e}", exc_info=True)
        
        # 对每组食材检查冲突
        for group in ingredient_groups:
            conflicts = []
            
            for ing_name in group:
                if ing_name not in ingredient_data_map:
                    continue
                
                ing_data = ingredient_data_map[ing_name]
                relations_str = ing_data.get("relations", "{}")
                
                try:
                    relations = json.loads(relations_str)
                    ing_conflicts = relations.get("conflicts_with", [])
                    
                    # 检查是否与组内其他食材冲突
                    for conflict in ing_conflicts:
                        if conflict["name"] in group:
                            conflicts.append({
                                "ingredient1": ing_name,
                                "ingredient2": conflict["name"],
                                "reason": conflict["reason"],
                                "severity": conflict["severity"]
                            })
                
                except json.JSONDecodeError as e:
                    logger.error(f"解析关系数据失败: {ing_name} - {e}")
            
            results.append(conflicts)
        
        logger.info(f"批量检查完成: 返回 {len(results)} 组结果")
        return results



def optimize_metadata_structure(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    优化元数据结构以提升查询性能
    
    将常用关系提取到顶层字段，避免频繁的 JSON 解析
    优化策略：
    1. 提取常用关系字段到顶层（字符串格式）
    2. 保留原始 relations JSON（用于完整数据访问）
    3. 为常用查询创建索引友好的字段
    
    Args:
        metadata: 原始元数据字典
    
    Returns:
        优化后的元数据字典
    
    Example:
        >>> original = {
        ...     "entity_type": "recipe",
        ...     "relations": '{"contains_ingredients": [{"id": "ing_001", "name": "鸡蛋"}]}'
        ... }
        >>> optimized = optimize_metadata_structure(original)
        >>> print(optimized.get("ingredient_ids"))
        'ing_001'
    """
    logger.debug(f"优化元数据结构: {metadata.get('entity_type')} - {metadata.get('name')}")
    
    # 复制原始元数据
    optimized = metadata.copy()
    
    # 解析关系数据
    relations_str = metadata.get("relations", "{}")
    try:
        relations = json.loads(relations_str)
    except json.JSONDecodeError as e:
        logger.error(f"解析关系数据失败: {e}")
        return optimized
    
    entity_type = metadata.get("entity_type")
    
    # 根据实体类型优化
    if entity_type == "recipe":
        # 1. 提取食材 ID 列表（用于快速过滤）
        ingredient_ids = [
            ing["id"]
            for ing in relations.get("contains_ingredients", [])
        ]
        optimized["ingredient_ids"] = ",".join(ingredient_ids)
        
        # 2. 提取食材名称列表（用于文本搜索）
        ingredient_names = [
            ing["name"]
            for ing in relations.get("contains_ingredients", [])
        ]
        optimized["ingredient_names"] = ",".join(ingredient_names)
        
        # 3. 提取场景列表（用于场景过滤）
        scenarios = relations.get("suitable_scenarios", [])
        optimized["scenarios"] = ",".join(scenarios)
        
        # 4. 提取相似食谱 ID（用于快速查找）
        similar_ids = [
            r["id"]
            for r in relations.get("similar_recipes", [])
        ]
        optimized["similar_recipe_ids"] = ",".join(similar_ids)
    
    elif entity_type == "ingredient":
        # 1. 提取搭配食材 ID（用于快速查找）
        pairing_ids = [
            p["id"]
            for p in relations.get("pairs_well_with", [])
        ]
        optimized["pairing_ids"] = ",".join(pairing_ids)
        
        # 2. 提取搭配食材名称
        pairing_names = [
            p["name"]
            for p in relations.get("pairs_well_with", [])
        ]
        optimized["pairing_names"] = ",".join(pairing_names)
        
        # 3. 提取禁忌食材 ID
        conflict_ids = [
            c["id"]
            for c in relations.get("conflicts_with", [])
        ]
        optimized["conflict_ids"] = ",".join(conflict_ids)
        
        # 4. 提取禁忌食材名称
        conflict_names = [
            c["name"]
            for c in relations.get("conflicts_with", [])
        ]
        optimized["conflict_names"] = ",".join(conflict_names)
        
        # 5. 提取功效列表（用于功效搜索）
        effects = [
            e["effect"]
            for e in relations.get("health_effects", [])
        ]
        optimized["health_effects_list"] = ",".join(effects)
    
    elif entity_type == "nutrient":
        # 1. 提取功能列表
        functions = relations.get("functions", [])
        optimized["functions_list"] = ",".join(functions)
        
        # 2. 提取富含来源 ID
        source_ids = [
            s["id"]
            for s in relations.get("rich_sources", [])
        ]
        optimized["rich_source_ids"] = ",".join(source_ids)
        
        # 3. 提取富含来源名称
        source_names = [
            s["name"]
            for s in relations.get("rich_sources", [])
        ]
        optimized["rich_source_names"] = ",".join(source_names)
    
    logger.debug(f"元数据优化完成: 添加了 {len(optimized) - len(metadata)} 个字段")
    return optimized


def batch_optimize_metadata(
    metadatas: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    批量优化元数据结构
    
    对多个元数据进行批量优化，提升处理效率
    
    Args:
        metadatas: 元数据列表
    
    Returns:
        优化后的元数据列表
    
    Example:
        >>> metadatas = [recipe1_meta, recipe2_meta, ingredient1_meta]
        >>> optimized = batch_optimize_metadata(metadatas)
        >>> print(len(optimized))
        3
    """
    logger.info(f"批量优化元数据: {len(metadatas)} 条")
    
    optimized_list = []
    for metadata in metadatas:
        try:
            optimized = optimize_metadata_structure(metadata)
            optimized_list.append(optimized)
        except Exception as e:
            logger.error(f"优化元数据失败: {e}", exc_info=True)
            # 失败时保留原始元数据
            optimized_list.append(metadata)
    
    logger.info(f"批量优化完成: {len(optimized_list)} 条")
    return optimized_list
