"""
关系查询工具 (V2)

提供基于知识图谱关系的查询功能
支持食材搭配、食谱查找、禁忌检查等关系推理
"""

import json
from typing import List, Dict, Any, Optional
from langchain_community.vectorstores import Chroma
from ..utils.logger import get_logger

logger = get_logger(__name__)


class RelationQuery:
    """关系查询工具类
    
    提供基于元数据中的关系信息进行查询和推理的功能
    """
    
    def __init__(self, vectorstore: Chroma):
        """
        初始化关系查询工具
        
        Args:
            vectorstore: ChromaDB 向量存储实例
        """
        self.vectorstore = vectorstore
    
    def get_ingredient_pairings(self, ingredient_name: str) -> List[Dict[str, Any]]:
        """
        获取食材搭配推荐
        
        查询指定食材的搭配推荐，返回搭配原因和评分
        
        Args:
            ingredient_name: 食材名称
        
        Returns:
            搭配推荐列表，每项包含:
            - id: 食材ID
            - name: 食材名称
            - reason: 搭配原因
            - score: 搭配评分 (0-1)
        
        Example:
            >>> pairings = query.get_ingredient_pairings("鸡蛋")
            >>> print(pairings[0])
            {'id': 'ing_002', 'name': '番茄', 'reason': '酸甜口味互补', 'score': 0.95}
        """
        try:
            # 1. 检索食材实体
            results = self.vectorstore.similarity_search(
                query=ingredient_name,
                k=1,
                filter={"entity_type": {"$eq": "ingredient"}}
            )
            
            if not results:
                logger.warning(f"未找到食材: {ingredient_name}")
                return []
            
            # 2. 提取搭配关系
            ingredient = results[0].metadata
            relations_str = ingredient.get("relations", "{}")
            
            # 解析 JSON 关系数据
            try:
                relations = json.loads(relations_str)
            except json.JSONDecodeError as e:
                logger.error(f"解析关系数据失败: {e}")
                return []
            
            pairings = relations.get("pairs_well_with", [])
            
            logger.info(f"找到 {len(pairings)} 个搭配推荐: {ingredient_name}")
            return pairings
        
        except Exception as e:
            logger.error(f"获取食材搭配失败: {e}", exc_info=True)
            return []
    
    def get_recipe_by_ingredients(
        self,
        ingredients: List[str],
        exact_match: bool = True
    ) -> List[Dict[str, Any]]:
        """
        根据食材列表查找食谱
        
        支持精确匹配（所有食材都包含）或部分匹配
        
        Args:
            ingredients: 食材名称列表
            exact_match: 是否精确匹配（True=所有食材都包含，False=包含任一食材）
        
        Returns:
            匹配的食谱列表，每项包含完整的食谱元数据
        
        Example:
            >>> recipes = query.get_recipe_by_ingredients(["鸡蛋", "番茄"])
            >>> print(recipes[0]['name'])
            '番茄炒蛋'
        """
        try:
            # 1. 检索所有食谱
            all_recipes = self.vectorstore.get(
                where={"entity_type": {"$eq": "recipe"}}
            )
            
            if not all_recipes or not all_recipes.get("metadatas"):
                logger.warning("未找到任何食谱")
                return []
            
            # 2. 过滤包含指定食材的食谱
            matched_recipes = []
            
            for recipe in all_recipes["metadatas"]:
                relations_str = recipe.get("relations", "{}")
                
                try:
                    relations = json.loads(relations_str)
                except json.JSONDecodeError:
                    continue
                
                recipe_ingredients = relations.get("contains_ingredients", [])
                recipe_ing_names = [ing["name"] for ing in recipe_ingredients]
                
                # 检查匹配条件
                if exact_match:
                    # 精确匹配：所有指定食材都包含
                    if all(ing in recipe_ing_names for ing in ingredients):
                        matched_recipes.append(recipe)
                else:
                    # 部分匹配：包含任一食材
                    if any(ing in recipe_ing_names for ing in ingredients):
                        matched_recipes.append(recipe)
            
            logger.info(
                f"找到 {len(matched_recipes)} 个食谱 "
                f"({'精确' if exact_match else '部分'}匹配): {ingredients}"
            )
            return matched_recipes
        
        except Exception as e:
            logger.error(f"根据食材查找食谱失败: {e}", exc_info=True)
            return []
    
    def check_ingredient_conflicts(
        self,
        ingredients: List[str]
    ) -> List[Dict[str, Any]]:
        """
        检查食材禁忌
        
        检查食材列表中是否存在搭配禁忌，返回冲突原因和严重程度
        
        Args:
            ingredients: 食材名称列表
        
        Returns:
            冲突列表，每项包含:
            - ingredient1: 第一个食材名称
            - ingredient2: 第二个食材名称
            - reason: 冲突原因
            - severity: 严重程度（低/中/高）
        
        Example:
            >>> conflicts = query.check_ingredient_conflicts(["鸡蛋", "豆浆"])
            >>> print(conflicts[0])
            {'ingredient1': '鸡蛋', 'ingredient2': '豆浆', 
             'reason': '影响蛋白质吸收', 'severity': '中'}
        """
        conflicts = []
        
        try:
            # 检索所有食材
            for ing_name in ingredients:
                results = self.vectorstore.similarity_search(
                    query=ing_name,
                    k=1,
                    filter={"entity_type": {"$eq": "ingredient"}}
                )
                
                if not results:
                    continue
                
                ing_data = results[0].metadata
                relations_str = ing_data.get("relations", "{}")
                
                try:
                    relations = json.loads(relations_str)
                except json.JSONDecodeError:
                    continue
                
                ing_conflicts = relations.get("conflicts_with", [])
                
                # 检查是否与其他食材冲突
                for conflict in ing_conflicts:
                    if conflict["name"] in ingredients:
                        conflicts.append({
                            "ingredient1": ing_name,
                            "ingredient2": conflict["name"],
                            "reason": conflict["reason"],
                            "severity": conflict["severity"]
                        })
            
            if conflicts:
                logger.warning(f"发现 {len(conflicts)} 个食材冲突")
            else:
                logger.info("未发现食材冲突")
            
            return conflicts
        
        except Exception as e:
            logger.error(f"检查食材禁忌失败: {e}", exc_info=True)
            return []
    
    def get_recipes_for_health_goal(
        self,
        health_goal: str,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        根据健康目标查找食谱
        
        使用元数据过滤查找适合指定健康目标的食谱
        
        Args:
            health_goal: 健康目标（如：减肥、增肌、养生）
            top_k: 返回结果数量上限
        
        Returns:
            匹配的食谱列表
        
        Example:
            >>> recipes = query.get_recipes_for_health_goal("减肥")
            >>> print(len(recipes))
            5
        """
        try:
            # 方式 1：通过元数据过滤（快速）
            # 注意：ChromaDB 的 where 子句需要精确匹配
            # 由于 health_goals 是逗号分隔的字符串，我们需要使用 contains
            
            # 获取所有食谱
            all_recipes = self.vectorstore.get(
                where={"entity_type": {"$eq": "recipe"}}
            )
            
            if not all_recipes or not all_recipes.get("metadatas"):
                logger.warning("未找到任何食谱")
                return []
            
            # 手动过滤包含指定健康目标的食谱
            matched_recipes = []
            for recipe in all_recipes["metadatas"]:
                health_goals_str = recipe.get("health_goals", "")
                if health_goal in health_goals_str:
                    matched_recipes.append(recipe)
                    if len(matched_recipes) >= top_k:
                        break
            
            logger.info(f"找到 {len(matched_recipes)} 个食谱适合健康目标: {health_goal}")
            return matched_recipes
        
        except Exception as e:
            logger.error(f"根据健康目标查找食谱失败: {e}", exc_info=True)
            return []
    
    def find_similar_recipes(
        self,
        recipe_id: str,
        top_k: int = 5,
        use_vector: bool = True
    ) -> List[Dict[str, Any]]:
        """
        查找相似食谱
        
        融合预定义关系和向量相似度查找相似食谱
        
        Args:
            recipe_id: 食谱ID
            top_k: 返回结果数量
            use_vector: 是否使用向量相似度（True=融合关系和向量，False=仅使用关系）
        
        Returns:
            相似食谱列表，按相似度排序
        
        Example:
            >>> similar = query.find_similar_recipes("recipe_001", top_k=3)
            >>> print(similar[0]['name'])
            '西红柿蛋汤'
        """
        try:
            # 1. 获取食谱
            recipe = self.vectorstore.get(ids=[recipe_id])
            
            if not recipe or not recipe.get("metadatas"):
                logger.warning(f"未找到食谱: {recipe_id}")
                return []
            
            recipe_data = recipe["metadatas"][0]
            relations_str = recipe_data.get("relations", "{}")
            
            try:
                relations = json.loads(relations_str)
            except json.JSONDecodeError:
                relations = {}
            
            # 2. 方式 A：使用预定义的相似关系
            similar_from_relation = relations.get("similar_recipes", [])
            similar_ids_from_relation = [r["id"] for r in similar_from_relation]
            
            result_recipes = []
            
            # 获取关系中的相似食谱详情
            if similar_ids_from_relation:
                for sim_id in similar_ids_from_relation:
                    sim_recipe = self.vectorstore.get(ids=[sim_id])
                    if sim_recipe and sim_recipe.get("metadatas"):
                        result_recipes.append(sim_recipe["metadatas"][0])
            
            # 3. 方式 B：使用向量相似度（如果启用）
            if use_vector and recipe.get("documents"):
                recipe_text = recipe["documents"][0]
                similar_from_vector = self.vectorstore.similarity_search(
                    query=recipe_text,
                    k=top_k + 1,  # +1 因为会包含自己
                    filter={"entity_type": {"$eq": "recipe"}}
                )
                
                # 添加向量检索的结果（排除自己）
                for doc in similar_from_vector:
                    if doc.metadata.get("id") != recipe_id:
                        # 避免重复
                        if doc.metadata.get("id") not in similar_ids_from_relation:
                            result_recipes.append(doc.metadata)
            
            # 4. 限制返回数量
            result_recipes = result_recipes[:top_k]
            
            logger.info(f"找到 {len(result_recipes)} 个相似食谱: {recipe_id}")
            return result_recipes
        
        except Exception as e:
            logger.error(f"查找相似食谱失败: {e}", exc_info=True)
            return []
    
    def _reciprocal_rank_fusion(
        self,
        result_lists: List[List[Dict[str, Any]]],
        k: int = 60
    ) -> List[Dict[str, Any]]:
        """
        RRF 融合算法
        
        使用 Reciprocal Rank Fusion 算法融合多个结果列表
        
        Args:
            result_lists: 多个结果列表
            k: RRF 参数（默认60）
        
        Returns:
            融合后的结果列表，按分数排序
        """
        scores = {}
        item_map = {}
        
        for results in result_lists:
            for rank, item in enumerate(results, 1):
                item_id = item.get("id", item.get("name"))
                
                # 累加 RRF 分数
                if item_id not in scores:
                    scores[item_id] = 0
                    item_map[item_id] = item
                
                scores[item_id] += 1 / (k + rank)
        
        # 按分数排序
        sorted_items = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [item_map[item_id] for item_id, score in sorted_items]
