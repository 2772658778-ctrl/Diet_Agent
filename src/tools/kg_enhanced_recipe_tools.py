"""
V2 食谱推荐工具 - 知识图谱增强版

提供基于混合检索（向量 + 关系）的食谱推荐功能
支持已有食材、健康目标、时间限制等多种过滤条件
"""

import json
from typing import Optional, List, Dict, Any
from langchain.tools import tool
from ..vectorstore.chroma_client import get_vectorstore
from ..vectorstore.hybrid_retriever import HybridRetriever
from ..vectorstore.relation_query import RelationQuery
from ..config import get_settings
from ..utils.logger import get_logger


logger = get_logger(__name__)


@tool
def search_recipes_v2(
    query: str,
    time_limit: Optional[int] = None,
    health_goal: Optional[str] = None,
    available_ingredients: Optional[str] = None,
    difficulty: Optional[str] = None
) -> str:
    """
    搜索食谱（V2 知识图谱增强版）
    
    使用混合检索策略（向量检索 + 关系推理）提供更准确的食谱推荐。
    支持基于已有食材的智能推荐。
    
    Args:
        query: 搜索查询（如：酸甜口味、快手菜）
        time_limit: 时间限制（分钟），可选
        health_goal: 健康目标（减肥/增肌/养生），可选
        available_ingredients: 已有食材列表（逗号分隔），可选
        difficulty: 难度要求（简单/中等/困难），可选
    
    Returns:
        JSON 字符串，包含食谱列表及关系信息
        
    Examples:
        >>> search_recipes_v2("酸甜口味")
        >>> search_recipes_v2("快手菜", time_limit=30, available_ingredients="鸡蛋,番茄")
        >>> search_recipes_v2("高蛋白", health_goal="增肌")
    """
    try:
        logger.info(
            f"开始 V2 搜索: query='{query}', time_limit={time_limit}, "
            f"health_goal={health_goal}, available_ingredients={available_ingredients}"
        )
        
        # 输入验证
        if not query or not query.strip():
            logger.warning("搜索查询为空")
            return json.dumps({
                "error": "参数错误",
                "message": "请提供搜索关键词"
            }, ensure_ascii=False)
        
        # 验证时间限制
        if time_limit is not None and time_limit <= 0:
            logger.warning(f"无效的时间限制: {time_limit}")
            return json.dumps({
                "error": "参数错误",
                "message": "时间限制必须大于 0"
            }, ensure_ascii=False)
        
        # 获取向量存储实例
        vectorstore = get_vectorstore()
        if vectorstore is None:
            error_msg = "向量数据库未初始化"
            logger.error(error_msg)
            return json.dumps({
                "error": "系统错误",
                "message": "数据库未就绪，请联系管理员"
            }, ensure_ascii=False)
        
        # 创建混合检索器
        retriever = HybridRetriever(vectorstore)
        
        # 构建用户上下文
        context = {}
        
        if time_limit is not None:
            context["time_limit"] = time_limit
        
        if health_goal:
            context["health_goal"] = health_goal
        
        if difficulty:
            context["difficulty"] = difficulty
        
        if available_ingredients:
            # 解析食材列表
            ingredients_list = [
                ing.strip() 
                for ing in available_ingredients.split(",") 
                if ing.strip()
            ]
            context["available_ingredients"] = ingredients_list
            logger.info(f"已有食材: {ingredients_list}")
        
        # 获取配置
        settings = get_settings()
        
        # 执行混合检索
        results = retriever.retrieve(
            query=query,
            user_context=context,
            top_k=settings.top_k
        )
        
        if not results:
            logger.info("未找到匹配的食谱")
            return json.dumps({
                "message": "未找到符合条件的食谱",
                "suggestion": "建议调整搜索条件，如放宽时间限制或更改口味偏好",
                "recipes": []
            }, ensure_ascii=False)
        
        # 格式化结果（包含关系信息）
        recipes = []
        for result in results:
            try:
                # 解析关系数据
                relations_str = result.get("relations", "{}")
                try:
                    relations = json.loads(relations_str)
                except json.JSONDecodeError:
                    logger.warning(f"解析关系数据失败: {result.get('name')}")
                    relations = {}
                
                # 提取食材列表
                ingredients = [
                    ing["name"] 
                    for ing in relations.get("contains_ingredients", [])
                ]
                
                # 提取适合场景
                suitable_scenarios = relations.get("suitable_scenarios", [])
                
                # 提取相似食谱
                similar_recipes = [
                    sim["name"]
                    for sim in relations.get("similar_recipes", [])[:3]
                ]
                
                recipe = {
                    "name": result.get("name", ""),
                    "time": result.get("time", 0),
                    "difficulty": result.get("difficulty", ""),
                    "calories": result.get("calories", 0),
                    "tags": result.get("tags", "").split(",") if result.get("tags") else [],
                    "ingredients": ingredients,
                    "suitable_scenarios": suitable_scenarios,
                    "similar_recipes": similar_recipes,
                    "health_goals": result.get("health_goals", "").split(",") if result.get("health_goals") else []
                }
                recipes.append(recipe)
            except Exception as e:
                logger.warning(f"格式化食谱失败: {e}")
                continue
        
        if not recipes:
            logger.warning("所有食谱格式化失败")
            return json.dumps({
                "error": "数据错误",
                "message": "食谱数据格式异常"
            }, ensure_ascii=False)
        
        logger.info(f"成功返回 {len(recipes)} 条 V2 食谱推荐")
        
        return json.dumps({
            "success": True,
            "count": len(recipes),
            "recipes": recipes,
            "version": "v2"
        }, ensure_ascii=False)
        
    except Exception as e:
        logger.error(f"V2 搜索执行失败: {e}", exc_info=True)
        return json.dumps({
            "error": "系统错误",
            "message": "搜索失败，请稍后再试"
        }, ensure_ascii=False)


@tool
def check_ingredient_pairing(ingredients: str) -> str:
    """
    检查食材搭配（知识图谱推理）
    
    检查食材之间的搭配关系，包括：
    - 搭配禁忌（冲突检测）
    - 搭配推荐（互补建议）
    - 可做的食谱（关系推理）
    
    Args:
        ingredients: 逗号分隔的食材列表（如："鸡蛋,番茄,葱"）
    
    Returns:
        JSON 字符串，包含搭配建议、禁忌和可做食谱
        
    Examples:
        >>> check_ingredient_pairing("鸡蛋,番茄")
        >>> check_ingredient_pairing("鸡蛋,豆浆")  # 会检测到禁忌
    """
    try:
        logger.info(f"开始检查食材搭配: {ingredients}")
        
        # 输入验证
        if not ingredients or not ingredients.strip():
            logger.warning("食材列表为空")
            return json.dumps({
                "error": "参数错误",
                "message": "请提供食材列表"
            }, ensure_ascii=False)
        
        # 解析食材列表
        ingredient_list = [ing.strip() for ing in ingredients.split(",") if ing.strip()]
        
        if len(ingredient_list) < 2:
            logger.warning("食材数量不足")
            return json.dumps({
                "error": "参数错误",
                "message": "至少需要提供 2 种食材"
            }, ensure_ascii=False)
        
        # 获取向量存储实例
        vectorstore = get_vectorstore()
        if vectorstore is None:
            error_msg = "向量数据库未初始化"
            logger.error(error_msg)
            return json.dumps({
                "error": "系统错误",
                "message": "数据库未就绪"
            }, ensure_ascii=False)
        
        # 创建关系查询工具
        relation_query = RelationQuery(vectorstore)
        
        # 1. 检查禁忌
        logger.info("  检查食材禁忌...")
        conflicts = relation_query.check_ingredient_conflicts(ingredient_list)
        
        # 2. 获取搭配推荐
        logger.info("  获取搭配推荐...")
        all_pairings = []
        for ingredient in ingredient_list:
            pairs = relation_query.get_ingredient_pairings(ingredient)
            # 只保留不在当前列表中的推荐
            for pair in pairs:
                if pair["name"] not in ingredient_list:
                    all_pairings.append({
                        "base_ingredient": ingredient,
                        "recommended": pair["name"],
                        "reason": pair["reason"],
                        "score": pair["score"]
                    })
        
        # 去重并排序
        seen = set()
        unique_pairings = []
        for pair in all_pairings:
            key = f"{pair['base_ingredient']}-{pair['recommended']}"
            if key not in seen:
                seen.add(key)
                unique_pairings.append(pair)
        
        # 按评分排序，取前5个
        unique_pairings.sort(key=lambda x: x["score"], reverse=True)
        top_pairings = unique_pairings[:5]
        
        # 3. 查找可做的食谱
        logger.info("  查找可做的食谱...")
        recipes = relation_query.get_recipe_by_ingredients(
            ingredient_list,
            exact_match=False  # 部分匹配，提高召回率
        )
        
        # 格式化食谱列表
        recipe_names = [r["name"] for r in recipes[:5]]
        
        # 构建结果
        result = {
            "success": True,
            "ingredients": ingredient_list,
            "conflicts": conflicts,
            "pairing_suggestions": top_pairings,
            "possible_recipes": recipe_names,
            "summary": {
                "has_conflicts": len(conflicts) > 0,
                "conflict_count": len(conflicts),
                "suggestion_count": len(top_pairings),
                "recipe_count": len(recipe_names)
            }
        }
        
        logger.info(
            f"搭配检查完成: {len(conflicts)} 个冲突, "
            f"{len(top_pairings)} 个建议, {len(recipe_names)} 个食谱"
        )
        
        return json.dumps(result, ensure_ascii=False)
        
    except Exception as e:
        logger.error(f"检查食材搭配失败: {e}", exc_info=True)
        return json.dumps({
            "error": "系统错误",
            "message": "搭配检查失败，请稍后再试"
        }, ensure_ascii=False)


@tool
def get_nutrition_advice(
    health_goal: str,
    current_diet: Optional[str] = None
) -> str:
    """
    获取营养建议（知识图谱推理）
    
    基于健康目标提供个性化营养建议，包括：
    - 营养摄入建议
    - 推荐食谱
    - 饮食注意事项
    
    Args:
        health_goal: 健康目标（减肥/增肌/养生）
        current_diet: 当前饮食描述（可选）
    
    Returns:
        JSON 字符串，包含营养建议和推荐食谱
        
    Examples:
        >>> get_nutrition_advice("减肥")
        >>> get_nutrition_advice("增肌", current_diet="每天吃3顿，主要是米饭和蔬菜")
    """
    try:
        logger.info(f"开始生成营养建议: health_goal={health_goal}")
        
        # 输入验证
        if not health_goal or not health_goal.strip():
            logger.warning("健康目标为空")
            return json.dumps({
                "error": "参数错误",
                "message": "请提供健康目标"
            }, ensure_ascii=False)
        
        # 验证健康目标
        valid_goals = ["减肥", "增肌", "养生"]
        if health_goal not in valid_goals:
            logger.warning(f"无效的健康目标: {health_goal}")
            return json.dumps({
                "error": "参数错误",
                "message": f"健康目标必须是以下之一: {', '.join(valid_goals)}"
            }, ensure_ascii=False)
        
        # 获取向量存储实例
        vectorstore = get_vectorstore()
        if vectorstore is None:
            error_msg = "向量数据库未初始化"
            logger.error(error_msg)
            return json.dumps({
                "error": "系统错误",
                "message": "数据库未就绪"
            }, ensure_ascii=False)
        
        # 创建关系查询工具
        relation_query = RelationQuery(vectorstore)
        
        # 1. 获取适合该目标的食谱
        logger.info(f"  查找适合 '{health_goal}' 的食谱...")
        recipes = relation_query.get_recipes_for_health_goal(health_goal, top_k=10)
        
        if not recipes:
            logger.warning(f"未找到适合 '{health_goal}' 的食谱")
            return json.dumps({
                "success": True,
                "health_goal": health_goal,
                "message": "暂无适合该健康目标的食谱",
                "advice": "建议咨询专业营养师",
                "recommended_recipes": []
            }, ensure_ascii=False)
        
        # 2. 分析营养特点
        logger.info("  分析营养特点...")
        total_calories = sum(r.get("calories", 0) for r in recipes)
        total_protein = sum(r.get("protein", 0) for r in recipes)
        avg_calories = total_calories / len(recipes) if recipes else 0
        avg_protein = total_protein / len(recipes) if recipes else 0
        
        # 提取常见标签
        all_tags = []
        for recipe in recipes:
            tags_str = recipe.get("tags", "")
            if tags_str:
                tags = [t.strip() for t in tags_str.split(",") if t.strip()]
                all_tags.extend(tags)
        
        # 统计标签频率
        from collections import Counter
        tag_counts = Counter(all_tags)
        common_tags = [tag for tag, count in tag_counts.most_common(5)]
        
        nutrition_summary = {
            "avg_calories": round(avg_calories, 1),
            "avg_protein": round(avg_protein, 1),
            "common_tags": common_tags,
            "recipe_count": len(recipes)
        }
        
        # 3. 生成建议
        logger.info("  生成营养建议...")
        advice = _generate_nutrition_advice(health_goal, nutrition_summary)
        
        # 4. 格式化推荐食谱
        recommended_recipes = []
        for recipe in recipes[:5]:
            recommended_recipes.append({
                "name": recipe.get("name", ""),
                "time": recipe.get("time", 0),
                "calories": recipe.get("calories", 0),
                "protein": recipe.get("protein", 0),
                "tags": recipe.get("tags", "").split(",") if recipe.get("tags") else []
            })
        
        # 构建结果
        result = {
            "success": True,
            "health_goal": health_goal,
            "nutrition_summary": nutrition_summary,
            "advice": advice,
            "recommended_recipes": recommended_recipes,
            "tips": _get_health_goal_tips(health_goal)
        }
        
        logger.info(f"营养建议生成完成: {len(recommended_recipes)} 个推荐食谱")
        
        return json.dumps(result, ensure_ascii=False)
        
    except Exception as e:
        logger.error(f"生成营养建议失败: {e}", exc_info=True)
        return json.dumps({
            "error": "系统错误",
            "message": "建议生成失败，请稍后再试"
        }, ensure_ascii=False)


def _generate_nutrition_advice(
    health_goal: str,
    nutrition_summary: Dict[str, Any]
) -> str:
    """
    生成营养建议文本
    
    Args:
        health_goal: 健康目标
        nutrition_summary: 营养摘要
    
    Returns:
        营养建议文本
    """
    advice_templates = {
        "减肥": (
            f"建议每餐控制在 {nutrition_summary['avg_calories']:.0f} 卡路里左右，"
            f"保证 {nutrition_summary['avg_protein']:.1f}克 蛋白质摄入。"
            f"多选择{', '.join(nutrition_summary['common_tags'][:3])}类食物。"
            f"配合适量运动，每周至少3次有氧运动。"
        ),
        
        "增肌": (
            f"建议每餐摄入 {nutrition_summary['avg_protein']:.1f}克 以上蛋白质，"
            f"总热量约 {nutrition_summary['avg_calories']:.0f} 卡路里。"
            f"推荐{', '.join(nutrition_summary['common_tags'][:3])}类食物。"
            f"配合力量训练，训练后30分钟内补充蛋白质。"
        ),
        
        "养生": (
            f"建议饮食清淡，每餐 {nutrition_summary['avg_calories']:.0f} 卡路里左右，"
            f"营养均衡。多选择{', '.join(nutrition_summary['common_tags'][:3])}类食物。"
            f"保持规律作息，适量运动，保持心情愉悦。"
        )
    }
    
    return advice_templates.get(health_goal, "建议均衡饮食，适量运动。")


def _get_health_goal_tips(health_goal: str) -> List[str]:
    """
    获取健康目标相关的小贴士
    
    Args:
        health_goal: 健康目标
    
    Returns:
        小贴士列表
    """
    tips_map = {
        "减肥": [
            "控制总热量摄入，创造热量缺口",
            "增加蛋白质摄入，提高饱腹感",
            "多吃蔬菜，增加膳食纤维",
            "避免高糖高脂食物",
            "保持规律运动，每周至少3次"
        ],
        
        "增肌": [
            "保证充足蛋白质摄入（每公斤体重1.6-2.2克）",
            "适量增加碳水化合物，提供能量",
            "训练后及时补充营养",
            "保证充足睡眠，促进肌肉恢复",
            "循序渐进增加训练强度"
        ],
        
        "养生": [
            "饮食清淡，少油少盐",
            "多吃新鲜蔬菜水果",
            "保持规律作息",
            "适量运动，不过度劳累",
            "保持心情愉悦，减少压力"
        ]
    }
    
    return tips_map.get(health_goal, ["建议咨询专业营养师"])
