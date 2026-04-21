"""
Agent 工具包

提供食谱推荐、营养分析和食材查询功能
包含 V1 和 V2 版本的工具
"""

from .recipe_tools import search_recipes, clear_search_cache, get_cache_info
from .nutrition_tools import analyze_nutrition
from .ingredient_tools import check_ingredients

# V2 知识图谱增强工具
from .kg_enhanced_recipe_tools import (
    search_recipes_v2,
    check_ingredient_pairing,
    get_nutrition_advice
)


__all__ = [
    # V1 工具
    "search_recipes",
    "clear_search_cache",
    "get_cache_info",
    "analyze_nutrition",
    "check_ingredients",
    
    # V2 工具
    "search_recipes_v2",
    "check_ingredient_pairing",
    "get_nutrition_advice"
]
