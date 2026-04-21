"""
数据加载和文档构建模块

负责将结构化食谱数据转换为适合向量化的自然语言文本

Requirements:
- 6.1: 将查询文本转换为向量
- 7.1: 从 JSON 文件加载食谱数据并向量化
"""

from typing import Dict, Any
from ..models import Recipe


def build_document_text(recipe: Dict[str, Any]) -> str:
    """
    构建食谱的描述性文本，用于向量化
    
    将结构化的食谱数据转换为自然语言文本，包含：
    - 食谱名称和描述
    - 烹饪时间和难度
    - 主要食材
    - 标签
    
    Args:
        recipe: 食谱字典，包含 name, description, time, difficulty, ingredients, tags 等字段
    
    Returns:
        str: 用于向量化的自然语言文本
    
    Requirements: 6.1, 7.1
    
    Example:
        >>> recipe = {
        ...     "name": "番茄炒蛋",
        ...     "description": "经典家常菜",
        ...     "time": 15,
        ...     "difficulty": "简单",
        ...     "ingredients": [{"name": "番茄"}, {"name": "鸡蛋"}],
        ...     "tags": ["快手", "酸甜", "家常", "下饭"]
        ... }
        >>> text = build_document_text(recipe)
        >>> print(text)
        番茄炒蛋：经典家常菜。烹饪时间15分钟，难度简单。包含食材：番茄、鸡蛋。标签：快手、酸甜、家常、下饭。
    """
    # 提取基本信息
    name = recipe.get("name", "")
    description = recipe.get("description", "")
    time = recipe.get("time", 0)
    difficulty = recipe.get("difficulty", "")
    
    # 提取食材名称
    ingredients = recipe.get("ingredients", [])
    ingredients_text = "、".join([ing.get("name", "") for ing in ingredients if ing.get("name")])
    
    # 提取标签
    tags = recipe.get("tags", [])
    tags_text = "、".join(tags) if tags else ""
    
    # 提取烹饪步骤
    steps = recipe.get("steps", [])
    steps_text = " ".join([f"第{i+1}步：{s}" for i, s in enumerate(steps)]) if steps else ""

    # 提取热量和营养
    calories = recipe.get("calories", 0)
    nutrition = recipe.get("nutrition", {})
    protein = nutrition.get("protein", 0)

    # 构建自然语言文本
    text_parts = []
    
    # 名称和描述
    if name and description:
        text_parts.append(f"{name}：{description}。")
    elif name:
        text_parts.append(f"{name}。")
    
    # 时间和难度
    if time and difficulty:
        text_parts.append(f"烹饪时间{time}分钟，难度{difficulty}。")
    elif time:
        text_parts.append(f"烹饪时间{time}分钟。")
    elif difficulty:
        text_parts.append(f"难度{difficulty}。")

    # 热量和营养
    if calories:
        text_parts.append(f"热量约{calories}卡。")
    if protein:
        text_parts.append(f"蛋白质{protein}克。")
    
    # 食材
    if ingredients_text:
        text_parts.append(f"包含食材：{ingredients_text}。")

    # 烹饪步骤（核心内容，对"怎么做"查询至关重要）
    if steps_text:
        text_parts.append(f"烹饪步骤：{steps_text}。")
    
    # 标签
    if tags_text:
        text_parts.append(f"标签：{tags_text}。")
    
    return "".join(text_parts)
