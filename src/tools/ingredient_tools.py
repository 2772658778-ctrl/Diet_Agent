"""
食材查询工具

提供食材信息查询和搭配建议功能

Requirements:
- 3.1: 返回食材的基本信息
- 3.2: 检查多个食材的搭配组合
- 3.3: 说明搭配的原因（如"营养互补"）
- 8.1: LLM API 调用失败时重试
- 8.2: 重试失败后返回友好错误提示
- 8.3: 向量数据库连接失败时记录错误
- 8.4: 工具执行失败时返回包含错误信息的 JSON 响应
"""

import json
from typing import List, Dict, Any, Tuple
from langchain.tools import tool
from ..utils.logger import get_logger


logger = get_logger(__name__)


# 食材基本信息数据库（MVP 阶段使用静态数据）
INGREDIENT_INFO = {
    "番茄": {
        "category": "蔬菜",
        "nutrition": "富含维生素C、番茄红素",
        "benefits": "抗氧化、保护心血管"
    },
    "鸡蛋": {
        "category": "蛋类",
        "nutrition": "高蛋白、富含卵磷脂",
        "benefits": "增强记忆力、补充优质蛋白"
    },
    "鸡胸肉": {
        "category": "肉类",
        "nutrition": "高蛋白、低脂肪",
        "benefits": "增肌减脂、提供优质蛋白"
    },
    "西兰花": {
        "category": "蔬菜",
        "nutrition": "富含维生素C、膳食纤维",
        "benefits": "抗癌、促进消化"
    },
    "生菜": {
        "category": "蔬菜",
        "nutrition": "低热量、富含水分和纤维",
        "benefits": "减肥、补充维生素"
    },
    "圣女果": {
        "category": "水果",
        "nutrition": "富含维生素C、番茄红素",
        "benefits": "美容养颜、抗氧化"
    },
    "黄瓜": {
        "category": "蔬菜",
        "nutrition": "低热量、富含水分",
        "benefits": "清热解毒、美容养颜"
    },
    "橄榄油": {
        "category": "油脂",
        "nutrition": "富含不饱和脂肪酸",
        "benefits": "保护心血管、抗氧化"
    },
    "豆腐": {
        "category": "豆制品",
        "nutrition": "高蛋白、富含钙质",
        "benefits": "补钙、提供植物蛋白"
    },
    "猪肉": {
        "category": "肉类",
        "nutrition": "富含蛋白质和铁",
        "benefits": "补铁、提供能量"
    },
    "牛肉": {
        "category": "肉类",
        "nutrition": "高蛋白、富含铁和锌",
        "benefits": "增肌、补铁"
    },
    "鲈鱼": {
        "category": "水产",
        "nutrition": "高蛋白、低脂肪、富含DHA",
        "benefits": "健脑、保护心血管"
    },
    "三文鱼": {
        "category": "水产",
        "nutrition": "富含Omega-3脂肪酸、高蛋白",
        "benefits": "保护心血管、健脑"
    },
    "虾": {
        "category": "水产",
        "nutrition": "高蛋白、低脂肪、富含钙",
        "benefits": "补钙、增强免疫力"
    },
    "胡萝卜": {
        "category": "蔬菜",
        "nutrition": "富含胡萝卜素、维生素A",
        "benefits": "保护视力、增强免疫力"
    },
    "土豆": {
        "category": "蔬菜",
        "nutrition": "富含淀粉、维生素C",
        "benefits": "提供能量、促进消化"
    },
    "青椒": {
        "category": "蔬菜",
        "nutrition": "富含维生素C",
        "benefits": "增强免疫力、促进新陈代谢"
    },
    "木耳": {
        "category": "菌类",
        "nutrition": "富含膳食纤维、铁",
        "benefits": "清肠排毒、补铁"
    },
    "花生": {
        "category": "坚果",
        "nutrition": "富含不饱和脂肪酸、蛋白质",
        "benefits": "保护心血管、提供能量"
    },
    "米饭": {
        "category": "主食",
        "nutrition": "富含碳水化合物",
        "benefits": "提供能量"
    }
}


# 食材搭配规则（MVP 阶段使用简单规则）
PAIRING_RULES = [
    {
        "ingredients": ["番茄", "鸡蛋"],
        "reason": "营养互补，番茄的维生素C促进鸡蛋蛋白质吸收",
        "rating": "推荐"
    },
    {
        "ingredients": ["鸡胸肉", "西兰花"],
        "reason": "高蛋白低脂，西兰花的纤维促进消化，适合健身人群",
        "rating": "推荐"
    },
    {
        "ingredients": ["鸡胸肉", "生菜"],
        "reason": "低脂高蛋白，生菜清爽解腻，适合减肥",
        "rating": "推荐"
    },
    {
        "ingredients": ["豆腐", "猪肉"],
        "reason": "动植物蛋白互补，豆腐吸收肉香，口感丰富",
        "rating": "推荐"
    },
    {
        "ingredients": ["鲈鱼", "姜"],
        "reason": "姜可去腥提鲜，保留鱼肉原味",
        "rating": "推荐"
    },
    {
        "ingredients": ["三文鱼", "柠檬"],
        "reason": "柠檬去腥增鲜，维生素C丰富",
        "rating": "推荐"
    },
    {
        "ingredients": ["胡萝卜", "木耳"],
        "reason": "营养互补，胡萝卜素和膳食纤维丰富",
        "rating": "推荐"
    },
    {
        "ingredients": ["土豆", "青椒"],
        "reason": "口感搭配好，青椒的维生素C保护土豆营养",
        "rating": "推荐"
    }
]


def _get_ingredient_info(ingredient_name: str) -> Dict[str, Any]:
    """
    获取食材基本信息
    
    Args:
        ingredient_name: 食材名称
    
    Returns:
        食材信息字典
    """
    if ingredient_name in INGREDIENT_INFO:
        info = INGREDIENT_INFO[ingredient_name].copy()
        info["name"] = ingredient_name
        return info
    else:
        # 未知食材，返回基本信息
        return {
            "name": ingredient_name,
            "category": "未知",
            "nutrition": "暂无详细信息",
            "benefits": "请查阅相关资料"
        }


def _check_pairing(ingredients: List[str]) -> List[Dict[str, Any]]:
    """
    检查食材搭配
    
    Args:
        ingredients: 食材列表
    
    Returns:
        搭配建议列表
    """
    pairings = []
    
    # 检查两两搭配
    for i in range(len(ingredients)):
        for j in range(i + 1, len(ingredients)):
            ing1 = ingredients[i]
            ing2 = ingredients[j]
            
            # 查找匹配的搭配规则
            for rule in PAIRING_RULES:
                rule_ings = rule["ingredients"]
                if (ing1 in rule_ings and ing2 in rule_ings) or \
                   (ing2 in rule_ings and ing1 in rule_ings):
                    pairings.append({
                        "ingredients": [ing1, ing2],
                        "reason": rule["reason"],
                        "rating": rule["rating"]
                    })
                    break
    
    return pairings


@tool
def check_ingredients(ingredients_str: str) -> str:
    """
    检查食材信息
    
    解析逗号分隔的食材列表，返回食材信息和搭配建议。
    
    Args:
        ingredients_str: 食材列表，逗号分隔（如："鸡胸肉,西兰花,橄榄油"）
    
    Returns:
        JSON 字符串，包含食材信息和搭配建议
        
    Examples:
        >>> check_ingredients("鸡胸肉,西兰花")
        >>> check_ingredients("番茄,鸡蛋")
        >>> check_ingredients("鲈鱼")
    """
    try:
        logger.info(f"开始检查食材: ingredients_str='{ingredients_str}'")
        
        # 1. 输入验证
        if not ingredients_str or not ingredients_str.strip():
            logger.warning("食材列表为空")
            return json.dumps({
                "error": "参数错误",
                "message": "请提供至少一个食材名称"
            }, ensure_ascii=False)
        
        # 检查输入长度
        if len(ingredients_str) > 200:
            logger.warning(f"食材列表过长: {len(ingredients_str)} 字符")
            return json.dumps({
                "error": "参数错误",
                "message": "食材列表过长，请限制在 200 字符以内"
            }, ensure_ascii=False)
        
        # 分割并清理食材名称
        try:
            ingredients = [ing.strip() for ing in ingredients_str.split(",") if ing.strip()]
        except Exception as e:
            logger.error(f"解析食材列表失败: {e}")
            return json.dumps({
                "error": "参数错误",
                "message": "食材列表格式错误，请使用逗号分隔"
            }, ensure_ascii=False)
        
        if not ingredients:
            logger.warning("解析后食材列表为空")
            return json.dumps({
                "error": "参数错误",
                "message": "请提供有效的食材名称"
            }, ensure_ascii=False)
        
        # 限制食材数量
        if len(ingredients) > 10:
            logger.warning(f"食材数量过多: {len(ingredients)}，将限制为前 10 个")
            ingredients = ingredients[:10]
        
        logger.info(f"解析到 {len(ingredients)} 个食材: {ingredients}")
        
        # 2. 获取每个食材的基本信息
        ingredients_info = []
        for ingredient in ingredients:
            try:
                # 验证食材名称
                if len(ingredient) > 20:
                    logger.warning(f"食材名称过长: {ingredient}")
                    continue
                
                info = _get_ingredient_info(ingredient)
                ingredients_info.append(info)
            except Exception as e:
                logger.warning(f"获取食材信息失败: {ingredient}, 错误: {e}")
                # 添加基本信息
                ingredients_info.append({
                    "name": ingredient,
                    "category": "未知",
                    "nutrition": "暂无详细信息",
                    "benefits": "请查阅相关资料"
                })
        
        if not ingredients_info:
            logger.error("所有食材信息获取失败")
            return json.dumps({
                "error": "数据错误",
                "message": "无法获取食材信息，请检查食材名称"
            }, ensure_ascii=False)
        
        # 3. 检查食材搭配（仅当有多个食材时）
        pairings = []
        if len(ingredients) > 1:
            try:
                pairings = _check_pairing(ingredients)
                logger.info(f"找到 {len(pairings)} 个搭配建议")
            except Exception as e:
                logger.warning(f"检查食材搭配失败: {e}")
        
        # 4. 格式化返回结果
        result = {
            "success": True,
            "count": len(ingredients_info),
            "ingredients": ingredients_info
        }
        
        # 添加搭配建议
        if pairings:
            result["pairings"] = pairings
            result["pairing_summary"] = f"找到 {len(pairings)} 个推荐搭配"
        else:
            if len(ingredients) > 1:
                result["pairing_summary"] = "暂无特定搭配建议，但这些食材可以一起使用"
            else:
                result["pairing_summary"] = "单个食材，无需搭配建议"
        
        logger.info(f"成功返回 {len(ingredients_info)} 个食材的信息")
        
        return json.dumps(result, ensure_ascii=False)
        
    except Exception as e:
        logger.error(f"check_ingredients 执行失败: {e}", exc_info=True)
        return json.dumps({
            "error": "系统错误",
            "message": "食材查询失败，请稍后再试"
        }, ensure_ascii=False)
