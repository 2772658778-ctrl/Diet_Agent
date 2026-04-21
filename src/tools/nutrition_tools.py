"""
营养分析工具

提供食谱营养成分查询和分析功能

Requirements:
- 2.1: 返回热量、蛋白质、碳水化合物和脂肪含量
- 2.2: 食谱不存在时返回明确的错误提示
- 2.3: 使用易于理解的单位（克、卡路里）
- 8.1: LLM API 调用失败时重试
- 8.2: 重试失败后返回友好错误提示
- 8.3: 向量数据库连接失败时记录错误
- 8.4: 工具执行失败时返回包含错误信息的 JSON 响应
"""

import json
from typing import Optional, Dict, Any
from langchain.tools import tool
from ..vectorstore.chroma_client import get_vectorstore
from ..utils.logger import get_logger


logger = get_logger(__name__)


def _get_health_advice(calories: int, protein: float, carbs: float, fat: float, fiber: float) -> str:
    """
    根据营养成分生成健康建议
    
    Args:
        calories: 热量（卡路里）
        protein: 蛋白质（克）
        carbs: 碳水化合物（克）
        fat: 脂肪（克）
        fiber: 纤维（克）
    
    Returns:
        健康建议文本
    """
    advice = []
    
    # 热量判断
    if calories < 200:
        advice.append("低热量")
    elif calories > 500:
        advice.append("高热量")
    
    # 蛋白质判断
    if protein > 25:
        advice.append("高蛋白")
    
    # 脂肪判断
    if fat < 10:
        advice.append("低脂")
    elif fat > 30:
        advice.append("高脂")
    
    # 纤维判断
    if fiber > 5:
        advice.append("高纤维")
    
    # 适合人群判断
    suitability = []
    if calories < 300 and fat < 15:
        suitability.append("适合减肥")
    if protein > 25:
        suitability.append("适合增肌")
    if fat < 15 and fiber > 3:
        suitability.append("适合养生")
    
    result = "、".join(advice) if advice else "营养均衡"
    if suitability:
        result += "，" + "、".join(suitability)
    
    return result


@tool
def analyze_nutrition(recipe_name: str) -> str:
    """
    分析食谱营养成分
    
    根据食谱名称查询营养信息，包括热量、蛋白质、碳水化合物、脂肪和纤维。
    
    Args:
        recipe_name: 食谱名称
    
    Returns:
        JSON 字符串，包含营养信息或错误信息
        
    Examples:
        >>> analyze_nutrition("番茄炒蛋")
        >>> analyze_nutrition("健身鸡胸肉沙拉")
    """
    try:
        logger.info(f"开始分析食谱营养: recipe_name='{recipe_name}'")
        
        # 输入验证
        if not recipe_name or not recipe_name.strip():
            logger.warning("食谱名称为空")
            return json.dumps({
                "error": "参数错误",
                "message": "请提供食谱名称"
            }, ensure_ascii=False)
        
        # 获取向量存储实例
        vectorstore = get_vectorstore()
        if vectorstore is None:
            error_msg = "向量数据库未初始化，请先运行初始化脚本"
            logger.error(error_msg)
            return json.dumps({
                "error": "系统错误",
                "message": "数据库未就绪，请联系管理员"
            }, ensure_ascii=False)
        
        # 1. 通过食谱名称精确查询（带重试机制）
        max_retries = 2
        results = None
        last_error = None
        
        for attempt in range(max_retries):
            try:
                # 使用名称作为查询，并限制返回数量为 1
                results = vectorstore.similarity_search(
                    query=recipe_name,
                    k=1
                )
                logger.info(f"查询返回 {len(results)} 条结果")
                break
            except Exception as e:
                last_error = e
                logger.warning(f"向量检索失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                
                if attempt < max_retries - 1:
                    import time
                    time.sleep(0.5)  # 短暂延迟后重试
        
        if results is None:
            logger.error(f"向量检索失败，已达到最大重试次数: {last_error}", exc_info=True)
            return json.dumps({
                "error": "检索失败",
                "message": "查询服务暂时不可用，请稍后再试"
            }, ensure_ascii=False)
        
        # 2. 检查是否找到食谱
        if not results:
            logger.info(f"未找到食谱: {recipe_name}")
            return json.dumps({
                "error": "食谱不存在",
                "message": f"未找到名为 '{recipe_name}' 的食谱，请检查食谱名称是否正确"
            }, ensure_ascii=False)
        
        # 3. 获取第一个结果（最相关的）
        doc = results[0]
        metadata = doc.metadata
        
        # 验证是否是精确匹配（名称相似度检查）
        found_name = metadata.get("name", "")
        if recipe_name.lower() not in found_name.lower() and found_name.lower() not in recipe_name.lower():
            logger.warning(f"找到的食谱 '{found_name}' 与查询 '{recipe_name}' 不完全匹配")
            # 仍然返回结果，但提示可能不是精确匹配
        
        # 4. 提取营养信息（带验证）
        try:
            calories = int(metadata.get("calories", 0))
            protein = float(metadata.get("protein", 0.0))
            carbs = float(metadata.get("carbs", 0.0))
            fat = float(metadata.get("fat", 0.0))
            fiber = float(metadata.get("fiber", 0.0))
        except (ValueError, TypeError) as e:
            logger.error(f"营养数据格式错误: {e}")
            return json.dumps({
                "error": "数据错误",
                "message": "食谱营养数据格式异常，请联系管理员"
            }, ensure_ascii=False)
        
        # 5. 生成健康建议
        try:
            health_advice = _get_health_advice(calories, protein, carbs, fat, fiber)
        except Exception as e:
            logger.warning(f"生成健康建议失败: {e}")
            health_advice = "营养信息已提供"
        
        # 6. 格式化返回结果
        nutrition_info = {
            "success": True,
            "recipe_name": found_name,
            "nutrition": {
                "calories": f"{calories} 卡路里",
                "protein": f"{protein} 克",
                "carbs": f"{carbs} 克",
                "fat": f"{fat} 克",
                "fiber": f"{fiber} 克"
            },
            "health_advice": health_advice
        }
        
        logger.info(f"成功返回食谱 '{found_name}' 的营养信息")
        
        return json.dumps(nutrition_info, ensure_ascii=False)
        
    except Exception as e:
        logger.error(f"analyze_nutrition 执行失败: {e}", exc_info=True)
        return json.dumps({
            "error": "系统错误",
            "message": "营养分析失败，请稍后再试"
        }, ensure_ascii=False)
