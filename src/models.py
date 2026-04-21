"""
数据模型定义
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class Ingredient(BaseModel):
    """食材数据结构"""
    name: str = Field(..., description="食材名称")
    amount: float = Field(..., description="食材数量")
    unit: str = Field(..., description="食材单位")


class Nutrition(BaseModel):
    """营养成分数据结构"""
    protein: float = Field(..., description="蛋白质（克）")
    carbs: float = Field(..., description="碳水化合物（克）")
    fat: float = Field(..., description="脂肪（克）")
    fiber: float = Field(..., description="纤维（克）")


class Recipe(BaseModel):
    """食谱数据结构"""
    id: str = Field(..., description="食谱ID")
    name: str = Field(..., description="食谱名称")
    description: str = Field(..., description="食谱描述")
    cuisine: str = Field(..., description="菜系")
    ingredients: List[Ingredient] = Field(..., description="食材列表")
    steps: List[str] = Field(..., description="烹饪步骤")
    time: int = Field(..., description="烹饪时间（分钟）")
    difficulty: str = Field(..., description="难度（简单/中等/困难）")
    calories: int = Field(..., description="热量（卡路里）")
    nutrition: Nutrition = Field(..., description="营养成分")
    tags: List[str] = Field(default_factory=list, description="标签列表")
    health_goals: List[str] = Field(default_factory=list, description="健康目标列表")


class ParsedContext(BaseModel):
    """解析后的用户上下文"""
    physical_state: Optional[str] = Field(None, description="身体状态（疲劳/精力充沛）")
    taste_preference: List[str] = Field(default_factory=list, description="口味偏好（酸/甜/辣等）")
    health_goal: Optional[str] = Field(None, description="健康目标（减肥/增肌/养生）")
    time_limit: Optional[int] = Field(None, description="时间限制（分钟）")
    meal_type: Optional[str] = Field(None, description="餐次（早餐/午餐/晚餐）")
    dietary_restrictions: List[str] = Field(default_factory=list, description="饮食限制（素食/无麸质等）")
    available_ingredients: List[str] = Field(default_factory=list, description="已有食材列表")


class UserRequest(BaseModel):
    """用户请求结构"""
    user_input: str = Field(..., description="用户输入文本")
    parsed_context: Optional[ParsedContext] = Field(None, description="解析后的上下文")


class ToolResponse(BaseModel):
    """工具响应结构"""
    success: bool = Field(..., description="是否成功")
    data: Optional[dict] = Field(None, description="响应数据")
    error: Optional[str] = Field(None, description="错误信息")
    message: Optional[str] = Field(None, description="提示信息")
