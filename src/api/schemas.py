# -*- coding: utf-8 -*-
"""
API 请求/响应模型

定义 FastAPI 端点的 Pydantic 模型：
- ChatRequest: 聊天请求
- ChatResponse: 聊天响应（同步模式）
- StreamEvent: SSE 流式事件
- HealthResponse: 健康检查响应
- FeedbackPayload: 反馈事件
"""

from typing import Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """聊天请求模型

    Attributes:
        query: 用户查询文本
        user_id: 用户 ID（用于加载语义记忆）
        stream: 是否使用流式返回
        session_id: 会话 ID（用于对话历史关联）
    """
    query: str = Field(..., min_length=1, max_length=2000, description="用户查询文本")
    user_id: str = Field(default="", description="用户 ID")
    stream: bool = Field(default=False, description="是否使用流式返回")
    session_id: Optional[str] = Field(default=None, description="会话 ID")
    available_ingredients: list[str] = Field(default_factory=list, description="可用食材列表")
    allergies: list[str] = Field(default_factory=list, description="过敏原列表")
    disliked_ingredients: list[str] = Field(default_factory=list, description="不喜欢的食材列表")
    max_cooking_time: Optional[int] = Field(default=None, description="最大烹饪时间（分钟）")
    health_goal: Optional[str] = Field(default=None, description="健康目标")
    meal_type: Optional[str] = Field(default=None, description="餐次类型")
    prefer_inventory_first: bool = Field(default=False, description="是否优先使用库存食材")
    feedback: Optional["FeedbackPayload"] = Field(default=None, description="本轮附带的反馈事件")


class FeedbackPayload(BaseModel):
    recipe_id: str = Field(..., description="反馈对应的食谱 ID")
    rating: int = Field(default=0, ge=0, le=5, description="评分（0-5）")
    liked: Optional[bool] = Field(default=None, description="是否喜欢")
    taste_rating: Optional[int] = Field(default=None, ge=0, le=5, description="口味评分")
    difficulty_rating: Optional[int] = Field(default=None, ge=0, le=5, description="难度评分")
    time_accurate: Optional[bool] = Field(default=None, description="耗时是否符合预期")
    comment: Optional[str] = Field(default=None, description="反馈备注")
    tags: list[str] = Field(default_factory=list, description="反馈标签")


class RetrievalMetadata(BaseModel):
    retrieved_count: int = Field(default=0, description="检索返回文档数")
    reranked_count: int = Field(default=0, description="精排后文档数")
    constraint_count: int = Field(default=0, description="识别出的约束数量")
    filtered_doc_count: int = Field(default=0, description="被硬过滤的文档数量")
    hard_filter_reasons: dict = Field(default_factory=dict, description="硬过滤原因统计")
    inventory_match_ratio: float = Field(default=0.0, description="库存食材匹配率")
    goal_fit_score: float = Field(default=0.0, description="健康目标匹配分")
    expiry_urgency_score: float = Field(default=0.0, description="临期食材优先分")
    fallback_triggered: bool = Field(default=False, description="是否触发兜底")


class MemoryMetadata(BaseModel):
    feedback_signal_count: int = Field(default=0, description="本轮注入的反馈信号数量")
    recommendation_anchor_count: int = Field(default=0, description="本轮注入的推荐锚点数量")
    feedback_summary: str = Field(default="", description="最近反馈摘要")
    memory_readback_ok: bool = Field(default=False, description="是否读到了可用的记忆信号")
    stable_preference_count: int = Field(default=0, description="读到的稳定用户偏好数量")
    stable_preference_keys: list[str] = Field(default_factory=list, description="稳定用户偏好键列表")
    applied_stable_preference_keys: list[str] = Field(default_factory=list, description="本轮真正用于补足约束的稳定偏好键")


class ChatResponseMetadata(BaseModel):
    extracted_params: dict = Field(default_factory=dict, description="标准化后的查询约束")
    current_step: int = Field(default=0, description="当前步骤索引")
    retry_count: int = Field(default=0, description="重试次数")
    goal_type: str = Field(default="", description="planner 判定的本轮目标类型")
    planner_next_action: str = Field(default="retrieve", description="planner 判定的下一步动作")
    inherit_followup_direction: bool = Field(default=False, description="是否继承上一轮推荐方向")
    response_type: str = Field(default="recommendation", description="响应类型：clarification/recommendation/fallback")
    clarification_needed: bool = Field(default=False, description="是否需要澄清")
    clarification_question: str = Field(default="", description="澄清问题")
    missing_slots: list[str] = Field(default_factory=list, description="缺失槽位")
    next_expected_slot: str = Field(default="", description="下一步期望补充的槽位")
    fallback_triggered: bool = Field(default=False, description="是否触发统一兜底")
    followup_mode: str = Field(default="", description="follow-up 模式")
    followup_anchor_names: list[str] = Field(default_factory=list, description="follow-up 承接的推荐锚点")
    retrieval_stats: dict = Field(default_factory=dict, description="检索统计原始信息")
    active_skill: str = Field(default="", description="当前激活的 skill")
    skill_contract: dict = Field(default_factory=dict, description="当前 skill 的运行时合同摘要")
    skill_capability: dict = Field(default_factory=dict, description="当前 skill 的运行时能力边界状态")
    history_message_count: int = Field(default=0, description="注入的历史消息数量")
    interaction_id: str = Field(default="", description="写回的交互记录 ID")
    feedback_logged: bool = Field(default=False, description="是否已写回反馈")
    recommended_recipes: list[dict] = Field(default_factory=list, description="本轮推荐结果摘要")
    retrieval: RetrievalMetadata = Field(default_factory=RetrievalMetadata, description="检索摘要")
    memory: MemoryMetadata = Field(default_factory=MemoryMetadata, description="记忆读回摘要")
    evaluation: dict = Field(default_factory=dict, description="评估结果")
    quality_signals: dict = Field(default_factory=dict, description="统一 skill 质量信号")


class ChatResponse(BaseModel):
    """聊天响应模型（同步模式）

    Attributes:
        response: Agent 生成的回复文本
        request_id: 请求唯一标识
        intent: 意图分类结果
        latency_ms: 总延迟（毫秒）
        token_usage: LLM token 使用统计
        metadata: 附加元数据
    """
    response: str = Field(..., description="Agent 生成的回复文本")
    request_id: str = Field(..., description="请求唯一标识")
    intent: str = Field(default="", description="意图分类结果")
    latency_ms: float = Field(default=0.0, description="总延迟（毫秒）")
    token_usage: Optional[dict] = Field(default=None, description="LLM token 使用统计")
    metadata: Optional[ChatResponseMetadata] = Field(default=None, description="附加元数据")


class SessionSummary(BaseModel):
    session_id: str = Field(default="", description="会话 ID")
    message_count: int = Field(default=0, description="内存态消息数量")
    interaction_count: int = Field(default=0, description="持久化交互数量")
    preview: str = Field(default="", description="最近会话预览")
    updated_at: str = Field(default="", description="最近更新时间")
    last_interaction_at: str = Field(default="", description="最近交互时间")
    source: str = Field(default="memory", description="会话摘要来源")


class SessionListResponse(BaseModel):
    user_id: str = Field(default="", description="用户 ID")
    sessions: list[SessionSummary] = Field(default_factory=list, description="最近会话摘要")


class SessionHistoryMessage(BaseModel):
    role: str = Field(default="assistant", description="消息角色")
    content: str = Field(default="", description="消息内容")


class SessionInteractionRecord(BaseModel):
    interaction_id: str = Field(default="", description="交互 ID")
    session_id: str = Field(default="", description="会话 ID")
    user_input: str = Field(default="", description="用户输入")
    agent_response: str = Field(default="", description="Agent 回复")
    recommended_recipes: list[dict] = Field(default_factory=list, description="推荐结果摘要")
    context: dict = Field(default_factory=dict, description="交互上下文")
    created_at: str = Field(default="", description="交互创建时间")


class SessionHistoryResponse(BaseModel):
    user_id: str = Field(default="", description="用户 ID")
    session_id: str = Field(default="", description="会话 ID")
    history_source: str = Field(default="empty", description="历史来源")
    messages: list[SessionHistoryMessage] = Field(default_factory=list, description="可恢复的对话消息")
    interactions: list[SessionInteractionRecord] = Field(default_factory=list, description="持久化交互记录")


class StreamEvent(BaseModel):
    """SSE 流式事件模型

    Attributes:
        event: 事件类型 (chunk / node_start / node_end / done / error)
        data: 事件数据
        request_id: 请求唯一标识
    """
    event: str = Field(..., description="事件类型")
    data: str = Field(default="", description="事件数据")
    request_id: str = Field(default="", description="请求唯一标识")


class HealthResponse(BaseModel):
    """健康检查响应模型

    Attributes:
        status: 服务状态 (ok / degraded)
        version: API 版本
        components: 各组件状态
    """
    status: str = Field(..., description="服务状态")
    version: str = Field(default="4.0.0", description="API 版本")
    components: dict = Field(default_factory=dict, description="各组件状态")


ChatRequest.model_rebuild()
