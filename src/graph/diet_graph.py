"""
主图编排

使用 LangGraph StateGraph 组装 Diet Agent V4 的完整流水线：
Router → Planner → Retriever → Generator → Evaluator

流程说明：
1. Router: 意图分类
2. Planner: 规划执行步骤（仅 recipe_search）
3. Retriever: 混合检索 + 精排
4. Generator: 基于上下文生成回复
5. Evaluator: 质量评估（Evaluator-Optimizer 循环）

本模块作为 V4 独立入口，与 V2 Agent (src/agent/) 并存。
"""

from typing import Optional, AsyncGenerator

from langgraph.graph import StateGraph, END

from .schemas import normalize_extracted_params
from .state import DietAgentState, build_initial_diet_agent_state, normalize_stable_user_preferences
from .nodes import (
    router_node,
    planner_node,
    retriever_node,
    generator_node,
    evaluator_node,
)
from .edges import route_after_planner, route_by_intent, should_regenerate, should_skip_eval
from ..utils.token_usage import get_current_token_usage, merge_token_usage, token_usage_scope
from ..utils.logger import get_logger


logger = get_logger(__name__)


def build_diet_graph():
    """构建饮食 Agent LangGraph
    
    创建 StateGraph，添加节点和条件边，编译并返回。
    
    图结构：
        router --[route_by_intent]--> planner / retriever / generator
        planner --> retriever
        retriever --> generator
        generator --[should_skip_eval]--> evaluator / END
        evaluator --[should_regenerate]--> generator(retry) / END
    
    Returns:
        编译后的 CompiledGraph 实例
    """
    logger.info("开始构建 Diet Agent LangGraph")
    
    builder = StateGraph(DietAgentState)
    
    # 添加节点
    builder.add_node("router", router_node)
    builder.add_node("planner", planner_node)
    builder.add_node("retriever", retriever_node)
    builder.add_node("generator", generator_node)
    builder.add_node("evaluator", evaluator_node)
    
    # 设置入口
    builder.set_entry_point("router")
    
    # Router → 条件路由
    builder.add_conditional_edges("router", route_by_intent, {
        "planner": "planner",
        "retriever": "retriever",
        "generator": "generator",
    })
    
    # Planner → 条件路由（澄清短路 or 正常检索）
    builder.add_conditional_edges("planner", route_after_planner, {
        "generator": "generator",
        "retriever": "retriever",
    })
    
    # Retriever → Generator
    builder.add_edge("retriever", "generator")
    
    # Generator → 条件路由（是否跳过评估）
    builder.add_conditional_edges("generator", should_skip_eval, {
        "evaluate": "evaluator",
        "pass": END,
    })
    
    # Evaluator → 条件路由（是否重试）
    builder.add_conditional_edges("evaluator", should_regenerate, {
        "retry": "generator",
        "pass": END,
    })
    
    graph = builder.compile()
    logger.info("Diet Agent LangGraph 构建完成")
    return graph


def run_diet_agent(
    query: str,
    user_id: str = "",
    config: Optional[dict] = None,
    return_state: bool = False,
    history_messages: Optional[list] = None,
    initial_extracted_params: Optional[dict] = None,
    initial_recent_feedback_signals: Optional[dict] = None,
    initial_recent_recommended_recipes: Optional[list[dict]] = None,
    initial_stable_user_preferences: Optional[dict] = None,
    skip_graph_eval: bool = False,
):
    """便捷函数：运行饮食 Agent
    
    构建初始 state，调用 graph.invoke()，提取并返回 response。
    
    Args:
        query: 用户查询文本
        user_id: 用户 ID（Phase 3 新增，用于激活语义记忆）
        config: 可选的 LangGraph 运行配置
        return_state: 若为 True，返回 (response, final_state) 元组
        history_messages: 上一轮 messages 列表，注入初始 state 实现多轮历史
    
    Returns:
        Agent 生成的回复文本（或 (response, state) 元组）
    """
    from langchain_core.messages import HumanMessage
    
    logger.info(f"运行 Diet Agent: query='{query[:100]}'")
    
    graph = build_diet_graph()
    
    # 构建初始 state（前置历史 messages 实现多轮对话）
    init_messages = list(history_messages or []) + [HumanMessage(content=query)]
    initial_state = build_initial_diet_agent_state(
        messages=init_messages,
        user_id=user_id,
    )
    if initial_extracted_params is not None:
        initial_state["extracted_params"] = normalize_extracted_params(initial_extracted_params)
    if initial_recent_feedback_signals:
        initial_state["recent_feedback_signals"] = dict(initial_recent_feedback_signals)
    if initial_recent_recommended_recipes:
        initial_state["recent_recommended_recipes"] = [dict(item) for item in initial_recent_recommended_recipes]
    if initial_stable_user_preferences:
        initial_state["stable_user_preferences"] = normalize_stable_user_preferences(initial_stable_user_preferences)
    initial_state["skip_graph_eval"] = bool(skip_graph_eval)

    # 可选注入 Langfuse 回调
    try:
        from ..observability.langfuse_integration import get_langfuse_callback
        langfuse_cb = get_langfuse_callback()
        if langfuse_cb is not None:
            config = config or {}
            existing_cbs = config.get("callbacks", [])
            config["callbacks"] = existing_cbs + [langfuse_cb]
    except Exception:
        pass  # Langfuse 不可用时降级
    
    # 运行图
    with token_usage_scope(initial_state.get("token_usage")):
        result = graph.invoke(initial_state, config=config)
        aggregated_token_usage = get_current_token_usage()

    merged_token_usage = merge_token_usage(result.get("token_usage"), aggregated_token_usage)
    if merged_token_usage:
        result["token_usage"] = merged_token_usage
    
    response = result.get("response", "抱歉，未能生成回复。")
    logger.info(f"Diet Agent 运行完成，回复长度: {len(response)}")
    if return_state:
        return response, result
    return response


async def arun_diet_agent_stream(
    query: str,
    user_id: str = "",
    config: Optional[dict] = None,
    initial_extracted_params: Optional[dict] = None,
) -> AsyncGenerator:
    """异步流式运行饮食 Agent

    使用 graph.astream_events() 逐步返回事件，
    供 FastAPI SSE 端点使用。

    Args:
        query: 用户查询文本
        user_id: 用户 ID
        config: LangGraph 运行配置（可包含 callbacks）

    Yields:
        LangGraph 事件字典
    """
    from langchain_core.messages import HumanMessage

    logger.info(f"异步流式运行 Diet Agent: query='{query[:100]}'")

    graph = build_diet_graph()

    initial_state = build_initial_diet_agent_state(
        messages=[HumanMessage(content=query)],
        user_id=user_id,
        request_id=(config or {}).get("request_id", ""),
    )
    if initial_extracted_params is not None:
        initial_state["extracted_params"] = normalize_extracted_params(initial_extracted_params)

    async for event in graph.astream_events(
        initial_state, version="v2", config=config
    ):
        yield event
