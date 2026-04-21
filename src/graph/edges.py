"""
条件边函数

定义 LangGraph StateGraph 中的条件路由逻辑：
- route_by_intent: 根据意图路由到不同节点
- should_regenerate: 评估后决定是否重新生成
- should_skip_eval: 决定是否跳过评估

复用模块:
- src/config.py::get_settings()
- src/utils/logger.py::get_logger()
"""

from .state import DietAgentState
from ..config import get_settings
from ..utils.logger import get_logger


logger = get_logger(__name__)


def route_by_intent(state: DietAgentState) -> str:
    """根据意图路由到下一个节点
    
    路由规则：
    - recipe_search → planner（需要规划执行步骤）
    - nutrition_query → retriever（直接检索）
    - ingredient_check → retriever（直接检索）
    - chitchat → generator（直接生成，无需检索）
    - video_summary → generator（直接执行 B 站视频总结 workflow）
    - 默认 → retriever（兜底）
    
    Args:
        state: LangGraph 全局状态
    
    Returns:
        下一个节点名称
    """
    intent = state.get("intent", "")
    
    route_map = {
        "recipe_search": "planner",
        "nutrition_query": "retriever",
        "ingredient_check": "retriever",
        "chitchat": "generator",
        "video_summary": "generator",
    }
    
    next_node = route_map.get(intent, "retriever")
    logger.info(f"意图路由: intent='{intent}' → next='{next_node}'")
    return next_node


def route_after_planner(state: DietAgentState) -> str:
    """根据 planner 决策决定从 planner 进入 generator 还是 retriever。"""
    planner_next_action = str(state.get("planner_next_action") or "").strip()
    clarification_needed = state.get("clarification_needed", False)

    if planner_next_action in {"clarify", "direct_followup"}:
        logger.info(f"规划后命中动作短路，planner_next_action={planner_next_action}，planner → generator")
        return "generator"
    if clarification_needed:
        logger.info("规划后命中澄清短路，planner → generator")
        return "generator"

    logger.info("规划后进入正常检索链路，planner → retriever")
    return "retriever"


def should_regenerate(state: DietAgentState) -> str:
    """评估后决定是否重新生成
    
    判断逻辑：
    - is_satisfactory == True → "pass"（结束）
    - retry_count >= max_retries → "pass"（超限，强制结束）
    - 否则 → "retry"（重新生成）
    
    Args:
        state: LangGraph 全局状态
    
    Returns:
        "pass" 或 "retry"
    """
    settings = get_settings()
    max_retries = settings.graph_max_retries
    
    evaluation = state.get("evaluation", {})
    is_satisfactory = evaluation.get("is_satisfactory", True)
    retry_count = state.get("retry_count", 0)
    
    if is_satisfactory:
        logger.info("评估通过，结束流程")
        return "pass"
    
    if retry_count >= max_retries:
        logger.warning(f"达到最大重试次数 ({max_retries})，强制结束")
        return "pass"
    
    logger.info(f"评估未通过，重试 (retry_count={retry_count}/{max_retries})")
    return "retry"


def should_skip_eval(state: DietAgentState) -> str:
    """决定是否跳过评估
    
    判断逻辑：
    - chitchat 意图 → "pass"（跳过评估直接结束）
    - 澄清回复 → "pass"（跳过评估）
    - skip_graph_eval → "pass"（跳过评估）
    - 其他意图 → "evaluate"（进入评估节点）
    
    Args:
        state: LangGraph 全局状态
    
    Returns:
        "pass" 或 "evaluate"
    """
    intent = state.get("intent", "")
    response_type = state.get("response_type", "recommendation")
    skip_graph_eval = bool(state.get("skip_graph_eval", False))

    if skip_graph_eval:
        logger.info("命中 skip_graph_eval，跳过评估")
        return "pass"
    
    if intent in {"chitchat", "video_summary"}:
        logger.info(f"意图 {intent} 跳过评估")
        return "pass"
    if response_type == "clarification":
        logger.info("澄清回复，跳过评估")
        return "pass"
    
    logger.info("非闲聊意图，进入评估")
    return "evaluate"
