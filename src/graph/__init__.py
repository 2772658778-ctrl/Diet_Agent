"""
Graph 模块 — LangGraph 饮食 Agent V4

使用 LangGraph StateGraph 实现 Router→Planner→Retriever→Generator→Evaluator 显式流水线，
替代原有 AgentExecutor (ReAct) 的黑盒循环。

主要导出：
- build_diet_graph: 构建编译后的 LangGraph
- DietAgentState: 全局状态 TypedDict
- run_diet_agent: 便捷运行函数
"""

from .diet_graph import build_diet_graph, run_diet_agent
from .state import DietAgentState

__all__ = [
    "build_diet_graph",
    "run_diet_agent",
    "DietAgentState",
]
