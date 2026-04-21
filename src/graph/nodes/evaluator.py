"""
质量评估节点

使用 LLM + Structured Output 评估生成回复的质量，维度包括：
- 相关性：回复是否与用户查询相关
- 完整性：回复是否完整地解答了问题
- 忠实性：回复是否基于检索文档

Phase 1 新增 Self-RAG 补充维度（当 rag_strategy='adaptive' 时启用）：
- 幻觉检测：检测不被文档支持的声明
- 有用性判断：回复是否真正解决用户问题

chitchat 意图跳过评估，直接标记为通过。

复用模块:
- src/graph/llm.py::get_graph_llm()
- src/rag/self_rag.py::SelfRAGJudge  (Phase 1)
- src/utils/logger.py::get_logger()
"""

from langchain_core.messages import HumanMessage, SystemMessage

from diet_agent.runtime import build_skill_quality_signals
from ..state import DietAgentState
from ..schemas import EvaluationResult
from ..llm import get_graph_llm
from ...config import get_settings
from ...utils.logger import get_logger


logger = get_logger(__name__)

EVALUATOR_SYSTEM_PROMPT = """你是一个回复质量评估器。请评估以下饮食助手的回复质量。

## 评估维度

1. **相关性**：回复是否与用户查询直接相关
2. **完整性**：回复是否完整地解答了用户的问题
3. **忠实性**：回复是否基于提供的参考文档，没有编造信息
4. **实用性**：回复是否提供了实用的建议（如烹饪时间、营养信息等）

输出改进建议时请额外遵守：
- 不要建议模型把“文档未直接覆盖”的食材组合、菜名或步骤伪装成证据事实
- 如果证据不足，可以建议模型：
  - 明确区分“证据支持”与“通用建议”
  - 增加明确标注的通用烹饪建议/营养建议
  - 引导用户补充条件，或发起下一轮更有针对性的检索
- 不要建议模型为了提高食材利用率，就把未覆盖食材硬并入已有菜谱做法

## 评估标准

- 如果回复在所有维度上基本合格，标记为 is_satisfactory=true
- 如果有明显问题（如编造信息、完全偏题、内容空洞），标记为 is_satisfactory=false
- 列出具体问题和改进建议

## 输入格式

用户查询：...
参考文档：...
助手回复：...
"""


def _normalize_evaluation_suggestion(suggestion: str) -> str:
    normalized = str(suggestion or "").strip()
    if not normalized:
        return ""

    risky_markers = [
        "融入其中",
        "贴合用户提供的全部食材",
        "豆腐番茄炒蛋",
        "豆腐番茄汤",
        "未直接覆盖",
        "加入未覆盖食材",
        "把食材加入现有做法",
    ]
    structure_markers = [
        "结构",
        "层次",
        "完整",
        "空洞",
        "具体",
        "结论",
    ]
    evidence_markers = [
        "证据",
        "文档",
        "引用",
        "转述",
        "检索",
        "步骤",
        "时间",
        "数值",
        "营养",
    ]
    layering_markers = [
        "通用建议",
        "区分",
        "分层",
        "文档事实",
        "grounded",
    ]

    normalized_parts: list[str] = []
    if any(marker in normalized for marker in structure_markers):
        normalized_parts.append("请优先把回复结构写清楚，先给主结论，再补充证据内理由。")
    if any(marker in normalized for marker in evidence_markers):
        normalized_parts.append("如果参考文档里有明确候选、步骤、时间或数值，请更明确引用或转述；如果没有，不要补写。")
    if any(marker in normalized for marker in layering_markers) or any(marker in normalized for marker in risky_markers):
        normalized_parts.append("如需补充常识或延伸建议，请明确区分“证据支持”和“通用建议”，不要把未覆盖组合、候选或做法写成文档事实。")

    if not normalized_parts:
        normalized_parts.append("请优先把回复结构写清楚，并更明确区分“证据支持”和“通用建议”。")

    return " ".join(dict.fromkeys(normalized_parts))


def evaluator_node(state: DietAgentState) -> dict:
    """质量评估节点

    评估 Generator 生成回复的质量。chitchat 意图跳过评估。
    每次评估不通过时 retry_count += 1。

    Phase 1: 当 rag_strategy='adaptive' 时，额外执行 Self-RAG 幻觉检测和有用性判断，
    结果合并到 evaluation 字典，不删除已有评估逻辑。

    Args:
        state: LangGraph 全局状态

    Returns:
        包含 evaluation, retry_count, self_rag_judgements 的字典
    """
    intent = state.get("intent", "chitchat")
    retry_count = state.get("retry_count", 0)
    settings = get_settings()

    logger.info(f"Evaluator 节点开始执行, intent={intent}, retry_count={retry_count}")

    # chitchat 跳过评估
    if intent == "chitchat":
        logger.info("闲聊意图，跳过评估，直接通过")
        evaluation = {
            "is_satisfactory": True,
            "passed": True,
            "issues": [],
            "suggestion": ""
        }
        evaluation["quality_signals"] = build_skill_quality_signals(
            state.get("active_skill", ""),
            response=state.get("response", ""),
            response_type=state.get("response_type", "recommendation"),
            retrieval_stats=state.get("retrieval_stats", {}),
            evaluation=evaluation,
        )
        return {
            "evaluation": evaluation,
            "retry_count": retry_count
        }

    # 获取评估所需信息
    messages = state.get("messages", [])
    user_query = ""
    if messages:
        # 查找最后一条用户消息
        for msg in reversed(messages):
            if hasattr(msg, "type") and msg.type == "human":
                user_query = msg.content
                break
            elif not hasattr(msg, "type"):
                user_query = str(msg)
                break

    response = state.get("response", "")
    reranked_docs = state.get("reranked_docs", [])

    # 构建文档摘要（与 generator 保持一致，包含完整元数据）
    docs_summary = ""
    for i, doc in enumerate(reranked_docs[:5], 1):
        if doc.get("text"):
            docs_summary += f"{i}. {doc['text'][:200]}\n"
        else:
            parts = [doc.get("name", "未知")]
            for key in ("cuisine", "time", "difficulty", "calories", "tags",
                        "health_goals", "protein", "carbs", "fat"):
                val = doc.get(key)
                if val:
                    parts.append(f"{key}={val}")
            docs_summary += f"{i}. {', '.join(parts)}\n"

    if not docs_summary:
        docs_summary = "（无参考文档）"

    evaluation: dict = {
        "is_satisfactory": True,
        "issues": [],
        "suggestion": ""
    }
    new_retry_count = retry_count

    try:
        llm = get_graph_llm()
        structured_llm = llm.with_structured_output(EvaluationResult)

        eval_input = (
            f"用户查询：{user_query}\n\n"
            f"参考文档：\n{docs_summary}\n"
            f"助手回复：\n{response}"
        )

        result: EvaluationResult = structured_llm.invoke([
            SystemMessage(content=EVALUATOR_SYSTEM_PROMPT),
            HumanMessage(content=eval_input)
        ])

        evaluation = {
            "is_satisfactory": result.is_satisfactory,
            "issues": result.issues,
            "suggestion": _normalize_evaluation_suggestion(result.suggestion)
        }
        new_retry_count = retry_count if result.is_satisfactory else retry_count + 1

        logger.info(
            f"评估结果: satisfactory={result.is_satisfactory}, "
            f"issues={result.issues}, retry_count={new_retry_count}"
        )

    except Exception as e:
        logger.error(f"Evaluator 节点执行失败: {e}", exc_info=True)
        # 降级：评估失败时默认通过，避免无限循环
        evaluation = {
            "is_satisfactory": True,
            "issues": [f"评估失败: {str(e)}"],
            "suggestion": ""
        }

    # ── Phase 1: Self-RAG 幻觉检测 + 有用性判断（补充维度）────────────────────
    self_rag_judgements: dict = dict(state.get("self_rag_judgements") or {})

    if settings.rag_strategy == "adaptive" and response:
        try:
            from ...rag.self_rag import get_self_rag_judge
            judge = get_self_rag_judge()

            if getattr(settings, "self_rag_lite", False):
                lite = judge.judge_quality_lite(user_query, response, reranked_docs)
                self_rag_judgements["hallucination"] = {
                    "has_hallucination": lite.has_hallucination,
                    "hallucinated_claims": lite.hallucinated_claims,
                }
                self_rag_judgements["usefulness"] = {
                    "is_useful": lite.is_useful,
                    "missing_info": lite.missing_info,
                }

                if lite.has_hallucination:
                    evaluation["issues"].append(
                        f"幻觉检测: {lite.hallucinated_claims[:2]}"
                    )
                    evaluation["is_satisfactory"] = False
                    new_retry_count = retry_count + 1
                    logger.warning(
                        f"Self-RAG Lite 检测到幻觉: {lite.hallucinated_claims}"
                    )

                if not lite.is_useful:
                    evaluation["issues"].append(
                        f"有用性不足: 缺失 {lite.missing_info}"
                    )
                    if evaluation["is_satisfactory"]:
                        evaluation["is_satisfactory"] = False
                        new_retry_count = retry_count + 1
                    logger.warning(
                        f"Self-RAG Lite 回复有用性不足: missing={lite.missing_info}"
                    )
            else:
                hallucination = judge.judge_hallucination(response, reranked_docs)
                self_rag_judgements["hallucination"] = {
                    "has_hallucination": hallucination.has_hallucination,
                    "hallucinated_claims": hallucination.hallucinated_claims,
                }
                if hallucination.has_hallucination:
                    evaluation["issues"].append(
                        f"幻觉检测: {hallucination.hallucinated_claims[:2]}"
                    )
                    evaluation["is_satisfactory"] = False
                    new_retry_count = retry_count + 1
                    logger.warning(
                        f"Self-RAG 检测到幻觉: {hallucination.hallucinated_claims}"
                    )

                usefulness = judge.judge_usefulness(user_query, response)
                self_rag_judgements["usefulness"] = {
                    "is_useful": usefulness.is_useful,
                    "missing_info": usefulness.missing_info,
                }
                if not usefulness.is_useful:
                    evaluation["issues"].append(
                        f"有用性不足: 缺失 {usefulness.missing_info}"
                    )
                    if evaluation["is_satisfactory"]:
                        evaluation["is_satisfactory"] = False
                        new_retry_count = retry_count + 1
                    logger.warning(
                        f"Self-RAG 回复有用性不足: missing={usefulness.missing_info}"
                    )
        except Exception as e:
            logger.error(f"Self-RAG 补充判断失败: {e}", exc_info=True)

    evaluation["passed"] = bool(evaluation.get("is_satisfactory", True))
    evaluation["quality_signals"] = build_skill_quality_signals(
        state.get("active_skill", ""),
        response=response,
        response_type=state.get("response_type", "recommendation"),
        retrieval_stats=state.get("retrieval_stats", {}),
        evaluation=evaluation,
    )

    return {
        "evaluation": evaluation,
        "retry_count": new_retry_count,
        "self_rag_judgements": self_rag_judgements,
    }
