"""
生成回复节点

根据意图选择不同的生成策略：
- chitchat: 直接用 LLM 生成闲聊回复（无需检索上下文）
- recipe_search / nutrition_query / ingredient_check: 基于 reranked_docs 生成回复

复用模块:
- src/llm/qwen_client.py::get_llm()
- src/agent/prompts_v2.py — 复用已有系统提示词风格
- src/utils/logger.py::get_logger()
"""

import re
from pathlib import Path
from typing import Any

from diet_agent.runtime import get_skill_runtime_policy

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from ...tutorials.pipeline import build_tutorial_chunk_corpus, export_tutorial_to_pdf, save_tutorial_to_json
from ...tutorials.storage import default_bilibili_artifact_root, resolve_bilibili_tutorial_storage
from ...tutorials.windsurf_skill_bridge import (
    WindsurfSkillBridgeError,
    expected_windsurf_result_json_path,
    load_bilibili_tutorial_from_windsurf_result,
    render_bilibili_tutorial_pdf_in_python,
)
from ...vectorstore.tutorial_store import batch_add_tutorial_chunks, get_tutorial_collection_count
from ..state import DietAgentState, normalize_stable_user_preferences
from ..llm import get_graph_llm
from ...config import get_settings
from ...utils.logger import get_logger


logger = get_logger(__name__)
settings = get_settings()


def build_bilibili_tutorial_from_url(*args: Any, **kwargs: Any):
    from ...tutorials.bilibili_summary import build_bilibili_tutorial_from_url as _impl

    return _impl(*args, **kwargs)

GENERATOR_SYSTEM_PROMPT = """你是一个专业的智能饮食助手，帮助用户找到合适的食谱、分析营养成分、检查食材搭配。

## 回复原则

1. 优先基于提供的参考文档生成回复，并结合通用烹饪/营养常识提高可用性
2. 回复简洁、专业、友好
3. 推荐食谱时说明推荐理由（符合口味、时间合适、食材匹配等）
4. 每次推荐 2-3 道菜即可，不要过多
5. 适当提供烹饪建议或营养提示
6. 如果参考文档不足以回答问题，诚实说明证据不足，并提供尽可能有用的通用建议
7. 不要自创新菜名、新组合菜、套餐名、配菜方案或文档中不存在的变体做法
8. 当文档未直接覆盖某个食材或组合时，不要把它说成文档事实；可以给出明确标注为“通用建议”的补充方向

## 回复格式

- 使用 Markdown 格式
- 食谱推荐包含：菜名、烹饪时间、难度、特点
- 营养建议包含：摄入建议、推荐食物
"""

CHITCHAT_SYSTEM_PROMPT = """你是一个友好的饮食助手。用户在进行闲聊，请用简短、友好的方式回复。
如果用户打招呼，可以简单回应并引导到饮食相关话题。
保持回复简短，2-3句话即可。"""

MISSING_SLOT_TEXT = {
    "available_ingredients": "食材信息",
    "max_cooking_time": "时间偏好",
    "health_goal": "健康目标",
    "dietary_restrictions": "饮食限制",
}


def _format_docs_context(docs: list[dict]) -> str:
    """将检索文档格式化为上下文字符串
    
    Args:
        docs: 检索到的文档列表
    
    Returns:
        格式化的上下文字符串
    """
    if not docs:
        return "（无相关参考文档）"
    
    context_parts = []
    for i, doc in enumerate(docs, 1):
        name = doc.get("name", doc.get("text", "未知"))
        description = doc.get("description", "")
        cuisine = doc.get("cuisine", "")
        time_val = doc.get("time", "")
        difficulty = doc.get("difficulty", "")
        calories = doc.get("calories", "")
        tags = doc.get("tags", "")
        score_breakdown = doc.get("score_breakdown", {})
        
        part = f"【文档{i}】{name}"
        if description:
            part += f"\n描述：{description}"
        if cuisine:
            part += f"\n菜系：{cuisine}"
        if time_val:
            part += f"\n时间：{time_val}分钟"
        if difficulty:
            part += f"\n难度：{difficulty}"
        if calories:
            part += f"\n热量：{calories}卡"
        if tags:
            part += f"\n标签：{tags}"
        if score_breakdown:
            part += (
                "\n解释："
                f"检索分 {score_breakdown.get('retrieval_score', 0)}, "
                f"食材匹配 {score_breakdown.get('ingredient_match_score', 0)}, "
                f"目标匹配 {score_breakdown.get('goal_fit_score', 0)}, "
                f"时间贴合 {score_breakdown.get('time_fit_score', 0)}"
            )
            matched_ingredients = score_breakdown.get("matched_ingredients", [])
            coverage = score_breakdown.get("ingredient_coverage", 0)
            matched_inventory_count = score_breakdown.get("matched_inventory_count", len(matched_ingredients))
            required_ingredient_count = score_breakdown.get("required_ingredient_count", 0)
            if matched_ingredients:
                part += f"\n命中食材：{', '.join(matched_ingredients)}"
            if required_ingredient_count:
                part += (
                    f"\n库存覆盖率：{matched_inventory_count}/{required_ingredient_count} "
                    f"({round(float(coverage) * 100)}%)"
                )
            expiring_soon_ingredients = score_breakdown.get("expiring_soon_ingredients", [])
            if expiring_soon_ingredients:
                part += f"\n优先消耗临期食材：{', '.join(expiring_soon_ingredients)}"
            missing_ingredients = score_breakdown.get("missing_ingredients", [])
            if missing_ingredients:
                part += f"\n还缺食材：{', '.join(missing_ingredients)}"
        
        context_parts.append(part)
    
    return "\n\n".join(context_parts)


def _prepare_docs_for_generation(docs: list[dict]) -> list[dict]:
    prepared_docs = []
    for doc in docs:
        doc_copy = dict(doc)
        score_breakdown = doc_copy.get("score_breakdown", {})
        if score_breakdown:
            explanation = (
                f"检索分 {score_breakdown.get('retrieval_score', 0)}, "
                f"食材匹配 {score_breakdown.get('ingredient_match_score', 0)}, "
                f"目标匹配 {score_breakdown.get('goal_fit_score', 0)}, "
                f"时间贴合 {score_breakdown.get('time_fit_score', 0)}"
            )
            matched_ingredients = score_breakdown.get("matched_ingredients", [])
            coverage = score_breakdown.get("ingredient_coverage", 0)
            matched_inventory_count = score_breakdown.get("matched_inventory_count", len(matched_ingredients))
            required_ingredient_count = score_breakdown.get("required_ingredient_count", 0)
            if matched_ingredients:
                explanation += f"。命中食材：{', '.join(matched_ingredients)}"
            if required_ingredient_count:
                explanation += (
                    f"。现有食材可直接利用 {matched_inventory_count}/{required_ingredient_count} 个"
                    f"（覆盖率 {round(float(coverage) * 100)}%）"
                )
            expiring_soon_ingredients = score_breakdown.get("expiring_soon_ingredients", [])
            if expiring_soon_ingredients:
                explanation += f"。可优先消耗临期食材：{', '.join(expiring_soon_ingredients)}"
            missing_ingredients = score_breakdown.get("missing_ingredients", [])
            if missing_ingredients:
                explanation += f"。仍缺少：{', '.join(missing_ingredients)}"
 
            text = str(doc_copy.get("text", "")).strip()
            if explanation not in text:
                doc_copy["text"] = f"{text}\n推荐解释：{explanation}".strip()
        prepared_docs.append(doc_copy)
    return prepared_docs


def _build_clarification_response(state: DietAgentState) -> str:
    question = state.get("clarification_question", "").strip()
    missing_slots = state.get("missing_slots", []) or []
    missing_slot_text = [MISSING_SLOT_TEXT.get(slot, slot) for slot in missing_slots]

    response_lines = ["为了给你更可执行的推荐，我想先确认一个最关键的问题："]
    if question:
        response_lines.append("")
        response_lines.append(question)
    if missing_slot_text:
        response_lines.append("")
        response_lines.append(f"当前还缺少：{', '.join(missing_slot_text)}。")
    response_lines.append("你补充这一点后，我就可以继续给你正式推荐。")
    return "\n".join(response_lines)


def _build_fallback_response(state: DietAgentState) -> str:
    extracted_params = state.get("extracted_params", {}) or {}
    retrieval_stats = state.get("retrieval_stats", {}) or {}
    hard_filter_reasons = retrieval_stats.get("hard_filter_reasons", {}) or {}

    response_lines = ["我暂时没找到完全符合你当前条件的食谱，可以先这样调整："]

    suggestion_parts: list[str] = []
    max_cooking_time = extracted_params.get("max_cooking_time")
    if hard_filter_reasons.get("time_budget_exceeded") and max_cooking_time:
        relaxed_time = max(int(max_cooking_time) + 10, int(max_cooking_time * 1.2))
        suggestion_parts.append(f"把烹饪时间从 {max_cooking_time} 分钟适度放宽到 {relaxed_time} 分钟")
    if hard_filter_reasons.get("allergy_conflict"):
        suggestion_parts.append("保留过敏约束不变，换一个主食材方向继续找")
    if hard_filter_reasons.get("disliked_ingredient_conflict"):
        suggestion_parts.append("把不喜欢的食材保留为硬约束，我可以改推相近口味的替代菜")
    if not suggestion_parts:
        suggestion_parts.append("补充 1-2 个更具体的条件，例如现有食材、时间预算或健康目标")

    missing_ingredients = extracted_params.get("available_ingredients") or []
    if suggestion_parts:
        response_lines.append("")
        response_lines.append("- 放宽建议：" + "；".join(suggestion_parts))
    response_lines.append("- 缺失食材：当前没有可直接执行的候选，建议补充你现有食材后我再重试")
    response_lines.append("- 替代方案：如果你愿意，我可以优先按清淡/快手/减脂等方向给你一版近似推荐")
    if missing_ingredients:
        response_lines.append(f"- 你已提供的食材：{', '.join(str(item) for item in missing_ingredients)}")
    return "\n".join(response_lines)


def _build_skill_fallback_response(
    state: DietAgentState,
    intent: str,
    response_contract: dict | None = None,
    fallback_policy: dict | None = None,
) -> str:
    response_contract = response_contract or {}
    fallback_policy = fallback_policy or {}
    fallback_mode = str(fallback_policy.get("on_low_evidence") or "").strip()
    if fallback_mode == "general_advice_only" and intent == "nutrition_query":
        return _build_general_advice_only_response(state, response_contract)
    if fallback_mode == "outline_only":
        return _build_outline_only_response(state)
    return _build_fallback_response(state)


def _should_use_outline_only_subgraph_fallback(
    state: DietAgentState,
    response_contract: dict | None = None,
    fallback_policy: dict | None = None,
) -> bool:
    response_contract = response_contract or {}
    fallback_policy = fallback_policy or {}
    planner_next_action = str(state.get("planner_next_action") or "").strip()
    fallback_mode = str(fallback_policy.get("on_low_evidence") or "").strip()
    return planner_next_action == "subgraph_candidate" and (
        bool(response_contract.get("allow_subgraph"))
        or fallback_mode == "outline_only"
    )


def _should_fallback(state: DietAgentState, intent: str, reranked_docs: list[dict]) -> bool:
    if intent == "chitchat":
        return False
    if state.get("response_type") == "fallback":
        return True
    retrieval_stats = state.get("retrieval_stats", {}) or {}
    if retrieval_stats.get("fallback_triggered"):
        return True
    return intent in {"recipe_search", "nutrition_query", "ingredient_check"} and not reranked_docs


def _resolve_generation_docs(state: DietAgentState) -> list[dict]:
    docs = list(state.get("reranked_docs", []) or state.get("retrieved_docs", []) or [])
    if docs:
        return docs

    if str(state.get("planner_next_action") or "").strip() != "direct_followup":
        return []

    recent_recommended_recipes = state.get("recent_recommended_recipes", []) or []
    return [dict(item) for item in recent_recommended_recipes if isinstance(item, dict)]


def _looks_like_recipe_steps_query(query: str, active_skill: str = "") -> bool:
    if str(active_skill or "").strip() == "recipe_tutorial":
        return True
    lowered = str(query or "")
    return any(keyword in lowered for keyword in ["怎么做", "做法", "步骤", "分步骤", "制作"])


def _looks_like_nutrition_fact_query(query: str) -> bool:
    lowered = str(query or "")
    return any(
        keyword in lowered
        for keyword in ["热量", "蛋白质", "脂肪", "碳水", "膳食纤维", "纤维", "营养成分", "营养信息", "卡路里"]
    )


def _looks_like_nutrition_guidance_query(query: str, intent: str) -> bool:
    return intent == "nutrition_query" and not _looks_like_nutrition_fact_query(query)


def _safe_float_value(value) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        numeric_text = "".join(ch for ch in str(value) if ch.isdigit() or ch == ".")
        if not numeric_text:
            return None
        try:
            return float(numeric_text)
        except ValueError:
            return None


def _format_numeric_value(value) -> str:
    numeric_value = _safe_float_value(value)
    if numeric_value is None:
        return ""
    if numeric_value.is_integer():
        return str(int(numeric_value))
    return f"{numeric_value:.1f}".rstrip("0").rstrip(".")


def _truncate_text(value: str, limit: int = 220) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _safe_int_value(value) -> int | None:
    numeric_value = _safe_float_value(value)
    if numeric_value is None:
        return None
    return int(numeric_value)


def _get_generation_contracts(state: DietAgentState) -> tuple[str, dict, dict, dict]:
    runtime_policy = get_skill_runtime_policy(state.get("active_skill") or "")
    active_skill = runtime_policy.skill_name
    if runtime_policy.skill_spec is None:
        return active_skill, {}, {}, {}
    return (
        active_skill,
        dict(runtime_policy.response_contract),
        dict(runtime_policy.evidence_policy),
        dict(runtime_policy.fallback_policy),
    )


def _resolve_response_contract_max_recipe_count(response_contract: dict, default: int = 2) -> int:
    max_recipe_count = _safe_int_value(response_contract.get("max_recipe_count"))
    if max_recipe_count is None:
        return default
    return max(1, min(max_recipe_count, 5))


def _resolve_tutorial_topic_anchor(state: DietAgentState, user_query: str) -> str:
    extracted_params = state.get("extracted_params", {}) or {}
    retrieval_stats = state.get("retrieval_stats", {}) or {}
    explicit_anchor = str(extracted_params.get("tutorial_topic_anchor") or retrieval_stats.get("tutorial_topic_anchor") or "").strip()
    if explicit_anchor:
        return explicit_anchor
    return str(user_query or "").strip()


def _build_native_general_cooking_fallback_response(
    state: DietAgentState,
    user_query: str,
    response_contract: dict | None = None,
) -> str:
    response_contract = response_contract or {}
    topic_anchor = _resolve_tutorial_topic_anchor(state, user_query)
    require_evidence_boundary = bool(response_contract.get("require_evidence_boundary"))
    llm = get_graph_llm()
    response = llm.invoke([
        SystemMessage(
            content=(
                "你是一名烹饪助手。当当前知识库没有直接检索到某道菜的现成教程时，"
                "可以基于通用烹饪知识回答，但必须先明确说明当前知识库没有直接命中，"
                "再把后续内容标成“通用做法/经验建议”，不要伪装成文档事实。"
                "请使用 Markdown，优先输出 4-7 条稳妥、可执行的编号步骤；"
                "对拿不准的温度、时间或配比，用范围表达；"
                "最后补一个简短的“提醒”或“可继续细化”小节。"
            )
        ),
        HumanMessage(
            content=(
                f"原始用户问题：{user_query}\n"
                f"当前讨论菜名：{topic_anchor}\n"
                f"证据边界要求：{'必须明确说明当前知识库未直接覆盖' if require_evidence_boundary else '需要说明当前知识库未直接命中'}\n"
                "请围绕这个菜名给出一版通用做法，不要改成别的菜。"
            )
        ),
    ])
    return response.content


def _build_general_advice_only_response(state: DietAgentState, response_contract: dict | None = None) -> str:
    response_contract = response_contract or {}
    extracted_params = state.get("extracted_params", {}) or {}
    health_goal = _normalize_nutrition_goal(str(extracted_params.get("health_goal") or "").strip())
    meal_type = str(extracted_params.get("meal_type") or "").strip()
    require_evidence_boundary = bool(response_contract.get("require_evidence_boundary"))

    lines = [
        "**当前证据情况**",
        "- 这次检索结果里没有足够直接的参考文档，我先不给你伪装成“文档已明确支持”的结论。",
        "",
        "**通用建议**",
    ]

    if health_goal == "减脂":
        lines.append(f"- {meal_type or '这一餐'}优先高蛋白、少油，并把总热量控制在更轻的范围内。")
    elif health_goal == "控脂":
        lines.append(f"- {meal_type or '这一餐'}优先低油、少饱和脂肪，尽量避开明显偏油或高脂的做法。")
    elif health_goal == "高蛋白":
        lines.append(f"- {meal_type or '这一餐'}优先保证优质蛋白来源，再搭配适量主食和蔬菜。")
    elif health_goal == "养胃":
        lines.append(f"- {meal_type or '这一餐'}尽量温热、清淡、易消化，避免辛辣刺激和过油做法。")
    else:
        lines.append(f"- {meal_type or '这一餐'}先优先选择更清淡、负担更低、容易执行的搭配。")

    if meal_type == "早餐":
        lines.append("- 早餐可以优先鸡蛋、无糖酸奶、牛奶或豆制品，再搭配适量主食。")
    elif meal_type == "晚餐":
        lines.append("- 晚餐尽量少油、不过量，主食按饥饿程度适量保留即可。")

    lines.append("- 如果你愿意，我可以继续按更具体的食材、时间或目标，帮你缩小到更可执行的范围。")
    if require_evidence_boundary:
        lines.append("- 以上内容属于通用建议，不代表当前参考文档已直接覆盖。")
    return "\n".join(lines)


def _build_outline_only_response(state: DietAgentState) -> str:
    extracted_params = state.get("extracted_params", {}) or {}
    active_skill = str(state.get("active_skill") or "").strip()
    health_goal = str(extracted_params.get("health_goal") or "").strip()
    meal_type = str(extracted_params.get("meal_type") or "").strip()
    target_text = health_goal or "当前目标"
    meal_text = meal_type or "每日"
    if active_skill == "meal_planning" or str(state.get("planner_next_action") or "").strip() == "subgraph_candidate":
        scope_text = f"{meal_type}安排" if meal_type else "每日安排"
        lines = [
            "我先给你一个更稳妥的一周饮食安排提纲：",
            "",
            "**本周安排结构**",
            f"- 先围绕“{target_text}”确定 2-3 个 {scope_text} 方向。",
            "- 每个方向先保留 1-2 个容易轮换的候选，再逐步补成更细的菜单。",
            "- 当前先不展开成完整七天细化版本；如果你补充现有食材、时间预算和忌口，我可以继续把这个提纲收紧。",
        ]
        return "\n".join(lines)

    lines = [
        "我先给你一个更稳妥的饮食安排大纲：",
        "",
        "**建议结构**",
        f"- 先围绕“{target_text}”确定 2-3 个 {meal_text} 主体方向。",
        "- 每个方向先保持食材和做法简单，后续再补具体候选。",
        "- 如果你补充现有食材、时间预算和忌口，我可以把这个大纲进一步收紧成可执行版本。",
    ]
    return "\n".join(lines)


def _normalize_list_field(value) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value in (None, ""):
        return []
    text = str(value).strip()
    return [text] if text else []


def _build_inventory_followup_hint(state: DietAgentState, docs: list[dict]) -> str:
    extracted_params = state.get("extracted_params", {}) or {}
    available_ingredients = [
        str(item).strip()
        for item in extracted_params.get("available_ingredients") or []
        if str(item).strip()
    ]
    if not available_ingredients or not docs:
        return ""

    used_ingredients: list[str] = []
    used_set: set[str] = set()
    for doc in docs[:3]:
        score_breakdown = doc.get("score_breakdown", {}) or {}
        matched_items = score_breakdown.get("matched_inventory_items") or score_breakdown.get("matched_ingredients") or []
        for item in matched_items:
            normalized_item = str(item).strip()
            if normalized_item and normalized_item not in used_set:
                used_set.add(normalized_item)
                used_ingredients.append(normalized_item)

    if not used_ingredients:
        return ""

    remaining_ingredients = [item for item in available_ingredients if item not in used_set]
    if not remaining_ingredients:
        return ""

    remaining_display = "、".join(remaining_ingredients[:3])
    if len(remaining_ingredients) > 3:
        remaining_display = f"{remaining_display}等食材"

    consume_target = "它" if len(remaining_ingredients) == 1 else "这些食材"
    lines = ["", "**通用建议**"]
    lines.append(
        f"- 当前证据还没有直接覆盖{remaining_display}；如果你想优先消耗{consume_target}，我可以按这个方向再给你一版不重复的候选。"
    )
    return "\n".join(lines)


def _append_inventory_followup_hint(response_text: str, state: DietAgentState, docs: list[dict]) -> str:
    hint = _build_inventory_followup_hint(state, docs)
    if not hint:
        return response_text
    normalized_response = str(response_text).rstrip()
    if hint.strip() in normalized_response:
        return normalized_response
    if "**通用建议**" in normalized_response or "**下轮可继续**" in normalized_response:
        return normalized_response
    return f"{normalized_response}\n{hint}"


def _strip_step_trailing_metadata(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""

    metadata_markers = [
        "标签：",
        "标签:",
        "适合场景：",
        "适合场景:",
        "适合目标：",
        "适合目标:",
        "营养特点：",
        "营养特点:",
        "推荐解释：",
        "推荐解释:",
    ]
    for marker in metadata_markers:
        if marker in text:
            text = text.split(marker, 1)[0].strip()

    return text.rstrip("。；，, ")


def _normalize_step_text(value: str) -> str:
    text = _strip_step_trailing_metadata(value)
    if not text:
        return ""

    for prefix in ["步骤：", "步骤:", "做法：", "做法:", "制作：", "制作:"]:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()

    text = re.sub(r"^(第\s*\d+\s*步|(?<!第)\d+\s*[\.、步])\s*[：:、.\-]*\s*", "", text)

    return text.strip(" ：:-")


def _looks_like_real_step(step: str) -> bool:
    normalized = _normalize_step_text(step)
    if len(normalized) < 2:
        return False

    intro_keywords = ["经典家常菜", "酸甜可口", "营养丰富", "口感", "风味", "简介", "特点"]
    if any(keyword in normalized for keyword in intro_keywords):
        return False

    cooking_keywords = ["切", "洗", "打散", "炒", "翻炒", "热锅", "下锅", "加入", "倒入", "搅拌", "焯", "煮", "蒸", "炖", "腌", "装盘"]
    return any(keyword in normalized for keyword in cooking_keywords)


def _extract_numbered_step_chunks(text: str) -> list[str]:
    normalized_text = _strip_step_trailing_metadata(text)
    if not normalized_text:
        return []

    step_pattern = re.compile(r"(第\s*\d+\s*步|(?<!第)\d+\s*[\.、步])\s*[：:、.\-]*")
    matches = list(step_pattern.finditer(normalized_text))
    if len(matches) < 2:
        return []

    chunks: list[str] = []
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(normalized_text)
        chunk = normalized_text[start:end].strip()
        normalized_chunk = _normalize_step_text(chunk)
        if normalized_chunk and _looks_like_real_step(normalized_chunk):
            chunks.append(normalized_chunk)
    return chunks


def _extract_steps_from_doc(doc: dict) -> list[str]:
    explicit_steps = [
        _normalize_step_text(step)
        for step in _normalize_list_field(doc.get("steps"))
    ]
    explicit_steps = [
        step
        for step in explicit_steps
        if step and not any(keyword in step for keyword in ["经典家常菜", "酸甜可口", "营养丰富", "简介", "特点"])
    ]
    if explicit_steps:
        return explicit_steps[:6]

    text = str(doc.get("text") or doc.get("description") or "").strip()
    if not text:
        return []

    numbered_chunks = _extract_numbered_step_chunks(text)
    if numbered_chunks:
        return numbered_chunks[:6]

    candidates: list[str] = []
    seen: set[str] = set()
    segments = text.replace("\n", "。").split("。")
    cooking_keywords = ["切", "炒", "煎", "煮", "焯", "拌", "腌", "蒸", "炖", "下锅", "翻炒", "热锅", "加入"]

    for raw_segment in segments:
        segment = _normalize_step_text(raw_segment)
        if not segment:
            continue
        if segment in seen:
            continue
        if _looks_like_real_step(segment) or any(keyword in segment for keyword in cooking_keywords):
            candidates.append(segment)
            seen.add(segment)
        if len(candidates) >= 6:
            break

    return candidates


def _build_structured_steps_response(docs: list[dict]) -> str:
    if not docs:
        return ""

    primary_doc = docs[0]
    steps = _extract_steps_from_doc(primary_doc)
    if len(steps) < 2:
        return ""

    title = str(primary_doc.get("name") or primary_doc.get("id") or "这道菜")
    lines = [f"**{title}做法**", ""]

    meta_parts = []
    if primary_doc.get("time") not in (None, ""):
        meta_parts.append(f"烹饪时间：约 {primary_doc.get('time')} 分钟")
    if primary_doc.get("difficulty"):
        meta_parts.append(f"难度：{primary_doc.get('difficulty')}")
    if meta_parts:
        lines.append("- " + " | ".join(meta_parts))
        lines.append("")

    lines.append("**步骤**")
    for index, step in enumerate(steps[:6], 1):
        lines.append(f"{index}. {step}")

    tip_source = str(primary_doc.get("description") or "").strip()
    if tip_source:
        lines.extend(["", f"**补充说明**：{_truncate_text(tip_source, 80)}"])

    lines.extend(["", "以上内容已按参考文档中的步骤整理。"])
    return "\n".join(lines)


def _build_structured_nutrition_response(query: str, docs: list[dict]) -> str:
    if not docs:
        return ""

    primary_doc = docs[0]
    field_specs = [
        ("calories", "热量", "千卡"),
        ("protein", "蛋白质", "克"),
        ("carbs", "碳水化合物", "克"),
        ("fat", "脂肪", "克"),
        ("fiber", "膳食纤维", "克"),
    ]
    nutrient_lines: list[str] = []
    asked_field_names: list[str] = []

    lowered_query = str(query or "")
    for field_name, label, _ in field_specs:
        if label in lowered_query:
            asked_field_names.append(field_name)
        elif field_name == "calories" and ("卡路里" in lowered_query or "卡" in lowered_query):
            asked_field_names.append(field_name)

    if not asked_field_names:
        asked_field_names = ["calories", "protein"] if any(keyword in lowered_query for keyword in ["营养", "营养信息", "营养成分"]) else [field_name for field_name, _, _ in field_specs]

    for field_name, label, unit in field_specs:
        if field_name not in asked_field_names:
            continue
        value = primary_doc.get(field_name)
        if value in (None, "", 0, 0.0):
            continue
        nutrient_lines.append(f"- {label}：{value} {unit}")

    if not nutrient_lines:
        return ""

    title = str(primary_doc.get("name") or primary_doc.get("id") or "该食物")
    lines = [f"**{title}营养信息**", "", "**参考文档中的明确数值**"]
    lines.extend(nutrient_lines)

    text = str(primary_doc.get("text") or primary_doc.get("description") or "").strip()
    if text:
        nutrition_segments = []
        for raw_segment in text.split("。"):
            segment = str(raw_segment).strip()
            if not segment:
                continue
            if any(keyword in segment for keyword in ["热量", "蛋白质", "脂肪", "碳水", "纤维", "营养"]):
                nutrition_segments.append(segment)
            if len(nutrition_segments) >= 2:
                break
        if nutrition_segments:
            lines.extend(["", "**补充说明**"])
            lines.extend(f"- {segment}" for segment in nutrition_segments)

    asked_labels = [label for field_name, label, _ in field_specs if field_name in asked_field_names]
    missing_labels = [label for label in asked_labels if not any(line.startswith(f"- {label}：") for line in nutrient_lines)]
    if missing_labels:
        lines.extend(["", f"未在参考文档中找到这些字段的明确数值：{', '.join(missing_labels)}。"])

    lines.extend(["", "以上数值直接来自参考文档，未做额外估算。"])
    return "\n".join(lines)


def _build_doc_search_blob(doc: dict) -> str:
    blob_parts = [
        str(doc.get("name") or ""),
        str(doc.get("description") or ""),
        str(doc.get("text") or ""),
        " ".join(_normalize_list_field(doc.get("tags"))),
        " ".join(_normalize_list_field(doc.get("health_goals"))),
        " ".join(_normalize_list_field(doc.get("health_goal_tags"))),
    ]
    return " ".join(part for part in blob_parts if part).lower()


def _normalize_nutrition_goal(health_goal: str) -> str:
    normalized_goal = str(health_goal or "").strip().lower()
    if not normalized_goal:
        return ""

    goal_aliases = (
        ("降血脂", "控脂"),
        ("血脂高", "控脂"),
        ("控脂", "控脂"),
        ("低脂", "控脂"),
        ("减肥", "减脂"),
        ("瘦身", "减脂"),
        ("减脂", "减脂"),
        ("高蛋白", "高蛋白"),
        ("控糖", "控糖"),
        ("养胃", "养胃"),
        ("清淡", "清淡口味"),
        ("低油", "低油"),
        ("少油", "低油"),
    )
    for alias, canonical in goal_aliases:
        if alias in normalized_goal:
            return canonical
    return normalized_goal


def _score_nutrition_guidance_penalty(blob: str, calories: float | None, health_goal: str) -> float:
    normalized_goal = _normalize_nutrition_goal(health_goal)
    if not normalized_goal:
        return 0.0

    penalty = 0.0
    negative_markers = {
        "减脂": ("红烧肉", "五花肉", "油炸", "肥而不腻", "干锅", "烧烤"),
        "控脂": ("红烧肉", "五花肉", "油炸", "肥而不腻", "高脂", "干锅", "烧烤"),
        "控糖": ("糖醋", "甜品", "蛋糕", "奶茶", "冰糖", "高糖"),
        "养胃": ("麻辣", "辛辣", "油炸", "干锅", "烧烤"),
        "清淡口味": ("麻辣", "重口", "油炸", "干锅", "红烧"),
        "低油": ("油炸", "干锅", "红烧", "肥肉", "五花肉"),
    }
    if any(marker in blob for marker in negative_markers.get(normalized_goal, ())):
        penalty += 4.0

    if calories is not None:
        if normalized_goal in {"减脂", "控脂", "清淡口味", "低油"}:
            if calories >= 600:
                penalty += 4.0
            elif calories >= 450:
                penalty += 2.5
        elif normalized_goal in {"控糖", "养胃"} and calories >= 500:
            penalty += 2.0

    return penalty


def _doc_mentions_ingredient(doc: dict, ingredient: str) -> bool:
    normalized_ingredient = str(ingredient or "").strip().lower()
    if not normalized_ingredient:
        return False

    candidate_fields = [
        str(doc.get("name") or ""),
        str(doc.get("description") or ""),
        str(doc.get("text") or ""),
        " ".join(_normalize_list_field(doc.get("tags"))),
        " ".join(_normalize_list_field(doc.get("health_goals"))),
        " ".join(_normalize_list_field(doc.get("health_goal_tags"))),
    ]
    blob = " ".join(part for part in candidate_fields if part).lower()
    return normalized_ingredient in blob


def _select_direct_ingredient_evidence_docs(docs: list[dict], ingredients: list[str]) -> list[dict]:
    normalized_ingredients = [str(item).strip() for item in ingredients if str(item).strip()]
    if not normalized_ingredients:
        return []
    direct_docs: list[dict] = []
    for doc in docs:
        if all(_doc_mentions_ingredient(doc, ingredient) for ingredient in normalized_ingredients):
            direct_docs.append(doc)
    return direct_docs


def _build_conservative_ingredient_check_response(state: DietAgentState, docs: list[dict]) -> str:
    extracted_params = state.get("extracted_params", {}) or {}
    ingredients = [
        str(item).strip()
        for item in extracted_params.get("available_ingredients") or []
        if str(item).strip()
    ]
    if len(ingredients) < 2:
        return ""

    direct_docs = _select_direct_ingredient_evidence_docs(docs, ingredients)
    if direct_docs:
        return ""

    ingredient_display = "和".join(ingredients[:2]) if len(ingredients) == 2 else "、".join(ingredients)
    meal_type = str(extracted_params.get("meal_type") or "").strip()
    health_goal = str(extracted_params.get("health_goal") or "").strip()
    nearby_candidates = [
        str(doc.get("name") or doc.get("id") or "").strip()
        for doc in docs[:2]
        if str(doc.get("name") or doc.get("id") or "").strip()
    ]

    lines = ["**当前证据情况**"]
    lines.append(f"- 现有参考文档里没有直接出现“{ingredient_display}”这个组合，我不能把它说成文档已明确验证的搭配。")
    if nearby_candidates:
        lines.append(f"- 当前更接近的参考候选是：{'、'.join(nearby_candidates)}，但它们不足以直接证明这个组合已被文档覆盖。")

    lines.extend(["", "**更稳妥的回答**"])
    scope_text = f"作为{meal_type}的一部分" if meal_type else "一起搭配"
    lines.append(f"- 如果你只是想判断能不能{scope_text}：从**通用建议**角度看，{ingredient_display}通常可以一起安排，注意按个人耐受情况适量食用。")
    if meal_type == "早餐":
        lines.append("- 作为**通用建议**，这类早餐搭配通常可以补充蛋白质，但这不是当前参考文档中的直接结论。")
    if health_goal:
        lines.append(f"- 如果你更看重“{health_goal}”这个目标，我可以继续按这个方向帮你找更直接覆盖该组合的候选。")
    else:
        lines.append("- 如果你愿意，我也可以继续帮你找文档里更直接覆盖这组食材的菜或做法。")
    return "\n".join(lines)


def _score_nutrition_guidance_doc(doc: dict, health_goal: str, meal_type: str) -> float:
    blob = _build_doc_search_blob(doc)
    normalized_goal = _normalize_nutrition_goal(health_goal)
    score = 0.0
    calories = _safe_float_value(doc.get("calories"))
    protein = _safe_float_value(doc.get("protein"))

    if normalized_goal == "减脂":
        if any(keyword in blob for keyword in ["减脂", "减肥", "低脂", "低卡", "低热量", "清淡", "鸡胸肉", "沙拉"]):
            score += 3.0
        if calories is not None and calories <= 350:
            score += 2.0
        if protein is not None and protein >= 15:
            score += 1.0
    elif normalized_goal == "控脂":
        if any(keyword in blob for keyword in ["控脂", "低脂", "低油", "清淡", "蒸", "鱼", "时蔬"]):
            score += 3.0
        if calories is not None and calories <= 350:
            score += 2.0
        if protein is not None and protein >= 12:
            score += 1.0
    elif normalized_goal == "高蛋白":
        if any(keyword in blob for keyword in ["高蛋白", "鸡胸肉", "鸡蛋", "蛋羹", "瘦肉", "鱼", "牛肉"]):
            score += 2.0
        if protein is not None and protein >= 15:
            score += 2.0
    elif normalized_goal == "清淡口味":
        if any(keyword in blob for keyword in ["清淡", "蒸", "粥", "汤", "沙拉"]):
            score += 2.0
        if calories is not None and calories <= 350:
            score += 1.0
    elif normalized_goal == "低油":
        if any(keyword in blob for keyword in ["低油", "少油", "清淡", "蒸", "沙拉"]):
            score += 2.0
        if calories is not None and calories <= 350:
            score += 1.0
    elif normalized_goal == "控糖":
        if any(keyword in blob for keyword in ["控糖", "低糖"]):
            score += 2.0
    elif normalized_goal == "养胃":
        if any(keyword in blob for keyword in ["养胃", "清淡", "粥", "汤", "蒸", "易消化"]):
            score += 2.0

    if meal_type == "早餐":
        if any(keyword in blob for keyword in ["早餐", "蛋羹", "粥", "面", "鸡蛋"]):
            score += 1.5
    elif meal_type == "晚餐":
        if any(keyword in blob for keyword in ["沙拉", "蒸", "鱼", "汤", "时蔬"]):
            score += 1.0

    score -= _score_nutrition_guidance_penalty(blob, calories, normalized_goal)
    return score


def _select_nutrition_guidance_docs(docs: list[dict], health_goal: str, meal_type: str) -> list[dict]:
    if not docs:
        return []
    scored_docs = [
        (_score_nutrition_guidance_doc(doc, health_goal, meal_type), index, doc)
        for index, doc in enumerate(docs)
    ]
    ranked_docs = sorted(scored_docs, key=lambda item: (-item[0], item[1]))
    positive_docs = [doc for score, _, doc in ranked_docs if score > 0]
    if positive_docs:
        return positive_docs[:3]
    return docs[:3]


def _build_structured_nutrition_guidance_response(query: str, state: DietAgentState, docs: list[dict]) -> str:
    if not docs:
        return ""

    extracted_params = state.get("extracted_params", {}) or {}
    health_goal = _normalize_nutrition_goal(str(extracted_params.get("health_goal") or "").strip())
    meal_type = str(extracted_params.get("meal_type") or "").strip()

    title_prefix = f"{health_goal}{meal_type}".strip()
    title = f"{title_prefix}建议" if title_prefix else "营养建议"
    lines = [f"**{title}**", "", "**通用建议**"]

    guidance_lines: list[str] = []
    if health_goal == "减脂":
        guidance_lines.extend([
            f"{meal_type or '这一餐'}优先高蛋白、低油、控制总热量。",
            "主食按饥饿程度适量保留，蔬菜占比可以更高。",
        ])
    elif health_goal == "控脂":
        guidance_lines.extend([
            f"{meal_type or '这一餐'}优先低油、少饱和脂肪，并把总热量控制在更轻的范围内。",
            "可以优先选择鱼类、蛋类、豆制品和时蔬，尽量避开明显偏油或高脂的候选。",
        ])
    elif health_goal == "高蛋白":
        guidance_lines.extend([
            f"{meal_type or '这一餐'}优先保证优质蛋白来源，再搭配适量主食和蔬菜。",
            "如果想更容易坚持，可以优先选择准备成本低、饱腹感更稳定的组合。",
        ])
    elif health_goal == "清淡口味":
        guidance_lines.extend([
            f"{meal_type or '这一餐'}尽量少油、少重口调味，优先蒸、煮、炖或凉拌。",
            "如果需要更稳妥，可以先从鱼类、蛋类和清爽蔬菜类候选里选。",
        ])
    elif health_goal == "低油":
        guidance_lines.extend([
            f"{meal_type or '这一餐'}尽量少油、少重口调味，优先蒸、煮、炖或凉拌。",
            "如果需要更稳妥，可以先从鱼类、蛋类和清爽蔬菜类候选里选。",
        ])
    elif health_goal == "养胃":
        guidance_lines.extend([
            f"{meal_type or '这一餐'}尽量温热、清淡、易消化，避免辛辣刺激和过油做法。",
            "如果想更稳妥，可以优先选粥、汤、蒸菜或软烂一些的搭配。",
        ])
    else:
        guidance_lines.append("优先围绕你的当前目标，选择更容易执行、负担更低的搭配。")

    if meal_type == "早餐":
        guidance_lines.append("早餐可以优先考虑鸡蛋、蛋羹、粥或其他高蛋白搭配；如果不局限于当前文档，通用上也可搭配牛奶或无糖酸奶。")
    elif meal_type == "晚餐":
        guidance_lines.append("晚餐尽量避免过油、过量精制主食和夜宵式吃法。")

    for item in guidance_lines[:3]:
        lines.append(f"- {item}")

    selected_docs = _select_nutrition_guidance_docs(docs, health_goal, meal_type)
    lines.extend(["", "**结合当前参考文档，可优先考虑**"])
    for doc in selected_docs:
        name = str(doc.get("name") or doc.get("id") or "参考候选")
        reason_parts: list[str] = []
        calories = _format_numeric_value(doc.get("calories"))
        protein = _format_numeric_value(doc.get("protein"))
        time_value = _format_numeric_value(doc.get("time"))
        if calories:
            reason_parts.append(f"热量约 {calories} 卡")
        if protein:
            reason_parts.append(f"蛋白质约 {protein} 克")
        if time_value:
            reason_parts.append(f"耗时约 {time_value} 分钟")
        description = _truncate_text(doc.get("description") or doc.get("text") or "", 60)
        if description:
            reason_parts.append(description)
        if not reason_parts:
            reason_parts.append("可作为当前目标下的参考选项")
        lines.append(f"- **{name}**：{'；'.join(reason_parts)}")

    lines.extend([
        "",
        "以上菜名、时间与明确数值来自参考文档；“通用建议”部分用于帮助你把这些候选落到当前目标里。",
    ])
    return "\n".join(lines)


def _get_followup_anchor_names(state: DietAgentState) -> list[str]:
    anchor_names = [
        str(name).strip()
        for name in state.get("followup_anchor_names", []) or []
        if str(name).strip()
    ]
    if anchor_names:
        return anchor_names

    recent_recommended_recipes = state.get("recent_recommended_recipes", []) or []
    resolved_names: list[str] = []
    seen_names: set[str] = set()
    for item in recent_recommended_recipes[:3]:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name or name in seen_names:
            continue
        seen_names.add(name)
        resolved_names.append(name)
    return resolved_names


def _get_doc_name(doc: dict) -> str:
    return str(doc.get("name") or doc.get("id") or "").strip()


def _select_followup_candidate_docs(docs: list[dict], anchor_names: list[str], limit: int = 2) -> list[dict]:
    selected_docs: list[dict] = []
    seen_names: set[str] = set()
    anchor_name_set = set(anchor_names)

    for doc in docs:
        name = _get_doc_name(doc)
        if not name or name in seen_names or name in anchor_name_set:
            continue
        seen_names.add(name)
        selected_docs.append(doc)
        if len(selected_docs) >= limit:
            return selected_docs

    if selected_docs:
        return selected_docs

    for doc in docs:
        name = _get_doc_name(doc)
        if not name or name in seen_names:
            continue
        seen_names.add(name)
        selected_docs.append(doc)
        if len(selected_docs) >= 1:
            break

    return selected_docs


def _build_followup_recommendation_reason(doc: dict, state: DietAgentState) -> str:
    extracted_params = state.get("extracted_params", {}) or {}
    score_breakdown = doc.get("score_breakdown", {}) or {}
    reason_parts: list[str] = []

    time_value = doc.get("time")
    if time_value not in (None, ""):
        reason_parts.append(f"约 {time_value} 分钟可完成")

    matched_ingredients = [
        str(item).strip()
        for item in score_breakdown.get("matched_inventory_items") or score_breakdown.get("matched_ingredients") or []
        if str(item).strip()
    ]
    if matched_ingredients:
        reason_parts.append(f"能直接利用{'、'.join(matched_ingredients[:3])}")

    health_goal = str(extracted_params.get("health_goal") or "").strip()
    try:
        goal_fit_score = float(score_breakdown.get("goal_fit_score", 0) or 0)
    except (TypeError, ValueError):
        goal_fit_score = 0.0
    if health_goal and goal_fit_score > 0:
        reason_parts.append(f"更贴近你这次的{health_goal}目标")

    difficulty = str(doc.get("difficulty") or "").strip()
    if difficulty:
        reason_parts.append(f"难度{difficulty}")

    if not reason_parts:
        tags = _normalize_list_field(doc.get("tags"))
        if tags:
            reason_parts.append(f"方向上更接近{tags[0]}")

    if not reason_parts:
        description = str(doc.get("description") or "").strip()
        if description:
            reason_parts.append(_truncate_text(description, 28))

    if not reason_parts:
        reason_parts.append("和你上一轮偏好方向更接近")

    return "，".join(reason_parts[:2]) + "。"


def _build_followup_general_advice(state: DietAgentState, docs: list[dict]) -> str:
    extracted_params = state.get("extracted_params", {}) or {}
    available_ingredients = [
        str(item).strip()
        for item in extracted_params.get("available_ingredients") or []
        if str(item).strip()
    ]
    if not available_ingredients:
        return ""

    used_ingredients: set[str] = set()
    for doc in docs[:2]:
        score_breakdown = doc.get("score_breakdown", {}) or {}
        for item in score_breakdown.get("matched_inventory_items") or score_breakdown.get("matched_ingredients") or []:
            normalized_item = str(item).strip()
            if normalized_item:
                used_ingredients.add(normalized_item)

    remaining_ingredients = [item for item in available_ingredients if item not in used_ingredients]
    if not remaining_ingredients:
        return ""

    return f"如果你这轮更想优先消耗{'、'.join(remaining_ingredients)}，我可以按这些食材改推一版不重复的候选。"


def _build_followup_next_turn_hint(state: DietAgentState, docs: list[dict]) -> str:
    extracted_params = state.get("extracted_params", {}) or {}
    available_ingredients = [
        str(item).strip()
        for item in extracted_params.get("available_ingredients") or []
        if str(item).strip()
    ]
    used_ingredients: set[str] = set()
    for doc in docs[:2]:
        score_breakdown = doc.get("score_breakdown", {}) or {}
        for item in score_breakdown.get("matched_inventory_items") or score_breakdown.get("matched_ingredients") or []:
            normalized_item = str(item).strip()
            if normalized_item:
                used_ingredients.add(normalized_item)

    remaining_ingredients = [item for item in available_ingredients if item not in used_ingredients]
    if remaining_ingredients:
        return f"如果你想继续，我可以按{'、'.join(remaining_ingredients)}优先消耗，再给你一版不重复的候选。"

    anchor_names = _get_followup_anchor_names(state)
    health_goal = str(extracted_params.get("health_goal") or "").strip()
    if anchor_names and health_goal:
        return f"如果你想继续，我可以沿着{'、'.join(anchor_names)}这条方向，再缩小到更适合{health_goal}的一版。"
    if anchor_names:
        return f"如果你想继续，我可以沿着{'、'.join(anchor_names)}这条方向，再分成更快手、更清淡或更下饭的一版。"
    if health_goal:
        return f"如果你想继续，我可以按更适合{health_goal}的方向，再缩小一版候选。"
    return "如果你想继续，我可以按更快手、更清淡或更高蛋白的方向再缩小一版。"


def _build_followup_carryover_line(state: DietAgentState, docs: list[dict]) -> str:
    anchor_names = _get_followup_anchor_names(state)
    candidate_names = [name for name in (_get_doc_name(doc) for doc in docs) if name]
    feedback_summary = _truncate_text(
        str((state.get("recent_feedback_signals", {}) or {}).get("summary") or "").strip(),
        24,
    )

    if anchor_names and candidate_names:
        return f"这次我沿着上一轮的{'、'.join(anchor_names)}方向，给你补一版不完全重复的候选：{'、'.join(candidate_names)}。"
    if anchor_names:
        return f"这次我继续沿着上一轮的{'、'.join(anchor_names)}方向，先给你补一版相近候选。"
    if feedback_summary:
        return f"这次我继续沿着你刚才认可的方向，补一版更贴近“{feedback_summary}”的候选。"
    return "这次我继续沿着刚才那轮推荐方向，给你补一版相近但不完全重复的候选。"


def _build_structured_followup_response(state: DietAgentState, docs: list[dict], response_contract: dict | None = None) -> str:
    followup_mode = str(state.get("followup_mode") or "").strip()
    if followup_mode not in {"anchored_followup", "generic_followup"} or not docs:
        return ""

    anchor_names = _get_followup_anchor_names(state)
    selected_docs = _select_followup_candidate_docs(docs, anchor_names, limit=_resolve_response_contract_max_recipe_count(response_contract or {}, default=2))
    if not selected_docs:
        return ""

    lines = [
        "**承接方向**",
        f"- {_build_followup_carryover_line(state, selected_docs)}",
        "",
        "**这轮推荐**",
    ]

    for doc in selected_docs:
        name = _get_doc_name(doc) or "这道菜"
        reason = _build_followup_recommendation_reason(doc, state)
        lines.append(f"- **{name}**：{reason}")

    general_advice = _build_followup_general_advice(state, selected_docs)
    if general_advice:
        lines.extend(["", "**通用建议**", f"- {general_advice}"])

    next_turn_hint = _build_followup_next_turn_hint(state, selected_docs)
    if next_turn_hint:
        lines.extend(["", "**下轮可继续**", f"- {next_turn_hint}"])

    return "\n".join(lines)


def _build_grounding_evidence(query: str, intent: str, docs: list[dict], active_skill: str = "") -> str:
    evidence_lines: list[str] = []
    use_steps = _looks_like_recipe_steps_query(query, active_skill)
    use_nutrition = intent == "nutrition_query" and _looks_like_nutrition_fact_query(query)

    for index, doc in enumerate(docs[:3], 1):
        title = str(doc.get("name") or doc.get("id") or f"文档{index}")
        doc_lines = [f"[证据{index}] {title}"]

        if use_steps:
            steps = doc.get("steps") or []
            if isinstance(steps, list) and steps:
                doc_lines.append("步骤：" + "；".join(str(step) for step in steps[:4]))
            text = str(doc.get("text") or "")
            if text:
                step_snippets = []
                for segment in text.split("。"):
                    segment = segment.strip()
                    if not segment:
                        continue
                    if "步骤" in segment or "第" in segment or any(k in segment for k in ["切", "炒", "煎", "煮", "焯", "翻炒"]):
                        step_snippets.append(segment)
                    if len(step_snippets) >= 3:
                        break
                if step_snippets:
                    doc_lines.append("做法片段：" + "；".join(step_snippets))

        if use_nutrition:
            nutrient_parts = []
            for field_name, label in [
                ("calories", "热量"),
                ("protein", "蛋白质"),
                ("carbs", "碳水"),
                ("fat", "脂肪"),
                ("fiber", "纤维"),
            ]:
                value = doc.get(field_name)
                if value not in (None, "", 0, 0.0):
                    nutrient_parts.append(f"{label}={value}")
            if nutrient_parts:
                doc_lines.append("营养字段：" + "，".join(nutrient_parts))
            text = str(doc.get("text") or "")
            if text:
                nutrition_snippets = []
                for segment in text.split("。"):
                    segment = segment.strip()
                    if not segment:
                        continue
                    if any(k in segment for k in ["热量", "蛋白质", "脂肪", "碳水", "卡"]):
                        nutrition_snippets.append(segment)
                    if len(nutrition_snippets) >= 2:
                        break
                if nutrition_snippets:
                    doc_lines.append("营养片段：" + "；".join(nutrition_snippets))

        if len(doc_lines) == 1:
            fallback_text = doc.get("text") or doc.get("description") or ""
            if fallback_text:
                doc_lines.append("摘要：" + _truncate_text(fallback_text))

        if len(doc_lines) > 1:
            evidence_lines.append("\n".join(doc_lines))

    if not evidence_lines:
        return ""
    return "\n\n".join(evidence_lines)


def _build_retry_guidance(suggestion: str, issues: list[str]) -> list[str]:
    suggestion_text = str(suggestion or "").strip()
    issue_text = " ".join(
        str(item).strip()
        for item in issues or []
        if str(item).strip()
    )
    source_text = f"{suggestion_text} {issue_text}".strip()
    if not source_text:
        return []

    guidance_lines: list[str] = []
    if any(marker in source_text for marker in ["结构", "层次", "完整", "空洞", "具体", "结论"]):
        guidance_lines.append("先给主结论，再补充证据内理由，保持主回答结构清晰。")
    if any(marker in source_text for marker in ["证据", "文档", "引用", "转述", "检索", "步骤", "时间", "数值", "营养"]):
        guidance_lines.append("如果证据里有明确候选、步骤、时间或数值，请更明确引用或转述这些信息。")
    if any(marker in source_text for marker in ["通用建议", "区分", "分层", "文档事实", "grounded", "未覆盖"]):
        guidance_lines.append("如需补充常识或延伸建议，请明确标注为“通用建议”，不要和证据事实混写。")

    if not guidance_lines:
        guidance_lines.append("把回答结构写清楚，并优先补强证据内信息。")

    guidance_lines.append("不要新增未覆盖组合、候选、菜名、步骤或做法。")
    return list(dict.fromkeys(guidance_lines))


def _build_grounded_query(
    user_query: str,
    intent: str,
    docs: list[dict],
    state: DietAgentState, 
    suggestion: str = "", 
    response_contract: dict | None = None, 
    evidence_policy: dict | None = None) -> str:
    response_contract = response_contract or {}
    evidence_policy = evidence_policy or {}
    require_evidence_boundary = bool(response_contract.get("require_evidence_boundary"))
    separate_evidence_from_general_advice = bool(evidence_policy.get("separate_evidence_from_general_advice"))
    allow_general_advice_when_insufficient = evidence_policy.get("allow_general_advice_when_insufficient")
    active_skill = str(state.get("active_skill") or "").strip()
    evidence = _build_grounding_evidence(user_query, intent, docs, active_skill)
    lines = [user_query]
    is_recipe_steps_query = _looks_like_recipe_steps_query(user_query, active_skill)
    is_recipe_recommendation = intent == "recipe_search" and not is_recipe_steps_query

    if evidence:
        lines.extend([
            "",
            "请优先参考以下证据回答，并遵守：",
            "- 如果证据里有明确步骤/数值，优先直接引用或转述这些内容。",
            "- 不要用常识估算去覆盖证据中的明确字段。",
            "- 如果你补充通用常识、经验做法或替代建议，必须明确标注为“通用建议/经验做法”，不要伪装成文档事实。",
            "- 如果证据中出现‘还缺食材/仍缺少’，这只表示候选覆盖率或制作门槛，不代表你可以把未出现的组合写成文档里已有的变体做法。",
            "",
            evidence,
        ])
        if require_evidence_boundary or separate_evidence_from_general_advice or allow_general_advice_when_insufficient is False:
            lines.extend([
                "",
                "当前 skill 对证据边界还有额外要求：",
            ])
            if require_evidence_boundary:
                lines.append("- 如果证据没有直接覆盖用户问题，先明确说明“当前证据未直接覆盖”，不要把推断写成文档结论。")
            if separate_evidence_from_general_advice:
                lines.append("- 将“证据结论”和“通用建议”分成独立小节，不要混写。")
            if allow_general_advice_when_insufficient is False:
                lines.append("- 如果证据不足，不要补充通用建议，只说明证据边界和下一步所需信息。")

        if is_recipe_recommendation:
            lines.extend([
                "",
                "这是一条“食谱推荐”类问题，请额外遵守：",
                "- 只能推荐证据中真实出现过名称的菜，不要新造菜名。",
                "- 不要生成‘A配B’、‘A加B汤’、‘套餐/拼盘/组合菜’这类证据中没有的组合推荐。",
                "- 如果某个食材没有被当前候选直接覆盖，可以先说明‘文档未直接覆盖’，再补充明确标注的通用建议，但不要把通用建议说成证据事实。",
                "- 不要因为常识上可行，就输出文档里没有明确给出的精确步骤、用量或烹饪流程并声称其来自证据。",
            ])
            followup_mode = str(state.get("followup_mode") or "").strip()
            followup_anchor_names = [
                str(name).strip()
                for name in state.get("followup_anchor_names", []) or []
                if str(name).strip()
            ]
            feedback_summary = str((state.get("recent_feedback_signals", {}) or {}).get("summary") or "").strip()
            if followup_mode in {"anchored_followup", "generic_followup"}:
                lines.extend([
                    "",
                    "这是一条“follow-up 延续推荐”类问题，请额外遵守：",
                    "- 主回答重点是延续推荐，先说明这次是基于上一轮方向做的新推荐。",
                    "- 优先推荐与上一轮方向相近、但不完全重复的候选；如果证据里只有重复项，再明确说明原因。",
                    "- 不要把大段通用建议写成主体；如需补充，只能放在最后，且保持很短。",
                    "- 不要复述过多系统解释，不要重新讲完整检索流程。",
                ])
                if followup_anchor_names:
                    lines.append(f"- 上一轮锚点：{'、'.join(followup_anchor_names)}。先承接这个方向，再给新的候选。")
        elif intent == "ingredient_check":
            lines.extend([
                "",
                "这是一条“食材搭配判断”类问题，请额外遵守：",
                "- 只有当证据里直接同时覆盖用户提到的食材组合时，才能说“文档明确支持/反对这个搭配”。",
                "- 如果证据没有直接覆盖该组合，只能给出明确标注为“通用建议”的保守判断，不要伪装成文档事实。",
                "- 不要补写未经证据直接支持的营养互补、禁忌原因、风险分级或医学化结论。",
            ])
        if is_recipe_steps_query:
            lines.extend([
                "",
                "这是一条“做法/步骤”类问题，请额外遵守：",
                "- 只输出证据中能支持的烹饪步骤，不要输出检索分数、标签、适合场景、适合目标、营养特点这类元数据。",
                "- 如果证据里有明确步骤，优先整理成连贯编号步骤。",
                "- 不要遗漏中间步骤，也不要跳号。",
                "- 如果证据里的步骤不完整，可以先给出证据支持的步骤，再单独补充明确标注为“通用建议”的注意事项；不要把这些通用建议混写成证据步骤。",
            ])

    retry_guidance = _build_retry_guidance(
        suggestion,
        [
            str(issue).strip()
            for issue in (state.get("evaluation", {}) or {}).get("issues", []) or []
            if str(issue).strip()
        ],
    )
    if retry_guidance:
        lines.extend([
            "",
            "这是一次基于评估反馈的重试，请仅在以下边界内改进：",
        ])
        lines.extend(f"- {item}" for item in retry_guidance)

    return "\n".join(lines)


def _sanitize_artifact_filename(value: str) -> str:
    raw = str(value or "artifact").strip() or "artifact"
    sanitized = "".join("_" if ch in '<>:"/\\|?*' else ch for ch in raw)
    sanitized = sanitized.rstrip(". ")
    return sanitized or "artifact"


def _render_bilibili_tutorial_pdf(
    tutorial_payload: dict,
    *,
    json_path: Path,
    pdf_path: Path,
    project_root: Path,
) -> Path:
    python_executable = str(settings.bilibili_pdf_python_executable or "").strip()
    if python_executable:
        try:
            return render_bilibili_tutorial_pdf_in_python(
                json_path,
                python_executable=python_executable,
                project_root=project_root,
                pdf_output=pdf_path,
            )
        except WindsurfSkillBridgeError as exc:
            logger.warning(f"configured bilibili pdf python failed, falling back to current runtime: {exc}")
    return export_tutorial_to_pdf(tutorial_payload, pdf_path)


def _build_video_summary_response(
    tutorial_payload: dict,
    artifacts: dict,
    *,
    json_path: Path,
    pdf_path: Path,
    chunk_count: int,
    success_count: int,
    fail_count: int,
    before_count: int | None,
    after_count: int | None,
) -> str:
    lines = ["**B站视频总结已生成**", ""]
    lines.append(f"- 视频标题：{tutorial_payload.get('recipe_name') or '-'}")
    if tutorial_payload.get("uploader"):
        lines.append(f"- UP 主：{tutorial_payload['uploader']}")
    if tutorial_payload.get("source_url"):
        lines.append(f"- 视频链接：{tutorial_payload['source_url']}")
    if artifacts.get("summary_source"):
        lines.append(f"- 总结来源：{artifacts['summary_source']}")
    lines.append(f"- 字幕来源：{artifacts.get('transcript_mode') or '-'}")
    lines.append(f"- 教程 JSON：`{json_path}`")
    if artifacts.get("result_json_path"):
        lines.append(f"- 运行时 result.json：`{artifacts['result_json_path']}`")
    if artifacts.get("source_result_json_path") and artifacts.get("source_result_json_path") != artifacts.get("result_json_path"):
        lines.append(f"- 技能原始 JSON：`{artifacts['source_result_json_path']}`")
    lines.append(f"- 教程 PDF：`{pdf_path}`")
    if artifacts.get("artifact_dir"):
        lines.append(f"- 原始产物目录：`{artifacts['artifact_dir']}`")
    if artifacts.get("subtitle_path"):
        lines.append(f"- 字幕文件：`{artifacts['subtitle_path']}`")
    if artifacts.get("cover_path"):
        lines.append(f"- 封面文件：`{artifacts['cover_path']}`")

    lines.extend(["", "**摘要**", str(tutorial_payload.get("summary") or "暂无摘要")])

    steps = list(tutorial_payload.get("steps") or [])
    if steps:
        lines.extend(["", "**章节结构**"])
        for step in steps[:6]:
            title = str(step.get("title") or "").strip()
            content = str(step.get("content") or "").strip()
            if title:
                lines.append(f"- **{title}**：{content}")

    lines.extend(["", "**入库结果**"])
    lines.append(f"- 生成 chunk 数：{chunk_count}")
    lines.append(f"- 成功写入/幂等跳过：{success_count}")
    lines.append(f"- 失败数：{fail_count}")
    if before_count is not None:
        lines.append(f"- collection 写入前：{before_count}")
    if after_count is not None:
        lines.append(f"- collection 写入后：{after_count}")
    return "\n".join(lines)


def _run_bilibili_video_summary_workflow(state: DietAgentState, user_query: str) -> tuple[str, str]:
    extracted_params = state.get("extracted_params", {}) or {}
    video_url = str(extracted_params.get("video_url") or "").strip()
    if not video_url:
        return (
            "请把要总结的 B 站视频链接发给我，我会按 `视频链接 -> 结构化总结 -> PDF -> 向量入库` 的流程处理。",
            "clarification",
        )

    try:
        project_root = Path(__file__).resolve().parents[3]
        artifact_root = default_bilibili_artifact_root(project_root)
        artifact_root.mkdir(parents=True, exist_ok=True)

        tutorial_payload: dict = {}
        artifacts: dict = {}
        storage = None

        if settings.bilibili_prefer_windsurf_skill_result:
            try:
                tutorial_payload, artifacts, storage = load_bilibili_tutorial_from_windsurf_result(
                    video_url=video_url,
                    project_root=project_root,
                    artifact_root=artifact_root,
                )
            except WindsurfSkillBridgeError as exc:
                expected_result_path = expected_windsurf_result_json_path(
                    video_url,
                    project_root=project_root,
                    artifact_root=artifact_root,
                )
                logger.info(
                    f"windsurf skill result not available for bilibili summary, fallback to native pipeline: {exc}; expected={expected_result_path}"
                )
                artifacts = {
                    "summary_source": "native_pipeline",
                    "expected_windsurf_result_json_path": str(expected_result_path),
                }

        if storage is None:
            tutorial_payload, native_artifacts = build_bilibili_tutorial_from_url(
                video_url,
                artifact_dir=artifact_root,
                cookies_from_browser=settings.bilibili_cookies_from_browser,
                cookies_file=settings.bilibili_cookies_file,
                whisper_model=settings.bilibili_whisper_model,
            )
            artifacts = {
                **artifacts,
                **native_artifacts,
                "summary_source": artifacts.get("summary_source") or "native_pipeline",
            }
            storage = resolve_bilibili_tutorial_storage(
                project_root=project_root,
                artifact_root=artifact_root,
                video_id=str(tutorial_payload.get("source_id") or native_artifacts.get("video_id") or ""),
                tutorial_id=str(tutorial_payload.get("tutorial_id") or ""),
                tutorial_title=str(tutorial_payload.get("title") or ""),
            )
        else:
            artifacts = {
                **artifacts,
                "summary_source": artifacts.get("summary_source") or "windsurf_skill_result",
            }

        json_path = save_tutorial_to_json(tutorial_payload, storage.tutorial_json_path)
        runtime_result_json_path = save_tutorial_to_json(tutorial_payload, storage.artifact_dir / "result.json")
        source_result_json_path = str(artifacts.get("result_json_path") or "").strip()
        artifacts = {
            **artifacts,
            "artifact_dir": str(storage.artifact_dir),
            "source_result_json_path": source_result_json_path,
            "result_json_path": str(runtime_result_json_path),
        }
        pdf_path = _render_bilibili_tutorial_pdf(
            tutorial_payload,
            json_path=json_path,
            pdf_path=storage.pdf_path,
            project_root=project_root,
        )

        chunk_corpus = build_tutorial_chunk_corpus([tutorial_payload])
        before_count = get_tutorial_collection_count(settings.bilibili_summary_collection_name)
        success_count, fail_count, errors = batch_add_tutorial_chunks(
            chunk_corpus,
            collection_name=settings.bilibili_summary_collection_name,
        )
        after_count = get_tutorial_collection_count(settings.bilibili_summary_collection_name)

        response_text = _build_video_summary_response(
            tutorial_payload,
            {**artifacts, "errors": errors},
            json_path=json_path,
            pdf_path=pdf_path,
            chunk_count=len(chunk_corpus),
            success_count=success_count,
            fail_count=fail_count,
            before_count=before_count,
            after_count=after_count,
        )
        if errors:
            response_text = f"{response_text}\n\n**入库错误**\n" + "\n".join(f"- {error}" for error in errors)
        return response_text, "recommendation"
    except Exception as exc:
        error_message = str(exc).strip()
        if error_message:
            if exc.__class__.__name__ == "BilibiliSummaryError":
                return f"我没能完成这次 B 站视频总结：{error_message}", "fallback"
            return f"B 站视频总结执行失败：{error_message}", "fallback"
        return "B 站视频总结执行失败，请稍后重试。", "fallback"


def generator_node(state: DietAgentState) -> dict:
    """生成回复节点
    
    根据意图选择不同的生成策略：
    - chitchat: 直接用 LLM 生成闲聊回复（无需检索上下文）
    - recipe_search / nutrition_query / ingredient_check: 基于 reranked_docs 生成回复

    复用模块:
    - src/llm/qwen_client.py::get_llm()
    - src/agent/prompts_v2.py — 复用已有系统提示词风格
    - src/utils/logger.py::get_logger()
    """
    intent = state.get("intent", "chitchat")
    logger.info(f"Generator 节点开始执行, intent={intent}")
    
    # 获取用户查询
    messages = state.get("messages", [])
    user_query = ""
    if messages:
        last_msg = messages[-1]
        user_query = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
    
    stable_user_preferences = normalize_stable_user_preferences(
        state.get("stable_user_preferences", {})
    )
    assembled_context: list = []
    memory_stats: dict = {}
    native_llm_fallback_used = False
    active_skill, response_contract, evidence_policy, fallback_policy = _get_generation_contracts(state)
    if stable_user_preferences:
        memory_stats["stable_preference_keys"] = sorted(stable_user_preferences.keys())
    
    try:
        if intent == "chitchat":
            llm = get_graph_llm()
            # 闲聊模式：不使用检索结果
            logger.info("闲聊模式，直接生成回复")
            response = llm.invoke([
                SystemMessage(content=CHITCHAT_SYSTEM_PROMPT),
                HumanMessage(content=user_query)
            ])
            response_text = response.content
            response_type = "recommendation"
        elif intent == "video_summary":
            logger.info("video_summary 模式，直接执行 B 站视频总结 workflow")
            response_text, response_type = _run_bilibili_video_summary_workflow(state, user_query)
        elif state.get("response_type") == "clarification":
            logger.info("澄清模式，直接输出澄清问题，不走 RAG")
            response_text = _build_clarification_response(state)
            response_type = "clarification"
        else:
            if _should_use_outline_only_subgraph_fallback(
                state,
                response_contract=response_contract,
                fallback_policy=fallback_policy,
            ):
                logger.info("命中 subgraph_candidate，当前降级为 outline_only fallback")
                response_text = _build_skill_fallback_response(
                    state,
                    intent,
                    response_contract=response_contract,
                    fallback_policy=fallback_policy,
                )
                response_type = "fallback"
            else:
                raw_reranked_docs = _resolve_generation_docs(state)
                if _should_fallback(state, intent, raw_reranked_docs):
                    fallback_mode = str(fallback_policy.get("on_low_evidence") or "").strip()
                    if active_skill == "recipe_tutorial" and fallback_mode == "native_general_cooking":
                        logger.info("recipe_tutorial 命中 low-evidence native_general_cooking 降级")
                        response_text = _build_native_general_cooking_fallback_response(
                            state,
                            user_query,
                            response_contract=response_contract,
                        )
                        response_type = "recommendation"
                        native_llm_fallback_used = True
                    elif intent == "ingredient_check":
                        conservative_response = _build_conservative_ingredient_check_response(state, raw_reranked_docs)
                        if conservative_response:
                            logger.info("ingredient_check 命中证据不足保守响应")
                            response_text = conservative_response
                            response_type = "recommendation"
                        else:
                            logger.info("命中 fallback 模式，输出统一兜底模板")
                            response_text = _build_skill_fallback_response(state, intent, response_contract=response_contract, fallback_policy=fallback_policy)
                            response_type = "fallback"
                    else:
                        logger.info("命中 fallback 模式，输出统一兜底模板")
                        response_text = _build_skill_fallback_response(state, intent, response_contract=response_contract, fallback_policy=fallback_policy)
                        response_type = "fallback"
                else:
                    reranked_docs = _prepare_docs_for_generation(raw_reranked_docs)
                    evaluation = state.get("evaluation", {})
                    suggestion = evaluation.get("suggestion", "")
                    used_assembler = False

                    structured_response = ""
                    if intent == "nutrition_query" and _looks_like_nutrition_fact_query(user_query):
                        structured_response = _build_structured_nutrition_response(user_query, raw_reranked_docs)
                    elif _looks_like_nutrition_guidance_query(user_query, intent):
                        structured_response = _build_structured_nutrition_guidance_response(user_query, state, raw_reranked_docs)
                    elif intent == "ingredient_check":
                        structured_response = _build_conservative_ingredient_check_response(state, raw_reranked_docs)
                    if not structured_response and intent == "recipe_search" and not _looks_like_recipe_steps_query(user_query, active_skill):
                        structured_response = _build_structured_followup_response(state, raw_reranked_docs, response_contract=response_contract)

                    if structured_response:
                        logger.info(
                            f"命中结构化生成模式, intent={intent}, docs={len(reranked_docs)}"
                        )
                        response_text = structured_response
                        response_type = "recommendation"
                        return {
                            "response": response_text,
                            "messages": [AIMessage(content=response_text)],
                            "response_type": response_type,
                            "reranked_docs": raw_reranked_docs,
                            "assembled_context": assembled_context,
                            "memory_stats": memory_stats,
                        }

                    llm = get_graph_llm()
                    try:
                        from ...context.context_assembler import ContextAssembler

                        history_dicts = []
                        for msg in messages[:-1]:
                            if hasattr(msg, "type"):
                                role = "user" if msg.type == "human" else "assistant"
                            elif hasattr(msg, "role"):
                                role = msg.role
                            else:
                                role = "user"
                            content = msg.content if hasattr(msg, "content") else str(msg)
                            history_dicts.append({"role": role, "content": content})

                        assembler = ContextAssembler(
                            token_budget=settings.context_total_token_budget
                        )
                        assembled_context = assembler.assemble(
                            query=user_query,
                            user_id=state.get("user_id", ""),
                            history=history_dicts,
                            retrieved_docs=reranked_docs,
                            intent=intent,
                            skill_name=active_skill,
                            stable_user_preferences=stable_user_preferences,
                        )

                        if suggestion and assembled_context:
                            last_msg = assembled_context[-1]
                            if last_msg.get("role") == "user":
                                assembled_context[-1] = {
                                    "role": "user",
                                    "content": _build_grounded_query(
                                        user_query=last_msg["content"],
                                        intent=intent,
                                        docs=reranked_docs,
                                        state=state,
                                        suggestion=suggestion,
                                        response_contract=response_contract,
                                        evidence_policy=evidence_policy,
                                    ),
                                }
                        elif assembled_context:
                            last_msg = assembled_context[-1]
                            if last_msg.get("role") == "user":
                                assembled_context[-1] = {
                                    "role": "user",
                                    "content": _build_grounded_query(
                                        user_query=last_msg["content"],
                                        intent=intent,
                                        docs=reranked_docs,
                                        state=state,
                                        response_contract=response_contract,
                                        evidence_policy=evidence_policy,
                                    ),
                                }

                        lc_messages = []
                        for m in assembled_context:
                            role = m.get("role", "user")
                            content = m.get("content", "")
                            if role == "system":
                                lc_messages.append(SystemMessage(content=content))
                            elif role == "assistant":
                                lc_messages.append(AIMessage(content=content))
                            else:
                                lc_messages.append(HumanMessage(content=content))

                        logger.info(
                            f"ContextAssembler 模式，组装 {len(assembled_context)} 条消息，"
                            f"参考 {len(reranked_docs)} 篇文档"
                        )
                        response = llm.invoke(lc_messages)
                        response_text = response.content
                        used_assembler = True
                        memory_stats = {
                            **memory_stats,
                            "assembler": True,
                            "messages": len(assembled_context),
                        }

                    except Exception as e:
                        logger.warning(f"ContextAssembler 不可用，降级为原有逻辑: {e}")

                    if not used_assembler:
                        context = _format_docs_context(reranked_docs)
                        generation_input = (
                            f"用户查询：{_build_grounded_query(user_query, intent, reranked_docs, state, suggestion, response_contract, evidence_policy)}"
                            f"\n\n参考文档：\n{context}"
                        )

                        logger.info(f"降级模式，参考 {len(reranked_docs)} 篇文档")
                        response = llm.invoke([
                            SystemMessage(content=GENERATOR_SYSTEM_PROMPT),
                            HumanMessage(content=generation_input)
                        ])
                        response_text = response.content
                    if intent == "recipe_search":
                        response_text = _append_inventory_followup_hint(response_text, state, reranked_docs)
                    response_type = "recommendation"

        logger.info(f"生成完成，回复长度: {len(response_text)}")
        
        update: dict = {
            "response": response_text,
            "messages": [AIMessage(content=response_text)],
            "response_type": response_type,
        }
        if response_type == "fallback" or native_llm_fallback_used:
            retrieval_stats = dict(state.get("retrieval_stats", {}) or {})
            if response_type == "fallback":
                retrieval_stats["fallback_triggered"] = True
            if native_llm_fallback_used:
                retrieval_stats["native_llm_fallback_used"] = True
            update["retrieval_stats"] = retrieval_stats
        if intent != "chitchat":
            update["reranked_docs"] = _resolve_generation_docs(state)
            update["assembled_context"] = assembled_context
            update["memory_stats"] = memory_stats
        return update
        
    except Exception as e:
        logger.error(f"Generator 节点执行失败: {e}", exc_info=True)
        error_response = "抱歉，我这次没能稳定生成推荐结果。你可以换一种说法，或补充食材、时间和目标后我再试一次。"
        return {
            "response": error_response,
            "messages": [AIMessage(content=error_response)],
            "response_type": "fallback",
        }
