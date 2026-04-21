"""
意图路由节点

使用 LLM + Structured Output 将用户查询分类为四种意图之一：
- recipe_search: 食谱搜索
- nutrition_query: 营养查询
- ingredient_check: 食材搭配检查
- chitchat: 闲聊

复用模块:
- src/llm/qwen_client.py::get_llm()
- src/utils/logger.py::get_logger()
"""

import re
from langchain_core.messages import HumanMessage, SystemMessage

from diet_agent.runtime import resolve_skill_runtime
from ..state import DietAgentState
from ..schemas import IntentClassification, normalize_extracted_params
from ..llm import get_graph_llm
from ...utils.logger import get_logger

logger = get_logger(__name__)

RECIPE_INTENT_HINTS = (
    "推荐",
    "吃什么",
    "晚饭",
    "晚餐",
    "午餐",
    "早餐",
    "夜宵",
    "做什么",
    "做饭",
    "家常菜",
    "快手菜",
    "清淡",
    "不油腻",
    "汤",
    "菜",
)

CHAT_INTENT_HINTS = (
    "你好",
    "您好",
    "嗨",
    "哈喽",
    "hello",
    "hi",
    "谢谢",
    "天气",
    "再见",
)

VIDEO_SUMMARY_HINTS = (
    "b站",
    "bilibili",
    "b23.tv",
    "视频总结",
    "总结这个视频",
    "总结下这个视频",
    "视频讲义",
    "视频教程总结",
)

VIDEO_URL_PATTERN = re.compile(r"(https?://(?:www\.)?(?:bilibili\.com/video/[A-Za-z0-9]+/?|b23\.tv/[A-Za-z0-9]+/?))", re.IGNORECASE)
BV_URL_HINT_PATTERN = re.compile(r"\bBV[0-9A-Za-z]+\b", re.IGNORECASE)

TUTORIAL_FOLLOWUP_HINTS = (
    "详细一点",
    "再详细一点",
    "再展开说说",
    "展开说说",
    "再展开",
    "具体步骤",
    "具体怎么做",
    "讲细一点",
    "说细一点",
    "展开一下",
    "这个怎么做",
    "这道菜怎么做",
    "它怎么做",
)

TUTORIAL_QUERY_STRIP_MARKERS = (
    "怎么做",
    "做法",
    "步骤",
    "分步骤",
    "制作",
    "教程",
    "如何做",
    "详细一点",
    "再详细一点",
    "再展开说说",
    "展开说说",
    "再展开",
    "具体步骤",
    "具体怎么做",
    "讲细一点",
    "说细一点",
    "展开一下",
    "这个",
    "这道菜",
    "这道",
    "它",
    "那个",
    "上一道",
    "上一个",
    "介绍一下",
    "简单介绍一下",
    "想吃",
    "请分步骤说明",
)

ROUTER_SYSTEM_PROMPT = """你是一个意图分类器，负责将用户的饮食相关查询分类为以下五种意图之一：

1. **recipe_search** — 用户想搜索或推荐食谱
   - 例："我想吃酸甜口味的菜"、"30分钟能做什么菜"、"推荐一道减肥菜"
   - 提取参数：ingredients(食材), time_limit(时间限制), health_goal(健康目标), difficulty(难度)

2. **nutrition_query** — 用户询问营养相关问题
   - 例："减肥应该怎么吃"、"高蛋白食物有哪些"、"每天需要多少蛋白质"
   - 提取参数：health_goal(健康目标), current_diet(当前饮食)

3. **ingredient_check** — 用户想检查食材搭配
   - 例："鸡蛋和豆浆能一起吃吗"、"番茄和什么搭配好"
   - 提取参数：ingredients(食材列表)

4. **chitchat** — 闲聊或与饮食无关的对话
   - 例："你好"、"今天天气怎么样"、"谢谢"
   - 通常不需要提取参数

5. **video_summary** — 用户希望总结 B 站视频内容，通常会提供 bilibili / b23.tv 链接
   - 例："帮我总结这个 B 站视频 https://www.bilibili.com/video/BVxxxx"、"把这个 b23.tv 视频整理成讲义"
   - 提取参数：video_url(视频链接), video_platform(固定为 bilibili), summary_scope(可选范围说明)

请根据用户最后一条消息进行分类，返回意图、置信度和提取的参数。

## 示例

用户："我有鸡蛋和番茄，想做个快手菜"
→ intent: recipe_search, confidence: 0.95, extracted_params: {"ingredients": ["鸡蛋", "番茄"], "time_limit": 30}

用户："鸡蛋和豆浆能一起吃吗"
→ intent: ingredient_check, confidence: 0.9, extracted_params: {"ingredients": ["鸡蛋", "豆浆"]}

用户："减肥期间应该怎么吃"
→ intent: nutrition_query, confidence: 0.9, extracted_params: {"health_goal": "减肥"}

用户："你好呀"
→ intent: chitchat, confidence: 0.95, extracted_params: {}

用户："帮我总结这个 B 站视频 https://www.bilibili.com/video/BV1xxxx"
→ intent: video_summary, confidence: 0.96, extracted_params: {"video_url": "https://www.bilibili.com/video/BV1xxxx", "video_platform": "bilibili"}
"""


ROUTER_RECHECK_PROMPT = """你是一个饮食助手的意图复核器。

你的任务是避免把真实的饮食需求误判为 chitchat。
只有在用户消息明确属于寒暄、感谢、告别或与饮食任务无关的轻聊天时，才返回 chitchat。
如果用户是在表达饮食偏好、想吃什么、想做什么菜、希望推荐餐食、询问营养、检查食材搭配，
即使说法模糊，也不能返回 chitchat，而应在以下三类里选择最合适的一类：
- recipe_search
- nutrition_query
- ingredient_check
请返回意图、置信度和提取参数。"""


def _normalize_query_for_rule_match(user_query: str) -> str:
    return re.sub(r"[\s，。！？!?、；;：:,]+", "", (user_query or "").strip().lower())


def _looks_like_capability_intro_query(user_query: str) -> bool:
    normalized_query = (user_query or "").strip().lower()
    compact_query = _normalize_query_for_rule_match(user_query)
    if not compact_query:
        return False

    explicit_params = _infer_explicit_query_params(user_query)
    has_actionable_constraints = bool(
        explicit_params.get("available_ingredients")
        or explicit_params.get("max_cooking_time") is not None
        or explicit_params.get("health_goal")
        or explicit_params.get("meal_type")
    )

    capability_chat_phrases = (
        "你能做什么",
        "你都能做什么",
        "你会什么",
        "你能帮我什么",
        "你能帮我做什么",
        "你可以帮我做什么",
        "可以帮我做什么",
        "能帮我做什么",
        "你是谁",
        "你是做什么的",
        "你会根据我的食材推荐菜吗",
        "你会根据我的食材推荐食谱吗",
        "你能根据我的食材推荐菜吗",
        "你可以根据我的食材推荐菜吗",
        "你也能回答营养相关的问题吗",
        "你能回答营养相关的问题吗",
        "你会回答营养相关的问题吗",
    )
    if any(
        phrase in normalized_query
        or _normalize_query_for_rule_match(phrase) in compact_query
        for phrase in capability_chat_phrases
    ):
        return True

    if has_actionable_constraints:
        return False

    capability_prefixes = ("你能", "你会", "你可以", "你也能", "你也会", "你也可以")
    capability_topics = (
        "根据我的食材推荐",
        "根据食材推荐",
        "回答营养相关的问题",
        "回答营养问题",
        "分析食材搭配",
    )
    if any(normalized_query.startswith(prefix) for prefix in capability_prefixes):
        return any(topic in normalized_query for topic in capability_topics) and normalized_query.endswith(("吗", "么", "嘛"))

    return False


def _should_force_chitchat(user_query: str) -> bool:
    compact_query = _normalize_query_for_rule_match(user_query)
    if not compact_query:
        return True

    if _looks_like_capability_intro_query(user_query):
        return True

    short_queries = {"你好", "您好", "嗨", "哈喽", "hello", "hi", "谢谢", "再见"}
    return compact_query in {_normalize_query_for_rule_match(item) for item in short_queries}


def _looks_like_tutorial_followup_query(query: str) -> bool:
    normalized_query = str(query or "").strip()
    if not normalized_query:
        return False
    return any(hint in normalized_query for hint in TUTORIAL_FOLLOWUP_HINTS)


def _extract_tutorial_topic_candidate(text: str) -> str:
    normalized_text = str(text or "").strip()
    if not normalized_text:
        return ""

    cleaned_text = normalized_text
    for marker in sorted(TUTORIAL_QUERY_STRIP_MARKERS, key=len, reverse=True):
        cleaned_text = cleaned_text.replace(marker, " ")

    candidates = [
        candidate.strip()
        for candidate in re.findall(r"[A-Za-z0-9\u4e00-\u9fff]+", cleaned_text)
        if candidate.strip()
    ]
    for candidate in sorted(candidates, key=len, reverse=True):
        if len(candidate) < 2:
            continue
        if candidate not in {"这个", "这道菜", "这道", "它", "那个", "上一道", "上一个", "再"}:
            return candidate
    return ""


def _resolve_recent_tutorial_topic_anchor(messages: list) -> str:
    prior_user_messages = []
    for message in messages[:-1]:
        if getattr(message, "type", "") == "human":
            prior_user_messages.append(message)

    for message in reversed(prior_user_messages):
        candidate = _extract_tutorial_topic_candidate(
            message.content if hasattr(message, "content") else str(message)
        )
        if candidate:
            return candidate
    return ""


def _extract_video_url(query: str) -> str:
    url_match = VIDEO_URL_PATTERN.search(str(query or ""))
    if url_match:
        return url_match.group(1).strip()

    bv_match = BV_URL_HINT_PATTERN.search(query)
    if bv_match:
        return f"https://www.bilibili.com/video/{bv_match.group(0)}"

    return ""


def _looks_like_video_summary_query(user_query: str) -> bool:
    original_query = str(user_query or "").strip()
    normalized_query = original_query.lower()
    if not normalized_query:
        return False
    extracted_video_url = _extract_video_url(original_query)
    if extracted_video_url:
        if any(hint in normalized_query for hint in VIDEO_SUMMARY_HINTS):
            return True
        stripped_without_url = normalized_query.replace(extracted_video_url.lower(), "")
        bv_match = BV_URL_HINT_PATTERN.search(original_query)
        if bv_match:
            stripped_without_url = stripped_without_url.replace(bv_match.group(0).lower(), "")
        stripped_without_url = stripped_without_url.strip(" ：:，,。.！？!?；;")
        return stripped_without_url == ""
    return False


def _infer_explicit_query_params(user_query: str) -> dict:
    normalized_query = (user_query or "").strip()
    inferred: dict[str, object] = {}

    meal_type_rules = {
        "早餐": ("早餐", "早饭"),
        "午餐": ("午餐", "午饭"),
        "晚餐": ("晚餐", "晚饭"),
        "夜宵": ("夜宵", "宵夜"),
    }
    for meal_type, phrases in meal_type_rules.items():
        if any(phrase in normalized_query for phrase in phrases):
            inferred["meal_type"] = meal_type
            break

    health_goal_rules = (
        ("高蛋白", ("高蛋白",)),
        ("减脂", ("减脂", "减肥", "瘦身")),
        ("增肌", ("增肌",)),
        ("控脂", ("控脂", "降血脂", "血脂高", "低脂")),
        ("控糖", ("控糖", "低糖")),
        ("清淡口味", ("清淡",)),
        ("低油", ("低油", "少油", "不油腻")),
    )
    for health_goal, phrases in health_goal_rules:
        if any(phrase in normalized_query for phrase in phrases):
            inferred["health_goal"] = health_goal
            break

    return normalize_extracted_params(inferred)


def _merge_extracted_params(state_params: dict, router_params: dict) -> dict:
    merged = normalize_extracted_params(state_params)
    normalized_router_params = normalize_extracted_params(router_params)

    for key, value in normalized_router_params.items():
        current_value = merged.get(key)
        if isinstance(value, list):
            if value:
                merged[key] = value
            continue
        if isinstance(value, bool):
            if value or current_value in (None, "", [], False):
                merged[key] = value
            continue
        if value not in (None, ""):
            merged[key] = value

    return merged


def _run_structured_intent_classification(prompt: str, user_query: str) -> IntentClassification:
    llm = get_graph_llm()
    structured_llm = llm.with_structured_output(IntentClassification)
    return structured_llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=user_query)
    ])


def router_node(state: DietAgentState) -> dict:
    """意图路由节点
    
    从 state["messages"] 取最后一条用户消息，使用 LLM Structured Output
    进行意图分类，返回分类结果。
    
    Args:
        state: LangGraph 全局状态
    
    Returns:
        包含 intent, confidence, extracted_params 的字典
    """
    logger.info("Router 节点开始执行")
    
    # 取最后一条用户消息
    messages = state.get("messages", [])
    if not messages:
        logger.warning("消息列表为空，默认为 chitchat")
        return {
            "intent": "chitchat",
            "confidence": 0.0,
            "active_skill": resolve_skill_runtime("chitchat", {"query": ""}).skill_name,
            "extracted_params": normalize_extracted_params({})
        }
    
    last_message = messages[-1]
    # 兼容 HumanMessage 和普通字符串
    if hasattr(last_message, "content"):
        user_query = last_message.content
    else:
        user_query = str(last_message)
    
    logger.info(f"路由用户查询: '{user_query[:100]}'")
    existing_params = state.get("extracted_params", {}) or {}

    if _looks_like_video_summary_query(user_query):
        logger.info("命中 Bilibili 视频总结规则短路，直接返回 video_summary")
        normalized_params = _merge_extracted_params(
            existing_params,
            {
                "video_url": _extract_video_url(user_query),
                "video_platform": "bilibili",
            },
        )
        resolved_skill = resolve_skill_runtime("video_summary", {**normalized_params, "query": user_query})
        return {
            "intent": "video_summary",
            "confidence": 0.99,
            "active_skill": resolved_skill.skill_name,
            "extracted_params": normalized_params,
        }

    if _looks_like_tutorial_followup_query(user_query):
        tutorial_topic_anchor = _resolve_recent_tutorial_topic_anchor(messages)
        if tutorial_topic_anchor:
            logger.info("命中教程 follow-up 规则短路，直接返回 recipe_search")
            normalized_params = _merge_extracted_params(
                existing_params,
                {"tutorial_topic_anchor": tutorial_topic_anchor},
            )
            resolved_skill = resolve_skill_runtime(
                "recipe_search",
                {**normalized_params, "query": user_query},
            )
            return {
                "intent": "recipe_search",
                "confidence": 0.99,
                "active_skill": resolved_skill.skill_name,
                "extracted_params": normalized_params,
            }

    if _should_force_chitchat(user_query):
        logger.info("命中闲聊规则短路，直接返回 chitchat")
        return {
            "intent": "chitchat",
            "confidence": 0.99,
            "active_skill": resolve_skill_runtime("chitchat", {"query": user_query}).skill_name,
            "extracted_params": normalize_extracted_params(existing_params),
        }
    
    try:
        result: IntentClassification = _run_structured_intent_classification(
            ROUTER_SYSTEM_PROMPT,
            user_query,
        )
        
        logger.info(
            f"路由结果: intent={result.intent}, "
            f"confidence={result.confidence}, "
            f"params={result.extracted_params}"
        )
        
        final_intent = result.intent
        final_confidence = result.confidence
        final_params = result.extracted_params

        if result.intent == "chitchat":
            rechecked_result: IntentClassification = _run_structured_intent_classification(
                ROUTER_RECHECK_PROMPT,
                user_query,
            )
            logger.info(
                f"路由复核结果: intent={rechecked_result.intent}, "
                f"confidence={rechecked_result.confidence}, "
                f"params={rechecked_result.extracted_params}"
            )
            if rechecked_result.intent != "chitchat":
                final_intent = rechecked_result.intent
                final_confidence = rechecked_result.confidence
                final_params = rechecked_result.extracted_params

        normalized_params = _merge_extracted_params(existing_params, final_params)
        if final_intent != "chitchat":
            normalized_params = _merge_extracted_params(
                normalized_params,
                _infer_explicit_query_params(user_query),
            )
         
        active_skill = ""
        if final_intent != "recipe_search":
            active_skill = resolve_skill_runtime(
                final_intent,
                {**normalized_params, "query": user_query},
            ).skill_name
          
        return {
            "intent": final_intent,
            "confidence": final_confidence,
            "active_skill": active_skill,
            "extracted_params": normalized_params
        }
        
    except Exception as e:
        logger.error(f"Router 节点执行失败: {e}", exc_info=True)
        # 降级为 chitchat
        return {
            "intent": "chitchat",
            "confidence": 0.0,
            "active_skill": resolve_skill_runtime("chitchat", {"query": user_query}).skill_name,
            "extracted_params": normalize_extracted_params({})
        }
