from typing import Any


def _contains_any(text: str, keywords: list[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def extract_query_features(query: str, extracted_params: dict[str, Any] | None = None) -> dict[str, float | int | bool]:
    params = extracted_params or {}
    query_text = str(query or "").strip().lower()

    available_ingredients = params.get("available_ingredients") or params.get("ingredients") or []
    allergies = params.get("allergies") or params.get("dietary_restrictions") or []
    disliked_ingredients = params.get("disliked_ingredients") or []
    max_cooking_time = params.get("max_cooking_time") or params.get("time_limit")
    health_goal = params.get("health_goal")
    meal_type = params.get("meal_type")

    active_constraints = sum(
        1
        for value in [available_ingredients, allergies, disliked_ingredients, max_cooking_time, health_goal, meal_type]
        if value not in (None, "", [], {})
    )

    query_constraint_hits = sum(
        1 for keyword in ["分钟", "减脂", "增肌", "低卡", "早餐", "午餐", "晚餐", "过敏", "不要", "别放"] if keyword in query_text
    )
    entity_constraint_density = min(1.0, round((active_constraints + query_constraint_hits * 0.5) / 4.0, 4))

    abstract_markers = ["吃什么", "来点", "推荐点", "随便", "不知道", "怎么吃", "方案", "建议"]
    concrete_markers = ["鸡蛋", "番茄", "牛肉", "分钟", "早餐", "午餐", "晚餐", "减脂", "增肌"]
    abstraction_raw = 0.2
    if _contains_any(query_text, abstract_markers):
        abstraction_raw += 0.45
    if active_constraints == 0:
        abstraction_raw += 0.2
    if _contains_any(query_text, concrete_markers) or active_constraints >= 2:
        abstraction_raw -= 0.35
    semantic_abstraction_score = round(min(1.0, max(0.0, abstraction_raw)), 4)

    inventory_markers = ["我有", "家里有", "冰箱里", "剩下", "现有", "手头有", "库存"]
    inventory_signal = 0.0
    if available_ingredients:
        inventory_signal = 1.0
    elif _contains_any(query_text, inventory_markers):
        inventory_signal = 0.75
    elif "食材" in query_text:
        inventory_signal = 0.35
    inventory_signal = round(inventory_signal, 4)

    return {
        "entity_constraint_density": entity_constraint_density,
        "semantic_abstraction_score": semantic_abstraction_score,
        "inventory_signal": inventory_signal,
        "active_constraint_count": active_constraints,
        "has_inventory_context": bool(inventory_signal >= 0.75),
    }
