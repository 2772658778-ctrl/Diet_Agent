import json
from typing import Any


def _normalize_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        items = [item.strip() for item in value.replace("，", ",").split(",")]
    elif isinstance(value, list):
        items = value
    else:
        items = [value]

    normalized: list[str] = []
    for item in items:
        if isinstance(item, dict):
            text = str(item.get("name") or item.get("id") or "").strip()
        else:
            text = str(item).strip()
        if text and text not in normalized:
            normalized.append(text)
    return normalized


def _safe_int(value: Any) -> int:
    if value is None or value == "":
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        digits = "".join(ch for ch in str(value) if ch.isdigit())
        return int(digits) if digits else 0


def _safe_float(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _extract_ingredient_names(recipe: dict[str, Any]) -> list[str]:
    direct_candidates = [
        recipe.get("ingredient_names"),
        recipe.get("main_ingredients"),
        recipe.get("ingredients"),
    ]
    for candidate in direct_candidates:
        names = _normalize_string_list(candidate)
        if names:
            return names

    relations = recipe.get("relations", {}) or {}
    relation_candidates = [
        relations.get("contains_ingredients"),
        relations.get("main_ingredients"),
        relations.get("ingredients"),
    ]
    for candidate in relation_candidates:
        names = _normalize_string_list(candidate)
        if names:
            return names

    return []


def _extract_goal_tags(recipe: dict[str, Any]) -> list[str]:
    direct_candidates = [
        recipe.get("goal_tags"),
        recipe.get("health_goals"),
    ]
    for candidate in direct_candidates:
        values = _normalize_string_list(candidate)
        if values:
            return values

    relations = recipe.get("relations", {}) or {}
    relation_candidates = [
        relations.get("suitable_for_goals"),
        relations.get("health_goals"),
    ]
    for candidate in relation_candidates:
        values = _normalize_string_list(candidate)
        if values:
            return values

    return _normalize_string_list(recipe.get("tags"))


def _extract_meal_type(recipe: dict[str, Any]) -> str:
    direct_value = recipe.get("meal_type") or recipe.get("meal_category") or recipe.get("scene")
    if direct_value:
        values = _normalize_string_list(direct_value)
        if values:
            return values[0]

    relations = recipe.get("relations", {}) or {}
    scenarios = _normalize_string_list(relations.get("suitable_scenarios"))
    if scenarios:
        return scenarios[0]
    return ""


def _extract_allergens(recipe: dict[str, Any], ingredient_names: list[str]) -> list[str]:
    direct_candidates = [
        recipe.get("allergens"),
        recipe.get("allergen_tags"),
    ]
    for candidate in direct_candidates:
        values = _normalize_string_list(candidate)
        if values:
            return values

    relations = recipe.get("relations", {}) or {}
    relation_candidates = [
        relations.get("allergens"),
        relations.get("contains_allergens"),
        relations.get("dietary_restrictions"),
    ]
    for candidate in relation_candidates:
        values = _normalize_string_list(candidate)
        if values:
            return values

    allergen_keywords = {
        "花生", "牛奶", "乳制品", "鸡蛋", "虾", "蟹", "贝类", "鱼", "大豆", "坚果", "麸质"
    }
    return [name for name in ingredient_names if name in allergen_keywords]


def build_recipe_metadata(recipe: dict[str, Any], include_entity_fields: bool = False) -> dict[str, Any]:
    ingredient_names = _extract_ingredient_names(recipe)
    goal_tags = _extract_goal_tags(recipe)
    meal_type = _extract_meal_type(recipe)
    allergens = _extract_allergens(recipe, ingredient_names)
    cook_time = _safe_int(recipe.get("cook_time") or recipe.get("time"))
    nutrition = recipe.get("nutrition", {}) or {}

    metadata = {
        "id": str(recipe.get("id", "")),
        "name": recipe.get("name", ""),
        "cuisine": recipe.get("cuisine", ""),
        "time": cook_time,
        "cook_time": cook_time,
        "difficulty": recipe.get("difficulty", ""),
        "calories": _safe_int(recipe.get("calories", 0)),
        "tags": ",".join(_normalize_string_list(recipe.get("tags", []))),
        "health_goals": ",".join(_normalize_string_list(recipe.get("health_goals", goal_tags))),
        "goal_tags": ",".join(goal_tags),
        "ingredient_names": ",".join(ingredient_names),
        "allergens": ",".join(allergens),
        "meal_type": meal_type,
        "protein": _safe_float(nutrition.get("protein", 0.0)),
        "carbs": _safe_float(nutrition.get("carbs", 0.0)),
        "fat": _safe_float(nutrition.get("fat", 0.0)),
        "fiber": _safe_float(nutrition.get("fiber", 0.0)),
    }

    if include_entity_fields:
        metadata["entity_type"] = "recipe"
        metadata["relations"] = json.dumps(recipe.get("relations", {}), ensure_ascii=False)

    return metadata
