from __future__ import annotations

import hashlib
import json
from html import escape
from pathlib import Path
from typing import Any, Iterable, Sequence


class TutorialPipelineError(RuntimeError):
    pass


def _normalize_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        items = [item.strip() for item in value.replace("，", ",").split(",")]
    elif isinstance(value, (list, tuple, set)):
        items = list(value)
    else:
        items = [value]

    normalized: list[str] = []
    for item in items:
        if isinstance(item, dict):
            text = str(item.get("name") or item.get("title") or item.get("id") or "").strip()
        else:
            text = str(item).strip()
        if text and text not in normalized:
            normalized.append(text)
    return normalized


def _safe_int(value: Any) -> int:
    if value in (None, ""):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        digits = "".join(ch for ch in str(value) if ch.isdigit())
        return int(digits) if digits else 0


def _safe_float(value: Any) -> float:
    if value in (None, ""):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _relations(recipe: dict[str, Any]) -> dict[str, Any]:
    relations = recipe.get("relations", {}) or {}
    return relations if isinstance(relations, dict) else {}


def _sanitize_filename(value: str) -> str:
    raw = str(value or "tutorial").strip() or "tutorial"
    sanitized = "".join("_" if ch in '<>:"/\\|?*' else ch for ch in raw)
    sanitized = sanitized.rstrip(". ")
    return sanitized or "tutorial"


def _stable_hash(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]


def _resolve_source_id(recipe: dict[str, Any]) -> str:
    recipe_id = str(recipe.get("id") or "").strip()
    if recipe_id:
        return recipe_id
    name = str(recipe.get("name") or "unnamed_recipe").strip() or "unnamed_recipe"
    return f"recipe_{_stable_hash(name)}"


def _format_amount(amount: Any, unit: str) -> str:
    if amount in (None, ""):
        return ""
    amount_text = str(amount).strip()
    unit_text = str(unit or "").strip()
    return f"{amount_text}{unit_text}".strip()


def _extract_ingredients(recipe: dict[str, Any]) -> list[dict[str, str]]:
    candidates: list[Any] = []
    direct_ingredients = recipe.get("ingredients")
    if isinstance(direct_ingredients, list):
        candidates.extend(direct_ingredients)

    relations = _relations(recipe)
    relation_ingredients = relations.get("contains_ingredients")
    if isinstance(relation_ingredients, list):
        candidates.extend(relation_ingredients)

    normalized: list[dict[str, str]] = []
    seen_names: set[str] = set()
    for item in candidates:
        if isinstance(item, dict):
            name = str(item.get("name") or item.get("id") or "").strip()
            amount_text = _format_amount(item.get("amount"), str(item.get("unit") or "").strip())
        else:
            name = str(item).strip()
            amount_text = ""
        if not name or name in seen_names:
            continue
        seen_names.add(name)
        display = f"{name}（{amount_text}）" if amount_text else name
        normalized.append(
            {
                "name": name,
                "amount": amount_text,
                "display": display,
            }
        )

    if normalized:
        return normalized

    fallback_names = _normalize_string_list(recipe.get("ingredient_names"))
    return [{"name": name, "amount": "", "display": name} for name in fallback_names]


def _extract_actual_steps(recipe: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("steps", "instructions", "method"):
        raw_steps = recipe.get(key)
        if not isinstance(raw_steps, list):
            continue
        normalized_steps: list[dict[str, Any]] = []
        for index, item in enumerate(raw_steps, start=1):
            if isinstance(item, dict):
                title = str(item.get("title") or item.get("name") or f"步骤{index}").strip() or f"步骤{index}"
                content = str(
                    item.get("content")
                    or item.get("detail")
                    or item.get("text")
                    or item.get("value")
                    or ""
                ).strip()
            else:
                title = f"步骤{index}"
                content = str(item).strip()
            if not content:
                continue
            normalized_steps.append(
                {
                    "index": index,
                    "title": title,
                    "content": content,
                }
            )
        if normalized_steps:
            return normalized_steps
    return []


def _build_summary(recipe: dict[str, Any], scenarios: Sequence[str], goals: Sequence[str]) -> str:
    description = str(recipe.get("description") or "").strip()
    time_minutes = _safe_int(recipe.get("time") or recipe.get("cook_time"))
    difficulty = str(recipe.get("difficulty") or "").strip()
    parts: list[str] = []
    if description:
        parts.append(description.rstrip("。"))
    if time_minutes:
        parts.append(f"预计完成时间约{time_minutes}分钟")
    if difficulty:
        parts.append(f"难度为{difficulty}")
    if scenarios:
        parts.append(f"适合{('、'.join(scenarios[:3]))}等场景")
    if goals:
        parts.append(f"常见目标包括{('、'.join(goals[:3]))}")
    if not parts:
        parts.append("基于现有 recipe 元数据自动整理的标准化教程")
    return "。".join(part.rstrip("。") for part in parts if part).strip() + "。"


def _build_template_steps(
    recipe: dict[str, Any],
    ingredients: Sequence[dict[str, str]],
    tags: Sequence[str],
    scenarios: Sequence[str],
    goals: Sequence[str],
) -> list[dict[str, Any]]:
    name = str(recipe.get("name") or "这道菜").strip() or "这道菜"
    description = str(recipe.get("description") or "").strip()
    cuisine = str(recipe.get("cuisine") or "").strip()
    difficulty = str(recipe.get("difficulty") or "").strip()
    time_minutes = _safe_int(recipe.get("time") or recipe.get("cook_time"))
    ingredient_preview = "、".join(item["display"] for item in ingredients[:6]) or name
    tag_preview = "、".join(tags[:3])
    goal_preview = "、".join(goals[:3])
    scenario_preview = "、".join(scenarios[:3])

    core_style = "按常规家庭做法完成主体烹饪"
    if "沙拉" in name or "沙拉" in tag_preview:
        core_style = "以分批处理主食材、混合配菜和最后调味为主"
    elif "汤" in name or "汤品" in tag_preview:
        core_style = "以煮制、调味和控制汤体口感为主"
    elif "粥" in name:
        core_style = "以小火加热、控制稠度和分段观察状态为主"

    steps = [
        {
            "index": 1,
            "title": "准备食材",
            "content": f"先准备{ingredient_preview}，再根据{name}的分量完成清洗、切配和分装。",
        },
        {
            "index": 2,
            "title": "完成预处理",
            "content": (
                f"围绕“{description or name}”这一路径整理主料和辅料；"
                f"如果你是{difficulty or '首次尝试'}，建议先把所有预处理一次性完成再进入加热阶段。"
            ),
        },
        {
            "index": 3,
            "title": "主体烹饪",
            "content": (
                f"结合{cuisine or '当前菜式'}特点，{core_style}；"
                f"整体节奏可参考约{time_minutes}分钟，重点围绕{tag_preview or '口感稳定'}调整火候和调味。"
            ),
        },
        {
            "index": 4,
            "title": "收尾上桌",
            "content": (
                f"出锅前检查熟度与口感，再根据{goal_preview or scenario_preview or '当前需求'}微调油盐和摆盘，完成上桌。"
            ),
        },
    ]
    return steps


def _build_nutrition_highlights(recipe: dict[str, Any], goals: Sequence[str]) -> list[str]:
    nutrition = recipe.get("nutrition", {}) or {}
    if not isinstance(nutrition, dict):
        nutrition = {}
    calories = _safe_int(recipe.get("calories"))
    protein = _safe_float(nutrition.get("protein"))
    carbs = _safe_float(nutrition.get("carbs"))
    fat = _safe_float(nutrition.get("fat"))
    fiber = _safe_float(nutrition.get("fiber"))
    vitamins = _normalize_string_list(nutrition.get("vitamins"))

    highlights: list[str] = []
    if calories:
        highlights.append(f"热量约{calories}千卡")
    if protein:
        highlights.append(f"蛋白质约{protein:g}克")
    if carbs:
        highlights.append(f"碳水约{carbs:g}克")
    if fat:
        highlights.append(f"脂肪约{fat:g}克")
    if fiber:
        highlights.append(f"膳食纤维约{fiber:g}克")
    if vitamins:
        highlights.append(f"维生素关注点：{'、'.join(vitamins[:4])}")
    if goals:
        highlights.append(f"常见适配目标：{'、'.join(goals[:3])}")
    return highlights


def _build_tips(
    recipe: dict[str, Any],
    scenarios: Sequence[str],
    goals: Sequence[str],
    nutrition_highlights: Sequence[str],
) -> list[str]:
    time_minutes = _safe_int(recipe.get("time") or recipe.get("cook_time"))
    difficulty = str(recipe.get("difficulty") or "").strip()
    alternative_recipes = _relations(recipe).get("alternative_recipes") or []
    alternatives = _normalize_string_list(alternative_recipes)

    tips: list[str] = []
    if time_minutes and time_minutes <= 15:
        tips.append("建议开火前一次性备好全部食材，保持快手出餐节奏。")
    if difficulty == "简单":
        tips.append("首次尝试时可先按标准配比完成，再根据口味做小幅调整。")
    elif difficulty:
        tips.append(f"这道菜难度为{difficulty}，建议分阶段检查熟度、火候和调味状态。")
    if scenarios:
        tips.append(f"适合场景：{'、'.join(scenarios[:3])}。")
    if goals:
        tips.append(f"如果你关注{'、'.join(goals[:2])}，优先控制额外油盐和高热量配料。")
    if nutrition_highlights:
        tips.append(f"营养关注点：{'；'.join(nutrition_highlights[:3])}。")
    if alternatives:
        tips.append(f"可横向对比：{'、'.join(alternatives[:2])}。")
    return tips


def _build_related_recipes(recipe: dict[str, Any]) -> list[str]:
    relations = _relations(recipe)
    related_names = _normalize_string_list(relations.get("similar_recipes"))
    alternative_names = _normalize_string_list(relations.get("alternative_recipes"))
    merged: list[str] = []
    for name in [*related_names, *alternative_names]:
        if name not in merged:
            merged.append(name)
    return merged


def _ensure_tutorial(payload: dict[str, Any]) -> dict[str, Any]:
    if isinstance(payload, dict) and payload.get("tutorial_id") and isinstance(payload.get("steps"), list):
        return payload
    return build_recipe_tutorial(payload)


def load_recipes_from_file(file_path: str | Path) -> list[dict[str, Any]]:
    path = Path(file_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("tutorial pipeline input must be a list of recipes")
    recipes = [item for item in data if isinstance(item, dict)]
    if len(recipes) != len(data):
        raise ValueError("tutorial pipeline input contains non-dict records")
    return recipes


def build_recipe_tutorial(recipe: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(recipe, dict):
        raise ValueError("recipe must be a dict")

    source_id = _resolve_source_id(recipe)
    name = str(recipe.get("name") or source_id).strip() or source_id
    tags = _normalize_string_list(recipe.get("tags"))
    relations = _relations(recipe)
    scenarios = _normalize_string_list(relations.get("suitable_scenarios"))
    goals = _normalize_string_list(recipe.get("health_goals")) or _normalize_string_list(relations.get("suitable_for_goals"))
    ingredients = _extract_ingredients(recipe)
    actual_steps = _extract_actual_steps(recipe)
    step_source = "recipe" if actual_steps else "template"
    steps = actual_steps or _build_template_steps(recipe, ingredients, tags, scenarios, goals)
    nutrition = recipe.get("nutrition", {}) or {}
    if not isinstance(nutrition, dict):
        nutrition = {}
    nutrition_highlights = _build_nutrition_highlights(recipe, goals)
    summary = _build_summary(recipe, scenarios, goals)
    tips = _build_tips(recipe, scenarios, goals, nutrition_highlights)

    return {
        "tutorial_id": f"tutorial_{source_id}",
        "source_type": "recipe",
        "source_id": source_id,
        "recipe_name": name,
        "title": f"{name}制作教程",
        "summary": summary,
        "difficulty": str(recipe.get("difficulty") or "").strip(),
        "time_minutes": _safe_int(recipe.get("time") or recipe.get("cook_time")),
        "cuisine": str(recipe.get("cuisine") or "").strip(),
        "calories": _safe_int(recipe.get("calories")),
        "tags": tags,
        "health_goals": goals,
        "scenarios": scenarios,
        "ingredients": ingredients,
        "steps": steps,
        "step_source": step_source,
        "tips": tips,
        "nutrition": {
            "protein": _safe_float(nutrition.get("protein")),
            "carbs": _safe_float(nutrition.get("carbs")),
            "fat": _safe_float(nutrition.get("fat")),
            "fiber": _safe_float(nutrition.get("fiber")),
            "vitamins": _normalize_string_list(nutrition.get("vitamins")),
        },
        "nutrition_highlights": nutrition_highlights,
        "related_recipes": _build_related_recipes(recipe),
    }


def build_recipe_tutorials(recipes: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [build_recipe_tutorial(recipe) for recipe in recipes]


def save_tutorial_to_json(tutorial_or_recipe: dict[str, Any], output_path: str | Path) -> Path:
    tutorial = _ensure_tutorial(tutorial_or_recipe)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(tutorial, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def save_tutorials_to_json(tutorials: Sequence[dict[str, Any]], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list(tutorials), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _paragraph_text(value: Any) -> str:
    return escape(str(value or "")).replace("\n", "<br/>")


def export_tutorial_to_pdf(tutorial_or_recipe: dict[str, Any], output_path: str | Path) -> Path:
    tutorial = _ensure_tutorial(tutorial_or_recipe)
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.cidfonts import UnicodeCIDFont
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except ImportError as exc:
        raise TutorialPipelineError("reportlab is required for PDF export") from exc

    try:
        pdfmetrics.getFont("STSong-Light")
    except KeyError:
        pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    font_name = "STSong-Light"
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TutorialTitle",
        parent=styles["Title"],
        fontName=font_name,
        fontSize=20,
        leading=26,
        textColor=colors.HexColor("#111827"),
        spaceAfter=10,
    )
    heading_style = ParagraphStyle(
        "TutorialHeading",
        parent=styles["Heading2"],
        fontName=font_name,
        fontSize=13,
        leading=18,
        textColor=colors.HexColor("#1f2937"),
        spaceBefore=6,
        spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "TutorialBody",
        parent=styles["BodyText"],
        fontName=font_name,
        fontSize=10.5,
        leading=16,
        textColor=colors.HexColor("#374151"),
    )
    meta_label_style = ParagraphStyle(
        "TutorialMetaLabel",
        parent=body_style,
        fontName=font_name,
        textColor=colors.HexColor("#6b7280"),
    )

    story: list[Any] = []
    story.append(Paragraph(_paragraph_text(tutorial.get("title")), title_style))
    story.append(Paragraph(_paragraph_text(tutorial.get("summary")), body_style))
    story.append(Spacer(1, 5 * mm))

    metadata_pairs = [
        ("菜系", tutorial.get("cuisine") or "-"),
        ("难度", tutorial.get("difficulty") or "-"),
        ("预计用时", f"{tutorial.get('time_minutes')}分钟" if tutorial.get("time_minutes") else "-"),
        ("热量", f"{tutorial.get('calories')}千卡" if tutorial.get("calories") else "-"),
        ("健康目标", "、".join(tutorial.get("health_goals") or []) or "-"),
        ("适合场景", "、".join(tutorial.get("scenarios") or []) or "-"),
    ]
    metadata_rows = [
        [Paragraph(_paragraph_text(label), meta_label_style), Paragraph(_paragraph_text(value), body_style)]
        for label, value in metadata_pairs
    ]
    metadata_table = Table(metadata_rows, colWidths=[28 * mm, 145 * mm])
    metadata_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f3f4f6")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#d1d5db")),
                ("INNERGRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#e5e7eb")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(metadata_table)
    story.append(Spacer(1, 5 * mm))

    story.append(Paragraph("食材准备", heading_style))
    ingredient_lines = [item.get("display") or item.get("name") or "" for item in tutorial.get("ingredients") or []]
    story.append(Paragraph(_paragraph_text("；".join(line for line in ingredient_lines if line) or "暂无食材信息"), body_style))
    story.append(Spacer(1, 4 * mm))

    story.append(Paragraph("步骤说明", heading_style))
    for step in tutorial.get("steps") or []:
        step_index = step.get("index") or 0
        step_title = step.get("title") or f"步骤{step_index}"
        step_content = step.get("content") or ""
        story.append(Paragraph(_paragraph_text(f"{step_index}. {step_title}：{step_content}"), body_style))
        story.append(Spacer(1, 2 * mm))

    tips = tutorial.get("tips") or []
    if tips:
        story.append(Spacer(1, 2 * mm))
        story.append(Paragraph("烹饪提示", heading_style))
        for tip in tips:
            story.append(Paragraph(_paragraph_text(f"- {tip}"), body_style))
            story.append(Spacer(1, 1 * mm))

    nutrition_highlights = tutorial.get("nutrition_highlights") or []
    if nutrition_highlights:
        story.append(Spacer(1, 2 * mm))
        story.append(Paragraph("营养重点", heading_style))
        for item in nutrition_highlights:
            story.append(Paragraph(_paragraph_text(f"- {item}"), body_style))
            story.append(Spacer(1, 1 * mm))

    document = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
    )
    document.build(story)
    return path


def export_tutorials_to_pdf(
    tutorials_or_recipes: Sequence[dict[str, Any]],
    output_dir: str | Path,
) -> list[Path]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    generated_paths: list[Path] = []
    for payload in tutorials_or_recipes:
        tutorial = _ensure_tutorial(payload)
        file_name = f"{_sanitize_filename(str(tutorial.get('title') or tutorial.get('tutorial_id') or 'tutorial'))}.pdf"
        generated_paths.append(export_tutorial_to_pdf(tutorial, output_root / file_name))
    return generated_paths


def build_tutorial_chunk_documents(
    tutorial_or_recipe: dict[str, Any],
    max_steps_per_chunk: int = 3,
) -> list[dict[str, Any]]:
    tutorial = _ensure_tutorial(tutorial_or_recipe)
    if max_steps_per_chunk <= 0:
        raise ValueError("max_steps_per_chunk must be positive")

    tutorial_id = str(tutorial.get("tutorial_id") or "").strip()
    title = str(tutorial.get("title") or tutorial_id).strip() or tutorial_id
    recipe_name = str(tutorial.get("recipe_name") or title).strip() or title
    source_id = str(tutorial.get("source_id") or "").strip()
    tags = ",".join(_normalize_string_list(tutorial.get("tags")))
    goals = ",".join(_normalize_string_list(tutorial.get("health_goals")))
    scenarios = ",".join(_normalize_string_list(tutorial.get("scenarios")))
    difficulty = str(tutorial.get("difficulty") or "").strip()
    cuisine = str(tutorial.get("cuisine") or "").strip()
    time_minutes = _safe_int(tutorial.get("time_minutes"))
    calories = _safe_int(tutorial.get("calories"))
    step_source = str(tutorial.get("step_source") or "").strip()

    base_metadata = {
        "entity_type": "tutorial_chunk",
        "tutorial_id": tutorial_id,
        "tutorial_title": title,
        "recipe_id": source_id,
        "recipe_name": recipe_name,
        "source_type": str(tutorial.get("source_type") or "tutorial"),
        "difficulty": difficulty,
        "time": time_minutes,
        "calories": calories,
        "cuisine": cuisine,
        "tags": tags,
        "health_goals": goals,
        "scenarios": scenarios,
        "step_source": step_source,
    }

    chunks: list[dict[str, Any]] = []

    overview_parts = [
        f"教程标题：{title}。",
        f"摘要：{tutorial.get('summary') or ''}。",
    ]
    if goals:
        overview_parts.append(f"适配目标：{goals}。")
    if scenarios:
        overview_parts.append(f"适合场景：{scenarios}。")
    overview_parts.append(f"难度：{difficulty or '-'}；用时：{time_minutes or '-'}分钟；菜系：{cuisine or '-'}。")
    chunks.append(
        {
            "id": f"{tutorial_id}::overview::0",
            "text": "".join(overview_parts),
            "metadata": {**base_metadata, "chunk_type": "overview", "chunk_index": 0},
        }
    )

    ingredients = tutorial.get("ingredients") or []
    ingredient_text = "；".join(
        str(item.get("display") or item.get("name") or "").strip()
        for item in ingredients
        if isinstance(item, dict) and str(item.get("display") or item.get("name") or "").strip()
    )
    chunks.append(
        {
            "id": f"{tutorial_id}::ingredients::0",
            "text": f"{title}食材清单：{ingredient_text or '暂无食材信息'}。",
            "metadata": {**base_metadata, "chunk_type": "ingredients", "chunk_index": 0},
        }
    )

    steps = tutorial.get("steps") or []
    for chunk_index, start in enumerate(range(0, len(steps), max_steps_per_chunk)):
        step_group = steps[start : start + max_steps_per_chunk]
        step_text = "".join(
            f"第{step.get('index') or (start + offset + 1)}步 {step.get('title') or f'步骤{start + offset + 1}'}：{step.get('content') or ''}。"
            for offset, step in enumerate(step_group)
            if isinstance(step, dict)
        )
        if not step_text:
            continue
        chunks.append(
            {
                "id": f"{tutorial_id}::steps::{chunk_index}",
                "text": f"{title}步骤讲解：{step_text}",
                "metadata": {
                    **base_metadata,
                    "chunk_type": "steps",
                    "chunk_index": chunk_index,
                    "step_start": _safe_int(step_group[0].get("index")) if step_group else 0,
                    "step_end": _safe_int(step_group[-1].get("index")) if step_group else 0,
                },
            }
        )

    closing_parts: list[str] = []
    tips = _normalize_string_list(tutorial.get("tips"))
    if tips:
        closing_parts.append(f"烹饪提示：{'；'.join(tips)}。")
    nutrition_highlights = _normalize_string_list(tutorial.get("nutrition_highlights"))
    if nutrition_highlights:
        closing_parts.append(f"营养重点：{'；'.join(nutrition_highlights)}。")
    related_recipes = _normalize_string_list(tutorial.get("related_recipes"))
    if related_recipes:
        closing_parts.append(f"相关菜品：{'、'.join(related_recipes[:4])}。")
    if closing_parts:
        chunks.append(
            {
                "id": f"{tutorial_id}::tips::0",
                "text": f"{title}补充信息：{''.join(closing_parts)}",
                "metadata": {**base_metadata, "chunk_type": "tips", "chunk_index": 0},
            }
        )

    return chunks


def build_tutorial_chunk_corpus(
    tutorials_or_recipes: Iterable[dict[str, Any]],
    max_steps_per_chunk: int = 3,
) -> list[dict[str, Any]]:
    corpus: list[dict[str, Any]] = []
    for payload in tutorials_or_recipes:
        corpus.extend(build_tutorial_chunk_documents(payload, max_steps_per_chunk=max_steps_per_chunk))
    return corpus
