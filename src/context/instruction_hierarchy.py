# -*- coding: utf-8 -*-
"""
指令层级与 Skill 管理

- InstructionHierarchy: 管理分层 system instruction（base + intent-specific）
- SkillRegistry: 封装常见任务为可复用 Skill（prompt_template + few_shot_examples）

复用模块:
- src/config.py::get_settings()
- src/utils/logger.py::get_logger()
"""

from typing import Dict, List, Optional, Any

from ..config import get_settings
from ..utils.logger import get_logger


logger = get_logger(__name__)
settings = get_settings()


# ── 内置 Prompt 常量 ──────────────────────────────────────────────────────────

_BASE_SYSTEM_PROMPT: str = """你是一个专业的智能饮食助手，具备丰富的食谱知识、营养学知识和食材搭配知识。

## 核心职责
1. 帮助用户找到符合需求的食谱（考虑口味、时间、食材、健康目标等）
2. 分析食谱或饮食计划的营养成分，提供科学建议
3. 检查食材搭配的合理性和安全性
4. 根据用户的饮食偏好和健康目标提供个性化建议

## 基本原则
- 基于提供的参考信息回复，不编造数据
- 回复简洁、专业、友好，使用 Markdown 格式
- 考虑用户的健康状况、过敏信息和饮食偏好
- 如参考文档不足，诚实说明并给出尽可能有用的建议"""

_INTENT_PROMPTS: Dict[str, str] = {
    "recipe_search": """## 当前任务：食谱搜索与推荐

请根据用户的需求（食材、口味、时间限制、健康目标等）推荐 2-3 道合适的食谱。

推荐格式：
- **菜名**：简要描述
- 烹饪时间：X 分钟 | 难度：XX | 热量：约 XX 卡
- 推荐理由：说明为何适合用户需求""",

    "nutrition_query": """## 当前任务：营养分析与建议

请基于参考文档，对用户查询的食物或食谱进行营养分析，并提供科学合理的饮食建议。

回复应包含：
- 主要营养成分（热量、蛋白质、脂肪、碳水化合物）
- 营养特点与健康价值
- 摄入建议和注意事项""",

    "ingredient_check": """## 当前任务：食材搭配检查

请检查用户提到的食材搭配是否合理、安全，并给出具体建议。

回复应包含：
- 搭配合理性评估（✅ 安全搭配 / ⚠️ 需注意 / ❌ 不建议搭配）
- 搭配说明（营养互补/口味协调/潜在风险等）
- 替代建议（如有问题）""",

    "chitchat": """## 当前任务：日常闲聊

用户在进行非饮食相关的闲聊，请用简短、友好的方式回复（2-3句话）。
如有机会，可自然引导到饮食健康相关的话题。""",

    "video_summary": """## 当前任务：B站视频总结

用户希望把 B 站视频整理成结构化中文讲义。

请优先完成以下目标：
- 识别并使用用户提供的视频链接
- 基于真实元数据和字幕整理教学型总结
- 明确返回 JSON、PDF 与本地产物路径
- 如执行失败，说明失败原因和下一步需要的信息""",
}

# ── 内置 Skill 定义 ───────────────────────────────────────────────────────────

BUILTIN_SKILLS: Dict[str, Dict[str, Any]] = {
    "recipe_recommendation": {
        "description": "根据用户需求推荐食谱",
        "required_tools": ["search_recipes_v2", "check_ingredient_pairing"],
        "intent": "recipe_search",
        "prompt_template": (
            "你是一名专业的饮食推荐师。根据用户的需求和提供的参考食谱，"
            "推荐最合适的 2-3 道菜。需综合考虑：食材匹配度、烹饪时间、"
            "口味偏好、热量要求和难度适配。"
        ),
        "few_shot_examples": [
            {
                "user": "我有土豆和鸡蛋，想做一道30分钟内能完成的家常菜",
                "assistant": (
                    "**推荐菜单**\n\n"
                    "1. **土豆烧鸡蛋**\n"
                    "   - 烹饪时间：20分钟 | 难度：简单 | 热量：约200卡\n"
                    "   - 推荐理由：食材完全匹配，家常做法简单，营养均衡\n\n"
                    "2. **西红柿炒鸡蛋配土豆饼**\n"
                    "   - 烹饪时间：25分钟 | 难度：简单 | 热量：约250卡\n"
                    "   - 推荐理由：若有西红柿，口味更丰富，营养互补"
                ),
            },
            {
                "user": "推荐一道适合减脂的午餐，不想吃太油腻的",
                "assistant": (
                    "**减脂午餐推荐**\n\n"
                    "1. **清蒸鲈鱼**\n"
                    "   - 烹饪时间：20分钟 | 难度：简单 | 热量：约150卡\n"
                    "   - 推荐理由：低脂高蛋白，清淡少油，饱腹感强\n\n"
                    "2. **鸡胸肉炒西兰花**\n"
                    "   - 烹饪时间：15分钟 | 难度：简单 | 热量：约180卡\n"
                    "   - 推荐理由：高蛋白低碳水，非常适合减脂期"
                ),
            },
        ],
    },
    "nutrition_analysis": {
        "description": "分析食谱或饮食的营养成分",
        "required_tools": ["get_nutrition_advice"],
        "intent": "nutrition_query",
        "prompt_template": (
            "你是一名注册营养师。基于提供的营养数据，给出专业、易懂的营养分析报告。"
            "重点关注用户的健康目标，给出有针对性的饮食建议。"
        ),
        "few_shot_examples": [
            {
                "user": "番茄炒鸡蛋的营养价值怎么样？",
                "assistant": (
                    "**番茄炒鸡蛋营养分析**\n\n"
                    "**主要营养成分**（每份约200g）：\n"
                    "- 热量：约180-220 卡路里\n"
                    "- 蛋白质：12-15g（鸡蛋提供优质蛋白）\n"
                    "- 脂肪：10-12g（以不饱和脂肪为主）\n"
                    "- 碳水化合物：8-10g\n\n"
                    "**营养特点**：维生素C（番茄）+ 完全蛋白（鸡蛋），"
                    "营养互补，性价比极高。\n\n"
                    "**建议**：搭配粗粮主食，膳食均衡效果更佳。"
                ),
            },
        ],
    },
    "meal_planning": {
        "description": "制定一周饮食计划",
        "required_tools": ["search_recipes_v2", "get_nutrition_advice"],
        "intent": "recipe_search",
        "prompt_template": (
            "你是一名专业的膳食规划师。根据用户的健康目标和饮食偏好，"
            "制定合理的一周饮食计划。确保每日营养均衡（三大营养素比例合理），"
            "兼顾口味多样性和烹饪便利性。"
        ),
        "few_shot_examples": [
            {
                "user": "帮我制定一个减脂的一周饮食计划",
                "assistant": (
                    "**减脂一周饮食计划**（每日约1400-1600卡）\n\n"
                    "**周一**\n"
                    "- 早餐：燕麦粥 + 水煮鸡蛋\n"
                    "- 午餐：鸡胸肉沙拉 + 糙米饭（小碗）\n"
                    "- 晚餐：清蒸鱼 + 炒青菜\n\n"
                    "**周二**\n"
                    "- 早餐：全麦面包 + 低脂牛奶\n"
                    "- 午餐：番茄炒鸡蛋 + 糙米饭\n"
                    "- 晚餐：豆腐汤 + 西兰花炒虾仁\n\n"
                    "（后续天数类似，保持多样化）\n\n"
                    "**注意事项**：减少精制碳水，增加膳食纤维，"
                    "保证每日饮水 1500-2000ml。"
                ),
            },
        ],
    },
}


# ── InstructionHierarchy ──────────────────────────────────────────────────────

class InstructionHierarchy:
    """指令层级管理器

    管理分层 system instruction，根据意图组装：
        base_prompt (通用饮食助手角色)
        + intent_prompt (意图特定指令)
        + user_context (用户画像注入，可选)
    """

    def __init__(self) -> None:
        """初始化指令层级管理器"""
        self._base_prompt: str = _BASE_SYSTEM_PROMPT
        self._intent_prompts: Dict[str, str] = _INTENT_PROMPTS
        logger.debug("InstructionHierarchy 初始化完成")

    def get_system_instruction(
        self,
        intent: str,
        skill_name: Optional[str] = None,
    ) -> str:
        """根据意图返回 system instruction

        Args:
            intent: 用户意图（recipe_search / nutrition_query / ingredient_check / chitchat）
            skill_name: 可选的 Skill 名称；若提供则优先使用 Skill 的 prompt_template

        Returns:
            组装后的 system instruction 字符串
        """
        if skill_name:
            try:
                from diet_agent.runtime import get_skill_assets

                asset_view = get_skill_assets(skill_name)
            except Exception:
                asset_view = None
            if asset_view and asset_view.has_prompt_template:
                logger.debug(f"使用 Skill '{skill_name}' 的资产 prompt_template")
                return f"{self._base_prompt}\n\n{asset_view.prompt_template}"

        if skill_name and skill_name in BUILTIN_SKILLS:
            skill_template = BUILTIN_SKILLS[skill_name].get("prompt_template", "")
            if skill_template:
                logger.debug(f"使用 Skill '{skill_name}' 的 prompt_template")
                return f"{self._base_prompt}\n\n{skill_template}"

        intent_prompt = self._intent_prompts.get(intent, "")
        if not intent_prompt:
            logger.warning(f"未找到意图 '{intent}' 对应的指令，使用 base prompt")
            return self._base_prompt

        instruction = f"{self._base_prompt}\n\n{intent_prompt}"
        logger.debug(f"生成意图 '{intent}' 的系统指令，长度: {len(instruction)}")
        return instruction

    def get_full_instruction(
        self,
        intent: str,
        user_profile: str = "",
    ) -> str:
        """组装完整指令（base + intent-specific + user profile context）

        Args:
            intent: 用户意图
            user_profile: 用户画像文本（供 prompt 注入）

        Returns:
            完整系统指令字符串
        """
        base_instruction = self.get_system_instruction(intent)

        if user_profile:
            full_instruction = (
                f"{base_instruction}\n\n"
                f"## 当前用户信息\n{user_profile}"
            )
            logger.debug(f"注入用户画像，完整指令长度: {len(full_instruction)}")
            return full_instruction

        return base_instruction


# ── SkillRegistry ─────────────────────────────────────────────────────────────

class SkillRegistry:
    """Skill 注册与选择

    封装常见任务为可复用 Skill，每个 Skill 的 schema：
        {
            "description": str,          # 任务描述
            "required_tools": list[str], # 所需工具名称列表
            "intent": str,               # 对应的意图类型
            "prompt_template": str,      # 任务特定 prompt
            "few_shot_examples": list[   # 2-3 个 few-shot 示例
                {"user": str, "assistant": str}
            ]
        }

    内置 Skill：recipe_recommendation / nutrition_analysis / meal_planning
    """

    def __init__(self) -> None:
        """初始化 Skill 注册表，加载内置 Skill"""
        self._skills: Dict[str, Dict[str, Any]] = {}
        for name, skill in BUILTIN_SKILLS.items():
            self._skills[name] = skill
        logger.info(f"SkillRegistry 初始化，内置 Skill: {list(self._skills.keys())}")

    def register_skill(self, name: str, skill: Dict[str, Any]) -> None:
        """注册一个 Skill

        Args:
            name: Skill 名称（唯一标识，重复注册会覆盖）
            skill: Skill 定义字典
        """
        self._skills[name] = skill
        logger.info(f"注册 Skill: {name}")

    def get_skill(self, name: str) -> Optional[Dict[str, Any]]:
        """获取 Skill 定义

        Args:
            name: Skill 名称

        Returns:
            Skill 定义字典，不存在时返回 None
        """
        return self._skills.get(name)

    def select_skill(
        self,
        intent: str,
        params: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """根据意图和参数选择最匹配的 Skill

        优先匹配 intent 一致的 Skill；若有多个，按关键词规则细化选择。

        Args:
            intent: 用户意图
            params: 查询参数（extracted_params）

        Returns:
            最匹配的 Skill 字典，无匹配时返回 None
        """
        matched = [
            (name, skill)
            for name, skill in self._skills.items()
            if skill.get("intent") == intent
        ]

        if not matched:
            logger.debug(f"未找到意图 '{intent}' 对应的 Skill")
            return None

        # 特殊规则：若 params 含一周/计划关键词，优先选 meal_planning
        query_hint = str(params.get("query", "")).lower()
        if intent == "recipe_search" and any(
            kw in query_hint for kw in ("week", "一周", "计划", "plan")
        ):
            meal_skill = self._skills.get("meal_planning")
            if meal_skill:
                logger.debug("选择 Skill: meal_planning（一周计划关键词匹配）")
                return meal_skill

        selected_name, selected_skill = matched[0]
        logger.debug(f"选择 Skill: {selected_name}")
        return selected_skill

    def get_few_shot_examples(self, skill_name: str) -> List[Dict[str, str]]:
        """获取 Skill 的 few-shot 示例列表

        Args:
            skill_name: Skill 名称

        Returns:
            few-shot 示例列表 [{"user": "...", "assistant": "..."}]，不存在时返回 []
        """
        skill = self._skills.get(skill_name)
        if skill is None:
            return []
        return skill.get("few_shot_examples", [])

    def list_skills(self) -> List[str]:
        """列出所有已注册 Skill 名称

        Returns:
            Skill 名称列表
        """
        return list(self._skills.keys())
