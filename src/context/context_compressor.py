"""
上下文压缩器
减少多轮对话的 token 消耗，保留关键信息
"""

from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import json
import tiktoken

from ..config import get_settings
from ..utils.token_usage import add_token_usage, estimate_text_token_usage, extract_token_usage
from ..utils.logger import get_logger


logger = get_logger(__name__)
settings = get_settings()


def _resolve_context_compressor_model(model: Optional[str]) -> str:
    return str(model or settings.llm_model or settings.openai_model or "gpt-3.5-turbo")


def _build_context_compressor_llm(model: str) -> ChatOpenAI:
    openai_api_base = settings.dashscope_openai_base_url
    if settings.llm_backend != "dashscope_openai_compat":
        openai_api_base = settings.openai_base_url or settings.dashscope_openai_base_url
    return ChatOpenAI(
        model=model,
        temperature=0,
        openai_api_key=settings.openai_api_key or settings.dashscope_api_key,
        openai_api_base=openai_api_base,
        max_retries=0,
        request_timeout=settings.llm_request_timeout_seconds,
    )


def _resolve_token_encoding(model: str):
    try:
        return tiktoken.encoding_for_model(model)
    except Exception:
        logger.warning(f"ContextCompressor 未识别 token 模型 {model}，回退到 cl100k_base")
        return tiktoken.get_encoding("cl100k_base")


class ContextCompressor:
    """上下文压缩器"""
    
    def __init__(
        self,
        window_size: int = 3,
        max_tokens: int = 2000,
        model: Optional[str] = None
    ):
        """
        初始化上下文压缩器
        
        Args:
            window_size: 滑动窗口大小（保留最近 N 轮对话）
            max_tokens: 最大 token 数
            model: LLM 模型（用于摘要）
        """
        self.window_size = window_size
        self.max_tokens = max_tokens
        model_name = _resolve_context_compressor_model(model)
        self.llm = _build_context_compressor_llm(model_name)
        self.encoding = _resolve_token_encoding(model_name)
        
        logger.info(f"初始化上下文压缩器: window_size={window_size}, max_tokens={max_tokens}")
    
    def compress(
        self,
        messages: List[Dict[str, str]],
        strategy: str = "hybrid"
    ) -> List[Dict[str, str]]:
        """
        压缩对话历史
        
        Args:
            messages: 对话消息列表 [{"role": "user/assistant", "content": "..."}]
            strategy: 压缩策略
                - "window": 滑动窗口
                - "summary": 摘要压缩
                - "hybrid": 混合策略（推荐）
                - "extract": 关键信息提取
        
        Returns:
            压缩后的消息列表
        """
        if not messages:
            return []
        
        # 计算当前 token 数
        current_tokens = self._count_tokens(messages)
        logger.info(f"当前对话 token 数: {current_tokens}")
        
        if current_tokens <= self.max_tokens:
            logger.info("无需压缩")
            return messages
        
        # 根据策略压缩
        if strategy == "window":
            compressed = self._window_compress(messages)
        elif strategy == "summary":
            compressed = self._summary_compress(messages)
        elif strategy == "extract":
            compressed = self._extract_compress(messages)
        else:  # hybrid
            compressed = self._hybrid_compress(messages)
        
        compressed_tokens = self._count_tokens(compressed)
        compression_rate = (1 - compressed_tokens / current_tokens) * 100
        
        logger.info(
            f"压缩完成: {current_tokens} -> {compressed_tokens} tokens "
            f"(压缩率: {compression_rate:.1f}%)"
        )
        
        return compressed
    
    def _window_compress(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """滑动窗口压缩：保留最近 N 轮对话"""
        # 保留最近的对话
        recent_messages = messages[-(self.window_size * 2):]
        
        logger.info(f"滑动窗口压缩: 保留最近 {len(recent_messages)} 条消息")
        return recent_messages
    
    def _summary_compress(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """摘要压缩：对旧对话生成摘要"""
        if len(messages) <= self.window_size * 2:
            return messages
        
        # 1. 保留最近的对话
        recent_messages = messages[-(self.window_size * 2):]
        
        # 2. 压缩旧对话
        old_messages = messages[:-(self.window_size * 2)]
        summary = self._generate_summary(old_messages)
        
        # 3. 组合
        compressed = [
            {"role": "system", "content": f"历史对话摘要：\n{summary}"}
        ] + recent_messages
        
        logger.info(f"摘要压缩: 旧对话 {len(old_messages)} 条 -> 摘要")
        return compressed
    
    def _extract_compress(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """关键信息提取：提取结构化信息"""
        # 提取关键信息
        key_info = self._extract_key_info(messages)
        
        # 保留最近的对话
        recent_messages = messages[-(self.window_size * 2):]
        
        # 组合
        compressed = [
            {"role": "system", "content": f"用户信息：\n{json.dumps(key_info, ensure_ascii=False, indent=2)}"}
        ] + recent_messages
        
        logger.info(f"关键信息提取: 提取 {len(key_info)} 个字段")
        return compressed
    
    def _hybrid_compress(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """混合策略：滑动窗口 + 摘要 + 关键信息提取"""
        if len(messages) <= self.window_size * 2:
            return messages
        
        # 1. 提取关键信息
        key_info = self._extract_key_info(messages)
        
        # 2. 保留最近的对话
        recent_messages = messages[-(self.window_size * 2):]
        
        # 3. 对中间对话生成摘要
        if len(messages) > self.window_size * 4:
            middle_messages = messages[self.window_size * 2:-(self.window_size * 2)]
            summary = self._generate_summary(middle_messages)
            
            compressed = [
                {"role": "system", "content": f"用户信息：\n{json.dumps(key_info, ensure_ascii=False, indent=2)}"},
                {"role": "system", "content": f"历史对话摘要：\n{summary}"}
            ] + recent_messages
        else:
            compressed = [
                {"role": "system", "content": f"用户信息：\n{json.dumps(key_info, ensure_ascii=False, indent=2)}"}
            ] + recent_messages
        
        logger.info(f"混合压缩: 关键信息 + 摘要 + 最近 {len(recent_messages)} 条消息")
        return compressed
    
    def _generate_summary(self, messages: List[Dict[str, str]]) -> str:
        """生成对话摘要"""
        conversation = "\n".join([
            f"{m['role']}: {m['content']}" for m in messages
        ])
        
        prompt = f"""
请总结以下对话的关键信息，包括：
1. 用户的健康目标和偏好（口味、菜系、食材）
2. 用户提到的约束条件（时间、预算、设备等）
3. 已推荐的食谱和用户反馈
4. 其他重要信息

对话内容：
{conversation}

请用简洁的语言总结（100字以内）：
"""
        
        try:
            response = self.llm.invoke(prompt)
            token_usage = extract_token_usage(response) or estimate_text_token_usage(prompt, response.content, self.llm.model_name)
            add_token_usage(token_usage)
            summary = response.content.strip()
            logger.info(f"生成摘要: {len(summary)} 字符")
            return summary
        except Exception as e:
            logger.error(f"生成摘要失败: {e}")
            return "对话历史摘要生成失败"
    
    def _extract_key_info(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """提取关键信息"""
        conversation = "\n".join([
            f"{m['role']}: {m['content']}" for m in messages
        ])
        
        prompt = f"""
从对话中提取结构化信息，以 JSON 格式输出：

对话内容：
{conversation}

请提取以下信息（如果对话中没有提到，则设为 null）：
{{
    "health_goal": "用户的健康目标（减肥/增肌/养生/null）",
    "preferences": {{
        "tastes": ["口味偏好"],
        "cuisines": ["菜系偏好"],
        "ingredients": ["食材偏好"]
    }},
    "dislikes": {{
        "tastes": ["不喜欢的口味"],
        "ingredients": ["不喜欢的食材"]
    }},
    "constraints": {{
        "time_limit": 时间限制（分钟，null 如果没有）,
        "available_ingredients": ["已有食材"],
        "budget": 预算（元，null 如果没有）
    }},
    "recommended_recipes": ["已推荐的食谱名称"],
    "feedback": "用户对推荐的反馈"
}}

只输出 JSON，不要其他内容：
"""
        
        try:
            response = self.llm.invoke(prompt)
            token_usage = extract_token_usage(response) or estimate_text_token_usage(prompt, response.content, self.llm.model_name)
            add_token_usage(token_usage)
            key_info = json.loads(response.content.strip())
            logger.info(f"提取关键信息: {len(key_info)} 个字段")
            return key_info
        except Exception as e:
            logger.error(f"提取关键信息失败: {e}")
            return {}
    
    def _count_tokens(self, messages: List[Dict[str, str]]) -> int:
        """计算消息的 token 数"""
        total_tokens = 0
        for message in messages:
            # 每条消息的基础 token（role + 格式）
            total_tokens += 4
            # 内容 token
            total_tokens += len(self.encoding.encode(message.get("content", "")))
        
        # 额外的格式 token
        total_tokens += 2
        
        return total_tokens


class AdaptiveContextCompressor(ContextCompressor):
    """自适应上下文压缩器"""
    
    def compress(
        self,
        messages: List[Dict[str, str]],
        strategy: str = "adaptive"
    ) -> List[Dict[str, str]]:
        """
        自适应压缩：根据对话长度自动选择策略
        
        Args:
            messages: 对话消息列表
            strategy: 压缩策略（"adaptive" 为自适应）
        
        Returns:
            压缩后的消息列表
        """
        if strategy != "adaptive":
            return super().compress(messages, strategy)
        
        # 计算当前 token 数
        current_tokens = self._count_tokens(messages)
        
        if current_tokens <= self.max_tokens:
            return messages
        
        # 根据对话长度选择策略
        num_messages = len(messages)
        
        if num_messages <= self.window_size * 2:
            # 短对话：直接保留
            return messages
        elif num_messages <= self.window_size * 4:
            # 中等对话：滑动窗口
            return self._window_compress(messages)
        elif num_messages <= self.window_size * 8:
            # 长对话：摘要压缩
            return self._summary_compress(messages)
        else:
            # 超长对话：混合策略
            return self._hybrid_compress(messages)


# 全局压缩器实例
_compressor: Optional[ContextCompressor] = None


def get_context_compressor(
    window_size: int = 3,
    max_tokens: int = 2000,
    adaptive: bool = True
) -> ContextCompressor:
    """获取上下文压缩器实例（单例）"""
    global _compressor
    
    if _compressor is None:
        if adaptive:
            _compressor = AdaptiveContextCompressor(
                window_size=window_size,
                max_tokens=max_tokens
            )
        else:
            _compressor = ContextCompressor(
                window_size=window_size,
                max_tokens=max_tokens
            )
    
    return _compressor
