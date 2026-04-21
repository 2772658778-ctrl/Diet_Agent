"""
Graph 专用 LLM 客户端

封装 dashscope.MultiModalConversation API，提供与 LangChain 兼容的接口。
qwen3.5-35b-a3b 仅支持 MultiModalConversation API（不支持 Generation / OpenAI-compatible endpoint）。

主要类:
- ChatDashScopeMultiModal: 自定义 BaseChatModel，支持 with_structured_output()

主要函数:
- get_graph_llm(): 获取 Graph 专用 LLM 实例（单例）
"""

import json
import re
from typing import Any, Iterator, List, Optional, Type, Union

import dashscope
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from ..config import get_settings
from ..utils.token_usage import add_token_usage, estimate_chat_token_usage, extract_token_usage
from ..utils.logger import get_logger


logger = get_logger(__name__)


def _coerce_input_messages(value: Any) -> List[BaseMessage]:
    if isinstance(value, list) and all(isinstance(item, BaseMessage) for item in value):
        return value
    if isinstance(value, BaseMessage):
        return [value]
    to_messages = getattr(value, "to_messages", None)
    if callable(to_messages):
        try:
            messages = to_messages()
        except Exception:
            messages = []
        if isinstance(messages, list) and all(isinstance(item, BaseMessage) for item in messages):
            return messages
    return [HumanMessage(content=str(value))]


def _to_dashscope_messages(messages: List[BaseMessage]) -> List[dict]:
    """将 LangChain 消息格式转换为 DashScope MultiModal 格式

    Args:
        messages: LangChain 消息列表

    Returns:
        DashScope MultiModalConversation 格式的消息列表
    """
    result = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            role = "user"
        elif isinstance(msg, AIMessage):
            role = "assistant"
        elif isinstance(msg, SystemMessage):
            role = "system"
        else:
            role = "user"

        # MultiModal API 要求 content 为列表格式
        result.append({
            "role": role,
            "content": [{"text": msg.content}]
        })
    return result


def _extract_json_from_text(text: str) -> str:
    """从 LLM 回复文本中提取 JSON 字符串

    支持 ```json ... ``` 代码块和裸 JSON 两种格式。

    Args:
        text: LLM 原始回复文本

    Returns:
        提取的 JSON 字符串
    """
    # 尝试匹配 ```json ... ``` 代码块
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        return match.group(1).strip()

    # 尝试匹配裸 JSON 对象
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        return match.group(0).strip()

    return text.strip()


class _StructuredOutputRunnable(Runnable):
    """with_structured_output 的可运行包装器

    通过 JSON prompt 注入 + 输出解析实现结构化输出，
    无需模型原生支持 function calling。
    """

    def __init__(self, llm: BaseChatModel, schema: Type[BaseModel]):
        self.llm = llm
        self.schema = schema

    def invoke(self, messages: List[BaseMessage], config=None, **kwargs) -> BaseModel:
        """调用 LLM 并解析为 Pydantic 模型

        Args:
            messages: 输入消息列表
            config: 运行配置（未使用）

        Returns:
            解析后的 Pydantic 模型实例
        """
        # 只展示 properties（字段名+说明），避免 LLM 把 schema 元字段当成答案
        full_schema = self.schema.model_json_schema()
        props = full_schema.get("properties", {})
        required = full_schema.get("required", [])

        # 构造字段说明（每行：字段名 | 类型/枚举 | 说明）
        lines = []
        for fname, finfo in props.items():
            req_mark = "*必填*" if fname in required else "可选"
            ftype = finfo.get("type", finfo.get("enum", ""))
            if isinstance(ftype, list):
                ftype = f"enum{ftype}"
            fdesc = finfo.get("description", "")
            lines.append(f"  - {fname} ({req_mark}, {ftype}): {fdesc}")
        fields_desc = "\n".join(lines)

        # 构造空模板，帮助 LLM 理解 "填值而非描述"
        template = {
            fname: (
                finfo.get("default", f"<填写{fname}>")
                if fname not in required
                else f"<填写{fname}>"
            )
            for fname, finfo in props.items()
        }
        template_json = json.dumps(template, ensure_ascii=False)

        # 注入 JSON 格式要求到系统提示
        json_instruction = (
            f"\n\n请将你的判断填入以下 JSON 格式并直接输出（不要输出字段说明，只输出填好值的 JSON）：\n"
            f"字段说明：\n{fields_desc}\n\n"
            f"输出模板（把 <填写...> 替换为实际值）：\n"
            f"```json\n{template_json}\n```\n"
            f"只输出 JSON，不要解释，不要输出 schema 或字段描述。"
        )

        # 找到或创建 SystemMessage，注入 schema 说明
        enriched = []
        has_system = False
        for msg in messages:
            if isinstance(msg, SystemMessage):
                enriched.append(SystemMessage(content=msg.content + json_instruction))
                has_system = True
            else:
                enriched.append(msg)

        if not has_system:
            enriched.insert(0, SystemMessage(content="你是一个助手。" + json_instruction))

        # 调用 LLM
        response = self.llm.invoke(enriched)
        raw_text = response.content

        # 解析 JSON
        json_str = _extract_json_from_text(raw_text)
        parsed = json.loads(json_str)
        return self.schema(**parsed)


class ChatDashScopeMultiModal(BaseChatModel):
    """基于 dashscope.MultiModalConversation 的 LangChain 兼容 LLM

    专为 qwen3.5-35b-a3b 等仅支持 MultiModal API 的模型设计。
    支持标准 LangChain 消息格式和 with_structured_output()。

    Attributes:
        model: 模型名称
        api_key: DashScope API Key
        temperature: 生成温度
        max_tokens: 最大 token 数
    """

    model: str = Field(default="qwen3.5-35b-a3b", description="模型名称")
    api_key: str = Field(default="", description="DashScope API Key")
    temperature: float = Field(default=0.1, description="生成温度")
    max_tokens: int = Field(default=2048, description="最大输出 token 数")
    enable_thinking: bool = Field(default=False, description="是否启用 Qwen3 thinking 模式（默认关闭以提速）")

    @property
    def _llm_type(self) -> str:
        return "dashscope_multimodal"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager=None,
        **kwargs,
    ) -> ChatResult:
        """调用 dashscope.MultiModalConversation API

        Args:
            messages: LangChain 消息列表
            stop: 停止词（暂不使用）
            run_manager: 运行管理器（暂不使用）

        Returns:
            ChatResult 包含生成的 AIMessage
        """
        dashscope_messages = _to_dashscope_messages(messages)

        logger.debug(f"调用 DashScope MultiModal API: model={self.model}, messages={len(messages)}")

        response = dashscope.MultiModalConversation.call(
            api_key=self.api_key,
            model=self.model,
            messages=dashscope_messages,
            enable_thinking=self.enable_thinking,
        )

        if response.status_code != 200:
            raise ValueError(
                f"DashScope API 调用失败: {response.status_code} "
                f"[{response.code}] {response.message}"
            )

        content_parts = response.output.choices[0].message.content
        # MultiModal 返回 list of dicts
        if isinstance(content_parts, list):
            text = "".join(
                part.get("text", "") for part in content_parts
                if isinstance(part, dict)
            )
        else:
            text = str(content_parts)

        token_usage = extract_token_usage(response) or estimate_chat_token_usage(messages, text, self.model)
        if token_usage:
            ai_message = AIMessage(content=text, response_metadata={"token_usage": token_usage})
        else:
            ai_message = AIMessage(content=text)
        return ChatResult(generations=[ChatGeneration(message=ai_message)])

    def invoke(self, input, config=None, *, stop=None, **kwargs):
        response = super().invoke(input, config=config, stop=stop, **kwargs)
        token_usage = extract_token_usage(response) or estimate_chat_token_usage(
            _coerce_input_messages(input),
            getattr(response, "content", ""),
            self.model,
        )
        if token_usage:
            add_token_usage(token_usage)
        return response

    def with_structured_output(
        self,
        schema: Union[Type[BaseModel], dict],
        **kwargs,
    ) -> Runnable:
        """返回支持 Pydantic 结构化输出的 Runnable

        使用 JSON prompt 注入 + 输出解析实现，无需模型原生函数调用支持。

        Args:
            schema: Pydantic BaseModel 类

        Returns:
            _StructuredOutputRunnable 实例
        """
        if isinstance(schema, dict):
            raise NotImplementedError("暂不支持 dict schema，请传入 Pydantic BaseModel 类")
        return _StructuredOutputRunnable(llm=self, schema=schema)


# ── Phase 5: OpenAI 兼容端点 LLM ──────────────────────────────────────────────


def _to_openai_messages(messages: List[BaseMessage]) -> List[dict]:
    """将 LangChain 消息格式转换为 OpenAI Chat Completions 格式

    Args:
        messages: LangChain 消息列表

    Returns:
        OpenAI Chat Completions 格式的消息列表
    """
    result = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            role = "user"
        elif isinstance(msg, AIMessage):
            role = "assistant"
        elif isinstance(msg, SystemMessage):
            role = "system"
        else:
            role = "user"
        result.append({"role": role, "content": msg.content})
    return result


class ChatDashScopeOpenAICompat(BaseChatModel):
    """基于 DashScope OpenAI 兼容端点的 LLM

    相比 ChatDashScopeMultiModal:
    - 响应速度快 ~5x（标准 HTTP API vs MultiModal API）
    - 支持原生流式输出（SSE chunk）
    - 支持 LangChain SQLiteCache

    Attributes:
        model: 模型名称
        api_key: DashScope API Key
        base_url: OpenAI 兼容端点地址
        temperature: 生成温度
        max_tokens: 最大输出 token 数
        streaming: 是否启用流式
    """

    model: str = Field(default="qwen3.5-flash", description="模型名称")
    api_key: str = Field(default="", description="DashScope API Key")
    base_url: str = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        description="OpenAI 兼容端点地址",
    )
    temperature: float = Field(default=0.1, description="生成温度")
    max_tokens: int = Field(default=2048, description="最大输出 token 数")
    streaming: bool = Field(default=False, description="是否启用流式")
    enable_thinking: bool = Field(default=False, description="是否启用 Qwen3 thinking 模式（默认关闭以提速）")
    request_timeout_seconds: float = Field(default=120.0, description="请求超时时间（秒）")
    connect_timeout_seconds: float = Field(default=20.0, description="连接超时时间（秒）")

    @property
    def _llm_type(self) -> str:
        return "dashscope_openai_compat"

    def _get_client(self):
        """获取 OpenAI SDK 客户端实例。

        Returns:
            openai.OpenAI 客户端
        """
        import httpx
        from openai import OpenAI

        return OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            http_client=httpx.Client(
                verify=False,
                timeout=httpx.Timeout(
                    timeout=self.request_timeout_seconds,
                    connect=self.connect_timeout_seconds,
                ),
            ),
        )

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager=None,
        **kwargs,
    ) -> ChatResult:
        """通过 OpenAI 兼容 API 调用 DashScope

        Args:
            messages: LangChain 消息列表
            stop: 停止词（可选）
            run_manager: 运行管理器（可选）

        Returns:
            ChatResult 包含生成的 AIMessage
        """
        client = self._get_client()
        openai_messages = _to_openai_messages(messages)

        logger.debug(
            f"调用 DashScope OpenAI compat API: model={self.model}, "
            f"messages={len(messages)}"
        )

        response = client.chat.completions.create(
            model=self.model,
            messages=openai_messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stop=stop,
            extra_body={"enable_thinking": self.enable_thinking},
        )

        text = response.choices[0].message.content or ""
        token_usage = extract_token_usage(response) or estimate_chat_token_usage(messages, text, self.model)
        if token_usage:
            ai_message = AIMessage(content=text, response_metadata={"token_usage": token_usage})
        else:
            ai_message = AIMessage(content=text)
        return ChatResult(generations=[ChatGeneration(message=ai_message)])

    def invoke(self, input, config=None, *, stop=None, **kwargs):
        response = super().invoke(input, config=config, stop=stop, **kwargs)
        token_usage = extract_token_usage(response) or estimate_chat_token_usage(
            _coerce_input_messages(input),
            getattr(response, "content", ""),
            self.model,
        )
        if token_usage:
            add_token_usage(token_usage)
        return response

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager=None,
        **kwargs,
    ) -> Iterator[ChatGenerationChunk]:
        """流式输出，逐 chunk 返回

        通过 OpenAI 兼容 API 的 stream=True 参数获取 SSE 流，
        逐 token 产出 ChatGenerationChunk。

        Args:
            messages: LangChain 消息列表
            stop: 停止词（可选）
            run_manager: 运行管理器（可选）

        Yields:
            ChatGenerationChunk 包含 AIMessageChunk
        """
        client = self._get_client()
        openai_messages = _to_openai_messages(messages)

        logger.debug(
            f"调用 DashScope OpenAI compat API (stream): model={self.model}"
        )

        response = client.chat.completions.create(
            model=self.model,
            messages=openai_messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stop=stop,
            stream=True,
            extra_body={"enable_thinking": self.enable_thinking},
        )

        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                chat_chunk = ChatGenerationChunk(
                    message=AIMessageChunk(content=content)
                )
                if run_manager:
                    run_manager.on_llm_new_token(content)
                yield chat_chunk

    def with_structured_output(
        self,
        schema: Union[Type[BaseModel], dict],
        **kwargs,
    ) -> Runnable:
        """返回支持 Pydantic 结构化输出的 Runnable

        复用 _StructuredOutputRunnable（JSON prompt 注入 + 输出解析）。

        Args:
            schema: Pydantic BaseModel 类

        Returns:
            _StructuredOutputRunnable 实例
        """
        if isinstance(schema, dict):
            raise NotImplementedError("暂不支持 dict schema，请传入 Pydantic BaseModel 类")
        return _StructuredOutputRunnable(llm=self, schema=schema)


# 模块级单例
_graph_llm: Optional[BaseChatModel] = None


def get_graph_llm(force_reload: bool = False) -> BaseChatModel:
    """获取 Graph 专用 LLM 实例（单例）

    Phase 5: 根据配置 llm_backend 选择后端：
    - dashscope_multimodal: 原有 MultiModalConversation API
    - dashscope_openai_compat: OpenAI 兼容端点（更快，支持流式）

    Args:
        force_reload: 是否强制重新创建实例

    Returns:
        BaseChatModel 实例
    """
    global _graph_llm
    if _graph_llm is None or force_reload:
        settings = get_settings()

        # Phase 5: 根据配置选择后端
        if settings.llm_backend == "dashscope_openai_compat":
            _graph_llm = ChatDashScopeOpenAICompat(
                model=settings.llm_model,
                api_key=settings.dashscope_api_key,
                base_url=settings.dashscope_openai_base_url,
                temperature=settings.llm_temperature,
                enable_thinking=getattr(settings, "llm_enable_thinking", False),
                request_timeout_seconds=settings.llm_request_timeout_seconds,
                connect_timeout_seconds=settings.llm_connect_timeout_seconds,
            )
            logger.info(
                f"Graph LLM 初始化完成 (OpenAI compat): model={settings.llm_model}"
            )
        else:
            _graph_llm = ChatDashScopeMultiModal(
                model=settings.llm_model,
                api_key=settings.dashscope_api_key,
                temperature=settings.llm_temperature,
                enable_thinking=getattr(settings, "llm_enable_thinking", False),
            )
            logger.info(
                f"Graph LLM 初始化完成 (MultiModal): model={settings.llm_model}"
            )

        # Phase 5: LLM 缓存
        if settings.llm_cache_enabled:
            try:
                import os
                from langchain_community.cache import SQLiteCache
                import langchain_core

                cache_dir = os.path.dirname(settings.llm_cache_path)
                if cache_dir:
                    os.makedirs(cache_dir, exist_ok=True)

                langchain_core.globals.set_llm_cache(
                    SQLiteCache(database_path=settings.llm_cache_path)
                )
                logger.info(f"LLM 缓存已启用: {settings.llm_cache_path}")
            except Exception as e:
                logger.warning(f"LLM 缓存初始化失败，继续运行: {e}")

    return _graph_llm
