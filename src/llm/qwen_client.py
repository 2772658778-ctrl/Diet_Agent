"""
Qwen LLM 客户端

初始化和配置 Qwen 大语言模型，提供重试机制和错误处理

Requirements:
- 4.1: 正确识别用户意图
- 8.1: LLM API 调用失败时重试最多 2 次
- 8.2: 重试失败后返回友好的错误提示
- 8.3: 记录详细的错误日志
"""

import time
from typing import Optional, Any, Dict
from langchain_openai import ChatOpenAI
from ..config import get_settings
from ..utils.logger import get_logger


logger = get_logger(__name__)


class QwenLLMClient:
    """
    Qwen LLM 客户端封装类
    
    提供 LLM 初始化、重试机制和错误处理功能
    """
    
    def __init__(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_retries: int = 2,
        retry_delay: float = 1.0
    ):
        """
        初始化 Qwen LLM 客户端
        
        Args:
            model: 模型名称，默认从配置读取
            temperature: 温度参数，默认从配置读取
            max_retries: 最大重试次数
            retry_delay: 重试延迟（秒）
        
        Requirements: 4.1, 8.1
        """
        self.settings = get_settings()
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # 使用提供的参数或配置中的默认值
        self.model = model or self.settings.llm_model
        self.temperature = temperature if temperature is not None else self.settings.llm_temperature
        
        logger.info(f"初始化 Qwen LLM 客户端: model={self.model}, temperature={self.temperature}")
        
        # 初始化 LLM
        self._llm = self._create_llm()
    
    def _create_llm(self) -> ChatOpenAI:
        """
        创建 LLM 实例
        
        使用 ChatOpenAI 兼容接口连接 DashScope
        
        Returns:
            ChatOpenAI 实例
        
        Requirements: 4.1
        """
        try:
            llm = ChatOpenAI(
                model=self.model,
                temperature=self.temperature,
                openai_api_key=self.settings.dashscope_api_key,
                openai_api_base=self.settings.dashscope_base_url,
                max_retries=0,  # 我们自己实现重试逻辑
                request_timeout=60,  # 60秒超时
            )
            logger.info("LLM 实例创建成功")
            return llm
        except Exception as e:
            logger.error(f"LLM 实例创建失败: {e}", exc_info=True)
            raise ValueError(f"无法创建 LLM 实例: {str(e)}")
    
    def get_llm(self) -> ChatOpenAI:
        """
        获取 LLM 实例
        
        Returns:
            ChatOpenAI 实例
        """
        return self._llm
    
    def invoke_with_retry(self, *args, **kwargs) -> Any:
        """
        带重试机制的 LLM 调用
        
        Args:
            *args: 传递给 LLM.invoke 的位置参数
            **kwargs: 传递给 LLM.invoke 的关键字参数
        
        Returns:
            LLM 响应
        
        Raises:
            Exception: 所有重试失败后抛出异常
        
        Requirements: 8.1, 8.2, 8.3
        """
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                logger.info(f"LLM 调用尝试 {attempt + 1}/{self.max_retries + 1}")
                
                # 调用 LLM
                response = self._llm.invoke(*args, **kwargs)
                
                logger.info(f"LLM 调用成功 (尝试 {attempt + 1})")
                return response
                
            except Exception as e:
                last_error = e
                logger.warning(
                    f"LLM 调用失败 (尝试 {attempt + 1}/{self.max_retries + 1}): {str(e)}",
                    exc_info=True
                )
                
                # 如果还有重试机会，等待后重试
                if attempt < self.max_retries:
                    logger.info(f"等待 {self.retry_delay} 秒后重试...")
                    time.sleep(self.retry_delay)
                else:
                    # 所有重试都失败了
                    logger.error(
                        f"LLM 调用失败，已达到最大重试次数 ({self.max_retries + 1})",
                        exc_info=True
                    )
        
        # 所有重试都失败，抛出异常
        error_msg = f"LLM 调用失败: {str(last_error)}"
        logger.error(error_msg)
        raise Exception(error_msg)


# 全局 LLM 客户端实例（单例模式）
_llm_client: Optional[QwenLLMClient] = None


def get_llm_client(force_reload: bool = False) -> QwenLLMClient:
    """
    获取 LLM 客户端实例（单例模式）
    
    Args:
        force_reload: 是否强制重新创建实例
    
    Returns:
        QwenLLMClient 实例
    
    Requirements: 4.1, 8.1
    """
    global _llm_client
    
    if _llm_client is None or force_reload:
        logger.info("创建新的 LLM 客户端实例")
        _llm_client = QwenLLMClient()
    
    return _llm_client


def get_llm(force_reload: bool = False) -> ChatOpenAI:
    """
    获取 LLM 实例（便捷函数）
    
    Args:
        force_reload: 是否强制重新创建实例
    
    Returns:
        ChatOpenAI 实例
    
    Requirements: 4.1
    """
    client = get_llm_client(force_reload=force_reload)
    return client.get_llm()


def test_llm_connection() -> bool:
    """
    测试 LLM 连接是否正常
    
    Returns:
        bool: 连接是否成功
    
    Requirements: 8.1, 8.2, 8.3
    """
    try:
        logger.info("测试 LLM 连接...")
        
        client = get_llm_client()
        llm = client.get_llm()
        
        # 发送简单的测试消息
        response = llm.invoke("你好")
        
        logger.info("LLM 连接测试成功")
        logger.debug(f"测试响应: {response}")
        
        return True
        
    except Exception as e:
        logger.error(f"LLM 连接测试失败: {e}", exc_info=True)
        return False
