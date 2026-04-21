"""
错误处理工具模块

提供统一的错误处理、分类和恢复机制

Requirements:
- 8.1: LLM API 调用失败时重试
- 8.2: 重试失败后返回友好错误提示
- 8.3: 向量数据库连接失败时记录错误并尝试重新连接
- 8.4: 工具执行失败时返回包含错误信息的 JSON 响应
- 8.5: 遇到任何错误时记录详细的错误日志
"""

import json
from typing import Optional, Dict, Any, Callable
from enum import Enum
from functools import wraps
from ..utils.logger import get_logger


logger = get_logger(__name__)


class ErrorType(Enum):
    """错误类型枚举"""
    VALIDATION_ERROR = "validation_error"  # 数据验证错误
    API_ERROR = "api_error"  # API 调用错误
    DATABASE_ERROR = "database_error"  # 数据库错误
    NETWORK_ERROR = "network_error"  # 网络错误
    TIMEOUT_ERROR = "timeout_error"  # 超时错误
    RATE_LIMIT_ERROR = "rate_limit_error"  # 速率限制错误
    SYSTEM_ERROR = "system_error"  # 系统错误
    UNKNOWN_ERROR = "unknown_error"  # 未知错误


class ErrorSeverity(Enum):
    """错误严重程度"""
    LOW = "low"  # 低 - 可以忽略或降级处理
    MEDIUM = "medium"  # 中 - 需要记录但不影响主流程
    HIGH = "high"  # 高 - 影响功能但系统可继续运行
    CRITICAL = "critical"  # 严重 - 系统无法继续运行


def classify_error(error: Exception) -> ErrorType:
    """
    分类错误类型
    
    Args:
        error: 异常对象
    
    Returns:
        错误类型
    
    Requirements: 8.5
    """
    error_str = str(error).lower()
    error_type_name = type(error).__name__.lower()
    
    # 速率限制（优先检查，因为可能包含其他关键词）
    if "rate limit" in error_str or "too many requests" in error_str:
        return ErrorType.RATE_LIMIT_ERROR
    
    # 超时错误（优先检查）
    if "timeout" in error_str or "timeout" in error_type_name:
        return ErrorType.TIMEOUT_ERROR
    
    # 数据库错误（优先检查，因为可能包含 connection 等关键词）
    if any(keyword in error_str for keyword in ["database", "chroma", "vector", "collection"]):
        return ErrorType.DATABASE_ERROR
    
    # API 错误
    if "api" in error_str or "api" in error_type_name:
        return ErrorType.API_ERROR
    
    # 网络错误
    if any(keyword in error_str for keyword in ["connection", "network", "unreachable"]):
        return ErrorType.NETWORK_ERROR
    
    # 验证错误
    if any(keyword in error_type_name for keyword in ["value", "type", "validation", "assertion"]):
        return ErrorType.VALIDATION_ERROR
    
    # 系统错误
    if any(keyword in error_type_name for keyword in ["system", "os", "io", "permission"]):
        return ErrorType.SYSTEM_ERROR
    
    return ErrorType.UNKNOWN_ERROR


def get_user_friendly_message(error_type: ErrorType, original_error: str = "") -> str:
    """
    获取用户友好的错误消息
    
    Args:
        error_type: 错误类型
        original_error: 原始错误消息
    
    Returns:
        用户友好的错误消息
    
    Requirements: 8.2
    """
    messages = {
        ErrorType.VALIDATION_ERROR: "输入数据格式不正确，请检查后重试",
        ErrorType.API_ERROR: "连接服务时遇到问题，请稍后再试",
        ErrorType.DATABASE_ERROR: "数据库暂时不可用，请稍后再试",
        ErrorType.NETWORK_ERROR: "网络连接失败，请检查网络后重试",
        ErrorType.TIMEOUT_ERROR: "请求超时，请稍后再试",
        ErrorType.RATE_LIMIT_ERROR: "请求过于频繁，请稍等片刻后再试",
        ErrorType.SYSTEM_ERROR: "系统遇到问题，请联系管理员",
        ErrorType.UNKNOWN_ERROR: "遇到未知错误，请稍后再试"
    }
    
    base_message = messages.get(error_type, messages[ErrorType.UNKNOWN_ERROR])
    
    # 对于某些错误类型，可以添加更多细节
    if error_type == ErrorType.VALIDATION_ERROR and original_error:
        # 提取有用的验证错误信息
        if "missing" in original_error.lower():
            base_message = "缺少必需的信息，请补充完整"
        elif "invalid" in original_error.lower():
            base_message = "输入信息无效，请检查格式"
    
    return base_message


def create_error_response(
    error: Exception,
    context: Optional[Dict[str, Any]] = None,
    include_details: bool = False
) -> str:
    """
    创建标准化的错误响应（JSON 格式）
    
    Args:
        error: 异常对象
        context: 错误上下文信息
        include_details: 是否包含详细错误信息（仅用于调试）
    
    Returns:
        JSON 格式的错误响应
    
    Requirements: 8.4
    """
    error_type = classify_error(error)
    user_message = get_user_friendly_message(error_type, str(error))
    
    response = {
        "error": error_type.value,
        "message": user_message
    }
    
    # 仅在调试模式下包含详细信息
    if include_details:
        response["details"] = str(error)
        if context:
            response["context"] = context
    
    return json.dumps(response, ensure_ascii=False)


def retry_on_error(
    max_retries: int = 2,
    delay: float = 1.0,
    error_types: Optional[tuple] = None
):
    """
    装饰器：在特定错误时自动重试
    
    Args:
        max_retries: 最大重试次数
        delay: 重试延迟（秒）
        error_types: 需要重试的错误类型元组，None 表示所有错误
    
    Requirements: 8.1
    
    Example:
        @retry_on_error(max_retries=3, delay=1.0)
        def call_api():
            ...
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            import time
            last_error = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    
                    # 检查是否需要重试此类型的错误
                    if error_types and not isinstance(e, error_types):
                        raise
                    
                    logger.warning(
                        f"{func.__name__} 失败 (尝试 {attempt + 1}/{max_retries + 1}): {e}"
                    )
                    
                    if attempt < max_retries:
                        logger.info(f"等待 {delay} 秒后重试...")
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"{func.__name__} 失败，已达到最大重试次数",
                            exc_info=True
                        )
            
            raise last_error
        
        return wrapper
    return decorator


def handle_tool_error(func: Callable):
    """
    装饰器：统一处理工具函数的错误
    
    自动捕获异常并返回标准化的 JSON 错误响应
    
    Requirements: 8.4, 8.5
    
    Example:
        @handle_tool_error
        @tool
        def my_tool(param: str) -> str:
            ...
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(
                f"工具 {func.__name__} 执行失败: {e}",
                exc_info=True
            )
            return create_error_response(
                error=e,
                context={"tool": func.__name__, "args": str(args)[:100]}
            )
    
    return wrapper


def validate_input(
    value: Any,
    value_name: str,
    required: bool = True,
    value_type: Optional[type] = None,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None
) -> None:
    """
    验证输入参数
    
    Args:
        value: 要验证的值
        value_name: 参数名称
        required: 是否必需
        value_type: 期望的类型
        min_length: 最小长度（字符串/列表）
        max_length: 最大长度（字符串/列表）
        min_value: 最小值（数字）
        max_value: 最大值（数字）
    
    Raises:
        ValueError: 验证失败时抛出
    
    Requirements: 8.4
    """
    # 检查是否为空
    if required and (value is None or (isinstance(value, str) and not value.strip())):
        raise ValueError(f"{value_name} 不能为空")
    
    if value is None:
        return
    
    # 检查类型
    if value_type and not isinstance(value, value_type):
        raise ValueError(f"{value_name} 必须是 {value_type.__name__} 类型")
    
    # 检查长度
    if isinstance(value, (str, list)):
        if min_length is not None and len(value) < min_length:
            raise ValueError(f"{value_name} 长度不能小于 {min_length}")
        if max_length is not None and len(value) > max_length:
            raise ValueError(f"{value_name} 长度不能大于 {max_length}")
    
    # 检查数值范围
    if isinstance(value, (int, float)):
        if min_value is not None and value < min_value:
            raise ValueError(f"{value_name} 不能小于 {min_value}")
        if max_value is not None and value > max_value:
            raise ValueError(f"{value_name} 不能大于 {max_value}")


def safe_execute(
    func: Callable,
    *args,
    fallback_value: Any = None,
    log_error: bool = True,
    **kwargs
) -> Any:
    """
    安全执行函数，捕获异常并返回降级值
    
    Args:
        func: 要执行的函数
        *args: 函数参数
        fallback_value: 发生错误时的降级返回值
        log_error: 是否记录错误日志
        **kwargs: 函数关键字参数
    
    Returns:
        函数返回值或降级值
    
    Requirements: 8.2, 8.5
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if log_error:
            logger.error(f"执行 {func.__name__} 失败: {e}", exc_info=True)
        return fallback_value
