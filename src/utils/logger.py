"""
日志系统
配置日志格式和级别，同时输出到文件和控制台，实现敏感信息脱敏

Requirements:
- 11.1: 记录操作的开始和结束
- 11.2: 记录完整的错误堆栈
- 11.3: 记录请求和响应的关键信息
- 11.4: 同时输出到文件和控制台
- 12.2: 不记录 API 密钥或其他敏感信息
"""

import io
import logging
import os
import re
import sys
import ctypes
from logging.handlers import RotatingFileHandler
from typing import Optional, Any
from functools import wraps
import traceback


def mask_sensitive_info(message: str) -> str:
    """
    脱敏敏感信息
    
    Args:
        message: 原始日志消息
    
    Returns:
        脱敏后的消息
    
    Requirements: 12.2
    """
    # 脱敏 API 密钥（保留前4位和后4位）
    message = re.sub(
        r'(api[_-]?key["\s:=]+)([a-zA-Z0-9]{4})[a-zA-Z0-9]+([a-zA-Z0-9]{4})',
        r'\1\2****\3',
        message,
        flags=re.IGNORECASE
    )
    
    # 脱敏完整的 API 密钥（sk- 开头的 OpenAI 格式）
    message = re.sub(
        r'(sk-[a-zA-Z0-9]{4})[a-zA-Z0-9]+([a-zA-Z0-9]{4})',
        r'\1****\2',
        message
    )
    
    # 脱敏 DashScope API 密钥格式
    message = re.sub(
        r'([a-zA-Z0-9]{4})[a-zA-Z0-9]{20,}([a-zA-Z0-9]{4})',
        r'\1****\2',
        message
    )
    
    # 脱敏密码
    message = re.sub(
        r'(password["\s:=]+)[^\s,}]+',
        r'\1****',
        message,
        flags=re.IGNORECASE
    )
    
    # 脱敏 token
    message = re.sub(
        r'(token["\s:=]+)[^\s,}]+',
        r'\1****',
        message,
        flags=re.IGNORECASE
    )
    
    # 脱敏 Authorization header
    message = re.sub(
        r'(Authorization["\s:=]+Bearer\s+)([a-zA-Z0-9]{4})[a-zA-Z0-9]+([a-zA-Z0-9]{4})',
        r'\1\2****\3',
        message,
        flags=re.IGNORECASE
    )
    
    return message


class SensitiveInfoFilter(logging.Filter):
    """敏感信息过滤器
    
    Requirements: 12.2
    """
    
    def filter(self, record: logging.LogRecord) -> bool:
        """过滤日志记录，脱敏敏感信息"""
        record.msg = mask_sensitive_info(str(record.msg))
        if record.args:
            record.args = tuple(
                mask_sensitive_info(str(arg)) if isinstance(arg, str) else arg
                for arg in record.args
            )
        return True


def _reconfigure_text_stream(stream: Any, encoding: str = "utf-8") -> Any:
    if stream is None:
        return stream

    try:
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding=encoding, errors="replace")
            return stream
    except Exception:
        pass

    buffer = getattr(stream, "buffer", None)
    if buffer is None:
        return stream

    try:
        return io.TextIOWrapper(buffer, encoding=encoding, errors="replace", line_buffering=True)
    except Exception:
        return stream


def configure_console_utf8() -> tuple[Any, Any]:
    if getattr(configure_console_utf8, "_configured", False):
        return sys.stdout, sys.stderr

    if sys.platform == "win32":
        try:
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleOutputCP(65001)
            kernel32.SetConsoleCP(65001)
        except Exception:
            pass

    try:
        sys.stdout = _reconfigure_text_stream(sys.stdout, encoding="utf-8")
    except Exception:
        pass

    try:
        sys.stderr = _reconfigure_text_stream(sys.stderr, encoding="utf-8")
    except Exception:
        pass

    try:
        if hasattr(sys.stdin, "reconfigure"):
            sys.stdin.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    setattr(configure_console_utf8, "_configured", True)
    return sys.stdout, sys.stderr


def setup_logger(
    name: str = "diet_agent",
    log_file: Optional[str] = None,
    log_level: str = "INFO",
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    配置日志系统
    
    Args:
        name: Logger 名称
        log_file: 日志文件路径
        log_level: 日志级别
        max_bytes: 单个日志文件最大大小
        backup_count: 保留的日志文件数量
    
    Returns:
        配置好的 logger 实例
    
    Requirements: 11.4
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # 避免重复添加 handler
    if logger.handlers:
        return logger
    
    # 日志格式（包含文件名和行号，便于调试）
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 控制台 handler（强制 UTF-8，避免 Windows GBK 乱码）
    _, _stream = configure_console_utf8()
    console_handler = logging.StreamHandler(_stream)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    console_handler.addFilter(SensitiveInfoFilter())
    logger.addHandler(console_handler)
    
    # 文件 handler（如果指定了日志文件）
    if log_file:
        # 确保日志目录存在
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        file_handler.addFilter(SensitiveInfoFilter())
        logger.addHandler(file_handler)
    
    return logger


def log_operation(operation_name: str):
    """
    装饰器：记录操作的开始和结束
    
    Args:
        operation_name: 操作名称
    
    Requirements: 11.1
    
    Example:
        @log_operation("搜索食谱")
        def search_recipes(query: str):
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = logging.getLogger("diet_agent")
            logger.info(f"开始执行: {operation_name}")
            try:
                result = func(*args, **kwargs)
                logger.info(f"完成执行: {operation_name}")
                return result
            except Exception as e:
                logger.error(f"执行失败: {operation_name} - {str(e)}", exc_info=True)
                raise
        return wrapper
    return decorator


def log_error_with_context(logger: logging.Logger, error: Exception, context: dict = None):
    """
    记录错误及其上下文信息
    
    Args:
        logger: Logger 实例
        error: 异常对象
        context: 上下文信息字典
    
    Requirements: 11.2
    """
    error_msg = f"错误: {type(error).__name__}: {str(error)}"
    if context:
        error_msg += f"\n上下文: {context}"
    error_msg += f"\n堆栈跟踪:\n{traceback.format_exc()}"
    logger.error(mask_sensitive_info(error_msg))


def log_api_call(logger: logging.Logger, api_name: str, request_data: Any = None, response_data: Any = None):
    """
    记录 API 调用的关键信息
    
    Args:
        logger: Logger 实例
        api_name: API 名称
        request_data: 请求数据（会被脱敏）
        response_data: 响应数据（会被脱敏）
    
    Requirements: 11.3
    """
    log_msg = f"API 调用: {api_name}"
    if request_data:
        log_msg += f"\n请求: {str(request_data)[:200]}..."  # 限制长度
    if response_data:
        log_msg += f"\n响应: {str(response_data)[:200]}..."  # 限制长度
    logger.info(mask_sensitive_info(log_msg))


# 全局 logger 实例
_global_logger: Optional[logging.Logger] = None


def get_logger(name: str = "diet_agent") -> logging.Logger:
    """
    获取全局 logger 实例
    
    Args:
        name: Logger 名称
    
    Returns:
        Logger 实例
    """
    global _global_logger
    if _global_logger is None:
        log_file = os.environ.get("DIET_LOG_FILE")
        _global_logger = setup_logger(name=name, log_file=log_file)
    return _global_logger

