"""
上下文工程模块 (Phase 3)

提供完整的上下文工程体系：
- ContextCompressor: 上下文压缩（V3 已有）
- ContextAssembler: 动态上下文组装器（按 token 预算分配多源信息）
- TieredMemoryManager: 三层记忆管理（工作记忆 / 情景记忆 / 语义记忆）
- InstructionHierarchy: 指令层级管理
- SkillRegistry: Skill 模式注册与选择

复用模块:
- src/context/context_compressor.py::ContextCompressor
- src/database/postgres_client.py::PostgreSQLClient
- src/config.py::get_settings()
- src/utils/logger.py::get_logger()
"""

from .context_compressor import ContextCompressor
from .context_assembler import ContextAssembler
from .memory_manager import TieredMemoryManager
from .instruction_hierarchy import InstructionHierarchy, SkillRegistry

__all__ = [
    "ContextCompressor",
    "ContextAssembler",
    "TieredMemoryManager",
    "InstructionHierarchy",
    "SkillRegistry",
]
