"""
配置管理模块
使用 Pydantic Settings 管理配置，支持环境变量和 .env 文件

Requirements:
- 10.1: 从环境变量或配置文件加载 API 密钥
- 10.2: 验证必需的配置项是否存在
- 10.3: 配置项缺失时使用合理的默认值或报错
- 10.4: 支持通过环境变量覆盖配置
"""

from pydantic_settings import BaseSettings
from pydantic import Field, field_validator
from typing import Optional
import os


class Settings(BaseSettings):
    """系统配置
    
    支持从以下来源加载配置（优先级从高到低）：
    1. 环境变量
    2. .env 文件
    3. 默认值
    """
    
    # API Keys (必填)
    dashscope_api_key: str = Field(..., description="DashScope API 密钥（必填）")
    openai_api_key: Optional[str] = Field(None, description="OpenAI API 密钥（兼容模式，可选）")
    
    # API Base URLs
    dashscope_base_url: str = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        description="DashScope API Base URL"
    )
    openai_base_url: Optional[str] = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        description="OpenAI API Base URL（兼容模式）"
    )
    
    # Model Names
    dashscope_model: str = Field(default="qwen3.5-flash", description="DashScope 模型名称")
    openai_model: Optional[str] = Field(default="qwen3.5-flash", description="OpenAI 模型名称（兼容模式）")
    
    # LLM 配置
    llm_model: str = Field(default="qwen3.5-flash", description="LLM 模型名称")
    llm_temperature: float = Field(default=0.1, description="LLM 温度参数")
    llm_request_timeout_seconds: float = Field(default=120.0, description="LLM 请求超时时间（秒）")
    llm_connect_timeout_seconds: float = Field(default=20.0, description="LLM 连接超时时间（秒）")
    
    # Embedding 配置
    embedding_model: str = Field(default="text-embedding-v1", description="Embedding 模型名称")
    openai_embedding_model: str = Field(default="text-embedding-v1", description="OpenAI Embedding 模型名称")
    
    # ChromaDB 配置
    chroma_db_path: str = Field(default="./chroma_db", description="ChromaDB 存储路径")
    collection_name: str = Field(default="recipes", description="Collection 名称")
    tutorial_collection_name: str = Field(default="recipe_tutorials", description="Tutorial Collection 名称")
    bilibili_summary_collection_name: str = Field(default="bilibili_tutorials", description="Bilibili 视频总结 Collection 名称")
    bilibili_cookies_from_browser: str = Field(default="", description="Bilibili 抓取时传给 yt-dlp 的浏览器 cookies 来源，例如 chrome")
    bilibili_cookies_file: str = Field(default="", description="Bilibili 抓取时传给 yt-dlp 的 cookies.txt 文件路径")
    bilibili_whisper_model: str = Field(default="base", description="Bilibili 视频无字幕时使用的 Whisper 模型")
    
    # Agent 配置
    max_iterations: int = Field(default=5, description="Agent 最大迭代次数")
    verbose: bool = Field(default=True, description="是否显示详细日志")
    
    # 检索配置
    top_k: int = Field(default=5, description="检索返回的结果数量")
    
    # PostgreSQL 配置 (V3)
    postgres_host: str = Field(default="localhost", description="PostgreSQL 主机地址")
    postgres_port: int = Field(default=5432, description="PostgreSQL 端口")
    postgres_user: str = Field(default="postgres", description="PostgreSQL 用户名")
    postgres_password: str = Field(default="", description="PostgreSQL 密码")
    postgres_database: str = Field(default="diet_agent_v3", description="PostgreSQL 数据库名")
    
    # Cross-Encoder 配置 (V3)
    reranker_model: str = Field(
        default="BAAI/bge-reranker-base",
        description="Cross-Encoder 模型名称（中文优化）"
    )
    reranker_top_k: int = Field(default=10, description="重排序后返回的结果数量")
    reranker_score_threshold: float = Field(default=0.0, description="重排序分数阈值")
    
    # 上下文压缩配置 (V3)
    context_compression_strategy: str = Field(
        default="hybrid",
        description="上下文压缩策略: window, summary, extract, hybrid"
    )
    context_window_size: int = Field(default=5, description="滑动窗口大小（对话轮数）")
    context_max_tokens: int = Field(default=2000, description="上下文最大 token 数")
    
    # LangGraph 配置 (V4)
    graph_max_retries: int = Field(default=1, description="Evaluator 最大重试次数")
    graph_retriever_top_k: int = Field(default=10, description="图检索 top_k")
    graph_reranker_top_k: int = Field(default=5, description="图精排 top_k")
    fusion_mode: str = Field(default="fixed_fusion", description="混合检索融合模式: fixed_fusion / query_aware_fusion")
    
    # RAG 策略配置 (Phase 1)
    rag_strategy: str = Field(default="adaptive", description="RAG 策略: adaptive / standard")
    rag_multi_query_count: int = Field(default=3, description="Multi-Query 生成的查询数量")
    rag_max_retrieval_retries: int = Field(default=2, description="Self-RAG 最大重检索次数")
    rag_relevance_threshold: float = Field(default=0.5, description="文档相关性阈值")

    # 语义分块配置 (Phase 1)
    chunking_strategy: str = Field(default="semantic", description="分块策略: semantic / fixed")
    chunking_breakpoint_type: str = Field(default="percentile", description="语义分块断点类型")
    chunking_breakpoint_threshold: float = Field(default=95.0, description="语义分块断点阈值")

    # ── 评测配置 (Phase 2) ────────────────────────────────────────────────────────
    eval_dataset_path: str = Field(
        default="src/evaluation/test_dataset.json",
        description="评测数据集路径"
    )
    eval_faithfulness_threshold: float = Field(
        default=0.7,
        description="Faithfulness 合格阈值"
    )
    eval_relevancy_threshold: float = Field(
        default=0.7,
        description="Answer Relevancy 合格阈值"
    )
    eval_retrieval_k: int = Field(
        default=5,
        description="Recall@K / nDCG@K 中的 K 值"
    )

    # ── 上下文工程配置 (Phase 3) ──────────────────────────────────────────────
    context_total_token_budget: int = Field(
        default=8000,
        description="上下文总 token 预算"
    )
    context_response_reserve_ratio: float = Field(
        default=0.5,
        description="为生成预留的 token 比例 (0.0~1.0)"
    )
    context_working_memory_window: int = Field(
        default=5,
        description="工作记忆窗口大小（对话轮数）"
    )
    context_episodic_update_threshold: int = Field(
        default=6,
        description="触发情景记忆摘要更新的消息数阈值"
    )
    context_enable_skills: bool = Field(
        default=True,
        description="是否启用 Skill 模式"
    )

    # ── Phase 4: 可观测性 + Streaming 配置 ──────────────────────────────────
    # FastAPI
    api_host: str = Field(default="0.0.0.0", description="API 监听地址")
    api_port: int = Field(default=8000, description="API 监听端口")
    api_cors_origins: str = Field(default="*", description="CORS 允许的来源（逗号分隔）")

    # Langfuse（默认禁用）
    langfuse_enabled: bool = Field(default=False, description="是否启用 Langfuse 追踪")
    langfuse_public_key: str = Field(default="", description="Langfuse Public Key")
    langfuse_secret_key: str = Field(default="", description="Langfuse Secret Key")
    langfuse_host: str = Field(
        default="https://cloud.langfuse.com",
        description="Langfuse 服务地址"
    )

    # 结构化日志
    structured_log_enabled: bool = Field(
        default=True,
        description="是否启用结构化请求日志"
    )

    # ── Phase 5: 性能优化配置 ────────────────────────────────────────────────
    # LLM 后端选择
    llm_backend: str = Field(
        default="dashscope_openai_compat",
        description="LLM 后端: dashscope_multimodal / dashscope_openai_compat"
    )
    dashscope_openai_base_url: str = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        description="DashScope OpenAI 兼容端点"
    )

    # 快速相关性过滤
    fast_relevance_enabled: bool = Field(
        default=True,
        description="使用 embedding 相似度替代 LLM 逐篇相关性过滤"
    )
    fast_relevance_threshold: float = Field(
        default=0.3,
        description="embedding 相关性阈值（余弦相似度）"
    )

    # Self-RAG 精简模式
    self_rag_lite: bool = Field(
        default=True,
        description="合并 Self-RAG 门控（幻觉+有用性→单次调用）"
    )

    # LLM 缓存
    llm_cache_enabled: bool = Field(
        default=True,
        description="启用 LLM 响应缓存（SQLite）"
    )
    llm_cache_path: str = Field(
        default="./cache/llm_cache.db",
        description="LLM 缓存数据库路径"
    )

    # 日志配置
    log_level: str = Field(default="INFO", description="日志级别")
    log_file: str = Field(default="logs/agent.log", description="日志文件路径")
    
    def model_post_init(self, __context) -> None:
        """模型初始化后的处理
        
        如果未设置 openai_api_key，使用 dashscope_api_key
        """
        if self.openai_api_key is None:
            self.openai_api_key = self.dashscope_api_key
    
    @property
    def postgres_connection_string(self) -> str:
        """获取 PostgreSQL 连接字符串
        
        Returns:
            PostgreSQL 连接字符串
        """
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_database}"
        )
    
    @field_validator('dashscope_api_key')
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        """验证 API 密钥（必填项）
        
        Requirements: 10.2, 10.3
        """
        if not v or v.strip() == "" or v == "your-api-key-here":
            raise ValueError(
                "DASHSCOPE_API_KEY 未配置或无效。\n"
                "请在 .env 文件中设置有效的 API 密钥，或通过环境变量 DASHSCOPE_API_KEY 设置。"
            )
        return v.strip()
    
    @field_validator('llm_temperature')
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        """验证温度参数
        
        Requirements: 10.3
        """
        if not 0 <= v <= 2:
            raise ValueError("llm_temperature 必须在 0 到 2 之间")
        return v
    
    @field_validator('max_iterations')
    @classmethod
    def validate_max_iterations(cls, v: int) -> int:
        """验证最大迭代次数
        
        Requirements: 10.3
        """
        if v <= 0:
            raise ValueError("max_iterations 必须大于 0")
        return v
    
    @field_validator('top_k')
    @classmethod
    def validate_top_k(cls, v: int) -> int:
        """验证检索结果数量
        
        Requirements: 10.3
        """
        if v <= 0:
            raise ValueError("top_k 必须大于 0")
        return v
    
    @field_validator('log_level')
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """验证日志级别
        
        Requirements: 10.3
        """
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"log_level 必须是以下之一: {', '.join(valid_levels)}")
        return v_upper
    
    class Config:
        """Pydantic 配置
        
        Requirements: 10.1, 10.4
        - env_file: 从 .env 文件加载配置
        - case_sensitive: 环境变量不区分大小写
        - 环境变量优先级高于 .env 文件
        """
        # 使用绝对路径确保无论从哪里运行都能找到 .env 文件
        _base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env_file = os.path.join(_base_dir, ".env")
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # 忽略未定义的额外字段


# 全局配置实例
_settings: Optional[Settings] = None


def get_settings(force_reload: bool = False) -> Settings:
    """获取配置实例（单例模式）
    
    Args:
        force_reload: 是否强制重新加载配置
    
    Returns:
        Settings: 配置实例
    
    Requirements: 10.1, 10.2, 10.4
    
    Raises:
        ValueError: 配置验证失败时抛出
    """
    global _settings
    if _settings is None or force_reload:
        try:
            _settings = Settings()
        except Exception as e:
            raise ValueError(f"配置加载失败: {str(e)}")
    return _settings


def validate_config() -> bool:
    """验证配置是否有效
    
    Returns:
        bool: 配置是否有效
    
    Requirements: 10.2
    """
    try:
        get_settings()
        return True
    except Exception:
        return False
