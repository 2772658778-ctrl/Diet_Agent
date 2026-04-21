"""
语义分块模块

基于 embedding 余弦相似度的断点检测，按语义边界切分文档。
对比固定大小分块，在中文食谱场景下保持语义完整性。

复用模块:
- src/config.py::get_settings() — 获取 embedding 模型配置
- src/vectorstore/data_loader.py::build_document_text() — 文档文本构建
- src/utils/logger.py::get_logger()

依赖:
- langchain_experimental.text_splitter.SemanticChunker
- langchain_community.embeddings.DashScopeEmbeddings
"""

from typing import Any, Optional

from ..config import get_settings
from ..utils.logger import get_logger


logger = get_logger(__name__)


class SemanticChunkerWrapper:
    """语义分块器

    基于 embedding 余弦相似度的断点检测，按语义边界切分文档。
    对比固定大小分块，在中文食谱场景下保持语义完整性。

    复用模块:
    - src/config.py::get_settings() — 获取 embedding 模型配置
    - src/vectorstore/data_loader.py::build_document_text()
    - src/utils/logger.py::get_logger()
    """

    def __init__(
        self,
        breakpoint_type: Optional[str] = None,
        breakpoint_threshold: Optional[float] = None,
    ) -> None:
        """初始化语义分块器。

        Args:
            breakpoint_type: 断点检测类型，默认从 config 读取
                ('percentile' / 'standard_deviation' / 'interquartile')
            breakpoint_threshold: 断点阈值，默认从 config 读取

        Raises:
            ImportError: langchain_experimental 未安装时抛出
        """
        settings = get_settings()

        self._breakpoint_type = breakpoint_type or settings.chunking_breakpoint_type
        self._breakpoint_threshold = (
            breakpoint_threshold
            if breakpoint_threshold is not None
            else settings.chunking_breakpoint_threshold
        )

        try:
            from langchain_community.embeddings import DashScopeEmbeddings
            from langchain_experimental.text_splitter import SemanticChunker

            embeddings = DashScopeEmbeddings(
                model=settings.embedding_model,
                dashscope_api_key=settings.dashscope_api_key,
            )

            self._chunker = SemanticChunker(
                embeddings=embeddings,
                breakpoint_threshold_type=self._breakpoint_type,
                breakpoint_threshold_amount=self._breakpoint_threshold,
            )
            logger.info(
                f"SemanticChunkerWrapper 初始化完成: "
                f"type={self._breakpoint_type}, threshold={self._breakpoint_threshold}"
            )
        except ImportError as e:
            raise ImportError(
                f"langchain_experimental 未安装，请执行: "
                f"pip install langchain-experimental>=0.0.47\n原始错误: {e}"
            )

    def chunk(self, text: str) -> list[str]:
        """语义切分文本。

        Args:
            text: 待切分的文本

        Returns:
            切分后的文本块列表，空文本返回空列表
        """
        if not text or not text.strip():
            logger.warning("语义分块：输入文本为空，返回空列表")
            return []

        try:
            docs = self._chunker.create_documents([text])
            chunks = [doc.page_content for doc in docs if doc.page_content.strip()]
            logger.info(f"语义分块完成: 原文 {len(text)} 字 → {len(chunks)} 个块")
            return chunks
        except Exception as e:
            logger.error(f"语义分块失败: {e}", exc_info=True)
            return [text]

    def chunk_recipe(self, recipe: dict[str, Any]) -> list[dict[str, Any]]:
        """对食谱进行语义分块，每个块保留原始元数据。

        Args:
            recipe: 食谱字典，包含 name / cuisine / time / difficulty 等字段

        Returns:
            带元数据的 chunk 列表，每个元素格式为：
            {'text': str, 'chunk_index': int, 'name': str, ...元数据}
        """
        from ..vectorstore.data_loader import build_document_text

        if not recipe:
            return []

        try:
            doc_text = build_document_text(recipe)
        except Exception as e:
            logger.error(f"构建食谱文档文本失败: {e}", exc_info=True)
            doc_text = recipe.get("name", "")

        chunks = self.chunk(doc_text)
        if not chunks:
            return []

        metadata = {
            "name": recipe.get("name", ""),
            "cuisine": recipe.get("cuisine", ""),
            "time": recipe.get("time", 0),
            "difficulty": recipe.get("difficulty", ""),
            "calories": recipe.get("calories", 0),
            "tags": recipe.get("tags", []),
        }

        result: list[dict[str, Any]] = []
        for idx, chunk_text in enumerate(chunks):
            chunk_doc = {"text": chunk_text, "chunk_index": idx}
            chunk_doc.update(metadata)
            result.append(chunk_doc)

        logger.info(
            f"食谱 '{metadata['name']}' 分块完成: {len(result)} 个块"
        )
        return result

    def compare_with_fixed(
        self,
        text: str,
        fixed_chunk_size: int = 500,
    ) -> dict[str, Any]:
        """对同一文本分别做语义分块和固定大小分块，返回对比统计。

        Args:
            text: 待分块的文本
            fixed_chunk_size: 固定分块大小（字符数）

        Returns:
            对比统计字典，包含：
            {
                'semantic': {'count': int, 'avg_size': float, 'min_size': int, 'max_size': int},
                'fixed':    {'count': int, 'avg_size': float, 'min_size': int, 'max_size': int},
            }
        """
        if not text or not text.strip():
            return {
                "semantic": {"count": 0, "avg_size": 0.0, "min_size": 0, "max_size": 0},
                "fixed": {"count": 0, "avg_size": 0.0, "min_size": 0, "max_size": 0},
            }

        semantic_chunks = self.chunk(text)

        # 固定大小分块
        fixed_chunks = [
            text[i: i + fixed_chunk_size]
            for i in range(0, len(text), fixed_chunk_size)
            if text[i: i + fixed_chunk_size].strip()
        ]

        def _stats(chunks: list[str]) -> dict[str, Any]:
            sizes = [len(c) for c in chunks]
            if not sizes:
                return {"count": 0, "avg_size": 0.0, "min_size": 0, "max_size": 0}
            return {
                "count": len(sizes),
                "avg_size": round(sum(sizes) / len(sizes), 1),
                "min_size": min(sizes),
                "max_size": max(sizes),
            }

        result = {
            "semantic": _stats(semantic_chunks),
            "fixed": _stats(fixed_chunks),
        }
        logger.info(
            f"分块对比: semantic={result['semantic']['count']}块, "
            f"fixed={result['fixed']['count']}块"
        )
        return result
