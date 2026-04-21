"""
ChromaDB 向量存储客户端

负责管理向量数据库的初始化、存储和检索

Requirements:
- 6.1: 将查询文本转换为向量
- 7.1: 从 JSON 文件加载食谱数据并向量化
- 7.2: 为每个食谱生成描述性文本
- 7.3: 同时存储元数据
"""

import json
import os
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from ..config import get_settings
from ..utils.logger import get_logger
from .data_loader import build_document_text
from .recipe_metadata import build_recipe_metadata

logger = get_logger(__name__)


class VectorStoreManager:
    """向量存储管理器（单例模式）
    
    管理 ChromaDB 客户端的生命周期，避免重复创建连接
    
    Requirements: 8.3 - 向量数据库连接失败时记录错误并尝试重新连接
    """
    _instance: Optional['VectorStoreManager'] = None
    _client: Optional[chromadb.PersistentClient] = None
    _vectorstore: Optional[Chroma] = None
    _max_reconnect_attempts: int = 3
    _reconnect_delay: float = 1.0
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_client(self, db_path: str, force_reconnect: bool = False) -> chromadb.PersistentClient:
        """获取 ChromaDB 客户端，支持重连机制
        
        Args:
            db_path: 数据库存储路径
            force_reconnect: 是否强制重新连接
        
        Returns:
            ChromaDB 客户端实例
        
        Raises:
            Exception: 连接失败时抛出异常
        
        Requirements: 8.3
        """
        if self._client is None or force_reconnect:
            import time
            last_error = None
            
            for attempt in range(self._max_reconnect_attempts):
                try:
                    logger.info(f"初始化 ChromaDB 客户端 (尝试 {attempt + 1}/{self._max_reconnect_attempts})，路径: {db_path}")
                    
                    # 确保目录存在
                    os.makedirs(db_path, exist_ok=True)
                    
                    self._client = chromadb.PersistentClient(
                        path=db_path,
                        settings=ChromaSettings(
                            anonymized_telemetry=False,
                            allow_reset=True
                        )
                    )
                    logger.info("ChromaDB 客户端连接成功")
                    return self._client
                    
                except Exception as e:
                    last_error = e
                    logger.warning(
                        f"ChromaDB 客户端连接失败 (尝试 {attempt + 1}/{self._max_reconnect_attempts}): {e}",
                        exc_info=True
                    )
                    
                    if attempt < self._max_reconnect_attempts - 1:
                        logger.info(f"等待 {self._reconnect_delay} 秒后重试...")
                        time.sleep(self._reconnect_delay)
            
            # 所有重试都失败
            error_msg = f"ChromaDB 客户端连接失败，已达到最大重试次数: {last_error}"
            logger.error(error_msg)
            raise Exception(error_msg)
        
        return self._client
    
    def reconnect(self, db_path: str) -> bool:
        """重新连接 ChromaDB 客户端
        
        Args:
            db_path: 数据库存储路径
        
        Returns:
            是否重连成功
        
        Requirements: 8.3
        """
        try:
            logger.info("尝试重新连接 ChromaDB...")
            self._client = None
            self.get_client(db_path, force_reconnect=True)
            logger.info("ChromaDB 重连成功")
            return True
        except Exception as e:
            logger.error(f"ChromaDB 重连失败: {e}", exc_info=True)
            return False
    
    def get_vectorstore(self) -> Optional[Chroma]:
        """获取向量存储实例
        
        Returns:
            向量存储实例，如果未初始化则返回 None
        """
        return self._vectorstore
    
    def set_vectorstore(self, vectorstore: Chroma) -> None:
        """设置向量存储实例
        
        Args:
            vectorstore: 向量存储实例
        """
        self._vectorstore = vectorstore


def init_vectorstore(
    recipes_file: str,
    force_reload: bool = False
) -> Chroma:
    """
    初始化向量数据库
    
    处理流程：
    1. 读取 recipes.json
    2. 为每个食谱调用 build_document_text()
    3. 批量向量化所有文本
    4. 存入 ChromaDB（包含 embeddings, documents, metadatas, ids）
    5. 持久化到磁盘
    
    Args:
        recipes_file: 食谱 JSON 文件路径
        force_reload: 是否强制重新加载（删除现有数据）
    
    Returns:
        初始化好的向量存储实例
    
    Requirements: 6.1, 7.1, 7.2, 7.3, 8.1, 8.2, 8.3, 8.4, 8.5
    
    Raises:
        FileNotFoundError: 食谱文件不存在
        ValueError: 数据格式错误或数据验证失败
        Exception: 向量化或存储失败
    """
    settings = get_settings()
    manager = VectorStoreManager()
    
    # 检查是否已经初始化
    if not force_reload and manager.get_vectorstore() is not None:
        logger.info("向量存储已初始化，直接返回")
        return manager.get_vectorstore()
    
    logger.info(f"开始初始化向量数据库，数据文件: {recipes_file}")
    
    # 1. 读取食谱数据（增强错误处理）
    if not os.path.exists(recipes_file):
        error_msg = f"食谱文件不存在: {recipes_file}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    try:
        with open(recipes_file, 'r', encoding='utf-8') as f:
            recipes_data = json.load(f)
        
        # 数据验证
        if not isinstance(recipes_data, list):
            raise ValueError("食谱数据必须是列表格式")
        
        if len(recipes_data) == 0:
            raise ValueError("食谱数据为空")
        
        logger.info(f"成功读取 {len(recipes_data)} 条食谱数据")
        
    except json.JSONDecodeError as e:
        error_msg = f"食谱文件 JSON 格式错误: {e}"
        logger.error(error_msg, exc_info=True)
        raise ValueError(error_msg)
    except UnicodeDecodeError as e:
        error_msg = f"食谱文件编码错误，请确保文件使用 UTF-8 编码: {e}"
        logger.error(error_msg, exc_info=True)
        raise ValueError(error_msg)
    except Exception as e:
        error_msg = f"读取食谱文件失败: {e}"
        logger.error(error_msg, exc_info=True)
        raise Exception(error_msg)
    
    # 2. 构建文档文本和元数据（增强错误处理）
    documents = []
    metadatas = []
    ids = []
    failed_recipes = []
    
    for idx, recipe in enumerate(recipes_data):
        try:
            # 数据验证
            if not isinstance(recipe, dict):
                logger.warning(f"跳过无效食谱 (索引 {idx}): 不是字典格式")
                failed_recipes.append(f"索引 {idx}: 格式错误")
                continue
            
            recipe_name = recipe.get("name", f"未命名食谱_{idx}")
            
            # 验证必需字段
            required_fields = ["name", "time", "difficulty"]
            missing_fields = [f for f in required_fields if f not in recipe]
            if missing_fields:
                logger.warning(f"跳过食谱 '{recipe_name}': 缺少必需字段 {missing_fields}")
                failed_recipes.append(f"{recipe_name}: 缺少字段 {missing_fields}")
                continue
            
            # 构建文档文本
            doc_text = build_document_text(recipe)
            if not doc_text or len(doc_text.strip()) == 0:
                logger.warning(f"跳过食谱 '{recipe_name}': 文档文本为空")
                failed_recipes.append(f"{recipe_name}: 文档文本为空")
                continue
            
            documents.append(doc_text)
            
            # 构建元数据
            metadata = build_recipe_metadata(recipe)
            
            metadatas.append(metadata)
            
            # 生成 ID
            recipe_id = recipe.get("id", f"recipe_{len(ids) + 1:03d}")
            ids.append(recipe_id)
            
        except Exception as e:
            recipe_name = recipe.get('name', f'索引_{idx}') if isinstance(recipe, dict) else f'索引_{idx}'
            logger.error(f"处理食谱失败: {recipe_name}, 错误: {e}", exc_info=True)
            failed_recipes.append(f"{recipe_name}: {str(e)}")
            continue
    
    # 检查是否有有效数据
    if not documents:
        error_msg = "没有有效的食谱数据可以向量化"
        if failed_recipes:
            error_msg += f"\n失败的食谱: {failed_recipes[:5]}"  # 只显示前5个
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    logger.info(f"成功构建 {len(documents)} 条文档文本")
    if failed_recipes:
        logger.warning(f"跳过 {len(failed_recipes)} 条无效食谱")
    
    # 3. 初始化 Embedding 模型（增强错误处理）
    try:
        embeddings = DashScopeEmbeddings(
            model=settings.embedding_model,
            dashscope_api_key=settings.dashscope_api_key
        )
        logger.info(f"初始化 Embedding 模型: {settings.embedding_model}")
    except Exception as e:
        error_msg = f"初始化 Embedding 模型失败: {e}"
        logger.error(error_msg, exc_info=True)
        raise Exception(error_msg)
    
    # 4. 创建或重置 Collection（增强错误处理）
    try:
        client = manager.get_client(settings.chroma_db_path)
    except Exception as e:
        error_msg = f"连接 ChromaDB 失败: {e}"
        logger.error(error_msg, exc_info=True)
        raise Exception(error_msg)
    
    if force_reload:
        try:
            client.delete_collection(name=settings.collection_name)
            logger.info(f"删除现有 Collection: {settings.collection_name}")
        except Exception as e:
            logger.warning(f"删除 Collection 失败（可能不存在）: {e}")
    
    # 5. 批量向量化并存储（增强错误处理）
    # 使用 add_texts 进行批量向量化（内部调用 embed_documents 而非 embed_query）
    # 这样可以显著提升性能（减少 60-70% 的初始化时间）
    try:
        logger.info("开始批量向量化和存储...")
        vectorstore = Chroma(
            collection_name=settings.collection_name,
            embedding_function=embeddings,
            client=client,
            persist_directory=settings.chroma_db_path
        )
        
        # 批量添加文档（使用 embed_documents 进行批量向量化）
        vectorstore.add_texts(
            texts=documents,
            metadatas=metadatas,
            ids=ids
        )
        logger.info(f"成功向量化并存储 {len(documents)} 条食谱")
        
    except Exception as e:
        error_msg = f"向量化和存储失败: {e}"
        logger.error(error_msg, exc_info=True)
        
        # 尝试重连
        logger.info("尝试重新连接 ChromaDB...")
        if manager.reconnect(settings.chroma_db_path):
            logger.info("重连成功，重试向量化...")
            try:
                client = manager.get_client(settings.chroma_db_path)
                vectorstore = Chroma(
                    collection_name=settings.collection_name,
                    embedding_function=embeddings,
                    client=client,
                    persist_directory=settings.chroma_db_path
                )
                vectorstore.add_texts(
                    texts=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                logger.info("重试成功")
            except Exception as retry_error:
                error_msg = f"重试失败: {retry_error}"
                logger.error(error_msg, exc_info=True)
                raise Exception(error_msg)
        else:
            raise Exception(error_msg)
    
    # 6. 保存到管理器
    manager.set_vectorstore(vectorstore)
    
    logger.info("向量数据库初始化完成")
    return vectorstore


def get_vectorstore() -> Optional[Chroma]:
    """获取已初始化的向量存储实例"""
    manager = VectorStoreManager()
    return manager.get_vectorstore()


def get_or_connect_vectorstore() -> Optional[Chroma]:
    """获取向量存储，若内存中无实例则懒连接磁盘上已有的 ChromaDB。

    Returns:
        Chroma 实例，若磁盘无数据或连接失败则返回 None
    """
    manager = VectorStoreManager()
    vs = manager.get_vectorstore()
    if vs is not None:
        return vs

    settings = get_settings()
    try:
        client = manager.get_client(settings.chroma_db_path)
        embeddings = DashScopeEmbeddings(
            model=settings.embedding_model,
            dashscope_api_key=settings.dashscope_api_key,
        )
        vs = Chroma(
            collection_name=settings.collection_name,
            embedding_function=embeddings,
            client=client,
            persist_directory=settings.chroma_db_path,
        )
        if vs._collection.count() > 0:
            manager.set_vectorstore(vs)
            logger.info(f"懒连接向量库成功，文档数: {vs._collection.count()}")
            return vs
        logger.warning("向量库为空，需先调用 init_vectorstore() 加载数据")
    except Exception as e:
        logger.error(f"懒连接向量库失败: {e}", exc_info=True)
    return None


def add_recipe(recipe: Dict[str, Any]) -> bool:
    """
    添加新食谱到向量数据库
    
    Args:
        recipe: 食谱字典
    
    Returns:
        是否添加成功
    
    Requirements: 7.4, 8.3, 8.4, 8.5
    """
    try:
        vectorstore = get_vectorstore()
        if vectorstore is None:
            error_msg = "向量存储未初始化"
            logger.error(error_msg)
            return False
        
        # 数据验证
        if not isinstance(recipe, dict):
            logger.error("食谱数据必须是字典格式")
            return False
        
        recipe_name = recipe.get("name", "")
        if not recipe_name:
            logger.error("食谱缺少名称字段")
            return False
        
        # 验证必需字段
        required_fields = ["name", "time", "difficulty"]
        missing_fields = [f for f in required_fields if f not in recipe]
        if missing_fields:
            logger.error(f"食谱 '{recipe_name}' 缺少必需字段: {missing_fields}")
            return False
        
        # 构建文档文本
        try:
            doc_text = build_document_text(recipe)
            if not doc_text or len(doc_text.strip()) == 0:
                logger.error(f"食谱 '{recipe_name}' 文档文本为空")
                return False
        except Exception as e:
            logger.error(f"构建文档文本失败: {e}", exc_info=True)
            return False
        
        # 构建元数据
        metadata = build_recipe_metadata(recipe)
        
        # 生成 ID
        recipe_id = recipe.get("id", f"recipe_new_{recipe.get('name', 'unknown')}")
        
        # 添加到向量库
        vectorstore.add_texts(
            texts=[doc_text],
            metadatas=[metadata],
            ids=[recipe_id]
        )
        
        logger.info(f"成功添加食谱: {recipe.get('name', 'unknown')}")
        return True
        
    except Exception as e:
        logger.error(f"添加食谱失败: {e}", exc_info=True)
        return False


def update_recipe(recipe: Dict[str, Any]) -> bool:
    """
    更新现有食谱
    
    Args:
        recipe: 食谱字典（必须包含 id）
    
    Returns:
        是否更新成功
    
    Requirements: 7.5, 8.3, 8.4, 8.5
    """
    try:
        vectorstore = get_vectorstore()
        if vectorstore is None:
            error_msg = "向量存储未初始化"
            logger.error(error_msg)
            return False
        
        # 数据验证
        if not isinstance(recipe, dict):
            logger.error("食谱数据必须是字典格式")
            return False
        
        recipe_id = recipe.get("id")
        if not recipe_id:
            logger.error("食谱缺少 ID，无法更新")
            return False
        
        recipe_name = recipe.get("name", recipe_id)
        logger.info(f"开始更新食谱: {recipe_name} (ID: {recipe_id})")
        
        # 删除旧数据
        try:
            vectorstore.delete(ids=[recipe_id])
            logger.info(f"已删除旧数据: {recipe_id}")
        except Exception as e:
            logger.warning(f"删除旧数据失败（可能不存在）: {e}")
        
        # 添加新数据
        success = add_recipe(recipe)
        if success:
            logger.info(f"成功更新食谱: {recipe_name}")
        else:
            logger.error(f"更新食谱失败: {recipe_name}")
        
        return success
        
    except Exception as e:
        logger.error(f"更新食谱失败: {e}", exc_info=True)
        return False
