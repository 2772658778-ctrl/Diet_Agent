"""
知识图谱增强的向量数据库客户端 (V2)

负责初始化包含三类实体（食谱、食材、营养素）的向量数据库
将知识图谱关系编码到元数据中，实现语义检索 + 关系推理的统一架构
"""

import json
import os
from typing import List, Dict, Any, Optional
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from ..config import get_settings
from ..utils.logger import get_logger
from .recipe_metadata import build_recipe_metadata
from .data_loader_v2 import (
    build_recipe_document_v2,
    build_ingredient_document_v2,
    build_nutrient_document_v2
)

logger = get_logger(__name__)


def load_json(file_path: str) -> List[Dict[str, Any]]:
    """
    加载 JSON 文件
    
    Args:
        file_path: JSON 文件路径
    
    Returns:
        解析后的数据列表
    
    Raises:
        FileNotFoundError: 文件不存在
        ValueError: JSON 格式错误
    """
    if not file_path:
        return []
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            raise ValueError(f"数据格式错误: 期望列表，实际为 {type(data)}")
        
        return data
    
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON 格式错误: {e}")
    except UnicodeDecodeError as e:
        raise ValueError(f"文件编码错误，请确保使用 UTF-8 编码: {e}")


def prepare_recipe_metadata(recipe: Dict[str, Any]) -> Dict[str, Any]:
    """
    准备食谱元数据（包含关系）
    
    将食谱数据转换为向量数据库的元数据格式
    关系数据以 JSON 字符串形式存储在 relations 字段中
    
    Args:
        recipe: 食谱字典，包含 V2 数据模型的所有字段
    
    Returns:
        元数据字典，包含所有可查询字段和关系数据
    """
    return build_recipe_metadata(recipe, include_entity_fields=True)


def prepare_ingredient_metadata(ingredient: Dict[str, Any]) -> Dict[str, Any]:
    """
    准备食材元数据（包含关系）
    
    将食材数据转换为向量数据库的元数据格式
    关系数据以 JSON 字符串形式存储在 relations 字段中
    
    Args:
        ingredient: 食材字典，包含 V2 数据模型的所有字段
    
    Returns:
        元数据字典，包含所有可查询字段和关系数据
    """
    metadata = {
        # 实体类型
        "entity_type": "ingredient",
        "id": ingredient.get("id", ""),
        "name": ingredient.get("name", ""),
        
        # 基础属性
        "category": ingredient.get("category", ""),
        "aliases": ",".join(ingredient.get("aliases", [])),
        "season": ",".join(ingredient.get("season", [])),
        "price_range": ingredient.get("price_range", ""),
        
        # 营养信息（每100g）
        "calories_per_100g": float(ingredient.get("nutrition_per_100g", {}).get("calories", 0.0)),
        "protein_per_100g": float(ingredient.get("nutrition_per_100g", {}).get("protein", 0.0)),
        
        # 关系（JSON 编码）
        "relations": json.dumps(ingredient.get("relations", {}), ensure_ascii=False)
    }
    
    return metadata


def prepare_nutrient_metadata(nutrient: Dict[str, Any]) -> Dict[str, Any]:
    """
    准备营养素元数据（包含关系）
    
    将营养素数据转换为向量数据库的元数据格式
    关系数据以 JSON 字符串形式存储在 relations 字段中
    
    Args:
        nutrient: 营养素字典，包含 V2 数据模型的所有字段
    
    Returns:
        元数据字典，包含所有可查询字段和关系数据
    """
    metadata = {
        # 实体类型
        "entity_type": "nutrient",
        "id": nutrient.get("id", ""),
        "name": nutrient.get("name", ""),
        
        # 基础属性
        "category": nutrient.get("category", ""),
        "unit": nutrient.get("unit", ""),
        "english_name": nutrient.get("english_name", ""),
        
        # 关系（JSON 编码）
        "relations": json.dumps(nutrient.get("relations", {}), ensure_ascii=False)
    }
    
    return metadata


def init_kg_enhanced_vectorstore(
    recipes_file: str,
    ingredients_file: str,
    nutrients_file: Optional[str] = None,
    collection_name: str = "kg_enhanced_recipes",
    force_reload: bool = False
) -> Chroma:
    """
    初始化知识图谱增强的向量数据库
    
    处理流程：
    1. 读取三类实体数据（食谱、食材、营养素）
    2. 为每个实体构建包含关系信息的文档文本
    3. 准备富元数据（包含 relations JSON）
    4. 批量向量化所有实体
    5. 存储到 ChromaDB
    
    Args:
        recipes_file: 食谱 JSON 文件路径
        ingredients_file: 食材 JSON 文件路径
        nutrients_file: 营养素 JSON 文件路径（可选）
        collection_name: Collection 名称
        force_reload: 是否强制重新加载（删除现有数据）
    
    Returns:
        初始化好的向量存储实例
    
    Raises:
        FileNotFoundError: 数据文件不存在
        ValueError: 数据格式错误
        Exception: 向量化或存储失败
    """
    settings = get_settings()
    
    logger.info("=" * 60)
    logger.info("开始初始化知识图谱增强向量数据库 (V2)")
    logger.info("=" * 60)
    
    # 1. 读取所有数据
    logger.info("步骤 1/5: 读取数据文件...")
    try:
        recipes = load_json(recipes_file)
        ingredients = load_json(ingredients_file)
        nutrients = load_json(nutrients_file) if nutrients_file else []
        
        logger.info(f"✓ 成功读取数据:")
        logger.info(f"  - 食谱: {len(recipes)} 条")
        logger.info(f"  - 食材: {len(ingredients)} 条")
        logger.info(f"  - 营养素: {len(nutrients)} 条")
        logger.info(f"  - 总计: {len(recipes) + len(ingredients) + len(nutrients)} 条")
    
    except Exception as e:
        logger.error(f"✗ 读取数据文件失败: {e}", exc_info=True)
        raise
    
    # 2. 构建文档和元数据
    logger.info("\n步骤 2/5: 构建文档文本和元数据...")
    documents = []
    metadatas = []
    ids = []
    failed_items = []
    
    # 处理食谱
    logger.info("  处理食谱...")
    for idx, recipe in enumerate(recipes):
        try:
            # 验证必需字段
            if not recipe.get("id") or not recipe.get("name"):
                logger.warning(f"  跳过食谱 (索引 {idx}): 缺少 id 或 name")
                failed_items.append(f"食谱_{idx}: 缺少必需字段")
                continue
            
            # 构建文档文本
            doc_text = build_recipe_document_v2(recipe)
            if not doc_text or len(doc_text.strip()) == 0:
                logger.warning(f"  跳过食谱 '{recipe.get('name')}': 文档文本为空")
                failed_items.append(f"{recipe.get('name')}: 文档文本为空")
                continue
            
            documents.append(doc_text)
            metadatas.append(prepare_recipe_metadata(recipe))
            ids.append(recipe["id"])
        
        except Exception as e:
            recipe_name = recipe.get('name', f'索引_{idx}') if isinstance(recipe, dict) else f'索引_{idx}'
            logger.error(f"  处理食谱失败: {recipe_name}, 错误: {e}")
            failed_items.append(f"{recipe_name}: {str(e)}")
            continue
    
    logger.info(f"  ✓ 成功处理 {len([m for m in metadatas if m['entity_type'] == 'recipe'])} 条食谱")
    
    # 处理食材
    logger.info("  处理食材...")
    for idx, ingredient in enumerate(ingredients):
        try:
            # 验证必需字段
            if not ingredient.get("id") or not ingredient.get("name"):
                logger.warning(f"  跳过食材 (索引 {idx}): 缺少 id 或 name")
                failed_items.append(f"食材_{idx}: 缺少必需字段")
                continue
            
            # 构建文档文本
            doc_text = build_ingredient_document_v2(ingredient)
            if not doc_text or len(doc_text.strip()) == 0:
                logger.warning(f"  跳过食材 '{ingredient.get('name')}': 文档文本为空")
                failed_items.append(f"{ingredient.get('name')}: 文档文本为空")
                continue
            
            documents.append(doc_text)
            metadatas.append(prepare_ingredient_metadata(ingredient))
            ids.append(ingredient["id"])
        
        except Exception as e:
            ingredient_name = ingredient.get('name', f'索引_{idx}') if isinstance(ingredient, dict) else f'索引_{idx}'
            logger.error(f"  处理食材失败: {ingredient_name}, 错误: {e}")
            failed_items.append(f"{ingredient_name}: {str(e)}")
            continue
    
    logger.info(f"  ✓ 成功处理 {len([m for m in metadatas if m['entity_type'] == 'ingredient'])} 条食材")
    
    # 处理营养素
    if nutrients:
        logger.info("  处理营养素...")
        for idx, nutrient in enumerate(nutrients):
            try:
                # 验证必需字段
                if not nutrient.get("id") or not nutrient.get("name"):
                    logger.warning(f"  跳过营养素 (索引 {idx}): 缺少 id 或 name")
                    failed_items.append(f"营养素_{idx}: 缺少必需字段")
                    continue
                
                # 构建文档文本
                doc_text = build_nutrient_document_v2(nutrient)
                if not doc_text or len(doc_text.strip()) == 0:
                    logger.warning(f"  跳过营养素 '{nutrient.get('name')}': 文档文本为空")
                    failed_items.append(f"{nutrient.get('name')}: 文档文本为空")
                    continue
                
                documents.append(doc_text)
                metadatas.append(prepare_nutrient_metadata(nutrient))
                ids.append(nutrient["id"])
            
            except Exception as e:
                nutrient_name = nutrient.get('name', f'索引_{idx}') if isinstance(nutrient, dict) else f'索引_{idx}'
                logger.error(f"  处理营养素失败: {nutrient_name}, 错误: {e}")
                failed_items.append(f"{nutrient_name}: {str(e)}")
                continue
        
        logger.info(f"  ✓ 成功处理 {len([m for m in metadatas if m['entity_type'] == 'nutrient'])} 条营养素")
    
    # 检查是否有有效数据
    if not documents:
        error_msg = "没有有效的数据可以向量化"
        if failed_items:
            error_msg += f"\n失败的项目: {failed_items[:5]}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    logger.info(f"\n✓ 总计构建 {len(documents)} 条文档")
    if failed_items:
        logger.warning(f"  跳过 {len(failed_items)} 条无效数据")
    
    # 3. 初始化 Embedding 模型
    logger.info("\n步骤 3/5: 初始化 Embedding 模型...")
    try:
        embeddings = DashScopeEmbeddings(
            model=settings.embedding_model,
            dashscope_api_key=settings.dashscope_api_key
        )
        logger.info(f"✓ 成功初始化 Embedding 模型: {settings.embedding_model}")
    except Exception as e:
        logger.error(f"✗ 初始化 Embedding 模型失败: {e}", exc_info=True)
        raise
    
    # 4. 创建或重置 Collection
    logger.info("\n步骤 4/5: 准备向量数据库...")
    try:
        # 确保目录存在
        os.makedirs(settings.chroma_db_path, exist_ok=True)
        
        # 创建 Chroma 客户端
        import chromadb
        from chromadb.config import Settings as ChromaSettings
        
        client = chromadb.PersistentClient(
            path=settings.chroma_db_path,
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # 如果强制重新加载，删除现有 Collection
        if force_reload:
            try:
                client.delete_collection(name=collection_name)
                logger.info(f"✓ 删除现有 Collection: {collection_name}")
            except Exception as e:
                logger.info(f"  Collection 不存在或删除失败: {e}")
        
        logger.info(f"✓ ChromaDB 客户端准备完成")
    
    except Exception as e:
        logger.error(f"✗ 准备向量数据库失败: {e}", exc_info=True)
        raise
    
    # 5. 批量向量化并存储
    logger.info("\n步骤 5/5: 批量向量化并存储...")
    try:
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            client=client,
            persist_directory=settings.chroma_db_path
        )
        
        # 批量添加文档
        logger.info(f"  开始向量化 {len(documents)} 条文档...")
        vectorstore.add_texts(
            texts=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        logger.info(f"✓ 成功向量化并存储所有数据")
        
        # 统计信息
        recipe_count = len([m for m in metadatas if m['entity_type'] == 'recipe'])
        ingredient_count = len([m for m in metadatas if m['entity_type'] == 'ingredient'])
        nutrient_count = len([m for m in metadatas if m['entity_type'] == 'nutrient'])
        
        logger.info("\n" + "=" * 60)
        logger.info("知识图谱增强向量数据库初始化完成!")
        logger.info("=" * 60)
        logger.info(f"Collection: {collection_name}")
        logger.info(f"存储路径: {settings.chroma_db_path}")
        logger.info(f"实体统计:")
        logger.info(f"  - 食谱: {recipe_count} 条")
        logger.info(f"  - 食材: {ingredient_count} 条")
        logger.info(f"  - 营养素: {nutrient_count} 条")
        logger.info(f"  - 总计: {len(documents)} 条")
        logger.info("=" * 60)
        
        return vectorstore
    
    except Exception as e:
        logger.error(f"✗ 向量化和存储失败: {e}", exc_info=True)
        raise


def get_vectorstore(
    collection_name: str = "kg_enhanced_recipes",
    force_init: bool = False
) -> Chroma:
    """
    获取已存在的向量数据库实例
    
    如果向量数据库不存在或 force_init=True，则初始化新的向量数据库
    
    Args:
        collection_name: Collection 名称
        force_init: 是否强制重新初始化
    
    Returns:
        向量存储实例
    
    Raises:
        ValueError: 向量数据库不存在且未提供初始化参数
    """
    settings = get_settings()
    
    try:
        # 初始化 Embedding 模型
        embeddings = DashScopeEmbeddings(
            model=settings.embedding_model,
            dashscope_api_key=settings.dashscope_api_key
        )
        
        # 创建 Chroma 客户端
        import chromadb
        from chromadb.config import Settings as ChromaSettings
        
        client = chromadb.PersistentClient(
            path=settings.chroma_db_path,
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # 检查 Collection 是否存在
        try:
            existing_collections = [col.name for col in client.list_collections()]
            if collection_name not in existing_collections:
                if force_init:
                    logger.warning(f"Collection '{collection_name}' 不存在，将自动初始化")
                    # 使用默认数据文件初始化
                    recipes_file = os.path.join("data", "recipes_v2.json")
                    ingredients_file = os.path.join("data", "ingredients_v2.json")
                    nutrients_file = os.path.join("data", "nutrients_v2.json")
                    return init_kg_enhanced_vectorstore(
                        recipes_file=recipes_file,
                        ingredients_file=ingredients_file,
                        nutrients_file=nutrients_file,
                        collection_name=collection_name,
                        force_reload=False
                    )
                else:
                    raise ValueError(
                        f"Collection '{collection_name}' 不存在。"
                        f"请先运行初始化脚本或设置 force_init=True"
                    )
        except Exception as e:
            logger.error(f"检查 Collection 失败: {e}")
            if force_init:
                logger.warning("将尝试初始化新的向量数据库")
                recipes_file = os.path.join("data", "recipes_v2.json")
                ingredients_file = os.path.join("data", "ingredients_v2.json")
                nutrients_file = os.path.join("data", "nutrients_v2.json")
                return init_kg_enhanced_vectorstore(
                    recipes_file=recipes_file,
                    ingredients_file=ingredients_file,
                    nutrients_file=nutrients_file,
                    collection_name=collection_name,
                    force_reload=False
                )
            raise
        
        # 创建向量存储实例
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            client=client,
            persist_directory=settings.chroma_db_path
        )
        
        logger.info(f"✓ 成功获取向量数据库: {collection_name}")
        return vectorstore
    
    except Exception as e:
        logger.error(f"获取向量数据库失败: {e}", exc_info=True)
        raise
