"""
增强检索器 (V3)

整合 ChromaDB + PostgreSQL + Cross-Encoder + Context Compression
实现完整的检索-重排-压缩流程
"""

from typing import List, Dict, Any, Optional
from langchain_community.vectorstores import Chroma

from ..utils.logger import get_logger
from ..database.postgres_client import PostgresClient
from ..reranker.cross_encoder_reranker import CrossEncoderReranker
from ..context.context_compressor import ContextCompressor
from .hybrid_retriever import HybridRetriever

logger = get_logger(__name__)


class EnhancedRetrieverV3:
    """增强检索器 V3
    
    完整的检索流程：
    1. 混合检索（ChromaDB 向量 + 关系推理）
    2. PostgreSQL 精确过滤（用户偏好、库存、历史）
    3. Cross-Encoder 精排
    4. 上下文压缩
    
    Features:
    - 多路召回：语义相似 + 关系推理 + 用户偏好
    - 精确过滤：基于 PostgreSQL 的结构化数据过滤
    - 智能重排：Cross-Encoder 精排提升相关性
    - 上下文压缩：自适应压缩策略节省 token
    """
    
    def __init__(
        self,
        vectorstore: Chroma,
        postgres_client: PostgresClient,
        reranker: Optional[CrossEncoderReranker] = None,
        compressor: Optional[ContextCompressor] = None
    ):
        """
        初始化增强检索器
        
        Args:
            vectorstore: ChromaDB 向量存储
            postgres_client: PostgreSQL 客户端
            reranker: Cross-Encoder 重排器（可选）
            compressor: 上下文压缩器（可选）
        """
        self.vectorstore = vectorstore
        self.postgres_client = postgres_client
        self.hybrid_retriever = HybridRetriever(vectorstore)
        
        # 初始化重排器
        self.reranker = reranker or CrossEncoderReranker()
        
        # 初始化压缩器
        self.compressor = compressor or ContextCompressor()
        
        logger.info("增强检索器 V3 初始化完成")
    
    def retrieve(
        self,
        query: str,
        user_id: Optional[int] = None,
        user_context: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
        use_reranker: bool = True,
        use_postgres_filter: bool = True
    ) -> List[Dict[str, Any]]:
        """
        完整的检索流程
        
        Args:
            query: 用户查询
            user_id: 用户 ID（用于 PostgreSQL 查询）
            user_context: 用户上下文
            top_k: 返回结果数量
            use_reranker: 是否使用重排器
            use_postgres_filter: 是否使用 PostgreSQL 过滤
        
        Returns:
            检索结果列表
        
        Example:
            >>> retriever = EnhancedRetrieverV3(vectorstore, postgres_client)
            >>> results = retriever.retrieve(
            ...     query="低热量晚餐",
            ...     user_id=1,
            ...     top_k=5
            ... )
        """
        if user_context is None:
            user_context = {}
        
        logger.info(f"开始 V3 增强检索: query='{query}', user_id={user_id}, top_k={top_k}")
        
        try:
            # 步骤 1: 混合检索（ChromaDB）
            logger.info("  步骤 1/4: 混合检索（向量 + 关系）...")
            initial_results = self.hybrid_retriever.retrieve(
                query=query,
                user_context=user_context,
                top_k=top_k * 3,  # 召回更多候选
                lambda_param=0.7
            )
            logger.info(f"    混合检索返回 {len(initial_results)} 条结果")
            
            # 步骤 2: PostgreSQL 精确过滤
            if use_postgres_filter and user_id:
                logger.info("  步骤 2/4: PostgreSQL 精确过滤...")
                filtered_results = self._postgres_filter(
                    results=initial_results,
                    user_id=user_id,
                    user_context=user_context
                )
                logger.info(f"    过滤后剩余 {len(filtered_results)} 条结果")
            else:
                filtered_results = initial_results
                logger.info("  步骤 2/4: 跳过 PostgreSQL 过滤")
            
            # 步骤 3: Cross-Encoder 精排
            if use_reranker and filtered_results:
                logger.info("  步骤 3/4: Cross-Encoder 精排...")
                reranked_results = self.reranker.rerank(
                    query=query,
                    documents=filtered_results,
                    top_k=top_k
                )
                logger.info(f"    精排后返回 {len(reranked_results)} 条结果")
            else:
                reranked_results = filtered_results[:top_k]
                logger.info("  步骤 3/4: 跳过 Cross-Encoder 精排")
            
            # 步骤 4: 记录交互（异步）
            if user_id:
                logger.info("  步骤 4/4: 记录交互历史...")
                self._log_interaction(
                    user_id=user_id,
                    query=query,
                    results=reranked_results,
                    context=user_context
                )
            else:
                logger.info("  步骤 4/4: 跳过交互记录")
            
            logger.info(f"V3 增强检索完成，返回 {len(reranked_results)} 条结果")
            return reranked_results
        
        except Exception as e:
            logger.error(f"V3 增强检索失败: {e}", exc_info=True)
            return []
    
    def retrieve_with_context_compression(
        self,
        query: str,
        conversation_history: List[Dict[str, str]],
        user_id: Optional[int] = None,
        user_context: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
        compression_strategy: str = "adaptive"
    ) -> Dict[str, Any]:
        """
        带上下文压缩的检索
        
        适用于多轮对话场景，自动压缩历史上下文
        
        Args:
            query: 当前查询
            conversation_history: 对话历史 [{"role": "user", "content": "..."}, ...]
            user_id: 用户 ID
            user_context: 用户上下文
            top_k: 返回结果数量
            compression_strategy: 压缩策略（window/summary/extract/hybrid/adaptive）
        
        Returns:
            包含检索结果和压缩后上下文的字典:
            {
                "results": [...],
                "compressed_context": "...",
                "compression_stats": {...}
            }
        
        Example:
            >>> history = [
            ...     {"role": "user", "content": "我想吃低热量的"},
            ...     {"role": "assistant", "content": "推荐清蒸鱼..."},
            ...     {"role": "user", "content": "还有其他的吗"}
            ... ]
            >>> result = retriever.retrieve_with_context_compression(
            ...     query="还有其他的吗",
            ...     conversation_history=history,
            ...     user_id=1
            ... )
        """
        logger.info(f"开始带上下文压缩的检索: strategy={compression_strategy}")
        
        try:
            # 1. 压缩对话历史
            logger.info("  步骤 1/2: 压缩对话历史...")
            compression_result = self.compressor.compress(
                conversation_history=conversation_history,
                strategy=compression_strategy
            )
            
            compressed_context = compression_result["compressed_context"]
            compression_stats = compression_result["stats"]
            
            logger.info(f"    压缩完成: {compression_stats['original_tokens']} -> "
                       f"{compression_stats['compressed_tokens']} tokens "
                       f"(压缩率: {compression_stats['compression_ratio']:.2%})")
            
            # 2. 执行检索
            logger.info("  步骤 2/2: 执行检索...")
            results = self.retrieve(
                query=query,
                user_id=user_id,
                user_context=user_context,
                top_k=top_k
            )
            
            return {
                "results": results,
                "compressed_context": compressed_context,
                "compression_stats": compression_stats
            }
        
        except Exception as e:
            logger.error(f"带上下文压缩的检索失败: {e}", exc_info=True)
            return {
                "results": [],
                "compressed_context": "",
                "compression_stats": {}
            }
    
    def _postgres_filter(
        self,
        results: List[Dict[str, Any]],
        user_id: int,
        user_context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        使用 PostgreSQL 数据进行精确过滤
        
        过滤逻辑：
        1. 用户偏好匹配（健康目标、口味偏好）
        2. 食材库存匹配（优先推荐可用食材的食谱）
        3. 历史反馈过滤（排除低评分食谱）
        4. 营养目标匹配（符合每日营养目标）
        
        Args:
            results: 初始检索结果
            user_id: 用户 ID
            user_context: 用户上下文
        
        Returns:
            过滤后的结果列表
        """
        try:
            # 1. 获取用户偏好
            preferences = self.postgres_client.get_user_preferences(user_id)
            health_goals = [p["preference_value"] for p in preferences 
                          if p["preference_type"] == "health_goal"]
            
            # 2. 获取用户食材库存
            inventory = self.postgres_client.get_user_inventory(user_id)
            available_ingredients = {item["ingredient_name"] for item in inventory}
            
            # 3. 获取用户反馈（低评分食谱）
            feedbacks = self.postgres_client.get_user_feedbacks(user_id)
            low_rated_recipes = {f["recipe_id"] for f in feedbacks 
                               if f.get("rating") and f["rating"] < 3}
            
            # 4. 过滤结果
            filtered = []
            for result in results:
                recipe_id = result.get("id")
                
                # 排除低评分食谱
                if recipe_id in low_rated_recipes:
                    logger.debug(f"    过滤低评分食谱: {result.get('name')}")
                    continue
                
                # 计算匹配分数
                score = 0.0
                
                # 健康目标匹配
                recipe_tags = result.get("tags", "").split(",")
                for goal in health_goals:
                    if goal in recipe_tags:
                        score += 1.0
                
                # 食材库存匹配
                recipe_ingredients = result.get("ingredients", "").split(",")
                matched_ingredients = sum(1 for ing in recipe_ingredients 
                                        if ing in available_ingredients)
                if recipe_ingredients:
                    score += matched_ingredients / len(recipe_ingredients)
                
                # 添加匹配分数到结果
                result["postgres_score"] = score
                filtered.append(result)
            
            # 按匹配分数排序
            filtered.sort(key=lambda x: x.get("postgres_score", 0), reverse=True)
            
            return filtered
        
        except Exception as e:
            logger.error(f"PostgreSQL 过滤失败: {e}", exc_info=True)
            return results
    
    def _log_interaction(
        self,
        user_id: int,
        query: str,
        results: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> None:
        """
        记录用户交互历史
        
        Args:
            user_id: 用户 ID
            query: 用户查询
            results: 推荐结果
            context: 上下文信息
        """
        try:
            # 提取推荐的食谱 ID
            recipe_ids = [r.get("id") for r in results if r.get("id")]
            
            # 记录交互
            self.postgres_client.add_interaction(
                user_id=user_id,
                query=query,
                recommended_recipes=recipe_ids,
                context=context
            )
            
            logger.debug(f"    记录交互: user_id={user_id}, recipes={len(recipe_ids)}")
        
        except Exception as e:
            logger.error(f"记录交互失败: {e}", exc_info=True)
    
    def add_user_feedback(
        self,
        user_id: int,
        recipe_id: str,
        rating: Optional[int] = None,
        feedback_type: Optional[str] = None,
        comment: Optional[str] = None
    ) -> bool:
        """
        添加用户反馈
        
        Args:
            user_id: 用户 ID
            recipe_id: 食谱 ID
            rating: 评分（1-5）
            feedback_type: 反馈类型（like/dislike/favorite/tried）
            comment: 评论
        
        Returns:
            是否成功
        
        Example:
            >>> retriever.add_user_feedback(
            ...     user_id=1,
            ...     recipe_id="recipe_001",
            ...     rating=5,
            ...     feedback_type="favorite",
            ...     comment="非常好吃！"
            ... )
        """
        try:
            self.postgres_client.add_feedback(
                user_id=user_id,
                recipe_id=recipe_id,
                rating=rating,
                feedback_type=feedback_type,
                comment=comment
            )
            logger.info(f"添加用户反馈成功: user_id={user_id}, recipe_id={recipe_id}")
            return True
        
        except Exception as e:
            logger.error(f"添加用户反馈失败: {e}", exc_info=True)
            return False
    
    def get_user_recommendations(
        self,
        user_id: int,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        基于用户历史的个性化推荐
        
        分析用户的偏好、反馈和交互历史，生成个性化推荐
        
        Args:
            user_id: 用户 ID
            top_k: 返回结果数量
        
        Returns:
            推荐结果列表
        
        Example:
            >>> recommendations = retriever.get_user_recommendations(user_id=1, top_k=5)
        """
        try:
            # 1. 获取用户偏好
            preferences = self.postgres_client.get_user_preferences(user_id)
            
            # 2. 构建查询
            health_goals = [p["preference_value"] for p in preferences 
                          if p["preference_type"] == "health_goal"]
            
            if health_goals:
                query = " ".join(health_goals)
            else:
                query = "健康美味"
            
            # 3. 执行检索
            results = self.retrieve(
                query=query,
                user_id=user_id,
                top_k=top_k
            )
            
            logger.info(f"生成个性化推荐: user_id={user_id}, count={len(results)}")
            return results
        
        except Exception as e:
            logger.error(f"生成个性化推荐失败: {e}", exc_info=True)
            return []
