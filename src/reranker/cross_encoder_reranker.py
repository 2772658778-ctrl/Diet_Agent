"""
Cross-Encoder 精排器
在粗排（向量检索 + 关系推理）后进行精排，提升推荐准确性
"""

from typing import List, Dict, Any, Optional
from sentence_transformers import CrossEncoder
import numpy as np

from ..utils.logger import get_logger


logger = get_logger(__name__)


class CrossEncoderReranker:
    """Cross-Encoder 精排器"""
    
    def __init__(
        self,
        model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2',
        batch_size: int = 32
    ):
        """
        初始化 Cross-Encoder 精排器
        
        Args:
            model_name: 模型名称
                - 'cross-encoder/ms-marco-MiniLM-L-6-v2': 快速，适合生产
                - 'BAAI/bge-reranker-base': 中文优化
                - 'BAAI/bge-reranker-large': 最高精度
            batch_size: 批处理大小
        """
        logger.info(f"加载 Cross-Encoder 模型: {model_name}")
        self.model = CrossEncoder(model_name)
        self.batch_size = batch_size
        logger.info("Cross-Encoder 模型加载完成")
    
    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        精排文档
        
        Args:
            query: 查询文本
            documents: 文档列表，每个文档需要包含 'text' 或 'name' + 'description' 字段
            top_k: 返回前 K 个结果
            score_threshold: 分数阈值，低于此分数的文档将被过滤
        
        Returns:
            重新排序后的文档列表
        """
        if not documents:
            return []
        
        logger.info(f"开始精排: query='{query}', documents={len(documents)}")
        
        # 1. 构建查询-文档对
        pairs = []
        for doc in documents:
            # 提取文档文本
            if 'text' in doc:
                doc_text = doc['text']
            elif 'name' in doc and 'description' in doc:
                doc_text = f"{doc['name']}: {doc['description']}"
            elif 'name' in doc:
                doc_text = doc['name']
            else:
                logger.warning(f"文档缺少文本字段: {doc}")
                doc_text = str(doc)
            
            pairs.append((query, doc_text))
        
        # 2. 批量计算相关性分数
        scores = self._batch_predict(pairs)
        
        # 3. 添加精排分数到文档
        for doc, score in zip(documents, scores):
            doc['rerank_score'] = float(score)
        
        # 4. 按分数排序
        reranked_docs = sorted(
            documents,
            key=lambda x: x['rerank_score'],
            reverse=True
        )
        
        # 5. 过滤低分文档
        if score_threshold is not None:
            reranked_docs = [
                doc for doc in reranked_docs
                if doc['rerank_score'] >= score_threshold
            ]
        
        # 6. 返回 Top-K
        if top_k is not None:
            reranked_docs = reranked_docs[:top_k]
        
        logger.info(f"精排完成: 返回 {len(reranked_docs)} 个结果")
        
        return reranked_docs
    
    def _batch_predict(self, pairs: List[tuple]) -> np.ndarray:
        """批量预测相关性分数"""
        all_scores = []
        
        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i:i+self.batch_size]
            batch_scores = self.model.predict(batch)
            all_scores.extend(batch_scores)
        
        return np.array(all_scores)
    
    def rerank_with_fusion(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        alpha: float = 0.7,
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        融合原始分数和精排分数
        
        Args:
            query: 查询文本
            documents: 文档列表（需要包含 'score' 字段）
            alpha: 融合权重，alpha * rerank_score + (1-alpha) * original_score
            top_k: 返回前 K 个结果
        
        Returns:
            重新排序后的文档列表
        """
        # 1. 精排
        reranked_docs = self.rerank(query, documents)
        
        # 2. 归一化原始分数
        if any('score' in doc for doc in reranked_docs):
            original_scores = [doc.get('score', 0) for doc in reranked_docs]
            max_score = max(original_scores) if original_scores else 1.0
            min_score = min(original_scores) if original_scores else 0.0
            score_range = max_score - min_score if max_score > min_score else 1.0
            
            normalized_scores = [
                (score - min_score) / score_range
                for score in original_scores
            ]
        else:
            normalized_scores = [0.0] * len(reranked_docs)
        
        # 3. 归一化精排分数
        rerank_scores = [doc['rerank_score'] for doc in reranked_docs]
        max_rerank = max(rerank_scores) if rerank_scores else 1.0
        min_rerank = min(rerank_scores) if rerank_scores else 0.0
        rerank_range = max_rerank - min_rerank if max_rerank > min_rerank else 1.0
        
        normalized_rerank_scores = [
            (score - min_rerank) / rerank_range
            for score in rerank_scores
        ]
        
        # 4. 融合分数
        for doc, orig_score, rerank_score in zip(
            reranked_docs, normalized_scores, normalized_rerank_scores
        ):
            doc['fused_score'] = alpha * rerank_score + (1 - alpha) * orig_score
        
        # 5. 按融合分数排序
        fused_docs = sorted(
            reranked_docs,
            key=lambda x: x['fused_score'],
            reverse=True
        )
        
        # 6. 返回 Top-K
        if top_k is not None:
            fused_docs = fused_docs[:top_k]
        
        return fused_docs


# 全局精排器实例
_reranker: Optional[CrossEncoderReranker] = None


def get_reranker(
    model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
) -> CrossEncoderReranker:
    """获取精排器实例（单例）"""
    global _reranker
    
    if _reranker is None:
        _reranker = CrossEncoderReranker(model_name=model_name)
    
    return _reranker
