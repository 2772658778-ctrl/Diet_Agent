"""
查询变换模块

提供三种 LLM 驱动的查询变换策略，用于提升检索召回率：
1. Multi-Query: 一个问题生成多个检索视角
2. HyDE: 先生成假设答案再检索
3. Step-Back: 从具体问题提升到抽象问题

复用模块:
- src/graph/llm.py::get_graph_llm()
- src/config.py::get_settings()
- src/utils/logger.py::get_logger()
"""

from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage

from ..graph.llm import get_graph_llm
from ..config import get_settings
from ..utils.logger import get_logger


logger = get_logger(__name__)


class QueryTransformer:
    """查询变换器

    提供三种查询变换策略，用于提升检索召回率：
    1. Multi-Query: 一个问题生成多个检索视角
    2. HyDE: 先生成假设答案再检索
    3. Step-Back: 从具体问题提升到抽象问题

    复用模块:
    - src/graph/llm.py::get_graph_llm()
    - src/config.py::get_settings()
    - src/utils/logger.py::get_logger()
    """

    def __init__(self) -> None:
        """初始化查询变换器，从配置读取默认参数。"""
        self._settings = get_settings()

    def multi_query(self, query: str, n: Optional[int] = None) -> list[str]:
        """将原始查询改写为 n 个不同检索视角的查询。

        Args:
            query: 原始查询文本
            n: 生成的查询数量，默认从 config.rag_multi_query_count 读取

        Returns:
            改写后的查询列表（长度为 n），异常时返回 [query]
        """
        count = n if n is not None else self._settings.rag_multi_query_count
        logger.info(f"Multi-Query 变换: query='{query[:50]}', n={count}")

        system_prompt = (
            "你是一个查询改写专家。对于用户给出的食谱/饮食相关问题，"
            f"请从 {count} 个不同的检索视角改写该问题，每行输出一个改写后的查询，不要编号，不要解释。"
        )

        try:
            llm = get_graph_llm()
            response = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"原始查询：{query}"),
            ])
            raw_text = response.content.strip()
            queries = [line.strip() for line in raw_text.splitlines() if line.strip()]
            # 确保返回 count 条，不足则用原始查询补齐
            if not queries:
                queries = [query]
            queries = queries[:count]
            while len(queries) < count:
                queries.append(query)
            logger.info(f"Multi-Query 变换完成，生成 {len(queries)} 条查询")
            return queries
        except Exception as e:
            logger.error(f"Multi-Query 变换失败，降级返回原始查询: {e}", exc_info=True)
            return [query]

    def hyde(self, query: str) -> str:
        """生成假设性文档（HyDE），供后续 embedding 检索使用。

        Args:
            query: 原始查询文本

        Returns:
            生成的假设文档文本，异常时返回原始查询
        """
        logger.info(f"HyDE 变换: query='{query[:50]}'")

        system_prompt = (
            "你是一个食谱专家。请根据用户的查询，用 2-3 句话写一段假设性的食谱描述，"
            "描述一道符合该查询条件的菜品（包含食材、做法简述、口味特点）。直接输出描述，不要标题。"
        )

        try:
            llm = get_graph_llm()
            response = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"查询：{query}"),
            ])
            hypothesis = response.content.strip()
            if not hypothesis:
                return query
            logger.info(f"HyDE 生成假设文档，长度: {len(hypothesis)}")
            return hypothesis
        except Exception as e:
            logger.error(f"HyDE 变换失败，降级返回原始查询: {e}", exc_info=True)
            return query

    def step_back(self, query: str) -> str:
        """将具体问题提升为更抽象的问题（Step-Back Prompting）。

        Args:
            query: 具体查询文本

        Returns:
            抽象化后的查询，异常时返回原始查询
        """
        logger.info(f"Step-Back 变换: query='{query[:50]}'")

        system_prompt = (
            "你是一个查询抽象专家。请将用户的具体饮食问题提升为一个更抽象、更通用的问题，"
            "从而帮助检索到更全面的背景知识。只输出抽象后的问题，不要解释。\n"
            "示例：'鸡蛋和牛奶能一起吃吗' → '蛋白质食材的搭配原则'"
        )

        try:
            llm = get_graph_llm()
            response = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"具体问题：{query}"),
            ])
            abstract_query = response.content.strip()
            if not abstract_query:
                return query
            logger.info(f"Step-Back 变换完成: '{abstract_query[:80]}'")
            return abstract_query
        except Exception as e:
            logger.error(f"Step-Back 变换失败，降级返回原始查询: {e}", exc_info=True)
            return query

    def transform(self, query: str, strategy: str = "multi_query") -> list[str]:
        """统一变换入口，根据 strategy 调用对应方法。

        Args:
            query: 原始查询文本
            strategy: 变换策略，可选 'multi_query' / 'hyde' / 'step_back'

        Returns:
            变换后的查询列表：
            - multi_query → [q1, q2, ..., qn]
            - hyde        → [假设文档]
            - step_back   → [原始查询, 抽象查询]

        Raises:
            ValueError: strategy 不合法时
        """
        logger.info(f"查询变换: strategy='{strategy}', query='{query[:50]}'")

        if strategy == "multi_query":
            return self.multi_query(query)
        elif strategy == "hyde":
            return [self.hyde(query)]
        elif strategy == "step_back":
            abstract = self.step_back(query)
            return [query, abstract] if abstract != query else [query]
        else:
            raise ValueError(
                f"不支持的变换策略: '{strategy}'，"
                "可选值为 'multi_query' / 'hyde' / 'step_back'"
            )


# 模块级单例
_query_transformer: Optional[QueryTransformer] = None


def get_query_transformer() -> QueryTransformer:
    """获取 QueryTransformer 单例。

    Returns:
        QueryTransformer 实例
    """
    global _query_transformer
    if _query_transformer is None:
        _query_transformer = QueryTransformer()
        logger.info("QueryTransformer 初始化完成")
    return _query_transformer
