"""
智能饮食 Agent 主逻辑

集成 LLM、工具和对话记忆，创建完整的 Agent 系统

Requirements:
- 4.1: 正确识别用户意图并调用工具
- 4.2: 提取查询中的多个条件
- 5.1: 记住之前的对话内容
- 5.2: 正确理解上下文引用
- 5.3: 结合之前的查询和新条件
- 5.4: 保持长对话连贯性
- 9.1: 在 3 秒内返回响应（通过 max_iterations 控制）
- 9.2: 向量检索在 1 秒内完成
"""

from typing import Optional, Dict, Any, List
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from ..llm.qwen_client import get_llm
from ..tools.recipe_tools import search_recipes
from ..tools.nutrition_tools import analyze_nutrition
from ..tools.ingredient_tools import check_ingredients
from .prompts import get_system_prompt
from ..config import get_settings
from ..utils.logger import get_logger


logger = get_logger(__name__)


def create_diet_agent(
    verbose: Optional[bool] = None,
    max_iterations: Optional[int] = None,
    use_memory: bool = True
):
    """
    创建智能饮食 Agent
    
    集成 LLM、工具和对话记忆，构建完整的 Agent 系统
    
    Args:
        verbose: 是否显示详细日志，默认从配置读取
        max_iterations: 最大迭代次数，默认从配置读取
        use_memory: 是否使用对话记忆
    
    Returns:
        AgentExecutor 实例
    
    Requirements: 4.1, 4.2, 5.1, 5.2, 5.3, 5.4, 9.1, 9.2
    
    Example:
        >>> agent = create_diet_agent()
        >>> response = agent.invoke({"input": "我想吃酸甜的"})
    """
    try:
        logger.info("开始创建智能饮食 Agent...")
        
        # 1. 获取配置
        settings = get_settings()
        verbose = verbose if verbose is not None else settings.verbose
        max_iterations = max_iterations if max_iterations is not None else settings.max_iterations
        
        logger.info(f"Agent 配置: verbose={verbose}, max_iterations={max_iterations}, use_memory={use_memory}")
        
        # 2. 获取 LLM
        logger.info("初始化 LLM...")
        llm = get_llm()
        
        # 3. 准备工具列表
        logger.info("准备工具列表...")
        tools = [
            search_recipes,
            analyze_nutrition,
            check_ingredients
        ]
        logger.info(f"已加载 {len(tools)} 个工具: {[tool.name for tool in tools]}")
        
        # 4. 获取系统提示词
        logger.info("加载系统提示词...")
        system_prompt_text = get_system_prompt(simple=False)
        
        # 5. 创建对话记忆（如果启用）
        memory = None
        if use_memory:
            logger.info("初始化对话记忆 (ConversationBufferMemory)...")
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="output"
            )
            logger.info("对话记忆初始化完成")
        
        # 6. 创建 Prompt 模板
        logger.info("创建 Prompt 模板...")
        prompt_messages = [
            ("system", system_prompt_text),
        ]
        
        # 如果使用记忆，添加 chat_history placeholder
        if use_memory:
            prompt_messages.append(MessagesPlaceholder(variable_name="chat_history"))
        
        prompt_messages.extend([
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        prompt = ChatPromptTemplate.from_messages(prompt_messages)
        
        # 7. 创建 Agent
        logger.info("创建 Tool Calling Agent...")
        agent = create_tool_calling_agent(llm, tools, prompt)
        
        # 8. 创建 AgentExecutor
        logger.info("创建 AgentExecutor...")
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            memory=memory,
            verbose=verbose,
            max_iterations=max_iterations,
            handle_parsing_errors=True
        )
        
        logger.info("智能饮食 Agent 创建成功！")
        
        return agent_executor
        
    except Exception as e:
        logger.error(f"创建 Agent 失败: {e}", exc_info=True)
        raise Exception(f"无法创建智能饮食 Agent: {str(e)}")


class DietAgentSession:
    """
    智能饮食 Agent 会话管理类
    
    提供会话级别的 Agent 管理，支持多轮对话
    
    Requirements: 5.1, 5.2, 5.3, 5.4
    """
    
    def __init__(
        self,
        verbose: Optional[bool] = None,
        max_iterations: Optional[int] = None
    ):
        """
        初始化 Agent 会话
        
        Args:
            verbose: 是否显示详细日志
            max_iterations: 最大迭代次数
        """
        logger.info("初始化 Agent 会话...")
        
        self.agent = create_diet_agent(
            verbose=verbose,
            max_iterations=max_iterations,
            use_memory=True  # 会话模式始终使用记忆
        )
        
        self.conversation_count = 0
        logger.info("Agent 会话初始化完成")
    
    def chat(self, user_input: str) -> str:
        """
        与 Agent 对话
        
        Args:
            user_input: 用户输入
        
        Returns:
            Agent 回复
        
        Requirements: 5.1, 5.2, 5.3, 5.4, 8.1, 8.2, 8.3, 8.4, 8.5
        """
        try:
            # 输入验证
            if not user_input or not user_input.strip():
                logger.warning("收到空输入")
                return "请输入您的问题或需求。"
            
            self.conversation_count += 1
            logger.info(f"对话轮次 {self.conversation_count}: 用户输入='{user_input}'")
            
            # 调用 Agent（ConversationBufferMemory 会自动管理历史）
            try:
                response = self.agent.invoke({"input": user_input})
            except TimeoutError as e:
                logger.error(f"Agent 调用超时: {e}", exc_info=True)
                return "抱歉，处理您的请求超时了。请尝试简化您的问题或稍后再试。"
            except Exception as e:
                logger.error(f"Agent 调用失败: {e}", exc_info=True)
                
                # 尝试降级响应
                if "api" in str(e).lower() or "connection" in str(e).lower():
                    return "抱歉，连接服务时遇到问题。请检查网络连接或稍后再试。"
                elif "rate limit" in str(e).lower():
                    return "抱歉，请求过于频繁。请稍等片刻后再试。"
                else:
                    return "抱歉，我遇到了一些问题。请稍后再试或重新描述您的需求。"
            
            # 提取输出
            output = response.get("output", "")
            
            # 验证输出
            if not output or len(output.strip()) == 0:
                logger.warning("Agent 返回空输出")
                return "抱歉，我没能理解您的需求。能否换个方式描述一下？"
            
            logger.info(f"对话轮次 {self.conversation_count}: Agent 回复长度={len(output)}")
            logger.debug(f"Agent 回复: {output}")
            
            return output
            
        except Exception as e:
            logger.error(f"对话处理失败: {e}", exc_info=True)
            return "抱歉，我遇到了一些问题。请稍后再试或重新描述您的需求。"
    
    def reset(self):
        """
        重置会话（清除对话记忆）
        
        Requirements: 5.1
        """
        logger.info("重置 Agent 会话...")
        
        # 重新创建 Agent（会清除记忆）
        self.agent = create_diet_agent(use_memory=True)
        self.conversation_count = 0
        
        logger.info("Agent 会话已重置")
    
    def get_conversation_count(self) -> int:
        """
        获取对话轮次
        
        Returns:
            对话轮次数
        """
        return self.conversation_count


def test_agent() -> bool:
    """
    测试 Agent 是否正常工作
    
    Returns:
        bool: 测试是否成功
    
    Requirements: 4.1, 4.2
    """
    try:
        logger.info("开始测试 Agent...")
        
        # 创建 Agent
        agent = create_diet_agent(verbose=False, use_memory=False)
        
        # 测试简单查询
        test_input = "推荐一道快手菜"
        logger.info(f"测试输入: {test_input}")
        
        response = agent.invoke({"input": test_input})
        
        # 提取输出
        output = response.get("output", "")
        
        logger.info(f"测试输出长度: {len(output)}")
        logger.debug(f"测试输出: {output}")
        
        # 检查是否有输出
        if output and len(output) > 0:
            logger.info("Agent 测试成功")
            return True
        else:
            logger.error("Agent 测试失败: 输出为空")
            return False
            
    except Exception as e:
        logger.error(f"Agent 测试失败: {e}", exc_info=True)
        return False


# 全局 Agent 会话实例（单例模式）
_agent_session: Optional[DietAgentSession] = None


def get_agent_session(force_reload: bool = False) -> DietAgentSession:
    """
    获取 Agent 会话实例（单例模式）
    
    Args:
        force_reload: 是否强制重新创建实例
    
    Returns:
        DietAgentSession 实例
    
    Requirements: 5.1, 5.2, 5.3, 5.4
    """
    global _agent_session
    
    if _agent_session is None or force_reload:
        logger.info("创建新的 Agent 会话实例")
        _agent_session = DietAgentSession()
    
    return _agent_session
