#!/usr/bin/env python3
"""
智能饮食 Agent - 命令行界面

提供交互式命令行界面，支持多轮对话

Requirements:
- 9.1: 在 3 秒内返回响应
- 9.2: 向量检索在 1 秒内完成

Usage:
    python main.py
"""

import sys
import io

# 设置 Windows 控制台编码为 UTF-8
if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except Exception:
        pass  # 如果设置失败，继续使用默认编码

from diet_agent.config import get_settings, validate_config
from diet_agent.integrations.vectorstore import get_vectorstore, init_vectorstore
from diet_agent.legacy import get_agent_session
from diet_agent.logging import get_logger


logger = get_logger(__name__)


def safe_print(text: str):
    """安全打印，处理 Windows 控制台编码问题
    
    Args:
        text: 要打印的文本
    """
    try:
        print(text)
    except UnicodeEncodeError:
        # 如果遇到编码错误，移除 emoji 字符
        import re
        text_no_emoji = re.sub(r'[^\u0000-\uFFFF]', '', text)
        # 替换常见 emoji
        text_no_emoji = text_no_emoji.replace('🍽️', '[食物]')
        text_no_emoji = text_no_emoji.replace('⚠️', '[警告]')
        text_no_emoji = text_no_emoji.replace('❌', '[错误]')
        text_no_emoji = text_no_emoji.replace('✅', '[成功]')
        text_no_emoji = text_no_emoji.replace('🔄', '[刷新]')
        text_no_emoji = text_no_emoji.replace('👋', '[再见]')
        text_no_emoji = text_no_emoji.replace('🤔', '[思考]')
        text_no_emoji = text_no_emoji.replace('📖', '[帮助]')
        print(text_no_emoji)


def print_welcome():
    """打印欢迎信息"""
    safe_print("\n" + "=" * 60)
    safe_print("🍽️  智能饮食 Agent - 您的专业饮食助手")
    safe_print("=" * 60)
    safe_print("\n我可以帮您：")
    safe_print("  • 根据口味和时间推荐食谱")
    safe_print("  • 分析食谱的营养成分")
    safe_print("  • 查询食材信息和搭配建议")
    safe_print("\n提示：")
    safe_print("  • 输入 'exit' 或 'quit' 退出程序")
    safe_print("  • 输入 'reset' 重置对话历史")
    safe_print("  • 输入 'help' 查看帮助信息")
    safe_print("=" * 60 + "\n")


def print_help():
    """打印帮助信息"""
    safe_print("\n" + "-" * 60)
    safe_print("📖 帮助信息")
    safe_print("-" * 60)
    safe_print("\n可用命令：")
    safe_print("  exit, quit  - 退出程序")
    safe_print("  reset       - 重置对话历史，开始新的会话")
    safe_print("  help        - 显示此帮助信息")
    safe_print("\n使用示例：")
    safe_print("  • 我想吃酸甜口味的快手菜")
    safe_print("  • 30分钟内能做什么菜？")
    safe_print("  • 番茄炒蛋的营养怎么样？")
    safe_print("  • 鸡胸肉和西兰花搭配好吗？")
    safe_print("-" * 60 + "\n")


def check_vectorstore():
    """检查向量数据库是否已初始化
    
    Returns:
        bool: 向量数据库是否可用
    """
    try:
        vectorstore = get_vectorstore()
        if vectorstore is None:
            # 尝试从磁盘加载已存在的数据库
            settings = get_settings()
            import os
            db_path = settings.chroma_db_path
            db_file = os.path.join(db_path, "chroma.sqlite3")
            
            if os.path.exists(db_file):
                # 数据库文件存在，尝试加载
                logger.info("检测到已存在的向量数据库，正在加载...")
                safe_print("\n🔄 检测到已存在的向量数据库，正在加载...")
                
                try:
                    # 使用测试数据文件路径（不会重新加载数据，只是连接到现有数据库）
                    recipes_file = os.path.join(os.path.dirname(__file__), "data", "recipes_test.json")
                    vectorstore = init_vectorstore(recipes_file, force_reload=False)
                    safe_print("✅ 向量数据库加载成功\n")
                    return True
                except Exception as e:
                    logger.error(f"加载向量数据库失败: {e}", exc_info=True)
                    safe_print(f"\n❌ 加载向量数据库失败: {e}")
                    return False
            else:
                # 数据库文件不存在
                safe_print("\n⚠️  向量数据库未初始化")
                safe_print("请先运行初始化脚本：")
                safe_print("  python scripts/init_database.py")
                safe_print("\n或者使用测试数据初始化：")
                safe_print("  python init_test_vectorstore.py")
                return False
        return True
    except Exception as e:
        logger.error(f"检查向量数据库失败: {e}", exc_info=True)
        safe_print(f"\n❌ 向量数据库检查失败: {e}")
        return False


def main():
    """主函数 - 运行交互式命令行界面
    
    Requirements: 9.1, 9.2
    """
    try:
        # 1. 验证配置
        logger.info("启动智能饮食 Agent CLI...")
        if not validate_config():
            safe_print("\n❌ 配置验证失败")
            safe_print("请检查 .env 文件中的配置，确保 DASHSCOPE_API_KEY 已设置")
            return 1
        
        settings = get_settings()
        logger.info(f"配置加载成功: model={settings.llm_model}")
        
        # 2. 检查向量数据库
        if not check_vectorstore():
            return 1
        
        # 3. 初始化 Agent 会话
        safe_print("\n🔄 正在初始化 Agent...")
        try:
            agent_session = get_agent_session()
            safe_print("✅ Agent 初始化成功\n")
        except Exception as e:
            logger.error(f"Agent 初始化失败: {e}", exc_info=True)
            safe_print(f"\n❌ Agent 初始化失败: {e}")
            safe_print("请检查配置和网络连接")
            return 1
        
        # 4. 打印欢迎信息
        print_welcome()
        
        # 5. 主循环 - 多轮对话
        while True:
            try:
                # 获取用户输入
                user_input = input("您: ").strip()
                
                # 处理空输入
                if not user_input:
                    continue
                
                # 处理命令
                if user_input.lower() in ['exit', 'quit']:
                    safe_print("\n👋 感谢使用智能饮食 Agent，再见！\n")
                    break
                
                if user_input.lower() == 'reset':
                    agent_session.reset()
                    safe_print("\n🔄 对话历史已重置\n")
                    continue
                
                if user_input.lower() == 'help':
                    print_help()
                    continue
                
                # 调用 Agent 处理用户输入
                safe_print("\n🤔 思考中...\n")
                response = agent_session.chat(user_input)
                
                # 显示 Agent 回复
                safe_print(f"Agent: {response}\n")
                safe_print("-" * 60 + "\n")
                
            except KeyboardInterrupt:
                safe_print("\n\n👋 检测到中断信号，退出程序\n")
                break
            
            except Exception as e:
                logger.error(f"处理用户输入时出错: {e}", exc_info=True)
                safe_print(f"\n❌ 抱歉，处理您的请求时出现错误: {e}")
                safe_print("请重试或输入 'help' 查看帮助\n")
        
        return 0
        
    except Exception as e:
        logger.error(f"程序运行失败: {e}", exc_info=True)
        safe_print(f"\n❌ 程序运行失败: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
