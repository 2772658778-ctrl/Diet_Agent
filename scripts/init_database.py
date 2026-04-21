#!/usr/bin/env python3
"""
向量数据库初始化脚本

从 JSON 文件读取食谱数据，初始化向量数据库

Requirements:
- 7.1: 从 JSON 文件加载食谱数据并向量化
- 7.2: 为每个食谱生成描述性文本
- 7.3: 同时存储元数据

Usage:
    python scripts/init_database.py [--recipes-file PATH] [--force]
"""

import sys
import os
import argparse
import io
from pathlib import Path

# 设置 Windows 控制台编码为 UTF-8
if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except Exception:
        pass  # 如果设置失败，继续使用默认编码

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from diet_agent.config import get_settings, validate_config
from diet_agent.integrations.vectorstore import init_vectorstore
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
        text_no_emoji = text_no_emoji.replace('🔧', '[工具]')
        text_no_emoji = text_no_emoji.replace('⚠️', '[警告]')
        text_no_emoji = text_no_emoji.replace('❌', '[错误]')
        text_no_emoji = text_no_emoji.replace('✅', '[成功]')
        text_no_emoji = text_no_emoji.replace('ℹ️', '[信息]')
        text_no_emoji = text_no_emoji.replace('📋', '[列表]')
        print(text_no_emoji)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="初始化智能饮食 Agent 向量数据库",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认测试数据初始化
  python scripts/init_database.py
  
  # 使用指定的食谱文件初始化
  python scripts/init_database.py --recipes-file data/recipes.json
  
  # 强制重新初始化（删除现有数据）
  python scripts/init_database.py --force
        """
    )
    
    parser.add_argument(
        '--recipes-file',
        type=str,
        default='data/recipes_test.json',
        help='食谱 JSON 文件路径（默认: data/recipes_test.json）'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='强制重新初始化，删除现有数据'
    )
    
    return parser.parse_args()


def print_progress(message: str, step: int = 0, total: int = 0):
    """打印进度信息
    
    Args:
        message: 进度消息
        step: 当前步骤
        total: 总步骤数
    """
    if total > 0:
        percentage = (step / total) * 100
        safe_print(f"[{step}/{total}] ({percentage:.1f}%) {message}")
    else:
        safe_print(f"• {message}")


def main():
    """主函数 - 初始化向量数据库
    
    Requirements: 7.1, 7.2, 7.3
    """
    args = parse_args()
    
    safe_print("\n" + "=" * 60)
    safe_print("🔧 智能饮食 Agent - 向量数据库初始化")
    safe_print("=" * 60 + "\n")
    
    try:
        # 1. 验证配置
        print_progress("验证配置...", 1, 5)
        if not validate_config():
            safe_print("\n❌ 配置验证失败")
            safe_print("请检查 .env 文件中的配置，确保 DASHSCOPE_API_KEY 已设置")
            return 1
        
        settings = get_settings()
        logger.info(f"配置加载成功: embedding_model={settings.embedding_model}")
        safe_print(f"✅ 配置验证成功")
        safe_print(f"   - Embedding 模型: {settings.embedding_model}")
        safe_print(f"   - 数据库路径: {settings.chroma_db_path}")
        safe_print(f"   - Collection 名称: {settings.collection_name}\n")
        
        # 2. 检查食谱文件
        print_progress("检查食谱文件...", 2, 5)
        recipes_file = args.recipes_file
        
        # 如果是相对路径，转换为绝对路径
        if not os.path.isabs(recipes_file):
            recipes_file = os.path.join(project_root, recipes_file)
        
        if not os.path.exists(recipes_file):
            safe_print(f"\n❌ 食谱文件不存在: {recipes_file}")
            safe_print("\n可用的食谱文件：")
            data_dir = os.path.join(project_root, 'data')
            if os.path.exists(data_dir):
                for file in os.listdir(data_dir):
                    if file.endswith('.json'):
                        safe_print(f"  • data/{file}")
            return 1
        
        safe_print(f"✅ 找到食谱文件: {recipes_file}\n")
        
        # 3. 显示初始化模式
        print_progress("准备初始化...", 3, 5)
        if args.force:
            safe_print("⚠️  强制模式：将删除现有数据并重新初始化")
            confirm = input("确认继续？(y/N): ").strip().lower()
            if confirm != 'y':
                safe_print("\n❌ 已取消初始化")
                return 0
        else:
            safe_print("ℹ️  增量模式：如果数据库已存在，将跳过初始化")
        safe_print("")
        
        # 4. 初始化向量数据库
        print_progress("初始化向量数据库...", 4, 5)
        safe_print("   这可能需要几分钟，请耐心等待...\n")
        
        try:
            vectorstore = init_vectorstore(
                recipes_file=recipes_file,
                force_reload=args.force
            )
            
            safe_print("✅ 向量数据库初始化成功\n")
            
        except FileNotFoundError as e:
            safe_print(f"\n❌ 文件错误: {e}")
            return 1
        except ValueError as e:
            safe_print(f"\n❌ 数据格式错误: {e}")
            return 1
        except Exception as e:
            logger.error(f"初始化失败: {e}", exc_info=True)
            safe_print(f"\n❌ 初始化失败: {e}")
            safe_print("请检查日志文件获取详细信息")
            return 1
        
        # 5. 验证初始化结果
        print_progress("验证初始化结果...", 5, 5)
        try:
            # 执行测试查询
            test_results = vectorstore.similarity_search("快手菜", k=3)
            
            if test_results:
                safe_print(f"✅ 验证成功，找到 {len(test_results)} 条测试结果\n")
                
                # 显示示例结果
                safe_print("📋 示例食谱：")
                for i, doc in enumerate(test_results[:3], 1):
                    metadata = doc.metadata
                    safe_print(f"   {i}. {metadata.get('name', 'Unknown')}")
                    safe_print(f"      时间: {metadata.get('time', 0)}分钟")
                    safe_print(f"      难度: {metadata.get('difficulty', 'Unknown')}")
                safe_print("")
            else:
                safe_print("⚠️  验证警告：未找到测试结果\n")
                
        except Exception as e:
            logger.warning(f"验证失败: {e}")
            safe_print(f"⚠️  验证警告: {e}\n")
        
        # 6. 完成
        safe_print("=" * 60)
        safe_print("✅ 初始化完成！")
        safe_print("=" * 60)
        safe_print("\n现在可以运行主程序：")
        safe_print("  python main.py")
        safe_print("")
        
        return 0
        
    except KeyboardInterrupt:
        safe_print("\n\n❌ 初始化被中断")
        return 1
    
    except Exception as e:
        logger.error(f"初始化脚本运行失败: {e}", exc_info=True)
        safe_print(f"\n❌ 初始化脚本运行失败: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

