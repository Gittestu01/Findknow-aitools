#!/usr/bin/env python3
"""
Findknow AI现代工具库启动脚本
"""

import os
import sys
from pathlib import Path

def check_environment():
    """检查运行环境"""
    print("🔍 检查运行环境...")
    
    # 检查Python版本
    if sys.version_info < (3, 8):
        print("❌ Python版本过低，需要3.8或更高版本")
        return False
    
    # 检查必要文件
    required_files = ['app.py', 'config.py', 'utils.py']
    for file in required_files:
        if not Path(file).exists():
            print(f"❌ 缺少必要文件: {file}")
            return False
    
    # 检查临时目录
    temp_dir = Path("temp_uploads")
    temp_dir.mkdir(exist_ok=True)
    
    print("✅ 环境检查完成")
    print("💡 提示：启动后请在界面中设置API密钥")
    return True

def main():
    """主函数"""
    print("🚀 启动Findknow AI现代工具库...")
    
    if not check_environment():
        print("❌ 环境检查失败，请修复问题后重试")
        sys.exit(1)
    
    print("✅ 启动Streamlit应用...")
    print("📱 应用将在浏览器中打开")
    print("🔗 本地地址: http://localhost:8501")
    
    # 启动Streamlit应用
    os.system("streamlit run app.py --server.port 8501 --server.address localhost")

if __name__ == "__main__":
    main()
