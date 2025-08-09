#!/usr/bin/env python3
"""
视频转GIF工具 - Streamlit Cloud入口点
专为Streamlit Cloud部署优化的入口文件
"""

import streamlit as st
import sys
import os
import traceback
from pathlib import Path

# 设置页面配置（必须在所有Streamlit操作之前）
st.set_page_config(
    page_title="视频转GIF工具 - Findknow AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def setup_python_path():
    """设置Python导入路径"""
    try:
        # 当前文件在 pages 目录中
        current_dir = Path(__file__).parent.absolute()
        
        # 父目录是项目根目录
        project_root = current_dir.parent.absolute()
        
        # 添加到Python路径
        paths_to_add = [str(project_root), str(current_dir)]
        for path in paths_to_add:
            if path not in sys.path:
                sys.path.insert(0, path)
        
        # 验证实现文件是否存在
        video_to_gif_1_path = current_dir / "video_to_gif_1.py"
        if not video_to_gif_1_path.exists():
            st.error(f"❌ 实现文件不存在: {video_to_gif_1_path}")
            return False
        
        return True
    except Exception as e:
        st.error(f"❌ 设置Python路径失败: {e}")
        return False

def check_dependencies():
    """检查必要的依赖"""
    try:
        import cv2
        import numpy as np
        from PIL import Image
        import openai
        return True
    except ImportError as e:
        st.error(f"❌ 缺少必要依赖: {e}")
        st.info("💡 请确保已安装所有依赖包")
        return False

def load_video_to_gif_module():
    """加载视频转GIF模块"""
    try:
        # 方法1: 尝试导入实际的实现文件
        try:
            # 首先尝试从当前目录导入 video_to_gif_1
            import importlib.util
            current_dir = Path(__file__).parent.absolute()
            module_path = current_dir / "video_to_gif_1.py"
            
            if module_path.exists():
                spec = importlib.util.spec_from_file_location("video_to_gif_1", module_path)
                video_to_gif_1 = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(video_to_gif_1)
                
                if hasattr(video_to_gif_1, 'main'):
                    st.success("✅ 成功导入 video_to_gif_1.py")
                    return video_to_gif_1.main
            
            st.error("❌ 无法找到 video_to_gif_1.py 实现文件")
            return None
            
        except ImportError as e:
            st.error(f"❌ 导入失败: {e}")
            pass
        

            
    except Exception as e:
        st.error(f"❌ 加载模块失败: {e}")
        st.error(f"详细错误: {traceback.format_exc()}")
        return None

def main():
    """主函数"""
    try:
        # 显示加载状态
        with st.spinner("🔄 正在加载视频转GIF工具..."):
            # 1. 设置Python路径
            if not setup_python_path():
                return
            
            # 2. 检查依赖
            if not check_dependencies():
                return
            
            # 3. 加载视频转GIF模块
            video_to_gif_main = load_video_to_gif_module()
            if video_to_gif_main is None:
                return
        
        # 4. 运行主函数
        video_to_gif_main()
        
    except Exception as e:
        st.error(f"❌ 运行视频转GIF工具时出错: {e}")
        st.error(f"详细错误信息:")
        st.code(traceback.format_exc())
        
        # 显示调试信息
        with st.expander("🔍 调试信息"):
            st.write("**当前工作目录:**", os.getcwd())
            st.write("**Python路径:**", sys.path[:5])  # 只显示前5个路径
            st.write("**当前文件位置:**", str(Path(__file__).parent.absolute()))
            
            # 检查文件存在性
            current_dir = Path(__file__).parent.absolute()
            video_gif_1_file = current_dir / "video_to_gif_1.py"
            
            st.write("**文件检查:**")
            st.write(f"- 当前目录 (pages): {current_dir}")
            st.write(f"- video_to_gif_1.py存在: {video_gif_1_file.exists()}")
            
            try:
                files_in_current = list(current_dir.glob("*.py"))
                st.write(f"- 当前目录中的Python文件: {[f.name for f in files_in_current]}")
            except Exception as e:
                st.write(f"- 无法列出当前目录文件: {e}")

if __name__ == "__main__":
    main()
