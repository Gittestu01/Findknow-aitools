"""
Findknow AI现代工具库配置文件
"""

import os
from pathlib import Path

# 基础配置
APP_NAME = "Findknow AI现代工具库"
APP_VERSION = "1.0.0"
APP_DESCRIPTION = "一个基于Streamlit开发的AI智能工具库"

# 页面配置
PAGE_CONFIG = {
    "page_title": APP_NAME,
    "page_icon": "🤖",
    "layout": "wide",
    "initial_sidebar_state": "collapsed"
}

# AI模型配置
AI_MODELS = {
    "qwen-plus": "Qwen-Plus",
    "qwen-long": "Qwen-Long",
    "qwen-turbo": "Qwen-Turbo",
    "deepseek-v3": "Deepseek-v3",
    "gpt-4": "GPT-4",
    "claude-3": "Claude-3"
}

# 默认模型
DEFAULT_MODEL = "qwen-plus"

# API配置
API_CONFIG = {
    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "timeout": 30,  # 添加超时配置
    "max_retries": 3  # 添加重试配置
}

# 文件上传配置
UPLOAD_CONFIG = {
    "allowed_types": ['txt', 'pdf', 'docx', 'doc'],
    "max_file_size": 10 * 1024 * 1024,  # 10MB
    "temp_dir": Path("temp_uploads")
}

# 工具配置
TOOLS_CONFIG = {
    "resume_assistant": {
        "name": "简历智能助手",
        "description": "智能分析简历与岗位匹配度，输出标准化表格",
        "icon": "📄",
        "page": "pages/resume_assistant.py"
    },
    "prompt_engineer": {
        "name": "提示词工程师", 
        "description": "专业提示词优化，提升AI对话效果",
        "icon": "🔧",
        "page": "pages/prompt_engineer.py"
    }
}

# 样式配置
STYLE_CONFIG = {
    "primary_color": "#667eea",
    "secondary_color": "#764ba2",
    "gradient": "linear-gradient(90deg, #667eea 0%, #764ba2 100%)",
    "card_style": """
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        color: white;
        transition: transform 0.3s ease;
    """
}

# 飞书配置
FEISHU_CONFIG = {
    "api_base": "https://open.feishu.cn/open-apis",
    "table_api": "/bitable/v1/apps/{app_token}/tables/{table_id}/records"
}

# 错误消息
ERROR_MESSAGES = {
    "api_key_missing": "API密钥未配置，请先设置API密钥",
    "client_init_failed": "初始化AI客户端失败",
    "file_upload_failed": "文件上传失败",
    "analysis_failed": "分析失败",
    "export_failed": "导出失败"
}

# 成功消息
SUCCESS_MESSAGES = {
    "file_uploaded": "文件上传成功",
    "analysis_completed": "分析完成",
    "export_completed": "导出完成",
    "optimization_completed": "优化完成"
}

# 验证函数
def validate_config():
    """验证配置是否正确"""
    errors = []
    
    # 检查临时目录
    temp_dir = UPLOAD_CONFIG["temp_dir"]
    temp_dir.mkdir(exist_ok=True)
    
    return errors

def get_client_config():
    """获取客户端配置"""
    return {
        "base_url": API_CONFIG["base_url"]
    } 