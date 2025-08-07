import streamlit as st
import os
from pathlib import Path
from openai import OpenAI
import pandas as pd
import json
from datetime import datetime
import requests
import re
import time

# 页面配置（必须是第一个Streamlit命令）
st.set_page_config(
    page_title="提示词工程师 - Findknow AI",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        color: white;
    }
    
    .prompt-area {
        background: rgba(255,255,255,0.1);
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .result-area {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        color: #333;
        font-weight: 500;
    }
    
    .placeholder-text {
        color: rgba(255,255,255,0.9);
        font-style: italic;
        font-weight: 500;
    }
    
    .model-selector {
        background: rgba(255,255,255,0.1);
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        color: #333;
    }
    
    .loading-spinner {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid #f3f3f3;
        border-top: 3px solid #667eea;
        border-radius: 50%;
        animation: spin 1s linear infinite;
        margin-right: 10px;
    }
    
    .chat-message {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
        color: #333;
    }
    
    .user-message {
        background: #e3f2fd;
        border-left-color: #2196f3;
        color: #333;
    }
    
    .assistant-message {
        background: #f3e5f5;
        border-left-color: #9c27b0;
        color: #333;
    }
    
    .model-info {
        background: #f0f8ff;
        border-radius: 8px;
        padding: 0.5rem;
        margin: 0.5rem 0;
        border-left: 3px solid #4CAF50;
        color: #333;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.main-header, .main-header * {
    font-family: 'Noto Emoji', 'Twemoji Mozilla', 'Segoe UI Emoji', 'Apple Color Emoji', 'Noto Color Emoji', 'Segoe UI', 'Arial', sans-serif !important;
}
</style>
""", unsafe_allow_html=True)

# 大模型配置
MODEL_CONFIGS = {
    "GPT-4": {
        "description": "OpenAI最强大的多模态模型，擅长复杂推理和创意任务",
        "strengths": ["复杂推理", "创意写作", "代码生成", "多模态理解"],
        "best_for": "需要深度思考和创造力的任务"
    },
    "GPT-3.5-turbo": {
        "description": "OpenAI的高效模型，平衡性能和速度",
        "strengths": ["快速响应", "成本效益", "通用任务", "对话"],
        "best_for": "日常对话和一般性任务"
    },
    "Claude-3": {
        "description": "Anthropic的安全导向模型，擅长分析和写作",
        "strengths": ["安全分析", "长文本处理", "学术写作", "逻辑推理"],
        "best_for": "需要安全性和准确性的任务"
    },
    "Gemini Pro": {
        "description": "Google的多模态模型，擅长创意和视觉任务",
        "strengths": ["多模态", "创意生成", "视觉理解", "实时信息"],
        "best_for": "需要视觉理解和创意的任务"
    },
    "Qwen Plus": {
        "description": "阿里云的通义千问模型，中文理解能力强",
        "strengths": ["中文理解", "本地化", "知识问答", "对话"],
        "best_for": "中文任务和本地化应用"
    },
    "豆包": {
        "description": "字节跳动的智能助手，擅长中文对话和创意内容生成",
        "strengths": ["中文对话", "创意写作", "内容生成", "情感理解"],
        "best_for": "中文创意写作和情感交流任务"
    },
    "DeepSeek-V3": {
        "description": "深度求索的代码专家模型，在编程和数学推理方面表现优异",
        "strengths": ["代码生成", "数学推理", "逻辑分析", "技术文档"],
        "best_for": "编程开发和技术分析任务"
    },
    "DeepSeek-R1": {
        "description": "深度求索的推理专家模型，擅长复杂推理和问题解决",
        "strengths": ["复杂推理", "问题解决", "逻辑分析", "学术研究"],
        "best_for": "需要深度推理和问题解决的任务"
    },
    "秘塔AI": {
        "description": "秘塔科技的智能助手，专注于中文理解和知识问答",
        "strengths": ["中文理解", "知识问答", "信息检索", "学习辅助"],
        "best_for": "中文知识问答和学习辅助任务"
    },
    "自定义模型": {
        "description": "其他特定的大模型",
        "strengths": ["根据具体模型而定"],
        "best_for": "特定领域或专业任务"
    }
}

# 提示词工程师系统提示词
PROMPT_ENGINEER_SYSTEM_PROMPT = """我想让你成为我的 Prompt 创作者。你的目标是帮助创建最佳的 Prompt，这个 Prompt 将由AI大模型使用。

你将遵循以下过程：
首先，你会接收到用户初版的提示词和目标大模型信息，请理解用户需求和内容，并根据特定大模型的特点进行针对性的润色
然后，你需要推断用户真实寻求，并根据初版提示词和大模型的特点，通过不断的重复来改进它，通过则进行下一步。
根据用户输入初版提示词和目标大模型，你会创建两个部分：
a) 修订后的 Prompt (你会编写修订后的 Prompt，应该清晰、精确、易于理解，并针对特定大模型优化)
c）问题（你提出相关问题，询问我需要哪些额外信息来改进 Prompt）

你提供的 Prompt 应该采用用户发出请求的形式，由大模型执行。
用户将和你继续这个迭代过程，用户会提供更多的信息，你会更新"修订后的 Prompt"部分的请求，直到它完整为止。

请确保：
1. 修订后的Prompt清晰、精确、易于理解
2. 针对特定大模型的特点和优势进行优化
3. 考虑目标模型的限制和最佳实践
4. 提出有针对性的问题来进一步完善
5. 保持专业性和实用性"""

def get_model_specific_prompt(selected_model):
    """根据选择的模型生成特定的优化提示"""
    model_info = MODEL_CONFIGS.get(selected_model, MODEL_CONFIGS["自定义模型"])
    
    return f"""
目标大模型：{selected_model}
模型特点：{model_info['description']}
模型优势：{', '.join(model_info['strengths'])}
适用场景：{model_info['best_for']}

请根据以上模型特点，为用户提供针对性的提示词优化建议。
"""

def init_client():
    """初始化AI客户端"""
    try:
        # 从session_state获取API密钥
        api_key = st.session_state.get("api_key")
        if not api_key:
            st.error("请先设置API密钥")
            return None
        
        client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        return client
    except Exception as e:
        st.error(f"初始化AI客户端失败: {e}")
        return None

def check_api_key():
    """检查API密钥是否已设置"""
    return st.session_state.get("api_key") is not None

def setup_api_key():
    """设置API密钥"""
    st.markdown("### 🔑 API密钥设置")
    
    # 如果已经有API密钥，显示当前状态
    if check_api_key():
        st.success("✅ API密钥已设置，可以使用所有功能")
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("🗑️ 清除密钥", use_container_width=True, key="clear_key_prompt"):
                # 清除API密钥
                st.session_state["api_key"] = None
                if "api_key_persistent" in st.session_state:
                    del st.session_state.api_key_persistent
                st.success("✅ API密钥已清除")
                st.rerun()
        with col2:
            st.write("")  # 占位符
    else:
        st.markdown("请输入您的API密钥以使用所有功能：")
        
        # API密钥输入区域
        api_key = st.text_input(
            "API密钥",
            type="password",
            placeholder="请输入您的API密钥...",
            help="请输入您的API密钥以启用AI功能",
            key="api_key_input_prompt"
        )
        
        # 设置按钮
        if st.button("✅ 设置密钥", use_container_width=True, key="set_key_prompt"):
            if api_key:
                # 保存到持久化存储
                st.session_state.api_key_persistent = api_key
                st.session_state["api_key"] = api_key
                st.success("✅ API密钥设置成功！")
                st.rerun()
            else:
                st.error("❌ 请输入有效的API密钥")
    
    st.markdown("---")

def optimize_prompt(client, user_input, selected_model, conversation_history=None):
    """优化提示词"""
    try:
        # 构建优化请求
        model_specific_prompt = get_model_specific_prompt(selected_model)
        
        optimization_prompt = f"""
{model_specific_prompt}

用户需求：
{user_input}

请根据目标大模型的特点，生成一个专业、高效的AI提示词，要求：
1. 清晰明确的目标和角色定义
2. 具体的任务描述和期望输出
3. 适当的约束和指导原则
4. 专业且易于理解的表达
5. 针对目标大模型的特点进行优化
6. 考虑模型的优势和适用场景

请直接输出优化后的提示词，无需其他说明。
"""
        
        messages = [
            {"role": "system", "content": PROMPT_ENGINEER_SYSTEM_PROMPT}
        ]
        
        # 添加对话历史
        if conversation_history:
            for msg in conversation_history:
                messages.append({"role": msg["role"], "content": msg["content"]})
        
        # 添加当前用户输入
        messages.append({"role": "user", "content": optimization_prompt})
        
        completion = client.chat.completions.create(
            model="qwen-plus",
            messages=messages,
            stream=False,
            temperature=0.3,
            max_tokens=1500
        )
        
        response = completion.choices[0].message.content
        return response.strip()
        
    except Exception as e:
        st.error(f"提示词优化失败: {e}")
        return None

def continue_optimization(client, conversation_history, user_input, selected_model):
    """继续优化提示词"""
    try:
        # 构建继续优化的请求
        model_specific_prompt = get_model_specific_prompt(selected_model)
        
        messages = [
            {"role": "system", "content": PROMPT_ENGINEER_SYSTEM_PROMPT}
        ]
        
        # 添加对话历史
        for msg in conversation_history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        
        # 添加模型信息和用户新输入
        messages.append({"role": "user", "content": f"{model_specific_prompt}\n\n用户反馈：{user_input}"})
        
        completion = client.chat.completions.create(
            model="qwen-plus",
            messages=messages,
            stream=False
        )
        
        return completion.choices[0].message.content
    except Exception as e:
        st.error(f"继续优化失败: {e}")
        return None

def init_session_state():
    """初始化会话状态"""
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    
    if "debug_mode" not in st.session_state:
        st.session_state.debug_mode = False
    
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "GPT-4"

def reset_conversation():
    """重置对话"""
    st.session_state.conversation_history = []
    st.rerun()

def display_conversation_history():
    """显示对话历史"""
    if st.session_state.conversation_history:
        st.markdown("### 💬 对话历史")
        
        for i, message in enumerate(st.session_state.conversation_history):
            if message["role"] == "user":
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>👤 用户:</strong><br>
                    {message["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <strong>🤖 助手:</strong><br>
                    {message["content"]}
                </div>
                """, unsafe_allow_html=True)

def display_model_info(selected_model):
    """显示模型信息"""
    if selected_model in MODEL_CONFIGS:
        model_info = MODEL_CONFIGS[selected_model]
        st.markdown(f"""
        <div class="model-info">
            <strong>📋 模型信息：{selected_model}</strong><br>
            <em>{model_info['description']}</em><br>
            <strong>优势：</strong>{', '.join(model_info['strengths'])}<br>
            <strong>适用场景：</strong>{model_info['best_for']}
        </div>
        """, unsafe_allow_html=True)

def main():
    """主函数"""
    # 初始化会话状态
    init_session_state()
    
    # 加载持久化的API密钥
    if "api_key_persistent" in st.session_state:
        api_key = st.session_state.api_key_persistent
        if api_key and api_key != st.session_state.get("api_key"):
            st.session_state["api_key"] = api_key
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 style="
            font-size: 2.5rem; 
            font-weight: bold; 
            color: #333;
            margin: 0;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        ">🔧 提示词工程师</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # API密钥设置区域
    setup_api_key()
    
    # 检查API密钥是否已设置
    if not check_api_key():
        # 如果API密钥未设置，显示提示信息
        st.markdown("""
        <div style="text-align: center; padding: 2rem; background: rgba(255,255,255,0.1); border-radius: 15px; margin: 2rem 0;">
            <h3>🔑 请先设置API密钥</h3>
            <p>设置API密钥后即可使用提示词优化功能</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # 返回首页按钮和重置对话按钮
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.markdown("""
        <div style="text-align: center;">
            <a href="https://findknow-aitools.streamlit.app/" target="_blank" style="
                display: inline-block;
                background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 0.5rem 1.5rem;
                text-decoration: none;
                border-radius: 25px;
                font-weight: bold;
                width: 100%;
                text-align: center;
                box-sizing: border-box;
            ">🏠 返回首页</a>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        if st.button("🔄 重置对话", use_container_width=True, help="清空所有对话历史并重新开始"):
            reset_conversation()
    with col3:
        st.write("")  # 占位符，保持布局平衡
    
    # 显示对话历史
    display_conversation_history()
    
    # 大模型选择区域
    st.markdown("### 🤖 选择目标大模型")
    st.markdown("请选择你想要将提示词应用的大模型，我将根据模型特点为您提供针对性的优化建议。")
    
    # 模型选择器
    selected_model = st.selectbox(
        "选择大模型",
        options=list(MODEL_CONFIGS.keys()),
        index=list(MODEL_CONFIGS.keys()).index(st.session_state.selected_model),
        key="model_selector"
    )
    
    # 更新会话状态
    st.session_state.selected_model = selected_model
    
    # 显示模型信息
    display_model_info(selected_model)
    
    # 提示信息
    st.markdown("""
    <div class="card">
        <p class="placeholder-text">请描述您的需求，我将根据选择的大模型为您优化提示词，让AI更好地理解您的意图...</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 用户输入区域
    st.markdown("### ✍️ 需求描述")
    user_input = st.text_area(
        "请描述您的需求或反馈",
        height=150,
        placeholder="例如：我需要一个能够分析财务报表的AI助手，能够识别关键指标并提供投资建议...",
        key="user_input_text"
    )
    
    # 调试选项
    debug_mode = st.checkbox("🔧 调试模式", help="显示详细的调试信息", key="debug_mode")
    
    # 按钮区域
    col1, col2 = st.columns([1, 1])
    
    with col1:
        optimize_clicked = st.button("🔧 开始优化", use_container_width=True)
    
    with col2:
        continue_clicked = st.button("💬 继续对话", use_container_width=True, disabled=not st.session_state.conversation_history)
    
    # 处理开始优化
    if optimize_clicked:
        if not user_input:
            st.warning("请输入您的需求描述！")
            return
        
        # 添加用户消息到对话历史
        st.session_state.conversation_history.append({
            "role": "user",
            "content": f"目标模型：{selected_model}\n\n需求：{user_input}"
        })
        
        # 开始优化
        with st.spinner("正在根据目标模型优化提示词..."):
            client = init_client()
            if client:
                result = optimize_prompt(client, user_input, selected_model, st.session_state.conversation_history[:-1])
                
                if result:
                    # 添加助手回复到对话历史
                    st.session_state.conversation_history.append({
                        "role": "assistant",
                        "content": result
                    })
                    
                    st.success("✅ 优化完成！")
                    st.rerun()
                else:
                    st.error("提示词优化失败，请重试")
    
    # 处理继续对话
    elif continue_clicked:
        if not user_input:
            st.warning("请输入您的反馈或问题！")
            return
        
        # 添加用户消息到对话历史
        st.session_state.conversation_history.append({
            "role": "user",
            "content": user_input
        })
        
        # 继续优化
        with st.spinner("正在处理您的反馈..."):
            client = init_client()
            if client:
                result = continue_optimization(client, st.session_state.conversation_history[:-1], user_input, selected_model)
                
                if result:
                    # 添加助手回复到对话历史
                    st.session_state.conversation_history.append({
                        "role": "assistant",
                        "content": result
                    })
                    
                    st.success("✅ 已更新优化结果！")
                    st.rerun()
                else:
                    st.error("处理失败，请重试")
    


if __name__ == "__main__":
    main() 