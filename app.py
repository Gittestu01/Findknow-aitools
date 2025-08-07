import streamlit as st
from config import PAGE_CONFIG, TOOLS_CONFIG
from utils import AIClient, FileManager, StyleManager, SessionManager
import time

# 页面配置
st.set_page_config(**PAGE_CONFIG)

# 应用自定义样式
StyleManager.apply_custom_style()

# 初始化工具类
ai_client = AIClient()
file_manager = FileManager()
session_manager = SessionManager()
session_manager.init_session_state()

# 页面刷新时的自动清理
if "page_refreshed" not in st.session_state:
    st.session_state.page_refreshed = True
    file_manager.cleanup_temp_files()

# 加载持久化的API密钥
ai_client.load_persistent_api_key()

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
            if st.button("🗑️ 清除密钥", use_container_width=True):
                ai_client.clear_api_key()
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
            key="api_key_input"
        )
        
        # 设置按钮
        if st.button("✅ 设置密钥", use_container_width=True):
            if api_key:
                ai_client.save_api_key_to_persistent(api_key)
                st.success("✅ API密钥设置成功！")
                st.rerun()
            else:
                st.error("❌ 请输入有效的API密钥")
        
        # 不显眼的建议
        st.markdown("""
        <div style="
            background: rgba(128, 128, 128, 0.1);
            border-radius: 8px;
            padding: 0.5rem;
            margin-top: 0.5rem;
            font-size: 0.8rem;
            color: rgba(200, 200, 200, 0.8);
            border-left: 3px solid rgba(169, 169, 169, 0.5);
        ">
            💡 <strong>建议</strong>: 您可以将API密钥保存在浏览器密码管理器中，用户名设置为"🤖AI tools"，这样下次使用时会更方便。
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")

def process_ai_response(completion, message_placeholder):
    """处理AI响应并显示"""
    full_response = ""
    if completion:
        try:
            for chunk in completion:
                if hasattr(chunk, 'choices') and chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, 'content') and delta.content is not None:
                        full_response += delta.content
                        message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
            return full_response
        except Exception as e:
            st.error(f"处理AI响应时出错: {e}")
            return None
    return None

def handle_chat_with_files(prompt, messages):
    """处理带文件的聊天"""
    try:
        # 处理上传的文件
        file_paths = []
        file_contents = []
        
        for uploaded_file in st.session_state.uploaded_files:
            file_path = file_manager.save_uploaded_file(uploaded_file)
            if file_path:
                file_paths.append(file_path)
                # 提取文件内容
                content = file_manager.extract_text_from_file(file_path)
                if content and not content.startswith("文件读取失败") and not content.startswith("需要安装") and not content.startswith("不支持的文件类型"):
                    file_contents.append(f"文件 {uploaded_file.name}:\n{content}")
                else:
                    st.warning(f"⚠️ 无法提取文件 {uploaded_file.name} 的内容")
            else:
                st.error(f"❌ 无法保存文件 {uploaded_file.name}")
        
        if file_paths:
            # 使用文件理解API - 处理多个文件
            if len(file_paths) == 1:
                # 单个文件，使用文件理解API
                completion = ai_client.file_understanding(file_paths[0], prompt)
            else:
                # 多个文件，合并内容使用普通对话
                combined_content = "\n\n".join(file_contents)
                # 构建包含文件内容的消息
                file_message = f"以下是上传的文件内容：\n\n{combined_content}\n\n请根据这些文件内容回答用户的问题：{prompt}"
                messages.append({"role": "user", "content": file_message})
                completion = ai_client.chat_completion(messages)
            
            return completion
        else:
            st.warning("没有成功保存的文件，使用普通对话...")
            return None
            
    except Exception as e:
        st.error(f"文件分析失败: {e}")
        return None

def main_page():
    """主页面"""
    # 添加居中的页面标题
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 style="
            font-size: 3rem; 
            font-weight: bold; 
            color: #333;
            margin: 0;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        ">🤖 Findknow AI现代工具库</h1>
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
            <p>设置API密钥后即可使用所有AI功能</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # 主体部分1 - AI对话
    st.markdown("### 💬 AI智能对话")
    
    # 创建两列布局：主对话区域和功能设置区域
    col_main, col_side = st.columns([3, 1])
    
    with col_main:
        # 显示聊天历史
        messages = session_manager.get_messages()
        for i, message in enumerate(messages):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # 为每条消息添加删除按钮
                if st.button(f"🗑️ 删除", key=f"delete_msg_{i}", help="删除此条消息"):
                    session_manager.delete_message(i)
                    st.rerun()
        
        # 聊天输入区域
        with st.form(key="chat_form"):
            col_input1, col_input2 = st.columns([4, 1])
            
            with col_input1:
                input_counter = st.session_state.get('input_counter', 0)
                prompt = st.text_input("请输入您的问题...", key=f"chat_input_{input_counter}", placeholder="在这里输入您的问题...")
            
            with col_input2:
                send_button = st.form_submit_button("发送", use_container_width=True)
            
            # 处理用户输入
            if prompt and send_button:
                # 添加用户消息
                session_manager.add_message("user", prompt)
                
                # AI响应
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    
                    if ai_client.is_available():
                        # 构建消息历史
                        messages = [{"role": m["role"], "content": m["content"]} for m in session_manager.get_messages()]
                        
                        # 检查是否有上传的文件
                        has_files = ('uploaded_files' in st.session_state and 
                                   st.session_state.uploaded_files and 
                                   len(st.session_state.uploaded_files) > 0)
                        
                        completion = None
                        if has_files:
                            # 处理带文件的聊天
                            completion = handle_chat_with_files(prompt, messages)
                        
                        # 如果没有文件或文件处理失败，使用普通对话
                        if not completion:
                            if st.session_state.get("enable_search", False):
                                completion = ai_client.network_search(prompt)
                            else:
                                completion = ai_client.chat_completion(messages)
                        
                        # 处理AI响应
                        full_response = process_ai_response(completion, message_placeholder)
                        if full_response:
                            session_manager.add_message("assistant", full_response)
                        else:
                            st.error("AI响应失败，请检查API配置或网络连接")
                    else:
                        st.error("AI客户端不可用，请检查API密钥配置")
                
                # 清空输入框 - 通过增加计数器来创建新的输入框
                st.session_state.input_counter = input_counter + 1
                st.rerun()
    
    with col_side:
        # 功能设置
        st.markdown("#### 功能设置")
        enable_search = st.checkbox("🔍 启用联网搜索", key="enable_search", value=False, help="启用实时网络搜索功能")
        show_file_upload = st.checkbox("📁 显示文件上传", key="show_file_upload", value=True, help="显示文件上传区域")
        
        # 对话管理
        st.markdown("#### 对话管理")
        if st.button("🔄 重新开始", key="restart_chat", help="清除所有对话记录", use_container_width=True):
            # 清除所有状态
            session_manager.clear_chat_history()
            # 设置文件上传器重置标志
            st.session_state.reset_file_uploader = True
            st.rerun()
        
        # 文件上传区域（可选显示）
        if show_file_upload:
            st.markdown("#### 文件上传")
            
            # 使用动态key来强制重置文件上传器
            file_uploader_key = "file_uploader"
            if st.session_state.get('reset_file_uploader', False):
                # 使用不同的key来强制重新创建组件
                file_uploader_key = f"file_uploader_{int(time.time())}"
                st.session_state.reset_file_uploader = False
            
            uploaded_files = st.file_uploader(
                "拖拽文件到此处或点击上传", 
                type=['txt', 'pdf', 'docx', 'doc'], 
                key=file_uploader_key, 
                help="支持多文件上传",
                accept_multiple_files=True
            )
            
            # 更新上传的文件状态
            if uploaded_files:
                st.session_state.uploaded_files = uploaded_files
                # 只在有新文件上传时显示一次成功消息
                current_count = len(uploaded_files)
                previous_count = st.session_state.get('previous_upload_count', 0)
                if current_count != previous_count:
                    st.success(f"✅ 已上传 {current_count} 个文件")
                    st.session_state.previous_upload_count = current_count
                
                # 上传文件后清理旧的临时文件
                file_manager.cleanup_temp_files()
            elif 'uploaded_files' in st.session_state and st.session_state.uploaded_files:
                # 只在有文件时显示一次信息，不重复显示
                if not st.session_state.get('files_displayed', False):
                    st.success(f"✅ 已上传 {len(st.session_state.uploaded_files)} 个文件")
                    st.session_state.files_displayed = True
            else:
                # 没有文件时，清除相关状态
                if 'uploaded_files' in st.session_state:
                    del st.session_state.uploaded_files
                if 'previous_upload_count' in st.session_state:
                    del st.session_state.previous_upload_count
                if 'files_displayed' in st.session_state:
                    del st.session_state.files_displayed
    
    # 分隔线
    st.markdown("---")
    
    # 主体部分2 - AI工具库
    st.markdown("### 🛠️ AI智能工具库")
    
    # 工具类型选择
    tool_type = st.selectbox(
        "选择工具类型",
        ["商业端", "生活端"],
        key="tool_type"
    )
    
    if tool_type == "生活端":
        st.info("生活端功能敬请期待...")
        return
    
    # 工具卡片
    col1, col2, col3 = st.columns(3)
    
    with col1:
        tool = TOOLS_CONFIG["resume_assistant"]
        st.markdown(f"""
        <div class="card">
            <h3>{tool['icon']} {tool['name']}</h3>
            <p>{tool['description']}</p>
        </div>
        """, unsafe_allow_html=True)
        # 使用可点击的链接
        st.markdown(f"""
        <div style="text-align: center;">
            <a href="/resume_assistant" target="_self" style="
                display: inline-block;
                background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 0.5rem 1.5rem;
                text-decoration: none;
                border-radius: 25px;
                font-weight: bold;
                margin-top: 1rem;
            ">进入简历助手</a>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        tool = TOOLS_CONFIG["prompt_engineer"]
        st.markdown(f"""
        <div class="card">
            <h3>{tool['icon']} {tool['name']}</h3>
            <p>{tool['description']}</p>
        </div>
        """, unsafe_allow_html=True)
        # 使用可点击的链接
        st.markdown(f"""
        <div style="text-align: center;">
            <a href="/prompt_engineer" target="_self" style="
                display: inline-block;
                background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 0.5rem 1.5rem;
                text-decoration: none;
                border-radius: 25px;
                font-weight: bold;
                margin-top: 1rem;
            ">进入提示词工程师</a>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="card">
            <h3>🚀 更多工具</h3>
            <p>敬请期待更多AI工具...</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main_page() 