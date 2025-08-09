# -*- coding: utf-8 -*-
import streamlit as st
import os
from pathlib import Path
import numpy as np
from PIL import Image
import io
import time
import gc
import json
from openai import OpenAI

# 尝试导入OpenCV，如果失败则提供错误信息
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError as e:
    OPENCV_AVAILABLE = False
    st.error(f"❌ OpenCV导入失败: {e}")
    st.info("💡 请确保安装了opencv-python-headless包")
    st.stop()
except Exception as e:
    OPENCV_AVAILABLE = False
    st.error(f"❌ OpenCV初始化失败: {e}")
    st.info("💡 这可能是由于缺少系统依赖库导致的，请尝试使用opencv-python-headless")
    st.stop()

# 页面配置（如果还没有设置的话）
try:
    st.set_page_config(
        page_title="视频转GIF工具 - Findknow AI",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
except st.errors.StreamlitAPIException:
    # 页面配置已经设置过了，跳过
    pass

# AI配置
AI_CONFIG = {
    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "model": "qwen-plus"
}

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        position: relative;
        overflow: hidden;
    }
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, rgba(255,255,255,0.1) 0%, transparent 100%);
        pointer-events: none;
        border-radius: 15px;
    }
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0;
        text-align: center;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
        position: relative;
        z-index: 1;
    }
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
        text-align: center;
        margin: 0.5rem 0 0 0;
        text-shadow: 0 1px 2px rgba(0,0,0,0.2);
        position: relative;
        z-index: 1;
    }

    .constraint-section {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    .ai-suggestion {
        background: rgba(255,255,255,0.15);
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 4px solid #4CAF50;
    }
    .success-card {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    .stButton > button {
        border-radius: 10px;
        font-weight: bold;
        transition: all 0.3s ease;
        border: none;
        padding: 0.75rem 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
        background-color: #4CAF50;
    }
    .stButton > button:active {
        transform: translateY(0);
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2rem;
        }
        .main-header p {
            font-size: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)

def get_ai_client():
    """获取AI客户端 - 支持代理环境"""
    api_key = st.session_state.get("api_key")
    if not api_key:
        return None
    
    try:
        # 首先尝试使用代理友好的配置
        import httpx
        import ssl
        
        # 创建SSL上下文，更宽松的SSL验证
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        # 创建httpx客户端，支持代理环境
        http_client = httpx.Client(
            timeout=httpx.Timeout(
                connect=15.0,
                read=60.0,
                write=15.0,
                pool=15.0
            ),
            verify=False,  # 在代理环境下禁用SSL验证
            limits=httpx.Limits(
                max_keepalive_connections=3,
                max_connections=10
            ),
            follow_redirects=True
        )
        
        client = OpenAI(
            api_key=api_key,
            base_url=AI_CONFIG["base_url"],
            http_client=http_client
        )
        return client
        
    except ImportError:
        # 如果没有httpx，使用基本配置
        try:
            client = OpenAI(
                api_key=api_key,
                base_url=AI_CONFIG["base_url"],
            )
            return client
        except Exception as e:
            # 静默处理，不显示错误
            return None
    except Exception as e:
        # 静默处理，不显示错误
        return None

def check_api_key():
    """检查API密钥是否已设置"""
    return st.session_state.get("api_key") is not None

def safe_encode_string(text):
    """安全地处理字符串编码，避免ASCII错误"""
    if not text:
        return ""
    
    try:
        # 尝试编码为UTF-8然后解码，确保字符串是安全的
        if isinstance(text, str):
            return text.encode('utf-8').decode('utf-8')
        else:
            return str(text).encode('utf-8').decode('utf-8')
    except (UnicodeEncodeError, UnicodeDecodeError):
        # 如果出现编码问题，移除非ASCII字符
        return ''.join(char for char in str(text) if ord(char) < 128)

def test_api_connection():
    """测试API连接是否正常"""
    client = get_ai_client()
    if not client:
        return False, "连接失败"
    
    # 简单重试机制，不显示过程
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # 发送最简单的测试请求
            completion = client.chat.completions.create(
                model=AI_CONFIG["model"],
                messages=[
                    {"role": "user", "content": "hi"}
                ],
                max_tokens=1,
                temperature=0
            )
            
            # 检查响应是否有效
            if completion and completion.choices and len(completion.choices) > 0:
                return True, "连接成功"
                
        except Exception as e:
            error_msg = str(e).lower()
            
            # 如果是最后一次尝试
            if attempt == max_retries - 1:
                if "unauthorized" in error_msg or "401" in error_msg:
                    return False, "API密钥无效"
                elif "forbidden" in error_msg or "403" in error_msg:
                    return False, "API权限不足"
                else:
                    return False, "连接失败"
            else:
                # 静默重试
                import time
                time.sleep(1)
                continue
                    
    return False, "连接失败"

def init_session_state():
    """初始化会话状态"""
    if 'video_file' not in st.session_state:
        st.session_state.video_file = None
    if 'gif_data' not in st.session_state:
        st.session_state.gif_data = None
    if 'conversion_params' not in st.session_state:
        st.session_state.conversion_params = {
            'fps': 10,
            'quality': 85,
            'width': None,
            'height': None,
            'optimize': True
        }
    if 'size_constraint' not in st.session_state:
        st.session_state.size_constraint = {
            'operator': '<',
            'value': 5.0,
            'unit': 'MB',
            'enabled': False,
            'target_size': 5.0 * 1024 * 1024  # 5MB in bytes
        }
    if 'ai_suggestions' not in st.session_state:
        st.session_state.ai_suggestions = []

def reset_agent():
    """重置智能体状态 - 完全重置所有状态，包括上传的文件"""
    # 清理临时文件
    cleanup_temp_files()
    
    # 清除所有可能影响UI的会话状态
    keys_to_clear = [
        'video_file', 'gif_data', 'conversion_params', 'size_constraint', 
        'ai_suggestions', 'uploaded_file', 'ai_suggestions_cache', 
        'size_estimate_cache', 'last_params_state_key', 'cached_estimated_size',
        'cached_constraint_satisfied', 'cached_constraint'
    ]
    
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    
    # 重新初始化会话状态
    init_session_state()
    
    # 强制垃圾回收以释放内存
    gc.collect()
    
    # 强制重新运行页面，清除所有UI状态
    st.rerun()

def validate_video_file(video_path):
    """验证视频文件的完整性和可读性"""
    try:
        # 检查文件是否存在
        if not os.path.exists(video_path):
            return False, "文件不存在"
        
        # 检查文件大小
        file_size = os.path.getsize(video_path)
        if file_size == 0:
            return False, "文件为空"
        
        if file_size < 1024:  # 小于1KB，可能是损坏的文件
            return False, "文件太小，可能已损坏"
        
        # 尝试打开视频文件进行基本验证
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            cap.release()
            return False, "无法打开视频文件"
        
        # 尝试读取第一帧以验证文件完整性
        ret, frame = cap.read()
        cap.release()
        
        if not ret or frame is None:
            return False, "无法读取视频帧，文件可能已损坏"
        
        return True, "文件验证通过"
        
    except Exception as e:
        return False, f"文件验证失败: {str(e)}"

def analyze_video_properties(video_path):
    """分析视频属性 - 增强版本，具有更强的错误处理"""
    try:
        # 首先验证视频文件
        is_valid, message = validate_video_file(video_path)
        if not is_valid:
            st.error(f"❌ 视频文件验证失败: {message}")
            return None
        
        # 使用更安全的方式打开视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("❌ 无法打开视频文件，可能是格式不支持或文件已损坏")
            return None
        
        try:
            # 获取基础视频属性，添加默认值防护
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # 验证获取的属性是否合理
            if fps <= 0 or fps > 120:  # FPS不合理
                fps = 25.0  # 使用默认FPS
                st.warning("⚠️ 检测到异常帧率，使用默认值25FPS")
            
            if frame_count <= 0:  # 帧数不合理
                frame_count = 1
                st.warning("⚠️ 无法获取准确帧数，使用估算值")
            
            if width <= 0 or height <= 0 or width > 4000 or height > 4000:  # 分辨率不合理
                st.error("❌ 检测到异常的视频分辨率，无法处理此视频")
                cap.release()
                return None
            
            # 计算时长
            duration = frame_count / fps if fps > 0 else 1.0
            
            # 获取文件大小
            file_size = os.path.getsize(video_path)
            
            # 尝试读取一帧以进一步验证文件完整性
            test_frame_read = False
            try:
                ret, frame = cap.read()
                if ret and frame is not None:
                    test_frame_read = True
                # 回到开始位置
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            except Exception as e:
                st.warning(f"⚠️ 读取测试帧时出现问题: {str(e)}")
            
            cap.release()
            
            if not test_frame_read:
                st.error("❌ 无法读取视频帧，文件可能已损坏或格式不兼容")
                return None
            
            video_props = {
                'fps': fps,
                'frame_count': frame_count,
                'width': width,
                'height': height,
                'duration': duration,
                'file_size': file_size
            }
            
            # 显示成功信息
            st.success("✅ 视频分析完成")
            
            return video_props
            
        except Exception as e:
            cap.release()
            st.error(f"❌ 分析视频属性时出错: {str(e)}")
            return None
            
    except Exception as e:
        st.error(f"❌ 视频分析失败: {str(e)}")
        st.info("💡 这可能是由于视频文件损坏、格式不支持或编码问题导致的")
        return None

def generate_ai_suggestions(video_props, user_input=""):
    """生成AI建议 - 使用真实AI大模型分析用户意图和视频特征"""
    
    # 检查API密钥
    if not check_api_key():
        st.warning("⚠️ 需要设置API密钥才能使用AI智能建议功能")
        st.info("💡 请返回主页设置API密钥，或使用下方的手动参数调整")
        return []
    
    if not video_props:
        return []
    
    # 生成缓存键 - 基于视频属性和用户输入，避免编码问题
    import hashlib
    user_input_safe = safe_encode_string(user_input.strip() if user_input else "")
    user_hash = hashlib.md5(user_input_safe.encode('utf-8')).hexdigest()[:8]
    cache_key = f"{video_props['width']}x{video_props['height']}_{video_props['fps']:.1f}fps_{video_props['duration']:.1f}s_{user_hash}"
    
    # 检查会话状态中的缓存
    if 'ai_suggestions_cache' not in st.session_state:
        st.session_state.ai_suggestions_cache = {}
    
    # 如果缓存中存在相同的建议，直接返回
    if cache_key in st.session_state.ai_suggestions_cache:
        cached_suggestions = st.session_state.ai_suggestions_cache[cache_key]
        st.success(f"✅ 已加载缓存的AI建议（{len(cached_suggestions)} 个方案）")
        return cached_suggestions
    
    # 获取AI客户端
    client = get_ai_client()
    if not client:
        st.error("❌ AI客户端初始化失败")
        return []
    
    try:
        # 构建优化的系统提示词（缩短长度以加快响应）
        system_prompt = """你是视频转GIF优化专家。

任务：分析视频参数和用户需求，提供2-3个GIF转换方案。

核心原则：
1. 平衡文件大小、质量和流畅度
2. 高分辨率视频需适当缩放
3. 长视频使用保守参数
4. 文件大小约束必须可实现

返回JSON数组格式：
[{
  "name": "方案名称",
  "description": "详细说明和适用场景（含预估大小）",
  "params": {
    "fps": 1-30,
    "quality": 50-100,
    "width": 目标宽度,
    "height": 目标高度,
    "optimize": true/false
  },
  "size_constraint": {
    "operator": "<",
    "value": 数值,
    "unit": "MB",
    "enabled": true
  }
}]

要求：
- 必须返回有效JSON
- 方案要有明显区别
- 约束要现实可行"""

        # 构建用户查询（简化以加快响应），确保用户输入安全
        user_input_safe = safe_encode_string(user_input.strip() if user_input else "")
        if not user_input_safe:
            user_input_safe = "通用优化建议"
            
        user_prompt = f"""视频参数：
分辨率：{video_props['width']}×{video_props['height']}
帧率：{video_props['fps']:.1f}FPS，时长：{video_props['duration']:.1f}秒
文件大小：{video_props['file_size'] / (1024*1024):.2f}MB

用户需求：{user_input_safe}

请提供2-3个GIF转换方案（JSON格式）。"""

        # 调用AI进行分析 - 静默重试机制
        with st.spinner("🤖 AI正在分析视频参数和用户需求..."):
            max_retries = 3
            ai_response = None
            
            for attempt in range(max_retries):
                try:
                    # 确保字符串编码正确
                    system_prompt_safe = safe_encode_string(system_prompt)
                    user_prompt_safe = safe_encode_string(user_prompt)
                    
                    completion = client.chat.completions.create(
                        model=AI_CONFIG["model"],
                        messages=[
                            {"role": "system", "content": system_prompt_safe},
                            {"role": "user", "content": user_prompt_safe}
                        ],
                        temperature=0.7,
                        max_tokens=1500
                    )
                    
                    ai_response = completion.choices[0].message.content
                    break  # 成功则跳出重试循环
                    
                except Exception as e:
                    error_msg = str(e).lower()
                    
                    # 如果是最后一次尝试
                    if attempt == max_retries - 1:
                        # 静默失败，使用默认建议
                        fallback_suggestions = get_fallback_suggestions(video_props, user_input)
                        st.session_state.ai_suggestions_cache[cache_key] = fallback_suggestions
                        return fallback_suggestions
                    else:
                        # 静默重试
                        import time
                        time.sleep(1)
                        continue
            
            # 如果ai_response为空，使用默认建议
            if not ai_response:
                fallback_suggestions = get_fallback_suggestions(video_props, user_input)
                st.session_state.ai_suggestions_cache[cache_key] = fallback_suggestions
                return fallback_suggestions
            
            # 解析AI响应
            try:
                # 提取JSON部分
                json_start = ai_response.find('[')
                json_end = ai_response.rfind(']') + 1
                
                if json_start != -1 and json_end != -1:
                    json_str = ai_response[json_start:json_end]
                    suggestions = json.loads(json_str)
                    
                    # 验证并处理建议
                    validated_suggestions = []
                    for suggestion in suggestions:
                        if validate_suggestion(suggestion, video_props):
                            # 进一步验证参数是否能满足大小约束
                            if 'size_constraint' in suggestion and suggestion['size_constraint'].get('enabled', False):
                                # 确保size_constraint包含target_size字段
                                constraint = suggestion['size_constraint']
                                if 'target_size' not in constraint:
                                    # 计算target_size
                                    unit_multipliers = {
                                        'B': 1,
                                        'KB': 1024,
                                        'MB': 1024 * 1024,
                                        'GB': 1024 * 1024 * 1024
                                    }
                                    multiplier = unit_multipliers.get(constraint.get('unit', 'MB').upper(), 1024 * 1024)
                                    constraint['target_size'] = constraint.get('value', 5.0) * multiplier
                                
                                adjusted_params = adjust_params_for_constraint(
                                    video_props, 
                                    suggestion['params'], 
                                    constraint
                                )
                                
                                # 更新建议的参数为调整后的参数
                                suggestion['params'] = adjusted_params
                                
                                # 验证调整后的参数
                                satisfied, estimated_size = validate_params_against_constraint(
                                    video_props, 
                                    adjusted_params, 
                                    constraint
                                )
                                
                                # 如果仍然不满足，调整约束
                                if not satisfied and estimated_size:
                                    new_target_size = estimated_size * 1.3  # 留30%余量
                                    new_target_mb = new_target_size / (1024 * 1024)
                                    constraint['value'] = round(new_target_mb, 1)
                                    constraint['target_size'] = new_target_size
                                    
                                    # 更新描述
                                    if "（预估" not in suggestion['description']:
                                        suggestion['description'] += f"（预估约{new_target_mb:.1f}MB）"
                                
                                # 添加预估大小信息
                                suggestion['estimated_size'] = estimate_gif_size(video_props, adjusted_params)
                            
                            # 清理和标准化建议数据
                            sanitized_suggestion = sanitize_suggestion(suggestion)
                            validated_suggestions.append(sanitized_suggestion)
                    
                    if validated_suggestions:
                        # 缓存有效的建议
                        st.session_state.ai_suggestions_cache[cache_key] = validated_suggestions
                        st.success(f"✅ AI成功生成了 {len(validated_suggestions)} 个专业建议")
                        return validated_suggestions
                    else:
                        st.warning("⚠️ AI生成的建议格式有误，使用默认建议")
                        fallback_suggestions = get_fallback_suggestions(video_props, user_input)
                        # 也缓存备选建议
                        st.session_state.ai_suggestions_cache[cache_key] = fallback_suggestions
                        return fallback_suggestions
                
                else:
                    st.warning("⚠️ AI响应格式异常，使用默认建议")
                    fallback_suggestions = get_fallback_suggestions(video_props, user_input)
                    st.session_state.ai_suggestions_cache[cache_key] = fallback_suggestions
                    return fallback_suggestions
                    
            except json.JSONDecodeError as e:
                st.warning(f"⚠️ AI响应解析失败: {str(e)}")
                fallback_suggestions = get_fallback_suggestions(video_props, user_input)
                st.session_state.ai_suggestions_cache[cache_key] = fallback_suggestions
                return fallback_suggestions
                
    except Exception as e:
        # 安全处理异常信息，避免编码问题
        error_msg = safe_encode_string(str(e))
        if not error_msg:
            error_msg = "未知错误"
        
        st.error(f"❌ AI分析失败: {error_msg}")
        st.info("💡 将使用默认建议作为备选方案")
        fallback_suggestions = get_fallback_suggestions(video_props, user_input)
        st.session_state.ai_suggestions_cache[cache_key] = fallback_suggestions
        return fallback_suggestions

def validate_suggestion(suggestion, video_props):
    """验证AI建议的有效性"""
    try:
        # 检查必需字段
        required_fields = ['name', 'description', 'params', 'size_constraint']
        if not all(field in suggestion for field in required_fields):
            return False
        
        # 验证名称和描述
        if not suggestion['name'] or not suggestion['description']:
            return False
        
        params = suggestion['params']
        if not isinstance(params, dict):
            return False
            
        required_params = ['fps', 'quality', 'width', 'height', 'optimize']
        if not all(param in params for param in required_params):
            return False
        
        # 验证参数类型和范围
        try:
            fps = int(params['fps'])
            quality = int(params['quality'])
            width = int(params['width'])
            height = int(params['height'])
            optimize = bool(params['optimize'])
        except (ValueError, TypeError):
            return False
        
        if not (1 <= fps <= 30):
            return False
        if not (50 <= quality <= 100):
            return False
        if not (10 <= width <= video_props['width'] * 2):
            return False
        if not (10 <= height <= video_props['height'] * 2):
            return False
        
        # 验证尺寸约束
        constraint = suggestion['size_constraint']
        if not isinstance(constraint, dict):
            return False
            
        required_constraint_fields = ['operator', 'value', 'unit', 'enabled']
        if not all(field in constraint for field in required_constraint_fields):
            return False
        
        if constraint['operator'] not in ['<', '>', '=', '<=', '>=']:
            return False
        if constraint['unit'] not in ['B', 'KB', 'MB', 'GB']:
            return False
        
        try:
            value = float(constraint['value'])
            if value <= 0:
                return False
        except (ValueError, TypeError):
            return False
        
        # 验证enabled字段
        if not isinstance(constraint['enabled'], bool):
            return False
        
        return True
        
    except Exception:
        return False

def sanitize_suggestion(suggestion):
    """清理和标准化AI建议数据"""
    try:
        sanitized = {
            'name': str(suggestion.get('name', '默认方案')),
            'description': str(suggestion.get('description', '暂无描述')),
            'params': {},
            'size_constraint': {}
        }
        
        # 清理参数
        params = suggestion.get('params', {})
        sanitized['params'] = {
            'fps': int(params.get('fps', 10)),
            'quality': int(params.get('quality', 85)),
            'width': int(params.get('width', 640)),
            'height': int(params.get('height', 480)),
            'optimize': bool(params.get('optimize', True))
        }
        
        # 清理约束
        constraint = suggestion.get('size_constraint', {})
        sanitized['size_constraint'] = {
            'operator': str(constraint.get('operator', '<')),
            'value': float(constraint.get('value', 5.0)),
            'unit': str(constraint.get('unit', 'MB')),
            'enabled': bool(constraint.get('enabled', True))
        }
        
        # 添加target_size
        unit_multipliers = {
            'B': 1,
            'KB': 1024,
            'MB': 1024 * 1024,
            'GB': 1024 * 1024 * 1024
        }
        multiplier = unit_multipliers.get(sanitized['size_constraint']['unit'].upper(), 1024 * 1024)
        sanitized['size_constraint']['target_size'] = sanitized['size_constraint']['value'] * multiplier
        
        # 保留预估大小（如果存在）
        if 'estimated_size' in suggestion:
            try:
                sanitized['estimated_size'] = float(suggestion['estimated_size'])
            except (ValueError, TypeError):
                pass
        
        return sanitized
        
    except Exception:
        # 如果清理失败，返回一个安全的默认建议
        return {
            'name': '安全默认方案',
            'description': '安全的默认转换参数',
            'params': {
                'fps': 10,
                'quality': 85,
                'width': 640,
                'height': 480,
                'optimize': True
            },
            'size_constraint': {
                'operator': '<',
                'value': 5.0,
                'unit': 'MB',
                'enabled': True,
                'target_size': 5.0 * 1024 * 1024
            }
        }

def get_fallback_suggestions(video_props, user_input=""):
    """获取备选建议（当AI失败时使用）- 基于实际预估的智能建议"""
    fps = video_props['fps']
    width = video_props['width']
    height = video_props['height']
    duration = video_props['duration']
    file_size = video_props['file_size']
    
    suggestions = []
    
    # 基础参数模板
    base_templates = [
        {
            'name': '高质量专业版',
            'description': '保持最高视觉质量，适合专业展示和演示用途',
            'params': {
                'fps': min(int(fps), 15),
                'quality': 95,
                'width': width,
                'height': height,
                'optimize': True
            },
            'target_size_mb': 10.0
        },
        {
            'name': '平衡优化版',
            'description': '质量与文件大小的最佳平衡，适合一般分享使用',
            'params': {
                'fps': min(int(fps), 12),
                'quality': 85,
                'width': min(width, 640),
                'height': min(height, 480),
                'optimize': True
            },
            'target_size_mb': 5.0
        },
        {
            'name': '小文件分享版',
            'description': '大幅压缩文件大小，适合网络传输和存储空间有限的场景',
            'params': {
                'fps': 8,
                'quality': 75,
                'width': min(width, 480),
                'height': min(height, 360),
                'optimize': True
            },
            'target_size_mb': 2.0
        }
    ]
    
    # 处理每个模板，确保参数可行
    for template in base_templates:
        # 跳过不合适的建议（例如短视频不需要小文件版本）
        if template['name'] == '小文件分享版' and duration <= 10 and file_size <= 20 * 1024 * 1024:
            continue
        
        target_size_bytes = template['target_size_mb'] * 1024 * 1024
        size_constraint = {
            'operator': '<',
            'value': template['target_size_mb'],
            'unit': 'MB',
            'enabled': True,
            'target_size': target_size_bytes
        }
        
        # 调整参数以满足大小约束
        adjusted_params = adjust_params_for_constraint(video_props, template['params'], size_constraint)
        
        # 验证调整后的参数
        satisfied, estimated_size = validate_params_against_constraint(video_props, adjusted_params, size_constraint)
        
        # 如果仍然不满足约束，进一步调整目标大小
        if not satisfied and estimated_size:
            # 将目标大小调整为预估大小的1.2倍（留一些余量）
            new_target_size = estimated_size * 1.2
            new_target_mb = new_target_size / (1024 * 1024)
            
            # 更新约束
            size_constraint['value'] = round(new_target_mb, 1)
            size_constraint['target_size'] = new_target_size
            
            # 更新描述，说明实际可达到的大小
            if template['name'] == '高质量专业版':
                template['description'] += f"（预估约{new_target_mb:.1f}MB）"
            elif template['name'] == '平衡优化版':
                template['description'] += f"（预估约{new_target_mb:.1f}MB）"
            elif template['name'] == '小文件分享版':
                template['description'] += f"（预估约{new_target_mb:.1f}MB）"
        
        suggestion = {
            'name': template['name'],
            'description': template['description'],
            'params': adjusted_params,
            'size_constraint': size_constraint,
            'estimated_size': estimate_gif_size(video_props, adjusted_params)
        }
        
        # 清理和标准化建议数据
        sanitized_suggestion = sanitize_suggestion(suggestion)
        suggestions.append(sanitized_suggestion)
    
    return suggestions

def get_real_gif_size_preview(video_path, params):
    """通过真实转换获得准确的GIF文件大小预估 - 高性能优化版本，增强错误处理"""
    cap = None
    try:
        # 首先验证视频文件
        is_valid, message = validate_video_file(video_path)
        if not is_valid:
            return None
        
        # 安全地打开视频文件
        try:
            cap = cv2.VideoCapture(video_path)
        except Exception as e:
            return None
            
        if not cap.isOpened():
            if cap:
                cap.release()
            return None
        
        # 预分配变量，增加安全检查
        fps = max(1, min(30, params.get('fps', 10)))  # 限制FPS范围
        target_width = max(10, min(2000, params.get('width', 640)))  # 限制宽度范围
        target_height = max(10, min(2000, params.get('height', 480)))  # 限制高度范围
        quality = max(50, min(100, params.get('quality', 85)))  # 限制质量范围
        
        # 安全地获取视频属性
        try:
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        except Exception as e:
            cap.release()
            return None
        
        # 验证获取的属性
        if original_fps <= 0 or original_fps > 120:
            original_fps = 25.0  # 使用默认值
        
        if total_frames <= 0:
            total_frames = 100  # 使用默认值
        
        # 计算采样间隔，添加边界检查
        sample_interval = max(1, int(original_fps / fps)) if original_fps > 0 else 1
        
        # 限制预估时的帧数以提高速度和稳定性
        max_preview_frames = min(30, total_frames // sample_interval)  # 大幅减少预估帧数
        if max_preview_frames <= 0:
            max_preview_frames = 5  # 最少处理5帧
        
        # 预分配帧数组
        frames = []
        frame_count = 0
        processed_frames = 0
        max_read_attempts = max_preview_frames * 3  # 防止无限循环
        read_attempts = 0
        
        # 预设置resize插值方法
        resize_interpolation = cv2.INTER_LINEAR
        
        # 安全的帧读取循环
        while processed_frames < max_preview_frames and read_attempts < max_read_attempts:
            try:
                ret, frame = cap.read()
                read_attempts += 1
                
                if not ret or frame is None:
                    break
                
                # 按间隔采样
                if frame_count % sample_interval == 0:
                    try:
                        # 验证帧的有效性
                        if frame.shape[0] <= 0 or frame.shape[1] <= 0:
                            frame_count += 1
                            continue
                        
                        # 安全地调整尺寸
                        if target_width and target_height:
                            try:
                                frame = cv2.resize(frame, (target_width, target_height), interpolation=resize_interpolation)
                            except Exception as resize_e:
                                # 如果resize失败，跳过这一帧
                                frame_count += 1
                                continue
                        
                        # 安全地进行颜色转换
                        try:
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        except Exception as color_e:
                            # 如果颜色转换失败，跳过这一帧
                            frame_count += 1
                            continue
                        
                        # 转换为PIL图像
                        try:
                            pil_image = Image.fromarray(frame_rgb)
                            frames.append(pil_image)
                            processed_frames += 1
                        except Exception as pil_e:
                            # 如果PIL转换失败，跳过这一帧
                            frame_count += 1
                            continue
                            
                    except Exception as frame_e:
                        # 处理单帧时的任何异常，继续处理下一帧
                        pass
                
                frame_count += 1
                
            except Exception as read_e:
                # 读取帧时的异常，尝试继续
                read_attempts += 1
                if read_attempts >= max_read_attempts:
                    break
                continue
        
        # 安全释放资源
        if cap:
            cap.release()
        
        # 检查是否成功处理了足够的帧
        if not frames or len(frames) < 2:
            return None
        
        # 安全地创建GIF
        try:
            gif_buffer = io.BytesIO()
            gif_duration = max(50, int(1000 / fps))  # 确保duration不会太小
            
            # 使用更保守的参数以避免错误
            frames[0].save(
                gif_buffer,
                format='GIF',
                save_all=True,
                append_images=frames[1:],
                duration=gif_duration,
                loop=0,
                optimize=bool(params.get('optimize', True)),
                quality=quality
            )
            
            gif_buffer.seek(0)
            gif_data = gif_buffer.getvalue()
            
            # 验证生成的GIF数据
            if len(gif_data) > 0:
                return len(gif_data)
            else:
                return None
            
        except Exception as gif_e:
            return None
        
    except Exception as e:
        # 捕获所有未预期的异常
        if cap:
            cap.release()
        return None

def get_fallback_estimate_size(video_props, params):
    """获取备用的文件大小估算"""
    try:
        # 基于视频属性和参数进行数学估算
        width = params.get('width', 640)
        height = params.get('height', 480)
        fps = params.get('fps', 10)
        quality = params.get('quality', 85)
        duration = video_props.get('duration', 10)
        
        # 基础估算公式（经验值）
        # 考虑分辨率、帧率、质量和时长
        pixels_per_frame = width * height
        total_frames = fps * duration
        
        # 每像素每帧的估算字节数（根据质量调整）
        bytes_per_pixel_frame = (quality / 100) * 0.8  # 基础值
        
        # 计算基础大小
        base_size = pixels_per_frame * total_frames * bytes_per_pixel_frame
        
        # 应用压缩因子（GIF压缩效率）
        compression_factor = 0.3  # GIF的平均压缩率
        estimated_size = base_size * compression_factor
        
        # 添加一些变化范围，确保估算在合理范围内
        min_size = 10 * 1024  # 最小10KB
        max_size = 50 * 1024 * 1024  # 最大50MB
        
        estimated_size = max(min_size, min(max_size, estimated_size))
        
        return int(estimated_size)
        
    except Exception as e:
        # 如果备用估算也失败，返回保守估计
        return video_props.get('file_size', 5 * 1024 * 1024) // 4

def estimate_gif_size(video_props, params, video_path=None):
    """预估GIF文件大小 - 使用真实转换获得准确预估，带备用机制"""
    
    # 验证输入参数
    if not video_props or not params:
        return 1024 * 1024  # 返回1MB作为默认值
    
    # 生成参数缓存键
    try:
        params_key = f"{params.get('width', 0)}x{params.get('height', 0)}_{params.get('fps', 10)}fps_{params.get('quality', 85)}q"
    except Exception:
        params_key = "default_params"
    
    # 初始化预估缓存
    if 'size_estimate_cache' not in st.session_state:
        st.session_state.size_estimate_cache = {}
    
    # 检查缓存
    if params_key in st.session_state.size_estimate_cache:
        cached_size = st.session_state.size_estimate_cache[params_key]
        if cached_size and cached_size > 0:
            return cached_size
    
    estimated_size = None
    
    # 首先尝试真实转换预估（只有在视频文件有效时）
    if video_path and os.path.exists(video_path):
        try:
            # 验证视频文件状态
            is_valid, _ = validate_video_file(video_path)
            if is_valid:
                estimated_size = get_real_gif_size_preview(video_path, params)
        except Exception as e:
            estimated_size = None
    
    # 如果真实转换失败，重试或使用保守估计
    if estimated_size is None or estimated_size <= 0:
        # 再次尝试真实转换预估，使用更宽松的参数
        if video_path and os.path.exists(video_path):
            try:
                # 使用简化参数重试
                simplified_params = params.copy()
                simplified_params['width'] = min(simplified_params.get('width', 320), 320)
                simplified_params['height'] = min(simplified_params.get('height', 240), 240)
                simplified_params['fps'] = min(simplified_params.get('fps', 5), 5)
                
                estimated_size = get_real_gif_size_preview(video_path, simplified_params)
                if estimated_size and estimated_size > 0:
                    # 根据原始参数调整估算大小
                    scale_factor = (params.get('width', 320) * params.get('height', 240)) / (simplified_params['width'] * simplified_params['height'])
                    fps_factor = params.get('fps', 10) / simplified_params['fps']
                    quality_factor = params.get('quality', 85) / 85
                    
                    estimated_size = int(estimated_size * scale_factor * fps_factor * quality_factor)
            except Exception:
                estimated_size = None
        
        # 如果仍然失败，使用基于视频文件大小的保守估计
        if estimated_size is None or estimated_size <= 0:
            file_size = video_props.get('file_size', 5 * 1024 * 1024)
            # 基于视频文件大小的简单估算：通常GIF是视频大小的1/3到1/2
            estimated_size = max(10 * 1024, file_size // 3)  # 最少10KB
    
    # 验证估算结果的合理性
    if estimated_size is None or estimated_size <= 0:
        estimated_size = 1024 * 1024  # 默认1MB
    
    # 缓存结果
    try:
        st.session_state.size_estimate_cache[params_key] = estimated_size
    except Exception:
        pass  # 缓存失败不影响功能
    
    return estimated_size

def validate_params_against_constraint(video_props, params, size_constraint, video_path=None):
    """验证参数是否能满足大小约束"""
    if not size_constraint or not size_constraint.get('enabled'):
        return True, None
    
    estimated_size = estimate_gif_size(video_props, params, video_path)
    target_size = size_constraint['target_size']
    operator = size_constraint['operator']
    
    # 检查是否满足约束
    satisfied = False
    if operator == '<':
        satisfied = estimated_size < target_size
    elif operator == '<=':
        satisfied = estimated_size <= target_size
    elif operator == '>':
        satisfied = estimated_size > target_size
    elif operator == '>=':
        satisfied = estimated_size >= target_size
    elif operator == '=':
        # 允许10%的误差
        satisfied = abs(estimated_size - target_size) / target_size <= 0.1
    
    return satisfied, estimated_size

def adjust_params_for_constraint(video_props, base_params, size_constraint, video_path=None):
    """根据大小约束调整参数"""
    if not size_constraint or not size_constraint.get('enabled'):
        return base_params
    
    target_size = size_constraint['target_size']
    operator = size_constraint['operator']
    
    # 只对小于类型的约束进行自动调整
    if operator not in ['<', '<=']:
        return base_params
    
    # 创建参数副本
    adjusted_params = base_params.copy()
    
    # 预估当前参数的文件大小
    current_estimated = estimate_gif_size(video_props, adjusted_params, video_path)
    
    if current_estimated <= target_size:
        return adjusted_params
    
    # 需要压缩，计算压缩比例
    compression_ratio = target_size / current_estimated
    
    # 保守的调整策略
    if compression_ratio >= 0.8:
        # 轻微调整
        adjusted_params['quality'] = max(75, int(adjusted_params['quality'] * 0.95))
        adjusted_params['fps'] = max(8, int(adjusted_params['fps'] * 0.9))
    elif compression_ratio >= 0.6:
        # 中等调整
        scale_factor = compression_ratio ** 0.3
        adjusted_params['width'] = max(160, int(adjusted_params['width'] * scale_factor))
        adjusted_params['height'] = max(120, int(adjusted_params['height'] * scale_factor))
        adjusted_params['quality'] = max(70, int(adjusted_params['quality'] * 0.9))
        adjusted_params['fps'] = max(6, int(adjusted_params['fps'] * 0.8))
    elif compression_ratio >= 0.4:
        # 较大调整
        scale_factor = compression_ratio ** 0.4
        adjusted_params['width'] = max(120, int(adjusted_params['width'] * scale_factor))
        adjusted_params['height'] = max(90, int(adjusted_params['height'] * scale_factor))
        adjusted_params['quality'] = max(65, int(adjusted_params['quality'] * 0.85))
        adjusted_params['fps'] = max(5, int(adjusted_params['fps'] * 0.7))
    else:
        # 激进调整
        scale_factor = compression_ratio ** 0.5
        adjusted_params['width'] = max(80, int(adjusted_params['width'] * scale_factor))
        adjusted_params['height'] = max(60, int(adjusted_params['height'] * scale_factor))
        adjusted_params['quality'] = max(60, int(adjusted_params['quality'] * 0.8))
        adjusted_params['fps'] = max(4, int(adjusted_params['fps'] * 0.6))
    
    return adjusted_params

def parse_size_constraint(operator, value, unit):
    """解析文件大小约束"""
    if not operator or not value or not unit:
        return None
    
    try:
        value = float(value)
        if value <= 0:
            return None
    except (ValueError, TypeError):
        return None
    
    # 转换为字节
    unit_multipliers = {
        'B': 1,
        'KB': 1024,
        'MB': 1024 * 1024,
        'GB': 1024 * 1024 * 1024
    }
    
    multiplier = unit_multipliers.get(unit.upper(), 1)
    target_size = value * multiplier
    
    return {
        'target_size': target_size,
        'operator': operator,
        'unit': unit.upper(),
        'display': f"{value}{unit} {operator}"
    }

def convert_video_to_gif(video_path, params, size_constraint=None):
    """将视频转换为GIF - 高性能优化版本，增强错误处理"""
    cap = None
    try:
        # 首先验证视频文件
        is_valid, message = validate_video_file(video_path)
        if not is_valid:
            st.error(f"❌ 视频文件验证失败: {message}")
            return None
        
        # 安全地打开视频文件
        try:
            cap = cv2.VideoCapture(video_path)
        except Exception as e:
            st.error(f"❌ 无法打开视频文件: {str(e)}")
            return None
            
        if not cap.isOpened():
            st.error("❌ 无法打开视频文件，可能是格式不支持或文件已损坏")
            return None
        
        # 预先分配变量，增加安全检查
        fps = max(1, min(30, params.get('fps', 10)))
        target_width = max(10, min(2000, params.get('width', 640)))
        target_height = max(10, min(2000, params.get('height', 480)))
        quality = max(50, min(100, params.get('quality', 85)))
        
        # 安全地获取视频属性
        try:
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        except Exception as e:
            cap.release()
            st.error(f"❌ 获取视频属性失败: {str(e)}")
            return None
        
        # 验证获取的属性
        if original_fps <= 0 or original_fps > 120:
            original_fps = 25.0
            st.warning("⚠️ 检测到异常帧率，使用默认值25FPS")
        
        if total_frames <= 0:
            total_frames = 100
            st.warning("⚠️ 无法获取准确帧数，使用估算值")
        
        # 计算采样间隔
        sample_interval = max(1, int(original_fps / fps)) if original_fps > 0 else 1
        
        # 限制最大帧数以提高速度和稳定性
        max_frames = min(150, total_frames // sample_interval)
        if max_frames <= 0:
            max_frames = 10  # 最少处理10帧
        
        # 预分配帧数组
        frames = []
        frame_count = 0
        processed_frames = 0
        max_read_attempts = max_frames * 2  # 防止无限循环
        read_attempts = 0
        
        # 创建进度条
        progress_bar = st.progress(0)
        status_text = st.empty()
        update_interval = max(1, max_frames // 20)
        
        # 预设置resize插值方法
        resize_interpolation = cv2.INTER_LINEAR
        
        # 安全的帧处理循环
        while processed_frames < max_frames and read_attempts < max_read_attempts:
            try:
                ret, frame = cap.read()
                read_attempts += 1
                
                if not ret or frame is None:
                    break
                
                # 按间隔采样
                if frame_count % sample_interval == 0:
                    try:
                        # 验证帧的有效性
                        if frame.shape[0] <= 0 or frame.shape[1] <= 0:
                            frame_count += 1
                            continue
                        
                        # 安全地调整尺寸
                        if target_width and target_height:
                            try:
                                frame = cv2.resize(frame, (target_width, target_height), interpolation=resize_interpolation)
                            except Exception as resize_e:
                                frame_count += 1
                                continue
                        
                        # 安全地进行颜色转换
                        try:
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        except Exception as color_e:
                            frame_count += 1
                            continue
                        
                        # 转换为PIL图像
                        try:
                            pil_image = Image.fromarray(frame_rgb)
                            frames.append(pil_image)
                            processed_frames += 1
                        except Exception as pil_e:
                            frame_count += 1
                            continue
                        
                        # 更新进度
                        if processed_frames % update_interval == 0 or processed_frames == max_frames:
                            try:
                                progress = processed_frames / max_frames
                                progress_bar.progress(progress)
                                status_text.text(f"正在处理视频帧... {processed_frames}/{max_frames}")
                            except Exception:
                                pass  # 进度更新失败不影响转换
                        
                        # 内存管理
                        if processed_frames % 50 == 0:
                            gc.collect()
                            
                    except Exception as frame_e:
                        # 处理单帧的异常，继续下一帧
                        pass
                
                frame_count += 1
                
            except Exception as read_e:
                # 读取帧的异常，尝试继续
                read_attempts += 1
                if read_attempts >= max_read_attempts:
                    break
                continue
        
        # 安全释放资源
        if cap:
            cap.release()
        
        # 检查是否成功处理了足够的帧
        if not frames or len(frames) < 2:
            st.error("❌ 没有提取到足够的有效帧，无法生成GIF")
            st.info("💡 这可能是由于视频文件损坏或格式不兼容导致的")
            return None
        
        # 安全地创建GIF
        try:
            status_text.text("正在生成GIF文件...")
            gif_buffer = io.BytesIO()
            
            # 预设置GIF参数，添加安全检查
            gif_duration = max(50, int(1000 / fps))  # 确保duration不会太小
            
            # 验证frames是否有效
            if not frames or len(frames) == 0:
                st.error("❌ 没有有效的帧数据")
                return None
            
            # 安全地保存GIF
            frames[0].save(
                gif_buffer,
                format='GIF',
                save_all=True,
                append_images=frames[1:] if len(frames) > 1 else [],
                duration=gif_duration,
                loop=0,
                optimize=bool(params.get('optimize', True)),
                quality=quality
            )
            
            gif_buffer.seek(0)
            gif_data = gif_buffer.getvalue()
            
            # 验证生成的GIF数据
            if not gif_data or len(gif_data) == 0:
                st.error("❌ 生成的GIF文件为空")
                return None
                
        except Exception as gif_e:
            st.error(f"❌ 生成GIF文件时出错: {str(gif_e)}")
            st.info("💡 请尝试调整参数（降低质量或分辨率）后重试")
            return None
        
        # 检查文件大小约束
        if size_constraint and size_constraint['enabled']:
            gif_size = len(gif_data)
            target_size = size_constraint['target_size']
            operator = size_constraint['operator']
            
            # 显示当前GIF文件大小信息
            gif_size_mb = gif_size / (1024 * 1024)
            gif_size_kb = gif_size / 1024
            target_size_mb = target_size / (1024 * 1024)
            target_size_kb = target_size / 1024
            
            if gif_size_mb >= 1:
                gif_size_display = f"{gif_size_mb:.2f}MB"
            else:
                gif_size_display = f"{gif_size_kb:.1f}KB"
                
            if target_size_mb >= 1:
                target_size_display = f"{target_size_mb:.2f}MB"
            else:
                target_size_display = f"{target_size_kb:.1f}KB"
            
            # 强制优化逻辑 - 当设置为小于某数值时，强制调整到目标大小以下
            if operator in ['<', '<=']:
                if gif_size > target_size:
                    status_text.text(f"正在智能优化GIF文件大小到 {target_size_display} 以下...")
                    optimized_data = optimize_gif_size(gif_data, target_size)
                    if optimized_data:
                        optimized_size = len(optimized_data)
                        optimized_size_mb = optimized_size / (1024 * 1024)
                        optimized_size_kb = optimized_size / 1024
                        
                        if optimized_size_mb >= 1:
                            optimized_display = f"{optimized_size_mb:.2f}MB"
                        else:
                            optimized_display = f"{optimized_size_kb:.1f}KB"
                        
                        if optimized_size <= target_size:
                            st.success(f"✅ 智能优化成功！文件大小从 {gif_size_display} 优化到 {optimized_display}")
                        else:
                            st.info(f"📊 文件大小: {optimized_display}")
                            st.info("💡 已达到在保持可接受质量下的最佳压缩效果")
                        
                        return optimized_data
                    else:
                        st.warning("⚠️ 智能优化未能显著减小文件大小，返回原始文件")
                        st.info("💡 建议：使用AI智能建议功能或手动调整参数（降低分辨率、帧率或质量）")
                        return gif_data
                else:
                    st.success(f"✅ 文件大小 {gif_size_display} 已满足约束要求 ≤ {target_size_display}")
            
            elif operator in ['>', '>=']:
                if gif_size < target_size:
                    st.info(f"📊 文件大小 {gif_size_display} 小于目标 {target_size_display}")
                else:
                    st.success(f"✅ 文件大小 {gif_size_display} 已满足约束要求 {target_size_display}")
            
            elif operator == '=':
                tolerance = 0.1  # 10%的容差
                if abs(gif_size - target_size) / target_size > tolerance:
                    if gif_size > target_size:
                        status_text.text(f"正在优化GIF文件大小到 {target_size_display}...")
                        optimized_data = optimize_gif_size(gif_data, target_size)
                        if optimized_data:
                            return optimized_data
                        else:
                            return gif_data
                    else:
                        st.info(f"📊 文件大小 {gif_size_display} 小于目标 {target_size_display}")
                else:
                    st.success(f"✅ 文件大小 {gif_size_display} 已满足约束要求 {target_size_display}")
        
        return gif_data
        
    except Exception as e:
        # 确保cap被正确释放
        if cap:
            cap.release()
        
        # 安全地处理异常信息
        error_msg = safe_encode_string(str(e))
        if not error_msg:
            error_msg = "未知错误"
            
        st.error(f"❌ 视频转换失败: {error_msg}")
        st.info("💡 这可能是由于以下原因导致的：")
        st.info("   • 视频文件损坏或格式不兼容")
        st.info("   • 参数设置不当（分辨率过大、质量过高等）")
        st.info("   • 系统内存不足")
        st.info("   • 请尝试上传不同的视频文件或调整参数")
        return None

def optimize_gif_size(gif_data, target_size_bytes):
    """优化GIF文件大小 - 高性能版本，保持最高可能质量"""
    try:
        original_size = len(gif_data)
        
        # 如果目标大小已经满足，直接返回
        if original_size <= target_size_bytes:
            return gif_data
        
        # 计算压缩比例
        compression_ratio = target_size_bytes / original_size
        
        # 将GIF数据转换为PIL图像
        gif_buffer = io.BytesIO(gif_data)
        with Image.open(gif_buffer) as img:
            frames = []
            durations = []
            
            # 预分配数组大小以减少内存重分配
            try:
                frame_count = 0
                while True:
                    frames.append(img.copy())
                    durations.append(img.info.get('duration', 100))
                    img.seek(len(frames))
                    frame_count += 1
                    
                    # 限制最大帧数以控制内存使用
                    if frame_count > 200:
                        break
            except EOFError:
                pass
        
        if not frames:
            return gif_data
        
        target_size_mb = target_size_bytes / (1024 * 1024)
        
        # 智能优化策略 - 优先保持质量，减少尝试次数
        strategies = []
        
        # 根据压缩比例智能选择最优策略
        if compression_ratio >= 0.8:
            # 轻微压缩 - 保持高质量
            strategies = [
                {'scale': 0.9, 'quality': 90, 'fps_reduction': 0.9},
                {'scale': 0.85, 'quality': 85, 'fps_reduction': 0.85}
            ]
        elif compression_ratio >= 0.5:
            # 中等压缩 - 平衡质量和大小
            strategies = [
                {'scale': 0.75, 'quality': 80, 'fps_reduction': 0.75},
                {'scale': 0.65, 'quality': 70, 'fps_reduction': 0.65}
            ]
        elif compression_ratio >= 0.3:
            # 高压缩 - 优先满足大小要求
            strategies = [
                {'scale': 0.55, 'quality': 65, 'fps_reduction': 0.55},
                {'scale': 0.45, 'quality': 55, 'fps_reduction': 0.45}
            ]
        else:
            # 极限压缩 - 确保满足大小要求
            strategies = [
                {'scale': 0.35, 'quality': 50, 'fps_reduction': 0.35},
                {'scale': 0.25, 'quality': 40, 'fps_reduction': 0.25}
            ]
        
        best_result = None
        best_quality_score = 0
        
        for i, strategy in enumerate(strategies):
            try:
                # 计算新的尺寸
                new_width = max(10, int(frames[0].width * strategy['scale']))
                new_height = max(10, int(frames[0].height * strategy['scale']))
                
                # 减少帧数
                frame_interval = max(1, int(1 / strategy['fps_reduction']))
                optimized_frames = frames[::frame_interval]
                optimized_durations = [durations[j] for j in range(0, len(durations), frame_interval)]
                
                # 调整帧尺寸
                resized_frames = []
                for frame in optimized_frames:
                    resized_frame = frame.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    resized_frames.append(resized_frame)
                
                # 保存优化后的GIF
                optimized_buffer = io.BytesIO()
                resized_frames[0].save(
                    optimized_buffer,
                    save_all=True,
                    append_images=resized_frames[1:],
                    duration=optimized_durations[0] if optimized_durations else 100,
                    loop=0,
                    optimize=True,
                    quality=strategy['quality']
                )
                
                optimized_buffer.seek(0)
                optimized_data = optimized_buffer.getvalue()
                optimized_size = len(optimized_data)
                
                # 检查是否满足大小要求
                if optimized_size <= target_size_bytes:
                    # 计算质量评分 (尺寸比例 * 质量比例 * 帧数比例)
                    quality_score = (strategy['scale'] ** 2) * (strategy['quality'] / 100) * strategy['fps_reduction']
                    
                    # 如果质量更好，更新最佳结果
                    if quality_score > best_quality_score:
                        best_quality_score = quality_score
                        best_result = optimized_data
                        
                    # 如果质量已经很高，直接返回
                    if quality_score > 0.8:
                        return optimized_data
                        
            except Exception as e:
                continue
        
        # 如果找到了满足要求的结果，返回最佳质量的那个
        if best_result:
            return best_result
        
        # 如果所有策略都无法满足要求，使用最激进的策略
        try:
            final_scale = max(0.1, compression_ratio ** 0.5)  # 使用平方根来平衡
            final_quality = max(30, int(compression_ratio * 100))
            
            new_width = max(10, int(frames[0].width * final_scale))
            new_height = max(10, int(frames[0].height * final_scale))
            
            # 大幅减少帧数
            frame_interval = max(1, int(1 / (compression_ratio * 0.5)))
            optimized_frames = frames[::frame_interval]
            optimized_durations = [durations[j] for j in range(0, len(durations), frame_interval)]
            
            resized_frames = []
            for frame in optimized_frames:
                resized_frame = frame.resize((new_width, new_height), Image.Resampling.LANCZOS)
                resized_frames.append(resized_frame)
            
            final_buffer = io.BytesIO()
            resized_frames[0].save(
                final_buffer,
                save_all=True,
                append_images=resized_frames[1:],
                duration=optimized_durations[0] if optimized_durations else 100,
                loop=0,
                optimize=True,
                quality=final_quality
            )
            
            final_buffer.seek(0)
            final_data = final_buffer.getvalue()
            
            if len(final_data) <= target_size_bytes:
                return final_data
                
        except Exception as e:
            pass
        
        # 如果仍然无法满足要求，返回最佳结果或原始数据
        if best_result:
            st.warning("⚠️ 无法完全达到目标大小，使用最佳优化结果")
            return best_result
        else:
            st.warning("⚠️ 无法达到目标大小，返回原始文件")
            return gif_data
        
    except Exception as e:
        st.error(f"❌ 优化失败: {str(e)}")
        return gif_data

def cleanup_temp_files():
    """清理临时文件"""
    try:
        temp_dir = Path("temp_uploads")
        if temp_dir.exists():
            for file_path in temp_dir.glob("*"):
                try:
                    if file_path.is_file():
                        file_path.unlink()
                except Exception as e:
                    pass
    except Exception as e:
        pass

def setup_api_key():
    """设置API密钥"""
    with st.expander("🔑 API密钥设置", expanded=not check_api_key()):
        # 如果已经有API密钥，显示当前状态
        if check_api_key():
            st.success("✅ API密钥已设置，可以使用AI智能建议功能")
            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                if st.button("🧪 测试连接", use_container_width=True):
                    with st.spinner("正在测试API连接..."):
                        is_connected, message = test_api_connection()
                        if is_connected:
                            st.success(f"✅ {message}")
                        else:
                            st.error(f"❌ {message}")
            with col2:
                if st.button("🗑️ 清除密钥", use_container_width=True):
                    # 清除密钥
                    st.session_state["api_key"] = None
                    if "api_key_persistent" in st.session_state:
                        del st.session_state.api_key_persistent
                    st.success("✅ API密钥已清除")
                    st.rerun()
            with col3:
                st.write("")  # 占位符
        else:
            st.markdown("请输入您的API密钥以使用AI智能建议功能：")
            
            # API密钥输入区域
            api_key = st.text_input(
                "API密钥",
                type="password",
                placeholder="请输入您的API密钥...",
                help="请输入您的API密钥以启用AI智能建议功能",
                key="api_key_input"
            )
            
            # 设置按钮
            col_set, col_test = st.columns([1, 1])
            with col_set:
                if st.button("✅ 设置密钥", use_container_width=True):
                    if api_key:
                        # 保存密钥
                        st.session_state.api_key_persistent = api_key
                        st.session_state["api_key"] = api_key
                        st.success("✅ API密钥设置成功！")
                        st.rerun()
                    else:
                        st.error("❌ 请输入有效的API密钥")
            
            with col_test:
                if st.button("🧪 测试并设置", use_container_width=True):
                    if api_key:
                        # 临时设置密钥进行测试
                        old_api_key = st.session_state.get("api_key")
                        st.session_state["api_key"] = api_key
                        
                        with st.spinner("正在测试API连接..."):
                            is_connected, message = test_api_connection()
                            
                        if is_connected:
                            # 测试成功，保存密钥
                            st.session_state.api_key_persistent = api_key
                            st.success("✅ 连接测试成功，API密钥已保存！")
                            st.rerun()
                        else:
                            # 测试失败，恢复原始密钥
                            if old_api_key:
                                st.session_state["api_key"] = old_api_key
                            else:
                                del st.session_state["api_key"]
                            st.error(f"❌ 连接测试失败: {message}")
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
                💡 <strong>提示</strong>: 没有API密钥也可以使用手动参数调整功能，但无法使用AI智能建议。
            </div>
            """, unsafe_allow_html=True)

def main():
    """主函数 - 增强的错误处理和容错机制"""
    try:
        # 初始化会话状态
        init_session_state()
    except Exception as e:
        st.error(f"❌ 初始化会话状态失败: {str(e)}")
        st.stop()
    
    # 页面头部
    st.markdown("""
    <div class="main-header">
        <h1>🎬 视频转GIF工具</h1>
        <p>智能视频转换，保持最佳质量 • 支持多种格式 • AI智能建议</p>
    </div>
    """, unsafe_allow_html=True)
    
    # API密钥设置区域
    setup_api_key()
    
    # 添加一些空间
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 简化说明
    if not check_api_key():
        st.info("💡 上传视频后可手动调整参数，设置API密钥后可使用AI智能建议功能")
    
    # 检查OpenCV是否可用
    if not OPENCV_AVAILABLE:
        st.error("❌ OpenCV不可用，无法进行视频处理")
        st.info("💡 请确保安装了opencv-python-headless包")
        return
    
    # 文件上传区域
    st.markdown("### 📁 文件上传")
    
    uploaded_file = st.file_uploader(
        "选择视频文件",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="支持MP4、AVI、MOV、MKV等格式，建议文件大小不超过100MB，转换时间取决于文件大小",
        key="uploaded_file"
    )
    
    if uploaded_file:
        # 保存上传的文件
        temp_dir = Path("temp_uploads")
        temp_dir.mkdir(exist_ok=True)
        
        input_path = temp_dir / uploaded_file.name
        try:
            with open(input_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        except Exception as e:
            st.error(f"❌ 文件保存失败: {str(e)}")
            return
        
        st.session_state.video_file = str(input_path)
        st.success(f"✅ 文件上传成功: {uploaded_file.name}")
        st.info("💡 系统将自动分析视频并设置最优参数")
        
        # 获取视频信息，增强错误处理
        try:
            video_info = analyze_video_properties(input_path)
        except Exception as e:
            st.error(f"❌ 分析视频文件时出现异常: {str(e)}")
            st.info("💡 请尝试上传不同的视频文件")
            video_info = None
        
        # 显示文件信息
        if video_info:
            try:
                file_size = input_path.stat().st_size
                file_size_mb = file_size / (1024*1024)
                file_size_kb = file_size / 1024
                
                size_display = f"{file_size_mb:.2f} MB" if file_size_mb >= 1 else f"{file_size_kb:.1f} KB"
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("时长", f"{video_info['duration']:.1f}秒")
                with col2:
                    st.metric("帧率", f"{video_info['fps']:.1f} FPS")
                with col3:
                    st.metric("分辨率", f"{video_info['width']}×{video_info['height']}")
                with col4:
                    st.metric("文件大小", size_display)
            except Exception as e:
                st.warning(f"⚠️ 获取文件信息失败: {str(e)}")
    
    # AI智能建议区域
    if uploaded_file and video_info:
        st.markdown("### 🤖 AI智能建议")
        
        # 自然语言输入
        user_input = st.text_input(
            "描述您的需求（可选）",
            placeholder="例如：我想要一个高清gif、压缩版gif、平衡版gif...",
            help="用自然语言描述您的需求，AI将为您推荐最优参数"
        )
        
        # 生成AI建议
        col_ai1, col_ai2 = st.columns([3, 1])
        with col_ai1:
            if st.button("🎯 获取AI建议", use_container_width=True):
                try:
                    with st.spinner("AI正在分析并生成建议..."):
                        suggestions = generate_ai_suggestions(video_info, user_input)
                        if suggestions:
                            st.session_state.ai_suggestions = suggestions
                        else:
                            st.warning("⚠️ 未能生成AI建议，将使用默认建议")
                            fallback_suggestions = get_fallback_suggestions(video_info, user_input)
                            st.session_state.ai_suggestions = fallback_suggestions
                except Exception as e:
                    st.error(f"❌ 生成AI建议时出错: {str(e)}")
                    st.info("💡 将使用默认建议作为替代方案")
                    try:
                        fallback_suggestions = get_fallback_suggestions(video_info, user_input)
                        st.session_state.ai_suggestions = fallback_suggestions
                    except Exception as fallback_error:
                        st.error(f"❌ 生成默认建议也失败了: {str(fallback_error)}")
                        st.info("💡 请手动调整参数")
        
        with col_ai2:
            if st.button("🔄 重新分析", use_container_width=True, help="清除缓存并重新获取AI建议"):
                # 清除AI建议缓存
                if 'ai_suggestions_cache' in st.session_state:
                    st.session_state.ai_suggestions_cache.clear()
                if 'ai_suggestions' in st.session_state:
                    del st.session_state.ai_suggestions
                
                # 清除所有建议应用状态
                keys_to_remove = []
                for key in st.session_state.keys():
                    if key.startswith('suggestion_') and key.endswith('_applied'):
                        keys_to_remove.append(key)
                for key in keys_to_remove:
                    del st.session_state[key]
                
                st.success("✅ 缓存已清除，请重新获取AI建议")
                st.rerun()
        
        # 显示AI建议
        if st.session_state.ai_suggestions:
            try:
                st.success(f"✅ AI为您生成了 {len(st.session_state.ai_suggestions)} 个优化建议")
                
                # 显示建议
                num_suggestions = len(st.session_state.ai_suggestions)
                if num_suggestions > 0:
                    cols = st.columns(min(num_suggestions, 4))  # 限制最大列数为4
                    for i, suggestion in enumerate(st.session_state.ai_suggestions):
                        if not suggestion:  # 跳过空建议
                            continue
                        col_index = i % len(cols)  # 循环使用列
                        with cols[col_index]:
                            # 构建文件大小约束和预估信息
                            size_info_parts = []
                            
                            # 添加预估大小信息
                            try:
                                if 'estimated_size' in suggestion and suggestion['estimated_size']:
                                    estimated_size = float(suggestion['estimated_size'])
                                    estimated_mb = estimated_size / (1024 * 1024)
                                    if estimated_mb >= 1:
                                        size_info_parts.append(f"预估: {estimated_mb:.1f}MB")
                                    else:
                                        estimated_kb = estimated_size / 1024
                                        size_info_parts.append(f"预估: {estimated_kb:.0f}KB")
                            except (ValueError, TypeError, ZeroDivisionError):
                                # 如果预估大小信息有问题，跳过显示
                                pass
                            
                            # 添加约束信息
                            try:
                                if ('size_constraint' in suggestion and 
                                    suggestion['size_constraint'] and 
                                    suggestion['size_constraint'].get('enabled', False)):
                                    constraint = suggestion['size_constraint']
                                    operator = constraint.get('operator', '<')
                                    value = constraint.get('value', 5.0)
                                    unit = constraint.get('unit', 'MB')
                                    size_info_parts.append(f"约束: {operator} {value}{unit}")
                            except (KeyError, TypeError):
                                # 如果约束信息有问题，跳过显示
                                pass
                            
                            size_constraint_info = ""
                            if size_info_parts:
                                size_constraint_info = " | " + " | ".join(size_info_parts)
                            
                            # 安全地获取建议信息
                            try:
                                name = suggestion.get('name', '未知方案')
                                description = suggestion.get('description', '暂无描述')
                                params = suggestion.get('params', {})
                                fps = params.get('fps', '未知')
                                quality = params.get('quality', '未知')
                                
                                st.markdown(f"""
                                <div class="ai-suggestion">
                                    <h4>{name}</h4>
                                    <p>{description}</p>
                                    <small>FPS: {fps} | 质量: {quality}%{size_constraint_info}</small>
                                </div>
                                """, unsafe_allow_html=True)
                            except Exception as e:
                                st.error(f"显示建议信息时出错: {str(e)}")
                                st.json(suggestion)  # 调试用，显示原始建议数据
                            
                            # 创建唯一的会话状态键来跟踪按钮点击
                            button_clicked_key = f"suggestion_{i}_applied"
                            
                            if button_clicked_key not in st.session_state:
                                st.session_state[button_clicked_key] = False
                            
                            # 如果已经应用过，显示不同的按钮
                            if st.session_state[button_clicked_key]:
                                st.button("✅ 已应用", key=f"applied_suggestion_{i}", disabled=True, use_container_width=True)
                            elif st.button(f"使用此建议", key=f"use_suggestion_{i}", use_container_width=True):
                                # 最简化的安全实现
                                success = False
                                try:
                                    # 检查建议是否有效
                                    if not suggestion or not isinstance(suggestion, dict):
                                        st.error("❌ 建议数据无效")
                                    else:
                                        # 尝试应用参数
                                        if 'params' in suggestion:
                                            params = suggestion['params']
                                            if params and isinstance(params, dict):
                                                # 使用最保守的方法设置参数
                                                if 'fps' in params:
                                                    st.session_state.conversion_params['fps'] = int(params['fps'])
                                                if 'quality' in params:
                                                    st.session_state.conversion_params['quality'] = int(params['quality'])
                                                if 'width' in params:
                                                    st.session_state.conversion_params['width'] = int(params['width'])
                                                if 'height' in params:
                                                    st.session_state.conversion_params['height'] = int(params['height'])
                                                if 'optimize' in params:
                                                    st.session_state.conversion_params['optimize'] = bool(params['optimize'])
                                                success = True
                                        
                                        # 尝试应用约束
                                        if 'size_constraint' in suggestion:
                                            constraint = suggestion['size_constraint']
                                            if constraint and isinstance(constraint, dict):
                                                if 'operator' in constraint:
                                                    st.session_state.size_constraint['operator'] = str(constraint['operator'])
                                                if 'value' in constraint:
                                                    st.session_state.size_constraint['value'] = float(constraint['value'])
                                                if 'unit' in constraint:
                                                    st.session_state.size_constraint['unit'] = str(constraint['unit'])
                                                if 'enabled' in constraint:
                                                    st.session_state.size_constraint['enabled'] = bool(constraint['enabled'])
                                                # 重新计算target_size
                                                value = st.session_state.size_constraint['value']
                                                unit = st.session_state.size_constraint['unit']
                                                multipliers = {'B': 1, 'KB': 1024, 'MB': 1048576, 'GB': 1073741824}
                                                st.session_state.size_constraint['target_size'] = value * multipliers.get(unit.upper(), 1048576)
                                                success = True
                                        
                                        if success:
                                            st.session_state[button_clicked_key] = True
                                            st.success("✅ 建议已成功应用！")
                                            st.balloons()
                                        else:
                                            st.warning("⚠️ 建议数据不完整，无法应用")
                                        
                                except Exception as e:
                                    st.error(f"❌ 应用失败: {type(e).__name__}")
                                    if hasattr(e, 'args') and e.args:
                                        st.code(str(e.args[0]))
                                    st.info("💡 请手动调整参数")
            except Exception as e:
                st.error(f"❌ 显示AI建议时出错: {str(e)}")
                st.info("💡 请尝试重新获取建议或使用手动参数调整")
    
    # 参数设置区域
    if uploaded_file:
        st.markdown("### ⚙️ 转换参数")
        st.info("💡 参数已根据视频特性自动优化，您可以根据需要调整，建议保持默认设置以获得最佳效果")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            fps = st.slider(
                "帧率 (FPS)", 
                min_value=1, 
                max_value=30, 
                value=st.session_state.conversion_params.get('fps', 10),
                help="控制GIF播放速度，建议8-15FPS，数值越高动画越流畅但文件越大"
            )
            
            quality = st.slider(
                "质量", 
                min_value=50, 
                max_value=100, 
                value=st.session_state.conversion_params.get('quality', 85),
                help="控制GIF画质，数值越高画质越好但文件越大"
            )
        
        with col2:
            if video_info:
                scale_ratio = st.slider(
                    "缩放比例", 
                    min_value=0.1, 
                    max_value=2.0, 
                    value=1.0,
                    step=0.1,
                    help="控制GIF分辨率比例，1.0为原始尺寸，数值越大画质越好但文件越大"
                )
                
                # 根据比例计算目标尺寸
                target_width = int(video_info['width'] * scale_ratio)
                target_height = int(video_info['height'] * scale_ratio)
                
                # 显示当前尺寸信息
                st.info(f"📐 目标尺寸: {target_width}×{target_height} (原始: {video_info['width']}×{video_info['height']})")
            else:
                target_width = 640
                target_height = 480
        
        optimize = st.checkbox(
            "启用优化", 
            value=st.session_state.conversion_params.get('optimize', True),
            help="优化GIF文件大小，建议启用"
        )
        
        # 更新参数
        st.session_state.conversion_params.update({
            'fps': fps,
            'quality': quality,
            'width': target_width,
            'height': target_height,
            'optimize': optimize
        })
        
        # 文件大小约束设置
        st.markdown('<div class="constraint-section">', unsafe_allow_html=True)
        st.markdown("### 📏 文件大小约束")
        st.info("💡 设置文件大小约束可自动优化输出文件，系统会智能调整参数以达到最佳效果")
        
        col_constraint1, col_constraint2, col_constraint3 = st.columns([1, 1, 1])
        
        with col_constraint1:
            # 获取当前操作符的索引
            operator_options = ["<", ">", "=", "<=", ">="]
            current_operator = st.session_state.size_constraint.get('operator', '<')
            operator_index = operator_options.index(current_operator) if current_operator in operator_options else 0
            
            operator = st.selectbox(
                "比较符号",
                operator_options,
                index=operator_index,
                help="选择文件大小约束条件"
            )
        
        with col_constraint2:
            size_value = st.number_input(
                "数值",
                min_value=0.1,
                max_value=1000.0,
                value=st.session_state.size_constraint.get('value', 5.0),
                step=0.1,
                help="输入目标文件大小数值"
            )
        
        with col_constraint3:
            # 获取当前单位的索引
            unit_options = ["MB", "KB", "B"]
            current_unit = st.session_state.size_constraint.get('unit', 'MB')
            unit_index = unit_options.index(current_unit) if current_unit in unit_options else 0
            
            size_unit = st.selectbox(
                "单位",
                unit_options,
                index=unit_index,
                help="选择文件大小单位"
            )
        
        # 解析约束
        size_constraint = parse_size_constraint(operator, size_value, size_unit)
        if size_constraint:
            st.session_state.size_constraint = {
                'operator': operator,
                'value': size_value,
                'unit': size_unit,
                'enabled': True,
                'target_size': size_constraint['target_size']
            }
            
            target_size_mb = size_constraint['target_size'] / (1024 * 1024)
            target_size_kb = size_constraint['target_size'] / 1024
            
            if target_size_mb >= 1:
                size_display = f"{target_size_mb:.2f}MB"
            else:
                size_display = f"{target_size_kb:.1f}KB"
            
            st.success(f"✅ 约束设置: {size_constraint['display']}")
        else:
            st.session_state.size_constraint['enabled'] = False
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 显示当前参数的预估文件大小（基于最终约束调整后的参数）
        if video_info:
            # 获取当前视频文件路径
            current_video_path = st.session_state.get('video_file')
            
            # 生成参数状态键，避免重复计算
            params_state_key = f"{st.session_state.conversion_params}_{st.session_state.size_constraint.get('enabled', False)}_{st.session_state.size_constraint.get('value', 0)}"
            
            # 检查是否需要重新计算
            if 'last_params_state_key' not in st.session_state or st.session_state.last_params_state_key != params_state_key:
                with st.spinner("🔄 正在预估参数，请稍后..."):
                    # 根据当前约束调整参数
                    if st.session_state.size_constraint.get('enabled', False):
                        constraint = st.session_state.size_constraint.copy()
                        
                        # 确保约束包含target_size字段
                        if 'target_size' not in constraint:
                            unit_multipliers = {
                                'B': 1,
                                'KB': 1024,
                                'MB': 1024 * 1024,
                                'GB': 1024 * 1024 * 1024
                            }
                            multiplier = unit_multipliers.get(constraint.get('unit', 'MB').upper(), 1024 * 1024)
                            constraint['target_size'] = constraint.get('value', 5.0) * multiplier
                        
                        # 基于约束调整参数
                        adjusted_params = adjust_params_for_constraint(
                            video_info, 
                            st.session_state.conversion_params, 
                            constraint,
                            current_video_path
                        )
                        
                        # 使用调整后的参数进行预估
                        estimated_size = estimate_gif_size(video_info, adjusted_params, current_video_path)
                        
                        # 验证是否满足约束
                        satisfied, _ = validate_params_against_constraint(
                            video_info, 
                            adjusted_params, 
                            constraint,
                            current_video_path
                        )
                        
                        # 显示约束信息
                        target_size_mb = constraint['target_size'] / (1024 * 1024)
                        target_size_kb = constraint['target_size'] / 1024
                        
                        if target_size_mb >= 1:
                            target_display = f"{target_size_mb:.1f}MB"
                        else:
                            target_display = f"{target_size_kb:.0f}KB"
                        
                        # 更新会话状态中的参数为调整后的参数
                        st.session_state.conversion_params.update(adjusted_params)
                        
                    else:
                        # 没有约束时，使用原始参数
                        estimated_size = estimate_gif_size(video_info, st.session_state.conversion_params, current_video_path)
                        satisfied = True
                        constraint = None
                    
                    # 缓存计算结果
                    st.session_state.last_params_state_key = params_state_key
                    st.session_state.cached_estimated_size = estimated_size
                    st.session_state.cached_constraint_satisfied = satisfied
                    st.session_state.cached_constraint = constraint
            
            else:
                # 使用缓存的结果
                estimated_size = st.session_state.cached_estimated_size
                satisfied = st.session_state.cached_constraint_satisfied
                constraint = st.session_state.cached_constraint
                if constraint:
                    target_size_mb = constraint['target_size'] / (1024 * 1024)
                    target_size_kb = constraint['target_size'] / 1024
                    
                    if target_size_mb >= 1:
                        target_display = f"{target_size_mb:.1f}MB"
                    else:
                        target_display = f"{target_size_kb:.0f}KB"
            
            # 显示预估结果
            estimated_mb = estimated_size / (1024 * 1024)
            estimated_kb = estimated_size / 1024
            
            if estimated_mb >= 1:
                size_display = f"{estimated_mb:.2f}MB"
            else:
                size_display = f"{estimated_kb:.1f}KB"
            
            # 如果有约束，在一个信息中显示预估大小和约束状态
            if constraint and constraint.get('enabled', False):
                if satisfied:
                    st.success(f"✅ 基于当前参数文件转换后的预估大小为: {size_display} (满足约束 {constraint['operator']} {target_display})")
                else:
                    st.warning(f"⚠️ 基于当前参数文件转换后的预估大小为: {size_display} (不满足约束 {constraint['operator']} {target_display})")
            else:
                st.info(f"📊 基于当前参数文件转换后的预估大小为: {size_display}")
            
            # 显示调整提示
            st.info("💡 提示：降低质量、帧率或分辨率可以减小文件大小")
        
        # 转换按钮
        st.markdown("---")
        st.markdown("### 🚀 开始转换")
        
        col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 1])
        
        with col_btn1:
            if st.button("🔄 开始转换", type="primary", use_container_width=True, help="点击开始转换视频为GIF格式"):
                # 创建输出文件名
                output_filename = f"{uploaded_file.name.rsplit('.', 1)[0]}.gif"
                
                # 显示转换进度
                with st.spinner("🔄 正在转换视频为GIF..."):
                    gif_data = convert_video_to_gif(
                        input_path, 
                        st.session_state.conversion_params,
                        st.session_state.size_constraint
                    )
                    
                    if gif_data:
                        st.session_state.gif_data = gif_data
                        
                        # 分模块展示结果
                        st.markdown('<div class="success-card">', unsafe_allow_html=True)
                        st.success("✅ 转换完成！")
                        st.balloons()
                        
                        # 转换完成后的信息模块
                        st.markdown("### 📊 转换结果")
                        col_info1, col_info2 = st.columns(2)
                        with col_info1:
                            gif_size = len(gif_data) / (1024 * 1024)
                            gif_size_kb = len(gif_data) / 1024
                            
                            if gif_size >= 1:
                                size_display = f"{gif_size:.2f} MB"
                            else:
                                size_display = f"{gif_size_kb:.1f} KB"
                            
                            st.metric("📊 输出文件大小", size_display)
                        
                        with col_info2:
                            # 获取GIF分辨率信息
                            try:
                                gif_buffer = io.BytesIO(gif_data)
                                with Image.open(gif_buffer) as img:
                                    st.metric("📐 输出文件分辨率", f"{img.width}×{img.height}")
                            except Exception as e:
                                st.metric("📐 输出文件分辨率", "未知")
                        
                        # 下载模块
                        st.markdown("### 📥 下载")
                        try:
                            st.download_button(
                                label="📥 下载GIF文件",
                                data=gif_data,
                                file_name=output_filename,
                                mime="image/gif",
                                use_container_width=True,
                                type="primary",
                                help="点击下载转换完成的GIF文件"
                            )
                        except Exception as e:
                            st.error(f"❌ 下载文件失败: {str(e)}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.error("❌ 转换失败")
                        st.info("💡 请检查视频文件格式或调整参数")
        
        with col_btn2:
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
        
        with col_btn3:
            if st.button("🔄 重置智能体", use_container_width=True, help="清空所有数据并重新开始"):
                reset_agent()

if __name__ == "__main__":
    main()
