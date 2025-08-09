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

# 页面配置
st.set_page_config(
    page_title="视频转GIF工具 - Findknow AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

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
    """获取AI客户端"""
    api_key = st.session_state.get("api_key") or st.session_state.get("api_key_persistent")
    if not api_key:
        return None
    
    try:
        client = OpenAI(
            api_key=api_key,
            base_url=AI_CONFIG["base_url"]
        )
        return client
    except Exception as e:
        st.error(f"AI客户端初始化失败: {e}")
        return None

def check_api_key():
    """检查API密钥是否已设置"""
    return st.session_state.get("api_key") or st.session_state.get("api_key_persistent")

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
            'enabled': False
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
        'ai_suggestions', 'uploaded_file'
    ]
    
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    
    # 重新初始化会话状态
    init_session_state()
    
    # 强制重新运行页面，清除所有UI状态
    st.rerun()

def analyze_video_properties(video_path):
    """分析视频属性"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        # 获取视频属性
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        
        # 获取文件大小
        file_size = os.path.getsize(video_path)
        
        cap.release()
        
        return {
            'fps': fps,
            'frame_count': frame_count,
            'width': width,
            'height': height,
            'duration': duration,
            'file_size': file_size
        }
    except Exception as e:
        st.error(f"分析视频属性失败: {e}")
        return None

def generate_ai_suggestions(video_props, user_input=""):
    """生成AI建议 - 使用真实AI大模型分析用户意图和视频特征"""
    
    # 检查API密钥
    if not check_api_key():
        st.warning("⚠️ 需要设置API密钥才能使用AI智能建议功能")
        st.info("💡 请返回主页设置API密钥，或使用下方的手动参数调整")
        return []
    
    # 获取AI客户端
    client = get_ai_client()
    if not client:
        st.error("❌ AI客户端初始化失败")
        return []
    
    if not video_props:
        return []
    
    try:
        # 构建专业的系统提示词
        system_prompt = """你是一位专业的视频转GIF优化专家，具备深厚的多媒体技术知识和用户体验设计经验。

核心职责：
1. 分析用户的自然语言需求，准确识别使用场景和质量期望
2. 结合视频技术参数，提供专业的转换参数建议
3. 平衡文件大小、视觉质量和加载速度的关系
4. 针对不同应用场景提供最优解决方案

技术专业知识：
- GIF格式特性：调色板限制、循环播放、逐帧压缩
- 关键参数影响：帧率影响流畅度、质量影响色彩保真度、分辨率影响清晰度
- 压缩策略：关键帧采样、色彩量化、尺寸缩放的综合运用
- 应用场景优化：社交媒体、网页展示、邮件附件等不同需求

回复格式要求：
返回一个JSON数组，包含2-4个建议方案。每个方案必须包含：
- name: 方案名称（简洁有力，体现特色）
- description: 详细说明（说明适用场景和优势）
- params: 技术参数
  - fps: 帧率 (1-30)
  - quality: 质量百分比 (50-100)
  - width: 目标宽度
  - height: 目标高度  
  - optimize: 是否启用优化 (true/false)
- size_constraint: 文件大小约束
  - operator: 比较符 ("<", ">", "=", "<=", ">=")
  - value: 数值
  - unit: 单位 ("B", "KB", "MB", "GB")
  - enabled: 是否启用 (true/false)

示例：
[
  {
    "name": "高质量专业版",
    "description": "保持最高视觉质量，适合专业展示和高质量要求场景",
    "params": {
      "fps": 15,
      "quality": 95,
      "width": 1280,
      "height": 720,
      "optimize": true
    },
    "size_constraint": {
      "operator": "<",
      "value": 10.0,
      "unit": "MB",
      "enabled": true
    }
  }
]

注意事项：
- 必须返回有效的JSON格式
- 参数值必须在合理范围内
- 建议方案要有明显区别和针对性
- 文件大小约束要现实可行"""

        # 构建用户查询
        user_prompt = f"""请为以下视频转GIF需求提供专业建议：

视频技术参数：
- 原始分辨率：{video_props['width']}×{video_props['height']}
- 原始帧率：{video_props['fps']:.1f} FPS
- 视频时长：{video_props['duration']:.1f}秒
- 文件大小：{video_props['file_size'] / (1024*1024):.2f}MB
- 总帧数：{video_props['frame_count']}帧

用户需求描述：
{user_input if user_input.strip() else "用户未提供具体需求，请提供通用的优化建议"}

请基于以上信息，提供2-4个专业的GIF转换方案建议。每个方案要有明确的定位和适用场景。"""

        # 调用AI进行分析
        with st.spinner("🤖 AI正在分析视频参数和用户需求..."):
            completion = client.chat.completions.create(
                model=AI_CONFIG["model"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            ai_response = completion.choices[0].message.content
            
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
                            validated_suggestions.append(suggestion)
                    
                    if validated_suggestions:
                        st.success(f"✅ AI成功生成了 {len(validated_suggestions)} 个专业建议")
                        return validated_suggestions
                    else:
                        st.warning("⚠️ AI生成的建议格式有误，使用默认建议")
                        return get_fallback_suggestions(video_props, user_input)
                
                else:
                    st.warning("⚠️ AI响应格式异常，使用默认建议")
                    return get_fallback_suggestions(video_props, user_input)
                    
            except json.JSONDecodeError as e:
                st.warning(f"⚠️ AI响应解析失败: {str(e)}")
                return get_fallback_suggestions(video_props, user_input)
                
    except Exception as e:
        st.error(f"❌ AI分析失败: {str(e)}")
        st.info("💡 将使用默认建议作为备选方案")
        return get_fallback_suggestions(video_props, user_input)

def validate_suggestion(suggestion, video_props):
    """验证AI建议的有效性"""
    try:
        # 检查必需字段
        required_fields = ['name', 'description', 'params', 'size_constraint']
        if not all(field in suggestion for field in required_fields):
            return False
        
        params = suggestion['params']
        required_params = ['fps', 'quality', 'width', 'height', 'optimize']
        if not all(param in params for param in required_params):
            return False
        
        # 验证参数范围
        if not (1 <= params['fps'] <= 30):
            return False
        if not (50 <= params['quality'] <= 100):
            return False
        if not (10 <= params['width'] <= video_props['width'] * 2):
            return False
        if not (10 <= params['height'] <= video_props['height'] * 2):
            return False
        
        # 验证尺寸约束
        constraint = suggestion['size_constraint']
        required_constraint_fields = ['operator', 'value', 'unit', 'enabled']
        if not all(field in constraint for field in required_constraint_fields):
            return False
        
        if constraint['operator'] not in ['<', '>', '=', '<=', '>=']:
            return False
        if constraint['unit'] not in ['B', 'KB', 'MB', 'GB']:
            return False
        if not isinstance(constraint['value'], (int, float)) or constraint['value'] <= 0:
            return False
        
        return True
        
    except Exception:
        return False

def get_fallback_suggestions(video_props, user_input=""):
    """获取备选建议（当AI失败时使用）"""
    fps = video_props['fps']
    width = video_props['width']
    height = video_props['height']
    duration = video_props['duration']
    file_size = video_props['file_size']
    
    suggestions = []
    
    # 高质量版本
    suggestions.append({
        'name': '高质量专业版',
        'description': '保持最高视觉质量，适合专业展示和演示用途',
        'params': {
            'fps': min(int(fps), 15),
            'quality': 95,
            'width': width,
            'height': height,
            'optimize': True
        },
        'size_constraint': {
            'operator': '<',
            'value': 10.0,
            'unit': 'MB',
            'enabled': True
        }
    })
    
    # 平衡版本
    suggestions.append({
        'name': '平衡优化版',
        'description': '质量与文件大小的最佳平衡，适合一般分享使用',
        'params': {
            'fps': min(int(fps), 12),
            'quality': 85,
            'width': min(width, 640),
            'height': min(height, 480),
            'optimize': True
        },
        'size_constraint': {
            'operator': '<',
            'value': 5.0,
            'unit': 'MB',
            'enabled': True
        }
    })
    
    # 压缩版本
    if duration > 10 or file_size > 20 * 1024 * 1024:
        suggestions.append({
            'name': '小文件分享版',
            'description': '大幅压缩文件大小，适合网络传输和存储空间有限的场景',
            'params': {
                'fps': 8,
                'quality': 75,
                'width': min(width, 480),
                'height': min(height, 360),
                'optimize': True
            },
            'size_constraint': {
                'operator': '<',
                'value': 2.0,
                'unit': 'MB',
                'enabled': True
            }
        })
    
    return suggestions

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
    """将视频转换为GIF - 优化版本"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("无法打开视频文件")
            return None
        
        frames = []
        fps = params['fps']
        target_width = params['width']
        target_height = params['height']
        quality = params['quality']
        
        # 计算采样间隔
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        if original_fps > 0:
            sample_interval = max(1, int(original_fps / fps))
        else:
            sample_interval = 1
        
        # 限制最大帧数以提高速度
        max_frames = min(200, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) // sample_interval))
        
        frame_count = 0
        processed_frames = 0
        
        # 创建进度条
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        while True:
            ret, frame = cap.read()
            if not ret or processed_frames >= max_frames:
                break
            
            # 按间隔采样
            if frame_count % sample_interval == 0:
                try:
                    # 调整尺寸
                    if target_width and target_height:
                        frame = cv2.resize(frame, (target_width, target_height))
                    
                    # BGR转RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # 转换为PIL图像
                    pil_image = Image.fromarray(frame_rgb)
                    frames.append(pil_image)
                    processed_frames += 1
                    
                    # 更新进度
                    progress = processed_frames / max_frames
                    progress_bar.progress(progress)
                    status_text.text(f"正在处理视频帧... {processed_frames}/{max_frames}")
                    
                    # 定期清理内存
                    if processed_frames % 50 == 0:
                        gc.collect()
                        
                except Exception as e:
                    continue
            
            frame_count += 1
        
        cap.release()
        
        if not frames:
            st.error("没有提取到有效帧")
            return None
        
        # 创建GIF
        status_text.text("正在生成GIF文件...")
        gif_buffer = io.BytesIO()
        frames[0].save(
            gif_buffer,
            format='GIF',
            save_all=True,
            append_images=frames[1:],
            duration=int(1000 / fps),  # 毫秒
            loop=0,
            optimize=params['optimize'],
            quality=quality
        )
        
        gif_buffer.seek(0)
        gif_data = gif_buffer.getvalue()
        
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
                    status_text.text(f"正在强制优化GIF文件大小到 {target_size_display} 以下...")
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
                            st.success(f"✅ 优化成功！文件大小从 {gif_size_display} 减小到 {optimized_display}")
                        else:
                            st.warning(f"⚠️ 优化后文件大小 {optimized_display}，仍超过目标 {target_size_display}")
                            st.info("💡 建议：请尝试调整'缩放比例'参数来进一步压缩文件大小")
                        
                        return optimized_data
                    else:
                        st.error("❌ 优化失败，返回原始文件")
                        return gif_data
                else:
                    st.success(f"✅ 文件大小 {gif_size_display} 已满足约束要求 {target_size_display}")
            
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
        st.error(f"转换失败: {e}")
        return None

def optimize_gif_size(gif_data, target_size_bytes):
    """优化GIF文件大小 - 保持最高可能质量"""
    try:
        # 将GIF数据转换为PIL图像
        gif_buffer = io.BytesIO(gif_data)
        with Image.open(gif_buffer) as img:
            frames = []
            durations = []
            
            try:
                while True:
                    frames.append(img.copy())
                    durations.append(img.info.get('duration', 100))
                    img.seek(len(frames))
            except EOFError:
                pass
        
        if not frames:
            return gif_data
        
        original_size = len(gif_data)
        target_size_mb = target_size_bytes / (1024 * 1024)
        
        # 如果目标大小已经满足，直接返回
        if original_size <= target_size_bytes:
            return gif_data
        
        # 计算压缩比例
        compression_ratio = target_size_bytes / original_size
        
        # 智能优化策略 - 优先保持质量
        strategies = []
        
        # 策略1: 轻微压缩 - 保持高质量
        if compression_ratio >= 0.8:
            strategies.extend([
                {'scale': 0.95, 'quality': 95, 'fps_reduction': 0.95},
                {'scale': 0.9, 'quality': 90, 'fps_reduction': 0.9},
                {'scale': 0.85, 'quality': 85, 'fps_reduction': 0.85}
            ])
        
        # 策略2: 中等压缩 - 平衡质量和大小
        elif compression_ratio >= 0.5:
            strategies.extend([
                {'scale': 0.8, 'quality': 85, 'fps_reduction': 0.8},
                {'scale': 0.75, 'quality': 80, 'fps_reduction': 0.75},
                {'scale': 0.7, 'quality': 75, 'fps_reduction': 0.7},
                {'scale': 0.65, 'quality': 70, 'fps_reduction': 0.65}
            ])
        
        # 策略3: 高压缩 - 优先满足大小要求
        elif compression_ratio >= 0.3:
            strategies.extend([
                {'scale': 0.6, 'quality': 70, 'fps_reduction': 0.6},
                {'scale': 0.55, 'quality': 65, 'fps_reduction': 0.55},
                {'scale': 0.5, 'quality': 60, 'fps_reduction': 0.5}
            ])
        
        # 策略4: 极限压缩 - 确保满足大小要求
        else:
            strategies.extend([
                {'scale': 0.4, 'quality': 55, 'fps_reduction': 0.4},
                {'scale': 0.35, 'quality': 50, 'fps_reduction': 0.35},
                {'scale': 0.3, 'quality': 45, 'fps_reduction': 0.3},
                {'scale': 0.25, 'quality': 40, 'fps_reduction': 0.25}
            ])
        
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
            col1, col2 = st.columns([1, 3])
            with col1:
                if st.button("🗑️ 清除密钥", use_container_width=True):
                    # 清除密钥
                    st.session_state["api_key"] = None
                    if "api_key_persistent" in st.session_state:
                        del st.session_state.api_key_persistent
                    st.success("✅ API密钥已清除")
                    st.rerun()
            with col2:
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
            if st.button("✅ 设置密钥", use_container_width=True):
                if api_key:
                    # 保存密钥
                    st.session_state.api_key_persistent = api_key
                    st.session_state["api_key"] = api_key
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
                💡 <strong>提示</strong>: 没有API密钥也可以使用手动参数调整功能，但无法使用AI智能建议。
            </div>
            """, unsafe_allow_html=True)

def main():
    """主函数"""
    # 初始化会话状态
    init_session_state()
    
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
    
    # 添加一些说明
    if check_api_key():
        st.info("💡 欢迎使用视频转GIF工具！上传视频后AI将自动分析并生成智能建议，也可手动调整参数")
    else:
        st.info("💡 欢迎使用视频转GIF工具！上传视频后可手动调整参数，设置API密钥后可使用AI智能建议功能")
    
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
        
        # 获取视频信息
        video_info = analyze_video_properties(input_path)
        
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
        if st.button("🎯 获取AI建议", use_container_width=True):
            with st.spinner("AI正在分析并生成建议..."):
                suggestions = generate_ai_suggestions(video_info, user_input)
                st.session_state.ai_suggestions = suggestions
        
        # 显示AI建议
        if st.session_state.ai_suggestions:
            st.success(f"✅ AI为您生成了 {len(st.session_state.ai_suggestions)} 个优化建议")
            
            # 显示建议
            cols = st.columns(len(st.session_state.ai_suggestions))
            for i, suggestion in enumerate(st.session_state.ai_suggestions):
                with cols[i]:
                    # 构建文件大小约束显示信息
                    size_constraint_info = ""
                    if 'size_constraint' in suggestion and suggestion['size_constraint']['enabled']:
                        constraint = suggestion['size_constraint']
                        size_constraint_info = f" | 文件大小: {constraint['operator']} {constraint['value']}{constraint['unit']}"
                    
                    st.markdown(f"""
                    <div class="ai-suggestion">
                        <h4>{suggestion['name']}</h4>
                        <p>{suggestion['description']}</p>
                        <small>FPS: {suggestion['params']['fps']} | 质量: {suggestion['params']['quality']}%{size_constraint_info}</small>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button(f"使用此建议", key=f"use_suggestion_{i}", use_container_width=True):
                        # 应用转换参数
                        st.session_state.conversion_params = suggestion['params'].copy()
                        
                        # 应用文件大小约束
                        if 'size_constraint' in suggestion:
                            st.session_state.size_constraint = suggestion['size_constraint'].copy()
                        
                        st.success("✅ 参数和文件大小约束已应用！")
                        st.rerun()
    
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
