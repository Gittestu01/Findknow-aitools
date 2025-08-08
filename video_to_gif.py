import streamlit as st
import os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import io
import time
import gc

# 页面配置
st.set_page_config(
    page_title="视频转GIF工具 - Findknow AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

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
    """生成AI建议 - 智能分析用户意图和需求"""
    suggestions = []
    
    if not video_props:
        return suggestions
    
    # 基于视频属性生成建议
    fps = video_props['fps']
    duration = video_props['duration']
    file_size = video_props['file_size']
    width = video_props['width']
    height = video_props['height']
    
    # 分析用户输入 - 智能意图识别
    user_input_lower = user_input.lower()
    
    # 意图分析：检测用户是否明确要求压缩
    explicit_compression_keywords = ["压缩", "小文件", "快速", "减小", "缩小", "节省空间", "网络分享", "上传限制"]
    has_explicit_compression = any(keyword in user_input_lower for keyword in explicit_compression_keywords)
    
    # 意图分析：检测用户是否要求高质量
    quality_keywords = ["高清", "高质量", "原版", "最佳质量", "最高质量", "无损", "专业"]
    has_quality_demand = any(keyword in user_input_lower for keyword in quality_keywords)
    
    # 意图分析：检测特定使用场景
    wechat_keywords = ["微信", "表情包", "动图", "表情", "聊天", "社交"]
    is_wechat_usage = any(keyword in user_input_lower for keyword in wechat_keywords)
    
    email_keywords = ["邮件", "email", "附件", "发送"]
    is_email_usage = any(keyword in user_input_lower for keyword in email_keywords)
    
    web_keywords = ["网页", "网站", "博客", "论坛", "在线"]
    is_web_usage = any(keyword in user_input_lower for keyword in web_keywords)
    
    # 智能推荐逻辑
    if has_quality_demand or (not has_explicit_compression and not is_wechat_usage and not is_email_usage and not is_web_usage):
        # 用户要求高质量或没有明确压缩需求时，优先推荐最高质量
        suggestions.append({
            'name': '高清优化版',
            'description': '保持原视频质量，优化色彩和细节，适合专业用途和高质量需求',
            'params': {
                'fps': min(fps, 15),
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
    
    if is_wechat_usage:
        # 微信表情包专用优化
        suggestions.append({
            'name': '微信表情包版',
            'description': '专为微信动图表情包优化，文件大小控制在20MB以内，保持良好视觉效果',
            'params': {
                'fps': min(fps, 12),
                'quality': 85,
                'width': min(width, 320),
                'height': min(height, 240),
                'optimize': True
            },
            'size_constraint': {
                'operator': '<',
                'value': 20.0,
                'unit': 'MB',
                'enabled': True
            }
        })
    
    if is_email_usage:
        # 邮件附件优化
        suggestions.append({
            'name': '邮件附件版',
            'description': '适合邮件发送的优化版本，文件大小控制在5MB以内',
            'params': {
                'fps': min(fps, 10),
                'quality': 80,
                'width': min(width, 480),
                'height': min(height, 360),
                'optimize': True
            },
            'size_constraint': {
                'operator': '<',
                'value': 5.0,
                'unit': 'MB',
                'enabled': True
            }
        })
    
    if is_web_usage:
        # 网页使用优化
        suggestions.append({
            'name': '网页优化版',
            'description': '适合网页展示的优化版本，快速加载，文件大小控制在2MB以内',
            'params': {
                'fps': min(fps, 8),
                'quality': 75,
                'width': min(width, 400),
                'height': min(height, 300),
                'optimize': True
            },
            'size_constraint': {
                'operator': '<',
                'value': 2.0,
                'unit': 'MB',
                'enabled': True
            }
        })
    
    if has_explicit_compression:
        # 用户明确要求压缩
        suggestions.append({
            'name': '压缩优化版',
            'description': '大幅压缩文件大小，适合网络分享和存储空间有限的情况',
            'params': {
                'fps': 8,
                'quality': 70,
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
    
    # 基于视频特征的智能建议（仅在用户没有明确需求时提供）
    if not user_input or (not has_explicit_compression and not has_quality_demand and not is_wechat_usage and not is_email_usage and not is_web_usage):
        if duration > 30:  # 长视频
            suggestions.append({
                'name': '长视频优化版',
                'description': '针对长视频的特殊优化，平衡质量和文件大小',
                'params': {
                    'fps': 8,
                    'quality': 80,
                    'width': min(width, 720),
                    'height': min(height, 540),
                    'optimize': True
                },
                'size_constraint': {
                    'operator': '<',
                    'value': 8.0,
                    'unit': 'MB',
                    'enabled': True
                }
            })
        
        if file_size > 50 * 1024 * 1024:  # 大文件
            suggestions.append({
                'name': '大文件优化版',
                'description': '针对大文件的智能压缩，保持可接受的视觉效果',
                'params': {
                    'fps': 10,
                    'quality': 75,
                    'width': min(width, 480),
                    'height': min(height, 360),
                    'optimize': True
                },
                'size_constraint': {
                    'operator': '<',
                    'value': 3.0,
                    'unit': 'MB',
                    'enabled': True
                }
            })
    
    # 如果没有生成任何建议，提供默认的平衡版本
    if not suggestions:
        suggestions.append({
            'name': '平衡优化版',
            'description': '质量与大小的最佳平衡，适合一般用途',
            'params': {
                'fps': min(fps, 12),
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

def main():
    """主函数"""
    # 初始化会话状态
    init_session_state()
    
    # 页面头部
    st.markdown("""
    <div class="main-header">
        <h1>🎬 视频转GIF工具</h1>
        <p>智能视频转换，保持最佳质量 • 支持多种格式 • 自动优化</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 添加一些空间
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 添加一些说明
    st.info("💡 欢迎使用视频转GIF工具！上传视频后系统将自动分析并设置最优参数，支持多种视频格式转换")
    
    # 检查OpenCV是否可用
    try:
        import cv2
        opencv_available = True
    except ImportError:
        opencv_available = False
        st.error("❌ OpenCV未安装，请运行: pip install opencv-python")
        st.info("💡 安装方法：pip install opencv-python")
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
