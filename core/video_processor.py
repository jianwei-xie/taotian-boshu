"""
视频处理模块

解决的问题：商家上传视频后，如何快速准确地提取话术

核心功能：
1. 用Whisper把视频语音转成文字（base模型，速度优先）
2. 按照直播说话节奏把话术切分成10-30秒一段
3. 自动过滤"嗯啊哦"这些语气词
4. 输出带时间戳的话术片段，方便后续归因

设计原则：
- 速度比准确率重要，错几个字不影响分类
- 对脏数据极度宽容，商家上传什么都不能崩
- 所有耗时操作都要有进度条
"""

import re
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import warnings

import numpy as np

# 忽略一些不必要的警告
warnings.filterwarnings('ignore')


@dataclass
class ScriptSegment:
    """
    话术片段数据类
    
    为什么用dataclass：比字典更清晰，字段有类型提示，不容易写错
    """
    start_time: float      # 开始时间（秒）
    end_time: float        # 结束时间（秒）
    text: str              # 话术文本
    segment_id: int        # 片段ID，方便后续关联
    
    @property
    def duration(self) -> float:
        """片段时长（秒）"""
        return self.end_time - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        """转成字典，方便序列化"""
        return {
            'segment_id': self.segment_id,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.duration,
            'text': self.text
        }


class VideoProcessor:
    """
    视频处理器
    
    这个类解决什么问题：
    商家上传一个直播视频，我要把它变成带时间戳的话术片段列表
    """
    
    def __init__(
        self,
        whisper_model: str = "base",
        segment_min_duration: float = 10.0,
        segment_max_duration: float = 30.0,
        pause_threshold: float = 1.5,
        filler_words: Optional[List[str]] = None
    ):
        """
        初始化视频处理器
        
        参数都是可选的，不给就用默认值，商家不需要知道这些
        """
        self.whisper_model_name = whisper_model
        self.segment_min_duration = segment_min_duration
        self.segment_max_duration = segment_max_duration
        self.pause_threshold = pause_threshold
        self.filler_words = filler_words or ['嗯', '啊', '哦', '呃', '那个', '这个', '就是', '然后', '对吧', '是吧']
        
        # Whisper模型延迟加载，用的时候再加载，节省内存
        self._whisper_model = None
        
    def _load_whisper_model(self):
        """
        延迟加载Whisper模型
        
        为什么延迟加载：
        1. 不是每次都会用到视频处理（比如商家只上传了订单数据）
        2. 模型加载要几秒钟，放在构造函数里会让初始化变慢
        3. 节省内存，不用的时候不占资源
        """
        if self._whisper_model is None:
            try:
                import whisper
                print(f"正在加载Whisper模型({self.whisper_model_name})，请稍候...")
                self._whisper_model = whisper.load_model(self.whisper_model_name)
                print("模型加载完成！")
            except Exception as e:
                raise RuntimeError(f"加载Whisper模型失败：{str(e)}。请确保已安装openai-whisper：pip install openai-whisper")
        return self._whisper_model
    
    def extract_audio_from_video(self, video_path: str, progress_callback=None) -> str:
        """
        从视频中提取音频
        
        解决的问题：Whisper需要音频文件，但商家上传的是视频
        
        参数：
            video_path: 视频文件路径
            progress_callback: 进度回调函数，给界面用
            
        返回：
            音频文件路径（临时文件）
        """
        if progress_callback:
            progress_callback(0.1, "正在提取音频...")
        
        try:
            from moviepy.editor import VideoFileClip
        except ImportError:
            raise RuntimeError("请安装moviepy：pip install moviepy")
        
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"找不到视频文件：{video_path}")
        
        # 生成临时音频文件路径
        audio_path = video_path.parent / f"{video_path.stem}_temp_audio.wav"
        
        try:
            # 提取音频
            video = VideoFileClip(str(video_path))
            
            if progress_callback:
                progress_callback(0.2, f"视频时长：{video.duration/60:.1f}分钟，正在提取音频...")
            
            # 只提取音频，降低采样率加快处理速度
            video.audio.write_audiofile(
                str(audio_path),
                fps=16000,  # Whisper推荐16kHz
                nbytes=2,
                codec='pcm_s16le',
                verbose=False,
                logger=None  # 不输出moviepy的日志
            )
            video.close()
            
            if progress_callback:
                progress_callback(0.3, "音频提取完成！")
            
            return str(audio_path)
            
        except Exception as e:
            raise RuntimeError(f"提取音频失败：{str(e)}")
    
    def transcribe_audio(self, audio_path: str, progress_callback=None) -> Dict[str, Any]:
        """
        用Whisper把音频转成文字
        
        解决的问题：把语音变成带时间戳的文本
        
        参数：
            audio_path: 音频文件路径
            progress_callback: 进度回调函数
            
        返回：
            Whisper的转录结果，包含segments（带时间戳的片段列表）
        """
        if progress_callback:
            progress_callback(0.35, "正在加载语音识别模型...")
        
        model = self._load_whisper_model()
        
        if progress_callback:
            progress_callback(0.4, "正在识别语音，这可能需要几分钟...")
        
        try:
            # 使用Whisper转录
            # language=zh确保识别中文
            # task=transcribe是转录，不是翻译
            result = model.transcribe(
                audio_path,
                language="zh",
                task="transcribe",
                verbose=False  # 不输出进度
            )
            
            if progress_callback:
                progress_callback(0.6, f"语音识别完成！共识别出{len(result['segments'])}个片段")
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"语音识别失败：{str(e)}")
    
    def remove_filler_words(self, text: str) -> str:
        """
        去除语气词
        
        解决的问题：直播里"嗯啊哦"太多了，去掉后更干净
        
        参数：
            text: 原始文本
            
        返回：
            清理后的文本
        """
        # 先把连续的语气词合并
        cleaned = text
        
        # 去除每个语气词
        for word in self.filler_words:
            # 匹配语气词，前面后面可以有标点或空格
            pattern = f"[，,\s]*{word}[，,\s]*"
            cleaned = re.sub(pattern, "，", cleaned)
        
        # 清理多余的标点
        cleaned = re.sub(r"，+", "，", cleaned)
        cleaned = re.sub(r"，\s*，", "，", cleaned)
        cleaned = re.sub(r"^，|，$", "", cleaned)
        
        # 清理多余空格
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        
        return cleaned
    
    def merge_segments(
        self,
        segments: List[Dict[str, Any]],
        progress_callback=None
    ) -> List[ScriptSegment]:
        """
        合并Whisper的片段，按照直播节奏切成10-30秒一段
        
        解决的问题：
        Whisper切得太碎了（通常3-5秒一句），不符合直播的话术节奏
        直播里一个完整的话术通常要10-30秒
        
        合并策略：
        1. 先按停顿切分（停顿超过1.5秒认为是一个话术结束）
        2. 如果切出来太短（<10秒），就和下一段合并
        3. 如果切出来太长（>30秒），就在句子边界处切开
        
        参数：
            segments: Whisper的原始片段列表
            progress_callback: 进度回调函数
            
        返回：
            合并后的话术片段列表
        """
        if progress_callback:
            progress_callback(0.65, "正在合并话术片段...")
        
        if not segments:
            return []
        
        # 第一步：按停顿阈值初步分组
        raw_groups = []
        current_group = [segments[0]]
        
        for i in range(1, len(segments)):
            prev_segment = segments[i-1]
            curr_segment = segments[i]
            
            # 计算两段之间的停顿时间
            pause_duration = curr_segment['start'] - prev_segment['end']
            
            if pause_duration > self.pause_threshold:
                # 停顿太长，开始新的一组
                raw_groups.append(current_group)
                current_group = [curr_segment]
            else:
                current_group.append(curr_segment)
        
        # 别忘了最后一组
        if current_group:
            raw_groups.append(current_group)
        
        # 第二步：调整每组时长，确保在10-30秒之间
        final_segments = []
        segment_id = 0
        
        i = 0
        while i < len(raw_groups):
            group = raw_groups[i]
            group_start = group[0]['start']
            group_end = group[-1]['end']
            group_duration = group_end - group_start
            
            # 合并文本
            group_text = "".join([s['text'] for s in group])
            
            if group_duration < self.segment_min_duration:
                # 太短了，尝试和下一段合并
                if i + 1 < len(raw_groups):
                    next_group = raw_groups[i + 1]
                    merged_duration = next_group[-1]['end'] - group_start
                    
                    if merged_duration <= self.segment_max_duration:
                        # 合并后不会太长，就合并
                        group.extend(next_group)
                        group_text = "".join([s['text'] for s in group])
                        group_end = next_group[-1]['end']
                        group_duration = merged_duration
                        i += 1  # 跳过下一段
                    # 合并后太长，就不合并，保留短的
                # 已经是最后一段了，保留
                
            elif group_duration > self.segment_max_duration:
                # 太长了，需要在句子边界处切开
                # 简单策略：按30秒切分
                sub_segments = []
                sub_start = group_start
                sub_text_parts = []
                
                for s in group:
                    if s['end'] - sub_start > self.segment_max_duration and sub_text_parts:
                        # 超过30秒了，先保存这一段
                        sub_text = "".join(sub_text_parts)
                        sub_text = self.remove_filler_words(sub_text)
                        
                        if sub_text and len(sub_text) > 5:  # 至少5个字符
                            final_segments.append(ScriptSegment(
                                start_time=sub_start,
                                end_time=s['start'],
                                text=sub_text,
                                segment_id=segment_id
                            ))
                            segment_id += 1
                        
                        sub_start = s['start']
                        sub_text_parts = [s['text']]
                    else:
                        sub_text_parts.append(s['text'])
                
                # 处理最后一段
                if sub_text_parts:
                    sub_text = "".join(sub_text_parts)
                    sub_text = self.remove_filler_words(sub_text)
                    
                    if sub_text and len(sub_text) > 5:
                        final_segments.append(ScriptSegment(
                            start_time=sub_start,
                            end_time=group_end,
                            text=sub_text,
                            segment_id=segment_id
                        ))
                        segment_id += 1
                
                i += 1
                continue
            
            # 处理正常长度的段落
            group_text = self.remove_filler_words(group_text)
            
            # 只保留有意义的话术（至少5个字符）
            if group_text and len(group_text) > 5:
                final_segments.append(ScriptSegment(
                    start_time=group_start,
                    end_time=group_end,
                    text=group_text,
                    segment_id=segment_id
                ))
                segment_id += 1
            
            i += 1
        
        if progress_callback:
            progress_callback(0.8, f"话术合并完成！共生成{len(final_segments)}个话术片段")
        
        return final_segments
    
    def process_video(
        self,
        video_path: str,
        progress_callback=None
    ) -> List[ScriptSegment]:
        """
        处理视频的主函数：提取音频 -> 语音识别 -> 话术分割
        
        这是商家调用的主要接口，一步完成所有操作
        
        参数：
            video_path: 视频文件路径
            progress_callback: 进度回调函数，接收(progress, message)两个参数
                             progress是0-1之间的浮点数
                             message是进度描述
        
        返回：
            话术片段列表
            
        异常：
            各种错误都会包装成RuntimeError，附带友好的错误信息
        """
        video_path = Path(video_path)
        
        # 检查文件是否存在
        if not video_path.exists():
            raise RuntimeError(f"您上传的视频文件不存在，请重新上传。路径：{video_path}")
        
        # 检查文件格式
        supported_formats = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
        if video_path.suffix.lower() not in supported_formats:
            raise RuntimeError(
                f"不支持的文件格式：{video_path.suffix}。"
                f"请上传以下格式的视频：{', '.join(supported_formats)}"
            )
        
        audio_path = None
        
        try:
            # 第一步：提取音频
            audio_path = self.extract_audio_from_video(video_path, progress_callback)
            
            # 第二步：语音识别
            transcription_result = self.transcribe_audio(audio_path, progress_callback)
            
            # 第三步：话术分割
            segments = self.merge_segments(
                transcription_result.get('segments', []),
                progress_callback
            )
            
            if progress_callback:
                progress_callback(1.0, f"视频处理完成！共提取{len(segments)}段话术")
            
            return segments
            
        except Exception as e:
            # 统一错误处理，给商家友好的提示
            error_msg = str(e)
            if "No such file" in error_msg:
                raise RuntimeError("视频文件读取失败，请检查文件是否损坏或重新上传")
            elif "codec" in error_msg.lower():
                raise RuntimeError("视频编码不支持，请尝试转换为MP4格式后重新上传")
            else:
                raise RuntimeError(f"视频处理出错：{error_msg}")
        
        finally:
            # 清理临时音频文件
            if audio_path and Path(audio_path).exists():
                try:
                    Path(audio_path).unlink()
                except:
                    pass  # 清理失败也没关系
    
    def process_audio_only(
        self,
        audio_path: str,
        progress_callback=None
    ) -> List[ScriptSegment]:
        """
        只处理音频文件（如果商家直接上传了音频）
        
        参数和返回值同process_video
        """
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            raise RuntimeError(f"您上传的音频文件不存在，请重新上传。路径：{audio_path}")
        
        try:
            # 语音识别
            transcription_result = self.transcribe_audio(str(audio_path), progress_callback)
            
            # 话术分割
            segments = self.merge_segments(
                transcription_result.get('segments', []),
                progress_callback
            )
            
            if progress_callback:
                progress_callback(1.0, f"音频处理完成！共提取{len(segments)}段话术")
            
            return segments
            
        except Exception as e:
            raise RuntimeError(f"音频处理出错：{str(e)}")


def create_sample_script_segments() -> List[ScriptSegment]:
    """
    创建示例话术片段，用于测试
    
    当商家没有上传视频时，可以用这个看效果
    """
    sample_texts = [
        (0, 15, "姐妹们好，欢迎来到我们的直播间，今天给大家带来的是一款超级好用的面膜"),
        (15, 35, "这款面膜主打补水保湿，里面添加了玻尿酸成分，敷完皮肤水水嫩嫩的"),
        (35, 50, "是不是有很多姐妹冬天皮肤特别干，化妆卡粉卡到怀疑人生"),
        (50, 70, "今天直播间拍下立减50，到手只要99，还送价值69的精华小样"),
        (70, 85, "库存只剩最后20单了，想要的姐妹赶紧拍，1号链接直接下单"),
        (85, 105, "我们是天猫旗舰店，正品保证，不满意7天无理由退换货"),
        (105, 120, "这个面膜敷15分钟就够了，不要太久，一周用2-3次效果最好"),
        (120, 140, "用过的姐妹在评论区扣个好用，让我看看有多少人回购了"),
        (140, 155, "还有最后3分钟，没付款的姐妹抓紧了，活动结束恢复原价"),
        (155, 175, "今天下单送运费险，不满意随时退，没有任何风险"),
    ]
    
    segments = []
    for i, (start, end, text) in enumerate(sample_texts):
        segments.append(ScriptSegment(
            start_time=start,
            end_time=end,
            text=text,
            segment_id=i
        ))
    
    return segments


# 简单的测试代码
if __name__ == "__main__":
    # 测试话术清理功能
    processor = VideoProcessor()
    
    test_text = "嗯，这个面膜啊，就是嗯，那个补水保湿的效果特别好，对吧"
    cleaned = processor.remove_filler_words(test_text)
    print(f"原文：{test_text}")
    print(f"清理后：{cleaned}")
    
    # 测试示例数据
    print("\n示例话术片段：")
    samples = create_sample_script_segments()
    for seg in samples[:3]:
        print(f"[{seg.start_time:.0f}s-{seg.end_time:.0f}s] {seg.text[:30]}...")
