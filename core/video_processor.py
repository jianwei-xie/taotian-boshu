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
    模拟一场双11淘宝直播（2024-11-11 20:00~次日02:00），涵盖所有话术类型

    产品与demo_data.csv一致（10款双11爆款）：
    1. 花西子空气蜜粉饼 (~129)
    2. 完美日记唇釉#R09 (~59.9)
    3. 珀莱雅双抗精华液 (~239)
    4. 优衣库U系列连衣裙 (~199)
    5. 波司登轻薄羽绒服 (~499)
    6. 蕉内无痕内衣套装 (~169)
    7. 美的智能电饭煲 (~399)
    8. 小米空气净化器 (~899)
    9. 三只松鼠坚果礼盒 (~89.9)
    10. 认养一头牛纯牛奶 (~69.9)

    共86个片段，覆盖0-80分钟（4800秒），模拟真实双11直播的前~80分钟
    后120分钟无话术（主播休息、回答弹幕、自由互动等），符合真实直播节奏。

    设计原则（对齐淘天官方GMV贡献基准）：
    - 每个产品完整话术序列：产品介绍(90s) -> 信任背书(60s) -> 价格福利(60s)
      -> 逼单催促(30s) -> 痛点共鸣(30s) -> 使用教程(30s) -> 售后承诺(30s) -> 互动引导(90s)
    - 产品间间隔10-12分钟，末尾有3-5分钟无话术间隙
    - 订单分布在间隙中，DID模型会将订单归因到最后一个话术的处理窗口
    - 产品介绍是每个产品的第一个话术，其0-120min窗口覆盖整个产品周期+间隙
    - 标签分布：产品介绍~23%, 互动引导~19%, 价格福利~12%, 信任背书~12%,
      逼单催促~12%, 痛点共鸣~12%, 使用教程~12%, 售后承诺~12%
    - 双11特色语言：双11价、全年最低、11.11、狂欢价、过了今晚恢复原价

    时间线:
    - 0-5min:   开场互动+双11预热 (3 segments)
    - 5-12min:  产品1-花西子蜜粉饼 (8 segments)
    - 12-19min: 产品2-完美日记唇釉 (8 segments)
    - 19-26min: 产品3-珀莱雅精华液 (8 segments)
    - 26-33min: 产品4-优衣库连衣裙 (8 segments)
    - 33-40min: 产品5-波司登羽绒服 (8 segments)
    - 40-47min: 产品6-蕉内内衣 (8 segments)
    - 47-54min: 产品7-美的电饭煲 (8 segments)
    - 54-61min: 产品8-小米净化器 (8 segments)
    - 61-68min: 产品9-三只松鼠礼盒 (8 segments)
    - 68-75min: 产品10-认养一头牛牛奶 (8 segments)
    - 75-80min: 结束互动+感谢 (3 segments)
    - 80-240min: [无话术]
    """
    sample_texts = [
        # ===== 开场互动+双11预热 (0-5min, 0-300s): 3 segments =====
        (0, 90, "欢迎宝宝们来到直播间！今天是双11狂欢夜，全场好货狂欢特惠，关注主播不错过任何福利，新进来的宝宝点点关注"),
        (90, 180, "双11只有今天，过了今晚十二点活动就结束了，今天给大家准备了十款超级爆款，每一款都是11.11专属福利，千万不要走开"),
        (180, 300, "来，在的宝宝在评论区打个在，让我看看直播间有多少人，点赞到一万我们发一波双11专属福袋，里面有免单名额"),

        # ===== 产品1 - 花西子空气蜜粉饼 (5-12min, 300-720s): 8 segments =====
        # 产品介绍 (300-390s, 90s)
        (300, 390, "第一款给大家介绍的是花西子空气蜜粉饼，这款是今年特别受欢迎的单品，采用的是微米级粉体技术，上脸特别轻薄透气，完全不卡粉，控油持妆效果特别好，早上化完妆到晚上都不会脱妆"),
        # 信任背书 (390-450s, 60s)
        (390, 450, "花西子是国货美妆头部品牌，天猫旗舰店月销超过50万件，好评率高达98%，这款蜜粉饼是小红书博主都在推的爆款，每盒都有防伪码可以查验，正品保证假一赔十"),
        # 价格福利 (450-510s, 60s)
        (450, 510, "花西子蜜粉饼日常价199，今天双11直播间只要129！直接打六五折，这个价格只有双11才有，全年最低价，129到手还送一个定制粉扑和旅行装散粉"),
        # 逼单催促 (510-540s, 30s)
        (510, 540, "花西子蜜粉饼库存只剩最后50盒了，双11只有今天，过了今晚就没有了，1号链接已经上了，3、2、1上链接，手慢无"),
        # 痛点共鸣 (540-570s, 30s)
        (540, 570, "是不是有很多姐妹到了下午T区就出油脱妆，毛孔粗大卡粉卡到怀疑人生，每次补妆都特别麻烦"),
        # 使用教程 (570-600s, 30s)
        (570, 600, "教大家这款蜜粉饼的上妆手法，用粉扑轻轻按压而不是涂抹，从T区开始往外拍开，少量多次上出来的妆感特别高级"),
        # 售后承诺 (600-630s, 30s)
        (600, 630, "花西子支持7天无理由退换货，运费险我全包了，不满意随时退，质量问题包退包换，大家放心拍"),
        # 互动引导 (630-720s, 90s)
        (630, 720, "拍到的宝宝在评论区打个已拍，我看看有多少人抢到了，没抢到的宝宝打想要，新进来的宝宝点点关注不迷路，双11福利持续放送，福袋已经准备好了大家扣666参与"),

        # ===== 产品2 - 完美日记唇釉 (12-19min, 720-1140s): 8 segments =====
        # 产品介绍 (720-810s, 90s)
        (720, 810, "第二款给大家带来的是完美日记唇釉R09号色，这款是今年的网红色号，丝绒雾面质地，上嘴特别顺滑不拔干不起皮，R09这个色号是复古红棕色调，黄皮白皮都能涂，素颜涂也完全没问题"),
        # 信任背书 (810-870s, 60s)
        (810, 870, "完美日记是国货美妆第一品牌，天猫旗舰店月销20万支，好评率98%，这款唇釉是李佳琦推荐过的同款，每支都有防伪码可以查验，正品保证假一赔十"),
        # 价格福利 (870-930s, 60s)
        (870, 930, "完美日记唇釉专柜价89，今天双11直播间只要59块9！不到60块钱一支大牌质感的唇釉，59块9还送一个唇部去死皮膏和润唇膏，拍三支再减15"),
        # 逼单催促 (930-960s, 30s)
        (930, 960, "R09号色库存告急只剩最后30支了，双11活动只有今天明天就结束了，2号链接已经上了，拍下的宝宝在评论区打个已拍，抓紧"),
        # 痛点共鸣 (960-990s, 30s)
        (960, 990, "是不是有很多姐妹买口红总是踩雷，不是太干就是掉色太快，涂上半小时就没了，每次买一堆真正爱用的没几支"),
        # 使用教程 (990-1020s, 30s)
        (990, 1020, "教大家唇釉的正确涂法，先涂下唇中间然后往两边晕开，上唇从嘴角往中间涂，最后用手指轻轻拍一下让边缘更自然"),
        # 售后承诺 (1020-1050s, 30s)
        (1020, 1050, "完美日记支持7天无理由退换货，运费险我全包了，不满意随时退，质量问题包退包换，双11购物无忧大家放心拍"),
        # 互动引导 (1050-1140s, 90s)
        (1050, 1140, "用过的宝宝在评论区打个好用，让我看看有多少回购的老粉，新进来的宝宝点点关注，双11福袋马上开奖在评论区打666参与，后面还有珀莱雅精华液等重磅好货不要走开"),

        # ===== 产品3 - 珀莱雅双抗精华液 (19-26min, 1140-1560s): 8 segments =====
        # 产品介绍 (1140-1230s, 90s)
        (1140, 1230, "第三款重磅产品来了珀莱雅双抗精华液，这款是抗氧抗糖双效合一的明星产品，里面添加了麦角硫因和虾青素两大核心成分，质地清爽好吸收不油腻不粘腻，坚持两周就能看到肤色明显提亮"),
        # 信任背书 (1230-1290s, 60s)
        (1230, 1290, "珀莱雅是国货护肤头部品牌，天猫旗舰店月销超过30万瓶，这款双抗精华是小红书口碑排名第一的抗老精华，很多皮肤科医生都在推荐，正品保证假一赔十"),
        # 价格福利 (1290-1350s, 60s)
        (1290, 1350, "珀莱雅双抗精华液日常价339，今天双11直播间只要239！直降100块，这个价格比旗舰店双11预售还便宜，239还送同款小样两瓶和一片面膜，买两瓶再减30"),
        # 逼单催促 (1350-1380s, 30s)
        (1350, 1380, "珀莱雅精华液双11库存紧张品牌方只给了100瓶的额度，卖完就没了，3号链接已经上了，3、2、1上链接，犹豫一下就没了"),
        # 痛点共鸣 (1380-1410s, 30s)
        (1380, 1410, "是不是有很多姐妹过了25岁皮肤就开始暗沉发黄，细纹越来越多，用了很多护肤品但效果都不明显，真的很心烦"),
        # 使用教程 (1410-1440s, 30s)
        (1410, 1440, "这款精华液早晚各用一次，洗完脸涂完爽肤水后取两滴在手心搓热按压上脸，再涂面霜锁住精华，坚持用两周就能看到效果"),
        # 售后承诺 (1440-1470s, 30s)
        (1440, 1470, "珀莱雅支持7天无理由退换货，运费险我全包了，不满意随时退，质量问题包退包换，品牌直发正品保证大家放心拍"),
        # 互动引导 (1470-1560s, 90s)
        (1470, 1560, "在的宝宝扣在让我看看直播间还有多少人，双11狂欢继续后面还有优衣库连衣裙和波司登羽绒服等重磅好货不要走开，福袋已经准备好了大家扣666参与"),

        # ===== 产品4 - 优衣库U系列连衣裙 (26-33min, 1560-1980s): 8 segments =====
        # 产品介绍 (1560-1650s, 90s)
        (1560, 1650, "第四款给大家带来的是优衣库U系列连衣裙，这款是今年秋冬的爆款，面料是高支棉加少量氨纶穿着特别舒服有弹性，这个版型是宽松H型设计不挑身材不挑身高，80斤到140斤都能穿"),
        # 信任背书 (1650-1710s, 60s)
        (1650, 1710, "优衣库是全球知名快时尚品牌，U系列是优衣库最高端的产品线，这款连衣裙是今年秋冬天猫销量排名前三的爆款，很多时尚博主都在推荐同款"),
        # 价格福利 (1710-1770s, 60s)
        (1710, 1770, "优衣库U系列连衣裙专柜价249，今天双11直播间只要199！直降50这个价格买到U系列真的太值了，199包邮到家，买两件减30两件368一件才184"),
        # 逼单催促 (1770-1800s, 30s)
        (1770, 1800, "U系列连衣裙M码和L码快没了其他码也只剩二三十件，双11活动只有今天，4号链接已经上了，拍下的宝宝在评论区打个已拍，库存清空不补"),
        # 痛点共鸣 (1800-1830s, 30s)
        (1800, 1830, "是不是有很多姐妹秋冬不知道穿什么，衣柜里一堆衣服但就是挑不出一件满意的，每天早上都在纠结买了很多又不满意"),
        # 使用教程 (1830-1860s, 30s)
        (1830, 1860, "这款连衣裙搭配方法特别多，单穿配小白鞋就是休闲风，外面搭一件大衣就是通勤风，配靴子就是法式优雅风，一衣多穿特别实用"),
        # 售后承诺 (1860-1890s, 30s)
        (1860, 1890, "优衣库支持7天无理由退换货，运费险我全包了，不满意随时退，质量问题包退包换，品牌直发正品保证大家放心拍"),
        # 互动引导 (1890-1980s, 90s)
        (1890, 1980, "用过的宝宝在评论区打个好用让我看看有多少回购的老粉，新进来的宝宝点点关注，双11福袋马上开奖在评论区打666参与，后面还有波司登羽绒服等重磅好货不要走开"),

        # ===== 产品5 - 波司登轻薄羽绒服 (33-40min, 1980-2400s): 8 segments =====
        # 产品介绍 (1980-2070s, 90s)
        (1980, 2070, "第五款重磅推荐波司登轻薄羽绒服，今年双11的王炸单品，采用的是90%白鹅绒填充蓬松度700+保暖又轻薄不臃肿，这款是短款设计小个子女生也能穿出好比例，面料防风防水秋冬通勤穿特别合适"),
        # 信任背书 (2070-2130s, 60s)
        (2070, 2130, "波司登是羽绒服行业第一品牌，连续28年全国销量领先，这款羽绒服获得过国家专利认证，登上过央视广告，是很多明星冬天出街的同款，正品保证假一赔十"),
        # 价格福利 (2130-2190s, 60s)
        (2130, 2190, "波司登羽绒服日常价799，今天双11直播间只要499！直降300块不到500块钱买波司登正品羽绒服，499还送收纳袋和同款围巾，买两件再减50，双11狂欢价全年最低"),
        # 逼单催促 (2190-2220s, 30s)
        (2190, 2220, "波司登羽绒服只剩最后40件了品牌方限量供货卖完就下架，5号链接已经上了，3、2、1上链接，抢到就是赚到"),
        # 痛点共鸣 (2220-2250s, 30s)
        (2220, 2250, "是不是有很多姐妹冬天穿羽绒服特别臃肿像个球一样，不穿又冷穿多了又显胖显矮，真的特别纠结"),
        # 使用教程 (2250-2280s, 30s)
        (2250, 2280, "教大家羽绒服的收纳方法，换季的时候用配套的收纳袋卷起来放不要挂，这样鹅绒不会结块明年拿出来跟新的一样蓬松"),
        # 售后承诺 (2280-2310s, 30s)
        (2280, 2310, "波司登支持7天无理由退换货，运费险我全包了，不满意随时退，质量问题包退包换，品牌直发正品保证大家放心拍"),
        # 互动引导 (2310-2400s, 90s)
        (2310, 2400, "在的宝宝扣在让我看看直播间还有多少人，双11狂欢继续后面还有蕉内内衣和美的电饭煲等重磅好货不要走开，福袋已经准备好了大家扣666参与"),

        # ===== 产品6 - 蕉内无痕内衣套装 (40-47min, 2400-2820s): 8 segments =====
        # 产品介绍 (2400-2490s, 90s)
        (2400, 2490, "第六款给大家带来的是蕉内无痕内衣套装，这款是蕉内的王牌产品，采用一体成型无痕工艺穿紧身衣服也看不出内衣痕迹，面料是莫代尔加氨纶的混纺面料亲肤透气不闷热弹性特别好不勒肉"),
        # 信任背书 (2490-2550s, 60s)
        (2490, 2550, "蕉内是新锐内衣头部品牌，天猫旗舰店内衣类目月销排名前三，好评率97%以上，这款无痕内衣套装是蕉内销量最高的王牌产品，很多时尚博主都在推荐"),
        # 价格福利 (2550-2610s, 60s)
        (2550, 2610, "蕉内无痕内衣套装日常价229，今天双11直播间只要169！直降60块一套两件只要169，169到手两件套文胸加内裤，买两套再减20，双11价全年最低"),
        # 逼单催促 (2610-2640s, 30s)
        (2610, 2640, "蕉内内衣套装库存告急75B和80A快没货了，双11只有今天这个活动，6号链接已经上了，拍下的宝宝在评论区打个已拍，犹豫就没了"),
        # 痛点共鸣 (2640-2670s, 30s)
        (2640, 2670, "是不是有很多姐妹穿内衣总是勒痕很明显，穿紧身衣服特别尴尬，换了好几个品牌都不舒服，真的特别烦"),
        # 使用教程 (2670-2700s, 30s)
        (2670, 2700, "这款内衣手洗机洗都可以，建议用内衣洗衣液轻柔模式，洗完自然晾干不要暴晒，莫代尔面料洗了也不变形"),
        # 售后承诺 (2700-2730s, 30s)
        (2700, 2730, "蕉内支持7天无理由退换货，运费险我全包了，不满意随时退，质量问题包退包换，品牌直发正品保证大家放心拍"),
        # 互动引导 (2730-2820s, 90s)
        (2730, 2820, "在的宝宝扣在让我看看直播间还有多少人，双11狂欢继续后面还有美的电饭煲和小米净化器等重磅好货不要走开，福袋已经准备好了大家扣666参与"),

        # ===== 产品7 - 美的智能电饭煲 (47-54min, 2820-3240s): 8 segments =====
        # 产品介绍 (2820-2910s, 90s)
        (2820, 2910, "第七款重磅产品美的智能电饭煲，IH电磁环绕加热技术煮出来的米饭粒粒分明口感Q弹比普通电饭煲好吃太多了，3L的黄金容量一到四口人都够用，内胆是陶瓷不粘涂层煮粥煮饭煲汤一锅搞定"),
        # 信任背书 (2910-2970s, 60s)
        (2910, 2970, "美的是家电行业第一品牌，全国联保一年有任何质量问题直接联系客服包退包换，我们是品牌旗舰店直发，正品保证假一赔十，大家放心拍"),
        # 价格福利 (2970-3030s, 60s)
        (2970, 3030, "美的智能电饭煲日常价599，今天双11直播间只要399！直降200块IH电磁加热电饭煲到手超值优惠，399还送蒸笼和量杯饭勺一套，双11狂欢价全年最低"),
        # 逼单催促 (3030-3060s, 30s)
        (3030, 3060, "美的电饭煲只剩最后25台了亏本冲量双11专属活动，7号链接已经上了，3、2、1上链接，要的宝宝赶快下手，抢完下架"),
        # 痛点共鸣 (3060-3090s, 30s)
        (3060, 3090, "是不是有很多姐妹煮饭不是太硬就是夹生，每次煮出来都不好吃，真的很浪费食材特别心烦"),
        # 使用教程 (3090-3120s, 30s)
        (3090, 3120, "教大家这款电饭煲的使用方法，用量杯量好米淘洗两遍加水到刻度线按煮饭键就好了，还可以用预约功能早上出门前放好米晚上回来就能吃"),
        # 售后承诺 (3120-3150s, 30s)
        (3120, 3150, "美的家电全国联保一年，有任何质量问题直接联系客服包退包换，品牌旗舰店直发正品保证假一赔十，大家放心拍"),
        # 互动引导 (3150-3240s, 90s)
        (3150, 3240, "在的宝宝扣在让我看看直播间还有多少人，双11狂欢继续后面还有小米净化器和三只松鼠礼盒等重磅好货不要走开，福袋已经准备好了大家扣666参与"),

        # ===== 产品8 - 小米空气净化器 (54-61min, 3240-3660s): 8 segments =====
        # 产品介绍 (3240-3330s, 90s)
        (3240, 3330, "第八款给大家带来的是小米空气净化器，这款是小米的旗舰款，采用三层复合滤芯HEPA滤网能有效过滤99.97%的PM2.5，适用面积30到60平米卧室客厅都能用，运行噪音低至32分贝晚上睡觉开着也完全不会被打扰"),
        # 信任背书 (3330-3390s, 60s)
        (3330, 3390, "小米是科技家电头部品牌，这款净化器是小米旗舰店销量排名前三的爆款，好评率96%以上，很多数码博主都在推荐，支持专柜验货正品保证假一赔十"),
        # 价格福利 (3390-3450s, 60s)
        (3390, 3450, "小米空气净化器日常价1299，今天双11直播间只要899！直降400块到手超值优惠，899还送两个替换滤芯相当于一年不用再买滤芯了，全年最低价"),
        # 逼单催促 (3450-3480s, 30s)
        (3450, 3480, "小米净化器库存只剩最后15台了品牌方限量供应双11专属活动，8号链接已经上了，3、2、1上链接，抢到就是赚到"),
        # 痛点共鸣 (3480-3510s, 30s)
        (3480, 3510, "是不是有很多姐妹家里空气质量不好，冬天开暖气更是闷得慌，老人小孩经常咳嗽真的特别担心"),
        # 使用教程 (3510-3540s, 30s)
        (3510, 3540, "教大家净化器滤芯的更换方法，打开前面板取出旧滤芯换上新滤芯卡紧就行，APP会自动提醒更换时间特别方便"),
        # 售后承诺 (3540-3570s, 30s)
        (3540, 3570, "小米家电全国联保一年，有任何质量问题直接联系客服包退包换，品牌旗舰店直发正品保证假一赔十，大家放心拍"),
        # 互动引导 (3570-3660s, 90s)
        (3570, 3660, "在的宝宝扣在让我看看直播间还有多少人，双11狂欢继续后面还有三只松鼠礼盒和认养一头牛牛奶等重磅好货不要走开，福袋已经准备好了大家扣666参与"),

        # ===== 产品9 - 三只松鼠坚果礼盒 (61-68min, 3660-4080s): 8 segments =====
        # 产品介绍 (3660-3750s, 90s)
        (3660, 3750, "第九款给大家带来的是三只松鼠坚果礼盒，这款是双11的爆款礼盒，里面有开心果、巴旦木、腰果、夏威夷果等八种坚果采用当季新货精选，每一颗都是个大饱满壳薄肉厚独立小包装干净卫生"),
        # 信任背书 (3750-3810s, 60s)
        (3750, 3810, "三只松鼠是坚果零食行业第一品牌，天猫旗舰店零食类目月销排名第一，好评率98%以上，这款礼盒是双11连续5年销量冠军，品牌直发正品保证"),
        # 价格福利 (3810-3870s, 60s)
        (3810, 3870, "三只松鼠坚果礼盒日常价139，今天双11直播间只要89块9！不到90块钱一整箱八种坚果，89块9还送每日坚果和牛肉干，拍两箱再减15，双11狂欢价太划算了"),
        # 逼单催促 (3870-3900s, 30s)
        (3870, 3900, "三只松鼠礼盒双11库存紧张品牌方只给了80箱的额度，9号链接已经上了，拍下的宝宝在评论区打个已拍，卖完就下架不补了"),
        # 痛点共鸣 (3900-3930s, 30s)
        (3900, 3930, "是不是有很多姐妹冬天在家追剧嘴巴寂寞，想吃零食又怕胖，每次都控制不住自己吃完又后悔，真的特别纠结"),
        # 使用教程 (3930-3960s, 30s)
        (3930, 3960, "这款坚果礼盒开袋即食特别方便，追剧办公来一包，也可以搭配酸奶和麦片做早餐，每天一小包营养均衡"),
        # 售后承诺 (3960-3990s, 30s)
        (3960, 3990, "三只松鼠支持7天无理由退换货，运费险我全包了，不满意随时退，质量问题包退包换，品牌直发正品保证大家放心拍"),
        # 互动引导 (3990-4080s, 90s)
        (3990, 4080, "在的宝宝扣在让我看看直播间还有多少人，双11狂欢继续后面还有认养一头牛牛奶最后一件爆款不要走开，福袋已经准备好了大家扣666参与"),

        # ===== 产品10 - 认养一头牛纯牛奶 (68-75min, 4080-4500s): 8 segments =====
        # 产品介绍 (4080-4170s, 90s)
        (4080, 4170, "第十款给大家带来的是认养一头牛纯牛奶，这款是双11必囤的日常好物，采用的是澳洲荷斯坦奶牛产的优质鲜奶蛋白质含量3.6克，口感特别香浓醇厚不是那种稀稀的水牛奶，老人小孩都爱喝"),
        # 信任背书 (4170-4230s, 60s)
        (4170, 4230, "认养一头牛是鲜奶行业头部品牌，天猫旗舰店鲜奶类目月销排名前三，好评率99%以上，品牌有自己的牧场从源头保证品质，每箱都有溯源码可以查验"),
        # 价格福利 (4230-4290s, 60s)
        (4230, 4290, "认养一头牛纯牛奶日常价89，今天双11直播间只要69块9！一箱24盒不到70块钱算下来一盒才两块九，拍三箱再减15，双11囤奶最佳时机太划算了"),
        # 逼单催促 (4290-4320s, 30s)
        (4290, 4320, "认养一头牛牛奶双11库存告急品牌方限量500箱，10号链接已经上了，3、2、1上链接，囤奶的宝宝赶紧拍，手慢无"),
        # 痛点共鸣 (4320-4350s, 30s)
        (4320, 4350, "是不是有很多宝妈担心孩子喝的牛奶不够好，市面上的牛奶品质参差不齐不知道怎么选，真的特别头疼"),
        # 使用教程 (4350-4380s, 30s)
        (4350, 4380, "这款牛奶开箱后建议冷藏保存，早上搭配面包和鸡蛋就是营养早餐，晚上睡前喝一杯温牛奶还有助睡眠"),
        # 售后承诺 (4380-4410s, 30s)
        (4380, 4410, "认养一头牛支持7天无理由退换货，运费险我全包了，不满意随时退，质量问题包退包换，品牌直发正品保证大家放心拍"),
        # 互动引导 (4410-4500s, 90s)
        (4410, 4500, "在的宝宝扣在让我看看直播间还有多少人，双11所有链接还在线喜欢的宝宝继续下单，福袋已经准备好了大家扣666参与"),

        # ===== 结束互动+感谢 (75-80min, 4500-4800s): 3 segments =====
        (4500, 4590, "感谢宝宝们的支持今天双11所有链接还在线喜欢的宝宝继续下单，双11狂欢夜快乐关注主播不错过每一场福利"),
        (4590, 4680, "点个关注点个赞我们下次直播不见不散，双11福利持续放送，明天同一时间继续给大家带来更多好货"),
        (4680, 4800, "晚安宝宝们双11快乐，所有订单今天全部发货，有问题随时联系客服我们24小时在线"),
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
