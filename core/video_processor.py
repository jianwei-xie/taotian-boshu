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
    模拟一场2小时的淘宝直播（2024-06-15 20:00~22:00），涵盖所有话术类型

    产品与demo_data.csv一致：
    - 口红（均价¥156，直播间价¥89）
    - 连衣裙（均价¥293，直播间价¥199）
    - 电饭煲（均价¥441，直播间价¥299）

    共50个片段，覆盖0-7200秒，每段30-60秒
    """
    sample_texts = [
        # ===== 1. 开场互动 (0-5min, 0-300s): 4 segments =====
        (0, 45, "欢迎宝宝们来到直播间！今天是618年中大促专场，全场好货低至五折，关注主播不迷路，右上角点亮关注灯牌"),
        (45, 90, "新进来的宝宝点点关注，今天直播间福利超级多，三款爆款产品轮番上阵，不要走开哦，先点个关注"),
        (90, 150, "来我们先抽一波福袋，大家扣666参与抽奖，福袋里面有免单名额，马上开奖，赶紧参与"),
        (150, 210, "恭喜中奖的宝宝！没中奖的别急，后面下单的宝宝还有机会抽免单，福利不断"),

        # ===== 2. 产品介绍-口红 (5-15min, 300-900s): 4 segments =====
        (300, 360, "第一款给大家介绍的是这款丝绒雾面口红，六个色号可选，不管是日常通勤还是约会都能hold住"),
        (360, 420, "这款口红是丝绒质地的，上嘴非常顺滑，不拔干不起皮，持久度特别好，吃饭都不怎么掉色"),
        (420, 480, "大家看这个试色，番茄红真的绝了，黄皮白皮都能涂，素颜涂也完全没问题，超级显白"),
        (480, 540, "还有这个烂番茄色，今年最火的色号，涂上就是高级感满满，姐妹们一定要入"),

        # ===== 3. 价格福利-口红 (15-20min, 900-1200s): 3 segments =====
        (900, 960, "这款口红专柜价156块钱，今天直播间只要89！直接打五折都不到，这个价格真的太划算了"),
        (960, 1020, "89块钱到手一支丝绒口红，还送一个定制唇刷和卸妆棉，相当于买一送一"),
        (1020, 1080, "拍两支再减10块，到手168两支，算下来一支才84！姐妹们可以跟闺蜜拼单"),

        # ===== 4. 逼单催促-口红 (20-25min, 1200-1500s): 3 segments =====
        (1200, 1260, "口红库存只剩最后80支了，每个色号库存不多了，喜欢的宝宝赶紧下手"),
        (1260, 1320, "1号链接已经上了，拍下的宝宝扣已拍，我看看有多少人下单了，321上链接"),
        (1320, 1380, "番茄红已经快断货了，只剩最后15支，没抢到的宝宝赶紧切换色号，手慢无"),

        # ===== 5. 信任背书 (25-30min, 1500-1800s): 3 segments =====
        (1500, 1560, "这款口红是李佳琦直播间推荐过的，月销5万支，好评率98%，品质绝对有保障"),
        (1560, 1620, "我们是品牌旗舰店直发，每支口红都有防伪码可以查验，正品保证假一赔十"),
        (1620, 1680, "小红书上好多博主都在推这款，你们可以去搜搜看，口碑真的非常好，回购率超高"),

        # ===== 6. 产品介绍-连衣裙 (30-45min, 1800-2700s): 4 segments =====
        (1800, 1860, "第二款给大家带来的是这款法式碎花连衣裙，面料是雪纺加内衬的，穿着特别舒服"),
        (1860, 1920, "这个版型是V领收腰设计，拉长颈部线条，收腰显瘦，微胖的宝宝也能穿出好身材"),
        (1920, 1980, "裙摆是A字大裙摆，走起路来飘逸又好看，约会穿这套绝对回头率百分百"),
        (1980, 2040, "S到XXL码都有，80斤到140斤的宝宝都能穿，弹性面料不挑身材，大家放心拍"),

        # ===== 7. 价格福利-连衣裙 (45-55min, 2700-3300s): 3 segments =====
        (2700, 2760, "这款连衣裙专柜价293，今天直播间199包邮到家！这个价格买到法式连衣裙真的太值了"),
        (2760, 2820, "199还送一条 matching 的腰带和一个小披肩，相当于一整套搭配都给你配好了"),
        (2820, 2880, "买两件再减20，两件378，一件才189！姐妹们可以跟闺蜜一起拼，一人一件"),

        # ===== 8. 逼单催促-连衣裙 (55-65min, 3300-3900s): 3 segments =====
        (3300, 3360, "连衣裙库存告急，M码和L码快没了，其他码也只剩二三十件，要的宝宝抓紧"),
        (3360, 3420, "2号链接已经上了，拍下的宝宝扣已拍，我看看有多少人，尺码不多了别犹豫"),
        (3420, 3480, "马上要下架了，最后10件！3、2、1，没抢到的宝宝真的要等下次直播了"),

        # ===== 9. 痛点共鸣 (65-75min, 3900-4500s): 3 segments =====
        (3900, 3960, "是不是有很多姐妹夏天不知道穿什么，衣柜里一堆衣服但就是挑不出一件满意的"),
        (3960, 4020, "每次买口红不是太干就是太油，涂上嘴不到两小时就掉色了，真的很心烦"),
        (4020, 4080, "还有在家做饭，煮出来的饭不是夹生就是糊底，每次都要倒掉重煮，特别浪费"),

        # ===== 10. 产品介绍-电饭煲 (75-85min, 4500-5100s): 4 segments =====
        (4500, 4560, "第三款重磅产品来了，这款智能电饭煲，IH电磁加热技术，煮出来的饭粒粒分明"),
        (4560, 4620, "3L的黄金容量，一到四口人都够用，内胆是陶瓷不粘涂层，煮粥煮饭煮汤都行"),
        (4620, 4680, "一键操作特别简单，老人也能用，预约功能可以提前设好，下班回家就能吃上热饭"),
        (4680, 4740, "还有蛋糕功能、煲汤功能、煮粥功能，一锅多用，买一个顶好几个厨房电器"),

        # ===== 11. 价格福利-电饭煲 (85-95min, 5100-5700s): 3 segments =====
        (5100, 5160, "这款电饭煲专柜价441，今天直播间只要299！还送蒸笼和量杯饭勺一套"),
        (5160, 5220, "299买一个IH电磁加热的电饭煲，送蒸笼送量杯送饭勺，到手就是一整套"),
        (5220, 5280, "这个价格比618还便宜，全年最低价，错过今天真的就没有了"),

        # ===== 12. 逼单催促-电饭煲 (95-105min, 5700-6300s): 3 segments =====
        (5700, 5760, "电饭煲只剩最后30台了，亏本冲量，抢到就是赚到，3、2、1上链接"),
        (5760, 5820, "3号链接已经上了，拍下的宝宝扣想要，我看看有多少人，库存真的不多了"),
        (5820, 5880, "只剩最后5台了，抢完下架，这个亏本价不会再有了，要的宝宝抓紧"),

        # ===== 13. 使用教程 (105-110min, 6300-6600s): 2 segments =====
        (6300, 6360, "教大家怎么用电饭煲煮出完美的米饭，先用量杯量好米，淘洗两遍后加水到刻度线"),
        (6360, 6420, "然后按煮饭键就可以了，大概40分钟就好，煮出来的饭又香又软，比外面卖的好吃"),

        # ===== 14. 售后承诺 (110-115min, 6600-6900s): 2 segments =====
        (6600, 6660, "所有商品支持7天无理由退换货，运费险我全包了，大家放心拍，不满意直接退"),
        (6660, 6720, "质量问题包退包换，电饭煲保修一年，口红和连衣裙有质量问题随时联系客服"),

        # ===== 15. 结束互动 (115-120min, 6900-7200s): 2 segments =====
        (6900, 6960, "感谢宝宝们的支持，今天所有链接还在线，没抢到的宝宝继续拍，库存不多啦"),
        (6960, 7020, "明天同一时间不见不散，关注主播不错过每一场福利，点个关注点个赞，宝宝们晚安"),
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
