"""
淘天播术-电商直播话术军师 - 核心模块

包含所有核心分析功能：
- video_processor: 视频处理（语音转文字、话术分割）
- data_aligner: 数据对齐（订单数据清洗、时间对齐）
- script_classifier: 话术分类（BERT分类、标签识别）
- did_attribution: DID归因（话术效果分析）
- optimizer: 话术优化（贝叶斯优化）
"""

from .video_processor import VideoProcessor, ScriptSegment, create_sample_script_segments
from .data_aligner import DataAligner, OrderRecord, create_sample_orders
from .script_classifier import ScriptClassifier, SimpleRuleClassifier, ClassificationResult
from .did_attribution import DIDAttributor, AttributionResult, LabelAttributionSummary
from .optimizer import ScriptOptimizer, OptimizationResult

__all__ = [
    # 视频处理
    'VideoProcessor',
    'ScriptSegment',
    'create_sample_script_segments',
    
    # 数据对齐
    'DataAligner',
    'OrderRecord',
    'create_sample_orders',
    
    # 话术分类
    'ScriptClassifier',
    'SimpleRuleClassifier',
    'ClassificationResult',
    
    # DID归因
    'DIDAttributor',
    'AttributionResult',
    'LabelAttributionSummary',
    
    # 优化
    'ScriptOptimizer',
    'OptimizationResult',
]
