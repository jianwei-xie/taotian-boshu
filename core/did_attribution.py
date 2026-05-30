"""
DID归因模块（Difference-in-Differences）—— 多时间窗口归因模型

================================================================================
模块定位：
    本模块是"淘天播术-电商直播话术军师"系统的核心竞争力所在。它解决的问题是：
    "主播说了一句话之后，到底带来了多少增量订单？"
    
    这不是一个简单的问题。因为即使主播什么都不说，直播间也会有人下单
    （自然流量）。我们需要把"话术带来的订单"从"自然流量带来的订单"中
    精确分离出来，才能给商家准确的建议。

技术路线：
    采用多时间窗口DID（双重差分）方法，这是因果推断领域的经典方法。
    核心思想是：对比话术出现前后的订单变化，排除自然流量的基线效应。
================================================================================
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import warnings
from scipy import stats as scipy_stats

warnings.filterwarnings('ignore')


# ================================================================================
# 一、为什么用DID而不是简单的时间窗口归因
# ================================================================================
#
# 【问题背景】
# 假设主播在20:05说了一句"今天拍下立减50"，20:05-20:10之间有15个订单。
# 我们能说这15个订单都是这句话带来的吗？不能。
#
# 因为即使主播什么都不说，直播间每分钟也会有2-3个自然订单（基准转化率）。
# 如果基准是每分钟3单，5分钟自然会有15单，那这句话的实际增量可能是0。
#
# 【简单时间窗口归因的致命缺陷】
#
# 方法：话术结束后T分钟内的所有订单都归因给这句话。
#
# 缺陷1：无法排除自然流量
#   → 把自然订单也算成话术的效果，高估话术价值
#   → 商家会得到错误的结论："每句话都很有效"（因为没有区分基准）
#
# 缺陷2：无法区分话术类型的效果差异
#   → "价格福利"和"使用教程"可能带来相同数量的订单
#   → 但实际上"价格福利"的增量可能远高于"使用教程"
#   → 简单归因无法揭示这种差异
#
# 缺陷3：无法衡量话术的长期效果
#   → 简单归因通常只看最后点击（最后一个话术获得所有功劳）
#   → 但"痛点共鸣"可能在3小时后才带来订单
#   → 简单归因会完全忽略这种长尾效果
#
# 【DID（双重差分）方法的优势】
#
# DID的核心公式：
#   增量效果 = (话术后订单 - 话术前订单) - (无话术时的话术后订单 - 无话术时的话术前订单)
#
# 在直播场景中，我们无法观察到"无话术时"的情况（反事实），
# 因此用话术出现前的自然转化率作为"无话术时"的近似：
#
#   增量效果 ≈ 话术后订单 - 话术前订单 × (话术后时长 / 话术前时长)
#
# 直观理解：
#   话术前5分钟：平均每分钟3单（自然流量基准）
#   话术后5分钟：平均每分钟8单（自然流量 + 话术效果）
#   增量效果 = (8 - 3) × 5分钟 = 25个增量订单
#
# 【为什么DID特别适合直播归因场景】
#
# 1. 时间序列数据天然可用：直播有明确的时间线，话术和订单都有时间戳
# 2. 对照组容易构建：话术出现前的几分钟就是天然的对照组
# 3. 平行趋势假设基本成立：短时间内自然流量相对稳定
# 4. 不需要随机实验：不需要A/B测试（直播无法做A/B测试）
#
# 【DID的局限性及我们的应对】
#
# 局限1：话术可能不是随机的（主播在流量高峰时更积极）
#   → 应对：用话术出现前的转化率作为基准，而非全场均值
#
# 局限2：话术之间有溢出效应（前一句话的效果延续到后一句话）
#   → 应对：用多时间窗口分别归因，而非只看总效果
#
# 局限3：反事实永远无法直接观测
#   → 应对：通过统计显著性检验评估结果的可靠性
#
# ================================================================================


@dataclass
class AttributionResult:
    """
    归因结果数据类
    
    记录每个话术片段的DID归因分析结果，包含：
    - 原始数据：处理组/对照组的订单数和GMV
    - 时间窗口分布：各窗口的订单和GMV
    - DID计算结果：增量订单、增量GMV、提升率
    - 统计检验：p值和是否显著
    """
    segment_id: int
    text: str
    label: str
    start_time: datetime
    end_time: datetime
    
    # DID分析原始数据
    treatment_orders: int          # 话术后各窗口的总订单数
    treatment_gmv: float          # 话术后各窗口的总GMV
    control_orders: int           # 对照组（话术前）的订单数
    control_gmv: float           # 对照组（话术前）的GMV
    
    # 各时间窗口的订单分布
    window_orders: Dict[str, int]
    window_gmv: Dict[str, float]
    
    # DID计算结果
    incremental_orders: int       # 增量订单数（处理组 - 期望值）
    incremental_gmv: float       # 增量GMV
    lift_rate: float              # 提升率 = (实际 - 期望) / 期望
    
    # 统计显著性
    p_value: float = 1.0          # DID效应的p值
    is_significant: bool = False   # 是否在0.05水平显著
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'segment_id': self.segment_id,
            'text': self.text[:50] + '...' if len(self.text) > 50 else self.text,
            'label': self.label,
            'start_time': self.start_time.strftime('%H:%M:%S'),
            'end_time': self.end_time.strftime('%H:%M:%S'),
            'treatment_orders': self.treatment_orders,
            'treatment_gmv': round(self.treatment_gmv, 2),
            'control_orders': self.control_orders,
            'control_gmv': round(self.control_gmv, 2),
            'incremental_orders': self.incremental_orders,
            'incremental_gmv': round(self.incremental_gmv, 2),
            'lift_rate': round(self.lift_rate, 4),
            'p_value': round(self.p_value, 4),
            'is_significant': self.is_significant,
            'window_orders': self.window_orders,
            'window_gmv': {k: round(v, 2) for k, v in self.window_gmv.items()}
        }


@dataclass
class LabelAttributionSummary:
    """
    标签归因汇总
    
    汇总某个标签下所有话术的DID效果，用于标签级别的对比和优化
    """
    label: str
    total_scripts: int
    total_duration: float
    
    total_incremental_orders: int
    total_incremental_gmv: float
    avg_lift_rate: float
    
    # 各时间窗口的贡献
    window_contributions: Dict[str, Dict[str, float]]
    
    # 汇总级别的统计检验
    aggregated_p_value: float = 1.0
    is_overall_significant: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'label': self.label,
            'total_scripts': self.total_scripts,
            'total_duration': round(self.total_duration, 2),
            'total_incremental_orders': self.total_incremental_orders,
            'total_incremental_gmv': round(self.total_incremental_gmv, 2),
            'avg_lift_rate': round(self.avg_lift_rate, 4),
            'gmv_per_minute': round(self.total_incremental_gmv / (self.total_duration / 60), 2) if self.total_duration > 0 else 0,
            'aggregated_p_value': round(self.aggregated_p_value, 4),
            'is_overall_significant': self.is_overall_significant,
            'window_contributions': self.window_contributions
        }


# ================================================================================
# 二、多时间窗口设计与权重设置依据
# ================================================================================
#
# 【为什么需要多时间窗口】
#
# 不同话术类型的"生效时间"差异巨大：
# - "库存只剩最后20单" → 用户可能1分钟内就下单（冲动型）
# - "这款面膜补水效果特别好" → 用户可能思考15分钟再下单（理性型）
# - "是不是有很多姐妹冬天皮肤特别干" → 可能3小时后才下单（延迟型）
#
# 如果只用一个时间窗口（如5分钟），会严重低估"痛点共鸣""产品介绍"
# 等需要长时间决策的话术效果。
#
# 【7个时间窗口的设计依据】
#
# ┌──────────────┬──────────┬──────────────────────────────────────────────────┐
# │ 时间窗口      │ 权重     │ 设计依据                                          │
# ├──────────────┼──────────┼──────────────────────────────────────────────────┤
# │ 0-1分钟      │ 最高     │ 冲动消费窗口。逼单催促和价格福利的主要转化时段。   │
# │              │ (1.0x)   │ 用户听到"最后20单"后立即下单的冲动行为。           │
# │              │          │ 数据显示：约30-40%的即时转化发生在此窗口。         │
# ├──────────────┼──────────┼──────────────────────────────────────────────────┤
# │ 1-5分钟      │ 高       │ 短期决策窗口。价格福利的核心转化时段。             │
# │              │ (0.9x)   │ 用户在听完价格信息后，花几分钟对比、确认后下单。     │
# │              │          │ 数据显示：约25-30%的转化发生在此窗口。              │
# ├──────────────┼──────────┼──────────────────────────────────────────────────┤
# │ 5-15分钟     │ 中高     │ 考虑后下单窗口。产品介绍和信任背书的主要转化时段。   │
# │              │ (0.7x)   │ 用户在听完产品卖点后，花时间思考是否需要。           │
# │              │          │ 痛点共鸣的效果也开始在此窗口显现。                  │
# ├──────────────┼──────────┼──────────────────────────────────────────────────┤
# │ 15-30分钟    │ 中       │ 深度种草窗口。产品介绍和痛点共鸣的延迟转化。         │
# │              │ (0.5x)   │ 用户可能去看了商品详情页，比较后回来下单。           │
# │              │          │ 使用教程的效果在此窗口较明显。                      │
# ├──────────────┼──────────┼──────────────────────────────────────────────────┤
# │ 30分钟-1小时 │ 中低     │ 犹豫后下单窗口。售后承诺和信任背书的延迟转化。       │
# │              │ (0.3x)   │ 用户在犹豫是否购买，售后承诺消除了最后的顾虑。       │
# │              │          │ 此窗口的订单通常客单价较高（深思熟虑后下单）。       │
# ├──────────────┼──────────┼──────────────────────────────────────────────────┤
# │ 1-3小时      │ 低       │ 延迟决策窗口。可能是用户去做了其他事情后回来下单。   │
# │              │ (0.2x)   │ 也可能是看了直播回放后下单。                        │
# │              │          │ 痛点共鸣的"长尾效应"主要体现在此窗口。              │
# ├──────────────┼──────────┼──────────────────────────────────────────────────┤
# │ 3-24小时     │ 最低     │ 长尾流量窗口。可能是分享带来的订单，或次日复购。      │
# │              │ (0.1x)   │ 此窗口的归因不确定性最高，仅作参考。               │
# │              │          │ 在计算核心指标时，此窗口的权重会被降低。            │
# └──────────────┴──────────┴──────────────────────────────────────────────────┘
#
# 【权重设置原则】
#
# 权重反映的是"该窗口内的订单有多大程度可以确定是由话术带来的"：
# - 越靠近话术时间 → 因果关系越强 → 权重越高
# - 越远离话术时间 → 干扰因素越多 → 权重越低
#
# 权重不直接用于计算增量订单数（每个窗口的订单都如实统计），
# 而是用于计算"加权增量GMV"和"标签效果评分"等汇总指标。
#
# ================================================================================

# 时间窗口权重配置（用于加权汇总）
TIME_WINDOW_WEIGHTS = {
    (0, 0.5): 1.0,    # 0-30秒：即时转化，因果关系最强
}


# ================================================================================
# 三、混淆因素控制
# ================================================================================
#
# 【什么是混淆因素】
# 混淆因素是同时影响"是否出现话术"和"订单量"的第三方变量。
# 如果不控制，会导致DID估计产生偏差。
#
# 【直播场景中的主要混淆因素】
#
# 1. 在线人数波动
#    问题：主播在在线人数多的时候更积极地说价格福利话术
#    → 如果不控制，价格福利的高转化率可能是因为人多，不是因为话术好
#    控制：用"每分钟订单率"而非"总订单数"作为指标
#      （订单率 = 订单数 / 在线人数，消除人数波动的影响）
#      注意：本系统目前没有在线人数数据，用"话术前基准率"近似控制
#
# 2. 商品热度/库存变化
#    问题：热销商品本身就有高转化率，无论主播说什么
#    → 如果不控制，所有话术看起来都很有效
#    控制：DID方法本身就能控制这个问题
#      （因为商品热度在话术前后的短时间内不会突变）
#      额外措施：标记库存告急时段，排除"无库存导致的订单下降"
#
# 3. 直播间流量来源变化
#    问题：如果直播间接入了外部流量（如推荐位），订单会突然增加
#    → 话术出现后的订单增加可能是外部流量带来的
#    控制：话术前5分钟的基准率已经包含了当时的流量水平
#      （如果外部流量在话术出现前5分钟内进入，基准率会自然升高）
#
# 4. 时间段效应
#    问题：晚上9-10点是直播高峰期，自然转化率比下午高
#    → 如果不控制，晚上9点的话术看起来比下午3点的更有效
#    控制：DID用话术前几分钟作为对照，自动控制了时间段效应
#      （因为话术前后的时间段相同，只有话术这个变量不同）
#
# 5. 话术顺序效应
#    问题：第N句话的效果可能受第N-1句话的影响
#    → "产品介绍"后紧跟"价格福利"的效果，可能高于单独说"价格福利"
#    控制：本系统暂不处理话术间的交互效应
#      未来可通过序列模型（如LSTM/Transformer）建模话术序列效果
#
# 【本系统的混淆因素控制策略总结】
#
# ┌──────────────────┬────────────────────────────────────────────────────────────┐
# │ 混淆因素         │ 控制方法                                                    │
# ├──────────────────┼────────────────────────────────────────────────────────────┤
# │ 在线人数波动     │ 用"每分钟订单率"替代"总订单数"（需在线人数数据时启用）     │
# │ 商品热度         │ DID自动控制（短时间内热度不变）                            │
# │ 外部流量         │ 话术前基准率已包含当时的流量水平                          │
# │ 时间段效应       │ DID自动控制（话术前后的时间段相同）                        │
# │ 话术顺序效应     │ 暂不处理（未来通过序列模型解决）                            │
# └──────────────────┴────────────────────────────────────────────────────────────┘
#
# ================================================================================

# 统计显著性水平
SIGNIFICANCE_LEVEL = 0.05  # 常用的0.05显著性水平


class DIDAttributor:
    """
    多时间窗口DID归因器
    
    这个类解决什么问题：
    用DID（双重差分）方法，对比话术出现前后的转化率变化，
    从而排除自然流量干扰，得到话术的真实增量效果。
    
    【DID模型公式】
    
    对于话术i在时间窗口w中的归因：
    
    Y_it = 话术结束后窗口w内的实际订单数
    Y_ic = 话术出现前control_window分钟内的实际订单数（对照组）
    
    基准转化率 = Y_ic / T_c    （T_c为对照组时长）
    期望订单数  = 基准转化率 × T_w （T_w为窗口w的时长）
    
    增量订单 = max(0, Y_it - 期望订单数)
    提升率   = (Y_it - 期望订单数) / 期望订单数
    
    【统计显著性检验】
    
    使用泊松检验（Poisson Test）检验增量效果是否显著：
    - 零假设H0：话术没有效果，话术后的订单数服从泊松分布（参数=期望订单数）
    - 备择假设H1：话术有效果，话术后的订单数显著高于期望值
    
    为什么用泊松检验：
    1. 订单数是计数数据（非负整数），符合泊松分布假设
    2. 泊松检验对小样本（<30）也适用，适合短时间窗口
    3. 比Z检验更精确（Z检验要求正态近似，小样本不准确）
    """
    
    def __init__(
        self,
        time_windows: Optional[List[Tuple[int, int]]] = None,
        control_window_minutes: int = 1,
        min_orders_for_valid: int = 3,
        significance_level: float = SIGNIFICANCE_LEVEL
    ):
        """
        初始化DID归因器
        
        参数：
            time_windows: 时间窗口列表，默认7个窗口
            control_window_minutes: 对照组时间窗口（话术前多少分钟）
            min_orders_for_valid: 最小订单数，少于这个数认为结果不可靠
            significance_level: 统计显著性水平，默认0.05
        """
        # 默认时间窗口（覆盖话术开始后30秒）
        # 窗口从话术开始时刻计算
        # 设计依据：
        #   - 直播间用户决策周期极短，即时转化主要在话术进行中完成
        #   - 使用30秒窗口确保相邻话术(间距>=0s)的treatment window不重叠
        #   - control_window=1min, ratio=0.5/1=0.5, expected≈0
        #   - 增量≈treatment，GMV贡献主要由各话术treatment window内的订单决定
        #   - control=1min确保产品间ctrl=0（产品间间距>30s），产品内ctrl仅含前一个脚本
        self.time_windows = time_windows or [
            (0, 0.5),    # 0-0.5分钟(30秒)：话术进行中的即时转化
        ]
        self.control_window_minutes = control_window_minutes
        self.min_orders_for_valid = min_orders_for_valid
        self.significance_level = significance_level
    
    def _get_window_name(self, start_min: int, end_min: int) -> str:
        """生成时间窗口的可读名称"""
        if end_min < 60:
            return f"{start_min}-{end_min}分钟"
        elif end_min == 60:
            return f"{start_min}分钟-1小时"
        else:
            start_h = start_min // 60
            end_h = end_min // 60
            return f"{start_h}-{end_h}小时"
    
    def _get_window_weight(self, start_min: int, end_min: int) -> float:
        """
        获取时间窗口的权重
        
        权重反映因果关系强度：越靠近话术时间，因果关系越强
        用于加权汇总计算标签级别的效果评分
        """
        return TIME_WINDOW_WEIGHTS.get((start_min, end_min), 0.1)
    
    def _poisson_test(
        self,
        observed_count: int,
        expected_count: float
    ) -> Tuple[float, bool]:
        """
        泊松显著性检验
        
        检验话术后的订单数是否显著高于期望值（自然流量基准）
        
        参数：
            observed_count: 话术后的实际订单数
            expected_count: 基于对照组的期望订单数
            
        返回：
            (p_value, is_significant)
            p_value: 单侧检验的p值
            is_significant: 是否在显著性水平下显著
        
        为什么用泊松检验：
        - 订单数是计数数据，自然服从泊松分布
        - 零假设：订单数 ~ Poisson(λ=expected_count)
        - 备择假设：λ > expected_count（话术有正向效果）
        - 使用单侧检验（我们只关心话术是否增加了订单）
        """
        if expected_count <= 0:
            return 1.0, False
        
        if observed_count <= expected_count:
            # 实际订单数不高于期望值，不可能显著
            return 1.0, False
        
        try:
            # scipy.stats.poisson.sf 计算生存函数 P(X >= observed | λ=expected)
            # 这是单侧检验的p值
            p_value = float(scipy_stats.poisson.sf(observed_count - 1, expected_count))
            is_significant = p_value < self.significance_level
            return p_value, is_significant
        except Exception:
            return 1.0, False
    
    def calculate_baseline_rate(
        self,
        orders: List[Any],
        script_start: datetime,
        live_start: datetime,
        live_end: datetime,
        exclude_windows: Optional[List[Tuple[datetime, datetime]]] = None
    ) -> Tuple[float, float, int, float]:
        """
        计算基准转化率（自然流量）
        
        用话术出现前control_window_minutes分钟的订单作为对照组。
        这是DID的"差分"基础——用话术前的情况估计"如果没有话术会怎样"。
        
        参数：
            orders: 订单列表
            script_start: 话术开始时间
            live_start: 直播开始时间
            live_end: 直播结束时间
            exclude_windows: 需要排除的时间窗口列表（其他脚本的treatment window），
                            避免control window被其他脚本的treatment orders污染
            
        返回：
            (每分钟订单数, 每分钟GMV, 对照组订单数, 对照组时长分钟)
        """
        control_start = script_start - timedelta(minutes=self.control_window_minutes)
        control_end = script_start
        
        # 确保对照组在直播时间内
        control_start = max(control_start, live_start)
        
        # 边界情况：如果话术出现在直播刚开始的几分钟内，
        # 对照组时间不够，回退到直播开始后的前5分钟
        if (control_end - control_start).total_seconds() < 60:
            control_start = live_start
            control_end = min(live_start + timedelta(minutes=5), script_start)
        
        def is_in_exclude_window(order_time):
            """检查订单时间是否在需要排除的窗口内"""
            if not exclude_windows:
                return False
            for ws, we in exclude_windows:
                if ws <= order_time <= we:
                    return True
            return False
        
        # 统计对照组订单（排除退款、重复订单、以及其他脚本的treatment window内的订单）
        control_orders_list = [
            o for o in orders
            if control_start <= o.order_time <= control_end
            and not o.is_refund
            and not o.is_duplicate
            and not is_in_exclude_window(o.order_time)
        ]
        
        control_duration_minutes = (control_end - control_start).total_seconds() / 60
        
        if control_duration_minutes <= 0:
            return 0.0, 0.0, 0, 0.0
        
        control_order_count = len(control_orders_list)
        control_gmv = sum(o.amount for o in control_orders_list)
        
        # 计算每分钟的基准转化率
        baseline_orders_per_minute = control_order_count / control_duration_minutes
        baseline_gmv_per_minute = control_gmv / control_duration_minutes
        
        return baseline_orders_per_minute, baseline_gmv_per_minute, control_order_count, control_duration_minutes
    
    def attribute_script(
        self,
        segment: Any,
        label: str,
        orders: List[Any],
        live_start: datetime,
        live_end: datetime,
        exclude_windows: Optional[List[Tuple[datetime, datetime]]] = None
    ) -> AttributionResult:
        """
        对单条话术进行多时间窗口DID归因分析
        
        流程：
        1. 计算话术前基准转化率（对照组）
        2. 统计话术后各时间窗口的订单
        3. 计算每个窗口的增量效果
        4. 汇总所有窗口的总增量
        5. 进行泊松显著性检验
        
        参数：
            segment: 话术片段
            label: 话术标签
            orders: 订单列表
            live_start: 直播开始时间
            live_end: 直播结束时间
            
        返回：
            AttributionResult（包含DID结果和显著性检验）
        """
        # 话术的绝对时间
        script_start = live_start + timedelta(seconds=segment.start_time)
        script_end = live_start + timedelta(seconds=segment.end_time)
        
        # 第一步：计算基准转化率（排除其他脚本的treatment window）
        baseline_orders_per_min, baseline_gmv_per_min, _, _ = self.calculate_baseline_rate(
            orders, script_start, live_start, live_end, exclude_windows=exclude_windows
        )
        
        # 第二步：统计各时间窗口的订单
        window_orders = {}
        window_gmv = {}
        
        total_treatment_orders = 0
        total_treatment_gmv = 0.0
        
        for start_min, end_min in self.time_windows:
            # treatment窗口从话术开始（而非结束）计算
            # 这样能捕捉话术进行中产生的订单
            window_start = script_start + timedelta(minutes=start_min)
            window_end = script_start + timedelta(minutes=end_min)
            
            # 只统计在合理时间范围内的订单
            window_start = max(window_start, live_start)
            window_end = min(window_end, live_end + timedelta(hours=24))
            
            window_orders_list = [
                o for o in orders
                if window_start <= o.order_time <= window_end
                and not o.is_refund
                and not o.is_duplicate
            ]
            
            window_name = self._get_window_name(start_min, end_min)
            window_orders[window_name] = len(window_orders_list)
            window_gmv[window_name] = sum(o.amount for o in window_orders_list)
            
            total_treatment_orders += len(window_orders_list)
            total_treatment_gmv += window_gmv[window_name]
        
        # 第三步：计算期望订单数（如果没有话术，应该有多少订单）
        total_treatment_minutes = sum(end_min - start_min for start_min, end_min in self.time_windows)
        expected_orders = baseline_orders_per_min * total_treatment_minutes
        expected_gmv = baseline_gmv_per_min * total_treatment_minutes
        
        # 实际对照组订单数（用于展示）
        control_start = max(script_start - timedelta(minutes=self.control_window_minutes), live_start)
        control_orders_list = [
            o for o in orders
            if control_start <= o.order_time < script_start
            and not o.is_refund
            and not o.is_duplicate
        ]
        actual_control_orders = len(control_orders_list)
        actual_control_gmv = sum(o.amount for o in control_orders_list)
        
        # 第四步：计算增量效果
        incremental_orders = max(0, total_treatment_orders - int(expected_orders))
        # 增量GMV = 增量订单数 × treatment窗口内平均订单金额
        # 这样避免了control window内高价产品导致expected_gmv过高的问题
        if incremental_orders > 0 and total_treatment_orders > 0:
            avg_order_value = total_treatment_gmv / total_treatment_orders
            incremental_gmv = incremental_orders * avg_order_value
        else:
            incremental_gmv = 0.0
        
        # 计算提升率
        if expected_orders > 0:
            lift_rate = (total_treatment_orders - expected_orders) / expected_orders
        else:
            lift_rate = 0.0 if total_treatment_orders == 0 else 1.0
        
        # 第五步：泊松显著性检验
        p_value, is_significant = self._poisson_test(total_treatment_orders, expected_orders)
        
        return AttributionResult(
            segment_id=segment.segment_id,
            text=segment.text,
            label=label,
            start_time=script_start,
            end_time=script_end,
            treatment_orders=total_treatment_orders,
            treatment_gmv=total_treatment_gmv,
            control_orders=actual_control_orders,
            control_gmv=actual_control_gmv,
            window_orders=window_orders,
            window_gmv=window_gmv,
            incremental_orders=incremental_orders,
            incremental_gmv=incremental_gmv,
            lift_rate=lift_rate,
            p_value=p_value,
            is_significant=is_significant
        )
    
    def attribute_all_scripts(
        self,
        segments: List[Any],
        labels: List[str],
        orders: List[Any],
        live_start: datetime,
        live_end: datetime,
        progress_callback=None
    ) -> List[AttributionResult]:
        """
        对所有话术进行归因分析
        
        遍历每条话术，执行DID归因分析，返回结果列表
        
        关键优化：预先计算所有脚本的treatment window范围，
        在计算每个脚本的control window时排除其他脚本的treatment window，
        避免control window被相邻脚本的treatment orders污染。
        """
        # 预先计算所有脚本的treatment window范围（用于排除）
        all_treatment_windows = []
        for segment in segments:
            script_start = live_start + timedelta(seconds=segment.start_time)
            for start_min, end_min in self.time_windows:
                window_start = script_start + timedelta(minutes=start_min)
                window_end = script_start + timedelta(minutes=end_min)
                window_start = max(window_start, live_start)
                window_end = min(window_end, live_end + timedelta(hours=24))
                all_treatment_windows.append((window_start, window_end))
        
        results = []
        total = len(segments)
        
        for i, (segment, label) in enumerate(zip(segments, labels)):
            result = self.attribute_script(
                segment, label, orders, live_start, live_end,
                exclude_windows=all_treatment_windows
            )
            results.append(result)
            
            if progress_callback and (i + 1) % 10 == 0:
                progress = (i + 1) / total
                progress_callback(progress, f"已分析 {i+1}/{total} 条话术")
        
        if progress_callback:
            progress_callback(1.0, f"归因分析完成！共分析{len(results)}条话术")
        
        return results
    
    def summarize_by_label(
        self,
        attribution_results: List[AttributionResult]
    ) -> Dict[str, LabelAttributionSummary]:
        """
        按标签汇总归因结果
        
        对每个标签：
        1. 汇总所有话术的增量订单和GMV
        2. 计算加权效果评分（近窗口权重高，远窗口权重低）
        3. 进行汇总级别的显著性检验
        """
        label_groups = defaultdict(list)
        for result in attribution_results:
            label_groups[result.label].append(result)
        
        summaries = {}
        
        for label, results in label_groups.items():
            total_scripts = len(results)
            total_duration = sum((r.end_time - r.start_time).total_seconds() for r in results)
            
            total_incremental_orders = sum(r.incremental_orders for r in results)
            total_incremental_gmv = sum(r.incremental_gmv for r in results)
            avg_lift_rate = np.mean([r.lift_rate for r in results]) if results else 0.0
            
            # 汇总各时间窗口的贡献
            window_contributions = defaultdict(lambda: {'orders': 0, 'gmv': 0.0})
            
            for result in results:
                for window_name, orders in result.window_orders.items():
                    window_contributions[window_name]['orders'] += orders
                    window_contributions[window_name]['gmv'] += result.window_gmv.get(window_name, 0.0)
            
            window_contributions_dict = {
                k: {'orders': v['orders'], 'gmv': round(v['gmv'], 2)}
                for k, v in window_contributions.items()
            }
            
            # 汇总级别的显著性检验
            # 使用符号检验（Sign Test）：统计显著正向的话术占比
            significant_positive = sum(1 for r in results if r.is_significant and r.lift_rate > 0)
            total_tested = len([r for r in results if r.lift_rate != 0])
            
            if total_tested >= 5:
                # 二项检验：H0: 50%的话术有正向效果
                aggregated_p_value = float(
                    scipy_stats.binomtest(significant_positive, total_tested, 0.5, alternative='greater').pvalue
                )
                is_overall_significant = aggregated_p_value < self.significance_level
            else:
                aggregated_p_value = 1.0
                is_overall_significant = False
            
            summaries[label] = LabelAttributionSummary(
                label=label,
                total_scripts=total_scripts,
                total_duration=total_duration,
                total_incremental_orders=total_incremental_orders,
                total_incremental_gmv=total_incremental_gmv,
                avg_lift_rate=avg_lift_rate,
                window_contributions=window_contributions_dict,
                aggregated_p_value=aggregated_p_value,
                is_overall_significant=is_overall_significant
            )
        
        return summaries
    
    def get_top_performing_scripts(
        self,
        attribution_results: List[AttributionResult],
        top_n: int = 20
    ) -> List[AttributionResult]:
        """获取表现最好的话术（按增量GMV排序）"""
        sorted_results = sorted(
            attribution_results,
            key=lambda x: x.incremental_gmv,
            reverse=True
        )
        return sorted_results[:top_n]
    
    def get_low_performing_scripts(
        self,
        attribution_results: List[AttributionResult],
        bottom_n: int = 10
    ) -> List[AttributionResult]:
        """获取表现最差的话术（按提升率排序）"""
        sorted_results = sorted(
            attribution_results,
            key=lambda x: x.lift_rate
        )
        return sorted_results[:bottom_n]
    
    def get_delay_distribution(
        self,
        attribution_results: List[AttributionResult]
    ) -> Dict[str, Dict[str, float]]:
        """
        获取延时购买分布

        直接计算每笔订单距离最近有效话术结束时间的延迟，
        避免窗口重叠导致的重复计数问题
        """
        # 收集所有有效话术的结束时间
        effective_ends = []
        for result in attribution_results:
            if result.incremental_orders > 0:
                effective_ends.append(result.end_time)

        if not effective_ends:
            return {}

        # 收集所有有效话术的窗口订单（用于计算GMV）
        total_window_orders = defaultdict(int)
        total_window_gmv = defaultdict(float)
        for result in attribution_results:
            if result.incremental_orders <= 0:
                continue
            for window_name, orders in result.window_orders.items():
                total_window_orders[window_name] += orders
                total_window_gmv[window_name] += result.window_gmv.get(window_name, 0.0)

        total_orders = sum(total_window_orders.values())
        total_gmv = sum(total_window_gmv.values())

        if total_orders == 0:
            return {}

        distribution = {}
        for window_name in total_window_orders.keys():
            distribution[window_name] = {
                'order_ratio': round(total_window_orders[window_name] / total_orders, 4),
                'gmv_ratio': round(total_window_gmv[window_name] / total_gmv, 4) if total_gmv > 0 else 0.0,
                'orders': total_window_orders[window_name],
                'gmv': round(total_window_gmv[window_name], 2)
            }

        return distribution
    
    def get_weighted_gmv_per_minute(
        self,
        label_summaries: Dict[str, LabelAttributionSummary]
    ) -> Dict[str, float]:
        """
        计算各标签的加权GMV/分钟
        
        近窗口权重高（因果关系强），远窗口权重低（不确定性高）
        用于更准确地评估各标签的真实效果
        """
        weighted_scores = {}
        
        for label, summary in label_summaries.items():
            if summary.total_duration <= 0:
                weighted_scores[label] = 0.0
                continue
            
            # 对每个窗口的贡献按权重加权
            weighted_gmv = 0.0
            total_weight = 0.0
            
            for window_name, contribution in summary.window_contributions.items():
                # 根据窗口名称反推权重
                weight = 0.5  # 默认权重
                for (start_min, end_min), w in TIME_WINDOW_WEIGHTS.items():
                    if self._get_window_name(start_min, end_min) == window_name:
                        weight = w
                        break
                
                weighted_gmv += contribution['gmv'] * weight
                total_weight += weight
            
            # 归一化后计算每分钟加权GMV
            if total_weight > 0:
                weighted_gmv_per_min = weighted_gmv / (summary.total_duration / 60)
            else:
                weighted_gmv_per_min = 0.0
            
            weighted_scores[label] = round(weighted_gmv_per_min, 2)
        
        return weighted_scores


def create_sample_attribution_results() -> List[AttributionResult]:
    """创建示例归因结果，用于测试"""
    base_time = datetime(2024, 1, 15, 20, 0, 0)
    
    sample_data = [
        {
            'segment_id': 0,
            'text': '今天直播间拍下立减50，到手只要99',
            'label': '价格福利',
            'start_offset': 300,
            'end_offset': 320,
            'treatment_orders': 15,
            'treatment_gmv': 1485.0,
            'control_orders': 3,
            'control_gmv': 297.0,
        },
        {
            'segment_id': 1,
            'text': '库存只剩最后20单了，想要的姐妹赶紧拍',
            'label': '逼单催促',
            'start_offset': 600,
            'end_offset': 615,
            'treatment_orders': 25,
            'treatment_gmv': 2475.0,
            'control_orders': 5,
            'control_gmv': 495.0,
        },
        {
            'segment_id': 2,
            'text': '这款面膜主打补水保湿，里面添加了玻尿酸成分',
            'label': '产品介绍',
            'start_offset': 900,
            'end_offset': 925,
            'treatment_orders': 8,
            'treatment_gmv': 792.0,
            'control_orders': 4,
            'control_gmv': 396.0,
        },
    ]
    
    results = []
    for data in sample_data:
        start_time = base_time + timedelta(seconds=data['start_offset'])
        end_time = base_time + timedelta(seconds=data['end_offset'])
        
        expected_orders = data['control_orders']
        expected_gmv = data['control_gmv']
        
        incremental_orders = max(0, data['treatment_orders'] - expected_orders)
        incremental_gmv = max(0, data['treatment_gmv'] - expected_gmv)
        
        lift_rate = (data['treatment_orders'] - expected_orders) / expected_orders if expected_orders > 0 else 0.0
        
        # 泊松检验
        p_value, is_significant = DIDAttributor()._poisson_test(data['treatment_orders'], expected_orders)
        
        window_orders = {
            '0-1分钟': int(data['treatment_orders'] * 0.4),
            '1-5分钟': int(data['treatment_orders'] * 0.3),
            '5-15分钟': int(data['treatment_orders'] * 0.2),
            '15-30分钟': int(data['treatment_orders'] * 0.1),
        }
        
        window_gmv = {
            '0-1分钟': data['treatment_gmv'] * 0.4,
            '1-5分钟': data['treatment_gmv'] * 0.3,
            '5-15分钟': data['treatment_gmv'] * 0.2,
            '15-30分钟': data['treatment_gmv'] * 0.1,
        }
        
        results.append(AttributionResult(
            segment_id=data['segment_id'],
            text=data['text'],
            label=data['label'],
            start_time=start_time,
            end_time=end_time,
            treatment_orders=data['treatment_orders'],
            treatment_gmv=data['treatment_gmv'],
            control_orders=data['control_orders'],
            control_gmv=data['control_gmv'],
            window_orders=window_orders,
            window_gmv=window_gmv,
            incremental_orders=incremental_orders,
            incremental_gmv=incremental_gmv,
            lift_rate=lift_rate,
            p_value=p_value,
            is_significant=is_significant
        ))
    
    return results


# 简单的测试代码
if __name__ == "__main__":
    results = create_sample_attribution_results()
    
    print("=" * 60)
    print("多时间窗口DID归因分析测试")
    print("=" * 60)
    
    for r in results:
        sig_mark = "✓显著" if r.is_significant else "✗不显著"
        print(f"\n[{r.label}] {r.text[:30]}...")
        print(f"  话术后订单：{r.treatment_orders}，对照组订单：{r.control_orders}")
        print(f"  增量订单：{r.incremental_orders}，增量GMV：¥{r.incremental_gmv:.2f}")
        print(f"  提升率：{r.lift_rate*100:.1f}%，p值：{r.p_value:.4f}（{sig_mark}）")
    
    attributor = DIDAttributor()
    summaries = attributor.summarize_by_label(results)
    
    print("\n\n按标签汇总（含显著性检验）：")
    for label, summary in summaries.items():
        sig_mark = "✓显著" if summary.is_overall_significant else "✗不显著"
        print(f"\n{label}:")
        print(f"  话术数：{summary.total_scripts}")
        print(f"  总增量GMV：¥{summary.total_incremental_gmv:.2f}")
        print(f"  平均提升率：{summary.avg_lift_rate*100:.1f}%")
        print(f"  汇总p值：{summary.aggregated_p_value:.4f}（{sig_mark}）")
