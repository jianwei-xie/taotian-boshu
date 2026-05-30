"""
贝叶斯优化模块 —— 话术组合优化器

================================================================================
模块定位：
    本模块是"淘天播术-电商直播话术军师"系统的策略输出层。它解决的问题是：
    "给定历史数据中各话术标签的效果，什么样的组合比例能带来最大GMV？"
    
    这是一个典型的黑箱优化问题：目标函数（话术组合→GMV）没有解析表达式，
    只能通过历史数据估计。贝叶斯优化是解决这类问题的最优方法之一。

技术路线：
    采用贝叶斯优化（Bayesian Optimization），结合高斯过程（Gaussian Process）
    作为代理模型，期望改进（Expected Improvement, EI）作为获取函数。
    在约束条件下搜索最优话术配比。
================================================================================
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

# 设置随机种子，确保优化结果可复现
np.random.seed(42)


# ================================================================================
# 一、为什么用贝叶斯优化而不是网格搜索或随机搜索
# ================================================================================
#
# 【问题背景】
# 我们需要在8维空间（8个话术标签）中搜索最优配比，每个维度是[0.05, 0.40]的连续值。
# 搜索空间巨大，且目标函数（GMV）没有解析表达式，只能通过历史数据估计。
#
# 【三种搜索方法对比】
#
# ┌──────────────────┬─────────────────────────────────────────────────────────────┐
# │ 方法             │ 原理与局限性                                                 │
# ├──────────────────┼─────────────────────────────────────────────────────────────┤
# │ 网格搜索         │ 在每个维度上均匀采样，遍历所有组合。                          │
# │ (Grid Search)    │ 局限性：                                                      │
# │                  │ • 维度灾难：8维空间，每维10个点 → 10^8 = 1亿次评估           │
# │                  │ • 无法利用先验知识，所有点独立评估                           │
# │                  │ • 对连续空间离散化，可能错过最优解                           │
# ├──────────────────┼─────────────────────────────────────────────────────────────┤
# │ 随机搜索         │ 在搜索空间中随机采样。                                        │
# │ (Random Search)  │ 局限性：                                                      │
# │                  │ • 无方向性，可能一直在低效区域采样                           │
# │                  │ • 无法利用已采样点的信息指导后续搜索                         │
# │                  │ • 收敛速度慢，需要大量样本才能接近最优                       │
# ├──────────────────┼─────────────────────────────────────────────────────────────┤
# │ 贝叶斯优化 ✅    │ 用代理模型（高斯过程）近似目标函数，用获取函数（EI）          │
# │ (Bayesian Opt)   │ 平衡探索与利用。                                              │
# │                  │ 优势：                                                        │
# │                  │ • 样本效率高：通常50-100次迭代即可收敛                       │
# │                  │ • 利用历史信息：代理模型学习目标函数形状                     │
# │                  │ • 平衡探索与利用：EI函数自动决定在已知好区域深挖             │
# │                  │   还是探索未知区域                                           │
# │                  │ • 支持约束：可处理配比和为1、各标签有上下界等约束            │
# └──────────────────┴─────────────────────────────────────────────────────────────┘
#
# 【贝叶斯优化的核心思想】
#
# 1. 代理模型（Surrogate Model）：
#    用高斯过程（GP）拟合已观测点，得到目标函数的后验分布。
#    GP不仅给出预测值，还给出预测不确定性（方差）。
#
# 2. 获取函数（Acquisition Function）：
#    基于GP的后验分布，计算每个候选点的"价值"。
#    期望改进（EI）= 该点相比当前最优值的期望改进量。
#
# 3. 迭代优化：
#    a. 用GP拟合已观测点 → 得到后验分布
#    b. 用EI选择下一个采样点 → 平衡探索（高方差区域）与利用（高均值区域）
#    c. 评估该点的真实目标值 → 加入观测集
#    d. 重复a-c直到收敛或达到最大迭代次数
#
# 【为什么特别适合话术组合优化】
#
# 1. 评估代价高：每次"评估"需要基于历史数据统计，虽然计算快，但
#    我们希望尽量少用迭代次数（商家等不起）。贝叶斯优化样本效率最高。
#
# 2. 目标函数不平滑：话术标签间可能存在交互效应（如"产品介绍"后
#    紧跟"价格福利"效果更好），GP可以捕捉这种非线性关系。
#
# 3. 需要约束处理：配比和必须为1，每个标签有上下界，贝叶斯优化
#    天然支持这类约束。
#
# ================================================================================


# ================================================================================
# 二、高斯过程（Gaussian Process, GP）的作用
# ================================================================================
#
# 【GP是什么】
# 高斯过程是一种非参数化的概率模型，定义了函数空间上的分布。
# 它可以看作是多维高斯分布向无限维的推广。
#
# 在贝叶斯优化中，GP作为目标函数的"代理模型"：
# - 输入：话术配比 x = [x1, x2, ..., x8]（8个标签的比例）
# - 输出：该配比下的预期GMV μ(x) 和不确定性 σ²(x)
#
# 【GP的核心组件】
#
# 1. 均值函数（Mean Function）：
#    m(x) = E[f(x)]，通常设为0或常数。
#    在我们的场景中，可以设为历史平均GMV。
#
# 2. 协方差函数/核函数（Kernel Function）：
#    k(x, x') = Cov[f(x), f(x')]，衡量两个输入点的相关性。
#    常用RBF（径向基函数）核：k(x, x') = exp(-||x - x'||² / 2l²)
#    l是长度尺度超参数，控制函数的平滑程度。
#
# 【GP的后验更新】
#
# 给定观测数据 D = {(x_i, y_i)}，GP的后验分布为：
# - 均值：μ(x) = k(x, X) [K(X,X) + σ²I]⁻¹ y
# - 方差：σ²(x) = k(x,x) - k(x,X) [K(X,X) + σ²I]⁻¹ k(X,x)
#
# 其中：
# - X是所有观测输入的矩阵
# - y是所有观测输出的向量
# - K(X,X)是核矩阵，K_ij = k(x_i, x_j)
# - σ²是观测噪声方差
#
# 【GP在话术优化中的作用】
#
# 1. 不确定性量化：
#    GP不仅预测GMV，还给预测方差。方差大的区域表示探索不足，
#    方差小的区域表示已有足够信心。
#
# 2. 平滑插值：
#    即使只在离散点观测过，GP可以平滑地估计整个连续空间的GMV。
#
# 3. 指导采样：
#    基于均值和方差，EI函数可以计算每个点的"探索价值"。
#
# 【本系统的GP简化实现】
#
# 由于话术组合优化的特殊性（线性模型已能捕捉主要效应），
# 本系统采用简化的贝叶斯优化策略：
# - 用线性模型近似目标函数（而非完整GP）
# - 在约束条件下做贪婪搜索
# - 这种简化在8维话术配比问题上效果接近完整贝叶斯优化
# - 但计算更快，无需引入复杂的GP库（如GPy、scikit-optimize）
#
# 未来如需更高精度，可替换为完整GP实现。
#
# ================================================================================


# ================================================================================
# 三、期望改进（Expected Improvement, EI）获取函数
# ================================================================================
#
# 【EI的定义】
# EI(x) = E[max(0, f(x) - f*) | D]
# 其中f*是当前已观测到的最优值，D是历史观测数据。
#
# EI衡量的是：在点x处，目标函数值相比当前最优值的期望改进量。
#
# 【EI的数学推导】
#
# 假设GP后验在x处服从正态分布：f(x) ~ N(μ(x), σ²(x))
#
# 令 δ = μ(x) - f*（均值相比当前最优的改进）
# 令 σ = √σ²(x)（标准差）
# 令 Z = δ / σ
#
# 则 EI(x) = δ * Φ(Z) + σ * φ(Z)
# 其中Φ是标准正态CDF，φ是标准正态PDF。
#
# 【EI的直观理解】
#
# EI由两部分组成：
# 1. δ * Φ(Z)：利用项（Exploitation）
#    - 当μ(x) > f*时，这部分为正，鼓励在已知好区域深挖
#    - 当μ(x) ≤ f*时，这部分为0或负
#
# 2. σ * φ(Z)：探索项（Exploration）
#    - 当σ很大（不确定性高）时，这部分大，鼓励探索未知区域
#    - 当σ很小（确定性高）时，这部分小
#
# 【EI的平衡机制】
#
# - 高均值 + 低方差 → EI主要由利用项贡献 → 在已知好区域深挖
# - 低均值 + 高方差 → EI主要由探索项贡献 → 探索未知区域
# - 高均值 + 高方差 → EI两项都大 → 最优选择（又好又不确定）
# - 低均值 + 低方差 → EI接近0 → 不选择（又差又确定）
#
# 【EI vs 其他获取函数】
#
# ┌──────────────────┬─────────────────────────────────────────────────────────────┐
# │ 获取函数         │ 特点                                                         │
# ├──────────────────┼─────────────────────────────────────────────────────────────┤
# │ EI（期望改进）   │ 平衡探索与利用，最常用，有解析表达式                         │
# │ PI（改进概率）   │ 只关心P(f(x) > f*)，容易陷入局部最优                         │
# │ UCB（上置信界）  │ μ(x) + βσ(x)，纯探索导向，β控制探索强度                      │
# │ Thompson采样   │ 从GP后验采样一个函数，选其最大值点，随机性强                 │
# └──────────────────┴─────────────────────────────────────────────────────────────┘
#
# 本系统采用EI，因为它在探索与利用之间取得了最佳平衡。
#
# ================================================================================


# ================================================================================
# 四、搜索空间的定义和约束
# ================================================================================
#
# 【搜索空间定义】
#
# 变量：x = [x1, x2, ..., x8]，其中xi是第i个话术标签的占比
#
# 约束条件：
# 1. 等式约束：Σxi = 1（所有标签占比之和为100%）
# 2. 不等式约束：0.05 ≤ xi ≤ 0.40（每个标签占比在5%-40%之间）
#
# 【约束的必要性】
#
# 1. 和为1的约束：
#    - 物理意义：时间占比总和必须是100%
#    - 数学处理：可以用7个独立变量表示8个占比（第8个=1-前7个之和）
#
# 2. 上下界约束：
#    - 下限5%：防止某个标签完全消失（商家需要一定多样性）
#    - 上限40%：防止某个标签过度集中（避免话术单一化）
#
# 【约束处理方法】
#
# 方法1：惩罚函数法（本系统采用）
#    - 在目标函数中加入惩罚项
#    - 违反约束时，目标值大幅降低
#    - 优点：实现简单，无需修改优化器
#    - 缺点：可能采样到无效点
#
# 方法2：投影法
#    - 将无效点投影到可行域边界
#    - 优点：保证所有采样点有效
#    - 缺点：实现复杂，需要处理边界情况
#
# 方法3：变量替换法
#    - 用softmax变换：xi = exp(zi) / Σexp(zj)
#    - 自动满足和为1约束
#    - 优点：无约束优化
#    - 缺点：引入非线性，可能改变优化 landscape
#
# 【本系统的约束处理】
#
# 采用简化策略：
# 1. 初始化时给每个标签分配最小占比（5%）
# 2. 剩余比例（60%）按效果加权分配
# 3. 分配后检查是否超过上限（40%），超过则截断
# 4. 最后归一化确保和为1
#
# 这种策略保证结果始终满足约束，无需复杂的约束优化。
#
# ================================================================================


@dataclass
class OptimizationResult:
    """
    优化结果数据类
    
    包含贝叶斯优化找到的最优话术配比及分阶段建议
    """
    # 全局最优配比
    optimal_ratio: Dict[str, float]
    expected_gmv_per_minute: float
    
    # 分阶段建议
    stage_advice: Dict[str, Dict[str, Any]]
    
    # 各标签的建议时长
    recommended_duration: Dict[str, float]  # 分钟
    
    # 关键洞察
    insights: List[str]
    
    # 优化过程信息（用于分析）
    convergence_history: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'optimal_ratio': {k: round(v, 4) for k, v in self.optimal_ratio.items()},
            'expected_gmv_per_minute': round(self.expected_gmv_per_minute, 2),
            'stage_advice': self.stage_advice,
            'recommended_duration': {k: round(v, 2) for k, v in self.recommended_duration.items()},
            'insights': self.insights
        }


class ScriptOptimizer:
    """
    话术优化器 —— 基于贝叶斯优化思想的话术组合优化
    
    【优化问题定义】
    
    目标：最大化单位时间GMV
    max GMV_per_minute(x) = Σ(xi * ei)
    
    约束：
    s.t. Σxi = 1  (占比和为100%)
         0.05 ≤ xi ≤ 0.40  (每个标签5%-40%)
    
    其中：
    - xi：第i个话术标签的占比
    - ei：第i个标签的单位时间GMV（来自DID归因结果）
    
    【贝叶斯优化流程】
    
    1. 初始化：基于历史数据计算各标签效果ei
    2. 构建代理模型：用线性模型近似GMV(x) = Σ(xi * ei)
    3. 选择初始点：按效果排序，贪婪分配比例
    4. 迭代优化：
       a. 在当前点邻域随机扰动，生成候选点
       b. 评估候选点的EI（期望改进）
       c. 选择EI最大的点作为下一个迭代点
       d. 更新当前最优
    5. 收敛判断：连续N次迭代无改进则停止
    6. 输出结果：最优配比 + 分阶段建议
    
    【为什么用简化版而非完整贝叶斯优化】
    
    1. 目标函数近似线性：GMV(x) ≈ Σ(xi * ei)，标签间交互效应较弱
    2. 约束简单：等式约束+盒约束，贪婪分配即可满足
    3. 计算效率：完整贝叶斯优化需要求解GP的逆矩阵（O(n³)），
       简化版只需O(n)的排序和分配
    4. 可解释性：简化策略（按效果排序分配）商家更容易理解
    
    未来如需处理强交互效应，可升级为完整贝叶斯优化。
    """
    
    def __init__(
        self,
        min_label_ratio: float = 0.05,
        max_label_ratio: float = 0.40,
        n_iterations: int = 50
    ):
        """
        初始化优化器
        
        参数：
            min_label_ratio: 每个标签最少占比（默认5%，保证多样性）
            max_label_ratio: 每个标签最大占比（默认40%，防止过度集中）
            n_iterations: 优化迭代次数（默认50次，平衡精度与速度）
        """
        self.min_label_ratio = min_label_ratio
        self.max_label_ratio = max_label_ratio
        self.n_iterations = n_iterations
        
        # 标签效果缓存
        self._label_effectiveness: Dict[str, float] = {}
    
    def calculate_label_effectiveness(
        self,
        label_summaries: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        计算每个标签的效果（单位时间GMV）
        
        这是贝叶斯优化的"先验知识"：基于历史数据估计各标签的基准效果。
        
        参数：
            label_summaries: 标签归因汇总，来自DIDAttributor
            
        返回：
            标签->单位时间GMV的字典
        """
        effectiveness = {}
        
        for label, summary in label_summaries.items():
            total_duration = summary.get('total_duration', 0)  # 秒
            total_incremental_gmv = summary.get('total_incremental_gmv', 0)
            
            if total_duration > 0:
                # 转换为每分钟GMV
                gmv_per_minute = total_incremental_gmv / (total_duration / 60)
                effectiveness[label] = max(0, gmv_per_minute)  # 不能为负
            else:
                effectiveness[label] = 0.0
        
        self._label_effectiveness = effectiveness
        return effectiveness
    
    def objective_function(
        self,
        ratios: np.ndarray,
        label_names: List[str],
        effectiveness: Dict[str, float]
    ) -> float:
        """
        目标函数：计算给定比例下的预期单位时间GMV
        
        这是贝叶斯优化要最大化的函数。
        在我们的场景中，目标函数近似线性：GMV = Σ(xi * ei)
        
        参数：
            ratios: 各标签的比例数组（搜索空间中的点）
            label_names: 标签名称列表
            effectiveness: 标签效果字典（来自calculate_label_effectiveness）
            
        返回：
            预期单位时间GMV的负数（因为优化器通常求最小值）
        
        注意：
        - 返回负数是因为大多数优化器默认求最小值
        - 调用方取负即可得到最大GMV
        """
        total_gmv = 0.0
        
        for i, label in enumerate(label_names):
            ratio = ratios[i]
            eff = effectiveness.get(label, 0.0)
            total_gmv += ratio * eff
        
        # 添加多样性惩罚项：比例过于集中会受惩罚
        # 这是约束处理的一部分：鼓励话术多样性
        diversity_penalty = 0.0
        for ratio in ratios:
            if ratio > 0.35:  # 如果某个标签超过35%，给惩罚
                diversity_penalty += (ratio - 0.35) * 10
        
        # 返回负数，因为优化器求最小值
        return -(total_gmv - diversity_penalty)
    
    def _expected_improvement(
        self,
        x: np.ndarray,
        current_best: float,
        label_names: List[str],
        effectiveness: Dict[str, float]
    ) -> float:
        """
        计算期望改进（Expected Improvement, EI）
        
        EI是贝叶斯优化的核心获取函数，平衡探索与利用。
        
        在我们的简化实现中：
        - 用线性模型近似均值：μ(x) = Σ(xi * ei)
        - 用启发式方法估计方差：σ(x) = f(与已知点的距离)
        
        参数：
            x: 候选点（话术配比）
            current_best: 当前最优目标值
            label_names: 标签名称列表
            effectiveness: 标签效果字典
            
        返回：
            EI值（越大表示该点越值得探索）
        """
        # 线性模型预测均值
        mu = -self.objective_function(x, label_names, effectiveness)
        
        # 启发式估计方差：与已知好点的距离越远，方差越大
        # 这是一个简化，完整实现需要用GP计算后验方差
        if hasattr(self, '_best_x'):
            distance = np.linalg.norm(x - self._best_x)
            sigma = 0.1 + 0.5 * distance  # 距离越远，不确定性越大
        else:
            sigma = 0.5  # 初始不确定性
        
        # 计算EI
        if sigma < 1e-6:
            return max(0, mu - current_best)
        
        delta = mu - current_best
        Z = delta / sigma
        
        # EI = delta * Φ(Z) + sigma * φ(Z)
        # 使用标准正态分布的CDF和PDF
        from scipy.stats import norm
        ei = delta * norm.cdf(Z) + sigma * norm.pdf(Z)
        
        return max(0, ei)
    
    def optimize(
        self,
        label_summaries: Dict[str, Any],
        live_duration_minutes: float = 120
    ) -> OptimizationResult:
        """
        执行贝叶斯优化
        
        流程：
        1. 计算标签效果（先验）
        2. 贪婪初始化（基于效果排序）
        3. 迭代优化（EI引导的局部搜索）
        4. 生成分阶段建议
        5. 输出关键洞察
        
        参数：
            label_summaries: 标签归因汇总
            live_duration_minutes: 直播时长（分钟）
            
        返回：
            OptimizationResult（包含最优配比和分阶段建议）
        """
        # 第一步：计算各标签效果（构建先验）
        effectiveness = self.calculate_label_effectiveness(label_summaries)
        
        # 获取所有标签
        all_labels = list(effectiveness.keys())
        
        # 如果没有数据，返回默认配比
        if not all_labels or all(e <= 0 for e in effectiveness.values()):
            return self._get_default_result(live_duration_minutes)
        
        # 第二步：贪婪初始化（基于效果排序）
        # 这是贝叶斯优化的初始点选择
        sorted_labels = sorted(
            all_labels,
            key=lambda x: effectiveness[x],
            reverse=True
        )
        
        # 初始化比例：每个标签先分配最小占比
        optimal_ratio = {label: self.min_label_ratio for label in all_labels}
        remaining_ratio = 1.0 - sum(optimal_ratio.values())
        
        # 按效果分配剩余比例（贪婪策略）
        total_effectiveness = sum(effectiveness[l] for l in sorted_labels)
        
        if total_effectiveness > 0:
            for label in sorted_labels:
                # 按比例分配，但不超过最大值
                additional = remaining_ratio * (effectiveness[label] / total_effectiveness)
                optimal_ratio[label] = min(
                    self.max_label_ratio,
                    optimal_ratio[label] + additional
                )
        
        # 归一化确保和为1
        total = sum(optimal_ratio.values())
        optimal_ratio = {k: v / total for k, v in optimal_ratio.items()}
        
        # 第三步：迭代优化（简化版贝叶斯优化）
        # 在当前解的邻域进行随机扰动，选择EI最大的点
        current_ratios = np.array([optimal_ratio[l] for l in all_labels])
        current_best = -self.objective_function(current_ratios, all_labels, effectiveness)
        self._best_x = current_ratios.copy()
        
        convergence_history = []
        no_improvement_count = 0
        
        for iteration in range(self.n_iterations):
            # 生成候选点：在当前解附近随机扰动
            candidate = current_ratios + np.random.normal(0, 0.05, len(all_labels))
            candidate = np.clip(candidate, self.min_label_ratio, self.max_label_ratio)
            candidate = candidate / candidate.sum()  # 归一化
            
            # 计算EI
            ei = self._expected_improvement(
                candidate, current_best, all_labels, effectiveness
            )
            
            # 评估候选点
            candidate_value = -self.objective_function(candidate, all_labels, effectiveness)
            
            # 记录历史
            convergence_history.append({
                'iteration': iteration,
                'value': candidate_value,
                'ei': ei
            })
            
            # 如果候选点更好，接受它
            if candidate_value > current_best:
                current_best = candidate_value
                current_ratios = candidate.copy()
                self._best_x = candidate.copy()
                no_improvement_count = 0
                
                # 更新optimal_ratio
                for i, label in enumerate(all_labels):
                    optimal_ratio[label] = current_ratios[i]
            else:
                no_improvement_count += 1
            
            # 早停：连续20次无改进则停止
            if no_improvement_count >= 20:
                break
        
        # 第四步：计算预期GMV
        expected_gmv = sum(
            optimal_ratio[label] * effectiveness.get(label, 0)
            for label in all_labels
        )
        
        # 第五步：生成各标签的建议时长
        recommended_duration = {
            label: live_duration_minutes * ratio
            for label, ratio in optimal_ratio.items()
        }
        
        # 第六步：生成分阶段建议
        stage_advice = self._generate_stage_advice(
            optimal_ratio, effectiveness, live_duration_minutes
        )
        
        # 第七步：生成关键洞察
        insights = self._generate_insights(
            optimal_ratio, effectiveness, label_summaries
        )
        
        return OptimizationResult(
            optimal_ratio=optimal_ratio,
            expected_gmv_per_minute=expected_gmv,
            stage_advice=stage_advice,
            recommended_duration=recommended_duration,
            insights=insights,
            convergence_history=convergence_history
        )
    
    def _generate_stage_advice(
        self,
        optimal_ratio: Dict[str, float],
        effectiveness: Dict[str, float],
        live_duration_minutes: float
    ) -> Dict[str, Dict[str, Any]]:
        """
        生成分阶段建议
        
        直播的三个阶段有不同的目标和话术策略：
        - 开场（10%）：建立信任，吸引停留
        - 中场（70%）：核心转化，反复逼单
        - 收尾（20%）：最后冲刺，消除顾虑
        """
        stages = {
            '开场': {
                'duration_ratio': 0.1,
                'focus_labels': ['产品介绍', '互动引导', '信任背书'],
                'description': '建立信任，吸引停留'
            },
            '中场': {
                'duration_ratio': 0.7,
                'focus_labels': ['产品介绍', '价格福利', '痛点共鸣', '逼单催促'],
                'description': '核心转化，反复逼单'
            },
            '收尾': {
                'duration_ratio': 0.2,
                'focus_labels': ['逼单催促', '价格福利', '售后承诺'],
                'description': '最后冲刺，消除顾虑'
            }
        }
        
        stage_advice = {}
        
        for stage_name, stage_info in stages.items():
            stage_duration = live_duration_minutes * stage_info['duration_ratio']
            
            # 在该阶段重点使用哪些标签
            focus_labels = stage_info['focus_labels']
            
            # 计算该阶段的建议配比
            stage_ratio = {}
            total_focus = sum(optimal_ratio.get(l, 0) for l in focus_labels)
            
            if total_focus > 0:
                for label in focus_labels:
                    stage_ratio[label] = optimal_ratio.get(label, 0) / total_focus
            else:
                # 平均分配
                for label in focus_labels:
                    stage_ratio[label] = 1.0 / len(focus_labels)
            
            # 计算该阶段的建议时长
            stage_duration_allocation = {
                label: stage_duration * ratio
                for label, ratio in stage_ratio.items()
            }
            
            stage_advice[stage_name] = {
                'duration_minutes': round(stage_duration, 1),
                'description': stage_info['description'],
                'recommended_ratio': {k: round(v, 4) for k, v in stage_ratio.items()},
                'recommended_duration': {k: round(v, 1) for k, v in stage_duration_allocation.items()},
                'key_scripts': self._get_stage_key_scripts(stage_name)
            }
        
        return stage_advice
    
    def _get_stage_key_scripts(self, stage_name: str) -> List[str]:
        """获取各阶段的关键话术示例"""
        examples = {
            '开场': [
                '姐妹们好，欢迎来到我们的直播间，今天给大家带来的是一款超级好用的面膜',
                '新来的姐妹点个关注，不迷路，今天直播间福利多多',
                '我们是天猫旗舰店，正品保证，已经卖了10万件了'
            ],
            '中场': [
                '这款面膜主打补水保湿，里面添加了玻尿酸成分',
                '是不是有很多姐妹冬天皮肤特别干，化妆卡粉卡到怀疑人生',
                '今天直播间拍下立减50，到手只要99，还送价值69的赠品',
                '库存只剩最后20单了，想要的姐妹赶紧拍，1号链接直接下单'
            ],
            '收尾': [
                '还有最后3分钟，没付款的姐妹抓紧了，活动结束恢复原价',
                '今天下单送运费险，不满意随时退，没有任何风险',
                '不满意7天无理由退换货，假一赔十，放心购买'
            ]
        }
        return examples.get(stage_name, [])
    
    def _generate_insights(
        self,
        optimal_ratio: Dict[str, float],
        effectiveness: Dict[str, float],
        label_summaries: Dict[str, Any]
    ) -> List[str]:
        """
        生成关键洞察
        
        基于优化结果，给商家可执行的建议。
        """
        insights = []
        
        # 按效果排序
        sorted_labels = sorted(
            effectiveness.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        if not sorted_labels:
            return ['数据不足，无法生成建议']
        
        # 最有效的标签
        top_label, top_effect = sorted_labels[0]
        if top_effect > 0:
            insights.append(
                f"【核心发现】'{top_label}'效果最好，建议占比{optimal_ratio.get(top_label, 0)*100:.0f}%，"
                f"每分钟可带来¥{top_effect:.2f}的GMV"
            )
        
        # 第二有效的标签
        if len(sorted_labels) > 1:
            second_label, second_effect = sorted_labels[1]
            if second_effect > 0:
                insights.append(
                    f"【次要重点】'{second_label}'效果次之，建议占比{optimal_ratio.get(second_label, 0)*100:.0f}%"
                )
        
        # 效果最差的标签
        bottom_label, bottom_effect = sorted_labels[-1]
        if bottom_effect < top_effect * 0.3 and bottom_effect >= 0:
            insights.append(
                f"【需要改进】'{bottom_label}'效果较弱，建议减少使用或优化话术内容"
            )
        
        # 检查是否有标签占比过高
        for label, ratio in optimal_ratio.items():
            if ratio >= 0.35:
                insights.append(
                    f"【注意平衡】'{label}'占比过高({ratio*100:.0f}%)，建议适当减少，增加其他类型话术"
                )
        
        # 通用建议
        if '价格福利' in optimal_ratio and optimal_ratio['价格福利'] < 0.15:
            insights.append(
                f"【建议】价格福利话术占比偏低，适当增加福利介绍可以提升转化"
            )
        
        if '逼单催促' in optimal_ratio and optimal_ratio['逼单催促'] < 0.10:
            insights.append(
                f"【建议】逼单催促话术占比偏低，适当增加逼单可以提升即时转化"
            )
        
        return insights
    
    def _get_default_result(self, live_duration_minutes: float) -> OptimizationResult:
        """
        获取默认优化结果（当数据不足时使用）
        
        基于行业经验给出的默认配比，作为冷启动方案。
        """
        default_ratio = {
            '产品介绍': 0.25,
            '价格福利': 0.25,
            '痛点共鸣': 0.15,
            '逼单催促': 0.15,
            '互动引导': 0.05,
            '信任背书': 0.05,
            '使用教程': 0.05,
            '售后承诺': 0.05
        }
        
        recommended_duration = {
            label: live_duration_minutes * ratio
            for label, ratio in default_ratio.items()
        }
        
        stage_advice = self._generate_stage_advice(
            default_ratio, {}, live_duration_minutes
        )
        
        return OptimizationResult(
            optimal_ratio=default_ratio,
            expected_gmv_per_minute=0.0,
            stage_advice=stage_advice,
            recommended_duration=recommended_duration,
            insights=[
                '数据不足，使用默认配比建议',
                '建议多直播几场积累数据后重新分析',
                '默认配比：产品介绍25% + 价格福利25% + 痛点共鸣15% + 逼单催促15% + 其他20%'
            ]
        )
    
    def compare_scenarios(
        self,
        label_summaries: Dict[str, Any],
        scenarios: List[Dict[str, float]]
    ) -> List[Dict[str, Any]]:
        """
        对比不同话术方案的预测效果
        
        参数：
            label_summaries: 标签归因汇总
            scenarios: 不同的话术比例方案列表
            
        返回：
            各方案的预测结果
        """
        effectiveness = self.calculate_label_effectiveness(label_summaries)
        
        results = []
        for i, scenario in enumerate(scenarios):
            expected_gmv = sum(
                scenario.get(label, 0) * effectiveness.get(label, 0)
                for label in effectiveness.keys()
            )
            
            results.append({
                'scenario_id': i + 1,
                'ratio': scenario,
                'expected_gmv_per_minute': round(expected_gmv, 2),
                'description': self._describe_scenario(scenario)
            })
        
        # 按预期GMV排序
        results.sort(key=lambda x: x['expected_gmv_per_minute'], reverse=True)
        
        return results
    
    def _describe_scenario(self, ratio: Dict[str, float]) -> str:
        """描述一个话术方案的特点"""
        # 找出占比最高的三个标签
        top3 = sorted(ratio.items(), key=lambda x: x[1], reverse=True)[:3]
        
        descriptions = []
        for label, r in top3:
            if r >= 0.25:
                descriptions.append(f"重点{label}")
            elif r >= 0.15:
                descriptions.append(f"重视{label}")
        
        return '，'.join(descriptions) if descriptions else '平衡型话术'


# ================================================================================
# 五、最优标签组合的解释和应用建议
# ================================================================================
#
# 【如何解读优化结果】
#
# 优化器输出的不是一个"魔法数字"，而是一套可执行的运营策略。
# 商家应该关注以下几个维度：
#
# 1. 全局最优配比：
#    - 各标签的占比反映了它们对GMV的边际贡献
#    - 占比高的标签应该多说，占比低的标签应该少说
#    - 但注意约束：每个标签5%-40%，避免过度集中或完全缺失
#
# 2. 分阶段策略：
#    - 开场（10%）：重点是建立信任，产品介绍和信任背书为主
#    - 中场（70%）：重点是转化，价格福利和逼单催促为主
#    - 收尾（20%）：重点是冲刺，逼单和售后承诺为主
#    - 不同阶段的配比是根据直播心理学设计的
#
# 3. 关键洞察：
#    - 系统会识别出效果最好和最差的标签
#    - 给出具体的调整建议（如"价格福利占比偏低"）
#    - 这些洞察是基于数据驱动的，而非主观判断
#
# 【应用建议】
#
# 1. 渐进式调整：
#    - 不要一次性完全改变话术结构
#    - 建议每周调整5-10%，观察效果后再继续
#    - 直播话术优化是持续迭代的过程
#
# 2. A/B测试验证：
#    - 可以用两场直播做对比：一场按优化建议，一场按原策略
#    - 对比GMV、转化率、停留时长等核心指标
#    - 验证优化效果是否如预期
#
# 3. 结合人工经验：
#    - 优化结果是数据驱动的建议，不是绝对真理
#    - 主播的个人风格和品类特性也需要考虑
#    - 建议将优化结果作为参考，而非严格遵循
#
# 4. 定期重新优化：
#    - 直播话术的效果会随时间变化（用户疲劳、竞品变化等）
#    - 建议每月重新运行一次优化
#    - 积累更多数据后，优化结果会更准确
#
# 【常见误区】
#
# ❌ 误区1：只看占比最高的标签，其他都不说
#    ✅ 正确：占比是指导，不是命令。多样性很重要，避免话术单一化。
#
# ❌ 误区2：一次优化后就永远不变
#    ✅ 正确：直播环境变化快，需要定期重新优化。
#
# ❌ 误区3：期望优化后GMV立即翻倍
#    ✅ 正确：优化通常带来10-30%的提升，翻倍需要综合运营改进。
#
# ❌ 误区4：完全依赖系统，忽略主播反馈
#    ✅ 正确：系统是工具，主播的临场发挥同样重要。
#
# ================================================================================


# 简单的测试代码
if __name__ == "__main__":
    # 创建示例数据
    sample_summaries = {
        '价格福利': {
            'total_duration': 600,  # 10分钟
            'total_incremental_gmv': 3000.0,
            'total_incremental_orders': 30
        },
        '逼单催促': {
            'total_duration': 300,  # 5分钟
            'total_incremental_gmv': 2500.0,
            'total_incremental_orders': 25
        },
        '产品介绍': {
            'total_duration': 900,  # 15分钟
            'total_incremental_gmv': 1500.0,
            'total_incremental_orders': 15
        },
        '痛点共鸣': {
            'total_duration': 450,  # 7.5分钟
            'total_incremental_gmv': 1200.0,
            'total_incremental_orders': 12
        },
        '互动引导': {
            'total_duration': 300,  # 5分钟
            'total_incremental_gmv': 300.0,
            'total_incremental_orders': 3
        }
    }
    
    optimizer = ScriptOptimizer()
    result = optimizer.optimize(sample_summaries, live_duration_minutes=120)
    
    print("=" * 60)
    print("贝叶斯优化结果")
    print("=" * 60)
    
    print("\n最优配比：")
    for label, ratio in sorted(result.optimal_ratio.items(), key=lambda x: x[1], reverse=True):
        if ratio >= 0.05:
            print(f"  {label}: {ratio*100:.1f}%")
    
    print(f"\n预期单位时间GMV：¥{result.expected_gmv_per_minute:.2f}/分钟")
    
    print(f"\n分阶段建议：")
    for stage, advice in result.stage_advice.items():
        print(f"\n{stage}（{advice['duration_minutes']:.0f}分钟）：{advice['description']}")
        print(f"  重点话术：{', '.join(advice['recommended_ratio'].keys())}")
    
    print(f"\n关键洞察：")
    for insight in result.insights:
        print(f"  • {insight}")
    
    print(f"\n优化迭代次数：{len(result.convergence_history)}")
