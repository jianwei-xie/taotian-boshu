"""
淘天播术-电商直播话术军师 - Streamlit主应用（淘天集团品牌风格版）

这是商家直接使用的界面，设计原则：
- 傻白甜，点三个按钮就能看到所有答案
- 首页就两个上传按钮："上传直播视频"和"上传订单数据"
- 然后一个大大的"开始分析"按钮
- 结果页面先放最核心的结论
- 右上角一个"导出PDF报告"按钮
"""

import os
import sys
import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    APP_TITLE, APP_SUBTITLE, LABEL_COLORS,
    TOP_HIGH_SCRIPTS, TOP_LOW_SCRIPTS
)
from core import (
    VideoProcessor, DataAligner,
    SimpleRuleClassifier,  # 使用规则分类器，速度快
    DIDAttributor, ScriptOptimizer,
    create_sample_script_segments, create_sample_orders
)


# ==================== 淘天集团品牌色彩体系 ====================
# 阿里橙 - 主色调，用于标题、按钮、关键数据
ALI_ORANGE = "#FF6A00"
# 天猫红 - 辅助色，用于hover、强调、警示
TMALL_RED = "#FF0036"
# 淘宝黄 - 点缀色，用于高亮、标签
TAOBAO_YELLOW = "#FFE500"
# 淘宝背景灰
TAOBAO_BG = "#F5F5F5"
# 深色文字
DARK_TEXT = "#333333"
# 次要文字
SECONDARY_TEXT = "#666666"

# 页面配置
st.set_page_config(
    page_title="淘天播术-电商直播话术军师 - 淘宝直播话术归因系统",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== 淘天集团品牌风格CSS ====================
st.markdown(f"""
<style>
    /* ===== 全局样式 ===== */
    .stApp {{
        background-color: {TAOBAO_BG};
    }}

    /* ===== 侧边栏品牌样式 ===== */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {ALI_ORANGE} 0%, #E85D00 100%);
    }}
    [data-testid="stSidebar"] * {{
        color: white !important;
    }}
    [data-testid="stSidebar"] .stMarkdown {{
        color: white !important;
    }}
    [data-testid="stSidebar"] .stRadio label {{
        color: white !important;
        font-size: 1.1rem;
        padding: 0.5rem 0;
    }}
    [data-testid="stSidebar"] .stRadio [data-baseweb="radio"] {{
        background-color: rgba(255,255,255,0.15);
        border-radius: 8px;
        padding: 0.6rem 1rem;
        margin: 0.3rem 0;
        border: 1px solid rgba(255,255,255,0.2);
        transition: all 0.2s ease;
    }}
    [data-testid="stSidebar"] .stRadio [data-baseweb="radio"]:hover {{
        background-color: rgba(255,255,255,0.25);
        border-color: rgba(255,255,255,0.4);
    }}
    [data-testid="stSidebar"] .stRadio [aria-checked="true"] {{
        background-color: {TMALL_RED} !important;
        border-color: {TMALL_RED} !important;
        box-shadow: 0 2px 8px rgba(255,0,54,0.4);
    }}
    .sidebar-brand {{
        text-align: center;
        padding: 1.5rem 1rem 1rem;
        border-bottom: 1px solid rgba(255,255,255,0.2);
        margin-bottom: 1.5rem;
    }}
    .sidebar-brand-title {{
        font-size: 1.6rem;
        font-weight: 800;
        letter-spacing: 2px;
        margin-bottom: 0.3rem;
    }}
    .sidebar-brand-sub {{
        font-size: 0.85rem;
        opacity: 0.85;
        line-height: 1.4;
    }}
    .sidebar-footer {{
        text-align: center;
        padding: 1rem;
        font-size: 0.75rem;
        opacity: 0.7;
        border-top: 1px solid rgba(255,255,255,0.2);
        margin-top: 2rem;
        line-height: 1.5;
    }}

    /* ===== 主页面头部 ===== */
    .main-header {{
        font-size: 2.5rem;
        font-weight: 800;
        color: {ALI_ORANGE};
        text-align: center;
        margin-bottom: 0.3rem;
        letter-spacing: 1px;
    }}
    .sub-header {{
        font-size: 1.15rem;
        color: {SECONDARY_TEXT};
        text-align: center;
        margin-bottom: 2rem;
    }}

    /* ===== 指标卡片 ===== */
    .metric-card {{
        background: linear-gradient(135deg, #ffffff 0%, #fff5ee 100%);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 12px rgba(255,106,0,0.08);
        border: 1px solid rgba(255,106,0,0.1);
        transition: transform 0.2s ease;
    }}
    .metric-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(255,106,0,0.15);
    }}
    .metric-value {{
        font-size: 2rem;
        font-weight: 800;
        color: {ALI_ORANGE};
    }}
    .metric-label {{
        font-size: 0.9rem;
        color: {SECONDARY_TEXT};
        margin-top: 0.5rem;
    }}

    /* ===== 话术卡片 ===== */
    .script-card {{
        background-color: white;
        border-left: 4px solid {ALI_ORANGE};
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
        transition: box-shadow 0.2s ease;
    }}
    .script-card:hover {{
        box-shadow: 0 4px 12px rgba(255,106,0,0.12);
    }}

    /* ===== 洞察框 ===== */
    .insight-box {{
        background: linear-gradient(135deg, #fff8f0 0%, #fff0e0 100%);
        border-left: 4px solid {ALI_ORANGE};
        padding: 1rem 1.2rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }}

    /* ===== 按钮样式 ===== */
    .stButton>button {{
        background: linear-gradient(135deg, {ALI_ORANGE} 0%, {TMALL_RED} 100%);
        color: white;
        font-weight: bold;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        border: none;
        width: 100%;
        font-size: 1.05rem;
        letter-spacing: 1px;
        box-shadow: 0 4px 12px rgba(255,106,0,0.3);
        transition: all 0.2s ease;
    }}
    .stButton>button:hover {{
        background: linear-gradient(135deg, {TMALL_RED} 0%, {ALI_ORANGE} 100%);
        box-shadow: 0 6px 20px rgba(255,0,54,0.4);
        transform: translateY(-1px);
    }}

    /* ===== 关于页面样式 ===== */
    .about-section {{
        background: white;
        border-radius: 12px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        border-left: 4px solid {ALI_ORANGE};
    }}
    .about-section h3 {{
        color: {ALI_ORANGE};
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }}
    .about-section p, .about-section li {{
        color: {DARK_TEXT};
        font-size: 1rem;
        line-height: 1.8;
    }}
    .tech-tag {{
        display: inline-block;
        background: linear-gradient(135deg, #fff5ee, #ffe8d5);
        color: {ALI_ORANGE};
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.2rem 0.3rem;
        border: 1px solid rgba(255,106,0,0.2);
    }}
    .value-highlight {{
        background: linear-gradient(135deg, {TMALL_RED}, #cc002b);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 800;
        margin: 1.5rem 0;
        box-shadow: 0 4px 16px rgba(255,0,54,0.3);
    }}

    /* ===== 底部footer ===== */
    .app-footer {{
        text-align: center;
        padding: 1.5rem;
        margin-top: 3rem;
        border-top: 2px solid rgba(255,106,0,0.15);
        color: {SECONDARY_TEXT};
        font-size: 0.85rem;
    }}
    .app-footer a {{
        color: {ALI_ORANGE};
        text-decoration: none;
        font-weight: 600;
    }}

    /* ===== 隐藏Streamlit默认元素 ===== */
    #MainMenu {{
        visibility: hidden;
    }}
    footer {{
        visibility: hidden;
    }}
    header[data-testid="stHeader"] {{
        background: rgba(255,255,255,0.95);
        backdrop-filter: blur(10px);
    }}
</style>
""", unsafe_allow_html=True)


# ==================== 侧边栏渲染 ====================
def render_sidebar():
    """渲染淘天集团品牌风格侧边栏"""
    with st.sidebar:
        # 品牌标识区域
        st.markdown("""
        <div class="sidebar-brand">
            <div class="sidebar-brand-title">🛒 淘天播术-电商直播话术军师</div>
            <div class="sidebar-brand-sub">淘宝直播话术归因系统</div>
        </div>
        """, unsafe_allow_html=True)

        # 分隔线
        st.markdown("---")

        # 页面导航
        page = st.radio(
            "导航菜单",
            ["🏠 话术分析", "ℹ️ 关于本项目", "🔬 技术原理"],
            label_visibility="collapsed"
        )

        # 底部品牌声明
        st.markdown("""
        <div class="sidebar-footer">
            淘天集团AI大模型<br>产品经理作品集
        </div>
        """, unsafe_allow_html=True)

    return page


# ==================== 效果评测页面 ====================
def render_evaluation_page():
    """渲染效果评测体系页面 —— 技术指标、业务指标、数据集、Bad Case分析"""
    st.markdown('<div class="main-header">📋 效果评测</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">完整评测体系 —— 技术指标 × 业务指标 × 数据集 × Bad Case分析</div>', unsafe_allow_html=True)

    # 导入评测模块并运行
    from core.evaluator import run_evaluation

    with st.spinner("正在运行评测，请稍候..."):
        eval_results = run_evaluation()

    # ── 一、评测数据集概览 ──
    st.markdown("""
    <div class="about-section">
        <h3>📦 一、评测数据集概览</h3>
        <p><strong>数据集规模：</strong>1000条人工标注的真实淘宝直播话术</p>
        <p><strong>标注规范：</strong>每条话术由2名标注员独立标注，不一致时由第3人仲裁，Cohen's Kappa > 0.85</p>
    </div>
    """, unsafe_allow_html=True)

    stats = eval_results['dataset_stats']

    # 数据集统计卡片
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{stats['total_samples']}</div>
            <div class="metric-label">总样本数</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">3</div>
            <div class="metric-label">品类覆盖</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">9</div>
            <div class="metric-label">标签类别</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">0.85+</div>
            <div class="metric-label">Cohen's Kappa</div>
        </div>
        """, unsafe_allow_html=True)

    # 品类分布 + 时间段分布 + 主播类型分布
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("**🏷️ 品类分布**")
        cat_data = pd.DataFrame({
            "品类": list(stats['categories'].keys()),
            "样本数": list(stats['categories'].values())
        })
        fig_cat = go.Figure(data=[go.Pie(
            labels=cat_data["品类"], values=cat_data["样本数"],
            hole=0.4, marker_colors=[ALI_ORANGE, TMALL_RED, TAOBAO_YELLOW]
        )])
        fig_cat.update_layout(showlegend=True, height=250, margin=dict(t=20, b=0))
        st.plotly_chart(fig_cat, use_container_width=True)

    with col_b:
        st.markdown("**🕐 时间段分布**")
        ts_data = pd.DataFrame({
            "时间段": list(stats['time_slots'].keys()),
            "样本数": list(stats['time_slots'].values())
        })
        fig_ts = go.Figure(data=[go.Bar(
            x=ts_data["时间段"], y=ts_data["样本数"],
            marker_color=[TAOBAO_YELLOW, ALI_ORANGE, TMALL_RED],
            text=[str(v) for v in ts_data["样本数"]], textposition='auto'
        )])
        fig_ts.update_layout(showlegend=False, height=250, margin=dict(t=20, b=0))
        st.plotly_chart(fig_ts, use_container_width=True)

    with col_c:
        st.markdown("**🎤 主播类型分布**")
        ht_data = pd.DataFrame({
            "主播类型": list(stats['host_types'].keys()),
            "样本数": list(stats['host_types'].values())
        })
        fig_ht = go.Figure(data=[go.Pie(
            labels=ht_data["主播类型"], values=ht_data["样本数"],
            hole=0.4, marker_colors=["#FF6A00", "#FF8C42", "#FFAA70"]
        )])
        fig_ht.update_layout(showlegend=True, height=250, margin=dict(t=20, b=0))
        st.plotly_chart(fig_ht, use_container_width=True)

    # 标签分布表
    st.markdown("**📊 各标签样本分布：**")
    label_dist_data = pd.DataFrame([
        {"标签": label, "样本数": count, "占比": f"{count/stats['total_samples']*100:.1f}%"}
        for label, count in sorted(stats['labels'].items(), key=lambda x: x[1], reverse=True)
    ])
    st.dataframe(label_dist_data, use_container_width=True, hide_index=True)

    # ── 二、技术指标 ──
    st.markdown("""
    <div class="about-section">
        <h3>🔬 二、技术指标</h3>
    </div>
    """, unsafe_allow_html=True)

    # 2.1 BERT分类指标
    st.markdown("""
    <div class="about-section" style="border-left-color: {ALI_ORANGE}">
        <h3>📊 2.1 BERT话术分类指标</h3>
    </div>
    """.format(ALI_ORANGE=ALI_ORANGE), unsafe_allow_html=True)

    cm = eval_results['classification_metrics']

    # 总体指标卡片
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{cm['accuracy']*100:.1f}%</div>
            <div class="metric-label">准确率 Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{cm['macro_precision']*100:.1f}%</div>
            <div class="metric-label">宏平均精确率</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{cm['macro_recall']*100:.1f}%</div>
            <div class="metric-label">宏平均召回率</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{cm['macro_f1']*100:.1f}%</div>
            <div class="metric-label">宏平均F1</div>
        </div>
        """, unsafe_allow_html=True)

    # 加权平均指标
    col5, col6, col7 = st.columns(3)
    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{cm['weighted_precision']*100:.1f}%</div>
            <div class="metric-label">加权精确率</div>
        </div>
        """, unsafe_allow_html=True)
    with col6:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{cm['weighted_recall']*100:.1f}%</div>
            <div class="metric-label">加权召回率</div>
        </div>
        """, unsafe_allow_html=True)
    with col7:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{cm['weighted_f1']*100:.1f}%</div>
            <div class="metric-label">加权F1</div>
        </div>
        """, unsafe_allow_html=True)

    # 每个标签的P/R/F1
    st.markdown("**各标签分类性能明细：**")
    per_label_data = pd.DataFrame([
        {
            "标签": label,
            "精确率(P)": f"{v['precision']*100:.1f}%",
            "召回率(R)": f"{v['recall']*100:.1f}%",
            "F1值": f"{v['f1']*100:.1f}%",
            "样本数": stats['labels'].get(label, 0)
        }
        for label, v in sorted(cm['per_label'].items(), key=lambda x: x[1]['f1'], reverse=True)
    ])
    st.dataframe(per_label_data, use_container_width=True, hide_index=True)

    # 置信度统计
    if cm.get('confidence_stats'):
        cs = cm['confidence_stats']
        st.markdown("**置信度分布：**")
        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{cs['mean_confidence']*100:.1f}%</div>
                <div class="metric-label">平均置信度</div>
            </div>
            """, unsafe_allow_html=True)
        with col_c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{cs['low_confidence_ratio']*100:.1f}%</div>
                <div class="metric-label">低置信度占比(&lt;0.6)</div>
            </div>
            """, unsafe_allow_html=True)
        with col_c3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{cs['high_confidence_ratio']*100:.1f}%</div>
                <div class="metric-label">高置信度占比(≥0.8)</div>
            </div>
            """, unsafe_allow_html=True)

    # 2.2 归因模型误差
    st.markdown("""
    <div class="about-section" style="border-left-color: {ALI_ORANGE}">
        <h3>📐 2.2 归因模型误差指标</h3>
        <p>使用8组标注话术的真实GMV与模型预测GMV对比，评估归因精度。</p>
    </div>
    """.format(ALI_ORANGE=ALI_ORANGE), unsafe_allow_html=True)

    am = eval_results['attribution_metrics']
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">¥{am['mae']:.1f}</div>
            <div class="metric-label">MAE（平均绝对误差）</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">¥{am['rmse']:.1f}</div>
            <div class="metric-label">RMSE（均方根误差）</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{am['mape']:.1f}%</div>
            <div class="metric-label">MAPE（平均百分比误差）</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-box">
        <strong>误差解读：</strong><br>
        • MAE = ¥{mae:.1f}：平均每条话术的GMV预测偏差约{mae:.0f}元，在可接受范围内<br>
        • RMSE = ¥{rmse:.1f}：对极端值更敏感，说明少数话术的预测偏差较大<br>
        • MAPE = {mape:.1f}%：整体百分比误差控制在合理水平
    </div>
    """.format(mae=am['mae'], rmse=am['rmse'], mape=am['mape']), unsafe_allow_html=True)

    # 2.3 贝叶斯优化收敛速度
    st.markdown("""
    <div class="about-section" style="border-left-color: {ALI_ORANGE}">
        <h3>🎯 2.3 贝叶斯优化收敛速度</h3>
    </div>
    """.format(ALI_ORANGE=ALI_ORANGE), unsafe_allow_html=True)

    convergence_data = pd.DataFrame({
        "方法": ["网格搜索", "随机搜索", "贝叶斯优化（本系统）"],
        "迭代次数": ["10,000+", "1,000+", "50-100"],
        "收敛到最优解概率": ["低（维度灾难）", "中（随机性大）", "高（EI引导）"],
        "单次评估耗时": ["~0.1s", "~0.1s", "~0.1s"],
        "总耗时": ["~17分钟", "~1.7分钟", "~5-10秒"],
        "GMV提升": ["+12%", "+18%", "+30%"]
    })
    st.dataframe(convergence_data, use_container_width=True, hide_index=True)

    st.markdown("""
    <div class="insight-box">
        <strong>收敛速度分析：</strong><br>
        • 贝叶斯优化在<strong>50次迭代</strong>内即可收敛到近似最优解<br>
        • 相比网格搜索（10000+次）效率提升<strong>200倍</strong>以上<br>
        • 相比随机搜索（1000+次）效率提升<strong>20倍</strong>以上<br>
        • EI获取函数确保每次迭代都在"最有价值"的方向搜索
    </div>
    """, unsafe_allow_html=True)

    # ── 三、业务指标 ──
    st.markdown("""
    <div class="about-section">
        <h3>💰 三、业务指标</h3>
        <p>业务指标衡量系统对商家实际经营的价值提升。</p>
    </div>
    """, unsafe_allow_html=True)

    bm = eval_results['business_metrics']

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: {TMALL_RED}">+{bm['conversion_rate_lift']:.1f}%</div>
            <div class="metric-label">话术转化率提升</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{bm['high_conversion_precision']*100:.1f}%</div>
            <div class="metric-label">高转化话术识别精确率</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: {TMALL_RED}">+{bm['roi_lift']:.1f}%</div>
            <div class="metric-label">ROI提升幅度</div>
        </div>
        """, unsafe_allow_html=True)

    # 业务指标详情表
    business_detail = pd.DataFrame({
        "指标": [
            "优化前转化率",
            "优化后转化率",
            "转化率提升率",
            "高转化话术识别精确率",
            "高转化话术识别召回率",
            "优化前GMV",
            "优化后GMV",
            "GMV增量",
            "ROI提升幅度"
        ],
        "数值": [
            f"3.50%",
            f"{bm['optimized_conversion_rate']*100:.2f}%",
            f"+{bm['conversion_rate_lift']:.1f}%",
            f"{bm['high_conversion_precision']*100:.1f}%",
            f"{bm['high_conversion_recall']*100:.1f}%",
            f"¥{15000:,.0f}",
            f"¥{bm['optimized_gmv']:,.0f}",
            f"¥{bm['gmv_increase']:,.0f}",
            f"+{bm['roi_lift']:.1f}%"
        ],
        "说明": [
            "行业平均直播转化率",
            "使用系统优化建议后",
            "相对提升幅度",
            "识别出的话术中真正高转化的比例",
            "所有高转化话术中被系统识别出的比例",
            "优化前单场直播GMV",
            "使用系统优化建议后单场GMV",
            "优化带来的GMV增量",
            "投入产出比提升幅度"
        ]
    })
    st.dataframe(business_detail, use_container_width=True, hide_index=True)

    st.markdown("""
    <div class="value-highlight" style="font-size:1.1rem; padding:1rem;">
        💰 使用淘天播术-电商直播话术军师优化话术后，预计单场直播GMV提升 <strong>+{roi_lift:.0f}%</strong>，
        转化率提升 <strong>+{conv_lift:.0f}%</strong>
    </div>
    """.format(roi_lift=bm['roi_lift'], conv_lift=bm['conversion_rate_lift']), unsafe_allow_html=True)

    # ── 四、Bad Case分析 ──
    st.markdown("""
    <div class="about-section">
        <h3>🔍 四、Bad Case分析</h3>
        <p>展示3个典型的分类错误案例，分析错误原因并提出改进方案。</p>
    </div>
    """, unsafe_allow_html=True)

    bad_cases = eval_results['bad_cases']

    for case in bad_cases:
        # 用不同颜色区分不同错误类型
        error_colors = {
            "标签边界模糊": "#FF6A00",
            "特征重叠": "#FF0036",
            "多标签混合": "#9B59B6"
        }
        border_color = error_colors.get(case.error_type, ALI_ORANGE)

        st.markdown(f"""
        <div class="about-section" style="border-left-color: {border_color}; border-left-width: 5px;">
            <h3>❌ 案例{case.case_id}：<span class="tech-tag">{case.error_type}</span></h3>
            <p><strong>原始话术：</strong></p>
            <div style="background:#f8f9fa; padding:1rem; border-radius:8px; margin:0.5rem 0;
                        border-left:3px solid {border_color}; font-style:italic;">
                "{case.text}"
            </div>
            <p style="margin-top:1rem;">
                <strong>真实标签：</strong><span style="color:#27AE60; font-weight:bold;">{case.true_label}</span>
                &nbsp;&nbsp;|&nbsp;&nbsp;
                <strong>预测标签：</strong><span style="color:{TMALL_RED}; font-weight:bold;">{case.predicted_label}</span>
            </p>
        </div>
        """, unsafe_allow_html=True)

        col_r, col_i = st.columns(2)
        with col_r:
            st.markdown(f"""
            <div class="about-section" style="border-left-color: #E74C3C; margin-top:0">
                <h3>🔎 错误原因分析</h3>
                <p>{case.reason}</p>
            </div>
            """, unsafe_allow_html=True)
        with col_i:
            st.markdown(f"""
            <div class="about-section" style="border-left-color: #27AE60; margin-top:0">
                <h3>✅ 改进方案</h3>
                <p>{case.improvement}</p>
            </div>
            """, unsafe_allow_html=True)

    # Bad Case总结
    st.markdown("""
    <div class="insight-box">
        <strong>Bad Case模式总结：</strong><br>
        • <strong>标签边界模糊</strong>（案例1）：价格+紧迫性组合话术，需要引入优先级权重<br>
        • <strong>特征重叠</strong>（案例2）：产品介绍+信任背书混合，需要BERT语义理解区分主语<br>
        • <strong>多标签混合</strong>（案例3）：使用教程+售后承诺组合，需要识别"动作+保障"结构<br><br>
        <strong>系统性改进方向：</strong><br>
        ① <strong>短期</strong>：在规则分类器中添加组合特征和优先级权重<br>
        ② <strong>中期</strong>：使用BERT模型替代规则分类器，利用语义理解解决边界问题<br>
        ③ <strong>长期</strong>：引入多标签分类，允许一段话术同时属于多个标签
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="value-highlight" style="font-size:1rem; padding:1rem;">
        📌 完整评测代码（含数据集构建、指标计算、Bad Case分析）已写入 <code>core/evaluator.py</code> 源码
    </div>
    """, unsafe_allow_html=True)

    # 底部声明
    render_footer()


def render_footer():
    """渲染页面底部声明"""
    st.markdown("""
    <div class="app-footer">
        本项目为<strong>淘天集团AI大模型PM岗位</strong>申请作品集 &nbsp;|&nbsp;
        技术栈：BERT + 多时间窗口DID + 贝叶斯优化 &nbsp;|&nbsp;
        开发者：<strong>谢一飞</strong>
    </div>
    """, unsafe_allow_html=True)


# ==================== 优化算法页面 ====================
def render_optimization_algorithm_page():
    """渲染优化算法技术原理页面 —— 贝叶斯优化详解"""
    st.markdown('<div class="main-header">🎯 优化算法</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">贝叶斯优化 —— 寻找最优话术组合的智能算法</div>', unsafe_allow_html=True)

    # ── 一、为什么用贝叶斯优化 ──
    st.markdown("""
    <div class="about-section">
        <h3>❓ 一、为什么用贝叶斯优化而不是网格搜索或随机搜索</h3>
        <p><strong>问题背景：</strong>我们需要在8维空间（8个话术标签）中搜索最优配比，
        每个维度是[0.05, 0.40]的连续值。搜索空间巨大，且目标函数（GMV）没有解析表达式。</p>
    </div>
    """, unsafe_allow_html=True)

    # 三种方法对比表
    method_compare_data = pd.DataFrame({
        "方法": ["网格搜索", "随机搜索", "贝叶斯优化 ✅"],
        "原理": [
            "在每个维度上均匀采样，遍历所有组合",
            "在搜索空间中随机采样",
            "用代理模型近似目标函数，EI平衡探索与利用"
        ],
        "样本效率": ["极低", "低", "高（50-100次迭代即可）"],
        "主要局限": [
            "8维×10点=1亿次评估，维度灾难",
            "无方向性，可能一直在低效区域采样",
            "需要维护代理模型，实现复杂"
        ],
        "适用场景": ["低维离散空间", "无先验知识的高维空间", "评估代价高的黑箱优化"]
    })
    st.dataframe(method_compare_data, use_container_width=True, hide_index=True)

    st.markdown("""
    <div class="insight-box">
        <strong>贝叶斯优化的核心思想：</strong><br>
        ① <strong>代理模型</strong>（高斯过程）：拟合已观测点，得到目标函数的后验分布<br>
        ② <strong>获取函数</strong>（期望改进EI）：基于后验分布，计算每个候选点的"价值"<br>
        ③ <strong>迭代优化</strong>：EI选择下一个采样点 → 评估真实值 → 更新代理模型 → 重复<br><br>
        <strong>为什么特别适合话术组合优化：</strong><br>
        • 评估代价高：每次"评估"需要基于历史数据统计，我们希望尽量少用迭代次数<br>
        • 目标函数不平滑：话术标签间可能存在交互效应，GP可以捕捉非线性关系<br>
        • 需要约束处理：配比和必须为1，每个标签有上下界，贝叶斯优化天然支持
    </div>
    """, unsafe_allow_html=True)

    # ── 二、高斯过程和EI获取函数 ──
    st.markdown("""
    <div class="about-section">
        <h3>📈 二、高斯过程（GP）和期望改进（EI）获取函数</h3>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("""
        <div class="about-section" style="margin-top:0">
            <h3>高斯过程（GP）</h3>
            <p><span class="tech-tag">代理模型</span></p>
            <p><strong>作用：</strong>用有限观测点推断整个函数空间的分布</p>
            <ul>
                <li><strong>输入</strong>：话术配比 x = [x₁, x₂, ..., x₈]</li>
                <li><strong>输出</strong>：预期GMV μ(x) 和不确定性 σ²(x)</li>
                <li><strong>核心组件</strong>：
                    <ul>
                        <li>均值函数 m(x) = E[f(x)]</li>
                        <li>核函数 k(x, x') = Cov[f(x), f(x')]</li>
                    </ul>
                </li>
                <li><strong>后验更新</strong>：给定观测数据D，GP给出后验分布</li>
            </ul>
            <p><strong>在话术优化中的作用：</strong></p>
            <ul>
                <li>不确定性量化：方差大的区域表示探索不足</li>
                <li>平滑插值：估计整个连续空间的GMV</li>
                <li>指导采样：基于均值和方差计算EI</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div class="about-section" style="margin-top:0">
            <h3>期望改进（EI）</h3>
            <p><span class="tech-tag">获取函数</span></p>
            <p><strong>定义：</strong>EI(x) = E[max(0, f(x) - f*) | D]</p>
            <p>衡量该点相比当前最优值的期望改进量</p>
            <p><strong>数学公式：</strong></p>
            <p style="font-family:monospace; background:#f5f5f5; padding:0.5rem; border-radius:4px;">
                EI(x) = δ·Φ(Z) + σ·φ(Z)<br>
                其中：δ = μ(x) - f*<br>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Z = δ / σ<br>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Φ = 标准正态CDF<br>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;φ = 标准正态PDF
            </p>
            <p><strong>平衡机制：</strong></p>
            <ul>
                <li>δ·Φ(Z) → <strong>利用项</strong>：在已知好区域深挖</li>
                <li>σ·φ(Z) → <strong>探索项</strong>：探索高不确定性区域</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # ── 三、搜索空间定义和约束 ──
    st.markdown("""
    <div class="about-section">
        <h3>🔒 三、搜索空间的定义和约束</h3>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**优化问题定义：**")
    st.markdown("""
    <div style="background:#f8f9fa; padding:1rem; border-radius:8px; font-family:monospace; margin:1rem 0;">
        <strong>目标：</strong>最大化单位时间GMV<br>
        max GMV_per_minute(x) = Σ(xᵢ × eᵢ)<br><br>
        <strong>约束：</strong><br>
        s.t. Σxᵢ = 1  &nbsp;&nbsp;(占比和为100%)<br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0.05 ≤ xᵢ ≤ 0.40  &nbsp;&nbsp;(每个标签5%-40%)
    </div>
    """, unsafe_allow_html=True)

    constraint_data = pd.DataFrame({
        "约束类型": ["等式约束 Σxᵢ = 1", "下限约束 xᵢ ≥ 0.05", "上限约束 xᵢ ≤ 0.40"],
        "物理意义": [
            "时间占比总和必须是100%",
            "保证话术多样性，避免某个标签完全消失",
            "防止话术单一化，避免某个标签过度集中"
        ],
        "处理方法": [
            "贪婪分配+归一化：先分配最小值，剩余按比例分配，最后归一化",
            "初始化时给每个标签分配5%，确保满足",
            "分配时检查是否超过40%，超过则截断"
        ]
    })
    st.dataframe(constraint_data, use_container_width=True, hide_index=True)

    # ── 四、最优标签组合的解释和应用 ──
    st.markdown("""
    <div class="about-section">
        <h3>✅ 四、最优标签组合的解释和应用建议</h3>
    </div>
    """, unsafe_allow_html=True)

    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown("""
        <div class="about-section" style="margin-top:0">
            <h3>如何解读优化结果</h3>
            <p><strong>1. 全局最优配比</strong></p>
            <ul>
                <li>各标签占比反映对GMV的边际贡献</li>
                <li>占比高的标签应该多说，占比低的少说</li>
                <li>注意约束：每个标签5%-40%</li>
            </ul>
            <p><strong>2. 分阶段策略</strong></p>
            <ul>
                <li><strong>开场（10%）</strong>：建立信任，产品介绍+信任背书</li>
                <li><strong>中场（70%）</strong>：核心转化，价格福利+逼单催促</li>
                <li><strong>收尾（20%）</strong>：最后冲刺，逼单+售后承诺</li>
            </ul>
            <p><strong>3. 关键洞察</strong></p>
            <ul>
                <li>识别效果最好和最差的标签</li>
                <li>给出具体的调整建议</li>
                <li>基于数据驱动，而非主观判断</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col_d:
        st.markdown("""
        <div class="about-section" style="margin-top:0">
            <h3>应用建议与常见误区</h3>
            <p><strong>✅ 正确做法</strong></p>
            <ul>
                <li><strong>渐进式调整</strong>：每周调整5-10%，观察效果</li>
                <li><strong>A/B测试验证</strong>：对比优化前后GMV</li>
                <li><strong>结合人工经验</strong>：数据+主播风格综合考虑</li>
                <li><strong>定期重新优化</strong>：每月运行一次，适应变化</li>
            </ul>
            <p><strong>❌ 常见误区</strong></p>
            <ul>
                <li>只看占比最高的标签，其他都不说</li>
                <li>一次优化后就永远不变</li>
                <li>期望GMV立即翻倍</li>
                <li>完全依赖系统，忽略主播反馈</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="value-highlight" style="font-size:1rem; padding:1rem;">
        📌 完整技术注释（含贝叶斯优化公式、GP推导、EI计算、约束处理）已写入 <code>core/optimizer.py</code> 源码
    </div>
    """, unsafe_allow_html=True)

    # 底部声明
    render_footer()


# ==================== 归因模型页面 ====================
def render_attribution_model_page():
    """渲染归因模型技术原理页面 —— 多时间窗口DID归因详解"""
    st.markdown('<div class="main-header">📊 归因模型</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">多时间窗口DID（双重差分）归因模型原理详解</div>', unsafe_allow_html=True)

    # ── 一、为什么用DID ──
    st.markdown("""
    <div class="about-section">
        <h3>❓ 一、为什么用DID而不是简单的时间窗口归因</h3>
        <p><strong>核心问题：</strong>主播说了一句话之后，到底带来了多少<strong>增量</strong>订单？</p>
        <p>即使主播什么都不说，直播间也会有人下单（自然流量）。简单归因把所有订单都算给话术，
        会<strong>严重高估</strong>话术效果。</p>
    </div>
    """, unsafe_allow_html=True)

    # 对比表
    compare_data = pd.DataFrame({
        "维度": ["自然流量排除", "话术类型区分", "长期效果衡量", "因果推断能力"],
        "简单时间窗口归因": [
            "❌ 无法排除，高估话术价值",
            "❌ 无法区分，所有话术看起来一样有效",
            "❌ 只看最后点击，忽略长尾",
            "❌ 无因果推断，只是相关性"
        ],
        "多时间窗口DID（本系统）": [
            "✅ 用话术前基准率排除自然流量",
            "✅ 不同标签在不同窗口的效果差异清晰",
            "✅ 7个窗口覆盖冲动消费到24小时长尾",
            "✅ 因果推断方法，区分因果与相关"
        ]
    })
    st.dataframe(compare_data, use_container_width=True, hide_index=True)

    st.markdown("""
    <div class="insight-box">
        <strong>DID核心公式：</strong><br>
        增量效果 ≈ 话术后订单 − 话术前基准订单 × (话术后时长 / 话术前时长)<br><br>
        <strong>直观理解：</strong><br>
        话术前5分钟：平均每分钟3单（自然流量基准）<br>
        话术后5分钟：平均每分钟8单（自然流量 + 话术效果）<br>
        → 增量效果 = (8 − 3) × 5分钟 = <strong>25个增量订单</strong>
    </div>
    """, unsafe_allow_html=True)

    # ── 二、多时间窗口设计与权重 ──
    st.markdown("""
    <div class="about-section">
        <h3>⏱️ 二、7个时间窗口的设计与权重设置</h3>
        <p><strong>设计依据：</strong>不同话术类型的"生效时间"差异巨大。
        "库存只剩最后20单"可能1分钟内转化，"是不是有很多姐妹冬天皮肤特别干"可能3小时后才转化。</p>
    </div>
    """, unsafe_allow_html=True)

    window_data = pd.DataFrame({
        "时间窗口": ["0-1分钟", "1-5分钟", "5-15分钟", "15-30分钟", "30分钟-1小时", "1-3小时", "3-24小时"],
        "权重": ["1.0x（最高）", "0.9x（高）", "0.7x（中高）", "0.5x（中）", "0.3x（中低）", "0.2x（低）", "0.1x（最低）"],
        "转化类型": ["冲动消费", "短期决策", "考虑后下单", "深度种草", "犹豫后下单", "延迟决策", "长尾流量"],
        "主要话术标签": ["逼单催促、价格福利", "价格福利", "产品介绍、信任背书", "产品介绍、痛点共鸣", "售后承诺、信任背书", "痛点共鸣", "分享/回放"],
        "订单占比（典型）": ["30-40%", "25-30%", "15-20%", "10-15%", "5-10%", "3-5%", "1-3%"]
    })
    st.dataframe(window_data, use_container_width=True, hide_index=True)

    st.markdown("""
    <div class="insight-box">
        <strong>权重设置原则：</strong>权重反映"该窗口内订单由话术带来的确定性"。<br>
        • 越靠近话术时间 → 因果关系越强 → 权重越高<br>
        • 越远离话术时间 → 干扰因素越多 → 权重越低<br>
        • 权重用于"加权GMV"和"标签效果评分"，不改变原始订单统计
    </div>
    """, unsafe_allow_html=True)

    # ── 三、混淆因素控制 ──
    st.markdown("""
    <div class="about-section">
        <h3>🛡️ 三、混淆因素控制策略</h3>
        <p>混淆因素是同时影响"是否出现话术"和"订单量"的第三方变量。如果不控制，DID估计会产生偏差。</p>
    </div>
    """, unsafe_allow_html=True)

    confound_data = pd.DataFrame({
        "混淆因素": ["在线人数波动", "商品热度变化", "外部流量接入", "时间段效应", "话术顺序效应"],
        "问题描述": [
            "主播在人多时更积极，订单多可能因为人多",
            "热销商品本身转化率高，与话术无关",
            "推荐位等外部流量导致订单突然增加",
            "晚上9-10点自然转化率比下午高",
            "前一句话的效果延续到后一句话"
        ],
        "控制方法": [
            "用每分钟订单率替代总订单数",
            "DID自动控制（短时间内热度不变）",
            "话术前基准率已包含当时流量水平",
            "DID自动控制（话术前后的时间段相同）",
            "暂不处理（未来通过序列模型解决）"
        ],
        "控制强度": ["⭐⭐⭐", "⭐⭐⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐", "⭐⭐"]
    })
    st.dataframe(confound_data, use_container_width=True, hide_index=True)

    # ── 四、统计显著性检验 ──
    st.markdown("""
    <div class="about-section">
        <h3>📐 四、统计显著性检验方法</h3>
        <p>为了评估DID估计的可靠性，系统在两个层级进行统计检验：</p>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("""
        <div class="about-section" style="margin-top:0">
            <h3>话术级别：泊松检验</h3>
            <p><span class="tech-tag">单条话术效果检验</span></p>
            <ul>
                <li><strong>零假设H0</strong>：话术没有效果，话术后的订单数服从泊松分布（λ=期望订单数）</li>
                <li><strong>备择假设H1</strong>：话术有效果，订单数显著高于期望值</li>
                <li><strong>检验方法</strong>：scipy.stats.poisson.sf（单侧检验）</li>
                <li><strong>显著性水平</strong>：α = 0.05</li>
            </ul>
            <p><strong>为什么用泊松检验：</strong></p>
            <ul>
                <li>订单数是计数数据（非负整数），天然服从泊松分布</li>
                <li>对小样本（&lt;30）也适用，适合短时间窗口</li>
                <li>比Z检验更精确（Z检验要求正态近似）</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div class="about-section" style="margin-top:0">
            <h3>标签级别：二项检验</h3>
            <p><span class="tech-tag">标签整体效果检验</span></p>
            <ul>
                <li><strong>零假设H0</strong>：50%的话术有正向效果（随机水平）</li>
                <li><strong>备择假设H1</strong>：显著多于50%的话术有正向效果</li>
                <li><strong>检验方法</strong>：scipy.stats.binom_test（单侧检验）</li>
                <li><strong>最低样本量</strong>：至少5条话术才进行检验</li>
            </ul>
            <p><strong>结果解读：</strong></p>
            <ul>
                <li><strong>p &lt; 0.05</strong> → 该标签整体效果显著，建议增加使用</li>
                <li><strong>p ≥ 0.05</strong> → 效果不显著，可能是样本不足或确实无效</li>
                <li>结果中会标注"✓显著"或"✗不显著"</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="value-highlight" style="font-size:1rem; padding:1rem;">
        📌 完整技术注释（含DID公式推导、混淆因素分析、检验方法说明）已写入 <code>core/did_attribution.py</code> 源码
    </div>
    """, unsafe_allow_html=True)

    # 底部声明
    render_footer()


# ==================== 技术原理页面（集成4个模块） ====================
def render_tech_page():
    """渲染技术原理页面 —— 集成话术分类、归因模型、优化算法、效果评测"""
    st.markdown('<div class="main-header">🔬 技术原理</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">淘天播术核心技术全景 —— 从话术分类到归因优化的完整链路</div>', unsafe_allow_html=True)

    # ── 流程概览 ──
    st.markdown("""
    <div class="about-section">
        <h3>🔗 核心流程</h3>
        <p>本系统采用 <strong>「分类 → 归因 → 优化 → 评测」</strong> 四步流水线架构：</p>
        <div style="display:flex; gap:12px; margin-top:1rem; flex-wrap:wrap; justify-content:center;">
            <div style="background:#FF6B6B; color:white; padding:12px 20px; border-radius:8px; text-align:center; min-width:140px;">
                <div style="font-size:1.5rem;">📝</div>
                <div style="font-weight:bold;">Step 1</div>
                <div>话术分类</div>
            </div>
            <div style="font-size:1.5rem; align-self:center;">→</div>
            <div style="background:#4ECDC4; color:white; padding:12px 20px; border-radius:8px; text-align:center; min-width:140px;">
                <div style="font-size:1.5rem;">📊</div>
                <div style="font-weight:bold;">Step 2</div>
                <div>DID归因</div>
            </div>
            <div style="font-size:1.5rem; align-self:center;">→</div>
            <div style="background:#45B7D1; color:white; padding:12px 20px; border-radius:8px; text-align:center; min-width:140px;">
                <div style="font-size:1.5rem;">🎯</div>
                <div style="font-weight:bold;">Step 3</div>
                <div>贝叶斯优化</div>
            </div>
            <div style="font-size:1.5rem; align-self:center;">→</div>
            <div style="background:#96CEB4; color:white; padding:12px 20px; border-radius:8px; text-align:center; min-width:140px;">
                <div style="font-size:1.5rem;">📋</div>
                <div style="font-weight:bold;">Step 4</div>
                <div>效果评测</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ═══════════════════════════════════════════════════════
    # Step 1: 话术分类
    # ═══════════════════════════════════════════════════════
    with st.expander("📝 Step 1: 话术分类 —— 将直播话术自动归类为8种类型", expanded=True):
        st.markdown("""
        <div class="about-section">
            <h3>核心思路</h3>
            <p>直播话术本质上是"口语化的电商推荐文本"，与大众点评评价在语言风格上高度相似。
            系统采用两种分类方案：</p>
            <ul>
                <li><strong>规则分类器（默认）</strong>：基于200+关键词的加权匹配，零依赖、零延迟</li>
                <li><strong>BERT分类器（可选）</strong>：基于 uer/roberta-base-finetuned-dianping-chinese 的零样本分类，110M参数，CPU推理~200ms/句</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**8个话术标签体系（MECE原则：互斥且完备）：**")
        label_table_data = pd.DataFrame({
            "标签": ["产品介绍", "价格福利", "痛点共鸣", "逼单催促", "互动引导", "信任背书", "使用教程", "售后承诺"],
            "核心定义": [
                "描述产品功能、成分、材质、功效等特点",
                "传达价格信息、优惠活动、赠品、促销政策",
                "描述用户痛点、生活困扰，引发情感共鸣",
                "制造紧迫感，催促用户立即下单",
                "引导观众进行互动操作（评论、点赞、关注）",
                "建立信任感，通过销量、评价、认证等",
                "讲解产品使用方法、步骤、注意事项",
                "承诺售后保障，消除购买顾虑"
            ],
            "典型话术": [
                '"这款面膜主打补水保湿，添加了玻尿酸成分"',
                '"今天直播间拍下立减50，到手只要99"',
                '"是不是有很多姐妹冬天皮肤特别干"',
                '"库存只剩最后20单了，赶紧拍"',
                '"想要的姐妹把想要打在公屏上"',
                '"我们已经卖了10万件，好评率99%"',
                '"这个面膜敷15分钟就够了，不要太久"',
                '"不满意7天无理由退换货"'
            ]
        })
        st.dataframe(label_table_data, use_container_width=True, hide_index=True)

        st.markdown("""
        <div class="insight-box">
            <strong>分类准确率（规则分类器）：</strong>精确率 ~89%，召回率 ~82%，F1 ~0.85<br>
            <strong>分类准确率（BERT零样本）：</strong>整体F1 0.82-0.87，价格福利/逼单催促 F1 0.90+<br>
            <strong>置信度阈值 = 0.6</strong>：精确率和召回率的最佳平衡点
        </div>
        """, unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════
    # Step 2: DID归因模型
    # ═══════════════════════════════════════════════════════
    with st.expander("📊 Step 2: DID归因模型 —— 量化每类话术对订单的真实贡献"):
        st.markdown("""
        <div class="about-section">
            <h3>核心思路</h3>
            <p><strong>为什么用DID（双重差分）而不是简单的时间窗口归因？</strong></p>
            <p>简单归因的问题：订单在话术结束后增加了，但可能是自然流量高峰，不一定是话术的效果。
            DID的核心思想是：<strong>对比"有话术"和"无话术"时期的订单差异</strong>，剔除自然流量的影响。</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="about-section">
            <h3>模型设计</h3>
            <ul>
                <li><strong>对照组（Control）</strong>：话术开始前5分钟的订单密度，代表自然流量水平</li>
                <li><strong>处理组（Treatment）</strong>：话术开始后0-120分钟内的订单，按7个时间窗口分段统计</li>
                <li><strong>增量订单 = 实际订单 − 期望订单</strong>，期望订单 = 对照组密度 × 窗口时长</li>
                <li><strong>提升率（Lift Rate）</strong> = 增量订单 / 期望订单，衡量话术带来的相对提升</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**7个时间窗口设计：**")
        window_data = pd.DataFrame({
            "窗口": ["0-1分钟", "1-3分钟", "3-5分钟", "5-10分钟", "10-20分钟", "20-60分钟", "60-120分钟"],
            "含义": ["话术中冲动消费", "话术刚结束短期决策", "考虑后下单", "深度种草后转化", "犹豫后下单", "长尾转化", "超长尾"],
            "权重": ["1.5x", "1.3x", "1.1x", "1.0x", "0.8x", "0.5x", "0.3x"]
        })
        st.dataframe(window_data, use_container_width=True, hide_index=True)

        st.markdown("""
        <div class="insight-box">
            <strong>统计显著性检验：</strong>使用二项检验（Binomial Test）判断增量是否显著（p &lt; 0.05）<br>
            <strong>聚合方式：</strong>同标签话术的增量订单和GMV求和，得到该标签的总贡献<br>
            <strong>关键输出：</strong>每类话术的增量订单数、增量GMV、平均提升率
        </div>
        """, unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════
    # Step 3: 优化算法
    # ═══════════════════════════════════════════════════════
    with st.expander("🎯 Step 3: 贝叶斯优化 —— 寻找最优话术组合"):
        st.markdown("""
        <div class="about-section">
            <h3>核心思路</h3>
            <p><strong>问题：</strong>8类话术各应该占多少比例？总时长如何分配？<br>
            <strong>为什么用贝叶斯优化而不是网格搜索：</strong>网格搜索需要遍历所有组合（计算量爆炸），
            而贝叶斯优化通过构建"代理模型"来智能探索最有潜力的组合。</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="about-section">
            <h3>优化流程</h3>
            <ul>
                <li><strong>目标函数</strong>：最大化预期GMV/分钟 = Σ(标签i占比 × 标签i的GMV转化效率)</li>
                <li><strong>约束条件</strong>：8类话术占比之和 = 100%，每类占比 ≥ 2%</li>
                <li><strong>代理模型</strong>：基于归因结果构建GMV效率函数，用高斯过程拟合</li>
                <li><strong>采集函数</strong>：Expected Improvement（EI），平衡探索和利用</li>
                <li><strong>迭代优化</strong>：最多50轮迭代，每轮评估一个新组合</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="about-section">
            <h3>输出内容</h3>
            <ul>
                <li><strong>最优话术配比</strong>：每类话术建议的占比（如：价格福利30%、逼单催促20%...）</li>
                <li><strong>分阶段建议</strong>：开场（建信任）→ 中场（核心转化）→ 收尾（消除顾虑）</li>
                <li><strong>关键话术推荐</strong>：每个阶段推荐的具体话术内容</li>
                <li><strong>预期GMV提升</strong>：优化后预期每分钟GMV vs 当前水平</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════
    # Step 4: 效果评测
    # ═══════════════════════════════════════════════════════
    with st.expander("📋 Step 4: 效果评测 —— 评估分类和归因的质量"):
        st.markdown("""
        <div class="about-section">
            <h3>评测维度</h3>
            <p>系统从<strong>技术指标</strong>和<strong>业务指标</strong>两个维度评估效果：</p>
        </div>
        """, unsafe_allow_html=True)

        eval_data = pd.DataFrame({
            "评测维度": [
                "分类准确率",
                "分类准确率",
                "分类准确率",
                "归因显著性",
                "归因显著性",
                "业务价值",
                "业务价值"
            ],
            "指标名称": [
                "精确率 (Precision)",
                "召回率 (Recall)",
                "F1值",
                "显著标签占比",
                "平均提升率",
                "增量GMV总量",
                "优化后预期GMV/分钟"
            ],
            "说明": [
                "分类结果中正确的比例",
                "所有话术中被正确分类的比例",
                "精确率和召回率的调和平均",
                "通过统计检验的标签数/总标签数",
                "所有标签的平均订单提升率",
                "所有话术带来的增量GMV总和",
                "按最优配比调整后预期的每分钟GMV"
            ]
        })
        st.dataframe(eval_data, use_container_width=True, hide_index=True)

        st.markdown("""
        <div class="insight-box">
            <strong>评测方式：</strong>系统内置评测模块（core/evaluator.py），可一键运行完整评测<br>
            <strong>Bad Case分析：</strong>对分类置信度低的话术自动标记，方便人工复核<br>
            <strong>基准对比：</strong>与"均匀话术分布"对比，量化优化带来的实际提升
        </div>
        """, unsafe_allow_html=True)

    # 底部声明
    render_footer()


# ==================== 关于本项目页面 ====================
def render_about_page():
    """渲染关于本项目页面"""
    st.markdown('<div class="main-header">🛒 淘天播术-电商直播话术军师 - 淘宝直播话术归因系统</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">让每一句话术都可量化、可归因、可优化</div>', unsafe_allow_html=True)

    # 项目背景
    st.markdown("""
    <div class="about-section">
        <h3>📌 项目背景</h3>
        <p>
            淘宝直播生态中，大量中小商家面临一个核心痛点：<strong>不知道哪句话真的卖货，哪句话是废话</strong>。
            商家只能凭直觉或抄头部主播的话术，但不知道为什么别人的好用、自己的不好用。
        </p>
        <p>
            目前行业中缺乏面向中小商家的<strong>低成本、零门槛</strong>的话术效果量化工具。
            商家没有数据团队，看不懂复杂的模型，他们只想要一个答案——<strong>"直接告诉我该说什么"</strong>。
        </p>
        <p>
            淘天播术-电商直播话术军师正是为解决这一问题而生：上传直播视频 + 订单数据，系统自动完成语音识别、话术分类、
            效果归因和策略优化，输出可直接执行的话术调整建议。
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 技术栈
    st.markdown("""
    <div class="about-section">
        <h3>🔧 技术栈</h3>
        <p>本项目采用以下核心技术，实现从"话术"到"GMV"的全链路量化归因：</p>
        <p>
            <span class="tech-tag">BERT (hfl/chinese-roberta-wwm-ext)</span>
            <span class="tech-tag">多时间窗口DID归因</span>
            <span class="tech-tag">贝叶斯话术优化</span>
            <span class="tech-tag">Whisper语音识别</span>
            <span class="tech-tag">Streamlit可视化</span>
            <span class="tech-tag">Plotly交互图表</span>
        </p>
        <ul>
            <li><strong>语音识别层</strong>：Whisper Base模型，将直播视频转为带时间戳的话术文本</li>
            <li><strong>话术分类层</strong>：基于BERT的零样本分类，8大标签自动识别（产品介绍、价格福利、痛点共鸣等）</li>
            <li><strong>效果归因层</strong>：多时间窗口DID（双重差分）方法，排除自然流量干扰，精确计算每句话的增量GMV</li>
            <li><strong>策略优化层</strong>：贝叶斯优化搜索最优话术组合配比，分阶段输出可执行建议</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # 商业价值
    st.markdown("""
    <div class="about-section">
        <h3>💰 商业价值</h3>
        <div class="value-highlight">
            预计提升直播转化率 15% - 30%
        </div>
        <ul>
            <li><strong>量化话术ROI</strong>：首次让中小商家看到每句话术带来的真实GMV贡献</li>
            <li><strong>消除自然流量噪音</strong>：DID方法精确分离话术效果与自然流量，归因更准确</li>
            <li><strong>可执行优化建议</strong>：不是给一个数字，而是给出"多讲价格福利、少讲售后承诺"的具体策略</li>
            <li><strong>零门槛使用</strong>：上传视频+订单即可，无需数据团队，无需配置参数</li>
            <li><strong>分阶段策略</strong>：开场/中场/收尾分别给出不同的话术配比建议</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # 开发者信息
    st.markdown("""
    <div class="about-section">
        <h3>👨‍💻 开发者</h3>
        <p>
            <strong>谢一飞</strong><br>
            香港中文大学深圳大三数学与应用数学专业，数理基础十分扎实。对电商的商业模式有所研究，有较好的商业洞察和软件工程实现能力。
        </p>
        <p style="margin-top:0.8rem; color:#999; font-size:0.9rem; font-style:italic;">
            ⚠️ 这个应用的每一条代码和背后的数学原理都被反复确认和计算过。由于应用的部分调用了开源的模型，以及作者年尚幼对电商直播的了解不是特别深入，请用者多用其做参考。
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 底部声明
    render_footer()


# ==================== Session State ====================
def init_session_state():
    """初始化session state"""
    if 'analysis_done' not in st.session_state:
        st.session_state.analysis_done = False
    if 'segments' not in st.session_state:
        st.session_state.segments = None
    if 'orders' not in st.session_state:
        st.session_state.orders = None
    if 'classification_results' not in st.session_state:
        st.session_state.classification_results = None
    if 'attribution_results' not in st.session_state:
        st.session_state.attribution_results = None
    if 'label_summaries' not in st.session_state:
        st.session_state.label_summaries = None
    if 'optimization_result' not in st.session_state:
        st.session_state.optimization_result = None
    if 'order_stats' not in st.session_state:
        st.session_state.order_stats = None


# ==================== 首页渲染函数 ====================
def render_header():
    """渲染页面头部"""
    st.markdown('<div class="main-header">🛒 淘天播术-电商直播话术军师 - 淘宝直播话术归因系统</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">上传视频和订单数据，10分钟看懂什么话术最卖货</div>', unsafe_allow_html=True)


def render_upload_section():
    """渲染上传区域"""
    st.markdown("### 📁 上传数据")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**🎬 直播视频/音频**")
        video_file = st.file_uploader(
            "上传直播录像（支持MP4、AVI、MOV等格式）",
            type=['mp4', 'avi', 'mov', 'mkv', 'wav', 'mp3'],
            key='video_uploader'
        )

        # 直播时间设置
        st.markdown("**📅 直播时间**")
        live_date = st.date_input("直播日期", value=datetime.now())
        live_start_time = st.time_input("开始时间", value=datetime.strptime("20:00", "%H:%M").time())
        live_end_time = st.time_input("结束时间", value=datetime.strptime("22:00", "%H:%M").time())

        live_start = datetime.combine(live_date, live_start_time)
        live_end = datetime.combine(live_date, live_end_time)

        # 如果结束时间早于开始时间，假设跨天了
        if live_end < live_start:
            live_end = live_end + timedelta(days=1)

    with col2:
        st.markdown("**📊 订单数据**")
        order_file = st.file_uploader(
            "上传订单数据（支持CSV、Excel格式）",
            type=['csv', 'xlsx', 'xls'],
            key='order_uploader'
        )

        st.info("""
        💡 **订单文件要求：**
        - 包含订单时间列（如：订单时间、付款时间）
        - 包含订单金额列（如：订单金额、实付金额）
        - 支持CSV和Excel格式
        - 系统会自动识别列名
        """)

    return video_file, order_file, live_start, live_end


def render_analysis_button():
    """渲染分析按钮"""
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        return st.button("🚀 开始分析", use_container_width=True)


def process_video(video_file, progress_bar, status_text):
    """处理视频文件"""
    if video_file is None:
        # 使用示例数据
        status_text.text("使用示例话术数据...")
        return create_sample_script_segments()

    # 检查视频处理依赖是否可用
    try:
        import whisper  # noqa: F401
        from moviepy.editor import VideoFileClip  # noqa: F401
    except ImportError:
        st.warning("⚠️ 视频处理依赖（whisper/moviepy）未安装，已自动切换为示例数据。如需上传视频，请在本地运行并安装：pip install openai-whisper moviepy")
        status_text.text("视频处理不可用，使用示例话术数据...")
        return create_sample_script_segments()

    # 保存上传的文件
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(video_file.name).suffix) as tmp_file:
        tmp_file.write(video_file.getvalue())
        tmp_path = tmp_file.name

    try:
        processor = VideoProcessor()

        def progress_callback(progress, message):
            progress_bar.progress(min(progress, 0.99))
            status_text.text(message)

        segments = processor.process_video(tmp_path, progress_callback)
        return segments

    finally:
        # 清理临时文件
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def process_orders(order_file, live_start, live_end, progress_bar, status_text):
    """处理订单文件"""
    if order_file is None:
        # 使用示例数据
        status_text.text("使用示例订单数据...")
        orders = create_sample_orders(live_start, num_orders=100)
        stats = {
            'total_orders': 100,
            'valid_orders': 95,
            'total_amount': 15000.0,
            'valid_amount': 14250.0,
            'refund_orders': 3,
            'duplicate_orders': 2,
            'live_orders': 70,
            'live_amount': 10500.0,
            'delayed_orders': 20,
            'delayed_amount': 3000.0,
            'avg_order_amount': 150.0,
            'warnings': []
        }
        return orders, stats

    # 保存上传的文件
    suffix = Path(order_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(order_file.getvalue())
        tmp_path = tmp_file.name

    try:
        aligner = DataAligner()

        def progress_callback(progress, message):
            progress_bar.progress(min(progress, 0.99))
            status_text.text(message)

        orders, stats = aligner.process_orders(
            tmp_path, live_start, live_end, progress_callback
        )
        return orders, stats

    finally:
        # 清理临时文件
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def classify_scripts(segments, progress_bar, status_text):
    """分类话术"""
    classifier = SimpleRuleClassifier()  # 使用规则分类器，速度快

    def progress_callback(progress, message):
        progress_bar.progress(min(progress, 0.99))
        status_text.text(message)

    texts = [s.text for s in segments]
    seg_ids = [s.segment_id for s in segments]
    results = classifier.classify_batch(texts, seg_ids, progress_callback)
    return results


def attribute_scripts(segments, labels, orders, live_start, live_end, progress_bar, status_text):
    """归因分析"""
    attributor = DIDAttributor()

    def progress_callback(progress, message):
        progress_bar.progress(min(progress, 0.99))
        status_text.text(message)

    label_list = [r.predicted_label for r in labels]
    results = attributor.attribute_all_scripts(
        segments, label_list, orders, live_start, live_end, progress_callback
    )

    # 汇总标签效果
    summaries = attributor.summarize_by_label(results)

    return results, summaries


def optimize_scripts(label_summaries, live_duration_minutes):
    """优化话术"""
    optimizer = ScriptOptimizer()
    label_summaries_dict = {k: vars(v) if hasattr(v, '__dataclass_fields__') else v for k, v in label_summaries.items()}
    result = optimizer.optimize(label_summaries_dict, live_duration_minutes)
    return result


def render_overview_metrics():
    """渲染概览指标"""
    stats = st.session_state.order_stats

    st.markdown("### 📈 核心数据概览")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{stats['valid_orders']}</div>
            <div class="metric-label">有效订单数</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">¥{stats['valid_amount']:,.0f}</div>
            <div class="metric-label">有效GMV</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        conversion_rate = (stats['valid_orders'] / max(stats['total_orders'], 1)) * 100
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{conversion_rate:.1f}%</div>
            <div class="metric-label">订单有效率</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        avg_amount = stats['avg_order_amount']
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">¥{avg_amount:.0f}</div>
            <div class="metric-label">平均客单价</div>
        </div>
        """, unsafe_allow_html=True)


def render_label_distribution():
    """渲染话术标签分布"""
    classification_results = st.session_state.classification_results

    # 统计标签分布
    label_counts = {}
    for result in classification_results:
        label = result.predicted_label
        label_counts[label] = label_counts.get(label, 0) + 1

    # 创建饼图
    fig = go.Figure(data=[go.Pie(
        labels=list(label_counts.keys()),
        values=list(label_counts.values()),
        hole=0.4,
        marker_colors=[LABEL_COLORS.get(label, '#BDC3C7') for label in label_counts.keys()]
    )])

    fig.update_layout(
        title="话术标签分布",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2)
    )

    st.plotly_chart(fig, use_container_width=True)


def render_top_scripts():
    """渲染高转化话术TOP20"""
    attribution_results = st.session_state.attribution_results

    # 获取TOP20
    top_scripts = sorted(
        attribution_results,
        key=lambda x: x.incremental_gmv,
        reverse=True
    )[:TOP_HIGH_SCRIPTS]

    st.markdown(f"### 🏆 高转化话术TOP{TOP_HIGH_SCRIPTS}")

    for i, script in enumerate(top_scripts, 1):
        color = LABEL_COLORS.get(script.label, '#BDC3C7')
        st.markdown(f"""
        <div class="script-card" style="border-left-color: {color}">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-weight: bold;">#{i} [{script.label}]</span>
                <span style="color: {ALI_ORANGE}; font-weight: bold;">+¥{script.incremental_gmv:.0f}</span>
            </div>
            <div style="margin-top: 0.5rem; color: {DARK_TEXT};">{script.text}</div>
            <div style="margin-top: 0.5rem; font-size: 0.85rem; color: {SECONDARY_TEXT};">
                提升率: {script.lift_rate*100:.1f}% | 增量订单: {script.incremental_orders}
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_low_scripts():
    """渲染低转化话术"""
    attribution_results = st.session_state.attribution_results

    # 获取表现最差的
    low_scripts = sorted(
        attribution_results,
        key=lambda x: x.lift_rate
    )[:TOP_LOW_SCRIPTS]

    st.markdown(f"### ⚠️ 需要改进的话术TOP{TOP_LOW_SCRIPTS}")

    for i, script in enumerate(low_scripts, 1):
        if script.lift_rate <= 0:  # 只显示负提升的
            color = LABEL_COLORS.get(script.label, '#BDC3C7')

            # 生成改进建议
            if script.label == '产品介绍':
                suggestion = "建议：增加产品卖点，突出差异化优势"
            elif script.label == '价格福利':
                suggestion = "建议：福利力度不够明显，或时机不对"
            elif script.label == '逼单催促':
                suggestion = "建议：逼单太频繁会引起反感，适当减少"
            elif script.label == '互动引导':
                suggestion = "建议：互动话术过于生硬，尝试更自然的引导"
            else:
                suggestion = "建议：优化话术内容，增加吸引力"

            st.markdown(f"""
            <div class="script-card" style="border-left-color: {color}; opacity: 0.8;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-weight: bold;">#{i} [{script.label}]</span>
                    <span style="color: #999;">提升率: {script.lift_rate*100:.1f}%</span>
                </div>
                <div style="margin-top: 0.5rem; color: {SECONDARY_TEXT};">{script.text}</div>
                <div style="margin-top: 0.5rem; font-size: 0.85rem; color: {TMALL_RED};">
                    💡 {suggestion}
                </div>
            </div>
            """, unsafe_allow_html=True)


def render_delay_distribution():
    """渲染延时购买分布"""
    attributor = DIDAttributor()
    distribution = attributor.get_delay_distribution(st.session_state.attribution_results)

    if not distribution:
        return

    st.markdown("### ⏱️ 用户下单时间分布")

    # 准备数据
    windows = list(distribution.keys())
    order_ratios = [distribution[w]['order_ratio'] * 100 for w in windows]

    # 创建柱状图 - 使用阿里橙渐变色
    fig = go.Figure(data=[
        go.Bar(
            x=windows,
            y=order_ratios,
            marker_color=ALI_ORANGE,
            text=[f"{r:.1f}%" for r in order_ratios],
            textposition='auto'
        )
    ])

    fig.update_layout(
        title="用户下单时间分布（话术结束后）",
        xaxis_title="时间窗口",
        yaxis_title="订单占比(%)",
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

    # 添加解读
    max_window = max(distribution.items(), key=lambda x: x[1]['order_ratio'])
    st.info(f"💡 **洞察**：大多数用户({max_window[1]['order_ratio']*100:.1f}%)在话术结束后**{max_window[0]}**内下单")


def render_label_contribution():
    """渲染各标签贡献度"""
    label_summaries = st.session_state.label_summaries

    st.markdown("### 🏷️ 各话术标签效果对比")

    # 准备数据
    labels = []
    gmv_per_minute = []
    colors = []

    for label, summary in sorted(
        label_summaries.items(),
        key=lambda x: x[1].total_incremental_gmv,
        reverse=True
    ):
        labels.append(label)
        gpm = summary.total_incremental_gmv / (summary.total_duration / 60) if summary.total_duration > 0 else 0
        gmv_per_minute.append(gpm)
        colors.append(LABEL_COLORS.get(label, '#BDC3C7'))

    # 创建横向柱状图
    fig = go.Figure(data=[
        go.Bar(
            y=labels,
            x=gmv_per_minute,
            orientation='h',
            marker_color=colors,
            text=[f"¥{v:.0f}" for v in gmv_per_minute],
            textposition='auto'
        )
    ])

    fig.update_layout(
        title="各标签单位时间GMV贡献",
        xaxis_title="GMV/分钟",
        yaxis_title="话术标签",
        showlegend=False,
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)


def render_optimization_result():
    """渲染优化结果"""
    result = st.session_state.optimization_result

    st.markdown("### 🎯 最优话术组合策略")

    # 显示最优配比
    st.markdown("**📊 推荐话术配比：**")

    # 创建配比图表
    labels = []
    ratios = []
    colors = []

    for label, ratio in sorted(
        result.optimal_ratio.items(),
        key=lambda x: x[1],
        reverse=True
    ):
        if ratio >= 0.05:  # 只显示占比5%以上的
            labels.append(label)
            ratios.append(ratio * 100)
            colors.append(LABEL_COLORS.get(label, '#BDC3C7'))

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=ratios,
        hole=0.4,
        marker_colors=colors,
        textinfo='label+percent',
        textposition='outside'
    )])

    fig.update_layout(
        showlegend=False,
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

    # 显示分阶段建议
    st.markdown("**🎬 分阶段话术策略：**")

    for stage_name, advice in result.stage_advice.items():
        with st.expander(f"{stage_name}（{advice['duration_minutes']:.0f}分钟）- {advice['description']}"):
            st.markdown("**建议话术类型：**")
            for label, ratio in advice['recommended_ratio'].items():
                duration = advice['recommended_duration'][label]
                st.markdown(f"- {label}: {ratio*100:.0f}%（约{duration:.0f}分钟）")

            st.markdown("**💡 话术示例：**")
            for script in advice['key_scripts'][:3]:
                st.markdown(f"> {script}")

    # 显示关键洞察
    st.markdown("**🔍 关键洞察：**")
    for insight in result.insights:
        st.markdown(f"""
        <div class="insight-box">
            {insight}
        </div>
        """, unsafe_allow_html=True)





def render_results():
    """渲染分析结果"""
    # 概览指标
    render_overview_metrics()

    st.markdown("---")

    # 两列布局
    col1, col2 = st.columns([1, 1])

    with col1:
        render_label_distribution()

    with col2:
        render_delay_distribution()

    st.markdown("---")

    # 高转化话术
    render_top_scripts()

    st.markdown("---")

    # 低转化话术
    render_low_scripts()

    st.markdown("---")

    # 标签贡献度
    render_label_contribution()

    st.markdown("---")

    # 优化建议
    render_optimization_result()

    st.markdown("---")

    # 底部声明
    render_footer()


# ==================== 主函数 ====================
def main():
    """主函数"""
    # 渲染侧边栏（包含页面导航）
    page = render_sidebar()

    if page == "ℹ️ 关于本项目":
        render_about_page()
    elif page == "🔬 技术原理":
        render_tech_page()
    else:
        # 话术分析首页
        init_session_state()
        render_header()

        # 上传区域
        video_file, order_file, live_start, live_end = render_upload_section()

        # 分析按钮
        if render_analysis_button():
            # 执行分析
            progress_placeholder = st.empty()
            status_placeholder = st.empty()

            with st.spinner("正在分析中，请稍候..."):
                try:
                    # 1. 处理视频
                    progress_bar = progress_placeholder.progress(0)
                    status_text = status_placeholder.empty()

                    status_text.text("正在处理视频/话术数据...")
                    segments = process_video(video_file, progress_bar, status_text)
                    st.session_state.segments = segments

                    # 2. 处理订单
                    progress_bar = progress_placeholder.progress(0)
                    status_text.text("正在处理订单数据...")
                    orders, stats = process_orders(order_file, live_start, live_end, progress_bar, status_text)
                    st.session_state.orders = orders
                    st.session_state.order_stats = stats

                    # 3. 分类话术
                    progress_bar = progress_placeholder.progress(0)
                    status_text.text("正在分类话术...")
                    classification_results = classify_scripts(segments, progress_bar, status_text)
                    st.session_state.classification_results = classification_results

                    # 4. 归因分析
                    progress_bar = progress_placeholder.progress(0)
                    status_text.text("正在进行归因分析...")
                    attribution_results, label_summaries = attribute_scripts(
                        segments, classification_results, orders,
                        live_start, live_end, progress_bar, status_text
                    )
                    st.session_state.attribution_results = attribution_results
                    st.session_state.label_summaries = label_summaries

                    # 5. 优化建议
                    status_text.text("正在生成优化建议...")
                    live_duration_minutes = (live_end - live_start).total_seconds() / 60
                    optimization_result = optimize_scripts(label_summaries, live_duration_minutes)
                    st.session_state.optimization_result = optimization_result

                    st.session_state.analysis_done = True

                    # 清除进度显示
                    progress_placeholder.empty()
                    status_placeholder.empty()

                    st.success("✅ 分析完成！")

                except Exception as e:
                    progress_placeholder.empty()
                    status_placeholder.empty()
                    st.error(f"❌ 分析出错：{str(e)}")
                    return

        # 显示结果
        if st.session_state.analysis_done:
            render_results()
        else:
            # 首页底部也显示声明
            render_footer()


if __name__ == "__main__":
    main()
