#!/usr/bin/env python3
"""
大乐透数据分析与预测系统 - Streamlit Web 界面
科技感暗色主题 | 玻璃形态设计 | 霓虹光效
"""

import sys, os
from io import StringIO
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

from config import (
    FRONT_MIN, FRONT_MAX, BACK_MIN, BACK_MAX,
    FRONT_COUNT, BACK_COUNT, RECOMMEND_GROUPS, OUTPUT_DIR,
)
from data_fetcher import update_data
from analysis.statistics import comprehensive_analysis
from analysis.visualization import (
    plot_frequency, plot_missing, plot_odd_even,
    plot_sum_distribution, plot_heatmap_frequency,
    plot_ac_value, plot_span_trend,
)
from models.statistical import FrequencyModel, PoissonModel, MonteCarloModel
from models.timeseries import ExponentialSmoothingModel
from models.ml_models import LSTMSequenceModel, XGBoostModel, RandomForestModel
from models.ensemble import EnsembleModel, generate_recommendations
from backtest.backtester import Backtester
from utils.helpers import print_disclaimer, format_numbers

# ── 页面配置 ──────────────────────────────────────────────────
st.set_page_config(
    page_title="DLT 大乐透 · 智能预测系统",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 自定义 CSS（国际清亮主题） ──────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

/* 全局基础 */
.stApp {
    background: #f0f2f5;
}
html, body, [data-testid="stAppViewContainer"] {
    color: #1e293b;
    font-family: 'Inter', -apple-system, sans-serif;
}
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #e2e8f0; }
::-webkit-scrollbar-thumb { background: #94a3b8; border-radius: 3px; }

h1, h2, h3 { font-weight: 700; letter-spacing: -0.02em; }
[data-testid="stAppViewBlockContainer"] { max-width: 1400px; padding: 1.5rem 2rem; }

/* ── 侧边栏 ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
    border-right: 1px solid #e2e8f0;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] { color: #475569; }
[data-testid="stSidebar"] .sidebar-brand {
    padding: 1rem 0 0.5rem;
    border-bottom: 1px solid #e2e8f0;
    margin-bottom: 1rem;
}
[data-testid="stSidebar"] .sidebar-brand h2 {
    color: #1e293b;
    font-size: 1.3rem;
    margin: 0;
}
[data-testid="stSidebar"] .sidebar-brand p {
    font-size: 0.75rem;
    color: #94a3b8;
    margin: 0.2rem 0 0 0;
}

/* ── 卡片 ── */
.glass-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.25rem;
    margin-bottom: 1rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    transition: all 0.2s ease;
}
.glass-card:hover {
    box-shadow: 0 4px 12px rgba(0,0,0,0.06);
    border-color: #cbd5e1;
}
.glass-card.glow {
    border-color: #2563eb;
    box-shadow: 0 2px 8px rgba(37,99,235,0.10);
}

/* ── 指标卡片 ── */
.metric-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.metric-card .value {
    font-size: 1.8rem;
    font-weight: 800;
    color: #1e293b;
    line-height: 1.2;
}
.metric-card .label {
    font-size: 0.72rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-top: 0.25rem;
}

/* ── 号码球 ── */
.number-ball {
    display: inline-flex; align-items: center; justify-content: center;
    width: 46px; height: 46px; border-radius: 50%;
    font-size: 1rem; font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    margin: 0 3px;
    transition: all 0.2s ease;
}
.number-ball.front {
    background: linear-gradient(135deg, #dc2626, #b91c1c);
    color: #fff;
    box-shadow: 0 2px 6px rgba(220,38,38,0.25);
}
.number-ball.back {
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
    color: #fff;
    box-shadow: 0 2px 6px rgba(37,99,235,0.25);
}
.number-ball.hit {
    box-shadow: 0 0 0 3px #fff, 0 0 0 5px #eab308;
    border: none;
}

/* ── 预测卡片 ── */
.pred-card .group-badge {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #94a3b8;
    margin-bottom: 0.5rem;
}
.pred-card .score-tag {
    font-size: 0.75rem;
    color: #2563eb;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
}
.pred-card .reason-tag {
    font-size: 0.7rem;
    color: #64748b;
    margin-top: 0.35rem;
}

/* ── 按钮 ── */
.stButton > button, div[data-testid="stButton"] > button {
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.85rem;
    transition: all 0.2s ease;
    border: 1px solid #d1d5db;
    background: #ffffff;
    color: #1e293b;
}
.stButton > button:hover {
    border-color: #2563eb;
    background: #f8fafc;
    color: #2563eb;
}
div[data-testid="stButton"] > button[kind="primary"] {
    background: #2563eb;
    border: none;
    color: #fff;
    box-shadow: 0 2px 8px rgba(37,99,235,0.2);
}
div[data-testid="stButton"] > button[kind="primary"]:hover {
    background: #1d4ed8;
    box-shadow: 0 4px 14px rgba(37,99,235,0.3);
    transform: translateY(-1px);
}

/* ── 滑块 ── */
.stSlider > div > div > div {
    background: #2563eb !important;
}

/* ── 导航选择栏 ── */
.stRadio div[role="radiogroup"] { gap: 0.25rem; }
.stRadio div[role="radiogroup"] label {
    border-radius: 8px;
    padding: 0.5rem 0.75rem;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    transition: all 0.2s ease;
    font-size: 0.85rem;
}
.stRadio div[role="radiogroup"] label:hover {
    border-color: #93c5fd;
    background: #f0f7ff;
}
.stRadio div[role="radiogroup"] label[data-checked="true"] {
    border-color: #2563eb;
    background: #eff6ff;
    color: #1d4ed8;
    font-weight: 600;
}

/* ── 展开器 ── */
.streamlit-expanderHeader {
    background: #ffffff !important;
    border-radius: 10px !important;
    border: 1px solid #e2e8f0 !important;
}
.streamlit-expanderHeader:hover {
    border-color: #93c5fd !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    overflow: hidden;
}
[data-testid="stDataFrame"] thead tr th {
    background: #f8fafc !important;
    color: #64748b !important;
    font-size: 0.75rem !important;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    border-bottom: 1px solid #e2e8f0 !important;
}
[data-testid="stDataFrame"] tbody tr {
    background: #ffffff !important;
}
[data-testid="stDataFrame"] tbody tr:nth-child(even) {
    background: #f8fafc !important;
}
[data-testid="stDataFrame"] tbody td {
    border-bottom: 1px solid #f1f5f9 !important;
    color: #334155 !important;
    font-size: 0.8rem !important;
}

/* ── Alert 框 ── */
[data-testid="stAlert"] {
    border-radius: 10px;
    border-left: 4px solid #2563eb;
    background: #ffffff !important;
}
[data-testid="stAlert"] [data-testid="stMarkdownContainer"] { color: #1e293b; }
.stAlert.success { border-left-color: #059669; }
.stAlert.info { border-left-color: #2563eb; }
.stAlert.warning { border-left-color: #d97706; }
.stAlert.error { border-left-color: #dc2626; }

/* ── 分隔线 ── */
hr {
    border: none;
    height: 1px;
    background: #e2e8f0;
    margin: 1.5rem 0;
}

/* ── 页脚 ── */
.footer {
    text-align: center;
    padding: 2rem 0 1rem;
    border-top: 1px solid #e2e8f0;
    margin-top: 2rem;
}
.footer p {
    font-size: 0.7rem;
    color: #94a3b8;
    margin: 0.25rem 0;
}
</style>
""", unsafe_allow_html=True)

# ── 辅助函数 ──────────────────────────────────────────────────
def render_ball(n, kind="front", is_hit=False):
    """渲染号码球 HTML"""
    css = f"number-ball {kind}"
    if is_hit:
        css += " hit"
    return f"<span class='{css}'>{n:02d}</span>"

def render_metric(value, label, delta=None):
    """渲染指标卡片"""
    delta_html = f"<div style='font-size:0.7rem;color:#10b981;margin-top:0.15rem'>{delta}</div>" if delta else ""
    return f"""
    <div class='metric-card'>
        <div class='value'>{value}</div>
        <div class='label'>{label}</div>
        {delta_html}
    </div>
    """

def render_prediction_card(rec, matches=None, is_active=True):
    """渲染一组预测号码的卡片"""
    g = rec["group"]
    g_matches = matches.get(g, {}) if matches else {}

    front_hits = set(g_matches.get("front_matches", [])) if g_matches else set()
    back_hits = set(g_matches.get("back_matches", [])) if g_matches else set()

    front_balls = "".join(
        render_ball(n, "front", n in front_hits) for n in rec["front"]
    )
    back_balls = "".join(
        render_ball(n, "back", n in back_hits) for n in rec["back"]
    )

    score_pct = rec["score"] * 100
    hit_text = ""
    if g_matches and g_matches.get("total_hits", 0) > 0:
        m = g_matches
        hit_text = (
            f"<div style='margin-top:0.5rem;font-size:0.85rem;color:#eab308;font-weight:600;'>"
            f"🎯 命中 {m['front_hits']}前 + {m['back_hits']}后 = {m['total_hits']}个</div>"
        )
    elif is_active and g_matches:
        hit_text = (
            f"<div style='margin-top:0.5rem;font-size:0.85rem;color:#64748b;'>"
            f"⏳ 待开奖</div>"
        )

    return f"""
    <div class='glass-card glow pred-card' style='padding:1rem;'>
        <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:0.75rem;'>
            <span class='group-badge'>第 {g} 组</span>
            <span class='score-tag'>综合得分 {score_pct:.2f}%</span>
        </div>
        <div style='margin-bottom:0.5rem;'>
            <div style='font-size:0.7rem;color:#64748b;margin-bottom:0.3rem;text-transform:uppercase;letter-spacing:0.05em;'>前区 · {FRONT_COUNT}码</div>
            <div>{front_balls}</div>
        </div>
        <div style='margin-bottom:0.5rem;'>
            <div style='font-size:0.7rem;color:#64748b;margin-bottom:0.3rem;text-transform:uppercase;letter-spacing:0.05em;'>后区 · {BACK_COUNT}码</div>
            <div>{back_balls}</div>
        </div>
        <div class='reason-tag'>📊 {rec['reason']}</div>
        {hit_text}
    </div>
    """

# ── 数据缓存 ──────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner="正在加载数据...")
def load_data(force_refresh=False):
    return update_data(force_refresh=force_refresh)


@st.cache_resource(ttl=3600, show_spinner="正在训练模型...")
def train_models(_df, use_lstm=True, use_xgboost=True):
    models = {}
    models["frequency"] = FrequencyModel(_df)
    models["poisson"] = PoissonModel(_df)
    models["monte_carlo"] = MonteCarloModel(_df, num_simulations=30000)
    models["exponential_smoothing"] = ExponentialSmoothingModel(_df, alpha=0.3)
    if use_lstm:
        try:
            lstm = LSTMSequenceModel(_df); lstm.train()
            models["lstm"] = lstm
        except Exception: pass
    if use_xgboost:
        try:
            xgb = XGBoostModel(_df); xgb.train()
            models["xgboost"] = xgb
        except Exception: pass
    try:
        rf = RandomForestModel(_df); rf.train()
        models["random_forest"] = rf
    except Exception: pass
    return models


# ── 加载数据 ──────────────────────────────────────────────────
with st.spinner("正在加载历史数据..."):
    df = load_data()

if df is None or len(df) == 0:
    st.error("无法获取数据，请检查网络连接或 CSV 文件")
    st.stop()


# ── 侧边栏 ──────────────────────────────────────────────────
with st.sidebar:
    # 品牌
    st.markdown(
        "<div class='sidebar-brand'>"
        "<h2>🎯 DLT 预测系统</h2>"
        "<p>大乐透 · 智能分析 · v2.0</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    # 导航
    st.markdown("<p style='font-size:0.75rem;color:#475569;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.5rem;'>导航</p>", unsafe_allow_html=True)
    page = st.radio(
        "导航",
        ["🎯 预测结果", "📊 数据分析", "📜 预测历史", "📈 回测"],
        label_visibility="collapsed",
        index=0,
    )

    st.markdown("<hr style='margin:1rem 0;'>", unsafe_allow_html=True)

    # 控制面板
    st.markdown("<p style='font-size:0.75rem;color:#475569;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.5rem;'>控制面板</p>", unsafe_allow_html=True)

    if st.button("🔄 刷新数据", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    num_groups = st.slider("推荐组数", 1, 20, RECOMMEND_GROUPS)
    use_morphological = st.checkbox("形态过滤", value=True, help="排除全奇/全偶等极低概率组合")
    enable_lstm = st.checkbox("LSTM 模型", value=True, help="需 TensorFlow")
    enable_xgboost = st.checkbox("XGBoost 模型", value=True)

    st.markdown("<hr style='margin:1rem 0;'>", unsafe_allow_html=True)
    st.caption("⚠️ 仅供娱乐参考 · 理性购彩")

# ── 加载数据 ──────────────────────────────────────────────────
with st.spinner("正在加载历史数据..."):
    df = load_data()

if df is None or len(df) == 0:
    st.error("无法获取数据，请检查网络连接或 CSV 文件")
    st.stop()

st.sidebar.success(f"✅ 已加载 {len(df):,} 期数据")


# ══════════════════════════════════════════════════════════════
#  页面：预测结果
# ══════════════════════════════════════════════════════════════
if page == "🎯 预测结果":
    from utils.predictions import (
        get_latest_prediction, get_all_predictions,
        compare_with_draw, save_prediction, auto_compare_latest
    )
    from models.statistical import FrequencyModel, PoissonModel, MonteCarloModel
    from models.timeseries import ExponentialSmoothingModel
    from models.ensemble import EnsembleModel, generate_recommendations

    # 自动比对
    auto_compare_latest(df)
    latest_pred = get_latest_prediction()

    # ── 头部 ──
    st.markdown(
        "<h1 style='background:linear-gradient(135deg,#2563eb 0%,#1d4ed8 50%,#7c3aed 100%);"
        "-webkit-background-clip:text;-webkit-text-fill-color:transparent;"
        "font-size:2rem;font-weight:800;margin-bottom:0.25rem;'>🎯 当前预测结果</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='color:#475569;font-size:0.85rem;margin-bottom:1.5rem;'>"
        "基于多模型集成 · 综合概率评分 · 冻结不刷新</p>",
        unsafe_allow_html=True,
    )

    if latest_pred is None:
        st.markdown("<div class='glass-card' style='text-align:center;padding:3rem;'>", unsafe_allow_html=True)
        st.markdown(
            "<p style='font-size:1.1rem;color:#64748b;'>暂无预测记录</p>"
            "<p style='font-size:0.85rem;color:#475569;margin-bottom:1.5rem;'>点击下方按钮生成第一期预测</p>",
            unsafe_allow_html=True,
        )
        if st.button("🚀 生成首期预测", type="primary", use_container_width=True):
            with st.spinner("正在生成预测..."):
                models_dict = {
                    "frequency": FrequencyModel(df),
                    "poisson": PoissonModel(df),
                    "exponential_smoothing": ExponentialSmoothingModel(df, alpha=0.3),
                    "monte_carlo": MonteCarloModel(df, num_simulations=20000),
                }
                ensemble = EnsembleModel(models_dict)
                recs = generate_recommendations(ensemble, num_groups=RECOMMEND_GROUPS)
                next_period = str(int(df.iloc[0]["period"]) + 1)
                save_prediction(period=next_period, draw_date="待定", recommendations=recs)
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        period = latest_pred["period"]
        status = latest_pred["status"]
        draw_date = latest_pred.get("draw_date", "待定")
        created = latest_pred.get("created_at", "")
        num_recs = len(latest_pred["recommendations"])

        # 状态横幅
        if status == "completed":
            s = latest_pred["summary"]
            st.success(f"✅ **期 {period}** 已开奖 · 最佳命中 **{s['best_hits']}** 个号码（第 {s['best_group']} 组）")
        else:
            st.info(f"⏳ **期 {period}** 预测已封存 · 开奖后自动比对")

        # ── 指标行 ──
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.markdown(render_metric(f"#{period}", "期号"), unsafe_allow_html=True)
        with col_m2:
            st.markdown(render_metric(f"{num_recs}", "推荐组数"), unsafe_allow_html=True)
        with col_m3:
            status_label = "已比对" if status == "completed" else "待开奖"
            status_color = "#10b981" if status == "completed" else "#f59e0b"
            st.markdown(
                f"<div class='metric-card'>"
                f"<div class='value' style='font-size:1.2rem;background:none;-webkit-text-fill-color:{status_color};color:{status_color};'>{status_label}</div>"
                f"<div class='label'>状态</div></div>",
                unsafe_allow_html=True,
            )
        with col_m4:
            st.markdown(render_metric(draw_date if draw_date != "待定" else created[:10], "日期"), unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)

        # ── 实际开奖号码（如已开奖） ──
        if status == "completed" and latest_pred.get("actual_draw"):
            actual = latest_pred["actual_draw"]
            st.markdown("<h3 style='font-size:1.1rem;margin-bottom:0.75rem;color:#e2e8f0;'>实际开奖号码</h3>", unsafe_allow_html=True)
            actual_front = "".join(render_ball(n, "front") for n in actual["front"])
            actual_back = "".join(render_ball(n, "back") for n in actual["back"])
            st.markdown(
                f"<div style='margin-bottom:0.5rem;'><span style='font-size:0.75rem;color:#64748b;text-transform:uppercase;letter-spacing:0.05em;margin-right:0.5rem;'>前区</span>{actual_front}</div>"
                f"<div><span style='font-size:0.75rem;color:#64748b;text-transform:uppercase;letter-spacing:0.05em;margin-right:0.5rem;'>后区</span>{actual_back}</div>",
                unsafe_allow_html=True,
            )
            st.markdown("<hr>", unsafe_allow_html=True)

        # ── 预测号码 ──
        st.markdown(
            f"<h3 style='font-size:1.1rem;margin-bottom:1rem;color:#e2e8f0;'>"
            f"推荐号码 <span style='color:#64748b;font-weight:400;font-size:0.85rem;'>({num_recs} 组)</span></h3>",
            unsafe_allow_html=True,
        )

        matches = {m["group"]: m for m in (latest_pred.get("matches") or [])}

        # 5组排成一行（3+2 或 5列）
        recs = latest_pred["recommendations"]
        cols = st.columns(min(5, len(recs)))
        for i, rec in enumerate(recs):
            with cols[i]:
                st.markdown(
                    render_prediction_card(rec, matches, status != "completed"),
                    unsafe_allow_html=True,
                )

        # 如果超过5组，继续
        if len(recs) > 5:
            remaining = recs[5:]
            cols2 = st.columns(min(5, len(remaining)))
            for i, rec in enumerate(remaining):
                with cols2[i]:
                    st.markdown(
                        render_prediction_card(rec, matches, status != "completed"),
                        unsafe_allow_html=True,
                    )

        # ── 比对摘要 ──
        if status == "completed" and latest_pred.get("summary"):
            s = latest_pred["summary"]
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown("<h3 style='font-size:1.1rem;margin-bottom:1rem;color:#e2e8f0;'>比对统计</h3>", unsafe_allow_html=True)
            col_a, col_b, col_c, col_d = st.columns(4)
            with col_a:
                st.markdown(render_metric(f"{s['best_hits']} 个", "最佳命中", f"第{s['best_group']}组"), unsafe_allow_html=True)
            with col_b:
                st.markdown(render_metric(f"{s['avg_front_hits']:.1f}", "前区平均命中"), unsafe_allow_html=True)
            with col_c:
                st.markdown(render_metric(f"{s['avg_back_hits']:.1f}", "后区平均命中"), unsafe_allow_html=True)
            with col_d:
                st.markdown(render_metric(f"{s['avg_total_hits']:.1f}", "综合平均命中"), unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)

        # ── 生成下一期 ──
        latest_period = str(df.iloc[0]["period"])
        has_pred = any(
            p["period"] == latest_period and p["status"] in ("active", "completed")
            for p in get_all_predictions()
        )

        if has_pred:
            st.info(f"✅ 期 {latest_period} 已有预测记录，刷新不会重新生成")

        if st.button("📅 生成下一期预测", type="primary", use_container_width=True):
            next_period = str(int(df.iloc[0]["period"]) + 1)
            existing = [p for p in get_all_predictions() if p["period"] == next_period]
            if existing:
                st.warning(f"期 {next_period} 已有预测")
            else:
                with st.spinner("正在生成新预测..."):
                    models_dict = {
                        "frequency": FrequencyModel(df),
                        "poisson": PoissonModel(df),
                        "exponential_smoothing": ExponentialSmoothingModel(df, alpha=0.3),
                        "monte_carlo": MonteCarloModel(df, num_simulations=20000),
                    }
                    ensemble = EnsembleModel(models_dict)
                    recs_new = generate_recommendations(ensemble, num_groups=RECOMMEND_GROUPS)
                    save_prediction(period=next_period, draw_date="待定", recommendations=recs_new)
                    st.rerun()


# ══════════════════════════════════════════════════════════════
#  页面：数据分析
# ══════════════════════════════════════════════════════════════
elif page == "📊 数据分析":
    st.markdown(
        "<h1 style='background:linear-gradient(135deg,#2563eb 0%,#1d4ed8 100%);"
        "-webkit-background-clip:text;-webkit-text-fill-color:transparent;"
        "font-size:2rem;font-weight:800;margin-bottom:0.25rem;'>📊 历史数据分析</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='color:#475569;font-size:0.85rem;margin-bottom:1.5rem;'>"
        f"基于 {len(df):,} 期历史数据 · 综合分析</p>",
        unsafe_allow_html=True,
    )

    analysis = comprehensive_analysis(df)

    # 指标行
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(render_metric(f"{len(df):,}", "历史期数"), unsafe_allow_html=True)
    with col2:
        hot = analysis["hot_cold"]["hot_front"][:5]
        st.markdown(render_metric(" ".join(f"{n:02d}" for n in hot), "前区热号 TOP5"), unsafe_allow_html=True)
    with col3:
        cold = analysis["hot_cold"]["cold_front"][:3]
        st.markdown(render_metric(" ".join(f"{n:02d}" for n in cold), "前区冷号 TOP3"), unsafe_allow_html=True)
    with col4:
        st.markdown(render_metric(f"{len(analysis['hot_cold']['hot_front'])}", "热号总数"), unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # 号码频率
    st.markdown("<h3 style='font-size:1.1rem;margin-bottom:1rem;color:#e2e8f0;'>🔢 号码频率分析</h3>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    with col_a:
        freq = analysis["frequency"]
        front_freq = pd.DataFrame(
            [(n, freq["front_freq"][n], freq["front_freq_pct"][n])
             for n in range(FRONT_MIN, FRONT_MAX + 1)],
            columns=["号码", "出现次数", "频率(%)"]
        ).sort_values("出现次数", ascending=False).reset_index(drop=True)
        front_freq.index = front_freq.index + 1
        st.markdown("<p style='color:#94a3b8;font-size:0.8rem;margin-bottom:0.5rem;'>前区频率</p>", unsafe_allow_html=True)
        st.dataframe(front_freq, width='stretch', height=360)

    with col_b:
        back_freq = pd.DataFrame(
            [(n, freq["back_freq"][n], freq["back_freq_pct"][n])
             for n in range(BACK_MIN, BACK_MAX + 1)],
            columns=["号码", "出现次数", "频率(%)"]
        ).sort_values("出现次数", ascending=False).reset_index(drop=True)
        back_freq.index = back_freq.index + 1
        st.markdown("<p style='color:#94a3b8;font-size:0.8rem;margin-bottom:0.5rem;'>后区频率</p>", unsafe_allow_html=True)
        st.dataframe(back_freq, width='stretch', height=360)

    st.markdown("<hr>", unsafe_allow_html=True)

    # 遗漏值
    st.markdown("<h3 style='font-size:1.1rem;margin-bottom:1rem;color:#e2e8f0;'>📌 遗漏值分析</h3>", unsafe_allow_html=True)
    missing = analysis["missing"]
    col_c, col_d = st.columns(2)

    with col_c:
        front_miss = pd.DataFrame(
            [(n, missing["front_missing"].get(n, 0))
             for n in range(FRONT_MIN, FRONT_MAX + 1)],
            columns=["号码", "遗漏期数"]
        ).sort_values("遗漏期数", ascending=False).reset_index(drop=True)
        front_miss.index = front_miss.index + 1
        st.markdown("<p style='color:#94a3b8;font-size:0.8rem;margin-bottom:0.5rem;'>前区遗漏</p>", unsafe_allow_html=True)
        st.dataframe(front_miss, width='stretch')

    with col_d:
        back_miss = pd.DataFrame(
            [(n, missing["back_missing"].get(n, 0))
             for n in range(BACK_MIN, BACK_MAX + 1)],
            columns=["号码", "遗漏期数"]
        ).sort_values("遗漏期数", ascending=False).reset_index(drop=True)
        back_miss.index = back_miss.index + 1
        st.markdown("<p style='color:#94a3b8;font-size:0.8rem;margin-bottom:0.5rem;'>后区遗漏</p>", unsafe_allow_html=True)
        st.dataframe(back_miss, width='stretch')

    st.markdown("<hr>", unsafe_allow_html=True)

    # 统计摘要
    st.markdown("<h3 style='font-size:1.1rem;margin-bottom:1rem;color:#e2e8f0;'>📋 统计分析摘要</h3>", unsafe_allow_html=True)
    stats = analysis["summary"]
    s = analysis["sum_analysis"]["front_sum"]
    sp = analysis["span"]
    ac = analysis["ac_value"]
    conc = analysis["consecutive"]
    oe = analysis["odd_even"]

    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        st.markdown(render_metric(f"{s['mean']:.1f}", "前区和值均值"), unsafe_allow_html=True)
        st.markdown(render_metric(f"{s['min']} ~ {s['max']}", "和值范围"), unsafe_allow_html=True)
    with col_s2:
        st.markdown(render_metric(f"{sp['mean']:.1f}", "前区跨度均值"), unsafe_allow_html=True)
        st.markdown(render_metric(f"{ac['mean']:.1f}", "AC 值均值"), unsafe_allow_html=True)
    with col_s3:
        st.markdown(render_metric(f"{conc['has_consecutive_pct']:.1f}%", "连号出现率"), unsafe_allow_html=True)
        st.markdown(render_metric(str(oe["most_common_front_ratio"]), "常见奇偶比"), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  页面：预测历史
# ══════════════════════════════════════════════════════════════
elif page == "📜 预测历史":
    from utils.predictions import get_all_predictions, compare_with_draw

    st.markdown(
        "<h1 style='background:linear-gradient(135deg,#2563eb 0%,#1d4ed8 100%);"
        "-webkit-background-clip:text;-webkit-text-fill-color:transparent;"
        "font-size:2rem;font-weight:800;margin-bottom:0.25rem;'>📜 预测历史记录</h1>",
        unsafe_allow_html=True,
    )

    all_preds = get_all_predictions()

    if not all_preds:
        st.markdown(
            "<div class='glass-card' style='text-align:center;padding:3rem;'>"
            "<p style='color:#64748b;'>暂无预测历史记录</p></div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<p style='color:#475569;font-size:0.85rem;margin-bottom:1.5rem;'>共 {len(all_preds)} 条记录 · 按时间倒序</p>",
            unsafe_allow_html=True,
        )

        for pred in all_preds:
            p_status = pred["status"]
            p_period = pred["period"]
            p_created = pred.get("created_at", "")
            p_recs = len(pred.get("recommendations", []))

            if p_status == "completed" and pred.get("summary"):
                s = pred["summary"]
                label = f"**期 {p_period}** ✅ 已比对 · 最佳 {s['best_hits']} 个 · {p_created}"
                with st.expander(label, expanded=False):
                    if pred.get("actual_draw"):
                        actual_f = " ".join(f"{n:02d}" for n in pred["actual_draw"]["front"])
                        actual_b = " ".join(f"{n:02d}" for n in pred["actual_draw"]["back"])
                        st.markdown(
                            f"<p style='color:#94a3b8;font-size:0.85rem;'>"
                            f"📌 实际开奖: "
                            f"{''.join(render_ball(n,'front') for n in pred['actual_draw']['front'])} "
                            f"{''.join(render_ball(n,'back') for n in pred['actual_draw']['back'])}"
                            f"</p>",
                            unsafe_allow_html=True,
                        )

                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.markdown(render_metric(f"{s['best_hits']} 个", "最佳命中"), unsafe_allow_html=True)
                    with col_b:
                        st.markdown(render_metric(f"{s['avg_front_hits']}", "前区平均"), unsafe_allow_html=True)
                    with col_c:
                        st.markdown(render_metric(f"{s['avg_back_hits']}", "后区平均"), unsafe_allow_html=True)

                    if pred.get("matches"):
                        st.markdown("<h4 style='font-size:0.95rem;margin:1rem 0 0.5rem;color:#475569;'>各组预测号码</h4>", unsafe_allow_html=True)
                        # Build a matches lookup
                        matches_map = {m["group"]: m for m in pred["matches"]}
                        for rec in pred["recommendations"]:
                            g = rec["group"]
                            m = matches_map.get(g, {})
                            front_hits = set(m.get("front_matches", [])) if m else set()
                            back_hits = set(m.get("back_matches", [])) if m else set()

                            front_balls = "".join(
                                render_ball(n, "front", n in front_hits) for n in rec["front"]
                            )
                            back_balls = "".join(
                                render_ball(n, "back", n in back_hits) for n in rec["back"]
                            )

                            total = m.get("total_hits", 0) if m else 0
                            hit_badge = (
                                f"<span style='color:#eab308;font-weight:700;font-size:0.85rem;'>🎯 命中 {total}个</span>"
                                if total > 0 else ""
                            )

                            st.markdown(
                                f"<div style='background:#fafbfc;border:1px solid #e2e8f0;border-radius:8px;"
                                f"padding:0.6rem 0.8rem;margin-bottom:0.4rem;display:flex;align-items:center;flex-wrap:wrap;gap:0.5rem;'>"
                                f"<span style='font-weight:700;color:#94a3b8;font-size:0.75rem;min-width:2.5rem;'>第{g}组</span>"
                                f"<span style='font-size:0.7rem;color:#94a3b8;'>前区</span>"
                                f"{front_balls}"
                                f"<span style='font-size:0.7rem;color:#94a3b8;margin-left:0.3rem;'>后区</span>"
                                f"{back_balls}"
                                f"{'&nbsp;' + hit_badge if hit_badge else ''}"
                                f"</div>",
                                unsafe_allow_html=True,
                            )
                    else:
                        st.markdown(
                            f"<p style='color:#94a3b8;font-size:0.85rem;'>暂无比对数据</p>",
                            unsafe_allow_html=True,
                        )
            else:
                label = f"**期 {p_period}** ⏳ 待开奖 · {p_recs}组 · {p_created}"
                with st.expander(label, expanded=False):
                    if pred.get("recommendations"):
                        for rec in pred["recommendations"]:
                            front_balls = "".join(render_ball(n, "front") for n in rec["front"])
                            back_balls = "".join(render_ball(n, "back") for n in rec["back"])
                            st.markdown(
                                f"<div style='background:#fafbfc;border:1px solid #e2e8f0;border-radius:8px;"
                                f"padding:0.5rem 0.8rem;margin-bottom:0.3rem;display:flex;align-items:center;flex-wrap:wrap;gap:0.5rem;'>"
                                f"<span style='font-weight:700;color:#94a3b8;font-size:0.75rem;min-width:2.5rem;'>第{rec['group']}组</span>"
                                f"<span style='font-size:0.7rem;color:#94a3b8;'>前区</span>"
                                f"{front_balls}"
                                f"<span style='font-size:0.7rem;color:#94a3b8;margin-left:0.3rem;'>后区</span>"
                                f"{back_balls}"
                                f"</div>",
                                unsafe_allow_html=True,
                            )


# ══════════════════════════════════════════════════════════════
#  页面：回测
# ══════════════════════════════════════════════════════════════
elif page == "📈 回测":
    st.markdown(
        "<h1 style='background:linear-gradient(135deg,#2563eb 0%,#1d4ed8 100%);"
        "-webkit-background-clip:text;-webkit-text-fill-color:transparent;"
        "font-size:2rem;font-weight:800;margin-bottom:0.25rem;'>📈 历史回测</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='color:#475569;font-size:0.85rem;margin-bottom:1.5rem;'>"
        "使用历史数据验证模型预测准确率</p>",
        unsafe_allow_html=True,
    )

    if st.button("▶️ 开始回测", type="primary", use_container_width=True):
        with st.spinner("正在运行回测（可能需要几分钟）..."):
            old_stdout = sys.stdout
            sys.stdout = captured = StringIO()
            bt = Backtester(df, test_window=30)
            stats = bt.run(verbose=False)
            sys.stdout = old_stdout

        if "error" in stats:
            st.error(f"回测失败: {stats['error']}")
        else:
            st.success("✅ 回测完成")

            # 关键指标
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(render_metric(f"{stats['total_tests']}", "测试期数"), unsafe_allow_html=True)
            with col2:
                st.markdown(render_metric(f"{stats['front_hits']['mean']:.2f}", "前区平均命中"), unsafe_allow_html=True)
            with col3:
                st.markdown(render_metric(f"{stats['back_hits']['mean']:.2f}", "后区平均命中"), unsafe_allow_html=True)
            with col4:
                st.markdown(render_metric(f"{stats['total_hits']['mean']:.2f}", "综合平均命中"), unsafe_allow_html=True)

            st.markdown("<hr>", unsafe_allow_html=True)

            col_a, col_b, col_c, col_d = st.columns(4)
            with col_a:
                st.markdown(render_metric(f"{stats['hit_rates']['front_3+']:.1f}%", "前区 3+ 命中率"), unsafe_allow_html=True)
            with col_b:
                st.markdown(render_metric(f"{stats['hit_rates']['front_4+']:.1f}%", "前区 4+ 命中率"), unsafe_allow_html=True)
            with col_c:
                st.markdown(render_metric(f"{stats['hit_rates']['back_1+']:.1f}%", "后区 1+ 命中率"), unsafe_allow_html=True)
            with col_d:
                st.markdown(render_metric(f"{stats['hit_rates']['back_2']:.1f}%", "后区 2 命中率"), unsafe_allow_html=True)

            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown("<h3 style='font-size:1.1rem;margin-bottom:1rem;color:#e2e8f0;'>详细命中分布</h3>", unsafe_allow_html=True)
            prize_df = pd.DataFrame([
                (k, v["count"], f"{v['pct']:.1f}%")
                for k, v in stats["prize_distribution"].items()
            ], columns=["命中等级", "次数", "占比"])
            st.dataframe(prize_df, width='stretch')

    st.markdown("<hr>", unsafe_allow_html=True)
    print_disclaimer()


# ══════════════════════════════════════════════════════════════
#  页脚
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class='footer'>
    <p>⚠️ 本系统仅供娱乐和研究目的 · 彩票开奖为独立随机事件 · 请理性购彩</p>
    <p>DLT Predictor v2.0 · Python · Streamlit · scikit-learn · TensorFlow</p>
</div>
""", unsafe_allow_html=True)
