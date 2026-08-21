#!/usr/bin/env python3
"""
大乐透统计分析 - 仅使用统计模型（frequency/poisson/exponential_smoothing/monte_carlo）
输出格式化的分析报告
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import random
import numpy as np
from collections import Counter

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)

from config import (
    FRONT_MIN, FRONT_MAX, BACK_MIN, BACK_MAX,
    FRONT_COUNT, BACK_COUNT, ENSEMBLE_WEIGHTS,
)
from data_fetcher import update_data
from analysis.statistics import frequency_analysis, hot_cold_analysis, comprehensive_analysis
from models.statistical import FrequencyModel, PoissonModel, MonteCarloModel
from models.timeseries import ExponentialSmoothingModel
from models.ensemble import EnsembleModel, generate_recommendations, MorphologicalFilter
from utils.helpers import format_numbers

# ─── Load Data ───────────────────────────────────────────────────
df = update_data(force_refresh=False)
total_draws = len(df)

# Last draw info
latest = df.iloc[0]
front_cols = [f"front_{i}" for i in range(1, 6)]
back_cols = [f"back_{i}" for i in range(1, 3)]
latest_period = latest["period"]
latest_front = [int(latest[c]) for c in front_cols]
latest_back = [int(latest[c]) for c in back_cols]
latest_str = format_numbers(latest_front, latest_back)

print(f"📡 数据加载完成：共 {total_draws} 期")
print(f"   最新期号：第 {latest_period} 期")
print(f"   最新开奖：{latest_str}")
print()

# ─── Hot/Cold Analysis ────────────────────────────────────────────
freq = frequency_analysis(df)
hot_cold = hot_cold_analysis(df)

# Top 10 front hot numbers
front_freq_sorted = sorted(freq["front_freq"].items(), key=lambda x: x[1], reverse=True)
top10_front = [(n, cnt) for n, cnt in front_freq_sorted[:10]]
top10_front_str = "  ".join(f"{n:02d}({cnt}次)" for n, cnt in top10_front)

# Top 5 back hot numbers
back_freq_sorted = sorted(freq["back_freq"].items(), key=lambda x: x[1], reverse=True)
top5_back = [(n, cnt) for n, cnt in back_freq_sorted[:5]]
top5_back_str = "  ".join(f"{n:02d}({cnt}次)" for n, cnt in top5_back)

print("🔥 前区热号 TOP 10（出现次数）:")
print(f"   {top10_front_str}")
print()
print("🔥 后区热号 TOP 5（出现次数）:")
print(f"   {top5_back_str}")
print()

# ─── Initialize Statistical Models Only ──────────────────────────
print("🧠 初始化统计模型...")

models = {}

models["frequency"] = FrequencyModel(df)
print("   ✓ 频率分析模型")

models["poisson"] = PoissonModel(df)
print("   ✓ 泊松分布模型")

mc = MonteCarloModel(df, num_simulations=50000)
models["monte_carlo"] = mc
print("   ✓ 蒙特卡洛模拟 (50000次)")

models["exponential_smoothing"] = ExponentialSmoothingModel(df, alpha=0.3)
print("   ✓ 指数平滑模型 (α=0.3)")
print()

# ─── Get Individual Model Probabilities ──────────────────────────
print("📊 各模型概率分布摘要（前区 TOP 10）:")

model_labels = {
    "frequency": "频率分析",
    "poisson": "泊松分布",
    "monte_carlo": "蒙特卡洛模拟",
    "exponential_smoothing": "指数平滑",
}

for name, model in models.items():
    if hasattr(model, "predict_front_probabilities"):
        probs = model.predict_front_probabilities()
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        top = [f"{n:02d}({p*100:.1f}%)" for n, p in sorted_probs[:5]]
        print(f"   [{model_labels.get(name, name)}] {'  '.join(top)}")

print()

# ─── Ensemble + Generate Recommendations ─────────────────────────
# Override weights for statistical-only ensemble
stat_weights = {"frequency": 0.35, "poisson": 0.25, "monte_carlo": 0.20, "exponential_smoothing": 0.20}

ensemble = EnsembleModel(models)

# Override the ensemble weights to use our stat-only weights
# The EnsembleModel reads ENSEMBLE_WEIGHTS from config, so models not present
# are just ignored and weights are re-normalized among present models.
# But the default weights give more weight to frequency. Let me directly
# adjust the internal state.

# Actually, EnsembleModel._compute_ensemble uses ENSEMBLE_WEIGHTS dict.
# The sum of our 4 stat models' weights from config = 0.25+0.15+0.10+0.10 = 0.60
# After normalization: freq=0.417, poisson=0.25, mc=0.167, es=0.167
# That's fine.

recommendations = generate_recommendations(ensemble, num_groups=8)

print("🎯 8 组推荐号码")
print("=" * 70)

for i, rec in enumerate(recommendations, 1):
    f_str = " ".join(f"{n:02d}" for n in rec["front"])
    b_str = " ".join(f"{n:02d}" for n in rec["back"])
    print()
    print(f"   [{i}]  {f_str}  +  {b_str}")
    print(f"       得分: {rec['score']:.4f}")
    print(f"       分析: {rec['reason']}")

print()
print("=" * 70)

# ─── Per-group detailed analysis ─────────────────────────────────
print()
print("📋 各组号码详细分析:")
print("-" * 70)

for i, rec in enumerate(recommendations, 1):
    front = rec["front"]
    back = rec["back"]
    f_str = " ".join(f"{n:02d}" for n in front)
    b_str = " ".join(f"{n:02d}" for n in back)

    # Front area stats
    f_sum = sum(front)
    f_span = max(front) - min(front)
    f_odd = sum(1 for n in front if n % 2 == 1)
    f_even = 5 - f_odd
    f_small = sum(1 for n in front if n <= 17)
    f_large = 5 - f_small

    # Back area
    b_odd = sum(1 for n in back if n % 2 == 1)
    b_even = 2 - b_odd

    # Hot/cold check
    hot_front_set = {n for n, _ in top10_front}
    hot_back_set = {n for n, _ in top5_back}
    hot_front_count = sum(1 for n in front if n in hot_front_set)
    hot_back_count = sum(1 for n in back if n in hot_back_set)

    # Identify which model contributed most
    # (by checking which model's probability sum is highest for this combo)
    best_model = "集成"
    best_score = 0
    for mname, model in models.items():
        if hasattr(model, "predict_front_probabilities"):
            fp = model.predict_front_probabilities()
            bp = model.predict_back_probabilities()
            sc = sum(fp.get(n, 0) for n in front) / 5 + sum(bp.get(n, 0) for n in back) / 2
            if sc > best_score:
                best_score = sc
                best_model = model_labels.get(mname, mname)

    print(f"\n  [{i}] {f_str}  +  {b_str}")
    print(f"      前区: 和值={f_sum}  跨度={f_span}  奇偶比={f_odd}:{f_even}  大小比={f_small}:{f_large}")
    print(f"      后区: {'奇偶' if b_odd == 1 else ('双奇' if b_odd == 2 else '双偶')}")
    print(f"      热号: 前区{hot_front_count}个热号 后区{hot_back_count}个热号")
    print(f"      主导模型: {best_model}")

print()
print("-" * 70)

# ─── Statistical Summary for Context ──────────────────────────────
analysis = comprehensive_analysis(df)
stats = analysis["summary"]
oe = analysis["odd_even"]
sa = analysis["sum_analysis"]
sp = analysis["span"]

print()
print("📊 统计基准参考:")
print(f"   前区常见奇偶比: {oe['most_common_front_ratio'][0]}:{oe['most_common_front_ratio'][1]}")
print(f"   前区和值均值: {sa['front_sum']['mean']:.1f}  跨度均值: {sp['mean']:.1f}")
print(f"   连号出现率: {analysis['consecutive']['has_consecutive_pct']:.1f}%")
print(f"   AC值均值: {analysis['ac_value']['mean']:.1f}")

# ─── Disclaimer ──────────────────────────────────────────────────
print()
print("=" * 70)
print("⚠️  免责声明")
print("=" * 70)
print("本程序仅供娱乐和研究目的。")
print("彩票开奖是独立随机事件，历史数据无法保证未来结果。")
print("任何预测分析均不构成投资建议，请理性购彩，量力而行。")
print("=" * 70)
