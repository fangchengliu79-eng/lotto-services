#!/usr/bin/env python3
"""
Cron analysis: Load data, run 4 statistical models, produce 8 recommendations.
No LSTM, no XGBoost — only frequency, poisson, exponential_smoothing, monte_carlo.
"""
import sys, os, random, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from collections import Counter

from config import (
    FRONT_MIN, FRONT_MAX, BACK_MIN, BACK_MAX,
    FRONT_COUNT, BACK_COUNT
)
from data_fetcher import load_from_csv
from analysis.statistics import frequency_analysis, missing_value_analysis
from models.statistical import FrequencyModel, PoissonModel, MonteCarloModel
from models.timeseries import ExponentialSmoothingModel

# ──────────────────────────────────────────────
# 1. Load data
# ──────────────────────────────────────────────
df = load_from_csv()
if df is None or len(df) == 0:
    print("NO_DATA")
    sys.exit(1)

front_cols = [f"front_{i}" for i in range(1, 6)]
back_cols = [f"back_{i}" for i in range(1, 3)]

latest_period = df.iloc[0]["period"]
latest_front = [df.iloc[0][c] for c in front_cols]
latest_back  = [df.iloc[0][c] for c in back_cols]

# ──────────────────────────────────────────────
# 2. Build 4 models
# ──────────────────────────────────────────────
freq_model    = FrequencyModel(df)
poisson_model = PoissonModel(df)
es_model      = ExponentialSmoothingModel(df, alpha=0.3)
mc_model      = MonteCarloModel(df, num_simulations=80000)

# ──────────────────────────────────────────────
# 3. Compute ensemble probabilities (equal weight for all 4)
# ──────────────────────────────────────────────
def get_probs(model, which="front"):
    if which == "front":
        return model.predict_front_probabilities()
    return model.predict_back_probabilities()

# Front ensemble
all_front_probs = []
for name, m in [("freq", freq_model), ("poisson", poisson_model),
                ("es", es_model), ("mc", mc_model)]:
    try:
        probs = get_probs(m, "front")
        if probs and len(probs) == 35:
            all_front_probs.append(probs)
    except Exception:
        pass

EQ_WEIGHT = 1.0 / len(all_front_probs) if all_front_probs else 0.25
front_ensemble = {}
for n in range(FRONT_MIN, FRONT_MAX + 1):
    front_ensemble[n] = sum(p.get(n, 0) * EQ_WEIGHT for p in all_front_probs)
total_f = sum(front_ensemble.values())
if total_f > 0:
    front_ensemble = {n: v / total_f for n, v in front_ensemble.items()}

# Back ensemble
all_back_probs = []
for name, m in [("freq", freq_model), ("poisson", poisson_model),
                ("es", es_model), ("mc", mc_model)]:
    try:
        probs = get_probs(m, "back")
        if probs and len(probs) == 12:
            all_back_probs.append(probs)
    except Exception:
        pass

EQ_WEIGHT_B = 1.0 / len(all_back_probs) if all_back_probs else 0.25
back_ensemble = {}
for n in range(BACK_MIN, BACK_MAX + 1):
    back_ensemble[n] = sum(p.get(n, 0) * EQ_WEIGHT_B for p in all_back_probs)
total_b = sum(back_ensemble.values())
if total_b > 0:
    back_ensemble = {n: v / total_b for n, v in back_ensemble.items()}

# ──────────────────────────────────────────────
# 4. Frequency data
# ──────────────────────────────────────────────
freq = frequency_analysis(df)

# Front hot TOP10
front_sorted = sorted(freq["front_freq"].items(), key=lambda x: x[1], reverse=True)
back_sorted  = sorted(freq["back_freq"].items(),  key=lambda x: x[1], reverse=True)

# Missing data
miss = missing_value_analysis(df)
front_miss_sorted = sorted(miss["front_missing"].items(), key=lambda x: x[1], reverse=True)
back_miss_sorted  = sorted(miss["back_missing"].items(),  key=lambda x: x[1], reverse=True)

# ──────────────────────────────────────────────
# 5. Generate 8 recommendation groups
# ──────────────────────────────────────────────
random.seed(42)
np.random.seed(42)

front_nums = list(range(FRONT_MIN, FRONT_MAX + 1))
back_nums  = list(range(BACK_MIN, BACK_MAX + 1))
front_weights = [max(front_ensemble[n], 0.001) for n in front_nums]
back_weights  = [max(back_ensemble[n], 0.001)  for n in back_nums]

# Hot / cold reference
top_hot_front = {n for n, _ in front_sorted[:10]}
top_hot_back  = {n for n, _ in back_sorted[:5]}
cold_front    = [n for n, _ in front_sorted[-5:]]
cold_back     = [n for n, _ in back_sorted[-3:]]

def weighted_sample(pop, w, k):
    """Weighted sampling without replacement."""
    pool = list(pop)
    ws   = list(w)
    sel  = []
    for _ in range(k):
        if sum(ws) <= 0:
            idx = random.randrange(len(pool))
        else:
            a = np.array(ws, dtype=float)
            a /= a.sum()
            idx = np.random.choice(len(pool), p=a)
        sel.append(pool[idx])
        pool.pop(idx)
        ws.pop(idx)
    return sorted(sel)

def calc_ac(front):
    diffs = set()
    for i in range(len(front)):
        for j in range(i + 1, len(front)):
            diffs.add(front[j] - front[i])
    return len(diffs) - 4

def morph_filter(front, back):
    """Morphological filter — reject extreme patterns."""
    odd = sum(1 for n in front if n % 2 == 1)
    if odd == 0 or odd == 5:
        return False
    small = sum(1 for n in front if n <= 17)
    if small == 0 or small == 5:
        return False
    s = sum(front)
    if s < 50 or s > 155:
        return False
    span = max(front) - min(front)
    if span < 8 or span > 32:
        return False
    consec = sum(1 for i in range(len(front) - 1) if front[i + 1] - front[i] == 1)
    if consec > 2:
        return False
    ac = calc_ac(front)
    if ac < 2:
        return False
    return True

# Model-specific pools for diversity
# We'll draw from each model's top-weighted recommendations
model_labels = []
model_variants = []

# Strategy 1: Frequency-dominated
fw = [max(freq["front_freq"][n] / len(df), 0.001) for n in front_nums]
bw = [max(freq["back_freq"][n] / len(df), 0.001) for n in back_nums]
model_variants.append((fw, bw, "频率权重优先"))
model_labels.append("频率模型")

# Strategy 2: Poisson + missing (补冷)
p_front = poisson_model.predict_front_probabilities()
p_back  = poisson_model.predict_back_probabilities()
fw2 = [max(p_front[n], 0.001) for n in front_nums]
bw2 = [max(p_back[n], 0.001) for n in back_nums]
model_variants.append((fw2, bw2, "泊松模型追击遗漏"))
model_labels.append("泊松分布")

# Strategy 3: Exponential smoothing (近期趋势)
es_front = es_model.predict_front_probabilities()
es_back  = es_model.predict_back_probabilities()
fw3 = [max(es_front[n], 0.001) for n in front_nums]
bw3 = [max(es_back[n], 0.001) for n in back_nums]
model_variants.append((fw3, bw3, "指数平滑关注近期趋势"))
model_labels.append("指数平滑")

# Strategy 4: MC simulation top results
mc_results = mc_model.simulate(top_n=200)
mc_front_counter = Counter()
mc_back_counter = Counter()
for f, b, _ in mc_results:
    mc_front_counter.update(f)
    mc_back_counter.update(b)
fw4 = [max(mc_front_counter.get(n, 1) / 200.0, 0.001) for n in front_nums]
bw4 = [max(mc_back_counter.get(n, 1) / 200.0, 0.001) for n in back_nums]
model_variants.append((fw4, bw4, "蒙特卡洛高频组合"))
model_labels.append("蒙特卡洛模拟")

# Strategy 5-8: Ensemble blends with different biases
# 5: Ensemble + hot bias
fw5 = [max(front_ensemble[n] * (1.2 if n in top_hot_front else 1.0), 0.001) for n in front_nums]
bw5 = [max(back_ensemble[n]  * (1.2 if n in top_hot_back  else 1.0), 0.001) for n in back_nums]
model_variants.append((fw5, bw5, "集成模型偏热号"))
model_labels.append("集成偏热")

# 6: Ensemble + cold bias (chase missing)
fw6 = [max(front_ensemble[n] * (1.5 if n in cold_front else 1.0), 0.001) for n in front_nums]
bw6 = [max(back_ensemble[n]  * (1.5 if n in cold_back  else 1.0), 0.001) for n in back_nums]
model_variants.append((fw6, bw6, "集成模型追冷号"))
model_labels.append("集成追冷")

# 7: Balanced (frequency + poisson blend)
fw7 = [max((freq["front_freq"][n]/len(df) + p_front[n]) / 2, 0.001) for n in front_nums]
bw7 = [max((freq["back_freq"][n]/len(df)  + p_back[n])  / 2, 0.001) for n in back_nums]
model_variants.append((fw7, bw7, "频率+泊松均衡"))
model_labels.append("频率泊松混合")

# 8: Recent trend (exponential smoothing + Monte Carlo)
fw8 = [max((es_front[n] + mc_front_counter.get(n, 1)/200.0) / 2, 0.001) for n in front_nums]
bw8 = [max((es_back[n]  + mc_back_counter.get(n, 1)/200.0)  / 2, 0.001) for n in back_nums]
model_variants.append((fw8, bw8, "指数平滑+蒙特卡洛"))
model_labels.append("近期趋势混合")

# Generate 8 groups
final_recommendations = []
used_front = Counter()
used_back  = Counter()

for idx, ((fw, bw, reason_prefix), label) in enumerate(zip(model_variants, model_labels)):
    attempts = 0
    best = None
    best_score = -1

    while attempts < 3000:
        attempts += 1

        # Diversity penalty
        adj_fw = list(fw)
        adj_bw = list(bw)
        for i, n in enumerate(front_nums):
            penalty = 0.05 * used_front.get(n, 0)
            adj_fw[i] = max(adj_fw[i] - penalty, 0.001)
        for i, n in enumerate(back_nums):
            penalty = 0.05 * used_back.get(n, 0)
            adj_bw[i] = max(adj_bw[i] - penalty, 0.001)

        sel_f = weighted_sample(front_nums, adj_fw, FRONT_COUNT)
        sel_b = weighted_sample(back_nums,  adj_bw, BACK_COUNT)

        if not morph_filter(sel_f, sel_b):
            continue

        # Score
        prob_score = sum(front_ensemble[n] for n in sel_f) / FRONT_COUNT \
                   + sum(back_ensemble[n]  for n in sel_b) / BACK_COUNT
        unique_f = sum(1 for n in sel_f if used_front[n] == 0)
        unique_b = sum(1 for n in sel_b if used_back[n] == 0)
        div_score = (unique_f + unique_b) / 7.0
        combined = 0.7 * prob_score + 0.3 * div_score

        if combined > best_score:
            best_score = combined
            best = (sel_f, sel_b, prob_score)

    if best is None:
        # Fallback: pure ensemble weighted sample
        sel_f = weighted_sample(front_nums, fw, FRONT_COUNT)
        sel_b = weighted_sample(back_nums,  bw, BACK_COUNT)
        prob_score = sum(front_ensemble[n] for n in sel_f) / FRONT_COUNT \
                   + sum(back_ensemble[n]  for n in sel_b) / BACK_COUNT
        best = (sel_f, sel_b, prob_score)

    sel_f, sel_b, prob_score = best

    # Build human-readable reason
    f_set = set(sel_f)
    b_set = set(sel_b)

    # Hot count
    hot_f_cnt = sum(1 for n in sel_f if n in top_hot_front)
    hot_b_cnt = sum(1 for n in sel_b if n in top_hot_back)
    cold_f_cnt = sum(1 for n in sel_f if n in cold_front)
    cold_b_cnt = sum(1 for n in sel_b if n in cold_back)

    details = []
    # Model basis
    details.append(reason_prefix)

    # Hot/cold
    if hot_f_cnt >= 3:
        details.append(f"前区{hot_f_cnt}热")
    if cold_f_cnt > 0:
        details.append(f"补{cold_f_cnt}冷")
    if hot_b_cnt >= 1:
        details.append(f"后区{hot_b_cnt}热")

    # Odd/even
    odd_cnt = sum(1 for n in sel_f if n % 2 == 1)
    details.append(f"奇偶{odd_cnt}:{5-odd_cnt}")

    # Sum
    s = sum(sel_f)
    if 80 <= s <= 130:
        details.append(f"和{s}(适中)")
    else:
        details.append(f"和{s}")

    # Span
    sp = max(sel_f) - min(sel_f)
    details.append(f"跨{sp}")

    # AC
    ac = calc_ac(sel_f)
    details.append(f"AC{ac}")

    # Back
    b_odd = sum(1 for n in sel_b if n % 2 == 1)
    details.append(f"后{'奇偶' if b_odd == 1 else '双' + ('奇' if b_odd == 2 else '偶')}")

    reason = " | ".join(details)

    final_recommendations.append({
        "front": sorted(sel_f),
        "back": sorted(sel_b),
        "score": round(prob_score, 4),
        "reason": reason,
        "model": label,
    })

    for n in sel_f:
        used_front[n] += 1
    for n in sel_b:
        used_back[n] += 1

# Sort by score descending
final_recommendations.sort(key=lambda x: x["score"], reverse=True)

# ──────────────────────────────────────────────
# 6. Format output
# ──────────────────────────────────────────────
print("=" * 62)
print("  大乐透数据分析与统计预测报告")
print("  Da Le Tou (Super Lotto) — Statistical Analysis")
print("=" * 62)
print()

# ── Section 1: Data info ──
print("📊 一、当前数据信息")
print("-" * 40)
print(f"  历史期数: {len(df)} 期 (第{df.iloc[-1]['period']}期 ~ 第{latest_period}期)")
print(f"  最新期号: {latest_period}")
print(f"  最新开奖: {' '.join(f'{n:02d}' for n in latest_front)}  +  {' '.join(f'{n:02d}' for n in latest_back)}")
print()

# ── Section 2: Hot numbers ──
print("🔥 二、热号排行")
print("-" * 40)
print("  前区热号 TOP10:")
for i, (n, cnt) in enumerate(front_sorted[:10], 1):
    pct = cnt / len(df) * 100
    print(f"    {i:2d}. {n:02d}  出现 {cnt} 次  ({pct:.1f}%)")
print()

print("  后区热号 TOP5:")
for i, (n, cnt) in enumerate(back_sorted[:5], 1):
    pct = cnt / len(df) * 100
    print(f"    {i:2d}. {n:02d}  出现 {cnt} 次  ({pct:.1f}%)")
print()

# Additional: Missing leaders
print("📌  当前遗漏最大（前区TOP5）:")
for n, m in front_miss_sorted[:5]:
    print(f"    {n:02d}  遗漏 {m} 期")
print()

print("📌  当前遗漏最大（后区TOP3）:")
for n, m in back_miss_sorted[:3]:
    print(f"    {n:02d}  遗漏 {m} 期")
print()

# ── Section 3: Recommendations ──
print("🎯 三、推荐号码（共 8 组）")
print("=" * 62)
print("  使用模型: 频率分析 + 泊松分布 + 指数平滑 + 蒙特卡洛模拟")
print(f"  蒙特卡洛模拟次数: 80,000")
print("=" * 62)
print()

for i, rec in enumerate(final_recommendations, 1):
    f_str = " ".join(f"{n:02d}" for n in rec["front"])
    b_str = " ".join(f"{n:02d}" for n in rec["back"])
    print(f"  [第{i:1d}组]  {f_str}  +  {b_str}")
    print(f"          模型: {rec['model']}")
    print(f"          得分: {rec['score']:.4f}")
    print(f"          分析: {rec['reason']}")
    print()

# ── Section 4: Model detail ──
print("📋 四、各模型贡献说明")
print("-" * 40)
print("  📐 频率模型: 基于30期历史出现频率，号码出现越多次排名越高")
print("  📐 泊松分布: 对遗漏期数建模，遗漏越久的号码补出概率越高")
print("  📐 指数平滑: α=0.3，给予近期数据更高权重，捕捉趋势变化")
print("  📐 蒙特卡洛: 80,000次加权随机模拟，筛选高频组合")
print("  集成策略: 4个模型等权融合 (各25%)，确保综合判断")
print()

# ── Section 5: Disclaimer ──
print("=" * 62)
print("  ⚠️  免责声明")
print("=" * 62)
print("  本报告仅供娱乐和研究目的。")
print("  彩票开奖为独立随机事件，历史数据无法保证未来结果。")
print("  任何预测分析均不构成购买建议。")
print("  请理性购彩，量力而行，切勿沉迷。")
print("=" * 62)
