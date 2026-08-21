#!/usr/bin/env python3
"""
大乐透统计分析 - 仅使用统计模型（frequency/poisson/exponential_smoothing/monte_carlo）
定时任务版（2026-07-21）
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import random
import numpy as np
from collections import Counter
from datetime import datetime

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)

from config import (
    FRONT_MIN, FRONT_MAX, BACK_MIN, BACK_MAX,
    FRONT_COUNT, BACK_COUNT,
)
from data_fetcher import update_data
from analysis.statistics import frequency_analysis, hot_cold_analysis, comprehensive_analysis
from models.statistical import FrequencyModel, PoissonModel, MonteCarloModel
from models.timeseries import ExponentialSmoothingModel
from models.ensemble import EnsembleModel, generate_recommendations, MorphologicalFilter
from utils.helpers import format_numbers

# ============================================================
# 1. DATA LOADING
# ============================================================
df = update_data(force_refresh=False)
total_draws = len(df)

front_cols = [f"front_{i}" for i in range(1, 6)]
back_cols = [f"back_{i}" for i in range(1, 3)]

latest = df.iloc[0]
latest_period = int(latest["period"])
latest_front = [int(latest[c]) for c in front_cols]
latest_back = [int(latest[c]) for c in back_cols]
latest_str = format_numbers(latest_front, latest_back)

print(f"📡 数据加载完成：共 {total_draws} 期")
print(f"   最新期号：第 {latest_period} 期")
print(f"   最新开奖：{latest_str}")
print()

# ============================================================
# 2. HOT NUMBER ANALYSIS
# ============================================================
freq = frequency_analysis(df)
hot_cold = hot_cold_analysis(df)

# Front area TOP 10 hot numbers (by frequency)
front_freq_sorted = sorted(freq["front_freq"].items(), key=lambda x: x[1], reverse=True)
top10_front = [(n, cnt) for n, cnt in front_freq_sorted[:10]]
top10_front_str = "  ".join(f"{n:02d}({cnt}次)" for n, cnt in top10_front)

# Back area TOP 5 hot numbers
back_freq_sorted = sorted(freq["back_freq"].items(), key=lambda x: x[1], reverse=True)
top5_back = [(n, cnt) for n, cnt in back_freq_sorted[:5]]
top5_back_str = "  ".join(f"{n:02d}({cnt}次)" for n, cnt in top5_back)

print("🔥 前区热号 TOP 10（出现次数）:")
print(f"   {top10_front_str}")
print()
print("🔥 后区热号 TOP 5（出现次数）:")
print(f"   {top5_back_str}")
print()

# ============================================================
# 3. MODEL INITIALIZATION (Statistical Only)
# ============================================================
print(f"🧠 初始化统计模型（{datetime.now().strftime('%H:%M:%S')}）...")

models = {}

models["frequency"] = FrequencyModel(df)
print("   ✓ 频率分析模型 (FrequencyModel)")

poisson = PoissonModel(df)
models["poisson"] = poisson
print("   ✓ 泊松分布模型 (PoissonModel)")

mc = MonteCarloModel(df, num_simulations=50000)
models["monte_carlo"] = mc
print("   ✓ 蒙特卡洛模拟 (MonteCarloModel, 50000次模拟)")

es = ExponentialSmoothingModel(df, alpha=0.3)
models["exponential_smoothing"] = es
print("   ✓ 指数平滑模型 (ExponentialSmoothing, α=0.3)")
print()

# ============================================================
# 4. MODEL PROBABILITY DISTRIBUTIONS
# ============================================================
model_labels = {
    "frequency": "频率分析",
    "poisson": "泊松分布（冷号）",
    "monte_carlo": "蒙特卡洛模拟",
    "exponential_smoothing": "指数平滑（近期趋势）",
}

print("📊 各模型概率分布摘要（前区 TOP 10）:")
for name, model in models.items():
    if name == "monte_carlo":
        mc_p = model.get_top_probabilities(top_n=100)
        probs = mc_p["front"]
    elif hasattr(model, "predict_front_probabilities"):
        probs = model.predict_front_probabilities()
    else:
        continue
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        top = [f"{n:02d}({p*100:.1f}%)" for n, p in sorted_probs[:10]]
        print(f"   [{model_labels.get(name, name)}]")
        print(f"     {'  '.join(top)}")
print()

# ============================================================
# 5. STRATEGY-BASED RECOMMENDATION GENERATION (8 Groups)
#    Each strategy uses different model combinations to ensure diversity
# ============================================================
print("🎯 生成推荐号码（多策略推理）...")
print()

# Individual model probabilities
front_probs_by_model = {}
back_probs_by_model = {}
for name, model in models.items():
    if name == "monte_carlo":
        mc_probs = model.get_top_probabilities(top_n=100)
        front_probs_by_model[name] = mc_probs["front"]
        back_probs_by_model[name] = mc_probs["back"]
    else:
        if hasattr(model, "predict_front_probabilities"):
            front_probs_by_model[name] = model.predict_front_probabilities()
        if hasattr(model, "predict_back_probabilities"):
            back_probs_by_model[name] = model.predict_back_probabilities()

# Ensemble probs
ensemble = EnsembleModel(models)
front_ensemble = ensemble.get_front_probabilities()
back_ensemble = ensemble.get_back_probabilities()

# ---- Strategy definitions ----
# Each strategy: (name, front_weight_src, back_weight_src, description)
strategies = [
    {
        "name": "📊 频率热号优先",
        "desc": "基于全历史频率加权，倾向近期高频号码，融合指数平滑捕捉趋势",
        "front_weights": {n: front_probs_by_model["frequency"].get(n, 0) * 0.6
                              + front_probs_by_model["exponential_smoothing"].get(n, 0) * 0.4
                          for n in range(FRONT_MIN, FRONT_MAX + 1)},
        "back_weights": {n: back_probs_by_model["frequency"].get(n, 0) * 0.6
                             + back_probs_by_model["exponential_smoothing"].get(n, 0) * 0.4
                         for n in range(BACK_MIN, BACK_MAX + 1)},
    },
    {
        "name": "🎯 集成均衡推荐",
        "desc": "四模型贝叶斯加权融合，兼顾频率、趋势、冷号、模拟的多维推荐",
        "front_weights": front_ensemble,
        "back_weights": back_ensemble,
    },
    {
        "name": "❄️ 泊松追冷策略",
        "desc": "泊松分布建模号码遗漏周期，优先选取逾期未出号码（冷号回补预期）",
        "front_weights": front_probs_by_model["poisson"],
        "back_weights": back_probs_by_model["poisson"],
    },
    {
        "name": "🎲 蒙特卡洛优选",
        "desc": "50000次加权随机模拟，从海量候选中按频率概率筛选最优组合",
        "front_weights": front_probs_by_model["monte_carlo"],
        "back_weights": back_probs_by_model["monte_carlo"],
    },
    {
        "name": "📈 趋势追踪策略",
        "desc": "指数平滑模型聚焦近期20期趋势变化，快速响应号码热度转变",
        "front_weights": front_probs_by_model["exponential_smoothing"],
        "back_weights": back_probs_by_model["exponential_smoothing"],
    },
    {
        "name": "⚖️ 冷热均衡策略",
        "desc": "频率热号(50%)+泊松冷号(50%)平衡，热号兜底+冷号博冷兼顾",
        "front_weights": {n: front_probs_by_model["frequency"].get(n, 0) * 0.5
                              + front_probs_by_model["poisson"].get(n, 0) * 0.5
                          for n in range(FRONT_MIN, FRONT_MAX + 1)},
        "back_weights": {n: back_probs_by_model["frequency"].get(n, 0) * 0.5
                             + back_probs_by_model["poisson"].get(n, 0) * 0.5
                         for n in range(BACK_MIN, BACK_MAX + 1)},
    },
    {
        "name": "🔄 蒙特卡洛+趋势混合",
        "desc": "蒙特卡洛模拟(60%)+指数平滑趋势(40%)，模拟中融入近期热点",
        "front_weights": {n: front_probs_by_model["monte_carlo"].get(n, 0) * 0.6
                              + front_probs_by_model["exponential_smoothing"].get(n, 0) * 0.4
                          for n in range(FRONT_MIN, FRONT_MAX + 1)},
        "back_weights": {n: back_probs_by_model["monte_carlo"].get(n, 0) * 0.6
                             + back_probs_by_model["exponential_smoothing"].get(n, 0) * 0.4
                         for n in range(BACK_MIN, BACK_MAX + 1)},
    },
    {
        "name": "🔮 全模型贝叶斯融合",
        "desc": "四模型log-space加权乘积融合，低频信号增强，冷热双向兼顾",
        "front_weights": {n: (front_probs_by_model["frequency"].get(n, 0) ** 0.25 *
                              front_probs_by_model["poisson"].get(n, 0) ** 0.25 *
                              front_probs_by_model["monte_carlo"].get(n, 0) ** 0.25 *
                              front_probs_by_model["exponential_smoothing"].get(n, 0) ** 0.25)
                          for n in range(FRONT_MIN, FRONT_MAX + 1)},
        "back_weights": {n: (back_probs_by_model["frequency"].get(n, 0) ** 0.25 *
                             back_probs_by_model["poisson"].get(n, 0) ** 0.25 *
                             back_probs_by_model["monte_carlo"].get(n, 0) ** 0.25 *
                             back_probs_by_model["exponential_smoothing"].get(n, 0) ** 0.25)
                         for n in range(BACK_MIN, BACK_MAX + 1)},
    },
]

# Normalize strategy weights
for strat in strategies:
    fw = strat["front_weights"]
    bw = strat["back_weights"]
    ft = sum(fw.values())
    bt = sum(bw.values())
    if ft > 0:
        strat["front_weights"] = {n: v / ft for n, v in fw.items()}
    if bt > 0:
        strat["back_weights"] = {n: v / bt for n, v in bw.items()}

# ---- Generate via weighted sampling for each strategy ----
all_recommendations = []

# Cross-group constraint tracking
used_front = Counter()
used_back = Counter()
used_front_pairs = {}
used_back_pairs = {}

# Hot sets (for reasoning)
hot_front_set = {n for n, _ in top10_front}
hot_back_set = {n for n, _ in top5_back}
cold_front_set = {n for n, _ in sorted(freq["front_freq"].items(), key=lambda x: x[1])[:5]}
cold_back_set = {n for n, _ in sorted(freq["back_freq"].items(), key=lambda x: x[1])[:3]}

unique_seeds = [101, 202, 303, 404, 505, 606, 707, 808]

for idx, strat in enumerate(strategies):
    seed = unique_seeds[idx]
    rng = random.Random(seed)
    np.random.seed(seed)

    front_weights_arr = [strat["front_weights"].get(n, 0.001) for n in range(FRONT_MIN, FRONT_MAX + 1)]
    back_weights_arr = [strat["back_weights"].get(n, 0.001) for n in range(BACK_MIN, BACK_MAX + 1)]
    front_weights_arr = [max(w, 0.001) for w in front_weights_arr]
    back_weights_arr = [max(w, 0.001) for w in back_weights_arr]

    front_nums_all = list(range(FRONT_MIN, FRONT_MAX + 1))
    back_nums_all = list(range(BACK_MIN, BACK_MAX + 1))

    best_combo = None
    best_score = -1
    best_front = None
    best_back = None

    for trial in range(500):
        # Apply usage penalty
        adj_fw = list(front_weights_arr)
        adj_bw = list(back_weights_arr)
        for i, n in enumerate(front_nums_all):
            if used_front.get(n, 0) >= 2:
                adj_fw[i] *= 0.1
        for i, n in enumerate(back_nums_all):
            if used_back.get(n, 0) >= 2:
                adj_bw[i] *= 0.1

        # Sample front
        pool_f = list(front_nums_all)
        w_f = list(adj_fw)
        sel_front = []
        for _ in range(FRONT_COUNT):
            if sum(w_f) <= 0:
                ci = rng.randint(0, len(pool_f) - 1)
            else:
                ws = np.array(w_f, dtype=float)
                ws /= ws.sum()
                ci = rng.choices(range(len(pool_f)), weights=ws, k=1)[0]
            sel_front.append(pool_f[ci])
            pool_f.pop(ci)
            w_f.pop(ci)

        # Sample back
        pool_b = list(back_nums_all)
        w_b = list(adj_bw)
        sel_back = []
        for _ in range(BACK_COUNT):
            if sum(w_b) <= 0:
                ci = rng.randint(0, len(pool_b) - 1)
            else:
                ws = np.array(w_b, dtype=float)
                ws /= ws.sum()
                ci = rng.choices(range(len(pool_b)), weights=ws, k=1)[0]
            sel_back.append(pool_b[ci])
            pool_b.pop(ci)
            w_b.pop(ci)

        sel_front.sort()
        sel_back.sort()

        # Morphological filter
        valid, _ = MorphologicalFilter.is_valid(sel_front, sel_back)
        if not valid:
            continue

        # Cross-group constraint check
        # Skip if any number appears >= 3 times
        if any(used_front.get(n, 0) >= 3 for n in sel_front):
            continue
        if any(used_back.get(n, 0) >= 3 for n in sel_back):
            continue
        # Skip if any pair appears >= 3 times
        sf = sorted(sel_front)
        pair_ok = True
        for i in range(len(sf)):
            for j in range(i + 1, len(sf)):
                pair = (sf[i], sf[j])
                if used_front_pairs.get(pair, 0) >= 3:
                    pair_ok = False
                    break
            if not pair_ok:
                break
        if not pair_ok:
            continue
        sb = sorted(sel_back)
        for i in range(len(sb)):
            for j in range(i + 1, len(sb)):
                pair = (sb[i], sb[j])
                if used_back_pairs.get(pair, 0) >= 3:
                    pair_ok = False
                    break
            if not pair_ok:
                break
        if not pair_ok:
            continue

        # Score = probability sum
        prob_score = sum(strat["front_weights"].get(n, 0) for n in sel_front) / FRONT_COUNT
        prob_score += sum(strat["back_weights"].get(n, 0) for n in sel_back) / BACK_COUNT

        # CRF-style structural bonus
        f_sum = sum(sel_front)
        f_span = max(sel_front) - min(sel_front)
        f_odd = sum(1 for n in sel_front if n % 2 == 1)
        f_even = 5 - f_odd
        # Ideal: sum ~ 80-110, span ~ 15-28, odd ~ 2-3
        struct_score = 0
        if 75 <= f_sum <= 120:
            struct_score += 0.05
        if 15 <= f_span <= 28:
            struct_score += 0.03
        if f_odd in (2, 3):
            struct_score += 0.02

        total_score = prob_score + struct_score

        if total_score > best_score:
            best_score = total_score
            best_front = list(sel_front)
            best_back = list(sel_back)
            best_combo = (best_front, best_back, best_score)

    if best_combo is None:
        # Fallback: relaxed filter
        for trial in range(500):
            pool_f = list(front_nums_all)
            w_f = list(front_weights_arr)
            sel_front = []
            for _ in range(FRONT_COUNT):
                if sum(w_f) <= 0:
                    ci = rng.randint(0, len(pool_f) - 1)
                else:
                    ws = np.array(w_f, dtype=float)
                    ws /= ws.sum()
                    ci = rng.choices(range(len(pool_f)), weights=ws, k=1)[0]
                sel_front.append(pool_f[ci])
                pool_f.pop(ci)
                w_f.pop(ci)
            sel_back = []
            pool_b = list(back_nums_all)
            w_b = list(back_weights_arr)
            for _ in range(BACK_COUNT):
                if sum(w_b) <= 0:
                    ci = rng.randint(0, len(pool_b) - 1)
                else:
                    ws = np.array(w_b, dtype=float)
                    ws /= ws.sum()
                    ci = rng.choices(range(len(pool_b)), weights=ws, k=1)[0]
                sel_back.append(pool_b[ci])
                pool_b.pop(ci)
                w_b.pop(ci)
            sel_front.sort()
            sel_back.sort()
            valid, _ = MorphologicalFilter.is_valid(sel_front, sel_back)
            if not valid:
                continue
            # Only check severe constraint violations (>3)
            if any(used_front.get(n, 0) >= 4 for n in sel_front):
                continue
            if any(used_back.get(n, 0) >= 4 for n in sel_back):
                continue
            prob_score = sum(strat["front_weights"].get(n, 0) for n in sel_front) / FRONT_COUNT
            prob_score += sum(strat["back_weights"].get(n, 0) for n in sel_back) / BACK_COUNT
            if prob_score > best_score:
                best_score = prob_score
                best_front = list(sel_front)
                best_back = list(sel_back)
                best_combo = (best_front, best_back, best_score)

    # Still None? Just take anything
    if best_combo is None:
        # Last resort: pick top probs
        sf = sorted(front_nums_all, key=lambda n: strat["front_weights"].get(n, 0), reverse=True)[:5]
        sb = sorted(back_nums_all, key=lambda n: strat["back_weights"].get(n, 0), reverse=True)[:2]
        best_front = sf
        best_back = sb
        best_score = sum(strat["front_weights"].get(n, 0) for n in sf) / 5 + sum(strat["back_weights"].get(n, 0) for n in sb) / 2

    # Update cross-group constraints
    for n in best_front:
        used_front[n] = used_front.get(n, 0) + 1
    for n in best_back:
        used_back[n] = used_back.get(n, 0) + 1
    sf = sorted(best_front)
    for i in range(len(sf)):
        for j in range(i + 1, len(sf)):
            pair = (sf[i], sf[j])
            used_front_pairs[pair] = used_front_pairs.get(pair, 0) + 1
    sb = sorted(best_back)
    for i in range(len(sb)):
        for j in range(i + 1, len(sb)):
            pair = (sb[i], sb[j])
            used_back_pairs[pair] = used_back_pairs.get(pair, 0) + 1

    # Build reason string
    f_str = " ".join(f"{n:02d}" for n in best_front)
    b_str = " ".join(f"{n:02d}" for n in best_back)
    f_sum = sum(best_front)
    f_span = max(best_front) - min(best_front)
    f_odd = sum(1 for n in best_front if n % 2 == 1)
    f_even = 5 - f_odd

    # Hot/cold count
    hot_f_cnt = sum(1 for n in best_front if n in hot_front_set)
    cold_f_cnt = sum(1 for n in best_front if n in cold_front_set)
    hot_b_cnt = sum(1 for n in best_back if n in hot_back_set)
    cold_b_cnt = sum(1 for n in best_back if n in cold_back_set)

    # Find dominant model
    best_model_name = "集成"
    best_model_score = 0
    for mname, mprobs in front_probs_by_model.items():
        fp = mprobs
        bp = back_probs_by_model.get(mname, {})
        sc = sum(fp.get(n, 0) for n in best_front) / 5 + sum(bp.get(n, 0) for n in best_back) / 2
        if sc > best_model_score:
            best_model_score = sc
            best_model_name = model_labels.get(mname, mname)

    # 大小比
    small_cnt = sum(1 for n in best_front if n <= 17)
    large_cnt = 5 - small_cnt

    reason_parts = [
        strat["name"],
        f"前区: 和值{f_sum} 跨度{f_span} 奇偶{f_odd}:{f_even} 大小{small_cnt}:{large_cnt}",
        f"热号{hot_f_cnt}个,冷号{cold_f_cnt}个",
        f"主导: {best_model_name}",
    ]
    if hot_b_cnt > 0:
        reason_parts.append(f"后区热号{hot_b_cnt}个")
    if cold_b_cnt > 0:
        reason_parts.append(f"后区冷号{cold_b_cnt}个")

    all_recommendations.append({
        "front": best_front,
        "back": best_back,
        "score": round(best_score, 4),
        "reason": strat["desc"],
        "strategy": strat["name"],
        "stats": {
            "sum": f_sum,
            "span": f_span,
            "odd_even": f"{f_odd}:{f_even}",
            "size_ratio": f"{small_cnt}:{large_cnt}",
            "hot_front": hot_f_cnt,
            "cold_front": cold_f_cnt,
            "hot_back": hot_b_cnt,
            "cold_back": cold_b_cnt,
            "dominant_model": best_model_name,
        },
    })

# ============================================================
# 6. OUTPUT: 8 Recommendation Groups
# ============================================================
print("=" * 72)
print(f"🎯 大乐透 8 组推荐号码（基于 {total_draws} 期历史数据）")
print(f"   数据截止: 第 {latest_period} 期 | {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print("=" * 72)

for i, rec in enumerate(all_recommendations, 1):
    f_str = " ".join(f"{n:02d}" for n in rec["front"])
    b_str = " ".join(f"{n:02d}" for n in rec["back"])
    s = rec["stats"]
    print(f"\n{'─' * 72}")
    print(f"  【{i}】{rec['strategy']}")
    print(f"  🏆  {f_str}  +  {b_str}")
    print(f"  📊 和值={s['sum']}  跨度={s['span']}  奇偶比={s['odd_even']}  大小比={s['size_ratio']}")
    print(f"  🔥 前区热号{s['hot_front']}个/冷号{s['cold_front']}个  后区热号{s['hot_back']}个/冷号{s['cold_back']}个")
    print(f"  💡 分析: {rec['reason']}")
    print(f"  ⭐ 得分: {rec['score']:.4f}")

print()
print("=" * 72)

# ============================================================
# 7. STATISTICAL REFERENCE
# ============================================================
analysis = comprehensive_analysis(df)
stats = analysis["summary"]
oe = analysis["odd_even"]
sa = analysis["sum_analysis"]
sp = analysis["span"]

print()
print("📊 统计基准参考（历史30期）:")
print(f"   前区常见奇偶比: {oe['most_common_front_ratio'][0]}:{oe['most_common_front_ratio'][1]}")
print(f"   前区和值均值: {sa['front_sum']['mean']:.1f}  跨度均值: {sp['mean']:.1f}")
print(f"   连号出现率: {analysis['consecutive']['has_consecutive_pct']:.1f}%")
print(f"   AC值均值: {analysis['ac_value']['mean']:.1f}")

print()
print("=" * 72)

# ============================================================
# 8. DISCLAIMER
# ============================================================
print()
print("⚠️  免责声明")
print("-" * 72)
print("本程序仅供娱乐和研究目的。")
print("彩票开奖是独立随机事件，历史数据无法保证未来结果。")
print("任何预测分析均不构成投资建议，请理性购彩，量力而行。")
print("所有号码基于统计模型（频率/泊松/指数平滑/蒙特卡洛）生成，")
print("不代表实际开奖结果。彩票有风险，投入须谨慎。")
print("=" * 72)
print()
