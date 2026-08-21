#!/usr/bin/env python3
"""
大乐透 Cron 分析脚本 - 仅使用统计模型（频率/泊松/指数平滑/蒙特卡洛）
"""
import sys
import os
sys.path.insert(0, "/root/lotti-analyzer")

import random
import numpy as np
from collections import Counter

# 固定随机种子以确保可复现，但不影响随机采样质量
random.seed(42)
np.random.seed(42)

from data_fetcher import update_data, load_from_csv, fetch_from_500com, save_to_csv, _merge_dfs
from analysis.statistics import comprehensive_analysis, frequency_analysis, hot_cold_analysis
from config import FRONT_MIN, FRONT_MAX, BACK_MIN, BACK_MAX, FRONT_COUNT, BACK_COUNT, ENSEMBLE_WEIGHTS
from models.statistical import FrequencyModel, PoissonModel, MonteCarloModel
from models.timeseries import ExponentialSmoothingModel
from models.ensemble import EnsembleModel, generate_recommendations, MorphologicalFilter

# ============================================================
# 1. 数据加载
# ============================================================
print("=" * 62)
print("  大乐透数据分析与预测系统（统计模型版）")
print("=" * 62)

# 尝试加载本地数据
print("\n📡 数据加载...")
df_local = load_from_csv()
if df_local is not None and len(df_local) >= 10:
    df = df_local
    print(f"  ✓ 从本地CSV加载 {len(df)} 期数据")
else:
    # 尝试网络抓取
    print("  尝试网络抓取...")
    df_local = df_local or None  # might be None or empty
    try:
        df_500 = fetch_from_500com()
        if df_500 is not None and len(df_500) > 0:
            df = df_500
            if df_local is not None:
                df = _merge_dfs(df, df_local)
            save_to_csv(df)
            print(f"  ✓ 从网络加载 {len(df)} 期数据")
        else:
            print("  ⚠ 网络抓取失败，使用本地数据")
            df = df_local if df_local is not None else load_from_csv()
    except Exception as e:
        print(f"  ⚠ 网络抓取异常: {e}")
        df = load_from_csv() if df is None else df

if df is None or len(df) == 0:
    print("  ❌ 无可用数据！")
    sys.exit(1)

total_draws = len(df)
latest_row = df.iloc[0]
latest_period = latest_row["period"]
front_cols = [f"front_{i}" for i in range(1, 6)]
back_cols = [f"back_{i}" for i in range(1, 3)]
latest_front = sorted([int(latest_row[c]) for c in front_cols])
latest_back = sorted([int(latest_row[c]) for c in back_cols])

print(f"  📊 共 {total_draws} 期历史数据")
print(f"  📅 最新期号: {latest_period}")
print(f"  🎯 最新开奖: {' '.join(f'{n:02d}' for n in latest_front)} + {' '.join(f'{n:02d}' for n in latest_back)}")

# ============================================================
# 2. 综合分析
# ============================================================
print("\n📊 统计分析...")
analysis = comprehensive_analysis(df)
stats = analysis["summary"]
print(f"  ✓ 数据范围: {stats['date_range']}")
print(f"  ✓ 常见奇偶比: {analysis['odd_even']['most_common_front_ratio']}")
print(f"  ✓ 前区平均和值: {analysis['sum_analysis']['front_sum']['mean']:.1f}")
print(f"  ✓ 前区平均跨度: {analysis['span']['mean']:.1f}")
print(f"  ✓ 连号出现率: {analysis['consecutive']['has_consecutive_pct']:.1f}%")

# ============================================================
# 3. 冷热号分析
# ============================================================
hot_cold = analysis["hot_cold"]
hot_front_top10 = hot_cold["hot_front"][:10]
hot_back_top5 = hot_cold["hot_back"][:5]

# 也做纯频率分析
freq = frequency_analysis(df)
front_freq_sorted = sorted(freq["front_freq"].items(), key=lambda x: x[1], reverse=True)
back_freq_sorted = sorted(freq["back_freq"].items(), key=lambda x: x[1], reverse=True)

print(f"\n🔥 前区热号 TOP10（综合频率+遗漏评分）: {', '.join(f'{n:02d}' for n in hot_front_top10)}")
print(f"🔥 后区热号 TOP5: {', '.join(f'{n:02d}' for n in hot_back_top5)}")

# ============================================================
# 4. 模型初始化 - 仅统计模型
# ============================================================
print("\n🧠 模型训练...")

models = {}

models["frequency"] = FrequencyModel(df)
print("  ✓ 频率分析模型")

models["poisson"] = PoissonModel(df)
print("  ✓ 泊松分布模型")

mc = MonteCarloModel(df, num_simulations=100000)
models["monte_carlo"] = mc
print("  ✓ 蒙特卡洛模拟 (10万次)")

models["exponential_smoothing"] = ExponentialSmoothingModel(df, alpha=0.3)
print("  ✓ 指数平滑模型")

# ============================================================
# 5. 调整集成权重（仅统计模型）
# ============================================================
# 只使用 4 个统计模型，重新归一化权重
STAT_WEIGHTS = {
    "frequency": 0.35,
    "poisson": 0.20,
    "monte_carlo": 0.20,
    "exponential_smoothing": 0.25,
}
# 临时覆盖 config 中的权重
import config
config.ENSEMBLE_WEIGHTS = STAT_WEIGHTS

print("\n🎯 集成预测（统计模型加权融合）...")
print(f"  权重: frequency={STAT_WEIGHTS['frequency']}, poisson={STAT_WEIGHTS['poisson']}, "
      f"monte_carlo={STAT_WEIGHTS['monte_carlo']}, exp_smooth={STAT_WEIGHTS['exponential_smoothing']}")

ensemble = EnsembleModel(models)

# 查看前区概率最高的号码
top_front_probs = sorted(ensemble.get_front_probabilities().items(), key=lambda x: x[1], reverse=True)
top_back_probs = sorted(ensemble.get_back_probabilities().items(), key=lambda x: x[1], reverse=True)

print(f"\n  前区概率TOP15: {', '.join(f'{n:02d}({p*100:.1f}%)' for n, p in top_front_probs[:15])}")
print(f"  后区概率TOP8: {', '.join(f'{n:02d}({p*100:.1f}%)' for n, p in top_back_probs[:8])}")

# ============================================================
# 6. 生成推荐 - 8组
# ============================================================
NUM_GROUPS = 8
print(f"\n🎯 生成 {NUM_GROUPS} 组推荐号码...")

recommendations = generate_recommendations(ensemble, num_groups=NUM_GROUPS, use_filter=True, diversity_weight=0.3)

# 按组号排序（而不是按得分），让第1组最均衡，第8组最冒险
# 先按得分排，然后编号
recommendations.sort(key=lambda x: x["score"], reverse=True)

# ============================================================
# 7. 输出结果
# ============================================================
print()
print("=" * 62)
print(f"  🎯 推荐号码（共 {len(recommendations)} 组）")
print("=" * 62)

for i, rec in enumerate(recommendations, 1):
    f_str = " ".join(f"{n:02d}" for n in rec["front"])
    b_str = " ".join(f"{n:02d}" for n in rec["back"])
    front_sum = sum(rec["front"])
    span = max(rec["front"]) - min(rec["front"])
    odd_count = sum(1 for n in rec["front"] if n % 2 == 1)
    back_odd = sum(1 for n in rec["back"] if n % 2 == 1)

    print(f"\n  [{i}]  {f_str}  +  {b_str}")
    print(f"      得分: {rec['score']:.4f}")
    print(f"      {rec['reason']}")

print()
print("=" * 62)
print("⚠️  免责声明")
print("=" * 62)
print("本程序仅供娱乐和研究目的。")
print("彩票开奖是独立随机事件，历史数据无法保证未来结果。")
print("任何预测分析均不构成投资建议，请理性购彩，量力而行。")
print("=" * 62)
