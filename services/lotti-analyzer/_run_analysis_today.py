"""
大乐透统计模型分析与预测 — 仅使用统计模型
运行日期: 2026-07-19
"""
import sys, os
sys.path.insert(0, '/root/lotti-analyzer')
from datetime import datetime

from data_fetcher import update_data
from models.statistical import FrequencyModel, PoissonModel, MonteCarloModel
from models.timeseries import ExponentialSmoothingModel
from models.ensemble import EnsembleModel, generate_recommendations
from config import ENSEMBLE_WEIGHTS

RUN_TIME = datetime.now().strftime('%Y-%m-%d %H:%M')

# 1. Load data
df = update_data(force_refresh=False)
total_draws = len(df)

# 2. Get latest draw info
front_cols = [f"front_{i}" for i in range(1, 6)]
back_cols = [f"back_{i}" for i in range(1, 3)]
latest = df.iloc[0]
latest_period = latest["period"]
latest_front = [int(latest[c]) for c in front_cols]
latest_back = [int(latest[c]) for c in back_cols]

# 3. Train models (statistical only)
models = {}
models["frequency"] = FrequencyModel(df)
models["poisson"] = PoissonModel(df)
mc = MonteCarloModel(df, num_simulations=50000)
models["monte_carlo"] = mc
models["exponential_smoothing"] = ExponentialSmoothingModel(df, alpha=0.3)

# 4. Run MC explicitly
mc.simulate(top_n=200)

# 5. Build ensemble with statistical-only weights
adjusted_weights = {
    "frequency": 0.30,
    "poisson": 0.25,
    "monte_carlo": 0.25,
    "exponential_smoothing": 0.20,
}
ENSEMBLE_WEIGHTS.update(adjusted_weights)
ensemble = EnsembleModel(models)

# 6. Generate 8 recommendation groups
recommendations = generate_recommendations(ensemble, num_groups=8, use_filter=True, diversity_weight=0.25)

# 7. Hot numbers
freq = ensemble.get_front_probabilities()
back_probs = ensemble.get_back_probabilities()
sorted_front = sorted(freq.items(), key=lambda x: x[1], reverse=True)
sorted_back = sorted(back_probs.items(), key=lambda x: x[1], reverse=True)
top10_front = [n for n, p in sorted_front[:10]]
top5_back = [n for n, p in sorted_back[:5]]

# === OUTPUT REPORT ===
print("=" * 70)
print("  大乐透 (DaLotto) 数据分析与统计预测报告")
print("=" * 70)
print()
print(f"📅 生成时间: {RUN_TIME}")
print()

# SECTION 1
print("─" * 70)
print("📊 1. 当前数据信息")
print("─" * 70)
print(f"   分析期数: {total_draws} 期")
print(f"   最新期号: {latest_period}")
print(f"   最新开奖号码: {' '.join(f'{n:02d}' for n in latest_front)}  +  {' '.join(f'{n:02d}' for n in latest_back)}")
print(f"   数据范围: 第{df['period'].iloc[-1]}期 ~ 第{df['period'].iloc[0]}期")
print()

# SECTION 2
print("─" * 70)
print("🔥 2. 热号分析（模型综合权重排名）")
print("─" * 70)
print(f"   前区热号 TOP10:")
for idx, n in enumerate(top10_front, 1):
    pct = freq[n] * 100
    print(f"     {idx:2d}. {n:02d}  — 权重 {pct:.2f}%")
print()
print(f"   后区热号 TOP5:")
for idx, n in enumerate(top5_back, 1):
    pct = back_probs[n] * 100
    print(f"     {idx:2d}. {n:02d}  — 权重 {pct:.2f}%")
print()

# SECTION 3
print("─" * 70)
print("🎯 3. 推荐号码（共 8 组）")
print("─" * 70)

for i, rec in enumerate(recommendations, 1):
    f_str = "  ".join(f"{n:02d}" for n in rec["front"])
    b_str = "  ".join(f"{n:02d}" for n in rec["back"])
    front_sum = sum(rec["front"])
    odd_count = sum(1 for n in rec["front"] if n % 2 == 1)
    span = max(rec["front"]) - min(rec["front"])
    
    print(f"\n  ┌─ 第 {i:02d} 组 ─────────────────────────────────────────┐")
    print(f"  │  前区: {f_str}")
    print(f"  │  后区:              {b_str}")
    print(f"  │  分析: {rec['reason']}")
    print(f"  │  和值{front_sum:3d} | 跨度{span:2d} | 得分 {rec['score']:.4f}")
    print(f"  └──────────────────────────────────────────────────────────┘")

print()

# SECTION 4 - Models
print("─" * 70)
print("📐 4. 模型配置说明")
print("─" * 70)
print(f"   频率分析模型 (Frequency):          {adjusted_weights['frequency']*100:.0f}%")
print(f"     → 基于历史出现频率加权，频率越高权重越大")
print(f"   泊松分布模型 (Poisson):            {adjusted_weights['poisson']*100:.0f}%")
print(f"     → 对遗漏值建模，遗漏越久条件概率越高")
print(f"   蒙特卡洛模拟 (Monte Carlo):        {adjusted_weights['monte_carlo']*100:.0f}%")
print(f"     → 50,000次加权随机模拟，筛选高得分组合")
print(f"   指数平滑模型 (Exp. Smoothing):     {adjusted_weights['exponential_smoothing']*100:.0f}%")
print(f"     → α=0.3，近期趋势权重更高，捕捉热度变化")
print()

# SECTION 5 - Disclaimer
print("=" * 70)
print("⚠️  免责声明")
print("=" * 70)
print("本报告基于统计学方法（频率分析、泊松分布、指数平滑、蒙特卡洛模拟）")
print("对历史开奖数据的数学建模，仅作为技术研究和娱乐参考。")
print()
print("彩票开奖结果为独立随机事件，历史数据无法预测未来开奖结果。")
print("任何预测分析均不构成投资建议，请您理性购彩，量力而行，切勿沉迷。")
print("=" * 70)
