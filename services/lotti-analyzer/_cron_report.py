#!/usr/bin/env python3
"""Generate detailed report for cron job delivery."""
import pandas as pd
from collections import Counter
from config import HISTORY_CSV, ENSEMBLE_WEIGHTS
from analysis.statistics import missing_value_analysis

df = pd.read_csv(HISTORY_CSV, dtype={"period": str})
latest = df.iloc[0]

# Latest draw
front_cols = [f"front_{i}" for i in range(1, 6)]
back_cols = [f"back_{i}" for i in range(1, 3)]
latest_front = [int(latest[c]) for c in front_cols]
latest_back = [int(latest[c]) for c in back_cols]

print("=" * 58)
print("  大乐透数据分析与预测系统 - 自动分析报告")
print("=" * 58)
print()
print(f"  报告时间: 2026-07-07 09:01")
print(f"  数据范围: 最近 {len(df)} 期")
print(f"  期号区间: {df['period'].iloc[-1]} ~ {df['period'].iloc[0]}")
print(f"  最新期号: {latest['period']}")
print(f"  最新开奖: {' '.join(f'{n:02d}' for n in latest_front)} + {' '.join(f'{n:02d}' for n in latest_back)}")
print()

# --- 前区热号 TOP10 ---
all_front = []
for _, row in df.iterrows():
    all_front.extend([int(row[c]) for c in front_cols])
front_counter = Counter(all_front)

print("─" * 58)
print("  前区热号 TOP10")
print("─" * 58)
for i, (n, cnt) in enumerate(front_counter.most_common(10), 1):
    print(f"  {i:2d}. {n:02d}  出现 {cnt} 次  {cnt/len(df)*100:.1f}%")
print()

# --- 后区热号 TOP5 ---
all_back = []
for _, row in df.iterrows():
    all_back.extend([int(row[c]) for c in back_cols])
back_counter = Counter(all_back)

print("─" * 58)
print("  后区热号 TOP5")
print("─" * 58)
for i, (n, cnt) in enumerate(back_counter.most_common(5), 1):
    print(f"  {i:2d}. {n:02d}  出现 {cnt} 次  {cnt/len(df)*100:.1f}%")
print()

# --- 冷号 ---
miss = missing_value_analysis(df)
cold_front = sorted(miss["front_missing"].items(), key=lambda x: x[1], reverse=True)[:5]
cold_back = sorted(miss["back_missing"].items(), key=lambda x: x[1], reverse=True)[:5]

print("─" * 58)
print("  前区冷号（遗漏最多）")
print("─" * 58)
for n, m in cold_front:
    print(f"  {n:02d}  遗漏 {m} 期")
print()

print("─" * 58)
print("  后区冷号（遗漏最多）")
print("─" * 58)
for n, m in cold_back:
    print(f"  {n:02d}  遗漏 {m} 期")
print()

# --- Model weights used ---
stat_weights = {k: v for k, v in ENSEMBLE_WEIGHTS.items()
                if k in ["frequency", "poisson", "monte_carlo", "exponential_smoothing"]}
total_w = sum(stat_weights.values())

print("─" * 58)
print("  模型融合权重（仅统计模型）")
print("─" * 58)
for k, v in stat_weights.items():
    labels = {
        "frequency": "频率分析",
        "poisson": "泊松分布",
        "monte_carlo": "蒙特卡洛模拟",
        "exponential_smoothing": "指数平滑",
    }
    print(f"  {labels.get(k, k):12s}  {v:.2f}  ({v/total_w*100:.0f}%)")
print()

# --- Statistical summary ---
print("─" * 58)
print("  关键统计指标")
print("─" * 58)
from analysis.statistics import comprehensive_analysis
analysis = comprehensive_analysis(df)
summ = analysis["summary"]
stats = analysis["sum_analysis"]
print(f"  前区平均和值: {stats['front_sum']['mean']:.1f}  (范围 {stats['front_sum']['min']}~{stats['front_sum']['max']})")
print(f"  前区平均跨度: {analysis['span']['mean']:.1f}")
print(f"  AC值均值:     {analysis['ac_value']['mean']:.1f}")
print(f"  连号出现率:   {analysis['consecutive']['has_consecutive_pct']:.1f}%")
print(f"  常见奇偶比:   {analysis['odd_even']['most_common_front_ratio'][0]}奇{analysis['odd_even']['most_common_front_ratio'][1]}偶")
print()
