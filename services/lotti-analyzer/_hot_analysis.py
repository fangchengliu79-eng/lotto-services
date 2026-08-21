import sys
sys.path.insert(0, '/root/lotti-analyzer')
from data_fetcher import update_data
from analysis.statistics import frequency_analysis, hot_cold_analysis

df = update_data(force_refresh=False)
print(f"总期数: {len(df)}")
latest = df.iloc[0]
front_cols = [f"front_{i}" for i in range(1, 6)]
back_cols = [f"back_{i}" for i in range(1, 3)]
print(f"最新期号: {latest['period']}")
print(f"最新开奖: {' '.join(f'{latest[c]:02d}' for c in front_cols)} + {' '.join(f'{latest[c]:02d}' for c in back_cols)}")
print()

freq = frequency_analysis(df)

print("=== 前区热号 TOP 10 (按出频排序) ===")
front_sorted = sorted(freq['front_freq'].items(), key=lambda x: x[1], reverse=True)
for n, cnt in front_sorted[:10]:
    pct = freq['front_freq_pct'][n]
    print(f"  {n:02d}: {cnt}次 ({pct:.1f}%)")

print()
print("=== 后区热号 TOP 5 (按出频排序) ===")
back_sorted = sorted(freq['back_freq'].items(), key=lambda x: x[1], reverse=True)
for n, cnt in back_sorted[:5]:
    pct = freq['back_freq_pct'][n]
    print(f"  {n:02d}: {cnt}次 ({pct:.1f}%)")

hc = hot_cold_analysis(df)
print()
print("=== 前区热号 (综合评分) TOP 10 ===")
vals = []
for n in hc['hot_front'][:10]:
    vals.append(f"{n:02d}")
print("  " + " ".join(vals))

print("=== 后区热号 (综合评分) TOP 5 ===")
vals = []
for n in hc['hot_back'][:5]:
    vals.append(f"{n:02d}")
print("  " + " ".join(vals))
