#!/usr/bin/env python3
"""Cron job: extract detailed analysis data for report generation."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_fetcher import load_from_csv
from analysis.statistics import frequency_analysis, hot_cold_analysis, missing_value_analysis

df = load_from_csv()
if df is None or len(df) == 0:
    print("NO_DATA")
    sys.exit(1)

front_cols = [f'front_{i}' for i in range(1, 6)]
back_cols = [f'back_{i}' for i in range(1, 3)]

# Data info
print(f'DRAW_COUNT={len(df)}')
print(f'LATEST_PERIOD={df.iloc[0]["period"]}')
latest_front = ' '.join(f'{df.iloc[0][c]:02d}' for c in front_cols)
latest_back = ' '.join(f'{df.iloc[0][c]:02d}' for c in back_cols)
print(f'LATEST_DRAW={latest_front} + {latest_back}')
print(f'OLDEST_PERIOD={df.iloc[-1]["period"]}')

# Frequency
freq = frequency_analysis(df)

# Front hot by frequency
front_sorted = sorted(freq["front_freq"].items(), key=lambda x: x[1], reverse=True)
print('---FRONT_HOT---')
for n, cnt in front_sorted[:10]:
    pct = cnt / len(df) * 100
    print(f'{n:02d}:{cnt}:{pct:.1f}')

# Back hot by frequency
back_sorted = sorted(freq["back_freq"].items(), key=lambda x: x[1], reverse=True)
print('---BACK_HOT---')
for n, cnt in back_sorted[:5]:
    pct = cnt / len(df) * 100
    print(f'{n:02d}:{cnt}:{pct:.1f}')

# Missing
miss = missing_value_analysis(df)
print('---FRONT_MISSING---')
front_miss = sorted(miss["front_missing"].items(), key=lambda x: x[1], reverse=True)
for n, m in front_miss[:5]:
    print(f'{n:02d}:{m}')
print('---BACK_MISSING---')
back_miss = sorted(miss["back_missing"].items(), key=lambda x: x[1], reverse=True)
for n, m in back_miss[:5]:
    print(f'{n:02d}:{m}')

# Hot-cold comprehensive
hc = hot_cold_analysis(df)
print('---HC_FRONT_HOT---')
print(','.join(f'{n:02d}' for n in hc['hot_front'][:10]))
print('---HC_BACK_HOT---')
print(','.join(f'{n:02d}' for n in hc['hot_back'][:5]))
print('---HC_FRONT_COLD---')
print(','.join(f'{n:02d}' for n in hc['cold_front'][:5]))
print('---HC_BACK_COLD---')
print(','.join(f'{n:02d}' for n in hc['cold_back'][:3]))

# Recent 10 draws
print('---RECENT_10---')
for i in range(min(10, len(df))):
    row = df.iloc[i]
    f = ' '.join(f'{row[c]:02d}' for c in front_cols)
    b = ' '.join(f'{row[c]:02d}' for c in back_cols)
    print(f'{row["period"]} | {f} + {b}')
