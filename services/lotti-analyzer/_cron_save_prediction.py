#!/usr/bin/env python3
"""保存最新预测到 predictions.json 并比对历史"""
import sys, os
sys.path.insert(0, "/root/lotti-analyzer")
import json
from datetime import datetime
from data_fetcher import load_from_csv
from utils.predictions import save_prediction, auto_compare_latest
from config import DATA_DIR

df = load_from_csv()

# 先比对已有预测
result = auto_compare_latest(df)
if result:
    print(f"已比对预测: 期{result['period']}")

# 当前最新期号
latest_period = int(df.iloc[0]["period"])
next_period = str(latest_period + 1)
print(f"本期预测期号: {next_period}")

# 读取刚刚生成的8组推荐
recommendations = [
    {"front": [5,13,20,26,34], "back": [2,3], "score": 0.3750, "reason": "频率+蒙特卡洛权重最高 | 奇偶2:3 和值适中(98) | 后区奇偶"},
    {"front": [6,7,12,17,20], "back": [5,8], "score": 0.3564, "reason": "指数平滑推荐连号06-07 | 奇偶2:3 和值62 | 后区奇偶"},
    {"front": [2,3,7,22,24], "back": [11,12], "score": 0.3518, "reason": "3个热号(02/03/07)+冷号22 | 奇偶2:3 和值58 | 后区奇偶"},
    {"front": [4,11,15,19,29], "back": [4,8], "score": 0.3391, "reason": "泊松模型推荐冷号补位 | 奇偶4:1 和值78 | 后区双偶"},
    {"front": [1,10,13,25,33], "back": [7,8], "score": 0.2756, "reason": "频率+指数平滑推荐 | 1冷号补位 | 奇偶4:1 和值82 | 后区奇偶"},
    {"front": [3,12,18,27,31], "back": [2,7], "score": 0.2589, "reason": "蒙特卡洛推荐跨度大组合 | 1冷号补位 | 奇偶3:2 和值91"},
    {"front": [8,29,32,33,35], "back": [8,12], "score": 0.2340, "reason": "高频尾号28-35区域 | 奇偶3:2 和值137 | 后区双偶"},
    {"front": [5,9,14,22,30], "back": [2,12], "score": 0.2165, "reason": "多样组合冷热均衡 | 1冷号补位 | 奇偶2:3 和值80"},
]

pred_id = save_prediction(
    period=next_period,
    draw_date=datetime.now().strftime("%Y-%m-%d"),
    recommendations=recommendations,
    models_used=["frequency", "poisson", "exponential_smoothing", "monte_carlo"],
)
print(f"预测已封存: {pred_id}")
