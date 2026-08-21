"""
统计分析模块 - 大乐透历史数据的多维统计分析

计算冷热号、奇偶比、和值、大小分布、连号、跨度、AC 值等指标。
"""
from typing import Dict, List, Tuple, Optional
from collections import Counter

import numpy as np
import pandas as pd

from config import (
    FRONT_MIN, FRONT_MAX, BACK_MIN, BACK_MAX,
    FRONT_COUNT, BACK_COUNT, TOP_HOT_COUNT, TOP_COLD_COUNT,
)
from utils.helpers import logger


def extract_front_numbers(df: pd.DataFrame) -> np.ndarray:
    """提取所有前区号码"""
    front_cols = [f"front_{i}" for i in range(1, 6)]
    return df[front_cols].values.flatten()


def extract_back_numbers(df: pd.DataFrame) -> np.ndarray:
    """提取所有后区号码"""
    back_cols = [f"back_{i}" for i in range(1, 3)]
    return df[back_cols].values.flatten()


def frequency_analysis(df: pd.DataFrame) -> Dict:
    """
    频率分析 - 计算每个号码的出现次数和频率
    返回前区和后区的频率统计
    """
    # 前区频率
    front_nums = extract_front_numbers(df)
    front_counter = Counter(front_nums)
    front_freq = {n: front_counter.get(n, 0) for n in range(FRONT_MIN, FRONT_MAX + 1)}

    # 后区频率
    back_nums = extract_back_numbers(df)
    back_counter = Counter(back_nums)
    back_freq = {n: back_counter.get(n, 0) for n in range(BACK_MIN, BACK_MAX + 1)}

    total_draws = len(df)
    return {
        "total_draws": total_draws,
        "front_freq": front_freq,
        "back_freq": back_freq,
        "front_freq_pct": {n: cnt / total_draws * 100 for n, cnt in front_freq.items()},
        "back_freq_pct": {n: cnt / total_draws * 100 for n, cnt in back_freq.items()},
    }


def missing_value_analysis(df: pd.DataFrame) -> Dict:
    """
    遗漏值分析 - 计算每个号码当前遗漏期数（最近 N 期未出现的次数）
    使用每期的号码数据，计算从上一次出现到现在的间隔
    """
    # 前区遗漏
    front_cols = [f"front_{i}" for i in range(1, 6)]
    front_missing = {}
    for n in range(FRONT_MIN, FRONT_MAX + 1):
        # 找到该号码最后一次出现的期次索引
        appeared_in = []
        for idx, row in df.iterrows():
            if n in row[front_cols].values:
                appeared_in.append(idx)
        if appeared_in:
            last_idx = appeared_in[0]  # df 已按最新在前排序
            front_missing[n] = last_idx  # 遗漏期数 = 从最新一期到该期之间的期数
        else:
            front_missing[n] = len(df)  # 从未出现

    # 后区遗漏
    back_cols = [f"back_{i}" for i in range(1, 3)]
    back_missing = {}
    for n in range(BACK_MIN, BACK_MAX + 1):
        appeared_in = []
        for idx, row in df.iterrows():
            if n in row[back_cols].values:
                appeared_in.append(idx)
        if appeared_in:
            last_idx = appeared_in[0]
            back_missing[n] = last_idx
        else:
            back_missing[n] = len(df)

    return {
        "front_missing": front_missing,
        "back_missing": back_missing,
    }


def hot_cold_analysis(df: pd.DataFrame, top_n: int = TOP_HOT_COUNT) -> Dict:
    """
    冷热号分析
    热号：最近统计期内出现频率高的号码
    冷号：最近统计期内出现频率低或遗漏时间长的号码
    """
    freq = frequency_analysis(df)
    miss = missing_value_analysis(df)
    total = freq["total_draws"]

    # 前区综合得分 = 频率 z-score 标准化 + 遗漏归一化
    front_scores = {}
    for n in range(FRONT_MIN, FRONT_MAX + 1):
        f = freq["front_freq"][n]
        m = miss["front_missing"][n]
        # 频率得分（越高越好）和遗漏得分（遗漏越大，冷号得分越低）
        freq_score = f / total if total > 0 else 0
        miss_score = 1 - (m / max(miss["front_missing"].values())) if max(miss["front_missing"].values()) > 0 else 0
        front_scores[n] = freq_score * 0.6 + miss_score * 0.4

    sorted_front = sorted(front_scores.items(), key=lambda x: x[1], reverse=True)
    hot_front = [n for n, _ in sorted_front[:top_n]]
    cold_front = [n for n, _ in sorted_front[-top_n:]]

    # 后区
    back_scores = {}
    for n in range(BACK_MIN, BACK_MAX + 1):
        f = freq["back_freq"][n]
        m = miss["back_missing"][n]
        freq_score = f / total if total > 0 else 0
        miss_score = 1 - (m / max(miss["back_missing"].values())) if max(miss["back_missing"].values()) > 0 else 0
        back_scores[n] = freq_score * 0.6 + miss_score * 0.4

    sorted_back = sorted(back_scores.items(), key=lambda x: x[1], reverse=True)
    top_back_hot = min(top_n // 3, BACK_MAX - BACK_MIN + 1)
    hot_back = [n for n, _ in sorted_back[:top_back_hot]]
    cold_back = [n for n, _ in sorted_back[-top_back_hot:]]

    return {
        "hot_front": hot_front,
        "cold_front": cold_front,
        "hot_back": hot_back,
        "cold_back": cold_back,
        "front_scores": front_scores,
        "back_scores": back_scores,
    }


def odd_even_analysis(df: pd.DataFrame) -> Dict:
    """
    奇偶比分析
    前区：奇数/偶数个数比（如 3:2, 2:3, 4:1 等）
    后区：奇数/偶数个数比
    """
    front_cols = [f"front_{i}" for i in range(1, 6)]
    back_cols = [f"back_{i}" for i in range(1, 3)]

    front_ratios = Counter()
    back_ratios = Counter()

    for _, row in df.iterrows():
        front_odd = sum(1 for c in front_cols if int(row[c]) % 2 == 1)
        front_even = FRONT_COUNT - front_odd
        front_ratios[(front_odd, front_even)] += 1

        back_odd = sum(1 for c in back_cols if int(row[c]) % 2 == 1)
        back_even = BACK_COUNT - back_odd
        back_ratios[(back_odd, back_even)] += 1

    total = len(df)
    return {
        "front_ratios": {str(k): {"count": v, "pct": v / total * 100} for k, v in front_ratios.most_common()},
        "back_ratios": {str(k): {"count": v, "pct": v / total * 100} for k, v in back_ratios.most_common()},
        "most_common_front_ratio": front_ratios.most_common(1)[0][0] if front_ratios else (3, 2),
        "most_common_back_ratio": back_ratios.most_common(1)[0][0] if back_ratios else (1, 1),
    }


def sum_analysis(df: pd.DataFrame) -> Dict:
    """
    和值分析 - 前区号码和值、后区号码和值的分布
    前区和值范围: 15(1+2+3+4+5) ~ 165(31+32+33+34+35)
    """
    front_cols = [f"front_{i}" for i in range(1, 6)]
    back_cols = [f"back_{i}" for i in range(1, 3)]

    front_sums = df[front_cols].sum(axis=1)
    back_sums = df[back_cols].sum(axis=1)

    return {
        "front_sum": {
            "mean": float(front_sums.mean()),
            "median": float(front_sums.median()),
            "std": float(front_sums.std()),
            "min": int(front_sums.min()),
            "max": int(front_sums.max()),
            "recent": [int(x) for x in front_sums.head(10).values],
        },
        "back_sum": {
            "mean": float(back_sums.mean()),
            "median": float(back_sums.median()),
            "std": float(back_sums.std()),
            "min": int(back_sums.min()),
            "max": int(back_sums.max()),
            "recent": [int(x) for x in back_sums.head(10).values],
        },
    }


def size_analysis(df: pd.DataFrame) -> Dict:
    """
    大小号分布分析
    前区: 01-17 为小号, 18-35 为大号（17/18分界）
    """
    front_cols = [f"front_{i}" for i in range(1, 6)]

    front_small_large = Counter()
    for _, row in df.iterrows():
        small = sum(1 for c in front_cols if int(row[c]) <= 17)
        large = FRONT_COUNT - small
        front_small_large[(small, large)] += 1

    total = len(df)
    result = {
        "front_small_large_ratios": {str(k): {"count": v, "pct": v / total * 100}
                                      for k, v in front_small_large.most_common()},
    }

    # 最常出现的比例
    if front_small_large:
        result["most_common_ratio"] = front_small_large.most_common(1)[0][0]

    return result


def consecutive_analysis(df: pd.DataFrame) -> Dict:
    """
    连号分析 - 统计前区中出现连号的期数和形式
    连号：相邻号码（如 05 06, 18 19 20）
    """
    front_cols = [f"front_{i}" for i in range(1, 6)]

    has_consecutive = 0
    max_consecutive_count = 0
    consecutive_pairs_count = 0

    for _, row in df.iterrows():
        nums = sorted([int(row[c]) for c in front_cols])
        consec = sum(1 for i in range(len(nums) - 1) if nums[i + 1] - nums[i] == 1)

        if consec > 0:
            has_consecutive += 1
            consecutive_pairs_count += consec

        # 最大连号长度
        max_len = 1
        cur_len = 1
        for i in range(len(nums) - 1):
            if nums[i + 1] - nums[i] == 1:
                cur_len += 1
                max_len = max(max_len, cur_len)
            else:
                cur_len = 1
        max_consecutive_count = max(max_consecutive_count, max_len)

    total = len(df)
    return {
        "has_consecutive_pct": has_consecutive / total * 100,
        "avg_consecutive_pairs_per_draw": consecutive_pairs_count / total if total > 0 else 0,
        "max_consecutive_count": max_consecutive_count,
        "consecutive_draws": has_consecutive,
        "total_draws": total,
    }


def span_analysis(df: pd.DataFrame) -> Dict:
    """
    跨度分析
    前区跨度 = 最大号 - 最小号（范围 4~34）
    """
    front_cols = [f"front_{i}" for i in range(1, 6)]

    spans = []
    for _, row in df.iterrows():
        nums = [int(row[c]) for c in front_cols]
        spans.append(max(nums) - min(nums))

    return {
        "mean": float(np.mean(spans)),
        "median": float(np.median(spans)),
        "std": float(np.std(spans)),
        "min": int(min(spans)),
        "max": int(max(spans)),
        "recent": spans[:10],
    }


def ac_value_analysis(df: pd.DataFrame) -> Dict:
    """
    AC 值（算术复杂度）分析
    AC值 = 号码间正差值个数 - (r-1)，其中 r 为选号个数（前区 r=5）
    AC值范围：0~10，反映号码的离散程度
    """
    front_cols = [f"front_{i}" for i in range(1, 6)]

    ac_values = []
    for _, row in df.iterrows():
        nums = sorted([int(row[c]) for c in front_cols])
        diffs = set()
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                diffs.add(nums[j] - nums[i])
        ac = len(diffs) - (FRONT_COUNT - 1)
        ac_values.append(max(0, ac))  # AC 值不能为负

    return {
        "mean": float(np.mean(ac_values)),
        "median": float(np.median(ac_values)),
        "std": float(np.std(ac_values)),
        "min": int(min(ac_values)),
        "max": int(max(ac_values)),
        "distribution": Counter(ac_values),
        "recent": ac_values[:10],
    }


def comprehensive_analysis(df: pd.DataFrame) -> Dict:
    """
    综合分析 - 运行所有分析并汇总结果
    """
    logger.info("正在执行综合分析...")

    result = {
        "summary": {
            "total_draws": len(df),
            "date_range": f"{df['period'].iloc[-1]} ~ {df['period'].iloc[0]}",
        },
        "frequency": frequency_analysis(df),
        "missing": missing_value_analysis(df),
        "hot_cold": hot_cold_analysis(df),
        "odd_even": odd_even_analysis(df),
        "sum_analysis": sum_analysis(df),
        "size_dist": size_analysis(df),
        "consecutive": consecutive_analysis(df),
        "span": span_analysis(df),
        "ac_value": ac_value_analysis(df),
    }

    logger.info("综合分析完成")
    return result
