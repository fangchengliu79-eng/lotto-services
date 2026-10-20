"""
时间序列模型 - 指数平滑/移动平均预测号码出现概率趋势

使用指数平滑 (Exponential Smoothing) 和移动平均对历史频率
进行时序分析，捕捉号码出现概率的近期变化趋势。
"""
from typing import Dict, List, Tuple
from collections import Counter

import numpy as np
import pandas as pd

from config import FRONT_MIN, FRONT_MAX, BACK_MIN, BACK_MAX, FRONT_COUNT, BACK_COUNT
from utils.helpers import logger


class ExponentialSmoothingModel:
    """
    指数平滑模型 - 对每个号码的出现概率进行加权平滑预测

    原理：给近期数据更高的权重，越旧的数据权重越小。
    公式：s_t = α * x_t + (1-α) * s_{t-1}
    其中 α 为平滑系数（0 < α < 1），x_t 为当期观测值。
    """

    def __init__(self, df: pd.DataFrame, alpha: float = 0.3):
        """
        alpha: 平滑系数，越大越关注近期数据
        """
        self.df = df
        self.alpha = alpha
        self.total_draws = len(df)
        self._compute_smoothed()

    def _compute_smoothed(self):
        """
        对每个号码应用指数平滑
        使用滑动窗口方式：每 N 期作为一个时间点，统计出现次数
        """
        n_windows = min(50, self.total_draws // 20)  # 约20期一个窗口
        if n_windows < 3:
            n_windows = max(1, self.total_draws // 5)

        window_size = max(1, self.total_draws // max(n_windows, 1))
        windows = []
        for i in range(0, self.total_draws, window_size):
            chunk = self.df.iloc[i:i + window_size]
            windows.append(chunk)

        # 对每个号码计算指数平滑值
        front_cols = [f"front_{i}" for i in range(1, 6)]
        back_cols = [f"back_{i}" for i in range(1, 3)]

        # 前区平滑
        front_smoothed = {}
        for n in range(FRONT_MIN, FRONT_MAX + 1):
            values = []
            for w in windows:
                count = sum(1 for _, r in w.iterrows() if n in r[front_cols].values)
                values.append(count / len(w))

            # 指数平滑
            s = values[0] if values else 0
            for v in values[1:]:
                s = self.alpha * v + (1 - self.alpha) * s
            front_smoothed[n] = s

        # 后区平滑
        back_smoothed = {}
        for n in range(BACK_MIN, BACK_MAX + 1):
            values = []
            for w in windows:
                count = sum(1 for _, r in w.iterrows() if n in r[back_cols].values)
                values.append(count / len(w))

            s = values[0] if values else 0
            for v in values[1:]:
                s = self.alpha * v + (1 - self.alpha) * s
            back_smoothed[n] = s

        # 归一化
        front_total = sum(front_smoothed.values())
        if front_total > 0:
            self.front_probs = {n: v / front_total for n, v in front_smoothed.items()}
        else:
            self.front_probs = {n: 1 / 35 for n in range(FRONT_MIN, FRONT_MAX + 1)}

        back_total = sum(back_smoothed.values())
        if back_total > 0:
            self.back_probs = {n: v / back_total for n, v in back_smoothed.items()}
        else:
            self.back_probs = {n: 1 / 12 for n in range(BACK_MIN, BACK_MAX + 1)}

    def predict_front_probabilities(self) -> Dict[int, float]:
        """返回前区预测概率"""
        return self.front_probs

    def predict_back_probabilities(self) -> Dict[int, float]:
        """返回后区预测概率"""
        return self.back_probs

    def get_recent_trend(self, n_windows: int = 10) -> Dict[str, List[float]]:
        """
        获取近期趋势数据，用于分析号码热度变化
        """
        window_size = max(1, self.total_draws // max(n_windows, 1))
        front_cols = [f"front_{i}" for i in range(1, 6)]

        # 只取最热的几个号看趋势
        hot_numbers = sorted(self.front_probs.items(), key=lambda x: x[1], reverse=True)[:8]
        trends = {}
        for num, _ in hot_numbers:
            trend = []
            for i in range(0, self.total_draws, window_size):
                chunk = self.df.iloc[i:i + window_size]
                count = sum(1 for _, r in chunk.iterrows() if num in r[front_cols].values)
                trend.append(count / len(chunk))
            trends[f"号{num:02d}"] = trend

        return trends


class MovingAverageModel:
    """
    移动平均模型 - 使用简单移动平均预测号码出现概率

    原理：取最近 N 期的出现频率作为下一期的预测值。
    比指数平滑更直观，但同等参数下对近期变化的响应较慢。
    """

    def __init__(self, df: pd.DataFrame, window: int = 30):
        """
        window: 移动平均窗口大小
        """
        self.df = df
        self.window = min(window, max(10, len(df)))
        self.front_probs = {}
        self.back_probs = {}
        self._compute_ma()

    def _compute_ma(self):
        """计算移动平均"""
        recent_df = self.df.head(self.window)  # 最新的 N 期
        front_cols = [f"front_{i}" for i in range(1, 6)]
        back_cols = [f"back_{i}" for i in range(1, 3)]

        # 前区
        front_counts = {}
        for n in range(FRONT_MIN, FRONT_MAX + 1):
            front_counts[n] = sum(1 for _, r in recent_df.iterrows()
                                  if n in r[front_cols].values)

        total = sum(front_counts.values())
        self.front_probs = {n: c / total for n, c in front_counts.items()} if total > 0 \
            else {n: 1 / 35 for n in range(FRONT_MIN, FRONT_MAX + 1)}

        # 后区
        back_counts = {}
        for n in range(BACK_MIN, BACK_MAX + 1):
            back_counts[n] = sum(1 for _, r in recent_df.iterrows()
                                 if n in r[back_cols].values)

        total = sum(back_counts.values())
        self.back_probs = {n: c / total for n, c in back_counts.items()} if total > 0 \
            else {n: 1 / 12 for n in range(BACK_MIN, BACK_MAX + 1)}

    def predict_front_probabilities(self) -> Dict[int, float]:
        return self.front_probs

    def predict_back_probabilities(self) -> Dict[int, float]:
        return self.back_probs
