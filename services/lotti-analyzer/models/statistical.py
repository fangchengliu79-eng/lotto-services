"""
统计模型模块 - 频率分析、泊松分布建模、蒙特卡洛模拟
"""
from typing import Dict, List, Tuple, Optional
import math
import random
from collections import Counter
from functools import lru_cache

import numpy as np
import pandas as pd

from config import (
    FRONT_MIN, FRONT_MAX, BACK_MIN, BACK_MAX,
    FRONT_COUNT, BACK_COUNT, TOP_HOT_COUNT, TOP_COLD_COUNT,
)
from analysis.statistics import frequency_analysis, missing_value_analysis
from utils.helpers import logger


class FrequencyModel:
    """频率分析模型 - 基于历史出现频率的加权推荐"""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.freq = frequency_analysis(df)

    def predict_front_probabilities(self) -> Dict[int, float]:
        """返回前区每个号码的出现概率（归一化）"""
        total = sum(self.freq["front_freq"].values())
        if total == 0:
            return {n: 1 / 35 for n in range(FRONT_MIN, FRONT_MAX + 1)}
        return {n: cnt / total for n, cnt in self.freq["front_freq"].items()}

    def predict_back_probabilities(self) -> Dict[int, float]:
        """返回后区每个号码的出现概率（归一化）"""
        total = sum(self.freq["back_freq"].values())
        if total == 0:
            return {n: 1 / 12 for n in range(BACK_MIN, BACK_MAX + 1)}
        return {n: cnt / total for n, cnt in self.freq["back_freq"].items()}

    def score_combination(self, front: List[int], back: List[int]) -> float:
        """评估一组号码的得分（基于频率加权）"""
        front_probs = self.predict_front_probabilities()
        back_probs = self.predict_back_probabilities()
        score = sum(front_probs.get(n, 0) for n in front) / FRONT_COUNT
        score += sum(back_probs.get(n, 0) for n in back) / BACK_COUNT
        return score


class PoissonModel:
    """
    泊松分布模型 - 对遗漏值进行泊松建模

    原理：号码出现的次数在固定期数内服从泊松分布。
    当号码遗漏期数远超期望间隔时，其出现概率相对较高。
    P(出现) = 1 - e^(-λ)，其中 λ = 1/期望间隔
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.missing = missing_value_analysis(df)
        self.freq = frequency_analysis(df)
        self.total_draws = len(df)

    def _expected_interval(self, num: int, zone: str = "front") -> float:
        """计算号码的期望出现间隔"""
        if zone == "front":
            freq = self.freq["front_freq"].get(num, 0)
        else:
            freq = self.freq["back_freq"].get(num, 0)

        if freq == 0:
            return self.total_draws + 1
        return self.total_draws / freq

    def _poisson_prob(self, k: int, lam: float) -> float:
        """泊松分布 PMF: P(X=k) = λ^k * e^(-λ) / k!"""
        if lam <= 0:
            return 0
        return (lam ** k) * math.exp(-lam) / math.factorial(k)

    def predict_front_probabilities(self) -> Dict[int, float]:
        """使用泊松分布预测前区号码出现概率"""
        probs = {}
        for n in range(FRONT_MIN, FRONT_MAX + 1):
            expected = self._expected_interval(n, "front")
            if expected <= 0:
                lam = 1.0
            else:
                lam = 1.0 / expected

            missing_periods = self.missing["front_missing"].get(n, 0)
            # 给定遗漏期数后的条件概率 = P(X <= missing_periods) 的补集
            # 即：遗漏了 missing_periods 期后，下一期出现的概率
            # 使用泊松累积分布
            prob = 0
            if lam > 0:
                # P(出现) ≈ 1 - P(X=0) where λ = missing_periods/expected
                # 更准确：P(在下一期出现 | 已遗漏k期) 需要条件概率
                # 简化：概率与遗漏期数成正比，但受泊松约束
                prob = min(1.0, missing_periods * lam * 1.5)

            probs[n] = prob

        # 归一化
        total = sum(probs.values())
        if total > 0:
            probs = {n: p / total for n, p in probs.items()}
        return probs

    def predict_back_probabilities(self) -> Dict[int, float]:
        """使用泊松分布预测后区号码出现概率"""
        probs = {}
        for n in range(BACK_MIN, BACK_MAX + 1):
            expected = self._expected_interval(n, "back")
            lam = 1.0 / max(expected, 0.1)
            missing_periods = self.missing["back_missing"].get(n, 0)
            prob = min(1.0, missing_periods * lam * 1.5)
            probs[n] = prob

        total = sum(probs.values())
        if total > 0:
            probs = {n: p / total for n, p in probs.items()}
        return probs


class MonteCarloModel:
    """
    蒙特卡洛模拟模型 - 使用加权随机组合生成大量候选

    原理：根据历史频率为每个号码赋予权重，然后随机生成大量
    组合，筛选出得分最高的组合。
    """

    def __init__(self, df: pd.DataFrame, num_simulations: int = 100000):
        self.df = df
        self.num_simulations = num_simulations
        self.freq = frequency_analysis(df)
        self._build_weights()

    def _build_weights(self):
        """根据频率构建权重向量"""
        # 前区权重
        front_total = sum(self.freq["front_freq"].values())
        self.front_weights = [
            self.freq["front_freq"].get(n, 1) / front_total
            for n in range(FRONT_MIN, FRONT_MAX + 1)
        ]
        # 平滑处理：避免概率为0
        self.front_weights = [max(w, 0.001) for w in self.front_weights]

        # 后区权重
        back_total = sum(self.freq["back_freq"].values())
        self.back_weights = [
            self.freq["back_freq"].get(n, 1) / back_total
            for n in range(BACK_MIN, BACK_MAX + 1)
        ]
        self.back_weights = [max(w, 0.001) for w in self.back_weights]

    def _weighted_sample_front(self) -> List[int]:
        """带权重的无放回前区抽样"""
        pool = list(range(FRONT_MIN, FRONT_MAX + 1))
        weights = self.front_weights.copy()
        selected = []
        for _ in range(FRONT_COUNT):
            if sum(weights) == 0:
                idx = random.randrange(len(pool))
            else:
                idx = random.choices(range(len(pool)), weights=weights, k=1)[0]
            selected.append(pool[idx])
            # 移除已选中的权重
            pool.pop(idx)
            weights.pop(idx)
        return sorted(selected)

    def _weighted_sample_back(self) -> List[int]:
        """带权重的无放回后区抽样"""
        pool = list(range(BACK_MIN, BACK_MAX + 1))
        weights = self.back_weights.copy()
        selected = []
        for _ in range(BACK_COUNT):
            if sum(weights) == 0:
                idx = random.randrange(len(pool))
            else:
                idx = random.choices(range(len(pool)), weights=weights, k=1)[0]
            selected.append(pool[idx])
            pool.pop(idx)
            weights.pop(idx)
        return sorted(selected)

    def _score_combination(self, front: List[int], back: List[int]) -> float:
        """评分函数 - 基于权重的得分"""
        front_all = list(range(FRONT_MIN, FRONT_MAX + 1))
        back_all = list(range(BACK_MIN, BACK_MAX + 1))

        front_score = sum(self.front_weights[front_all.index(n)] for n in front)
        back_score = sum(self.back_weights[back_all.index(n)] for n in back)

        return front_score + back_score

    def simulate(self, top_n: int = 100) -> List[Tuple[List[int], List[int], float]]:
        """
        运行蒙特卡洛模拟，返回得分最高的组合
        """
        logger.info(f"运行蒙特卡洛模拟: {self.num_simulations} 次模拟...")
        results = []

        for i in range(self.num_simulations):
            front = self._weighted_sample_front()
            back = self._weighted_sample_back()
            score = self._score_combination(front, back)
            results.append((front, back, score))

        # 按得分降序排列
        results.sort(key=lambda x: x[2], reverse=True)
        logger.info(f"蒙特卡洛模拟完成，已筛选出前 {top_n} 组")
        return results[:top_n]

    def get_top_probabilities(self, top_n: int = 100) -> Dict:
        """从模拟结果中统计每个号码的出现频率"""
        results = self.simulate(top_n)

        front_counter = Counter()
        back_counter = Counter()
        for front, back, _ in results:
            for n in front:
                front_counter[n] += 1
            for n in back:
                back_counter[n] += 1

        front_probs = {n: front_counter[n] / top_n for n in range(FRONT_MIN, FRONT_MAX + 1)}
        back_probs = {n: back_counter[n] / top_n for n in range(BACK_MIN, BACK_MAX + 1)}

        return {"front": front_probs, "back": back_probs}
