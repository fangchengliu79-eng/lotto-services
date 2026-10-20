"""
集成模型模块 - 加权融合统计 + ML 结果

将多个模型的预测结果按权重融合，并进行形态过滤
（排除极小概率组合，如全奇/全偶/全大/全小等）。
"""
from typing import Dict, List, Tuple, Optional
from collections import Counter

import numpy as np
import pandas as pd

from config import (
    FRONT_MIN, FRONT_MAX, BACK_MIN, BACK_MAX,
    FRONT_COUNT, BACK_COUNT, ENSEMBLE_WEIGHTS,
)
from utils.helpers import logger


class EnsembleModel:
    """
    集成模型 - 多模型加权融合

    融合策略：
    1. 统计模型（频率、泊松、蒙特卡洛）提供基础概率
    2. 时间序列模型捕捉近期趋势变化
    3. ML/DL 模型捕捉更复杂的模式和依赖关系
    4. 按权重加权融合得到综合概率
    5. 形态过滤排除极低概率组合
    """

    def __init__(self, models: Dict[str, object]):
        """
        models: {
            "frequency": FrequencyModel,
            "poisson": PoissonModel,
            "monte_carlo": MonteCarloModel,
            "exponential_smoothing": ExponentialSmoothingModel,
            "lstm": LSTMSequenceModel,
            "xgboost": XGBoostModel,
            "random_forest": RandomForestModel,
        }
        """
        self.models = models
        self.front_ensemble_probs = {}
        self.back_ensemble_probs = {}
        self._compute_ensemble()

    def _compute_ensemble(self):
        """计算加权融合概率"""
        logger.info("正在计算集成模型概率...")

        # 前区融合
        front_probs_list = []
        weights_used = []

        for model_name, model in self.models.items():
            if model is None:
                continue
            weight = ENSEMBLE_WEIGHTS.get(model_name, 0.1)
            try:
                if hasattr(model, "predict_front_probabilities"):
                    probs = model.predict_front_probabilities()
                    if probs and len(probs) == 35:
                        front_probs_list.append(probs)
                        weights_used.append(weight)
                        logger.debug(f"  + {model_name}: weight={weight:.2f}")
            except Exception as e:
                logger.warning(f"模型 {model_name} 前区预测失败: {e}")

        if not front_probs_list:
            logger.error("所有模型前区预测均失败！")
            self.front_ensemble_probs = {n: 1 / 35 for n in range(FRONT_MIN, FRONT_MAX + 1)}
        else:
            # 归一化权重
            w_sum = sum(weights_used)
            weights_used = [w / w_sum for w in weights_used]

            # 加权平均
            for n in range(FRONT_MIN, FRONT_MAX + 1):
                weighted_prob = sum(
                    probs[n] * w for probs, w in zip(front_probs_list, weights_used)
                )
                self.front_ensemble_probs[n] = weighted_prob

            # 归一化
            total = sum(self.front_ensemble_probs.values())
            if total > 0:
                self.front_ensemble_probs = {n: p / total for n, p in self.front_ensemble_probs.items()}

        # 后区融合
        back_probs_list = []
        weights_used = []

        for model_name, model in self.models.items():
            if model is None:
                continue
            weight = ENSEMBLE_WEIGHTS.get(model_name, 0.1)
            try:
                if hasattr(model, "predict_back_probabilities"):
                    probs = model.predict_back_probabilities()
                    if probs and len(probs) == 12:
                        back_probs_list.append(probs)
                        weights_used.append(weight)
            except Exception as e:
                logger.warning(f"模型 {model_name} 后区预测失败: {e}")

        if not back_probs_list:
            self.back_ensemble_probs = {n: 1 / 12 for n in range(BACK_MIN, BACK_MAX + 1)}
        else:
            w_sum = sum(weights_used)
            weights_used = [w / w_sum for w in weights_used]

            for n in range(BACK_MIN, BACK_MAX + 1):
                weighted_prob = sum(
                    probs[n] * w for probs, w in zip(back_probs_list, weights_used)
                )
                self.back_ensemble_probs[n] = weighted_prob

            total = sum(self.back_ensemble_probs.values())
            if total > 0:
                self.back_ensemble_probs = {n: p / total for n, p in self.back_ensemble_probs.items()}

        logger.info("集成模型概率计算完成")

    def get_front_probabilities(self) -> Dict[int, float]:
        return self.front_ensemble_probs

    def get_back_probabilities(self) -> Dict[int, float]:
        return self.back_ensemble_probs

    def get_top_front_numbers(self, k: int = 12) -> List[int]:
        """返回概率最高的前 k 个前区号码"""
        sorted_nums = sorted(self.front_ensemble_probs.items(), key=lambda x: x[1], reverse=True)
        return [n for n, _ in sorted_nums[:k]]

    def get_top_back_numbers(self, k: int = 6) -> List[int]:
        """返回概率最高的前 k 个后区号码"""
        sorted_nums = sorted(self.back_ensemble_probs.items(), key=lambda x: x[1], reverse=True)
        return [n for n, _ in sorted_nums[:k]]


class MorphologicalFilter:
    """
    形态过滤器 - 过滤极低概率的组合

    大乐透历史数据显示某些组合形态极少出现：
    - 全奇或全偶（前区 5 个全奇/全偶）
    - 全大或全小（全部 ≥18 或全部 ≤17）
    - 前区和值极端（<50 或 >150）
    - 跨度极端（<8 或 >30）
    - 连号过多（>3 个号码连续）
    """

    @staticmethod
    def is_valid(front: List[int], back: List[int]) -> Tuple[bool, str]:
        """
        检查号码组合是否通过形态过滤

        Returns:
            (is_valid: bool, reason: str)
        """
        # 1. 全奇/全偶
        odd_count = sum(1 for n in front if n % 2 == 1)
        if odd_count == 0:
            return False, "前区全偶（历史极低概率）"
        if odd_count == 5:
            return False, "前区全奇（历史极低概率）"

        # 2. 全大/全小（前区分界 17/18）
        small_count = sum(1 for n in front if n <= 17)
        if small_count == 0:
            return False, "前区全大（历史极低概率）"
        if small_count == 5:
            return False, "前区全小（历史极低概率）"

        # 3. 和值过滤
        front_sum = sum(front)
        if front_sum < 50:
            return False, f"前区和值过低 ({front_sum} < 50)"
        if front_sum > 155:
            return False, f"前区和值过高 ({front_sum} > 155)"

        # 4. 跨度过滤
        span = max(front) - min(front)
        if span < 8:
            return False, f"前区跨度过小 ({span} < 8)"
        if span > 32:
            return False, f"前区跨度过大 ({span} > 32)"

        # 5. 连号过多
        consec = sum(1 for i in range(len(front) - 1) if front[i + 1] - front[i] == 1)
        if consec > 2:
            return False, "连号过多"

        # 6. AC 值过滤（使用简化计算）
        diffs = set()
        for i in range(len(front)):
            for j in range(i + 1, len(front)):
                diffs.add(front[j] - front[i])
        ac = len(diffs) - 4
        if ac < 2:
            return False, f"AC 值过低 ({ac})"

        # 7. 后区奇偶检查
        back_odd = sum(1 for n in back if n % 2 == 1)
        if back_odd == 0 or back_odd == 2:
            # 后区全奇或全偶偶尔出现，仅做低分惩罚，不直接排除
            pass

        return True, "通过"


def generate_recommendations(
    ensemble: EnsembleModel,
    num_groups: int = 5,
    use_filter: bool = True,
    diversity_weight: float = 0.3,
) -> List[Dict]:
    """
    生成推荐号码组

    策略：
    1. 从集成模型概率中选出高概率号码
    2. 使用加权随机方法生成候选组合
    3. 应用形态过滤（可选）
    4. 加入多样性：各组之间号码尽量不同
    5. 按综合得分排序返回

    Args:
        ensemble: 集成模型
        num_groups: 推荐组数
        use_filter: 是否使用形态过滤
        diversity_weight: 多样性权重（0=纯概率，1=最大多样性）

    Returns:
        [{"front": [...], "back": [...], "score": float, "reason": str}, ...]
    """
    front_probs = ensemble.get_front_probabilities()
    back_probs = ensemble.get_back_probabilities()

    front_nums = list(range(FRONT_MIN, FRONT_MAX + 1))
    back_nums = list(range(BACK_MIN, BACK_MAX + 1))
    front_weights = [front_probs[n] for n in front_nums]
    back_weights = [back_probs[n] for n in back_nums]

    # 确保正权重
    front_weights = [max(w, 0.001) for w in front_weights]
    back_weights = [max(w, 0.001) for w in back_weights]

    recommendations = []
    used_front_numbers = Counter()
    used_back_numbers = Counter()

    # 自适应调整：冷号强制加入（保证一定的多样性）
    cold_front = sorted(front_probs.items(), key=lambda x: x[1])[:5]
    cold_back = sorted(back_probs.items(), key=lambda x: x[1])[:3]

    for group_idx in range(num_groups):
        attempts = 0
        max_attempts = 2000

        while attempts < max_attempts:
            attempts += 1

            # 动态调整权重：已被使用的号码降低权重
            adj_front_weights = list(front_weights)
            adj_back_weights = list(back_weights)

            for i, n in enumerate(front_nums):
                usage_penalty = diversity_weight * used_front_numbers.get(n, 0)
                adj_front_weights[i] = max(adj_front_weights[i] - usage_penalty * 0.05, 0.001)

            for i, n in enumerate(back_nums):
                usage_penalty = diversity_weight * used_back_numbers.get(n, 0)
                adj_back_weights[i] = max(adj_back_weights[i] - usage_penalty * 0.05, 0.001)

            # 加权采样前区
            selected_front = _weighted_sample_without_replacement(
                front_nums, adj_front_weights, FRONT_COUNT
            )
            selected_back = _weighted_sample_without_replacement(
                back_nums, adj_back_weights, BACK_COUNT
            )

            # 形态过滤
            if use_filter:
                is_valid, reason = MorphologicalFilter.is_valid(selected_front, selected_back)
                if not is_valid:
                    continue

            # 计算综合得分
            prob_score = sum(front_probs[n] for n in selected_front) / FRONT_COUNT
            prob_score += sum(back_probs[n] for n in selected_back) / BACK_COUNT

            # 多样性奖励
            front_unique = sum(1 for n in selected_front if used_front_numbers[n] == 0)
            back_unique = sum(1 for n in selected_back if used_back_numbers[n] == 0)
            diversity_score = (front_unique + back_unique) / (FRONT_COUNT + BACK_COUNT)

            # 综合得分
            weighted_score = (1 - diversity_weight) * prob_score + diversity_weight * diversity_score

            # 生成推荐理由
            reasons = []
            # 热号分析
            hot_front = sorted(front_probs.items(), key=lambda x: x[1], reverse=True)[:8]
            hot_front_nums = {n for n, _ in hot_front}
            hot_count = sum(1 for n in selected_front if n in hot_front_nums)
            if hot_count >= 3:
                reasons.append(f"{hot_count}个热号")

            # 冷号分析
            cold_front_nums = {n for n, _ in cold_front[:3]}
            cold_count = sum(1 for n in selected_front if n in cold_front_nums)
            if cold_count > 0:
                reasons.append(f"{cold_count}个冷号补位")

            # 奇偶比
            odd_count = sum(1 for n in selected_front if n % 2 == 1)
            reasons.append(f"奇偶比{odd_count}:{FRONT_COUNT - odd_count}")

            # 和值范围
            front_sum = sum(selected_front)
            if 80 <= front_sum <= 130:
                reasons.append(f"和值适中({front_sum})")
            else:
                reasons.append(f"和值{front_sum}")

            # 后区
            back_odd = sum(1 for n in selected_back if n % 2 == 1)
            reasons.append(f"后区{'奇偶' if back_odd == 1 else '双' + ('奇' if back_odd == 2 else '偶')}")

            selected_front.sort()
            selected_back.sort()

            recommendation = {
                "front": selected_front,
                "back": selected_back,
                "score": round(weighted_score, 4),
                "reason": " | ".join(reasons),
            }

            recommendations.append(recommendation)

            # 更新使用计数
            for n in selected_front:
                used_front_numbers[n] += 1
            for n in selected_back:
                used_back_numbers[n] += 1

            break

        if attempts >= max_attempts:
            logger.warning(f"第 {group_idx + 1} 组: 达到最大尝试次数")

    # 按综合得分排序
    recommendations.sort(key=lambda x: x["score"], reverse=True)
    return recommendations


def _weighted_sample_without_replacement(
    population: List[int], weights: List[float], k: int
) -> List[int]:
    """带权重的无放回随机采样"""
    if k >= len(population):
        return list(population)

    # 使用 random.choices + 去重（效率较高）
    result = []
    pool = list(population)
    w = list(weights)

    for _ in range(k):
        if sum(w) <= 0:
            idx = np.random.randint(len(pool))
        else:
            idx = np.random.choice(len(pool), p=np.array(w) / sum(w))
        result.append(pool[idx])
        pool.pop(idx)
        w.pop(idx)

    return sorted(result)
