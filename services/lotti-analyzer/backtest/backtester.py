"""
回测模块 - 对历史数据进行回测，评估模型预测能力

回测策略：使用历史 N-1 期数据训练，预测第 N 期，
滚动测试并统计命中率。
"""
from typing import Dict, List, Tuple, Optional
from collections import Counter
from copy import deepcopy

import numpy as np
import pandas as pd

from config import FRONT_COUNT, BACK_COUNT
from analysis.statistics import frequency_analysis
from models.statistical import FrequencyModel, PoissonModel
from models.timeseries import ExponentialSmoothingModel
from models.ensemble import EnsembleModel, generate_recommendations, MorphologicalFilter
from utils.helpers import logger, format_numbers


class Backtester:
    """
    回测引擎

    滚动回测：从某个起始点开始，每次使用过去 N 期训练，
    预测未来 1 期，统计命中情况。
    """

    def __init__(self, df: pd.DataFrame, test_start: int = None, test_window: int = 50):
        """
        Args:
            df: 完整历史数据（已按期号降序排列）
            test_start: 回测起始索引（从最新一期算起）
            test_window: 测试期数
        """
        self.df = df
        self.total_draws = len(df)
        self.test_window = min(test_window, max(10, self.total_draws // 3))

        if test_start is None:
            # 默认从倒数第 test_window*2 期开始测试，但保证有足够训练数据
            self.test_start = max(
                35,  # 至少需要 30+ 行训练数据
                self.total_draws - self.test_window * 3
            )
        else:
            self.test_start = max(35, min(test_start, self.total_draws - self.test_window - 10))

        self.results = []

    def run(self, verbose: bool = True) -> Dict:
        """
        运行回测

        对每一测试期进行：
        1. 使用该期之前的所有数据进行训练
        2. 生成推荐号码
        3. 与实际开奖号码比较，统计命中情况
        """
        logger.info(f"开始回测: 共 {self.test_window} 期测试")
        if verbose:
            print(f"\n📊 回测启动: {self.test_window} 期滚动测试")
            print(f"   测试范围: 第 {self.test_start} ~ {self.test_start + self.test_window} 期")
            print()

        front_cols = [f"front_{i}" for i in range(1, 6)]
        back_cols = [f"back_{i}" for i in range(1, 3)]

        for test_idx in range(self.test_start, min(self.test_start + self.test_window, self.total_draws)):
            if test_idx < 20 or test_idx >= self.total_draws - 5:  # 数据不足跳过
                continue

            # 训练数据：test_idx 之前的所有数据
            train_df = self.df.iloc[test_idx + 1:].copy()  # 更旧的数据
            if len(train_df) < 30:
                continue

            # 实际开奖号码
            actual_row = self.df.iloc[test_idx]
            actual_front = sorted([int(actual_row[c]) for c in front_cols])
            actual_back = sorted([int(actual_row[c]) for c in back_cols])

            # 使用统计模型（快速，不需要 ML 训练时间）
            models = {
                "frequency": FrequencyModel(train_df),
                "poisson": PoissonModel(train_df),
                "exponential_smoothing": ExponentialSmoothingModel(train_df, alpha=0.3),
            }
            ensemble = EnsembleModel(models)
            recommendations = generate_recommendations(ensemble, num_groups=5)

            # 评估每组推荐的命中情况
            best_hits = (0, 0)  # (front_hits, back_hits)

            for rec in recommendations:
                pred_front = rec["front"]
                pred_back = rec["back"]

                front_hits = len(set(pred_front) & set(actual_front))
                back_hits = len(set(pred_back) & set(actual_back))

                if front_hits + back_hits > sum(best_hits):
                    best_hits = (front_hits, back_hits)
                elif front_hits + back_hits == sum(best_hits) and front_hits > best_hits[0]:
                    best_hits = (front_hits, back_hits)

            result = {
                "period": actual_row.get("period", f"idx_{test_idx}"),
                "actual_front": actual_front,
                "actual_back": actual_back,
                "front_hits": best_hits[0],
                "back_hits": best_hits[1],
                "total_hits": best_hits[0] + best_hits[1],
            }
            self.results.append(result)

            if verbose and (len(self.results) % 10 == 0 or len(self.results) == self.test_window):
                progress = len(self.results)
                total = min(self.test_window, self.total_draws - self.test_start)
                print(f"   回测进度: {progress}/{total} 期")

        return self._compute_stats(verbose)

    def _compute_stats(self, verbose: bool = True) -> Dict:
        """计算回测统计指标"""
        if not self.results:
            return {"error": "无回测结果"}

        total_tests = len(self.results)
        front_hits_list = [r["front_hits"] for r in self.results]
        back_hits_list = [r["back_hits"] for r in self.results]
        total_hits_list = [r["total_hits"] for r in self.results]

        stats = {
            "total_tests": total_tests,
            "front_hits": {
                "mean": float(np.mean(front_hits_list)),
                "median": float(np.median(front_hits_list)),
                "max": int(max(front_hits_list)),
                "min": int(min(front_hits_list)),
            },
            "back_hits": {
                "mean": float(np.mean(back_hits_list)),
                "median": float(np.median(back_hits_list)),
                "max": int(max(back_hits_list)),
                "min": int(min(back_hits_list)),
            },
            "total_hits": {
                "mean": float(np.mean(total_hits_list)),
                "median": float(np.median(total_hits_list)),
                "max": int(max(total_hits_list)),
                "min": int(min(total_hits_list)),
            },
        }

        # 各等级命中统计（大乐透奖项对照）
        prize_stats = Counter()

        # 0+0, 0+1, 0+2
        # 1+0, 1+1, 1+2
        # 2+0, 2+1, 2+2
        # 3+0, 3+1, 3+2  (3+1=六等奖)
        # 4+0, 4+1, 4+2  (4+2=四等奖)
        # 5+0, 5+1, 5+2  (5+2=一等奖)
        for r in self.results:
            key = f"{r['front_hits']}+{r['back_hits']}"
            prize_stats[key] += 1

        stats["prize_distribution"] = {
            k: {"count": v, "pct": v / total_tests * 100}
            for k, v in sorted(prize_stats.items(), key=lambda x: (
                -int(x[0].split("+")[0]), -int(x[0].split("+")[1])
            ))
        }

        # 各档次命中率（简化版）
        front_3_plus = sum(1 for r in self.results if r["front_hits"] >= 3)
        front_4_plus = sum(1 for r in self.results if r["front_hits"] >= 4)
        front_5_plus = sum(1 for r in self.results if r["front_hits"] == 5)
        back_1_plus = sum(1 for r in self.results if r["back_hits"] >= 1)
        back_2 = sum(1 for r in self.results if r["back_hits"] == 2)

        stats["hit_rates"] = {
            "front_3+": front_3_plus / total_tests * 100,
            "front_4+": front_4_plus / total_tests * 100,
            "front_5": front_5_plus / total_tests * 100,
            "back_1+": back_1_plus / total_tests * 100,
            "back_2": back_2 / total_tests * 100,
            "front_avg_hits": stats["front_hits"]["mean"],
            "back_avg_hits": stats["back_hits"]["mean"],
        }

        if verbose:
            print()
            print("📊 回测结果统计")
            print("=" * 50)
            print(f"  测试期数:       {total_tests}")
            print(f"  前区平均命中:   {stats['front_hits']['mean']:.2f} 个 (最大 {stats['front_hits']['max']})")
            print(f"  后区平均命中:   {stats['back_hits']['mean']:.2f} 个 (最大 {stats['back_hits']['max']})")
            print(f"  综合平均命中:   {stats['total_hits']['mean']:.2f} 个")
            print(f"  前区 3+ 命中率: {stats['hit_rates']['front_3+']:.1f}%")
            print(f"  前区 4+ 命中率: {stats['hit_rates']['front_4+']:.1f}%")
            print(f"  后区 1+ 命中率: {stats['hit_rates']['back_1+']:.1f}%")
            print(f"  后区 2  命中率: {stats['hit_rates']['back_2']:.1f}%")
            print()
            print("  ⚠️  回测结果仅供参考。历史表现不能保证未来结果。")
            print("  彩票本质是完全随机的概率游戏。")
            print("=" * 50)
            print()
            print("  详细命中分布:")
            for key, val in stats["prize_distribution"].items():
                print(f"    {key} 命中: {val['count']} 次 ({val['pct']:.1f}%)")

        return stats


def run_backtest(df: pd.DataFrame = None, verbose: bool = True) -> Dict:
    """
    运行回测的便捷入口

    Args:
        df: 历史数据（None 则自动加载）
        verbose: 是否输出详细信息

    Returns:
        回测统计结果
    """
    if df is None:
        from data_fetcher import update_data
        df = update_data()

    bt = Backtester(df)
    return bt.run(verbose=verbose)


if __name__ == "__main__":
    run_backtest()
