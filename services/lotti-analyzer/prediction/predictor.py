"""
预测/推荐模块 - 运行完整分析流程并生成推荐
"""
from typing import Dict, List, Optional
import numpy as np

from config import RECOMMEND_GROUPS
from data_fetcher import update_data
from analysis.statistics import comprehensive_analysis
from analysis.visualization import generate_all_charts
from models.statistical import FrequencyModel, PoissonModel, MonteCarloModel
from models.timeseries import ExponentialSmoothingModel, MovingAverageModel
from models.ml_models import LSTMSequenceModel, XGBoostModel, RandomForestModel
from models.ensemble import EnsembleModel, generate_recommendations
from utils.helpers import logger, print_disclaimer, format_numbers


def run_prediction(
    num_groups: int = RECOMMEND_GROUPS,
    force_refresh_data: bool = False,
    generate_charts: bool = True,
    use_lstm: bool = True,
    use_xgboost: bool = True,
    use_random_forest: bool = True,
    verbose: bool = True,
) -> Dict:
    """
    运行完整预测流程

    Args:
        num_groups: 推荐组数
        force_refresh_data: 是否强制刷新数据
        generate_charts: 是否生成图表
        use_lstm: 是否使用 LSTM
        use_xgboost: 是否使用 XGBoost
        use_random_forest: 是否使用 RandomForest
        verbose: 是否输出详情

    Returns:
        { "analysis": ..., "recommendations": [...], "chart_paths": [...] }
    """
    if verbose:
        print("=" * 60)
        print("  大乐透数据分析与预测系统")
        print("=" * 60)
        print()
        print_disclaimer()

    # 1. 数据获取
    if verbose:
        print("📡 [1/5] 数据获取...")
    df = update_data(force_refresh=force_refresh_data)
    if verbose:
        print(f"   ✓ 已加载 {len(df)} 期历史数据")
        print()

    # 2. 综合分析
    if verbose:
        print("📊 [2/5] 数据分析...")
    analysis = comprehensive_analysis(df)

    if verbose:
        stats = analysis["summary"]
        print(f"   ✓ 数据范围: {stats['date_range']}")
        print(f"   ✓ 前区热号: {analysis['hot_cold']['hot_front'][:6]}")
        print(f"   ✓ 前区冷号: {analysis['hot_cold']['cold_front'][:4]}")
        print(f"   ✓ 常见奇偶比: {analysis['odd_even']['most_common_front_ratio']}")
        print(f"   ✓ 前区平均和值: {analysis['sum_analysis']['front_sum']['mean']:.1f}")
        print(f"   ✓ 前区平均跨度: {analysis['span']['mean']:.1f}")
        print(f"   ✓ AC值均值: {analysis['ac_value']['mean']:.1f}")
        print(f"   ✓ 连号出现率: {analysis['consecutive']['has_consecutive_pct']:.1f}%")
        print()

    # 3. 初始化各个模型
    if verbose:
        print("🧠 [3/5] 模型训练...")

    models = {}

    # 统计模型
    models["frequency"] = FrequencyModel(df)
    if verbose:
        print("   ✓ 频率分析模型")
    models["poisson"] = PoissonModel(df)
    if verbose:
        print("   ✓ 泊松分布模型")

    mc = MonteCarloModel(df, num_simulations=50000)
    models["monte_carlo"] = mc  # 这里只存模型对象，模拟在 ensemble 中按需调用
    if verbose:
        print("   ✓ 蒙特卡洛模拟")

    # 时间序列
    models["exponential_smoothing"] = ExponentialSmoothingModel(df, alpha=0.3)
    if verbose:
        print("   ✓ 指数平滑模型")

    # ML 模型
    lstm_model = LSTMSequenceModel(df)
    if use_lstm:
        try:
            lstm_model.train()
            models["lstm"] = lstm_model
            if verbose:
                print("   ✓ LSTM 序列模型")
        except Exception as e:
            if verbose:
                print(f"   ⚠ LSTM 模型加载失败: {e}")

    if use_xgboost:
        try:
            xgb_model = XGBoostModel(df)
            xgb_model.train()
            models["xgboost"] = xgb_model
            if verbose:
                print("   ✓ XGBoost 模型")
        except Exception as e:
            if verbose:
                print(f"   ⚠ XGBoost 跳过: {e}")

    if use_random_forest:
        try:
            rf_model = RandomForestModel(df)
            rf_model.train()
            models["random_forest"] = rf_model
            if verbose:
                print("   ✓ RandomForest 模型")
        except Exception as e:
            if verbose:
                print(f"   ⚠ RandomForest 跳过: {e}")

    if verbose:
        print()

    # 4. 集成预测
    if verbose:
        print("🎯 [4/5] 集成预测与推荐生成...")

    ensemble = EnsembleModel(models)
    recommendations = generate_recommendations(ensemble, num_groups=num_groups)

    if verbose:
        print(f"   ✓ 已生成 {len(recommendations)} 组推荐号码")
        print()

    # 5. 图表生成
    chart_paths = []
    if generate_charts:
        if verbose:
            print("📈 [5/5] 生成图表...")
        chart_paths = generate_all_charts(analysis)
        if verbose:
            print(f"   ✓ 已生成 {len(chart_paths)} 张图表")
        print()

    # 输出推荐结果
    if verbose:
        print("=" * 60)
        print(f"  🎯 推荐号码（共 {len(recommendations)} 组）")
        print("=" * 60)

        for i, rec in enumerate(recommendations, 1):
            f_str = " ".join(f"{n:02d}" for n in rec["front"])
            b_str = " ".join(f"{n:02d}" for n in rec["back"])
            print(f"\n  [{i}] {f_str}  +  {b_str}")
            print(f"      得分: {rec['score']:.4f} | {rec['reason']}")

        print()
        print("  提示: 彩票开奖为独立随机事件，以上推荐仅供娱乐参考。")
        print("=" * 60)

    return {
        "analysis": analysis,
        "recommendations": recommendations,
        "chart_paths": chart_paths,
        "model_count": len(models),
        "data_count": len(df),
    }


if __name__ == "__main__":
    run_prediction()
