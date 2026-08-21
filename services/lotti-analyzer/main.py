#!/usr/bin/env python3
"""
大乐透数据分析与预测系统 - CLI 入口

用法:
    python main.py                  # 完整预测流程
    python main.py --groups 5       # 指定推荐组数
    python main.py --backtest       # 运行回测
    python main.py --refresh        # 强制刷新数据
    python main.py --no-charts      # 不生成图表
    python main.py --no-lstm        # 不使用 LSTM
    python main.py --no-ml          # 不使用 ML 模型（仅统计模型）
    python main.py --analyze-only   # 仅分析，不预测

示例:
    python main.py --backtest --groups 3
    python main.py --refresh --no-charts --groups 10
"""
import argparse
import sys
import os

# 确保项目根目录在路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import RECOMMEND_GROUPS
from data_fetcher import update_data
from utils.helpers import logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="大乐透数据分析与预测系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py                    完整预测流程
  python main.py --backtest         运行回测
  python main.py --refresh          强制刷新数据
  python main.py --no-charts        不生成图表
  python main.py --analyze-only     仅分析
        """,
    )

    parser.add_argument(
        "--groups", "-g", type=int, default=RECOMMEND_GROUPS,
        help=f"推荐组数 (默认: {RECOMMEND_GROUPS})"
    )
    parser.add_argument(
        "--backtest", "-b", action="store_true",
        help="运行回测模式"
    )
    parser.add_argument(
        "--refresh", "-r", action="store_true",
        help="强制从网络刷新数据"
    )
    parser.add_argument(
        "--no-charts", action="store_true",
        help="不生成图表"
    )
    parser.add_argument(
        "--no-lstm", action="store_true",
        help="不使用 LSTM 模型"
    )
    parser.add_argument(
        "--no-xgboost", action="store_true",
        help="不使用 XGBoost 模型"
    )
    parser.add_argument(
        "--no-rf", action="store_true",
        help="不使用 RandomForest 模型"
    )
    parser.add_argument(
        "--no-ml", action="store_true",
        help="不使用任何 ML 模型"
    )
    parser.add_argument(
        "--analyze-only", action="store_true",
        help="仅进行分析，不生成预测推荐"
    )
    parser.add_argument(
        "--streamlit", "-s", action="store_true",
        help="启动 Streamlit Web 界面"
    )
    parser.add_argument(
        "--info", "-i", action="store_true",
        help="显示系统信息"
    )

    return parser.parse_args()


def show_info():
    """显示系统信息"""
    print("=" * 60)
    print("  大乐透数据分析与预测系统")
    print("=" * 60)
    print()
    print("  规则说明:")
    print("    前区: 01-35 选 5 个不重复号码")
    print("    后区: 01-12 选 2 个不重复号码")
    print()
    print("  模型架构:")
    print("    统计模型:")
    print("      • 频率分析            - 基于历史出现频率")
    print("      • 泊松分布            - 遗漏值概率建模")
    print("      • 蒙特卡洛模拟        - 加权随机组合搜索")
    print("    时间序列:")
    print("      • 指数平滑            - 近期趋势加权")
    print("    机器学习:")
    print("      • LSTM 序列模型       - 时序依赖学习")
    print("      • XGBoost 分类模型    - 特征工程预测")
    print("      • RandomForest 分类    - 多分类器集成")
    print("    集成策略:")
    print("      • 加权融合 + 形态过滤 + 多样性调度")
    print()
    print("  技术栈: Python, pandas, numpy, scikit-learn,")
    print("          TensorFlow/Keras, XGBoost, matplotlib")
    print()
    print("  免责声明: 本程序仅供娱乐和研究目的。")
    print("  彩票开奖是独立随机事件，不可预测。")
    print("=" * 60)


def main():
    args = parse_args()

    if args.info:
        show_info()
        return

    if args.streamlit:
        print("启动 Streamlit Web 界面...")
        os.system("streamlit run app.py")
        return

    if args.analyze_only:
        print("📊 分析模式（仅分析，不预测）")
        df = update_data(force_refresh=args.refresh)
        from analysis.statistics import comprehensive_analysis
        from analysis.visualization import generate_all_charts
        analysis = comprehensive_analysis(df)
        if not args.no_charts:
            paths = generate_all_charts(analysis)
            print(f"\n📈 已生成 {len(paths)} 张图表到 output/ 目录")
        print("\n✅ 分析完成")
        return

    if args.backtest:
        print("📊 回测模式")
        from backtest.backtester import run_backtest
        df = update_data(force_refresh=args.refresh)
        stats = run_backtest(df)
        return

    # 默认：完整预测流程
    from prediction.predictor import run_prediction

    use_lstm = not (args.no_lstm or args.no_ml)
    use_xgboost = not (args.no_xgboost or args.no_ml)
    use_rf = not (args.no_rf or args.no_ml)

    result = run_prediction(
        num_groups=args.groups,
        force_refresh_data=args.refresh,
        generate_charts=not args.no_charts,
        use_lstm=use_lstm,
        use_xgboost=use_xgboost,
        use_random_forest=use_rf,
    )

    print(f"\n✅ 分析完成。使用了 {result['model_count']} 个模型，")
    print(f"   基于 {result['data_count']} 期历史数据。")
    print(f"   图表已保存到 output/ 目录。")

    if not args.no_charts:
        print(f"\n📁 输出文件:")
        print(f"   - 数据文件: data/dlt_history.csv")
        for path in result.get("chart_paths", []):
            print(f"   - {path}")


if __name__ == "__main__":
    main()
