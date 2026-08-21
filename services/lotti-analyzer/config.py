"""
大乐透分析与预测系统 - 配置文件
"""
import os
from pathlib import Path

# 项目根目录
ROOT_DIR = Path(__file__).parent.absolute()

# 数据目录
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "output"

# 历史数据文件
HISTORY_CSV = DATA_DIR / "dlt_history.csv"
MODEL_DIR = DATA_DIR / "models"

# 大乐透规则
FRONT_MIN, FRONT_MAX = 1, 35  # 前区范围
BACK_MIN, BACK_MAX = 1, 12   # 后区范围
FRONT_COUNT = 5               # 前区选号数
BACK_COUNT = 2                # 后区选号数

# 默认参数
MIN_HISTORY_DRAWS = 100       # 最少需要的历史期数
RECOMMEND_GROUPS = 5          # 默认推荐注数
TOP_HOT_COUNT = 12            # 热号前 N 个
TOP_COLD_COUNT = 8            # 冷号前 N 个

# 模型权重（集成用）
ENSEMBLE_WEIGHTS = {
    "frequency": 0.25,
    "poisson": 0.15,
    "monte_carlo": 0.10,
    "exponential_smoothing": 0.10,
    "lstm": 0.15,
    "xgboost": 0.15,
    "random_forest": 0.10,
}

# 机器学习参数
LSTM_EPOCHS = 50
LSTM_BATCH_SIZE = 32
XGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1,
    "random_state": 42,
}

# 数据抓取源
DATA_SOURCES = {
    "500com": "https://datachart.500.com/dlt/history/newinc/history.php?start={start}&end={end}",
    "cwl_gov": "https://www.lottery.gov.cn/historykj/history.jspx?_ltype=dlt",
}

# 默认抓取网站（500.com 较稳定）
DEFAULT_SOURCE = "500com"

# Streamlit 配置
STREAMLIT_TITLE = "🎯 大乐透数据分析与预测系统"
STREAMLIT_ICON = "🎯"

# 确保目录存在
for d in [DATA_DIR, OUTPUT_DIR, MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)
