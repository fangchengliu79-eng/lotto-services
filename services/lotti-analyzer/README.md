# 🎯 大乐透数据分析与预测系统

中国体育彩票超级大乐透（大乐透）的统计分析与辅助预测程序。

> ⚠️ **免责声明**: 本程序仅供**娱乐和研究目的**。彩票开奖是独立随机事件，历史数据和任何分析模型均**无法保证未来结果**。请理性购彩，量力而行。

---

## 大乐透规则

- **前区**: 从 01-35 中选出 **5 个** 不重复号码
- **后区**: 从 01-12 中选出 **2 个** 不重复号码
- 总组合数: C(35,5) × C(12,2) = 324,632 × 66 = **21,425,712** 种

---

## 功能概览

| 功能 | CLI | Web |
|------|:---:|:---:|
| 📊 多维度统计分析 | ✅ | ✅ |
| 🎯 模型预测推荐（5组） | ✅ | ✅ |
| 📈 历史回测 | ✅ | ✅ |
| 🖼️ 可视化图表 | ✅ | ✅ |
| 📡 数据自动更新 | ✅ | ✅ |
| 🌓 科技暗色主题 UI | – | ✅ v2.0 |

---

## 快速开始

### 1. 安装依赖

```bash
cd lotti-analyzer
pip install -r requirements.txt
```

如需完整功能（含 LSTM/XGBoost），取消 `requirements.txt` 中注释的行：

```bash
pip install tensorflow xgboost
```

### 2. CLI 模式

```bash
# 完整预测流程
python main.py

# 自定义推荐组数
python main.py --groups 5

# 运行回测
python main.py --backtest

# 强制刷新数据
python main.py --refresh

# 仅分析不预测
python main.py --analyze-only

# 不使用 ML 模型（仅统计模型，速度快）
python main.py --no-ml

# 不生成图表
python main.py --no-charts
```

### 3. Web 界面

```bash
streamlit run app.py
```

---

## 项目结构

```
lotti-analyzer/
├── main.py                 # CLI 入口
├── app.py                  # Streamlit Web 界面
├── config.py               # 配置文件
├── requirements.txt        # 依赖列表
├── README.md               # 本文件
│
├── data_fetcher.py         # 数据获取模块
├── data/                   # 数据存储目录
│   ├── dlt_history.csv     # 历史开奖数据
│   └── models/             # 训练好的模型
│
├── analysis/
│   ├── statistics.py       # 统计分析（冷热号、奇偶比、和值等）
│   └── visualization.py    # 可视化图表生成
│
├── models/
│   ├── statistical.py      # 统计模型（频率、泊松、蒙特卡洛）
│   ├── timeseries.py       # 时间序列模型（指数平滑）
│   ├── ml_models.py        # ML 模型（LSTM、XGBoost、RF）
│   └── ensemble.py         # 集成融合 + 形态过滤
│
├── prediction/
│   └── predictor.py        # 推荐生成
│
├── backtest/
│   └── backtester.py       # 滚动回测引擎
│
├── utils/
│   └── helpers.py          # 工具函数
│
└── output/                 # 图表输出目录
```

---

## 模型架构详解

### 1. 统计模型

| 模型 | 原理 | 输入 | 输出 |
|------|------|------|------|
| **频率分析** | 历史出现频率 | 所有历史数据 | 各号码概率 |
| **泊松分布** | 遗漏值的泊松建模 | 遗漏期数 | 条件概率 |
| **蒙特卡洛** | 加权随机组合 + 评分 | 频率权重 | 高分组合 |

### 2. 时间序列

| 模型 | 原理 |
|------|------|
| **指数平滑** | α=0.3 平滑近期趋势，更关注近期的号码热度变化 |
| **移动平均** | 最近 N 期频率作为预测 |

### 3. 机器学习

| 模型 | 架构 | 说明 |
|------|------|------|
| **LSTM** | 双层 LSTM (64+32) | 历史 N 期出现向量 → 预测下一期概率 |
| **XGBoost** | 200 trees, max_depth=6 | 基于 8 维特征（频率、遗漏、近期趋势等） |
| **RandomForest** | 100 trees, per-number | 每个号码单独二分类器 |

### 4. 集成策略

```
加权融合 → 形态过滤 → 多样性调度 → 最终推荐

权重分配:
  - 频率分析:      25%
  - 泊松分布:      15%
  - 蒙特卡洛:      10%
  - 指数平滑:      10%
  - LSTM:          15%
  - XGBoost:       15%
  - RandomForest:  10%
```

### 5. 形态过滤

排除以下低概率组合：
- ❌ 全奇 / 全偶
- ❌ 全大（≥18）/ 全小（≤17）
- ❌ 和值 < 50 或 > 155
- ❌ 跨度 < 8 或 > 32
- ❌ 连号超过 2 组
- ❌ AC 值 < 2

---

## 数据分析指标

- **冷热号**: 频率统计 + 遗漏值综合评分
- **奇偶比**: 前区 5 个号码中奇数/偶数比例分布
- **和值**: 前区号码和值分布（均值约 90）
- **大小分布**: 小号(01-17) / 大号(18-35) 比例
- **跨度**: 最大号 - 最小号
- **AC 值**: 号码离散程度指标（均值约 5-7）
- **连号**: 相邻号码出现情况

---

## 数据来源

默认从 [500.com](https://datachart.500.com/dlt/) 抓取历史开奖数据。
首次运行会自动获取最新数据并保存到 `data/dlt_history.csv`。

数据格式：
```csv
period,front_1,front_2,front_3,front_4,front_5,back_1,back_2
25001,05,12,18,23,31,03,08
...
```

---

## 命令行参数

```
usage: python main.py [-h] [--groups GROUPS] [--backtest] [--refresh]
                      [--no-charts] [--no-lstm] [--no-xgboost] [--no-rf]
                      [--no-ml] [--analyze-only] [--streamlit] [--info]

示例:
  python main.py                    完整预测流程
  python main.py --backtest         运行回测
  python main.py --refresh          强制刷新数据
  python main.py --no-charts        不生成图表
  python main.py --analyze-only     仅分析
```

---

## 回测结果解读

回测使用滚动验证：每次用过去数据训练，预测下一期。

示例输出：
```
前区平均命中: 1.2 个
后区平均命中: 0.5 个
前区 3+ 命中率: 8.3%
```

> 对比纯随机：前区 5/35 ≈ 0.14 个/期，后区 2/12 ≈ 0.17 个/期
> 模型的命中率高于纯随机，但**远不能保证中奖**

---

## 注意事项

1. **数据时效性**: 推荐使用最新数据（`--refresh` 强制更新）
2. **LSTM 训练**: 首次运行 TensorFlow 会下载依赖，可能需要几分钟
3. **网络请求**: 数据抓取需要网络连接，如 500.com 访问失败，会使用本地缓存
4. **中文字体**: 图表会自动配置中文字体（SimHei），如缺失会回退到英文显示

---

## 技术栈

- Python 3.8+
- pandas / numpy (数据处理)
- requests + BeautifulSoup (数据抓取)
- matplotlib + seaborn (可视化)
- scikit-learn / TensorFlow / XGBoost (机器学习)
- Streamlit (Web 界面)

---

## License

仅供学习和研究使用。
