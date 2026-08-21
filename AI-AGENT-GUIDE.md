# 🤖 AI AGENT 重建指南 — 彩票预测服务

> **阅读对象**: 新安装的 AI agent（Hermes、Claude、ChatGPT、或其他任何 AI 助手）
> **任务**: 读取本文件后，在新机器上完整重建主人的两个彩票预测服务
> **版本**: 2026-08-21 (v1.0) — 请勿删除或重命名本文件，这是恢复服务的唯一入口文档

---

## 0. 一句话说明

本仓库包含 **两个独立的彩票预测服务** 的完整代码、历史数据、安装脚本和本指南。
新机器上只需执行 `bash setup.sh` 即可恢复全部服务与数据。
**不要修改任何业务代码**，除非主人明确要求。

---

## 1. 服务总览

| | 服务 1: lotto | 服务 2: lotti-analyzer |
|---|---|---|
| **类型** | Streamlit Web 应用 | CLI 分析器（定时任务） |
| **目录** | `services/lotto/` | `services/lotti-analyzer/` |
| **彩种** | 大乐透 (DLT) + 双色球 (SSQ) | 大乐透 (DLT) |
| **端口** | 8502 | 无（cron 触发） |
| **算法** | 统计算法 (ensemble) + 螺旋矩阵 (spiral_matrix) | 统计模型 8 策略 |
| **入口** | `app.py` | `_cron_dlt_direct.py` |
| **数据** | `data/dlt/` + `data/ssq/` (201+500 期) | `data/` (30 期) |
| **运行方式** | systemd: `lotto-web.service` | cron: 每周二/四/日 09:00 |

**访问地址**: `http://<服务器IP>:8502`（大乐透 + 双色球两个标签页）

---

## 2. 快速恢复（3 步，5 分钟内完成）

```bash
# 1. 拉取仓库
git clone https://github.com/fangchengliu79-eng/lotto-services.git
cd lotto-services

# 2. 一键安装（自动创建 venv + 装依赖 + 注册 systemd + cron）
bash setup.sh

# 3. 验证
curl -s http://localhost:8502 | head -5   # 应返回 HTML
systemctl status lotto-web.service        # 应显示 active (running)
```

如果环境中没有 systemd（如容器），用 `bash setup.sh --no-systemd`，脚本会打印手动启动命令。

---

## 3. 手动安装（如果 setup.sh 不可用或失败）

```bash
# 3.1 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 3.2 安装依赖
pip install --upgrade pip
pip install -r requirements.txt
# 依赖清单: streamlit / pandas / numpy / requests / beautifulsoup4 / lxml
#            scikit-learn / scipy / matplotlib / seaborn

# 3.3 启动 Web 服务 (服务 1)
cd services/lotto
streamlit run app.py --server.port 8502 --server.headless true
# 后台运行: nohup streamlit run app.py --server.port 8502 --server.headless true >> ../../logs/lotto.log 2>&1 &

# 3.4 运行 CLI 分析器 (服务 2, 手动跑一次验证)
cd ../lotti-analyzer
python3 _cron_dlt_direct.py

# 3.5 注册定时任务 (可选, 每周二/四/日 09:00 与开奖日对应)
crontab -e
# 添加一行:
# 0 9 * * 2,4,0 cd /绝对路径/lotto-services/services/lotti-analyzer && /绝对路径/lotto-services/.venv/bin/python3 _cron_dlt_direct.py >> /绝对路径/lotto-services/logs/lotti-analyzer.log 2>&1
```

---

## 4. 架构详解（AI agent 修改代码前必读）

### 4.1 服务 1: lotto（Streamlit Web）

```
services/lotto/
├── app.py                  # ★ 唯一入口 (含 st.tabs() 双彩种 + render_lottery)
├── config_base.py          # LotteryConfig 数据类 (所有差异参数)
├── config_dlt.py           # 大乐透配置实例
├── config_ssq.py           # 双色球配置实例
├── data_fetcher.py         # 数据抓取 (CWL API + 500.com HTML)
├── models/
│   ├── ensemble.py         # ★ 统计算法: EnsembleModel + generate_recommendations()
│   ├── spiral_matrix.py    # ★ 螺旋矩阵: SpiralMatrixPredictor + LotteryWheelingFilter
│   ├── statistical.py      # FrequencyModel / PoissonModel / MonteCarloModel
│   ├── timeseries.py       # ExponentialSmoothingModel
│   └── ml_models.py        # LSTM / XGBoost / RandomForest (备用)
├── utils/predictions.py    # 预测封存与比对 (每期独立文件存储)
├── data/
│   ├── dlt/dlt_history.csv       # 大乐透历史 (201期)
│   ├── dlt/predictions.json      # 联合索引
│   ├── dlt/predictions/          # 每期独立预测文件 (26079_ensemble.json 等)
│   ├── ssq/ssq_history.csv       # 双色球历史 (500期)
│   └── ssq/predictions/          # 每期独立预测文件
└── README.md               # 人类可读版文档
```

**⚠️ 关键规则（违反会破坏主人信任）**:
1. **入口是 `app.py`，不是 `lottery_page.py`** — `lottery_page.py` 是死代码，改它不影响运行
2. **算法入口是 `models/ensemble.py` 的 `generate_recommendations()`** — 不要用 `engine/prediction_engine.py` 的 `AdvancedPredictionEngine`（不兼容且未使用）
3. **5 组推荐 = 5 种不同推理策略**（热号/趋势/冷号/马尔可夫/平衡），绝不"去重"号码
4. **号码约束**: 同一算法 5 组内，任何单号 ≤3 次，任何号码对 ≤3 次（前区/后区分别独立计数）
5. **只比对当期**: 每条预测只与自己期号的开奖结果比对，绝不跨期
6. **永不删除预测记录**: 历史预测对主人有参考价值。`_save_all` 最多保留 20 条（active + completed），这是系统既定行为
7. **数据真实性**: 绝不使用合成数据冒充真实开奖。数据源: CWL API (SSQ) + 500.com HTML (DLT)

**修改算法后必须**:
```bash
# 1. 删除当前 active 预测 (保留 completed)
# 2. 重新生成预测
# 3. 重启服务
cd services/lotto && kill $(pgrep -f "streamlit run app.py") ; streamlit run app.py --server.port 8502 --server.headless true
```

### 4.2 服务 2: lotti-analyzer（CLI）

```
services/lotti-analyzer/
├── _cron_dlt_direct.py     # ★ 定时任务入口 (纯统计模型 8 策略)
├── _cron_*.py              # 其他 cron 变体 (历史遗留)
├── config.py               # 配置 (FRONT_MAX 等常量)
├── data_fetcher.py         # 数据加载 (含内置 30 期真实数据)
├── analysis/statistics.py  # 统计分析
├── models/                 # 统计模型 (frequency/poisson/exponential/monte_carlo)
├── data/dlt_history.csv    # 大乐透历史 (30期, 内置)
└── data/predictions.json   # 预测记录 (2条)
```

**说明**: 此服务独立于服务 1，有自己的数据副本（30 期）。由 cron 每周二/四/日 09:00 触发，输出 8 组推荐号码到日志。

---

## 5. 数据说明与备份

| 数据 | 位置 | 内容 |
|---|---|---|
| 大乐透历史 | `services/lotto/data/dlt/dlt_history.csv` | 201 期真实开奖 |
| 双色球历史 | `services/lotto/data/ssq/ssq_history.csv` | 500 期真实开奖 |
| 预测记录 | `services/lotto/data/{dlt,ssq}/predictions/` | 每期独立 JSON 文件（容灾） |
| 联合索引 | `services/lotto/data/{dlt,ssq}/predictions.json` | 兼容旧版读取 |
| 分析器数据 | `services/lotti-analyzer/data/` | 30 期 + 预测记录 |

**数据更新机制**: Web 页面有"🔄 重新计算"按钮 → 增量拉取最新 10 期 → 合并进 CSV → 双算法重新生成预测。网络不通时使用本地数据，不卡死。

**⚠️ 数据保护原则**: 预测历史是主人的资产。未经主人明确许可，**绝不删除任何预测记录文件**。

---

## 6. 常见运维操作

```bash
# 查看服务状态
systemctl status lotto-web.service
journalctl -u lotto-web.service -n 50 --no-pager

# 重启服务
systemctl restart lotto-web.service

# 查看分析器最近输出
tail -50 logs/lotti-analyzer.log

# 数据刷新 (手动触发)
cd services/lotto && python3 -c "
from data_fetcher import update_data
import config_dlt, config_ssq
df = update_data(config_dlt.cfg, force_refresh=True)
print('DLT 最新', len(df), '期')
"
```

---

## 7. 故障排查

| 症状 | 原因 | 解决 |
|---|---|---|
| 页面打不开 | 服务未启动/端口被占 | `systemctl status lotto-web.service`; 换端口 `setup.sh --port=8503` |
| 预测显示"待开奖"但已过期 | 网络不通未拉到新数据 | 点页面"🔍 强制检查逾期比对"按钮 |
| 数据源报错 | CWL/500.com 改版 | 检查 `data_fetcher.py`，优先 CWL API (SSQ) + 500.com HTML (DLT) |
| 期号 7 位数 (2025890) | 数据源损坏 | SSQ 数据源已改版，重新用 CWL API 拉取 |
| 修改代码后无变化 | 改错了文件 | 确认改的是 `app.py` 而非 `lottery_page.py` |

---

## 8. 环境要求

- **OS**: Linux (Ubuntu/Debian 系最佳; WSL 亦可)
- **Python**: ≥ 3.10 (实测 3.11 / 3.13)
- **网络**: 需要能访问 github.com (拉取) + cwl.gov.cn / 500.com (数据)
- **资源**: ≥ 1GB RAM, 1GB 磁盘 (仓库本身 < 10MB)

---

*本文件由主人与 AI 协作维护。任何重大架构变更后请同步更新本指南。*
