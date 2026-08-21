# 🎰 彩票预测服务 (lotto-services)

一键部署的彩票预测服务套件，包含两个独立服务：

| 服务 | 目录 | 说明 | 访问 |
|---|---|---|---|
| **lotto** | `services/lotto/` | Streamlit Web — 大乐透 + 双色球，统计算法 + 螺旋矩阵双算法并行 | `http://localhost:8502` |
| **lotti-analyzer** | `services/lotti-analyzer/` | CLI 分析器 — 大乐透统计模型 8 策略推荐 | cron 每周二/四/日 09:00 |

## 🚀 快速开始

```bash
git clone https://github.com/fangchengliu79-eng/lotto-services.git
cd lotto-services
bash setup.sh
```

完成后浏览器打开 `http://localhost:8502`。

## 📖 文档

- **`AI-AGENT-GUIDE.md`** — 🤖 **给 AI agent 的重建指南**（新机器恢复服务的唯一入口文档，含架构详解、运维、排障）
- `setup.sh` — 一键安装脚本（venv + 依赖 + systemd + cron）
- `requirements.txt` — 锁定依赖清单

## 🗂 服务 1: lotto (Web)

- 彩种: 大乐透 (前区 01-35 选 5 + 后区 01-12 选 2)、双色球 (红球 01-33 选 6 + 蓝球 01-16 选 1)
- 双算法并行: 统计算法 (`models/ensemble.py`) + 螺旋矩阵 (`models/spiral_matrix.py`)
- 数据: 大乐透 201 期 / 双色球 500 期真实开奖 (data/ 目录，随仓库携带)
- 预测: 每期独立文件存储 `data/{dlt,ssq}/predictions/{期号}_{算法}.json`
- 入口: `app.py` (⚠️ 不要改 `lottery_page.py`，那是死代码)

## 🗂 服务 2: lotti-analyzer (CLI)

- 大乐透统计模型 (frequency / poisson / exponential_smoothing / monte_carlo)
- 入口: `_cron_dlt_direct.py`
- 定时: 每周二/四/日 09:00 (对应开奖日)

## ⚙️ 安装选项

```bash
bash setup.sh                    # 默认: 端口 8502 + systemd + cron
bash setup.sh --port=8503        # 自定义端口
bash setup.sh --no-systemd       # 无 systemd 环境 (容器)，打印手动命令
bash setup.sh --skip-venv        # 使用系统 Python 环境
```

## 🔄 数据更新

页面内"🔄 重新计算"按钮：增量拉取最新 10 期开奖 → 合并本地 CSV → 双算法重新生成预测。网络不通自动使用本地数据。

## 🔒 数据保护

预测历史是主人的资产，**绝不擅自删除**。每期预测独立文件存储，单文件损坏不影响其他期。

---
*由 Forest Liu 与 Grove (Hermes Agent) 维护 · 2026-08-21*
