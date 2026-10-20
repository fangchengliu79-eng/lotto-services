#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
# 彩票预测服务 — 一键安装/恢复脚本
# 仓库: github.com/fangchengliu79-eng/lotto-services
# 用途: 在新机器上从零恢复两个彩票服务
#   service-1: lotto          Streamlit Web (大乐透+双色球, 端口 8502)
#   service-2: lotti-analyzer CLI 分析器 (cron 每周二/四/日 09:00 跑)
#
# 用法: bash setup.sh [--port 8502] [--no-systemd] [--skip-venv]
# ═══════════════════════════════════════════════════════════════════
set -euo pipefail

# ── 参数解析 ──
PORT="${PORT:-8502}"
NO_SYSTEMD=0
SKIP_VENV=0
for arg in "$@"; do
  case "$arg" in
    --port=*) PORT="${arg#*=}" ;;
    --no-systemd) NO_SYSTEMD=1 ;;
    --skip-venv) SKIP_VENV=1 ;;
    *) echo "未知参数: $arg (支持: --port=N --no-systemd --skip-venv)"; exit 1 ;;
  esac
done

# ── 路径 ──
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_ROOT="${LOTTO_HOME:-$SCRIPT_DIR}"
VENV_DIR="$INSTALL_ROOT/.venv"
SERVICE_DIR="$INSTALL_ROOT/services"
LOG_DIR="$INSTALL_ROOT/logs"

echo "════════════════════════════════════════════════"
echo "  彩票预测服务安装器"
echo "  安装根目录: $INSTALL_ROOT"
echo "  Web 端口:   $PORT"
echo "════════════════════════════════════════════════"

# ── 0. 前置检查 ──
command -v python3 >/dev/null 2>&1 || { echo "❌ 需要 python3"; exit 1; }
PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "✅ Python $PY_VER 检测到"
if ! python3 -c "import sys; exit(0 if sys.version_info >= (3,10) else 1)" 2>/dev/null; then
  echo "❌ 需要 Python >= 3.10"; exit 1
fi

# ── 1. 虚拟环境 ──
if [ "$SKIP_VENV" = "0" ]; then
  echo "── [1/5] 创建虚拟环境 ──"
  python3 -m venv "$VENV_DIR"
  source "$VENV_DIR/bin/activate"
  echo "✅ venv 就绪: $VENV_DIR"
  echo "── [2/5] 安装依赖 (requirements.txt) ──"
  pip install --upgrade pip -q
  pip install -r "$INSTALL_ROOT/requirements.txt" -q
  echo "✅ 依赖安装完成"
else
  echo "── [1-2/5] 跳过 venv 创建 (--skip-venv) ──"
  VENV_DIR=""
fi

# ── 3. 数据完整性校验 ──
echo "── [3/5] 数据完整性校验 ──"
DLT_CSV="$SERVICE_DIR/lotto/data/dlt/dlt_history.csv"
SSQ_CSV="$SERVICE_DIR/lotto/data/ssq/ssq_history.csv"
LOTTIA_CSV="$SERVICE_DIR/lotti-analyzer/data/dlt_history.csv"
ok=1
[ -f "$DLT_CSV" ] && [ "$(wc -l < "$DLT_CSV")" -gt 100 ] || { echo "⚠️ DLT 历史数据缺失: $DLT_CSV"; ok=0; }
[ -f "$SSQ_CSV" ] && [ "$(wc -l < "$SSQ_CSV")" -gt 100 ] || { echo "⚠️ SSQ 历史数据缺失: $SSQ_CSV"; ok=0; }
[ -f "$LOTTIA_CSV" ] || { echo "⚠️ lotti-analyzer 数据缺失: $LOTTIA_CSV"; ok=0; }
if [ "$ok" = "1" ]; then echo "✅ 数据完整 (DLT $(($(wc -l < "$DLT_CSV")-1))期 / SSQ $(($(wc -l < "$SSQ_CSV")-1))期)"; fi

# ── 4. 功能自检 (导入测试) ──
echo "── [4/5] 代码自检 ──"
PYBIN="${VENV_DIR:+$VENV_DIR/bin/}python3"
cd "$SERVICE_DIR/lotto"
"$PYBIN" -c "
import sys; sys.path.insert(0, '.')
import config_dlt, config_ssq, config_base
from models.ensemble import EnsembleModel, generate_recommendations
from models.spiral_matrix import SpiralMatrixPredictor
print('✅ lotto: 所有模块导入成功 (ensemble + spiral_matrix)')
" || { echo "❌ lotto 自检失败"; exit 1; }
cd "$SERVICE_DIR/lotti-analyzer"
"$PYBIN" -c "
import sys; sys.path.insert(0, '.')
import config, data_fetcher
from analysis.statistics import frequency_analysis, hot_cold_analysis, comprehensive_analysis
from models.statistical import FrequencyModel, PoissonModel, MonteCarloModel
from models.timeseries import ExponentialSmoothingModel
from models.ensemble import EnsembleModel, generate_recommendations
from utils.helpers import format_numbers
print('✅ lotti-analyzer: 模块导入成功 (含 _cron_dlt_direct 依赖链)')
" || { echo "❌ lotti-analyzer 自检失败"; exit 1; }
cd "$INSTALL_ROOT"
echo "✅ 代码自检通过"

# ── 5. 服务注册 ──
mkdir -p "$LOG_DIR"
if [ "$NO_SYSTEMD" = "0" ] && [ -d /run/systemd/system ]; then
  echo "── [5/5] 注册 systemd 服务 ──"
  PYBIN_ABS="$VENV_DIR/bin/python3"
  STREAMLIT_BIN="$VENV_DIR/bin/streamlit"
  cat > /etc/systemd/system/lotto-web.service <<EOF
[Unit]
Description=Lotto Prediction Web (Streamlit :$PORT)
After=network.target

[Service]
Type=simple
WorkingDirectory=$SERVICE_DIR/lotto
Environment=PORT=$PORT
ExecStart=$PYBIN_ABS $STREAMLIT_BIN run app.py --server.port $PORT --server.headless true
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
  systemctl daemon-reload
  systemctl enable lotto-web.service
  systemctl restart lotto-web.service
  echo "✅ systemd 服务 lotto-web.service 已注册并启动 (:${PORT})"

  # cron: lotti-analyzer 每周二/四/日 09:00 跑分析
  cat > /etc/cron.d/lotti-analyzer <<EOF
# 大乐透 CLI 分析器: 每周二/四/日 09:00 (与开奖日对应)
SHELL=/bin/bash
0 9 * * 2,4,0 root cd $SERVICE_DIR/lotti-analyzer && $PYBIN_ABS _cron_dlt_direct.py >> $LOG_DIR/lotti-analyzer.log 2>&1
EOF
  chmod 644 /etc/cron.d/lotti-analyzer
  echo "✅ cron 已注册: 每周二/四/日 09:00 运行 lotti-analyzer"
else
  echo "── [5/5] 跳过 systemd (--no-systemd 或非 systemd 环境) ──"
  echo "手动启动 Web 服务:"
  echo "  cd $SERVICE_DIR/lotto && ${VENV_DIR:+$VENV_DIR/bin/}streamlit run app.py --server.port $PORT --server.headless true"
  echo "手动运行分析器:"
  echo "  cd $SERVICE_DIR/lotti-analyzer && ${VENV_DIR:+$VENV_DIR/bin/}python3 _cron_dlt_direct.py"
  echo "建议用 crontab 注册定时任务 (每周二/四/日 09:00):"
  echo "  crontab -e  # 添加: 0 9 * * 2,4,0 cd $SERVICE_DIR/lotti-analyzer && ${VENV_DIR:+$VENV_DIR/bin/}python3 _cron_dlt_direct.py"
fi

echo ""
echo "════════════════════════════════════════════════"
echo "  ✅ 安装完成！"
echo "  Web 服务: http://localhost:$PORT"
echo "  日志目录: $LOG_DIR"
echo "  完整文档: $INSTALL_ROOT/AI-AGENT-GUIDE.md"
echo "════════════════════════════════════════════════"
