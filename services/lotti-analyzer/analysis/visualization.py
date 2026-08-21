"""
可视化模块 - 使用 matplotlib/seaborn 生成分析图表
"""
import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # 非交互式后端，适合 CLI
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

from config import FRONT_MIN, FRONT_MAX, BACK_MIN, BACK_MAX, OUTPUT_DIR
from utils.helpers import logger

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "Noto Sans CJK SC", "SimHei", "DejaVu Sans", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_style("whitegrid")


def plot_frequency(freq_data: Dict, zone: str = "front", save_path: str = None) -> str:
    """绘制号码出现频率柱状图"""
    key = f"{zone}_freq"
    if key not in freq_data:
        return ""

    nums = sorted(freq_data[key].keys())
    counts = [freq_data[key][n] for n in nums]
    total = freq_data.get("total_draws", 1)

    fig, ax = plt.subplots(figsize=(14, 5))
    colors = ["#e74c3c" if c > np.mean(counts) + np.std(counts)
              else "#3498db" if c < np.mean(counts) - np.std(counts)
              else "#95a5a6" for c in counts]

    bars = ax.bar(nums, counts, color=colors, edgecolor="white", linewidth=0.5)

    # 标注最热/最冷号
    for i, (n, c) in enumerate(zip(nums, counts)):
        if c == max(counts) or c == min(counts) or c == counts[-1] or c == counts[0]:
            ax.annotate(f"{c}", (n, c), textcoords="offset points",
                        xytext=(0, 5), ha="center", fontsize=8)

    title = f"{'前区' if zone == 'front' else '后区'}号码出现频率（共 {total} 期）"
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("号码", fontsize=11)
    ax.set_ylabel("出现次数", fontsize=11)
    ax.set_xticks(nums)

    plt.tight_layout()
    path = save_path or str(OUTPUT_DIR / f"{zone}_frequency.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"频率图已保存: {path}")
    return path


def plot_missing(missing_data: Dict, zone: str = "front", save_path: str = None) -> str:
    """绘制遗漏值条形图"""
    key = f"{zone}_missing"
    if key not in missing_data:
        return ""

    nums = sorted(missing_data[key].keys())
    miss_vals = [missing_data[key][n] for n in nums]

    fig, ax = plt.subplots(figsize=(14, 5))
    colors = ["#e67e22" if m > np.mean(miss_vals) + np.std(miss_vals)
              else "#2ecc71" if m < np.mean(miss_vals) - np.std(miss_vals)
              else "#95a5a6" for m in miss_vals]

    ax.bar(nums, miss_vals, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(y=np.mean(miss_vals), color="red", linestyle="--", alpha=0.5, label=f"平均遗漏")

    title = f"{'前区' if zone == 'front' else '后区'}号码遗漏期数"
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("号码", fontsize=11)
    ax.set_ylabel("遗漏期数", fontsize=11)
    ax.set_xticks(nums)
    ax.legend()

    plt.tight_layout()
    path = save_path or str(OUTPUT_DIR / f"{zone}_missing.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_odd_even(oe_data: Dict, save_path: str = None) -> str:
    """绘制奇偶比饼图"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for idx, (zone, label) in enumerate([("front", "前区"), ("back", "后区")]):
        key = f"{zone}_ratios"
        if key not in oe_data:
            continue
        ratios = oe_data[key]
        labels_ = list(ratios.keys())
        sizes = [ratios[k]["pct"] for k in labels_]

        axes[idx].pie(sizes, labels=labels_, autopct="%1.1f%%",
                      startangle=90, colors=sns.color_palette("pastel"))
        axes[idx].set_title(f"{label}奇偶比分布", fontsize=12, fontweight="bold")

    plt.tight_layout()
    path = save_path or str(OUTPUT_DIR / "odd_even_ratio.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_sum_distribution(sum_data: Dict, save_path: str = None) -> str:
    """绘制和值分布直方图"""
    # 从数据中重建和值序列
    front_sum_mean = sum_data.get("front_sum", {}).get("mean", 90)
    front_sum_std = sum_data.get("front_sum", {}).get("std", 20)

    # 模拟一下分布展示（实际应使用原始数据）
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (zone, label) in enumerate([("front", "前区"), ("back", "后区")]):
        s = sum_data.get(f"{zone}_sum", {})
        mean, std = s.get("mean", 90), s.get("std", 20)
        min_v, max_v = s.get("min", 15), s.get("max", 165)

        # 生成正态分布曲线
        x = np.linspace(min_v, max_v, 100)
        y = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)

        axes[idx].plot(x, y * len(sum_data.get("front_sum", {}).get("recent", [])), "b-", linewidth=2)
        axes[idx].axvline(mean, color="red", linestyle="--", alpha=0.7, label=f"均值 {mean:.1f}")

        # 标注均值±1σ区域
        axes[idx].axvspan(mean - std, mean + std, alpha=0.15, color="green", label=f"±1σ ({mean-std:.0f}~{mean+std:.0f})")

        axes[idx].set_title(f"{label}和值分布", fontsize=12, fontweight="bold")
        axes[idx].set_xlabel("和值", fontsize=11)
        axes[idx].set_ylabel("概率密度", fontsize=11)
        axes[idx].legend()

    plt.tight_layout()
    path = save_path or str(OUTPUT_DIR / "sum_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_heatmap_frequency(freq_data: Dict, save_path: str = None) -> str:
    """绘制前区号码频率热力图（5x7 网格，模拟选号卡）"""
    front_freq = freq_data.get("front_freq", {})
    if not front_freq:
        return ""

    # 35个号码排成 5行 x 7列
    grid = np.zeros((5, 7))
    for num, freq in front_freq.items():
        row = (num - 1) // 7
        col = (num - 1) % 7
        grid[row, col] = freq

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(grid, annot=True, fmt=".0f", cmap="YlOrRd",
                xticklabels=[str(i) for i in range(1, 8)],
                yticklabels=[f"{1+i*7}-{7+i*7}" for i in range(5)],
                linewidths=1, ax=ax, cbar_kws={"label": "出现次数"})

    ax.set_title("前区号码频率热力图（选号卡布局）", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = save_path or str(OUTPUT_DIR / "frequency_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_ac_value(ac_data: Dict, save_path: str = None) -> str:
    """绘制 AC 值分布"""
    distribution = ac_data.get("distribution", {})
    if not distribution:
        return ""

    values = sorted(distribution.keys())
    counts = [distribution[v] for v in values]
    total = sum(counts)

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(values, counts, color="#8e44ad", alpha=0.7, edgecolor="white")
    ax.bar_label(bars, labels=[f"{c/total*100:.1f}%" for c in counts], padding=3)

    ax.set_title("前区 AC 值分布", fontsize=14, fontweight="bold")
    ax.set_xlabel("AC 值", fontsize=11)
    ax.set_ylabel("出现次数", fontsize=11)
    ax.set_xticks(values)

    plt.tight_layout()
    path = save_path or str(OUTPUT_DIR / "ac_value_dist.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_span_trend(span_data: Dict, save_path: str = None) -> str:
    """绘制跨度趋势图"""
    recent = span_data.get("recent", [])
    if not recent:
        return ""

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(range(len(recent)), recent, "o-", color="#2980b9", linewidth=1.5, markersize=4)
    ax.axhline(y=span_data.get("mean", 0), color="red", linestyle="--", alpha=0.6, label=f"均值 {span_data['mean']:.1f}")

    ax.set_title("前区跨度趋势（最近 N 期）", fontsize=13, fontweight="bold")
    ax.set_xlabel("期次（从新到旧）", fontsize=11)
    ax.set_ylabel("跨度", fontsize=11)
    ax.legend()
    ax.fill_between(range(len(recent)), span_data.get("mean", 0) - span_data.get("std", 0),
                     span_data.get("mean", 0) + span_data.get("std", 0),
                     alpha=0.1, color="blue")

    plt.tight_layout()
    path = save_path or str(OUTPUT_DIR / "span_trend.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def generate_all_charts(analysis_result: Dict) -> List[str]:
    """生成所有分析图表"""
    logger.info("正在生成分析图表...")
    paths = []

    p = plot_frequency(analysis_result.get("frequency", {}), "front")
    if p: paths.append(p)

    p = plot_frequency(analysis_result.get("frequency", {}), "back")
    if p: paths.append(p)

    p = plot_missing(analysis_result.get("missing", {}), "front")
    if p: paths.append(p)

    p = plot_missing(analysis_result.get("missing", {}), "back")
    if p: paths.append(p)

    p = plot_odd_even(analysis_result.get("odd_even", {}))
    if p: paths.append(p)

    p = plot_sum_distribution(analysis_result.get("sum_analysis", {}))
    if p: paths.append(p)

    p = plot_heatmap_frequency(analysis_result.get("frequency", {}))
    if p: paths.append(p)

    p = plot_ac_value(analysis_result.get("ac_value", {}))
    if p: paths.append(p)

    p = plot_span_trend(analysis_result.get("span", {}))
    if p: paths.append(p)

    logger.info(f"共生成 {len(paths)} 张图表")
    return paths
