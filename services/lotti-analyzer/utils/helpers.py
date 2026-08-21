"""
工具函数模块
"""
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

from config import FRONT_MIN, FRONT_MAX, BACK_MIN, BACK_MAX, FRONT_COUNT, BACK_COUNT, OUTPUT_DIR

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("dlt")


def validate_numbers(front: List[int], back: List[int]) -> bool:
    """验证号码是否符合大乐透规则"""
    if len(front) != FRONT_COUNT or len(back) != BACK_COUNT:
        return False
    if len(set(front)) != FRONT_COUNT or len(set(back)) != BACK_COUNT:
        return False
    if not all(FRONT_MIN <= n <= FRONT_MAX for n in front):
        return False
    if not all(BACK_MIN <= n <= BACK_MAX for n in back):
        return False
    return True


def parse_draw_row(row: dict) -> Tuple[List[int], List[int], Optional[str]]:
    """从字典行解析前区、后区号码和期号"""
    try:
        front = sorted([int(row[f"front_{i}"]) for i in range(1, 6)])
        back = sorted([int(row[f"back_{i}"]) for i in range(1, 3)])
        period = row.get("period", None)
        return front, back, period
    except (KeyError, ValueError, TypeError) as e:
        logger.warning(f"解析行数据失败: {e}, 行: {row}")
        return [], [], None


def format_numbers(front: List[int], back: List[int]) -> str:
    """格式化号码输出，如 '05 12 18 23 31 + 03 08'"""
    f_str = " ".join(f"{n:02d}" for n in sorted(front))
    b_str = " ".join(f"{n:02d}" for n in sorted(back))
    return f"{f_str}  +  {b_str}"


def combinations_count(n: int, k: int) -> int:
    """组合数 C(n,k)"""
    from math import comb
    return comb(n, k)


def total_combinations() -> int:
    """大乐透所有组合数"""
    return combinations_count(35, 5) * combinations_count(12, 2)


def save_output(filename: str, content: str) -> str:
    """保存输出到文件"""
    path = OUTPUT_DIR / filename
    path.write_text(content, encoding="utf-8")
    logger.info(f"输出已保存: {path}")
    return str(path)


def print_disclaimer():
    """打印免责声明"""
    print("=" * 60)
    print("⚠️  免责声明")
    print("=" * 60)
    print("本程序仅供娱乐和研究目的。")
    print("彩票开奖是独立随机事件，历史数据无法保证未来结果。")
    print("任何预测分析均不构成投资建议，请理性购彩，量力而行。")
    print("=" * 60)
    print()
