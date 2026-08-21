"""
数据获取模块 - 自动爬取大乐透历史开奖数据

主要数据源：中彩网 (zhcw.com) - 官方数据合作伙伴
备用：500.com, 本地 CSV
"""
import json
import re
import time
from typing import List, Optional
from datetime import datetime

import pandas as pd
import requests
from bs4 import BeautifulSoup

from config import HISTORY_CSV, MIN_HISTORY_DRAWS
from utils.helpers import logger

# zhcw.com 大乐透页面
ZHCW_URL = "https://www.zhcw.com/kjxx/dlt/"

# 尝试使用的备用 API
FALLBACK_SOURCES = [
    {
        "name": "500com",
        "url": "https://datachart.500.com/dlt/history/newinc/history.php?start={start}&end={end}",
    },
]


def fetch_from_zhcw() -> Optional[pd.DataFrame]:
    """
    从中彩网 (zhcw.com) 抓取大乐透历史数据
    使用 browser 工具获取数据，然后通过文件传递
    """
    logger.info("正在从中彩网抓取数据...")

    try:
        # 使用 requests + BeautifulSoup 尝试直接抓取
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) "
                           "Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        }
        resp = requests.get(ZHCW_URL, headers=headers, timeout=15)
        resp.encoding = "utf-8"
        html = resp.text

        # 解析 HTML 表格
        soup = BeautifulSoup(html, "html.parser")
        records = []

        # 查找 tbody 中的行
        for tr in soup.select("tbody tr"):
            tds = tr.select("td")
            if len(tds) < 4:
                continue

            period = tds[0].get_text(strip=True)
            if not re.match(r"^\d+$", period):
                continue

            # 前区号码在 class="jqh" 的 span 中
            front_spans = tr.select(".jqh")
            back_spans = tr.select(".jql")

            front = [int(s.get_text(strip=True)) for s in front_spans if s.get_text(strip=True).isdigit()]
            back = [int(s.get_text(strip=True)) for s in back_spans if s.get_text(strip=True).isdigit()]

            if len(front) == 5 and len(back) == 2:
                records.append({
                    "period": period,
                    "front_1": front[0], "front_2": front[1],
                    "front_3": front[2], "front_4": front[3], "front_5": front[4],
                    "back_1": back[0], "back_2": back[1],
                })

        if records:
            df = pd.DataFrame(records)
            df = df.sort_values("period", ascending=False).reset_index(drop=True)
            logger.info(f"从中彩网抓取 {len(df)} 期数据")
            return df
        else:
            logger.warning("中彩网页面未找到开奖数据（可能需要 JavaScript 渲染）")
            return None

    except Exception as e:
        logger.warning(f"中彩网抓取失败: {e}")
        return None


def fetch_from_500com(start: int = 1, end: int = 2000) -> Optional[pd.DataFrame]:
    """从 500.com 抓取（备选方案）"""
    url = FALLBACK_SOURCES[0]["url"].format(start=start, end=end)
    logger.info(f"正在从 500.com 抓取数据...")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/120.0.0.0 Safari/537.36",
    }

    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.encoding = "gb2312"
        html = resp.text

        soup = BeautifulSoup(html, "html.parser")
        records = []

        for tr in soup.select("tr"):
            tds = tr.select("td")
            if len(tds) < 7:
                continue

            period = tds[0].get_text(strip=True)
            if not re.match(r"^\d+$", period):
                continue

            # 尝试多种号码提取方式
            numbers = []
            for cell in tds[1:]:
                nums = re.findall(r"\d+", cell.get_text())
                numbers.extend(nums)
                if len(numbers) >= 7:
                    break

            if len(numbers) >= 7:
                try:
                    front = [int(numbers[i]) for i in range(5)]
                    back = [int(numbers[i]) for i in range(5, 7)]
                    if all(1 <= n <= 35 for n in front) and all(1 <= n <= 12 for n in back):
                        records.append({
                            "period": period,
                            "front_1": front[0], "front_2": front[1],
                            "front_3": front[2], "front_4": front[3], "front_5": front[4],
                            "back_1": back[0], "back_2": back[1],
                        })
                except (ValueError, IndexError):
                    continue

        if records:
            df = pd.DataFrame(records)
            df = df.sort_values("period", ascending=False).reset_index(drop=True)
            logger.info(f"从 500.com 抓取 {len(df)} 期数据")
            return df

        logger.warning("500.com 未找到数据")
        return None

    except Exception as e:
        logger.warning(f"500.com 抓取失败: {e}")
        return None


def load_from_csv(filepath: str = None) -> Optional[pd.DataFrame]:
    """从本地 CSV 文件加载历史数据"""
    path = filepath or str(HISTORY_CSV)
    if not pd.io.common.file_exists(path):
        logger.warning(f"CSV 文件不存在: {path}")
        return None

    try:
        df = pd.read_csv(path, dtype={"period": str})
        required_cols = ["period"] + [f"front_{i}" for i in range(1, 6)] + [f"back_{i}" for i in range(1, 3)]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            logger.error(f"CSV 缺少必要列: {missing}")
            return None
        df = df.sort_values("period", ascending=False).reset_index(drop=True)
        logger.info(f"从 CSV 加载 {len(df)} 期数据: {path}")
        return df
    except Exception as e:
        logger.error(f"读取 CSV 失败: {e}")
        return None


def save_to_csv(df: pd.DataFrame, filepath: str = None) -> bool:
    """保存数据到 CSV"""
    path = filepath or str(HISTORY_CSV)
    try:
        df.to_csv(path, index=False, encoding="utf-8-sig")
        logger.info(f"数据已保存到: {path}")
        return True
    except Exception as e:
        logger.error(f"保存 CSV 失败: {e}")
        return False


def _extract_browser_capture() -> Optional[pd.DataFrame]:
    """
    从浏览器捕获文件加载数据（运行时由 browser_console 写入）
    文件格式: JSON数组 [{"period":"26058","front":[7,12,13,18,34],"back":[1,5]},...]
    """
    import os
    capture_path = "/root/lotti-analyzer/data/browser_capture.json"
    if not os.path.exists(capture_path):
        return None

    try:
        with open(capture_path) as f:
            data = json.load(f)

        records = []
        for item in data:
            if "period" in item and "front" in item and "back" in item:
                front = item["front"]
                back = item["back"]
                if len(front) == 5 and len(back) == 2:
                    records.append({
                        "period": str(item["period"]),
                        "front_1": front[0], "front_2": front[1],
                        "front_3": front[2], "front_4": front[3], "front_5": front[4],
                        "back_1": back[0], "back_2": back[1],
                    })

        if records:
            df = pd.DataFrame(records)
            df = df.sort_values("period", ascending=False).reset_index(drop=True)
            logger.info(f"从浏览器捕获文件加载 {len(df)} 期数据")
            return df
    except Exception as e:
        logger.warning(f"浏览器捕获文件读取失败: {e}")

    return None


def update_data(force_refresh: bool = False) -> pd.DataFrame:
    """
    获取并更新数据主函数
    优先级: 浏览器捕获 > 网络抓取(zhcw) > 网络抓取(500com) > 本地CSV
    """
    # 1. 先尝试本地 CSV（如果不是强制刷新且有足够数据）
    if not force_refresh:
        df_local = load_from_csv()
        if df_local is not None and len(df_local) >= MIN_HISTORY_DRAWS:
            logger.info(f"本地已有 {len(df_local)} 期数据，足够使用")
            return df_local

    # 2. 尝试从网络抓取
    logger.info("正在从网络更新数据...")

    # 2a. 先尝试中彩网
    df = fetch_from_zhcw()

    # 2b. 尝试 500.com
    if df is None or len(df) < MIN_HISTORY_DRAWS:
        df_500 = fetch_from_500com()
        if df_500 is not None:
            df = df_500 if df is None else _merge_dfs(df, df_500)

    # 2c. 尝试浏览器捕获
    if df is None or len(df) < 20:
        df_browser = _extract_browser_capture()
        if df_browser is not None:
            df = df_browser if df is None else _merge_dfs(df, df_browser)

    # 3. 如果网络获取失败，回退到本地
    if df is None or len(df) == 0:
        df_local = load_from_csv()
        if df_local is not None and len(df_local) > 0:
            logger.warning("网络数据获取失败，使用本地数据")
            return df_local
        logger.error("无可用数据源！")
        return pd.DataFrame()

    # 4. 合并本地数据（增量更新）
    df_local = load_from_csv()
    if df_local is not None:
        df = _merge_dfs(df, df_local)

    # 5. 保存到 CSV
    save_to_csv(df)
    logger.info(f"数据更新完成: {len(df)} 期")
    return df


def _merge_dfs(df_new: pd.DataFrame, df_old: pd.DataFrame) -> pd.DataFrame:
    """合并两个数据框，去重"""
    combined = pd.concat([df_new, df_old], ignore_index=True)
    combined = combined.drop_duplicates(subset=["period"], keep="first")
    combined = combined.sort_values("period", ascending=False).reset_index(drop=True)
    return combined


if __name__ == "__main__":
    df = update_data(force_refresh=True)
    print(f"数据加载完成: {len(df)} 期")
    if len(df) > 0:
        front_cols = [f"front_{i}" for i in range(1, 6)]
        back_cols = [f"back_{i}" for i in range(1, 3)]
        print(f"最新: 期{df.iloc[0]['period']} → "
              f"{' '.join(f'{df.iloc[0][c]:02d}' for c in front_cols)} + "
              f"{' '.join(f'{df.iloc[0][c]:02d}' for c in back_cols)}")
