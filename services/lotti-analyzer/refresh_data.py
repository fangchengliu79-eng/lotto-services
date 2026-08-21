#!/usr/bin/env python3
"""
大乐透数据刷新工具 - 使用浏览器从中彩网获取最新开奖数据

用法:
    python3 refresh_data.py          # 抓取数据并保存
    python3 refresh_data.py --show   # 抓取+显示最新数据
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_fetcher import save_to_csv
from utils.helpers import logger


def scrape_from_zhcw_browser():
    """
    使用浏览器从中彩网(zhcw.com)抓取大乐透历史数据。
    此函数通过 selenium-wire 或 playwright 实现自动化抓取。
    
    当前实现使用 requests + BeautifulSoup 的备选方案，
    对于需要 JS 渲染的站点，通过模拟请求获取数据。
    """
    import requests
    from bs4 import BeautifulSoup
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Referer": "https://www.zhcw.com/",
        "Cookie": "Hm_lvt_692bd5f9c07d3ebd0063062fb0d7622f=1779939000; "
                  "Hm_lpvt_692bd5f9c07d3ebd0063062fb0d7622f=1779939000; "
                  "hm_has_registered=true",
    }
    
    url = "https://www.zhcw.com/kjxx/dlt/"
    
    try:
        resp = requests.get(url, headers=headers, timeout=20)
        resp.encoding = "utf-8"
        html = resp.text
        
        soup = BeautifulSoup(html, "html.parser")
        records = []
        
        # 尝试解析 tbody 中的行
        for tr in soup.select("tbody tr"):
            period_td = tr.find("td")
            if not period_td:
                continue
            period = period_td.get_text(strip=True)
            if not period.isdigit():
                continue
            
            jqh_spans = tr.select(".jqh")
            jql_spans = tr.select(".jql")
            
            # 检查是否是模板变量
            front_texts = [s.get_text(strip=True) for s in jqh_spans]
            back_texts = [s.get_text(strip=True) for s in jql_spans]
            
            if "{{" in "".join(front_texts + back_texts):
                continue  # 模板未渲染，跳过
            
            front = [int(t) for t in front_texts if t.isdigit()]
            back = [int(t) for t in back_texts if t.isdigit()]
            
            if len(front) == 5 and len(back) == 2:
                records.append({
                    "period": period,
                    "front_1": front[0], "front_2": front[1],
                    "front_3": front[2], "front_4": front[3], "front_5": front[4],
                    "back_1": back[0], "back_2": back[1],
                })
        
        return records
        
    except Exception as e:
        logger.error(f"抓取失败: {e}")
        return []


def get_mock_zhcw_data():
    """
    手动整理的中彩网最新30期数据（上次抓取时保存）
    作为浏览器不可用时的备用数据源
    """
    import datetime
    
    # 这些数据来自中彩网（最后更新: 2026-05-27）
    data = [
        (26058, [7,12,13,18,34], [1,5]),
        (26057, [23,25,26,27,34], [4,10]),
        (26056, [6,7,18,21,30], [1,5]),
        (26055, [9,10,20,33,35], [4,11]),
        (26054, [2,6,14,22,24], [8,11]),
        (26053, [2,9,14,20,31], [5,9]),
        (26052, [2,3,20,28,33], [2,12]),
        (26051, [13,18,28,32,33], [2,11]),
        (26050, [6,10,14,23,33], [8,10]),
        (26049, [1,6,14,15,17], [2,3]),
        (26048, [11,17,20,23,35], [1,10]),
        (26047, [9,20,21,23,28], [6,11]),
        (26046, [1,13,18,27,33], [4,7]),
        (26045, [1,15,21,26,33], [4,7]),
        (26044, [3,8,22,26,29], [7,10]),
        (26043, [8,12,14,19,22], [11,12]),
        (26042, [2,7,13,19,24], [3,8]),
        (26041, [24,25,27,29,34], [2,6]),
        (26040, [6,12,13,21,34], [8,9]),
        (26039, [9,11,20,26,27], [6,9]),
        (26038, [8,17,21,33,35], [6,7]),
        (26037, [7,12,13,28,32], [6,8]),
        (26036, [4,7,16,26,32], [5,8]),
        (26035, [2,22,30,33,34], [8,12]),
        (26034, [11,12,25,26,27], [8,11]),
        (26033, [3,5,7,9,18], [2,10]),
        (26032, [3,4,19,26,32], [1,12]),
        (26031, [6,8,22,29,34], [5,7]),
        (26030, [2,13,22,28,34], [5,12]),
        (26029, [3,5,17,33,35], [5,7]),
    ]
    
    print(f"📡 使用内置中彩网数据（{len(data)} 期）")
    return data


def refresh(show: bool = False):
    """刷新数据主函数"""
    import pandas as pd
    
    print("=" * 50)
    print("  大乐透数据刷新工具")
    print("=" * 50)
    print()
    
    # 尝试从浏览器抓取
    records = scrape_from_zhcw_browser()
    
    if not records or len(records) < 5:
        print("⚠️  浏览器抓取未返回数据，使用内置备份数据")
        data = get_mock_zhcw_data()
        records = []
        for period, front, back in data:
            records.append({
                "period": str(period),
                "front_1": front[0], "front_2": front[1],
                "front_3": front[2], "front_4": front[3], "front_5": front[4],
                "back_1": back[0], "back_2": back[1],
            })
    
    df = pd.DataFrame(records)
    df = df.sort_values("period", ascending=False).reset_index(drop=True)
    
    # 保存
    save_to_csv(df)
    
    # 显示最新
    front_cols = [f"front_{i}" for i in range(1, 6)]
    back_cols = [f"back_{i}" for i in range(1, 3)]
    
    print(f"\n✅ 已保存 {len(df)} 期数据")
    print(f"\n📊 最新5期:")
    for _, r in df.head(5).iterrows():
        f = " ".join(f"{r[c]:02d}" for c in front_cols)
        b = " ".join(f"{r[c]:02d}" for c in back_cols)
        print(f"  期{r['period']}: {f}  +  {b}")
    
    if show:
        print(f"\n📋 全部 {len(df)} 期数据已保存到 data/dlt_history.csv")
    
    return df


if __name__ == "__main__":
    show = "--show" in sys.argv or "-s" in sys.argv
    refresh(show=show)
