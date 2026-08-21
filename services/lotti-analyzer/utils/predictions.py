"""
预测封存与比对模块

功能：
1. 每次生成预测时自动封存到 JSON 文件
2. 当新的开奖数据到来时，自动比对并标出命中号码
3. 页面刷新不重新生成预测，保持一致性
"""
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd

from config import DATA_DIR, FRONT_COUNT, BACK_COUNT
from utils.helpers import logger

PREDICTIONS_FILE = DATA_DIR / "predictions.json"


def _load_all() -> List[Dict]:
    """加载所有历史预测"""
    if not PREDICTIONS_FILE.exists():
        return []
    try:
        with open(PREDICTIONS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except (json.JSONDecodeError, Exception) as e:
        logger.warning(f"读取预测历史失败: {e}")
        return []


def _save_all(predictions: List[Dict]):
    """保存所有预测历史"""
    PREDICTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PREDICTIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)


def save_prediction(
    period: str,
    draw_date: str,
    recommendations: List[Dict],
    models_used: List[str] = None,
) -> str:
    """
    封存一组预测
    
    Args:
        period: 预测对应的期号（如 "26059"）
        draw_date: 开奖日期
        recommendations: 推荐号码列表
        models_used: 使用的模型列表
    
    Returns:
        pred_id: 预测ID
    """
    predictions = _load_all()
    
    pred_id = f"pred_{period}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    entry = {
        "id": pred_id,
        "period": str(period),
        "draw_date": draw_date,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "active",  # "active" = 待开奖, "completed" = 已比对
        "models_used": models_used or ["frequency", "poisson", "exponential_smoothing", "monte_carlo"],
        "recommendations": [
            {
                "group": i + 1,
                "front": r["front"],
                "back": r["back"],
                "score": round(r["score"], 4),
                "reason": r["reason"],
            }
            for i, r in enumerate(recommendations)
        ],
        "actual_draw": None,
        "matches": None,
        "summary": None,
    }
    
    # 检查是否已有同一期号的预测
    existing = [p for p in predictions if p["period"] == str(period) and p["status"] != "archived"]
    if existing:
        logger.info(f"期号 {period} 已有预测，覆盖更新")
        predictions = [p for p in predictions if p["id"] != existing[0]["id"]]
    
    predictions.append(entry)
    _save_all(predictions)
    logger.info(f"预测已封存: {period} ({pred_id})")
    
    return pred_id


def get_latest_prediction() -> Optional[Dict]:
    """获取最新的预测记录"""
    predictions = _load_all()
    if not predictions:
        return None
    # 按创建时间排序，取最新的
    predictions.sort(key=lambda p: p.get("created_at", ""), reverse=True)
    return predictions[0]


def get_prediction_by_period(period: str) -> Optional[Dict]:
    """根据期号获取预测"""
    predictions = _load_all()
    for p in predictions:
        if p["period"] == str(period):
            return p
    return None


def get_all_predictions() -> List[Dict]:
    """获取所有预测（按时间倒序）"""
    predictions = _load_all()
    predictions.sort(key=lambda p: p.get("created_at", ""), reverse=True)
    return predictions


def compare_with_draw(prediction: Dict, df: pd.DataFrame) -> Dict:
    """
    将预测与实际开奖号码比对
    
    Args:
        prediction: 预测记录
        df: 历史数据 DataFrame
    
    Returns:
        更新后的 prediction（含比对结果）
    """
    period = prediction["period"]
    front_cols = [f"front_{i}" for i in range(1, 6)]
    back_cols = [f"back_{i}" for i in range(1, 3)]
    
    # 查找实际开奖
    draw_row = df[df["period"] == period]
    if draw_row.empty:
        logger.info(f"期号 {period} 的开奖数据尚未获取")
        return prediction
    
    actual_front = sorted([int(draw_row.iloc[0][c]) for c in front_cols])
    actual_back = sorted([int(draw_row.iloc[0][c]) for c in back_cols])
    
    # 比对每组推荐
    matches = []
    total_front_hits = 0
    total_back_hits = 0
    best_group = 0
    best_hits = 0
    
    for rec in prediction["recommendations"]:
        pred_front = rec["front"]
        pred_back = rec["back"]
        
        front_matches = sorted([n for n in pred_front if n in actual_front])
        back_matches = sorted([n for n in pred_back if n in actual_back])
        total = len(front_matches) + len(back_matches)
        
        match_info = {
            "group": rec["group"],
            "front_matches": front_matches,
            "back_matches": back_matches,
            "front_hits": len(front_matches),
            "back_hits": len(back_matches),
            "total_hits": total,
        }
        matches.append(match_info)
        
        total_front_hits += len(front_matches)
        total_back_hits += len(back_matches)
        
        if total > best_hits:
            best_hits = total
            best_group = rec["group"]
    
    num_groups = len(prediction["recommendations"])
    
    summary = {
        "actual_front": actual_front,
        "actual_back": actual_back,
        "best_group": best_group,
        "best_hits": best_hits,
        "avg_front_hits": round(total_front_hits / num_groups, 2) if num_groups > 0 else 0,
        "avg_back_hits": round(total_back_hits / num_groups, 2) if num_groups > 0 else 0,
        "avg_total_hits": round((total_front_hits + total_back_hits) / num_groups, 2) if num_groups > 0 else 0,
    }
    
    prediction["actual_draw"] = {
        "front": actual_front,
        "back": actual_back,
    }
    prediction["matches"] = matches
    prediction["summary"] = summary
    prediction["status"] = "completed"
    
    # 保存更新
    all_predictions = _load_all()
    for i, p in enumerate(all_predictions):
        if p["id"] == prediction["id"]:
            all_predictions[i] = prediction
            break
    _save_all(all_predictions)
    
    logger.info(f"预测 {period} 比对完成: 最佳命中 {best_hits} 个 (第{best_group}组)")
    
    return prediction


def auto_compare_latest(df: pd.DataFrame) -> Optional[Dict]:
    """
    自动查找最新可比对的预测并执行比对
    
    Args:
        df: 历史数据 DataFrame
    
    Returns:
        如有可比对预测则返回更新后的预测，否则 None
    """
    predictions = _load_all()
    
    for pred in predictions:
        if pred["status"] == "active":
            period = pred["period"]
            # 检查历史数据中是否有该期号
            if period in df["period"].values:
                logger.info(f"发现可比对预测: {period}")
                pred = compare_with_draw(pred, df)
                return pred
    
    return None


def has_prediction_for_latest(df: pd.DataFrame) -> Tuple[bool, Optional[Dict]]:
    """
    检查最新一期是否有比对过的预测
    
    Returns:
        (has_prediction, prediction_or_None)
    """
    if df.empty:
        return False, None
    
    latest_period = str(df.iloc[0]["period"])
    predictions = _load_all()
    
    for p in predictions:
        if p["period"] == latest_period and p["status"] == "completed":
            return True, p
    
    return False, None


def get_pending_period(df: pd.DataFrame) -> Optional[str]:
    """
    获取当前应该预测的期号（最新一期之后的下一期）
    
    大乐透每周一、三、六开奖，所以下一期需要根据当前日期推算。
    这里简化为：如果最新一期是今天之前开的，下一期就是最新期号+1
    """
    if df.empty:
        return None
    
    latest_period = int(df.iloc[0]["period"])
    
    # 检查是否已有该期的预测
    predictions = _load_all()
    for p in predictions:
        if p["period"] == str(latest_period) and p["status"] in ("active", "completed"):
            return None  # 该期已有预测，不需要新建
    
    return str(latest_period + 1)


def archive_old_predictions(keep: int = 20):
    """归档旧预测（保留最近 N 条）"""
    predictions = _load_all()
    predictions.sort(key=lambda p: p.get("created_at", ""), reverse=True)
    
    if len(predictions) > keep:
        for p in predictions[keep:]:
            p["status"] = "archived"
        _save_all(predictions)
        logger.info(f"已归档 {len(predictions) - keep} 条旧预测")
