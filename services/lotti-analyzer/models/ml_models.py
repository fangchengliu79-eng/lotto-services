"""
机器学习模型模块 - LSTM 序列模型和 XGBoost/RandomForest 分类模型

LSTM：将号码出现视为时间序列，使用历史 N 期作为输入特征，
       预测下一期各号码的出现概率。
XGBoost/RF：基于历史频率、遗漏值、近期趋势等特征，
            分类预测热门号码。
"""
import os
import pickle
import warnings
from typing import Dict, List, Tuple, Optional
from collections import Counter

import numpy as np
import pandas as pd

from config import (
    FRONT_MIN, FRONT_MAX, BACK_MIN, BACK_MAX,
    FRONT_COUNT, BACK_COUNT, MODEL_DIR, LSTM_EPOCHS, LSTM_BATCH_SIZE, XGB_PARAMS,
)
from analysis.statistics import frequency_analysis, missing_value_analysis
from utils.helpers import logger

warnings.filterwarnings("ignore")


def _build_features(df: pd.DataFrame) -> np.ndarray:
    """
    构建特征矩阵用于 ML 模型
    特征包括：历史频率、遗漏值、近期出现次数、奇偶特征等
    """
    total_draws = len(df)
    front_cols = [f"front_{i}" for i in range(1, 6)]
    back_cols = [f"back_{i}" for i in range(1, 3)]

    # 计算基础统计
    freq = frequency_analysis(df)
    missing = missing_value_analysis(df)

    features = []
    for n in range(FRONT_MIN, FRONT_MAX + 1):
        f = []
        # 频率
        f.append(freq["front_freq"].get(n, 0) / max(total_draws, 1))
        # 遗漏
        f.append(missing["front_missing"].get(n, total_draws) / max(total_draws, 1))
        # 最近 20 期出现次数
        recent_20 = df.head(20)
        count_recent = sum(1 for _, r in recent_20.iterrows() if n in r[front_cols].values)
        f.append(count_recent / 20)
        # 最近 50 期出现次数
        recent_50 = df.head(50)
        count_recent_50 = sum(1 for _, r in recent_50.iterrows() if n in r[front_cols].values)
        f.append(count_recent_50 / 50)
        # 平均遗漏（历史）
        appeared_periods = []
        for idx, row in df.iterrows():
            if n in row[front_cols].values:
                appeared_periods.append(idx)
        if len(appeared_periods) > 1:
            intervals = np.diff(appeared_periods)
            f.append(np.mean(intervals) / max(total_draws, 1))
            f.append(np.std(intervals) / max(total_draws, 1))
        else:
            f.append(1.0)
            f.append(0.5)
        # 奇偶特征（该号码本身）
        f.append(1 if n % 2 == 1 else 0)
        # 大小特征
        f.append(1 if n > 17 else 0)

        features.append(f)

    return np.array(features)


class LSTMSequenceModel:
    """
    LSTM 序列预测模型

    原理：将历史期次视为时间序列，用过去 N 期的号码出现模式
    （每个号码是否出现）作为输入，通过 LSTM 网络学习时序依赖关系，
    输出下一期各号码的出现概率。

    架构：使用 Embedding + LSTM 层，输出为 35 分类（前区）
    和 12 分类（后区），分别训练。
    """

    def __init__(self, df: pd.DataFrame, seq_length: int = 20):
        self.df = df
        self.seq_length = min(seq_length, max(10, len(df) // 10))
        self.total_draws = len(df)
        self.front_model = None
        self.back_model = None
        self.front_predictions = None
        self.back_predictions = None
        self.model_trained = False

    def _prepare_sequences(self, zone: str = "front") -> Tuple[np.ndarray, np.ndarray]:
        """
        准备好序列数据
        输入：过去 N 期的号码出现向量
        输出：下一期各号码的出现情况
        """
        max_num = FRONT_MAX if zone == "front" else BACK_MAX
        min_num = FRONT_MIN if zone == "front" else BACK_MIN
        num_count = FRONT_COUNT if zone == "front" else BACK_COUNT
        cols = [f"{zone}_{i}" for i in range(1, num_count + 1)]

        # 构建出现矩阵: [期次, 号码] -> 0/1
        appears = np.zeros((self.total_draws, max_num - min_num + 1))
        for idx, row in self.df.iterrows():
            for c in cols:
                n = int(row[c])
                appears[idx, n - min_num] = 1

        # 构建序列
        X, y = [], []
        for i in range(self.seq_length, self.total_draws):
            X.append(appears[i - self.seq_length:i])
            y.append(appears[i])

        return np.array(X), np.array(y)

    def train(self, force_retrain: bool = False):
        """
        训练 LSTM 模型
        如果已保存的模型存在且不强制重新训练，则加载预训练模型。
        """
        front_path = MODEL_DIR / "lstm_front.keras"
        back_path = MODEL_DIR / "lstm_back.keras"
        front_pred_path = MODEL_DIR / "lstm_front_probs.npy"
        back_pred_path = MODEL_DIR / "lstm_back_probs.npy"

        # 尝试加载已保存的结果
        if not force_retrain:
            if front_pred_path.exists() and back_pred_path.exists():
                try:
                    self.front_predictions = np.load(str(front_pred_path))
                    self.back_predictions = np.load(str(back_pred_path))
                    self.model_trained = True
                    logger.info("已加载预训练的 LSTM 预测结果")
                    return True
                except Exception as e:
                    logger.warning(f"加载 LSTM 结果失败: {e}")

        # 如果数据太少，跳过 LSTM
        if self.total_draws < self.seq_length + 10:
            logger.warning(f"数据不足 ({self.total_draws} 期)，跳过 LSTM 训练")
            self._fallback_prediction()
            return False

        try:
            import tensorflow as tf
            from tensorflow import keras
            from tensorflow.keras import layers
        except ImportError:
            logger.warning("TensorFlow/Keras 未安装，跳过 LSTM 训练")
            self._fallback_prediction()
            return False

        logger.info("开始训练 LSTM 模型（前区）...")
        try:
            X_front, y_front = self._prepare_sequences("front")

            if len(X_front) < 10:
                logger.warning("序列数据不足，跳过 LSTM")
                self._fallback_prediction()
                return False

            # 构建模型
            front_input_dim = FRONT_MAX - FRONT_MIN + 1  # 35
            front_model = keras.Sequential([
                layers.LSTM(64, return_sequences=True, input_shape=(self.seq_length, front_input_dim)),
                layers.Dropout(0.2),
                layers.LSTM(32),
                layers.Dropout(0.2),
                layers.Dense(32, activation="relu"),
                layers.Dense(front_input_dim, activation="sigmoid"),
            ])
            front_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

            # 训练
            front_model.fit(
                X_front, y_front,
                epochs=min(LSTM_EPOCHS, 20),
                batch_size=LSTM_BATCH_SIZE,
                validation_split=0.1,
                verbose=0,
                callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
            )

            # 预测下一期
            last_seq = X_front[-1:]  # 最新序列
            self.front_predictions = front_model.predict(last_seq, verbose=0).flatten()

            # 保存模型和预测
            try:
                front_model.save(str(front_path))
                np.save(str(front_pred_path), self.front_predictions)
            except Exception:
                pass

            logger.info("LSTM 前区模型训练完成")
        except Exception as e:
            logger.error(f"LSTM 前区训练失败: {e}")
            self._fallback_prediction("front")
            return False

        logger.info("开始训练 LSTM 模型（后区）...")
        try:
            X_back, y_back = self._prepare_sequences("back")
            back_input_dim = BACK_MAX - BACK_MIN + 1  # 12
            back_model = keras.Sequential([
                layers.LSTM(32, input_shape=(self.seq_length, back_input_dim)),
                layers.Dropout(0.2),
                layers.Dense(16, activation="relu"),
                layers.Dense(back_input_dim, activation="sigmoid"),
            ])
            back_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

            back_model.fit(
                X_back, y_back,
                epochs=min(LSTM_EPOCHS, 20),
                batch_size=LSTM_BATCH_SIZE,
                validation_split=0.1,
                verbose=0,
                callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
            )

            last_seq = X_back[-1:]
            self.back_predictions = back_model.predict(last_seq, verbose=0).flatten()

            try:
                back_model.save(str(back_path))
                np.save(str(back_pred_path), self.back_predictions)
            except Exception:
                pass

            logger.info("LSTM 后区模型训练完成")
        except Exception as e:
            logger.error(f"LSTM 后区训练失败: {e}")
            self._fallback_prediction("back")

        self.model_trained = True
        return True

    def _fallback_prediction(self, zone: str = "all"):
        """回退：使用历史频率作为预测"""
        freq = frequency_analysis(self.df)
        if zone in ("all", "front"):
            front_total = sum(freq["front_freq"].values())
            self.front_predictions = np.array([
                freq["front_freq"].get(n, 0) / max(front_total, 1)
                for n in range(FRONT_MIN, FRONT_MAX + 1)
            ])
        if zone in ("all", "back"):
            back_total = sum(freq["back_freq"].values())
            self.back_predictions = np.array([
                freq["back_freq"].get(n, 0) / max(back_total, 1)
                for n in range(BACK_MIN, BACK_MAX + 1)
            ])

    def predict_front_probabilities(self) -> Dict[int, float]:
        """获取前区预测概率"""
        if self.front_predictions is None:
            self.train()
        probs = self.front_predictions
        # 归一化
        total = probs.sum()
        if total > 0:
            probs = probs / total
        return {i + 1: float(probs[i]) for i in range(len(probs))}

    def predict_back_probabilities(self) -> Dict[int, float]:
        """获取后区预测概率"""
        if self.back_predictions is None:
            self.train()
        probs = self.back_predictions
        total = probs.sum()
        if total > 0:
            probs = probs / total
        return {i + 1: float(probs[i]) for i in range(len(probs))}


class XGBoostModel:
    """
    XGBoost 分类模型

    原理：基于历史频率、遗漏值、近期趋势等工程特征，
    使用 XGBoost 分类器预测哪些号码更可能在下期出现。
    每个号码视为一个样本，特征为该号码的各种统计指标，
    标签为该号码是否在下一期出现。
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.total_draws = len(df)
        self.front_probs = {}
        self.back_probs = {}
        self.model_trained = False

    def train(self):
        """训练 XGBoost 模型"""
        try:
            import xgboost as xgb
        except ImportError:
            logger.warning("XGBoost 未安装，跳过训练")
            self._fallback()
            return False

        front_cols = [f"front_{i}" for i in range(1, 6)]

        try:
            # 构建训练数据
            features_list = []
            labels_list = []

            # 滑动窗口方式构建样本
            window = 50
            for start_idx in range(0, self.total_draws - window - 5, 5):
                train_df = self.df.iloc[start_idx:start_idx + window]
                test_idx = start_idx + window
                test_row = self.df.iloc[test_idx]

                # 构建特征
                feat = _build_features(train_df)
                # 标签：下期实际出现的号码
                label = np.zeros(35)
                for c in front_cols:
                    n = int(test_row[c])
                    label[n - 1] = 1

                features_list.append(feat)
                labels_list.append(label)

            if not features_list:
                self._fallback()
                return False

            X = np.vstack(features_list)
            y = np.vstack(labels_list)

            # 训练多标签分类器
            model = xgb.XGBClassifier(
                **XGB_PARAMS,
                objective="multi:softprob",
                num_class=35,
                eval_metric="mlogloss",
                use_label_encoder=False,
                verbosity=0,
            )
            model.fit(X, y, verbose=False)

            # 预测下一期
            latest_features = _build_features(self.df)
            pred_probs = model.predict_proba(latest_features)

            # XGBoost predict_proba 返回的格式不同
            # 对 multi:softprob, 使用 predict_proba
            if isinstance(pred_probs, list):
                # multiclass 返回 list of arrays
                self.front_probs = {i + 1: float(np.mean([p[i] for p in pred_probs]))
                                    for i in range(35)}
            else:
                self.front_probs = {i + 1: float(pred_probs[i][1]) if pred_probs.ndim > 1 and pred_probs.shape[1] > 1
                                    else float(pred_probs[i])
                                    for i in range(min(len(pred_probs), 35))}

            # 后区 - 简化版本（特征较少，使用简单频率）
            back_freq = frequency_analysis(self.df)["back_freq"]
            total = sum(back_freq.values())
            self.back_probs = {n: back_freq.get(n, 0) / max(total, 1)
                               for n in range(BACK_MIN, BACK_MAX + 1)}

            # 归一化
            front_total = sum(self.front_probs.values())
            if front_total > 0:
                self.front_probs = {n: p / front_total for n, p in self.front_probs.items()}

            self.model_trained = True
            logger.info("XGBoost 模型训练完成")
            return True

        except Exception as e:
            logger.error(f"XGBoost 训练失败: {e}")
            self._fallback()
            return False

    def _fallback(self):
        """回退到频率分析"""
        freq = frequency_analysis(self.df)
        total_f = sum(freq["front_freq"].values())
        self.front_probs = {n: freq["front_freq"].get(n, 0) / max(total_f, 1)
                           for n in range(FRONT_MIN, FRONT_MAX + 1)}
        total_b = sum(freq["back_freq"].values())
        self.back_probs = {n: freq["back_freq"].get(n, 0) / max(total_b, 1)
                          for n in range(BACK_MIN, BACK_MAX + 1)}

    def predict_front_probabilities(self) -> Dict[int, float]:
        if not self.model_trained:
            self.train()
        return self.front_probs

    def predict_back_probabilities(self) -> Dict[int, float]:
        if not self.model_trained:
            self.train()
        return self.back_probs


class RandomForestModel:
    """RandomForest 分类模型（XGBoost 的备选）"""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.total_draws = len(df)
        self.front_probs = {}
        self.back_probs = {}
        self.model_trained = False

    def train(self):
        """训练 RandomForest 模型（简化版：单分类器 + 特征工程）"""
        try:
            from sklearn.ensemble import RandomForestClassifier
        except ImportError:
            logger.warning("scikit-learn 未安装，跳过 RandomForest")
            self._fallback()
            return False

        try:
            # 使用所有历史数据构建特征
            features = _build_features(self.df)
            front_cols = [f"front_{i}" for i in range(1, 6)]

            # 标签：每个号码在过去 N 期中出现的平均次数
            labels = np.zeros(35)
            recent_n = min(20, self.total_draws)
            recent_df = self.df.head(recent_n)
            for n in range(FRONT_MIN, FRONT_MAX + 1):
                count = sum(1 for _, r in recent_df.iterrows() if n in r[front_cols].values)
                labels[n - 1] = count / recent_n

            # 训练单个多输出回归器
            clf = RandomForestClassifier(
                n_estimators=100, max_depth=6,
                random_state=42, n_jobs=1, verbose=0
            )
            # 转二进制标签：将出现次数超过均值+1std的设为1
            threshold = np.mean(labels) + np.std(labels)
            binary_labels = (labels > threshold).astype(int)
            clf.fit(features, binary_labels)

            # 预测下期概率
            probs = clf.predict_proba(features)
            # RandomForest predict_proba 返回 [n_samples, n_classes]
            # 对二元分类，取类别1的概率
            if isinstance(probs, list):
                self.front_probs = {i + 1: float(probs[i][0][1]) if probs[i].ndim > 1 and probs[i].shape[1] > 1
                                    else float(probs[i][0] if isinstance(probs[i], (list, np.ndarray)) else probs[i])
                                    for i in range(min(len(probs), 35))}
            elif probs.ndim == 2 and probs.shape[1] == 2:
                self.front_probs = {i + 1: float(probs[i][1]) for i in range(min(len(probs), 35))}
            else:
                self.front_probs = {i + 1: float(probs[i]) for i in range(min(len(probs), 35))}

            # 后区简单频率
            back_freq = frequency_analysis(self.df)["back_freq"]
            b_total = sum(back_freq.values())
            self.back_probs = {n: back_freq.get(n, 0) / max(b_total, 1)
                              for n in range(BACK_MIN, BACK_MAX + 1)}

            # 归一化
            total = sum(self.front_probs.values())
            if total > 0:
                self.front_probs = {n: p / total for n, p in self.front_probs.items()}

            self.model_trained = True
            logger.info("RandomForest 模型训练完成")
            return True

        except Exception as e:
            logger.error(f"RandomForest 训练失败: {e}")
            self._fallback()
            return False

    def _fallback(self):
        freq = frequency_analysis(self.df)
        total_f = sum(freq["front_freq"].values())
        self.front_probs = {n: freq["front_freq"].get(n, 0) / max(total_f, 1)
                           for n in range(FRONT_MIN, FRONT_MAX + 1)}
        total_b = sum(freq["back_freq"].values())
        self.back_probs = {n: freq["back_freq"].get(n, 0) / max(total_b, 1)
                          for n in range(BACK_MIN, BACK_MAX + 1)}

    def predict_front_probabilities(self) -> Dict[int, float]:
        if not self.model_trained:
            self.train()
        return self.front_probs

    def predict_back_probabilities(self) -> Dict[int, float]:
        if not self.model_trained:
            self.train()
        return self.back_probs
