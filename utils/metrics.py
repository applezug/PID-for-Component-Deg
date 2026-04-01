"""
RUL evaluation: RMSE, PHM Score, PICP (prediction interval coverage probability).
"""

import numpy as np
import torch


def rul_rmse(pred: np.ndarray, true: np.ndarray) -> float:
    return np.sqrt(np.mean((pred - true) ** 2))


def phm_score(pred: np.ndarray, true: np.ndarray) -> float:
    """
    PHM2012 scoring: early penalty 10*(e^(d/13)-1), late penalty 10*(e^(-d/10)-1), d = pred - true.
    """
    d = pred - true
    s = np.where(d < 0, 10 * (np.exp(-d / 10) - 1), 10 * (np.exp(d / 13) - 1))
    return np.sum(s)


def picp(
    pred_lower: np.ndarray,
    pred_upper: np.ndarray,
    true: np.ndarray,
) -> float:
    """
    预测区间覆盖率（Prediction Interval Coverage Probability）。
    PICP = (真值落在 [pred_lower, pred_upper] 内的样本数) / 总样本数。
    要求 pred_lower <= pred_upper，且与 true 同长度。
    """
    pred_lower = np.asarray(pred_lower).ravel()
    pred_upper = np.asarray(pred_upper).ravel()
    true = np.asarray(true).ravel()
    n = min(len(pred_lower), len(pred_upper), len(true))
    if n == 0:
        return 0.0
    covered = np.sum((true[:n] >= pred_lower[:n]) & (true[:n] <= pred_upper[:n]))
    return float(covered) / n


def mean_interval_width(pred_lower: np.ndarray, pred_upper: np.ndarray) -> float:
    """预测区间平均宽度（与 PICP 一起汇报，衡量不确定性区间大小）。"""
    pred_lower = np.asarray(pred_lower).ravel()
    pred_upper = np.asarray(pred_upper).ravel()
    n = min(len(pred_lower), len(pred_upper))
    if n == 0:
        return 0.0
    return float(np.mean(np.maximum(0.0, pred_upper[:n] - pred_lower[:n])))
