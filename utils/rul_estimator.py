"""
RUL 预测与评估模块

- 加载 CMAPSS 测试集与 RUL 真值
- 从轨迹估计 RUL（退化指标越过失效阈值）
- 计算 RMSE、PHM Score
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
from .metrics import rul_rmse, phm_score


def load_rul_labels(data_root: str, dataset: str = "FD001") -> np.ndarray:
    """加载 RUL 真值。RUL_FD001.txt 每行一个发动机的 RUL。"""
    p = Path(data_root) / f"RUL_{dataset}.txt"
    if not p.exists():
        raise FileNotFoundError(f"RUL file not found: {p}")
    return np.loadtxt(p, dtype=np.float64)


def load_test_sequences_last_window(
    data_root: str,
    dataset: str = "FD001",
    seq_length: int = 256,
    sensor_indices: Optional[List[int]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    加载测试集：每个发动机取最后 seq_length 个周期作为观测序列。
    返回: sequences (N, T, F), last_cycle_per_engine (N,), engine_ids (N,)
    """
    import pandas as pd
    sensor_indices = sensor_indices or [2, 3, 7, 14]
    p = Path(data_root) / f"test_{dataset}.txt"
    if not p.exists():
        raise FileNotFoundError(f"Test file not found: {p}")
    df = pd.read_csv(p, sep=r"\s+", header=None)
    data = df.values
    units = np.unique(data[:, 0]).astype(int)
    seqs = []
    last_cycles = []
    for u in units:
        block = data[data[:, 0] == u]
        block = block[np.argsort(block[:, 1])]
        x = block[:, sensor_indices].astype(np.float32)
        cycles = block[:, 1].astype(np.float32)
        if len(x) < 1:
            continue
        # 取最后 min(seq_length, len(x)) 个周期；不足则左侧用首行重复填充，保证所有发动机都参与评估
        n_take = min(seq_length, len(x))
        x_last = x[-n_take:]
        cycles_last = cycles[-n_take:]
        if n_take < seq_length:
            pad = np.repeat(x_last[0:1], seq_length - n_take, axis=0)
            x_last = np.concatenate([pad, x_last], axis=0)
            cycles_last = np.concatenate([np.repeat(cycles_last[0], seq_length - n_take), cycles_last])
        seqs.append(x_last)
        last_cycles.append(cycles_last[-1])
    if not seqs:
        return np.zeros((0, seq_length, len(sensor_indices))), np.array([]), np.array([])
    return np.stack(seqs), np.array(last_cycles), units[: len(seqs)]


def estimate_rul_from_trajectory(
    trajectory: np.ndarray,
    deg_feature_idx: int = 1,
    failure_threshold: float = 0.8,
    normalize: bool = True,
) -> float:
    """
    从轨迹估计 RUL：当退化指标首次超过 (1-failure_threshold) 时的步数。
    轨迹: (T, F)，deg_feature_idx 为退化特征列（如 T30=1）。
    归一化空间 [-1,1] 下：degradation_proxy = (x + 1)/2，失效当 proxy > (1 - failure_threshold)。
    返回: 首次失效步数，若未失效则返回 len(trajectory)
    """
    feat = trajectory[:, deg_feature_idx]
    # 退化代理: [-1,1] -> [0,1]，越高越退化
    if normalize:
        deg = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)
    else:
        deg = (feat + 1) / 2
    thresh = 1.0 - failure_threshold
    idx = np.where(deg >= thresh)[0]
    return float(idx[0]) if len(idx) > 0 else float(len(trajectory))


def evaluate_rul(
    data_root: str,
    dataset: str = "FD001",
    pred_rul: Optional[np.ndarray] = None,
    results_dir: Optional[str] = None,
    failure_threshold: float = 0.8,
) -> dict:
    """
    评估 RUL。若 pred_rul 为 None 且 results_dir 存在，则从 results/rul_predictions_{dataset}.npy 加载。
    返回: {'rmse': float, 'phm_score': float}
    """
    true_rul = load_rul_labels(data_root, dataset)
    if pred_rul is None and results_dir:
        p = Path(results_dir) / f"rul_predictions_{dataset}.npy"
        if p.exists():
            pred_rul = np.load(p)
    if pred_rul is None:
        return {"rmse": None, "phm_score": None, "error": "No predictions"}
    n = min(len(true_rul), len(pred_rul))
    true_rul = true_rul[:n]
    pred_rul = np.asarray(pred_rul, dtype=np.float64)[:n]
    return {
        "rmse": float(rul_rmse(pred_rul, true_rul)),
        "phm_score": float(phm_score(pred_rul, true_rul)),
        "n_samples": n,
    }
