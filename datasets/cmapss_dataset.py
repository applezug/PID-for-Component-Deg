"""
CMAPSS dataset: load FD001-FD004, extract HPC sensors (T24, T30, P30, Nc).
Columns: 0=unit, 1=time, 2-25=sensors. Use sensor_indices [2,3,7,14] -> T24,T30,P30,Nc (0-indexed: 2,3,7,14).
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Tuple
from .base_dataset import BaseDegradationDataset


def load_cmapss_file(path: str) -> np.ndarray:
    """Load single CMAPSS txt (no header). Columns: unit, time, s1, s2, ..."""
    df = pd.read_csv(path, sep=r'\s+', header=None)
    return df.values


def sliding_windows(data: np.ndarray, unit_col: int, seq_length: int, sensor_cols: List[int], step: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    data: (n_rows, n_cols). unit_col=0, time in 1, sensors in sensor_cols.
    Returns: sequences (N, T, F), time_normalized (N, T), unit_ids (N,).
    """
    units = np.unique(data[:, unit_col])
    sequences = []
    time_norms = []
    unit_ids = []
    for u in units:
        block = data[data[:, unit_col] == u]
        block = block[np.argsort(block[:, 1])]  # sort by time
        x = block[:, sensor_cols].astype(np.float32)
        t = block[:, 1].astype(np.float32)
        t = (t - t.min()) / (t.max() - t.min() + 1e-8)
        for i in range(0, len(x) - seq_length + 1, step):
            sequences.append(x[i:i + seq_length])
            time_norms.append(t[i:i + seq_length])
            unit_ids.append(u)
    if not sequences:
        return np.zeros((0, seq_length, len(sensor_cols)), dtype=np.float32), np.zeros((0, seq_length), dtype=np.float32), np.array([], dtype=np.int64)
    return np.stack(sequences), np.stack(time_norms), np.array(unit_ids)


class CMAPSSDataset(BaseDegradationDataset):
    def __init__(
        self,
        data_root: str,
        dataset: str = 'FD001',
        seq_length: int = 256,
        step: int = 10,
        period: str = 'train',
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        sensor_indices: Optional[List[int]] = None,
        seed: int = 42,
        normalize: bool = True,
        norm_stats: Optional[Tuple] = None,
    ):
        """
        data_root: folder containing train_FD001.txt, test_FD001.txt, RUL_FD001.txt.
        sensor_indices: 0-based column indices (e.g. [2,3,7,14] for T24,T30,P30,Nc).
        """
        self.data_root = Path(data_root)
        self.dataset = dataset
        self.seq_length = seq_length
        self.sensor_indices = sensor_indices or [2, 3, 7, 14]
        self.period = period
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.seed = seed

        train_path = self.data_root / f'train_{dataset}.txt'
        if not train_path.exists():
            raise FileNotFoundError(f'CMAPSS not found: {train_path}. Put train_FD001.txt etc. in {data_root}')

        data = load_cmapss_file(str(train_path))
        seqs, times, units = sliding_windows(data, unit_col=0, seq_length=seq_length, sensor_cols=self.sensor_indices, step=step)

        np.random.seed(seed)
        n = len(seqs)
        idx = np.random.permutation(n)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val
        train_idx = idx[:n_train]
        val_idx = idx[n_train:n_train + n_val]
        test_idx = idx[n_train + n_val:]

        if period == 'train':
            idx_use = train_idx
        elif period == 'val':
            idx_use = val_idx
        else:
            idx_use = test_idx

        sequences = seqs[idx_use]
        time_points = times[idx_use]
        engine_ids = units[idx_use]
        masks = np.ones_like(sequences, dtype=bool)

        if normalize and norm_stats is None and len(sequences) > 0:
            min_v = sequences.reshape(-1, sequences.shape[-1]).min(axis=0)
            max_v = sequences.reshape(-1, sequences.shape[-1]).max(axis=0)
            norm_stats = (min_v, max_v)

        super().__init__(
            sequences=sequences,
            time_points=time_points,
            masks=masks,
            engine_ids=engine_ids,
            normalize=normalize,
            norm_stats=norm_stats,
        )
        self.global_norm_stats = norm_stats
