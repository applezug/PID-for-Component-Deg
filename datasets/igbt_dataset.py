"""
NASA IGBT 数据集：从 data/NASA IGBT/Data 下各 Part 的 Turn On.csv（或指定 CSV）加载曲线，
重采样为固定 seq_length，接口与 BaseDegradationDataset 一致。
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Tuple
from .base_dataset import BaseDegradationDataset


def _find_igbt_csvs(data_root: Path, csv_name: str = "Turn On.csv") -> List[Path]:
    """递归查找所有 Part* 目录下的指定 CSV。"""
    data_root = Path(data_root).resolve()
    data_dir = data_root / "Data" if (data_root / "Data").exists() else data_root
    if not data_dir.exists():
        return []
    found = []
    for p in data_dir.rglob(csv_name):
        # 路径中任一段包含 "Part" 即视为 Part 目录（如 "Part 1", "Part 8L"）
        if any("Part" in part for part in p.parts):
            found.append(p)
    return sorted(found)


def _load_and_resample(csv_path: Path, seq_length: int) -> np.ndarray:
    """加载两列 CSV，重采样为 seq_length 长度，(seq_length, 2)。"""
    try:
        df = pd.read_csv(csv_path, header=None, sep=",", dtype=np.float64, on_bad_lines="skip")
    except Exception:
        return np.zeros((seq_length, 2), dtype=np.float32)
    if df.shape[0] < 2 or df.shape[1] < 2:
        return np.zeros((seq_length, 2), dtype=np.float32)
    raw = df.iloc[:, :2].values.astype(np.float64)
    # 去除 NaN/Inf
    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    n = raw.shape[0]
    if n < 2:
        return np.zeros((seq_length, 2), dtype=np.float32)
    old_idx = np.linspace(0, 1, n)
    new_idx = np.linspace(0, 1, seq_length)
    out = np.column_stack([
        np.interp(new_idx, old_idx, raw[:, 0]),
        np.interp(new_idx, old_idx, raw[:, 1]),
    ]).astype(np.float32)
    return out


class IGTBDataset(BaseDegradationDataset):
    """
    从 data/NASA IGBT 目录加载各 Part 的 IV/特性曲线，每条曲线重采样为 (seq_length, 2)。
    period: 'train' | 'val' | 'test'，按 train_ratio / val_ratio 划分。
    """

    def __init__(
        self,
        data_root: str,
        seq_length: int = 256,
        period: str = "train",
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        csv_name: str = "Turn On.csv",
        seed: int = 42,
        normalize: bool = True,
        norm_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ):
        self.data_root = Path(data_root)
        self.seq_length = seq_length
        self.period = period
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.seed = seed

        csv_paths = _find_igbt_csvs(self.data_root, csv_name=csv_name)
        if not csv_paths:
            raise FileNotFoundError(
                f"未在 {self.data_root} 下找到任何 Part 目录中的 {csv_name}。"
                "请确认路径为 data/NASA IGBT 且 Data/ 内包含 Part*/*.csv。"
            )

        sequences = []
        for p in csv_paths:
            seq = _load_and_resample(p, seq_length)
            sequences.append(seq)
        sequences = np.stack(sequences)

        n = len(sequences)
        np.random.seed(seed)
        idx = np.random.permutation(n)
        n_train = max(1, int(n * train_ratio))
        n_val = max(0, int(n * val_ratio))
        n_test = n - n_train - n_val
        train_idx = idx[:n_train]
        val_idx = idx[n_train : n_train + n_val]
        test_idx = idx[n_train + n_val :]

        if period == "train":
            idx_use = train_idx
        elif period == "val":
            idx_use = val_idx if len(val_idx) > 0 else train_idx[:1]
        else:
            idx_use = test_idx if len(test_idx) > 0 else train_idx[:1]

        sequences = sequences[idx_use]
        time_points = np.tile(np.linspace(0, 1, seq_length, dtype=np.float32), (len(sequences), 1))
        masks = np.ones_like(sequences, dtype=bool)
        engine_ids = np.arange(len(sequences))

        if normalize and norm_stats is None and len(sequences) > 0:
            flat = sequences.reshape(-1, sequences.shape[-1])
            min_v = flat.min(axis=0)
            max_v = flat.max(axis=0)
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
        self._n_total = n
        self._csv_paths = csv_paths
