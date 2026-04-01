"""Base dataset for degradation sequences."""

import torch
from torch.utils.data import Dataset
import numpy as np


class BaseDegradationDataset(Dataset):
    """Returns (sequence, time_points, mask, engine_id). sequence (T,F), time (T), mask (T,F)."""

    def __init__(self, sequences, time_points=None, masks=None, engine_ids=None, normalize=True, norm_stats=None):
        self.sequences = np.asarray(sequences)
        self.n_samples, self.seq_length, self.feature_size = self.sequences.shape
        self.time_points = np.asarray(time_points) if time_points is not None else None
        self.masks = np.asarray(masks) if masks is not None else None
        self.engine_ids = np.asarray(engine_ids) if engine_ids is not None else None
        self.normalize = normalize
        self.norm_stats = norm_stats

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x = torch.from_numpy(self.sequences[idx]).float()
        if self.time_points is not None:
            t = torch.from_numpy(self.time_points[idx]).float()
        else:
            t = torch.linspace(0, 1, self.seq_length)
        if self.masks is not None:
            m = torch.from_numpy(self.masks[idx])
        else:
            m = torch.ones_like(x, dtype=torch.bool)
        if self.normalize and self.norm_stats is not None:
            min_v, max_v = self.norm_stats
            x = 2.0 * (x - min_v) / (max_v - min_v + 1e-8) - 1.0
        eid = self.engine_ids[idx] if self.engine_ids is not None else idx
        return x, t, m, eid
