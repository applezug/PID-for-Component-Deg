"""
Model-Based Diffusion 工具函数（From project 3）
"""

import torch
import numpy as np
from typing import Tuple, Optional


def create_missing_mask(
    seq_length: int,
    feature_size: int,
    missing_rate: float,
    missing_pattern: str = 'random',
    device: str = 'cpu',
) -> torch.Tensor:
    mask = torch.ones(seq_length, feature_size, dtype=torch.bool, device=device)
    n_missing = int(seq_length * feature_size * missing_rate)

    if missing_pattern == 'random':
        indices = torch.randperm(seq_length * feature_size, device=device)[:n_missing]
        mask.view(-1)[indices] = False
    elif missing_pattern == 'block':
        block_size = max(1, n_missing // feature_size)
        start_idx = torch.randint(0, max(1, seq_length - block_size), (1,), device=device).item()
        end_idx = min(start_idx + block_size, seq_length)
        mask[start_idx:end_idx, :] = False
    elif missing_pattern == 'start':
        n_missing_steps = int(seq_length * missing_rate)
        mask[:n_missing_steps, :] = False
    elif missing_pattern == 'end':
        n_missing_steps = int(seq_length * missing_rate)
        mask[-n_missing_steps:, :] = False
    else:
        raise ValueError(f'unknown missing pattern: {missing_pattern}')

    return mask


def normalize_data(
    data: torch.Tensor,
    method: str = 'minmax',
    stats: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    if stats is not None:
        if method == 'minmax':
            min_val, max_val = stats
            normalized = (data - min_val) / (max_val - min_val + 1e-8)
            normalized = normalized * 2.0 - 1.0
        elif method == 'zscore':
            mean, std = stats
            normalized = (data - mean) / (std + 1e-8)
        else:
            raise ValueError(f'unknown normalization method: {method}')
        return normalized, stats

    if method == 'minmax':
        if data.dim() == 2:
            min_val = data.min(dim=0, keepdim=True)[0]
            max_val = data.max(dim=0, keepdim=True)[0]
        else:
            min_val = data.min(dim=1, keepdim=True)[0].min(dim=0, keepdim=True)[0]
            max_val = data.max(dim=1, keepdim=True)[0].max(dim=0, keepdim=True)[0]
        normalized = (data - min_val) / (max_val - min_val + 1e-8)
        normalized = normalized * 2.0 - 1.0
        stats = (min_val, max_val)
    elif method == 'zscore':
        if data.dim() == 2:
            mean = data.mean(dim=0, keepdim=True)
            std = data.std(dim=0, keepdim=True)
        else:
            mean = data.mean(dim=1, keepdim=True).mean(dim=0, keepdim=True)
            std = data.std(dim=1, keepdim=True).std(dim=0, keepdim=True)
        normalized = (data - mean) / (std + 1e-8)
        stats = (mean, std)
    else:
        raise ValueError(f'unknown normalization method: {method}')

    return normalized, stats


def denormalize_data(
    normalized_data: torch.Tensor,
    stats: Tuple[torch.Tensor, torch.Tensor],
    method: str = 'minmax',
) -> torch.Tensor:
    if method == 'minmax':
        min_val, max_val = stats
        normalized = (normalized_data + 1.0) / 2.0
        denormalized = normalized * (max_val - min_val + 1e-8) + min_val
    elif method == 'zscore':
        mean, std = stats
        denormalized = normalized_data * (std + 1e-8) + mean
    else:
        raise ValueError(f'unknown normalization method: {method}')
    return denormalized


def compute_metrics(
    imputed_data: torch.Tensor,
    true_data: torch.Tensor,
    mask: torch.Tensor,
) -> dict:
    missing_mask = ~mask
    if missing_mask.sum() == 0:
        return {'mse': 0.0, 'mae': 0.0, 'mape': 0.0, 'rmse': 0.0}

    imputed_missing = imputed_data[missing_mask]
    true_missing = true_data[missing_mask]
    mse = torch.mean((imputed_missing - true_missing) ** 2).item()
    mae = torch.mean(torch.abs(imputed_missing - true_missing)).item()
    rmse = np.sqrt(mse)
    mape = torch.mean(torch.abs((imputed_missing - true_missing) / (true_missing + 1e-8))).item() * 100
    return {'mse': mse, 'mae': mae, 'mape': mape, 'rmse': rmse}


def create_time_points(
    seq_length: int,
    start_time: float = 0.0,
    end_time: Optional[float] = None,
    device: str = 'cpu',
) -> torch.Tensor:
    if end_time is None:
        end_time = float(seq_length)
    return torch.linspace(start_time, end_time, seq_length, device=device)
