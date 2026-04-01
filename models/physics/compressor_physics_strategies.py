"""
三种物理损失策略实现：
1. CompressorPhysicsDenorm: 反归一化后使用物理量纲
2. CompressorPhysicsNorm: 归一化空间的新约束形式
3. CompressorPhysicsScaled: 原物理 + λ 缩放（由调用方传入 λ）
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple
from .base_physics import _monotonicity_violation_ratio
from .compressor_physics import CompressorPhysics


def denormalize(x_norm: torch.Tensor, norm_stats: Tuple) -> torch.Tensor:
    """x_norm in [-1,1] -> x_phys"""
    min_v, max_v = norm_stats
    min_v = torch.as_tensor(min_v, device=x_norm.device, dtype=x_norm.dtype)
    max_v = torch.as_tensor(max_v, device=x_norm.device, dtype=x_norm.dtype)
    return (x_norm + 1.0) / 2.0 * (max_v - min_v) + min_v


class CompressorPhysicsDenorm(CompressorPhysics):
    """
    策略1: 在物理层反归一化，使用物理量纲计算损失。
    输入为归一化数据 [-1,1]，内部反归一化后再调用原有物理公式。
    """

    def __init__(
        self,
        norm_stats: Tuple[np.ndarray, np.ndarray],
        speed_pressure_coeff: float,
        efficiency_bounds: tuple = (0.7, 0.9),
        device: str = 'cpu',
        **kwargs,
    ):
        super().__init__(
            efficiency_bounds=efficiency_bounds,
            speed_pressure_coeff=speed_pressure_coeff,
            device=device,
            **kwargs,
        )
        self.norm_stats = norm_stats

    def __call__(
        self,
        generated_trajectory: torch.Tensor,
        condition: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        x_phys = denormalize(generated_trajectory, self.norm_stats)
        return super().__call__(x_phys, condition)


class CompressorPhysicsNorm(CompressorPhysics):
    """
    策略2: 为归一化空间 [-1,1] 设计的新约束形式。
    - 速度-压力: 使用归一化空间的线性残差 (P30_norm 与 Nc_norm^2 的正相关)
    - 效率: eta_norm = (T30-T24)/(|P30|+0.5)，边界 [-1.5, 1.5]
    - 单调性: 与原版相同（尺度不变性）
    """

    def __init__(
        self,
        speed_pressure_coeff_norm: Optional[float] = None,
        eta_norm_bounds: tuple = (-1.5, 1.5),
        device: str = 'cpu',
        **kwargs,
    ):
        super().__init__(device=device, **kwargs)
        self.speed_pressure_coeff_norm = speed_pressure_coeff_norm
        self.eta_min_norm, self.eta_max_norm = eta_norm_bounds

    def fit_coeff_norm(self, data_norm: torch.Tensor) -> float:
        """从归一化数据拟合 P30_norm ~ c * (Nc_norm+1)^2"""
        if data_norm.dim() == 2:
            data_norm = data_norm.unsqueeze(0)
        P30 = data_norm[:, :, 2].reshape(-1)
        Nc = data_norm[:, :, 3].reshape(-1)
        x = ((Nc + 1) / 2) ** 2  # map to [0,1]
        c = (P30 * x).sum() / (x * x).sum().clamp(min=1e-8)
        return c.item()

    def speed_pressure_loss_norm(self, P30: torch.Tensor, Nc: torch.Tensor) -> torch.Tensor:
        """归一化空间: P30_norm 与 (Nc_norm+1)^2 正相关"""
        c = self.speed_pressure_coeff_norm
        if c is None:
            return torch.tensor(0.0, device=self.device)
        x = ((Nc + 1) / 2) ** 2
        pred = c * x
        return torch.mean((P30 - pred) ** 2)

    def efficiency_loss_norm(self, T24: torch.Tensor, T30: torch.Tensor, P30: torch.Tensor) -> torch.Tensor:
        """eta_norm = (T30-T24)/(|P30|+0.5)，边界适应 [-1,1] 空间"""
        eta = (T30 - T24) / (torch.abs(P30) + 0.5)
        low = torch.clamp(self.eta_min_norm - eta, min=0.0)
        high = torch.clamp(eta - self.eta_max_norm, min=0.0)
        return torch.mean(low ** 2 + high ** 2)

    def __call__(
        self,
        generated_trajectory: torch.Tensor,
        condition: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        if generated_trajectory.dim() == 2:
            generated_trajectory = generated_trajectory.unsqueeze(0)
        batch, T, F = generated_trajectory.shape
        idx = self.sensor_indices
        T24 = generated_trajectory[:, :, idx['T24']]
        T30 = generated_trajectory[:, :, idx['T30']]
        P30 = generated_trajectory[:, :, idx['P30']]
        Nc = generated_trajectory[:, :, idx['Nc']]

        loss = torch.tensor(0.0, device=self.device, dtype=generated_trajectory.dtype)
        c = (condition or {}).get('speed_pressure_coeff_norm', self.speed_pressure_coeff_norm)
        if c is not None:
            loss = loss + self.speed_pressure_loss_norm(P30, Nc)
        loss = loss + self.efficiency_loss_norm(T24, T30, P30)
        mono_feat = generated_trajectory[:, :, self.monotonic_feature_idx]
        loss = loss + self.monotonicity_loss(mono_feat)
        return loss

    def violation_ratios(self, generated_trajectory: torch.Tensor) -> Dict[str, float]:
        """效率违反比例、单调性违反比例（方案 7.1 可汇报指标）。"""
        if generated_trajectory.dim() == 2:
            generated_trajectory = generated_trajectory.unsqueeze(0)
        idx = self.sensor_indices
        T24 = generated_trajectory[:, :, idx['T24']]
        T30 = generated_trajectory[:, :, idx['T30']]
        P30 = generated_trajectory[:, :, idx['P30']]
        eta = (T30 - T24) / (torch.abs(P30) + 0.5)
        out_low = (eta < self.eta_min_norm).float().sum().item()
        out_high = (eta > self.eta_max_norm).float().sum().item()
        total = eta.numel()
        efficiency_violation_ratio = (out_low + out_high) / total if total else 0.0
        mono_feat = generated_trajectory[:, :, self.monotonic_feature_idx]
        monotonicity_violation_ratio = _monotonicity_violation_ratio(mono_feat)
        return {
            'efficiency_violation_ratio': efficiency_violation_ratio,
            'monotonicity_violation_ratio': monotonicity_violation_ratio,
        }
