"""
High-pressure compressor (HPC) physics constraints for CMAPSS.
Ref: doc 18/20 - speed-pressure relation, efficiency, monotonicity.
"""

import torch
from typing import Dict, Any, Optional, List
from .base_physics import BasePhysics


class CompressorPhysics(BasePhysics):
    """
    HPC physics:
    - speed_pressure_loss(P30, Nc, coeff): P30 vs Nc^2
    - efficiency_loss(T24, T30, P30): eta in [eta_min, eta_max]
    - monotonicity_loss(degradation_feature): e.g. T30 residual monotonic increase
    """

    def __init__(
        self,
        sensor_indices: Optional[Dict[str, int]] = None,
        efficiency_bounds: tuple = (0.7, 0.9),
        speed_pressure_coeff: Optional[float] = None,
        monotonic_feature_idx: int = 1,
        device: str = 'cpu',
        **kwargs,
    ):
        super().__init__(device=device, **kwargs)
        # Default CMAPSS: T24=0, T30=1, P30=2, Nc=3 in selected 4-dim feature
        self.sensor_indices = sensor_indices or {'T24': 0, 'T30': 1, 'P30': 2, 'Nc': 3}
        self.eta_min, self.eta_max = efficiency_bounds
        self.speed_pressure_coeff = speed_pressure_coeff
        self.monotonic_feature_idx = monotonic_feature_idx

    def speed_pressure_loss(self, P30: torch.Tensor, Nc: torch.Tensor, coeff: Optional[float] = None) -> torch.Tensor:
        """P30 proportional to Nc^2 (or similar). coeff from healthy segment fit if None."""
        c = coeff if coeff is not None else self.speed_pressure_coeff
        if c is None:
            return torch.tensor(0.0, device=self.device)
        # Residual: P30 - c * Nc^2 (or normalized form)
        pred = c * (Nc ** 2)
        return torch.mean((P30 - pred) ** 2)

    def efficiency_loss(self, T24: torch.Tensor, T30: torch.Tensor, P30: torch.Tensor) -> torch.Tensor:
        """Ideal gas: eta ~ (T30 - T24) / (P30^((gamma-1)/gamma) - 1) or simplified. Keep eta in [eta_min, eta_max]."""
        # Simplified: eta_proxy = (T30 - T24) / (P30 + 1e-8); penalize out-of-range
        eta_proxy = (T30 - T24) / (P30 + 1e-8)
        low = torch.clamp(self.eta_min - eta_proxy, min=0.0)
        high = torch.clamp(eta_proxy - self.eta_max, min=0.0)
        return torch.mean(low ** 2 + high ** 2)

    def monotonicity_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Penalize when x[t+1] < x[t] (degradation feature should increase over time)."""
        if x.dim() == 2:
            diff = x[:, 1:] - x[:, :-1]
        else:
            diff = x[1:] - x[:-1]
        # penalize negative diff
        neg = torch.clamp(-diff, min=0.0)
        return torch.mean(neg ** 2)

    def __call__(
        self,
        generated_trajectory: torch.Tensor,
        condition: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        generated_trajectory: (batch, T, features) or (T, features).
        condition can contain time_points, coeff, etc.
        """
        if generated_trajectory.dim() == 2:
            generated_trajectory = generated_trajectory.unsqueeze(0)
        batch, T, F = generated_trajectory.shape

        idx = self.sensor_indices
        T24 = generated_trajectory[:, :, idx['T24']]
        T30 = generated_trajectory[:, :, idx['T30']]
        P30 = generated_trajectory[:, :, idx['P30']]
        Nc = generated_trajectory[:, :, idx['Nc']]

        loss = torch.tensor(0.0, device=self.device, dtype=generated_trajectory.dtype)
        coeff = (condition or {}).get('speed_pressure_coeff', self.speed_pressure_coeff)
        loss = loss + self.speed_pressure_loss(P30, Nc, coeff)
        loss = loss + self.efficiency_loss(T24, T30, P30)
        mono_feat = generated_trajectory[:, :, self.monotonic_feature_idx]
        loss = loss + self.monotonicity_loss(mono_feat)
        return loss
