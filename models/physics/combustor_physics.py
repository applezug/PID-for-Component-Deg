"""
燃烧室组件物理约束（方案 5.3.2）。
归一化空间：温升 (T50-T30) ~ k*farB；燃烧效率代理在合理范围；Wf/Ps30 退化时单调上升。
轨迹特征顺序：T30=0, T50=1, farB=2, Wf/Ps30=3。
"""

import torch
from typing import Dict, Any, Optional
from .base_physics import BasePhysics, _monotonicity_violation_ratio


class CombustorPhysicsNorm(BasePhysics):
    """
    燃烧室物理约束（归一化空间 [-1,1]）：
    - 油气比-温升：(T50 - T30)_norm ~ k * (farB_norm+1)
    - 燃烧效率范围：eta_proxy = (T50-T30)/(|farB|+0.5) 约束在 [eta_min_norm, eta_max_norm]
    - 单调性：Wf/Ps30 随退化单调上升（monotonic_feature_idx=3）
    """

    def __init__(
        self,
        dT_farb_coeff_norm: Optional[float] = None,
        eta_norm_bounds: tuple = (-1.2, 1.2),
        monotonic_feature_idx: int = 3,
        device: str = 'cpu',
        **kwargs,
    ):
        super().__init__(device=device, **kwargs)
        self.dT_farb_coeff_norm = dT_farb_coeff_norm
        self.eta_min_norm, self.eta_max_norm = eta_norm_bounds
        self.monotonic_feature_idx = monotonic_feature_idx

    def dT_farb_loss(self, T30: torch.Tensor, T50: torch.Tensor, farB: torch.Tensor) -> torch.Tensor:
        """温升 (T50-T30) 与 (farB+1) 的线性关系"""
        if self.dT_farb_coeff_norm is None:
            return torch.tensor(0.0, device=self.device)
        dT = T50 - T30
        x = (farB + 1) / 2
        pred = self.dT_farb_coeff_norm * x
        return torch.mean((dT - pred) ** 2)

    def efficiency_loss_norm(self, T30: torch.Tensor, T50: torch.Tensor, farB: torch.Tensor) -> torch.Tensor:
        """效率代理 (T50-T30)/(|farB|+0.5) 约束在 [eta_min_norm, eta_max_norm]"""
        dT = T50 - T30
        denom = torch.abs(farB) + 0.5
        eta = dT / denom
        low = torch.clamp(self.eta_min_norm - eta, min=0.0)
        high = torch.clamp(eta - self.eta_max_norm, min=0.0)
        return torch.mean(low ** 2 + high ** 2)

    def monotonicity_loss(self, x: torch.Tensor) -> torch.Tensor:
        """退化特征单调上升：惩罚 x[t+1] < x[t]"""
        if x.dim() == 2:
            diff = x[:, 1:] - x[:, :-1]
        else:
            diff = x[1:] - x[:-1]
        neg = torch.clamp(-diff, min=0.0)
        return torch.mean(neg ** 2)

    def __call__(
        self,
        generated_trajectory: torch.Tensor,
        condition: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        if generated_trajectory.dim() == 2:
            generated_trajectory = generated_trajectory.unsqueeze(0)
        # 特征顺序: 0=T30, 1=T50, 2=farB, 3=Wf/Ps30
        T30 = generated_trajectory[:, :, 0]
        T50 = generated_trajectory[:, :, 1]
        farB = generated_trajectory[:, :, 2]

        loss = torch.tensor(0.0, device=self.device, dtype=generated_trajectory.dtype)
        loss = loss + self.dT_farb_loss(T30, T50, farB)
        loss = loss + self.efficiency_loss_norm(T30, T50, farB)
        mono_feat = generated_trajectory[:, :, self.monotonic_feature_idx]
        loss = loss + self.monotonicity_loss(mono_feat)
        return loss

    def violation_ratios(self, generated_trajectory: torch.Tensor) -> Dict[str, float]:
        """效率代理违反比例、单调性违反比例。"""
        if generated_trajectory.dim() == 2:
            generated_trajectory = generated_trajectory.unsqueeze(0)
        T30 = generated_trajectory[:, :, 0]
        T50 = generated_trajectory[:, :, 1]
        farB = generated_trajectory[:, :, 2]
        eta = (T50 - T30) / (torch.abs(farB) + 0.5)
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
