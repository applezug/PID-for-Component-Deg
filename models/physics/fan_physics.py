"""
风扇组件物理约束（方案 5.1.2）。
归一化空间：P2 ~ c1*(Nf+1)^2，T2 ~ c2*(Nf+1)；退化时 T2 单调上升。
轨迹特征顺序：T2=0, P2=1, Nf=2。
"""

import torch
from typing import Dict, Any, Optional
from .base_physics import BasePhysics, _monotonicity_violation_ratio


class FanPhysicsNorm(BasePhysics):
    """
    风扇物理约束（归一化空间 [-1,1]）：
    - 转速-压力：P2_norm ~ c1 * (Nf_norm+1)^2
    - 温度-转速：T2_norm ~ c2 * (Nf_norm+1)
    - 单调性：T2 随退化单调上升（monotonic_feature_idx=0）
    """

    def __init__(
        self,
        p2_nf2_coeff_norm: Optional[float] = None,
        t2_nf_coeff_norm: Optional[float] = None,
        monotonic_feature_idx: int = 0,
        device: str = 'cpu',
        **kwargs,
    ):
        super().__init__(device=device, **kwargs)
        self.p2_nf2_coeff_norm = p2_nf2_coeff_norm
        self.t2_nf_coeff_norm = t2_nf_coeff_norm
        self.monotonic_feature_idx = monotonic_feature_idx

    def p2_nf2_loss(self, P2: torch.Tensor, Nf: torch.Tensor) -> torch.Tensor:
        """P2_norm 与 (Nf_norm+1)^2 的线性关系"""
        if self.p2_nf2_coeff_norm is None:
            return torch.tensor(0.0, device=self.device)
        x = ((Nf + 1) / 2) ** 2
        pred = self.p2_nf2_coeff_norm * x
        return torch.mean((P2 - pred) ** 2)

    def t2_nf_loss(self, T2: torch.Tensor, Nf: torch.Tensor) -> torch.Tensor:
        """T2_norm 与 (Nf_norm+1) 的线性关系"""
        if self.t2_nf_coeff_norm is None:
            return torch.tensor(0.0, device=self.device)
        x = (Nf + 1) / 2
        pred = self.t2_nf_coeff_norm * x
        return torch.mean((T2 - pred) ** 2)

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
        # 特征顺序: 0=T2, 1=P2, 2=Nf
        T2 = generated_trajectory[:, :, 0]
        P2 = generated_trajectory[:, :, 1]
        Nf = generated_trajectory[:, :, 2]

        loss = torch.tensor(0.0, device=self.device, dtype=generated_trajectory.dtype)
        loss = loss + self.p2_nf2_loss(P2, Nf)
        loss = loss + self.t2_nf_loss(T2, Nf)
        mono_feat = generated_trajectory[:, :, self.monotonic_feature_idx]
        loss = loss + self.monotonicity_loss(mono_feat)
        return loss

    def violation_ratios(self, generated_trajectory: torch.Tensor) -> Dict[str, float]:
        """风扇无效率边界，仅汇报单调性违反比例。"""
        if generated_trajectory.dim() == 2:
            generated_trajectory = generated_trajectory.unsqueeze(0)
        mono_feat = generated_trajectory[:, :, self.monotonic_feature_idx]
        return {
            'efficiency_violation_ratio': 0.0,
            'monotonicity_violation_ratio': _monotonicity_violation_ratio(mono_feat),
        }
