"""
高压涡轮组件物理约束（方案 5.2.2）。
归一化空间：T50 ~ c1*(Nc+1)^2，W31 ~ c2*(T50+1)；退化时 T50 单调上升。
轨迹特征顺序：T50=0, P40=1, W31=2, W32=3, Nc=4。
"""

import torch
from typing import Dict, Any, Optional
from .base_physics import BasePhysics, _monotonicity_violation_ratio


class TurbinePhysicsNorm(BasePhysics):
    """
    高压涡轮物理约束（归一化空间 [-1,1]）：
    - 出口温度-转速：T50_norm ~ c1 * (Nc_norm+1)^2
    - 冷却流量-温度：W31_norm ~ c2 * (T50_norm+1)（正相关）
    - 单调性：T50 随退化单调上升（monotonic_feature_idx=0）
    """

    def __init__(
        self,
        t50_nc2_coeff_norm: Optional[float] = None,
        w31_t50_coeff_norm: Optional[float] = None,
        monotonic_feature_idx: int = 0,
        device: str = 'cpu',
        **kwargs,
    ):
        super().__init__(device=device, **kwargs)
        self.t50_nc2_coeff_norm = t50_nc2_coeff_norm
        self.w31_t50_coeff_norm = w31_t50_coeff_norm
        self.monotonic_feature_idx = monotonic_feature_idx

    def t50_nc2_loss(self, T50: torch.Tensor, Nc: torch.Tensor) -> torch.Tensor:
        """T50_norm 与 (Nc_norm+1)^2 的线性关系"""
        if self.t50_nc2_coeff_norm is None:
            return torch.tensor(0.0, device=self.device)
        x = ((Nc + 1) / 2) ** 2
        pred = self.t50_nc2_coeff_norm * x
        return torch.mean((T50 - pred) ** 2)

    def w31_t50_loss(self, W31: torch.Tensor, T50: torch.Tensor) -> torch.Tensor:
        """W31_norm 与 (T50_norm+1) 的线性关系（正相关）"""
        if self.w31_t50_coeff_norm is None:
            return torch.tensor(0.0, device=self.device)
        x = (T50 + 1) / 2
        pred = self.w31_t50_coeff_norm * x
        return torch.mean((W31 - pred) ** 2)

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
        # 特征顺序: 0=T50, 1=P40, 2=W31, 3=W32, 4=Nc
        T50 = generated_trajectory[:, :, 0]
        P40 = generated_trajectory[:, :, 1]
        W31 = generated_trajectory[:, :, 2]
        W32 = generated_trajectory[:, :, 3]
        Nc = generated_trajectory[:, :, 4]

        loss = torch.tensor(0.0, device=self.device, dtype=generated_trajectory.dtype)
        loss = loss + self.t50_nc2_loss(T50, Nc)
        loss = loss + self.w31_t50_loss(W31, T50)
        mono_feat = generated_trajectory[:, :, self.monotonic_feature_idx]
        loss = loss + self.monotonicity_loss(mono_feat)
        return loss

    def violation_ratios(self, generated_trajectory: torch.Tensor) -> Dict[str, float]:
        """涡轮无效率边界，仅汇报单调性违反比例。"""
        if generated_trajectory.dim() == 2:
            generated_trajectory = generated_trajectory.unsqueeze(0)
        mono_feat = generated_trajectory[:, :, self.monotonic_feature_idx]
        return {
            'efficiency_violation_ratio': 0.0,
            'monotonicity_violation_ratio': _monotonicity_violation_ratio(mono_feat),
        }
