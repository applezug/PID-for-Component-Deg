"""
IGBT physics constraints (monotonicity implemented, coupling/smoothness optional).
Design ref: Design-实验三-PINN-04-IGBT扩展实验 改进.md
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional
from .base_physics import BasePhysics, _monotonicity_violation_ratio


class IGBTPhysics(BasePhysics):
    """
    IGBT-specific physics:
    - Monotonicity: dVce/dt >= 0 (degradation indicator; implemented, always active).
    - Temperature–voltage coupling and smoothness: optional, only used when the
      required channels / condition entries are provided.

    The implementation is backwards-compatible with earlier versions that only
    used the monotonicity term: if no Tj channel or auxiliary information is
    available, the coupling and smoothness losses are skipped and only
    L_mono contributes to the total physics loss.
    """

    def __init__(
        self,
        coffin_manson_exponent: float = 2.0,
        vce_channel: int = 0,
        tj_channel: int = 1,
        use_coupling: bool = True,
        use_smoothness: bool = True,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(device=device, **kwargs)
        # Placeholder for possible Coffin–Manson-type extensions; currently unused.
        self.alpha = coffin_manson_exponent
        self.vce_channel = vce_channel
        self.tj_channel = tj_channel
        self.use_coupling = use_coupling
        self.use_smoothness = use_smoothness

    def _monotonicity_loss(self, trajectory: torch.Tensor) -> torch.Tensor:
        """Penalize negative time differences of Vce (or first channel). (B,T,F) -> scalar."""
        if trajectory.dim() == 2:
            trajectory = trajectory.unsqueeze(0)
        vce = trajectory[:, :, self.vce_channel]
        diff = vce[:, 1:] - vce[:, :-1]
        return F.relu(-diff).mean()

    def _coupling_loss(
        self,
        trajectory: torch.Tensor,
        condition: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        Temperature–voltage coupling loss.

        If a Tj trajectory is available (either as a separate channel or via
        condition['tj']), we fit a simple linear relation Vce ≈ α Tj + β on the
        first portion of the sequence (interpreted as "healthy" regime) and
        penalize deviations from this relation over the full window.

        If no suitable Tj is available, returns 0 so that the total loss falls
        back to L_mono only.
        """
        if not self.use_coupling:
            return torch.zeros((), device=self.device)

        if trajectory.dim() == 2:
            trajectory = trajectory.unsqueeze(0)

        B, T, Fdim = trajectory.shape

        # Try to get Tj from condition first (if provided).
        tj: Optional[torch.Tensor] = None
        if condition is not None and "tj" in condition:
            tj = condition["tj"].to(self.device)
            if tj.dim() == 2:  # (T,F) -> (1,T,F) or (T,) -> (1,T)
                tj = tj.unsqueeze(0)

        # Fallback: treat tj_channel as the Tj feature if it exists.
        if tj is None and self.tj_channel < Fdim:
            tj = trajectory[:, :, self.tj_channel]
        elif tj is None:
            # No Tj information available: skip this term.
            return torch.zeros((), device=self.device)

        vce = trajectory[:, :, self.vce_channel]
        if tj.dim() == 3 and tj.shape[-1] == 1:
            tj = tj.squeeze(-1)

        # Use the first 20% of the window as a "healthy" segment to fit α, β.
        healthy_T = max(1, int(0.2 * T))
        tj_h = tj[:, :healthy_T]
        vce_h = vce[:, :healthy_T]

        # Fit α, β in a least-squares sense for each batch element.
        # We keep the implementation simple; this is only used to define a soft prior.
        ones = torch.ones_like(tj_h)
        X = torch.stack([tj_h, ones], dim=-1)  # (B, Th, 2)
        # Solve normal equations X^T X theta = X^T y
        XtX = torch.matmul(X.transpose(1, 2), X)  # (B,2,2)
        Xty = torch.matmul(X.transpose(1, 2), vce_h.unsqueeze(-1))  # (B,2,1)
        # Add small ridge term for numerical stability
        ridge = 1e-6 * torch.eye(2, device=self.device).view(1, 2, 2)
        XtX = XtX + ridge
        theta = torch.linalg.solve(XtX, Xty)  # (B,2,1)
        alpha = theta[:, 0, 0].view(B, 1, 1)
        beta = theta[:, 1, 0].view(B, 1, 1)

        # Predicted Vce over the whole window
        vce_pred = alpha * tj + beta
        return ((vce - vce_pred) ** 2).mean()

    def _smoothness_loss(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        Smoothness loss on the degradation rate (second-order difference of Vce).
        Penalizes abrupt changes in the degradation slope.
        """
        if not self.use_smoothness:
            return torch.zeros((), device=self.device)

        if trajectory.dim() == 2:
            trajectory = trajectory.unsqueeze(0)
        vce = trajectory[:, :, self.vce_channel]
        # Second-order difference along time: Δ² Vce(t) = V(t+1) - 2V(t) + V(t-1)
        if vce.shape[1] < 3:
            return torch.zeros((), device=self.device)
        vce_prev = vce[:, :-2]
        vce_curr = vce[:, 1:-1]
        vce_next = vce[:, 2:]
        second_diff = vce_next - 2.0 * vce_curr + vce_prev
        return (second_diff ** 2).mean()

    def __call__(
        self,
        generated_trajectory: torch.Tensor,
        condition: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        Total physics loss for IGBT trajectories.

        By default, this includes:
          - L_mono: monotonicity loss on normalized Vce;
          - L_couple: optional temperature–voltage coupling loss (if Tj is available);
          - L_smooth: optional smoothness loss on the degradation rate.

        If no Tj or sufficient time steps are available, the corresponding terms
        gracefully return zero and the loss falls back to L_mono only, preserving
        compatibility with older experiments that only relied on monotonicity.
        """
        if generated_trajectory.dim() == 2:
            generated_trajectory = generated_trajectory.unsqueeze(0)
        generated_trajectory = generated_trajectory.to(self.device)

        l_mono = self._monotonicity_loss(generated_trajectory)
        l_couple = self._coupling_loss(generated_trajectory, condition=condition)
        l_smooth = self._smoothness_loss(generated_trajectory)

        return l_mono + l_couple + l_smooth

    def violation_ratios(self, generated_trajectory: torch.Tensor) -> Dict[str, float]:
        """
        For reporting: violation ratios of the implemented constraints.

        At present we always report the monotonicity violation ratio on the Vce
        channel. Future extensions may add ratios for the coupling and smoothness
        terms once clear threshold definitions are chosen.
        """
        if generated_trajectory.dim() == 2:
            generated_trajectory = generated_trajectory.unsqueeze(0)
        vce = generated_trajectory[:, :, self.vce_channel]
        return {
            "monotonicity_violation_ratio": _monotonicity_violation_ratio(vce)
        }

