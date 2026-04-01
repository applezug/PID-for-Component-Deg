"""
Base class for physics constraints.
Design reference: project 2 (battery PINN) + project 4 (PI-TCN fusion strategies).
"""

import torch
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


def _monotonicity_violation_ratio(x: torch.Tensor) -> float:
    """(B,T) 或 (T,)：违反单调上升的比例（x[t+1] < x[t] 的步数 / 总步数）。"""
    if x.dim() == 1:
        x = x.unsqueeze(0)
    diff = x[:, 1:] - x[:, :-1]
    total = diff.numel()
    if total == 0:
        return 0.0
    violations = (diff < 0).sum().item()
    return violations / total


class BasePhysics(ABC):
    """
    Unified interface for physics constraints.
    __call__(generated_trajectory, condition) -> total_physics_loss (scalar tensor).
    violation_ratios(trajectory) -> dict for reporting (efficiency_violation_ratio, monotonicity_violation_ratio).
    """

    def __init__(self, device: str = 'cpu', **kwargs):
        self.device = device
        self.kwargs = kwargs

    def violation_ratios(self, generated_trajectory: torch.Tensor) -> Dict[str, float]:
        """可汇报的违反比例。子类可覆盖。默认返回空。"""
        return {}

    @abstractmethod
    def __call__(
        self,
        generated_trajectory: torch.Tensor,
        condition: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        Compute physics loss for generated trajectory.

        Args:
            generated_trajectory: (batch, T, features) or (T, features)
            condition: optional dict with time_points, sensor_names, etc.

        Returns:
            loss: scalar tensor
        """
        pass
