"""
Mixed loss for physics-informed diffusion.
Design reference: project 2 CustomLossHC (data + physics + optional terms), weighted sum.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any


class PhysicsInformedDiffusionLoss(nn.Module):
    """
    Total loss = loss_diffusion + lambda_physics * loss_physics.
    Optional: loss_data (MSE to observed/ground truth) if labels available.
    """

    def __init__(
        self,
        physics_model: Optional[Any] = None,
        lambda_physics: float = 0.5,
        scaling_factors: Optional[list] = None,
    ):
        super().__init__()
        self.physics_model = physics_model
        self.lambda_physics = lambda_physics
        # scaling_factors: [w_diffusion, w_physics, w_data] for flexibility
        self.scaling_factors = scaling_factors or [1.0, 1.0, 1.0]

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        generated_trajectory: Optional[torch.Tensor] = None,
        condition: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        pred/target: diffusion loss terms (e.g. predicted noise vs noise).
        generated_trajectory: denoised or current estimate (T, features) for physics.
        """
        loss_diffusion = torch.mean((pred - target) ** 2)
        total = self.scaling_factors[0] * loss_diffusion

        if self.physics_model is not None and generated_trajectory is not None and self.lambda_physics > 0:
            loss_physics = self.physics_model(generated_trajectory, condition)
            total = total + self.lambda_physics * self.scaling_factors[1] * loss_physics

        return total
