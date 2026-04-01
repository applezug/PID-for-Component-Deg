"""
噪声预测网络，用于扩散模型训练。
输入: (B, T, F) 含噪序列 + 时间步 t；输出: 预测噪声 (B, T, F)。
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class NoisePredictionNet(nn.Module):
    """
    1D 时序噪声预测网络。
    输入: x_noisy (B,T,F), timestep (B,) 或标量
    输出: pred_noise (B,T,F)
    """

    def __init__(
        self,
        seq_length: int,
        feature_size: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        timesteps: int = 200,
        device: str = "cpu",
    ):
        super().__init__()
        self.seq_length = seq_length
        self.feature_size = feature_size
        self.timesteps = timesteps
        self.device = device

        # 时间步嵌入 (标量 -> hidden_dim)
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 输入: (B, T, F) -> 扩展时间嵌入为 (B, T, H)
        self.input_proj = nn.Linear(feature_size + hidden_dim, hidden_dim)

        convs = []
        for i in range(num_layers):
            convs.append(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
            )
            convs.append(nn.SiLU())
            convs.append(nn.GroupNorm(4, hidden_dim))
        self.convs = nn.Sequential(*convs)

        self.out_proj = nn.Linear(hidden_dim, feature_size)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        x: (B, T, F) 含噪序列
        t: (B,) 或标量，时间步索引 [0, timesteps-1]
        """
        B = x.shape[0]
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(B).to(x.device)
        t_norm = (t.float() / max(1, self.timesteps - 1)).unsqueeze(-1)  # (B, 1)
        time_emb = self.time_embed(t_norm)  # (B, hidden_dim)
        time_emb = time_emb.unsqueeze(1).expand(-1, self.seq_length, -1)  # (B, T, H)
        h = torch.cat([x, time_emb], dim=-1)  # (B, T, F+H)
        h = self.input_proj(h)  # (B, T, H)
        h = h.transpose(1, 2)  # (B, H, T)
        h = self.convs(h)  # (B, H, T)
        h = h.transpose(1, 2)  # (B, T, H)
        pred = self.out_proj(h)  # (B, T, F)
        return pred


def get_cosine_beta_schedule(timesteps: int, s: float = 0.008, device: str = "cpu") -> torch.Tensor:
    """与 MBD 一致的余弦 beta 调度。"""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64, device=device)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0, 0.999).float()


def forward_diffusion(
    x0: torch.Tensor,
    t: torch.Tensor,
    alphas_cumprod: torch.Tensor,
    device: str = "cpu",
) -> tuple:
    """
    前向扩散：x_t = sqrt(alpha_bar_t) * x0 + sqrt(1-alpha_bar_t) * eps
    x0: (B, T, F), t: (B,) 或标量
    返回: (x_t, eps)
    """
    B = x0.shape[0]
    if t.dim() == 0:
        t = t.unsqueeze(0).expand(B)
    t = t.long().clamp(0, alphas_cumprod.shape[0] - 1)
    alpha_bar = alphas_cumprod[t].view(-1, 1, 1).to(x0.device)
    eps = torch.randn_like(x0, device=x0.device, dtype=x0.dtype)
    x_t = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar + 1e-8) * eps
    return x_t, eps
