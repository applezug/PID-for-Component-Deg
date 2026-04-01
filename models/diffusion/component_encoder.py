"""
组件健康表征编码器（方案 5.4）。
从训练好的 NoisePredictionNet 提取 backbone（time_embed=0 + input_proj + convs），
全局平均池化后接线性层，输出固定 128 维，供实验三加载使用。
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any


class ComponentEncoder(nn.Module):
    """
    输入: (B, T, F) 归一化传感器序列
    输出: (B, output_dim) 健康表征向量，默认 output_dim=128
    可与 NoisePredictionNet 共享 time_embed、input_proj、convs 权重（用 t=0 表示“清洁”观测）。
    """

    def __init__(
        self,
        seq_length: int,
        feature_size: int,
        hidden_dim: int = 64,
        output_dim: int = 128,
        timesteps: int = 200,
    ):
        super().__init__()
        self.seq_length = seq_length
        self.feature_size = feature_size
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.input_proj = nn.Linear(feature_size + hidden_dim, hidden_dim)
        convs = []
        for _ in range(3):
            convs.append(nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1))
            convs.append(nn.SiLU())
            convs.append(nn.GroupNorm(4, hidden_dim))
        self.convs = nn.Sequential(*convs)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self._timesteps = timesteps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, F) 归一化序列
        返回: (B, output_dim)
        """
        B = x.shape[0]
        t_norm = torch.zeros(B, 1, device=x.device, dtype=x.dtype)  # t=0 表示清洁
        time_emb = self.time_embed(t_norm)
        time_emb = time_emb.unsqueeze(1).expand(-1, self.seq_length, -1)
        h = torch.cat([x, time_emb], dim=-1)
        h = self.input_proj(h)
        h = h.transpose(1, 2)
        h = self.convs(h)
        h = self.pool(h).squeeze(-1)
        return self.fc(h)

    def load_from_noise_net(self, state_dict: Dict[str, Any]) -> None:
        """从 NoisePredictionNet 的 state_dict 复制 time_embed、input_proj、convs 权重。"""
        own = self.state_dict()
        for name, param in state_dict.items():
            if name in own and own[name].shape == param.shape:
                own[name].copy_(param)


def build_encoder_from_noise_net(
    noise_net: nn.Module,
    seq_length: int,
    feature_size: int,
    hidden_dim: int = 64,
    output_dim: int = 128,
    device: str = "cpu",
) -> ComponentEncoder:
    """从已训练的 NoisePredictionNet 构建 ComponentEncoder 并复制 backbone 权重。"""
    encoder = ComponentEncoder(
        seq_length=seq_length,
        feature_size=feature_size,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
    ).to(device)
    sd = noise_net.state_dict()
    encoder.load_from_noise_net(sd)
    return encoder
