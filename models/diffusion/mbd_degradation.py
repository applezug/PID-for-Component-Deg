"""
Model-Based Diffusion for Degradation Data Imputation

实现基于退化模型的扩散过程，用于退化数据插补。
（From project 3 - Degradation_Model_Based_Diffusion）
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Dict
from tqdm import tqdm
from .degradation_models import DegradationModel


class MBDDegradationImputation(nn.Module):
    """
    基于退化模型的扩散插补模型

    使用已知的退化模型（如线性、指数退化）来约束扩散过程，
    替代传统的神经网络分数估计。
    """

    def __init__(
        self,
        seq_length: int,
        feature_size: int,
        degradation_model: DegradationModel,
        timesteps: int = 1000,
        sampling_timesteps: Optional[int] = None,
        beta_schedule: str = 'cosine',
        beta_start: float = 1e-4,
        beta_end: float = 1e-2,
        Nsample: int = 2048,
        temp_sample: float = 0.1,
        device: str = 'cpu',
    ):
        """
        Args:
            seq_length: 序列长度
            feature_size: 特征维度
            degradation_model: 退化模型实例
            timesteps: 扩散时间步数
            sampling_timesteps: 采样时间步数（如果小于 timesteps，使用快速采样）
            beta_schedule: beta 调度方式 ('linear' 或 'cosine')
            beta_start: 初始 beta 值
            beta_end: 最终 beta 值
            Nsample: 每个时间步采样的候选序列数量
            temp_sample: 采样温度参数
            device: 计算设备
        """
        super().__init__()

        self.seq_length = seq_length
        self.feature_size = feature_size
        self.degradation_model = degradation_model
        self.Nsample = Nsample
        self.temp_sample = temp_sample
        self.device = device

        # 设置扩散参数
        if beta_schedule == 'linear':
            betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64, device=device)
        elif beta_schedule == 'cosine':
            betas = self._cosine_beta_schedule(timesteps, device=device)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), alphas_cumprod[:-1]])

        self.register_buffer('betas', betas.float())
        self.register_buffer('alphas', alphas.float())
        self.register_buffer('alphas_cumprod', alphas_cumprod.float())
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev.float())
        self.register_buffer('sigmas', torch.sqrt(1 - alphas_cumprod).float())

        self.num_timesteps = timesteps
        self.sampling_timesteps = sampling_timesteps or timesteps
        self.fast_sampling = self.sampling_timesteps < timesteps

    @staticmethod
    def _cosine_beta_schedule(timesteps: int, s: float = 0.008, device: str = 'cpu') -> torch.Tensor:
        """余弦 beta 调度"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, dtype=torch.float64, device=device)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, 0, 0.999)

    def _evaluate_degradation_sequences(
        self,
        candidate_sequences: torch.Tensor,
        observed_data: torch.Tensor,
        time_points: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        使用退化模型评估候选插补序列

        Args:
            candidate_sequences: 候选插补序列 (Nsample, T, features)
            observed_data: 观测到的数据（包含缺失值）(T, features)
            time_points: 时间点序列 (T,)
            mask: 缺失值掩码 (T, features)，True 表示观测值，False 表示缺失值

        Returns:
            consistency_scores: 一致性评分 (Nsample,)
        """
        Nsample = candidate_sequences.shape[0]
        consistency_scores = []

        for i in range(Nsample):
            complete_sequence = observed_data.clone()
            complete_sequence[~mask] = candidate_sequences[i][~mask]

            score = self.degradation_model.evaluate_consistency(
                complete_sequence.unsqueeze(0),
                time_points.unsqueeze(0),
                mask.unsqueeze(0)
            )
            consistency_scores.append(score)

        return torch.stack(consistency_scores)

    def _reverse_once(
        self,
        i: int,
        Xbar_i: torch.Tensor,
        observed_data: torch.Tensor,
        time_points: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        """执行单步 MBD 反向扩散（针对退化数据插补）"""
        alphas_bar_i = self.alphas_cumprod[i]
        sigmas_i = self.sigmas[i]

        Yi = torch.sqrt(alphas_bar_i) * Xbar_i

        eps = torch.randn(self.Nsample, self.seq_length, self.feature_size,
                          device=self.device, dtype=Xbar_i.dtype)
        X0s = eps * sigmas_i + Xbar_i
        X0s = torch.clamp(X0s, -1.0, 1.0)

        candidate_sequences = observed_data.unsqueeze(0).expand(self.Nsample, -1, -1).clone()
        candidate_sequences[~mask.unsqueeze(0).expand(self.Nsample, -1, -1)] = \
            X0s[~mask.unsqueeze(0).expand(self.Nsample, -1, -1)]

        consistency_scores = self._evaluate_degradation_sequences(
            candidate_sequences, observed_data, time_points, mask
        )

        score_std = consistency_scores.std()
        score_std = torch.clamp(score_std, min=1e-4)
        score_mean = consistency_scores.mean()
        logp0 = (consistency_scores - score_mean) / score_std / self.temp_sample

        weights = torch.softmax(logp0, dim=0)

        Xbar = torch.zeros_like(Xbar_i)
        for f in range(self.feature_size):
            feature_mask = mask[:, f]
            if not feature_mask.all():
                missing_mask = ~feature_mask
                Xbar[missing_mask, f] = torch.einsum('n,nm->m', weights, X0s[:, missing_mask, f])
            else:
                Xbar[:, f] = observed_data[:, f]

        alphas_i = self.alphas[i]
        score = 1.0 / (1.0 - alphas_bar_i) * (-Yi + torch.sqrt(alphas_bar_i) * Xbar)

        Xim1 = 1.0 / torch.sqrt(alphas_i) * (Yi + (1.0 - alphas_bar_i) * score)

        alphas_bar_im1 = self.alphas_cumprod[i - 1]
        Xbar_im1 = Xim1 / torch.sqrt(alphas_bar_im1)

        Xbar_im1[mask] = observed_data[mask]

        avg_consistency = consistency_scores.mean().item()

        return Xbar_im1, avg_consistency

    @torch.no_grad()
    def sample(
        self,
        observed_data: torch.Tensor,
        time_points: torch.Tensor,
        mask: torch.Tensor,
        clip_denoised: bool = True,
    ) -> torch.Tensor:
        """执行完整的 MBD 反向扩散过程，生成插补数据"""
        observed_data = observed_data.to(self.device)
        time_points = time_points.to(self.device)
        mask = mask.to(self.device)

        XN = torch.randn(self.seq_length, self.feature_size,
                        device=self.device, dtype=observed_data.dtype)

        Xbar_N = XN.clone()
        Xbar_N[mask] = observed_data[mask]

        if self.fast_sampling:
            times = torch.linspace(self.num_timesteps - 1, 0, self.sampling_timesteps + 1,
                                  device=self.device).long()
            times = list(reversed(times.tolist()))
            time_pairs = list(zip(times[:-1], times[1:]))

            Xbar = Xbar_N
            for time, time_next in tqdm(time_pairs, desc='MBD Sampling'):
                Xbar, _ = self._reverse_once(time, Xbar, observed_data, time_points, mask)
        else:
            Xbar = Xbar_N
            for i in tqdm(range(self.num_timesteps - 1, 0, -1), desc='MBD Sampling'):
                Xbar, _ = self._reverse_once(i, Xbar, observed_data, time_points, mask)

        imputed_data = Xbar.clone()
        imputed_data[mask] = observed_data[mask]

        if clip_denoised:
            imputed_data = torch.clamp(imputed_data, -1.0, 1.0)

        return imputed_data

    def fit_degradation_params(
        self,
        observed_data: torch.Tensor,
        time_points: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        """从观测数据拟合退化模型参数"""
        if mask is None:
            mask = torch.ones_like(observed_data, dtype=torch.bool)

        self.degradation_model.fit_params(observed_data, time_points, mask)
        print(f"Fitted degradation model parameters: {self.degradation_model.params}")
