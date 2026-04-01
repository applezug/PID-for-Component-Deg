"""
Degradation models (from project 3): linear, exponential, power_law.
"""

import torch
import numpy as np
from typing import Dict, Optional
from abc import ABC, abstractmethod


class DegradationModel(ABC):
    def __init__(self, model_type: str = 'linear', params: Optional[Dict] = None, device: str = 'cpu'):
        self.model_type = model_type
        self.params = params or {}
        self.device = device

    @abstractmethod
    def forward(self, time_points: torch.Tensor, initial_value: Optional[torch.Tensor] = None) -> torch.Tensor:
        pass

    def evaluate_consistency(self, data_sequence: torch.Tensor, time_points: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if data_sequence.dim() == 2:
            data_sequence = data_sequence.unsqueeze(0)
            time_points = time_points.unsqueeze(0) if time_points.dim() == 1 else time_points
            squeeze_output = True
        else:
            squeeze_output = False
        batch_size, seq_len, n_features = data_sequence.shape
        consistency_scores = []
        for f in range(n_features):
            feature_data = data_sequence[:, :, f]
            feature_mask = mask[:, :, f] if mask is not None else torch.ones_like(feature_data, dtype=torch.bool)
            initial_values = []
            for b in range(batch_size):
                observed_indices = torch.where(feature_mask[b])[0]
                initial_values.append(feature_data[b, observed_indices[0]].item() if len(observed_indices) > 0 else 0.0)
            initial_values = torch.tensor(initial_values, device=self.device)
            predicted = self.forward(time_points, initial_value=initial_values)
            if mask is not None:
                mse = torch.mean((feature_data[feature_mask] - predicted[feature_mask]) ** 2)
            else:
                mse = torch.mean((feature_data - predicted) ** 2)
            consistency_scores.append(-mse)
        final_score = torch.mean(torch.stack(consistency_scores))
        return final_score.squeeze(0) if squeeze_output else final_score

    def fit_params(self, observed_data: torch.Tensor, time_points: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict:
        return self.params


class LinearDegradationModel(DegradationModel):
    def __init__(self, params: Optional[Dict] = None, device: str = 'cpu'):
        super().__init__('linear', {**{'slope': 0.01, 'intercept': 0.0}, **(params or {})}, device)

    def forward(self, time_points: torch.Tensor, initial_value: Optional[torch.Tensor] = None) -> torch.Tensor:
        s, i = self.params['slope'], self.params['intercept']
        if initial_value is not None:
            if time_points.dim() == 1:
                i = initial_value - s * time_points[0]
            else:
                i = initial_value.unsqueeze(-1) - s * time_points[:, 0:1]
        s = s if isinstance(s, torch.Tensor) else torch.tensor(s, device=self.device, dtype=time_points.dtype)
        i = i if isinstance(i, torch.Tensor) else torch.tensor(i, device=self.device, dtype=time_points.dtype)
        return s * time_points + i

    def fit_params(self, observed_data: torch.Tensor, time_points: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict:
        if observed_data.dim() == 2:
            observed_data, time_points = observed_data.unsqueeze(0), time_points.unsqueeze(0) if time_points.dim() == 1 else time_points
        sl, ic = [], []
        for f in range(observed_data.shape[-1]):
            fd = observed_data[0, :, f]
            fm = mask[0, :, f] if mask is not None else torch.ones_like(fd, dtype=torch.bool)
            ot, ov = time_points[0][fm], fd[fm]
            if len(ot) < 2:
                sl.append(self.params['slope']); ic.append(self.params['intercept'])
                continue
            n, st, sy = len(ot), torch.sum(ot), torch.sum(ov)
            den = n * torch.sum(ot ** 2) - st ** 2
            if abs(den.item()) < 1e-8:
                sl.append(self.params['slope']); ic.append(self.params['intercept'])
                continue
            a = (n * torch.sum(ot * ov) - st * sy) / den
            b = (sy - a * st) / n
            sl.append(a.item()); ic.append(b.item())
        self.params = {'slope': np.mean(sl) if sl else self.params['slope'], 'intercept': np.mean(ic) if ic else self.params['intercept']}
        return self.params


class ExponentialDegradationModel(DegradationModel):
    def __init__(self, params: Optional[Dict] = None, device: str = 'cpu'):
        super().__init__('exponential', {**{'decay_rate': 0.01, 'initial': 1.0}, **(params or {})}, device)

    def forward(self, time_points: torch.Tensor, initial_value: Optional[torch.Tensor] = None) -> torch.Tensor:
        lam = self.params['decay_rate']
        y0 = initial_value if initial_value is not None else self.params['initial']
        lam = lam if isinstance(lam, torch.Tensor) else torch.tensor(lam, device=self.device, dtype=time_points.dtype)
        y0 = y0 if isinstance(y0, torch.Tensor) else torch.tensor(y0, device=self.device, dtype=time_points.dtype)
        if time_points.dim() == 2 and y0.dim() == 1:
            y0 = y0.unsqueeze(-1)
        return y0 * torch.exp(-lam * time_points)

    def fit_params(self, observed_data: torch.Tensor, time_points: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict:
        default_decay, default_initial = 0.01, 1.0
        if observed_data.dim() == 2:
            observed_data = observed_data.unsqueeze(0)
            time_points = time_points.unsqueeze(0) if time_points.dim() == 1 else time_points
        dr, ini = [], []
        for f in range(observed_data.shape[-1]):
            fd = observed_data[0, :, f]
            fm = mask[0, :, f] if mask is not None else torch.ones_like(fd, dtype=torch.bool)
            ot, ov = time_points[0][fm], fd[fm]
            if len(ot) < 2:
                dr.append(default_decay); ini.append(default_initial)
                continue
            # 数据可能归一化到 [-1,1]，先线性缩放到正区间再取 log，避免 log(<=0)=NaN
            ov_min, ov_max = ov.min().item(), ov.max().item()
            span = (ov_max - ov_min) or 1.0
            ov_pos = (ov - ov_min) / span + 1e-6
            lv = torch.log(ov_pos)
            n, st, sl = len(ot), torch.sum(ot), torch.sum(lv)
            den = n * torch.sum(ot ** 2) - st ** 2
            if abs(den.item()) < 1e-8:
                dr.append(default_decay); ini.append(default_initial)
                continue
            lam = -(n * torch.sum(ot * lv) - st * sl) / den
            log_y0 = (sl + lam * st) / n
            lam_val = max(0.0, lam.item())
            try:
                y0_val = float(torch.exp(log_y0).item())
            except Exception:
                y0_val = np.nan
            if np.isfinite(lam_val) and np.isfinite(y0_val) and y0_val > 0:
                dr.append(lam_val); ini.append(y0_val)
            else:
                dr.append(default_decay); ini.append(default_initial)
        dr_finite = [x for x in dr if np.isfinite(x)]
        ini_finite = [x for x in ini if np.isfinite(x) and x > 0]
        decay = float(np.mean(dr_finite)) if dr_finite else default_decay
        initial = float(np.mean(ini_finite)) if ini_finite else default_initial
        if not (np.isfinite(decay) and np.isfinite(initial) and initial > 0):
            decay, initial = default_decay, default_initial
        self.params = {'decay_rate': max(0.0, decay), 'initial': max(1e-6, initial)}
        return self.params


class PowerLawDegradationModel(DegradationModel):
    def __init__(self, params: Optional[Dict] = None, device: str = 'cpu'):
        super().__init__('power_law', {**{'exponent': 0.5, 'initial': 1.0}, **(params or {})}, device)

    def forward(self, time_points: torch.Tensor, initial_value: Optional[torch.Tensor] = None) -> torch.Tensor:
        n = self.params['exponent']
        y0 = initial_value if initial_value is not None else self.params['initial']
        n = n if isinstance(n, torch.Tensor) else torch.tensor(n, device=self.device, dtype=time_points.dtype)
        y0 = y0 if isinstance(y0, torch.Tensor) else torch.tensor(y0, device=self.device, dtype=time_points.dtype)
        if time_points.dim() == 2 and y0.dim() == 1:
            y0 = y0.unsqueeze(-1)
        return y0 * (torch.clamp(time_points, min=1e-8) ** (-n))

    def fit_params(self, observed_data: torch.Tensor, time_points: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict:
        """log(y) = log(y0) - n*log(t) => 回归 log(y) ~ log(t) 得 n=-slope, y0=exp(intercept). 数据取正: t+eps, (y+1)/2+eps."""
        if observed_data.dim() == 2:
            observed_data = observed_data.unsqueeze(0)
            time_points = time_points.unsqueeze(0) if time_points.dim() == 1 else time_points
        exps, inis = [], []
        eps = 1e-8
        for f in range(observed_data.shape[-1]):
            fd = observed_data[0, :, f]
            fm = mask[0, :, f] if mask is not None else torch.ones_like(fd, dtype=torch.bool)
            ot = time_points[0][fm].float()
            ov = fd[fm].float()
            if len(ot) < 2:
                exps.append(self.params['exponent'])
                inis.append(self.params['initial'])
                continue
            t_pos = ot + 1.0
            y_pos = (ov + 1.0) / 2.0 + eps
            log_t = torch.log(t_pos)
            log_y = torch.log(y_pos)
            n_pt, st, sy = len(ot), torch.sum(log_t), torch.sum(log_y)
            den = n_pt * torch.sum(log_t ** 2) - st ** 2
            if abs(den.item()) < 1e-8:
                exps.append(self.params['exponent'])
                inis.append(self.params['initial'])
                continue
            b = (n_pt * torch.sum(log_t * log_y) - st * sy) / den
            a = (sy - b * st) / n_pt
            n_exp = -b.item()
            y0 = torch.exp(a).item()
            exps.append(max(0.01, min(2.0, n_exp)))
            inis.append(y0)
        ex_finite = [x for x in exps if np.isfinite(x)]
        ini_finite = [x for x in inis if np.isfinite(x) and x > 0]
        exp_val = float(np.mean(ex_finite)) if ex_finite else self.params['exponent']
        ini_val = float(np.mean(ini_finite)) if ini_finite else self.params['initial']
        if not (np.isfinite(exp_val) and np.isfinite(ini_val) and ini_val > 0):
            exp_val, ini_val = 0.5, 1.0
        self.params = {'exponent': max(0.01, min(2.0, exp_val)), 'initial': max(1e-6, ini_val)}
        return self.params
