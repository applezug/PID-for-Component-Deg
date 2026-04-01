"""
Run HPC (high-pressure compressor) experiment from config.
Usage: python experiments/run_hpc.py --config config/hpc_cmapss_paper.yaml

Flow: 1) Fit LinearDegradationModel from training data
      2) [可选] 扩散训练：前向扩散 + 噪声预测 + 物理损失联合优化
      3) MBD sampling for future trajectory imputation (mask last portion)
      4) Evaluate physics loss on generated trajectories (策略2: 归一化空间约束)
      5) Save trajectories, history, metrics
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import json
import argparse
import pickle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.diffusion import (
    MBDDegradationImputation,
    LinearDegradationModel,
    ExponentialDegradationModel,
    PowerLawDegradationModel,
    NoisePredictionNet,
    get_cosine_beta_schedule,
    forward_diffusion,
    build_encoder_from_noise_net,
)
from models.physics import CompressorPhysicsNorm, FanPhysicsNorm, TurbinePhysicsNorm, CombustorPhysicsNorm
from models.losses import PhysicsInformedDiffusionLoss
from datasets.cmapss_dataset import CMAPSSDataset
from utils.io_utils import load_yaml_config


def _average_degradation_params(param_list, model_type):
    """对多样本拟合的退化参数做平均（按模型类型聚合）；过滤 NaN/非有限值避免污染均值。"""
    if not param_list:
        return {}
    if model_type == 'linear':
        sl = [p['slope'] for p in param_list if np.isfinite(p.get('slope', np.nan))]
        ic = [p['intercept'] for p in param_list if np.isfinite(p.get('intercept', np.nan))]
        return {'slope': np.mean(sl) if sl else 0.01, 'intercept': np.mean(ic) if ic else 0.0}
    if model_type == 'exponential':
        dr = [p['decay_rate'] for p in param_list if np.isfinite(p.get('decay_rate', np.nan))]
        ini = [p['initial'] for p in param_list if np.isfinite(p.get('initial', np.nan)) and p.get('initial', 0) > 0]
        return {'decay_rate': max(0.0, float(np.mean(dr))) if dr else 0.01, 'initial': float(np.mean(ini)) if ini else 1.0}
    if model_type == 'power_law':
        ex = [p['exponent'] for p in param_list if np.isfinite(p.get('exponent', np.nan))]
        ini = [p['initial'] for p in param_list if np.isfinite(p.get('initial', np.nan)) and p.get('initial', 0) > 0]
        return {'exponent': float(np.mean(ex)) if ex else 0.5, 'initial': float(np.mean(ini)) if ini else 1.0}
    return param_list[0]


def fit_degradation_from_dataset(deg_model, dataset, n_samples=50, device='cpu', dataset_name=None):
    """从训练集采样子序列，拟合退化模型参数（支持 linear / exponential / power_law）。
    dataset_name 用于日志标识，便于消融并行时区分（如 FD001、FD002）。"""
    import copy
    param_list = []
    n = min(n_samples, len(dataset))
    indices = np.random.choice(len(dataset), n, replace=False)
    for idx in indices:
        x_t = torch.from_numpy(dataset.sequences[idx]).float().unsqueeze(0)
        t_t = torch.from_numpy(dataset.time_points[idx]).float().unsqueeze(0)
        m_t = torch.from_numpy(dataset.masks[idx]).unsqueeze(0)
        if dataset.normalize and dataset.norm_stats is not None:
            min_v, max_v = dataset.norm_stats
            x_t = 2.0 * (x_t - min_v) / (max_v - min_v + 1e-8) - 1.0
        deg_model.fit_params(x_t, t_t, m_t)
        param_list.append(copy.deepcopy(deg_model.params))
    deg_model.params = _average_degradation_params(param_list, deg_model.model_type)
    label = f"{deg_model.model_type}" + (f" @ {dataset_name}" if dataset_name else "")
    print(f"Fitted degradation ({label}): {deg_model.params}")


def fit_coeff_norm_for_strategy2(train_ds, n_samples=100) -> float:
    """
    策略2 专用：从归一化训练数据拟合 P30_norm ~ c * (Nc_norm+1)^2 的系数。
    用于 CompressorPhysicsNorm 的速度-压力约束。
    """
    indices = np.random.choice(len(train_ds), min(n_samples, len(train_ds)), replace=False)
    data_list = []
    for idx in indices:
        seq = train_ds.sequences[idx]
        if train_ds.normalize and train_ds.norm_stats is not None:
            min_v, max_v = train_ds.norm_stats
            seq_norm = 2.0 * (seq - min_v) / (max_v - min_v + 1e-8) - 1.0
        else:
            seq_norm = seq.astype(np.float32)
        data_list.append(seq_norm)
    data = np.vstack(data_list)
    P30 = data[:, 2].ravel()
    Nc = data[:, 3].ravel()
    x = ((Nc + 1) / 2) ** 2
    c = np.sum(P30 * x) / (np.sum(x * x) + 1e-8)
    return float(c)


def fit_fan_coeff_norm(train_ds, n_samples=100):
    """
    风扇组件：从归一化训练数据拟合 P2~(Nf+1)^2 与 T2~(Nf+1) 的系数。
    特征顺序 0=T2, 1=P2, 2=Nf。返回 (p2_nf2_coeff, t2_nf_coeff)。
    """
    indices = np.random.choice(len(train_ds), min(n_samples, len(train_ds)), replace=False)
    data_list = []
    for idx in indices:
        seq = train_ds.sequences[idx]
        if train_ds.normalize and train_ds.norm_stats is not None:
            min_v, max_v = train_ds.norm_stats
            seq_norm = 2.0 * (seq - min_v) / (max_v - min_v + 1e-8) - 1.0
        else:
            seq_norm = seq.astype(np.float32)
        data_list.append(seq_norm)
    data = np.vstack(data_list)
    T2, P2, Nf = data[:, 0].ravel(), data[:, 1].ravel(), data[:, 2].ravel()
    x_nf2 = ((Nf + 1) / 2) ** 2
    x_nf = (Nf + 1) / 2
    c1 = float(np.sum(P2 * x_nf2) / (np.sum(x_nf2 * x_nf2) + 1e-8))
    c2 = float(np.sum(T2 * x_nf) / (np.sum(x_nf * x_nf) + 1e-8))
    return c1, c2


def fit_turbine_coeff_norm(train_ds, n_samples=100):
    """
    高压涡轮组件：从归一化训练数据拟合 T50~(Nc+1)^2 与 W31~(T50+1) 的系数。
    特征顺序 0=T50, 1=P40, 2=W31, 3=W32, 4=Nc。返回 (t50_nc2_coeff, w31_t50_coeff)。
    """
    indices = np.random.choice(len(train_ds), min(n_samples, len(train_ds)), replace=False)
    data_list = []
    for idx in indices:
        seq = train_ds.sequences[idx]
        if train_ds.normalize and train_ds.norm_stats is not None:
            min_v, max_v = train_ds.norm_stats
            seq_norm = 2.0 * (seq - min_v) / (max_v - min_v + 1e-8) - 1.0
        else:
            seq_norm = seq.astype(np.float32)
        data_list.append(seq_norm)
    data = np.vstack(data_list)
    T50, W31, Nc = data[:, 0].ravel(), data[:, 2].ravel(), data[:, 4].ravel()
    x_nc2 = ((Nc + 1) / 2) ** 2
    x_t50 = (T50 + 1) / 2
    c1 = float(np.sum(T50 * x_nc2) / (np.sum(x_nc2 * x_nc2) + 1e-8))
    c2 = float(np.sum(W31 * x_t50) / (np.sum(x_t50 * x_t50) + 1e-8))
    return c1, c2


def fit_combustor_coeff_norm(train_ds, n_samples=100) -> float:
    """
    燃烧室组件：从归一化训练数据拟合 (T50-T30) ~ k*(farB+1) 的系数。
    特征顺序 0=T30, 1=T50, 2=farB, 3=Wf/Ps30。返回 k。
    """
    indices = np.random.choice(len(train_ds), min(n_samples, len(train_ds)), replace=False)
    data_list = []
    for idx in indices:
        seq = train_ds.sequences[idx]
        if train_ds.normalize and train_ds.norm_stats is not None:
            min_v, max_v = train_ds.norm_stats
            seq_norm = 2.0 * (seq - min_v) / (max_v - min_v + 1e-8) - 1.0
        else:
            seq_norm = seq.astype(np.float32)
        data_list.append(seq_norm)
    data = np.vstack(data_list)
    T30, T50, farB = data[:, 0].ravel(), data[:, 1].ravel(), data[:, 2].ravel()
    dT = T50 - T30
    x = (farB + 1) / 2
    k = float(np.sum(dT * x) / (np.sum(x * x) + 1e-8))
    return k


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/hpc_cmapss_paper.yaml')
    parser.add_argument('--diffusion-train', action='store_true', help='覆盖配置，执行扩散训练')
    parser.add_argument('--epochs', type=int, default=None, help='覆盖配置的 epochs（用于快速测试）')
    parser.add_argument('--trajectory-subdir', type=str, default=None, help='将轨迹保存到 save_dir/此子目录（用于基线 vs 扩散+PINN 对比）')
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)

    cfg = load_yaml_config(args.config)
    data_cfg = cfg.get('data', {})
    diff_cfg = cfg.get('diffusion', {})
    phys_cfg = cfg.get('physics', {})
    train_cfg = cfg.get('training', {}).copy()
    out_cfg = cfg.get('output', {})

    if args.diffusion_train:
        train_cfg['diffusion_train'] = True
    if args.epochs is not None:
        train_cfg['epochs'] = args.epochs

    device = train_cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    seed = train_cfg.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    save_dir = out_cfg.get('save_dir', 'results')
    if args.trajectory_subdir:
        trajectory_dir = os.path.join(save_dir, args.trajectory_subdir)
    else:
        trajectory_dir = save_dir
    exp_name = out_cfg.get('exp_name', 'hpc_cmapss')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(trajectory_dir, exist_ok=True)

    # Data
    try:
        train_ds = CMAPSSDataset(
            data_root=data_cfg.get('data_root', 'data/cmapss'),
            dataset=data_cfg.get('dataset', 'FD001'),
            seq_length=data_cfg.get('seq_length', 256),
            period='train',
            train_ratio=data_cfg.get('train_ratio', 0.7),
            val_ratio=data_cfg.get('val_ratio', 0.15),
            sensor_indices=data_cfg.get('sensor_indices', [2, 3, 7, 14]),
            seed=seed,
            normalize=data_cfg.get('normalize', True),
        )
        val_ds = CMAPSSDataset(
            data_root=data_cfg.get('data_root', 'data/cmapss'),
            dataset=data_cfg.get('dataset', 'FD001'),
            seq_length=data_cfg.get('seq_length', 256),
            period='val',
            train_ratio=data_cfg.get('train_ratio', 0.7),
            val_ratio=data_cfg.get('val_ratio', 0.15),
            sensor_indices=data_cfg.get('sensor_indices', [2, 3, 7, 14]),
            seed=seed,
            normalize=True,
            norm_stats=train_ds.global_norm_stats,
        )
    except FileNotFoundError as e:
        print('CMAPSS data not found. Put train_FD001.txt in data/cmapss/.', e)
        return

    seq_length = train_ds.seq_length
    feature_size = train_ds.feature_size

    # 1. Fit degradation model（支持 linear / exponential / power_law）
    deg_type = diff_cfg.get('degradation_model', 'linear').lower()
    if deg_type == 'exponential':
        deg_model = ExponentialDegradationModel(device=device)
    elif deg_type == 'power_law':
        deg_model = PowerLawDegradationModel(device=device)
    else:
        deg_model = LinearDegradationModel(device=device)
    fit_degradation_from_dataset(deg_model, train_ds, n_samples=min(50, len(train_ds)), device=device)

    # 2. Build MBD
    diffusion_model = MBDDegradationImputation(
        seq_length=seq_length,
        feature_size=feature_size,
        degradation_model=deg_model,
        timesteps=diff_cfg.get('timesteps', 200),
        sampling_timesteps=diff_cfg.get('sampling_timesteps', 50),
        Nsample=diff_cfg.get('Nsample', 512),
        temp_sample=diff_cfg.get('temp_sample', 0.1),
        beta_schedule=diff_cfg.get('beta_schedule', 'cosine'),
        device=device,
    )
    diffusion_model.eval()

    # 3. Physics model for evaluation（策略2: 归一化空间约束）
    physics_model = None
    if phys_cfg.get('enabled', True):
        if phys_cfg.get('fan'):
            # 风扇组件（方案 5.1）
            fan_cfg = phys_cfg.get('fan', {})
            c1, c2 = fit_fan_coeff_norm(train_ds)
            physics_model = FanPhysicsNorm(
                p2_nf2_coeff_norm=c1,
                t2_nf_coeff_norm=c2,
                monotonic_feature_idx=int(fan_cfg.get('monotonic_feature_idx', 0)),
                device=device,
            )
            print(f"Physics (FanPhysicsNorm): p2_nf2_coeff={c1:.6e}, t2_nf_coeff={c2:.6e}")
        elif phys_cfg.get('turbine'):
            # 高压涡轮组件（方案 5.2）
            turbine_cfg = phys_cfg.get('turbine', {})
            c1, c2 = fit_turbine_coeff_norm(train_ds)
            physics_model = TurbinePhysicsNorm(
                t50_nc2_coeff_norm=c1,
                w31_t50_coeff_norm=c2,
                monotonic_feature_idx=int(turbine_cfg.get('monotonic_feature_idx', 0)),
                device=device,
            )
            print(f"Physics (TurbinePhysicsNorm): t50_nc2_coeff={c1:.6e}, w31_t50_coeff={c2:.6e}")
        elif phys_cfg.get('combustor'):
            # 燃烧室组件（方案 5.3）
            combustor_cfg = phys_cfg.get('combustor', {})
            k = fit_combustor_coeff_norm(train_ds)
            eta_bounds = tuple(combustor_cfg.get('eta_norm_bounds', [-1.2, 1.2]))
            physics_model = CombustorPhysicsNorm(
                dT_farb_coeff_norm=k,
                eta_norm_bounds=eta_bounds,
                monotonic_feature_idx=int(combustor_cfg.get('monotonic_feature_idx', 3)),
                device=device,
            )
            print(f"Physics (CombustorPhysicsNorm): dT_farb_coeff={k:.6e}, eta_norm_bounds={eta_bounds}")
        else:
            comp_cfg = phys_cfg.get('compressor', {})
            coeff_norm = fit_coeff_norm_for_strategy2(train_ds)
            eta_bounds = tuple(comp_cfg.get('eta_norm_bounds', [-1.5, 1.5]))
            physics_model = CompressorPhysicsNorm(
                speed_pressure_coeff_norm=coeff_norm,
                eta_norm_bounds=eta_bounds,
                device=device,
            )
            print(f"Physics (策略2 CompressorPhysicsNorm): coeff_norm={coeff_norm:.6e}, eta_norm_bounds={eta_bounds}")

    # 3b. 扩散训练：前向扩散 + 噪声预测 + 物理损失联合优化
    history = {'train_loss': [], 'val_loss': [], 'physics_loss': []}
    if train_cfg.get('diffusion_train', False) and physics_model is not None:
        timesteps = diff_cfg.get('timesteps', 200)
        betas = get_cosine_beta_schedule(timesteps, device=device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        noise_net = NoisePredictionNet(
            seq_length=seq_length,
            feature_size=feature_size,
            hidden_dim=64,
            num_layers=3,
            timesteps=timesteps,
            device=device,
        ).to(device)
        criterion = PhysicsInformedDiffusionLoss(
            physics_model=physics_model,
            lambda_physics=phys_cfg.get('lambda_physics', 0.5),
        )
        optimizer = torch.optim.Adam(noise_net.parameters(), lr=train_cfg.get('learning_rate', 1e-4))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
        train_loader = DataLoader(train_ds, batch_size=train_cfg.get('batch_size', 32), shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=train_cfg.get('batch_size', 32), shuffle=False, num_workers=0)

        epochs = train_cfg.get('epochs', 200)
        patience = train_cfg.get('patience', 30)
        best_val, counter = float('inf'), 0

        for ep in range(epochs):
            noise_net.train()
            epoch_loss = 0.0
            for batch in train_loader:
                x, t_pts, m, _ = [b.to(device) for b in batch]
                B = x.shape[0]
                tau = torch.randint(0, timesteps, (B,), device=device)
                x_t, eps = forward_diffusion(x, tau, alphas_cumprod, device=device)
                eps_pred = noise_net(x_t, tau)
                alpha_bar = alphas_cumprod[tau].view(-1, 1, 1)
                x0_pred = (x_t - torch.sqrt(1 - alpha_bar + 1e-8) * eps_pred) / (torch.sqrt(alpha_bar) + 1e-8)
                x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

                loss = criterion(
                    pred=eps_pred,
                    target=eps,
                    generated_trajectory=x0_pred,
                    condition={'time_points': t_pts, 'mask': m},
                )
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(noise_net.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
            train_loss = epoch_loss / len(train_loader)
            history['train_loss'].append(train_loss)

            noise_net.eval()
            val_loss_sum = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    x, t_pts, m, _ = [b.to(device) for b in batch]
                    B = x.shape[0]
                    tau = torch.randint(0, timesteps, (B,), device=device)
                    x_t, eps = forward_diffusion(x, tau, alphas_cumprod, device=device)
                    eps_pred = noise_net(x_t, tau)
                    alpha_bar = alphas_cumprod[tau].view(-1, 1, 1)
                    x0_pred = (x_t - torch.sqrt(1 - alpha_bar + 1e-8) * eps_pred) / (torch.sqrt(alpha_bar) + 1e-8)
                    x0_pred = torch.clamp(x0_pred, -1.0, 1.0)
                    loss = criterion(pred=eps_pred, target=eps, generated_trajectory=x0_pred, condition={'time_points': t_pts, 'mask': m})
                    val_loss_sum += loss.item()
            val_loss = val_loss_sum / len(val_loader) if len(val_loader) > 0 else train_loss
            history['val_loss'].append(val_loss)
            scheduler.step(val_loss)

            if val_loss < best_val:
                best_val = val_loss
                counter = 0
                torch.save(noise_net.state_dict(), os.path.join(save_dir, f'{exp_name}_noise_net.pt'))
            else:
                counter += 1
                if counter >= patience:
                    print(f'[扩散训练] 早停 @ epoch {ep + 1}')
                    break
            if ep % 20 == 0:
                print(f'[扩散训练] Epoch {ep}: train={train_loss:.4f}, val={val_loss:.4f}')
        print(f'[扩散训练] 完成，最佳 val_loss={best_val:.4f}，模型已保存至 {exp_name}_noise_net.pt')

    # 4. Future trajectory generation: mask last mask_ratio, impute via MBD
    mask_ratio = 0.3
    n_gen = min(5, len(val_ds))
    metrics = {}
    violation_records = []

    for i in tqdm(range(n_gen), desc='Generating trajectories'):
        x, t, m, _ = val_ds[i]
        x_np = val_ds.sequences[i]
        t_np = val_ds.time_points[i]
        if train_ds.normalize and train_ds.norm_stats is not None:
            min_v, max_v = train_ds.norm_stats
            x_norm = 2.0 * (x_np - min_v) / (max_v - min_v + 1e-8) - 1.0
        else:
            x_norm = x_np.astype(np.float32)

        n_obs = int(seq_length * (1 - mask_ratio))
        mask = np.ones_like(x_norm, dtype=bool)
        mask[n_obs:, :] = False

        obs = torch.from_numpy(x_norm).float().to(device)
        t_t = torch.from_numpy(t_np).float().to(device)
        m_t = torch.from_numpy(mask).to(device)

        with torch.no_grad():
            imputed = diffusion_model.sample(obs, t_t, m_t, clip_denoised=True)

        imputed_np = imputed.cpu().numpy()
        future_part = imputed_np[n_obs:]
        truth_full = x_norm

        np.save(os.path.join(trajectory_dir, f'trajectory_{i}.npy'), truth_full)
        np.save(os.path.join(trajectory_dir, f'future_trajectory_{i}.npy'), future_part)
        np.save(os.path.join(trajectory_dir, f'imputed_trajectory_{i}.npy'), imputed_np)

        if physics_model is not None:
            imp_t = imputed.unsqueeze(0)
            ploss = physics_model(imp_t, condition={}).item()
            history['physics_loss'].append(ploss)
            vr = getattr(physics_model, 'violation_ratios', lambda _: {})(imp_t)
            if vr:
                violation_records.append(vr)

    if history['physics_loss']:
        metrics['physics_loss_mean'] = np.mean(history['physics_loss'])
        metrics['physics_loss_std'] = np.std(history['physics_loss'])
        print(f"Physics loss (generated): mean={metrics['physics_loss_mean']:.6f}, std={metrics['physics_loss_std']:.6f}")
    if violation_records:
        for key in violation_records[0].keys():
            metrics[key] = float(np.mean([r[key] for r in violation_records]))
        print(f"Violation ratios (generated): efficiency={metrics.get('efficiency_violation_ratio', 0):.4f}, monotonicity={metrics.get('monotonicity_violation_ratio', 0):.4f}")

    # 5. Save
    with open(os.path.join(save_dir, f'{exp_name}_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(save_dir, f'{exp_name}_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    dataset_name = data_cfg.get('dataset', 'FD001')
    torch.save({'degradation_params': deg_model.params}, os.path.join(save_dir, f'{exp_name}_degradation_{dataset_name}_{deg_type}.pt'))

    # 5.4 组件编码器与归一化参数（方案 5.4，供实验三使用）
    norm_path = os.path.join(save_dir, f'{exp_name}_norm.pkl')
    min_v, max_v = getattr(train_ds, 'global_norm_stats', None) or (train_ds.norm_stats if (train_ds.normalize and getattr(train_ds, 'norm_stats', None) is not None) else None)
    if min_v is not None and max_v is not None:
        with open(norm_path, 'wb') as f:
            pickle.dump({
                'min': np.asarray(min_v),
                'max': np.asarray(max_v),
                'feature_size': feature_size,
                'seq_length': seq_length,
            }, f)
        print(f'Norm params saved: {norm_path}')
    noise_net_path = os.path.join(save_dir, f'{exp_name}_noise_net.pt')
    if os.path.exists(noise_net_path):
        noise_net = NoisePredictionNet(
            seq_length=seq_length,
            feature_size=feature_size,
            hidden_dim=64,
            num_layers=3,
            timesteps=diff_cfg.get('timesteps', 200),
            device=device,
        ).to(device)
        noise_net.load_state_dict(torch.load(noise_net_path, map_location=device, weights_only=False))
        encoder = build_encoder_from_noise_net(
            noise_net, seq_length, feature_size, hidden_dim=64, output_dim=128, device=device,
        )
        encoder_path = os.path.join(save_dir, f'{exp_name}_encoder.pth')
        torch.save(encoder.state_dict(), encoder_path)
        print(f'Component encoder saved: {encoder_path} (128-dim health representation)')

    print(f'Done. Results saved to {save_dir}/')


if __name__ == '__main__':
    main()
