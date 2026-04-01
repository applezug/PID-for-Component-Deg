"""
Run IGBT experiment (optional).
Usage: python experiments/run_igbt.py --config config/igbt.yaml

流程：1) 从 data/NASA IGBT 加载 Part 曲线（IGTBDataset）
     2) 拟合线性退化模型，构建 MBD 扩散
     3) 可选：扩散+物理损失联合训练（IGBTPhysics，当前为占位）
     4) 对验证集做轨迹补全，保存结果与指标
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.diffusion import (
    MBDDegradationImputation,
    LinearDegradationModel,
    NoisePredictionNet,
    get_cosine_beta_schedule,
    forward_diffusion,
)
from models.physics import IGBTPhysics
from models.losses import PhysicsInformedDiffusionLoss
from datasets.igbt_dataset import IGTBDataset
from utils.io_utils import load_yaml_config


def fit_degradation_from_dataset(deg_model, dataset, n_samples=50, device='cpu'):
    """从训练集采样子序列，拟合退化模型参数"""
    slopes, intercepts = [], []
    n = min(n_samples, len(dataset))
    indices = np.random.choice(len(dataset), n, replace=False)
    for idx in indices:
        x_t = torch.from_numpy(dataset.sequences[idx]).float().unsqueeze(0)
        t_t = torch.from_numpy(dataset.time_points[idx]).float().unsqueeze(0)
        m_t = torch.from_numpy(dataset.masks[idx]).unsqueeze(0)
        if dataset.normalize and dataset.norm_stats is not None:
            min_v, max_v = dataset.norm_stats
            min_v = torch.from_numpy(np.asarray(min_v)).float().to(x_t.device)
            max_v = torch.from_numpy(np.asarray(max_v)).float().to(x_t.device)
            x_t = 2.0 * (x_t - min_v) / (max_v - min_v + 1e-8) - 1.0
        deg_model.fit_params(x_t, t_t, m_t)
        slopes.append(deg_model.params['slope'])
        intercepts.append(deg_model.params['intercept'])
    deg_model.params['slope'] = np.mean(slopes)
    deg_model.params['intercept'] = np.mean(intercepts)
    print(f"Fitted degradation: slope={deg_model.params['slope']:.6f}, intercept={deg_model.params['intercept']:.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/igbt.yaml')
    parser.add_argument('--diffusion-train', action='store_true', help='执行扩散+物理联合训练')
    parser.add_argument('--epochs', type=int, default=None, help='覆盖配置的 epochs')
    parser.add_argument('--n-gen', type=int, default=None, help='轨迹生成条数，默认 min(5, len(val_ds))')
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
    if not os.path.isabs(save_dir):
        save_dir = os.path.normpath(os.path.join(root, save_dir))
    exp_name = out_cfg.get('exp_name', 'igbt')
    os.makedirs(save_dir, exist_ok=True)

    # 数据：NASA IGBT（相对路径转为基于项目根的绝对路径）
    data_root_raw = data_cfg.get('data_root', 'data/NASA IGBT')
    data_root = os.path.normpath(os.path.join(root, data_root_raw)) if not os.path.isabs(data_root_raw) else data_root_raw
    seq_length = data_cfg.get('seq_length', 256)
    train_ratio = data_cfg.get('train_ratio', 0.7)
    val_ratio = data_cfg.get('val_ratio', 0.15)
    csv_name = data_cfg.get('csv_name', 'Turn On.csv')

    try:
        train_ds = IGTBDataset(
            data_root=data_root,
            seq_length=seq_length,
            period='train',
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            csv_name=csv_name,
            seed=seed,
            normalize=data_cfg.get('normalize', True),
        )
        val_ds = IGTBDataset(
            data_root=data_root,
            seq_length=seq_length,
            period='val',
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            csv_name=csv_name,
            seed=seed,
            normalize=True,
            norm_stats=train_ds.global_norm_stats,
        )
    except FileNotFoundError as e:
        print('IGBT 数据未找到:', e)
        return 1

    print(f"IGBT 数据: train={len(train_ds)}, val={len(val_ds)}, seq_length={seq_length}, feature_size={train_ds.feature_size}")

    feature_size = train_ds.feature_size

    # 1. 拟合退化模型
    deg_model = LinearDegradationModel(device=device)
    fit_degradation_from_dataset(deg_model, train_ds, n_samples=min(50, len(train_ds)), device=device)

    # 2. MBD 扩散
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

    # 3. 物理模型（IGBT：当前为占位，损失为 0）
    physics_model = None
    if phys_cfg.get('enabled', True):
        igbt_cfg = phys_cfg.get('igbt', {})
        physics_model = IGBTPhysics(
            coffin_manson_exponent=float(igbt_cfg.get('coffin_manson_exponent', 2.0)),
            device=device,
        )
        print("Physics: IGBTPhysics (Coffin-Manson placeholder, loss=0 until formula implemented)")

    # 3b. 可选：扩散训练
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
            train_loss = epoch_loss / len(train_loader) if train_loader else 0.0
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
            val_loss = val_loss_sum / len(val_loader) if val_loader else train_loss
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
        print(f'[扩散训练] 完成，最佳 val_loss={best_val:.4f}')

    # 4. 轨迹生成（mask 后段，MBD 补全）
    mask_ratio = 0.3
    n_gen = args.n_gen if getattr(args, 'n_gen', None) is not None else min(5, len(val_ds))
    n_gen = max(1, min(n_gen, len(val_ds)))
    metrics = {}

    for i in tqdm(range(n_gen), desc='IGBT trajectory generation'):
        x_np = val_ds.sequences[i]
        t_np = val_ds.time_points[i]
        if train_ds.normalize and train_ds.norm_stats is not None:
            min_v, max_v = train_ds.norm_stats
            x_norm = 2.0 * (x_np - min_v) / (max_v - min_v + 1e-8) - 1.0
        else:
            x_norm = x_np.astype(np.float32)

        seq_length_actual = x_norm.shape[0]
        n_obs = int(seq_length_actual * (1 - mask_ratio))
        mask = np.ones_like(x_norm, dtype=bool)
        mask[n_obs:, :] = False

        obs = torch.from_numpy(x_norm).float().to(device)
        t_t = torch.from_numpy(t_np).float().to(device)
        m_t = torch.from_numpy(mask).to(device)

        with torch.no_grad():
            imputed = diffusion_model.sample(obs, t_t, m_t, clip_denoised=True)

        imputed_np = imputed.cpu().numpy()
        np.save(os.path.join(save_dir, f'igbt_trajectory_{i}.npy'), x_norm)
        np.save(os.path.join(save_dir, f'igbt_imputed_trajectory_{i}.npy'), imputed_np)

        if physics_model is not None:
            ploss = physics_model(imputed.unsqueeze(0), condition={}).item()
            history['physics_loss'].append(ploss)

    if history.get('physics_loss'):
        metrics['physics_loss_mean'] = float(np.mean(history['physics_loss']))
        metrics['physics_loss_std'] = float(np.std(history['physics_loss']))

    # 5. 保存
    with open(os.path.join(save_dir, f'{exp_name}_history.json'), 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    with open(os.path.join(save_dir, f'{exp_name}_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    torch.save({'degradation_params': deg_model.params}, os.path.join(save_dir, f'{exp_name}_degradation.pt'))

    print(f'Done. Results saved to {save_dir}/ (exp_name={exp_name})')
    return 0


if __name__ == '__main__':
    sys.exit(main())
