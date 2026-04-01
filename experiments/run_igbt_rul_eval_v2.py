"""
V2 IGBT RUL and uncertainty evaluation.

This script is intended for the improved IGBT experiments (e.g.,
igbt_lambda0_full.yaml, igbt_lambda01_full.yaml, igbt_lambda05_full.yaml),
where the training and evaluation share the same network structure and
exp_name prefix.

It reuses the main logic from experiments/run_igbt_rul_eval.py but makes the
following assumptions:

- Checkpoints are named:
    results/{exp_name}_degradation.pt
    results/{exp_name}_noise_net.pt
  where exp_name is given by --exp-name.
- IGTBDataset splits (train/val/test) are consistent with the configs used
  during training.
- True RUL labels for IGBT are not yet available; this script therefore
  focuses on RUL *sample distributions* and interval widths rather than RMSE.

Usage (example):

    python experiments/run_igbt_rul_eval_v2.py \\
        --config config/igbt_lambda05_full.yaml \\
        --exp-name igbt_lambda05_full \\
        --n-samples 20 \\
        --interval-quantile 0.9
"""

import os
import sys
import json
import argparse
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.igbt_dataset import IGTBDataset  # noqa: E402
from models.diffusion import (  # noqa: E402
    MBDDegradationImputation,
    LinearDegradationModel,
    NoisePredictionNet,
    get_cosine_beta_schedule,
    forward_diffusion,
)
from utils.io_utils import load_yaml_config  # noqa: E402


def compute_rul_from_trajectory(vce_norm: np.ndarray, threshold: float = 0.8) -> int:
    """
    Compute a simple, relative RUL from a normalized Vce trajectory.

    Parameters
    ----------
    vce_norm : np.ndarray
        Normalized Vce trajectory of shape (T,).
    threshold : float
        Failure threshold in normalized units.

    Returns
    -------
    int
        RUL in time steps, defined as max(0, T - 1 - t_fail), where t_fail is the
        first index such that vce_norm[t_fail] >= threshold. If no failure step is
        found, returns 0.
    """
    T = vce_norm.shape[0]
    idx = np.where(vce_norm >= threshold)[0]
    if idx.size == 0:
        return 0
    t_fail = int(idx[0])
    return max(0, T - 1 - t_fail)


def _load_trained_degradation_and_noise_net(
    root: str,
    cfg: Dict[str, Any],
    exp_name: str,
    device: torch.device,
) -> Tuple[LinearDegradationModel, NoisePredictionNet, MBDDegradationImputation]:
    """
    Helper to rebuild degradation model, noise net, and MBD sampler using the
    improved IGBT pipeline (v2).
    """
    data_cfg = cfg.get("data", {})
    diff_cfg = cfg.get("diffusion", {})

    seq_length = data_cfg.get("seq_length", 256)

    # Load degradation parameters (v2 checkpoint)
    save_dir = cfg.get("output", {}).get("save_dir", "results")
    if not os.path.isabs(save_dir):
        save_dir = os.path.normpath(os.path.join(root, save_dir))
    deg_ckpt = os.path.join(save_dir, f"{exp_name}_degradation.pt")
    if not os.path.exists(deg_ckpt):
        raise FileNotFoundError(f"Degradation checkpoint not found: {deg_ckpt}")
    ckpt = torch.load(deg_ckpt, map_location="cpu")
    deg_model = LinearDegradationModel(device=str(device))
    deg_model.params = ckpt.get("degradation_params", {})

    # Build diffusion sampler (feature_size will be updated after dataset is loaded)
    diffusion_model = MBDDegradationImputation(
        seq_length=seq_length,
        feature_size=1,  # placeholder
        degradation_model=deg_model,
        timesteps=diff_cfg.get("timesteps", 200),
        sampling_timesteps=diff_cfg.get("sampling_timesteps", 50),
        Nsample=diff_cfg.get("Nsample", 512),
        temp_sample=diff_cfg.get("temp_sample", 0.1),
        beta_schedule=diff_cfg.get("beta_schedule", "cosine"),
        device=str(device),
    )
    diffusion_model.eval()

    # Load trained noise net (v2 checkpoint)
    noise_ckpt = os.path.join(save_dir, f"{exp_name}_noise_net.pt")
    if not os.path.exists(noise_ckpt):
        raise FileNotFoundError(f"Noise network checkpoint not found: {noise_ckpt}")

    timesteps = diff_cfg.get("timesteps", 200)
    noise_net = NoisePredictionNet(
        seq_length=seq_length,
        feature_size=1,  # placeholder
        hidden_dim=64,
        num_layers=3,
        timesteps=timesteps,
        device=str(device),
    ).to(device)
    noise_net.load_state_dict(torch.load(noise_ckpt, map_location=device))
    noise_net.eval()

    return deg_model, noise_net, diffusion_model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/igbt_lambda05_full.yaml")
    parser.add_argument("--exp-name", type=str, default="igbt_lambda05_full")
    parser.add_argument("--n-samples", type=int, default=20)
    parser.add_argument("--interval-quantile", type=float, default=0.9)
    parser.add_argument("--threshold", type=float, default=0.8)
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)

    cfg = load_yaml_config(args.config)
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})
    out_cfg = cfg.get("output", {})

    device_str = train_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    seed = train_cfg.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Dataset (reuse the same split as training/validation)
    data_root_raw = data_cfg.get("data_root", "data/NASA IGBT")
    data_root = os.path.normpath(os.path.join(root, data_root_raw)) if not os.path.isabs(data_root_raw) else data_root_raw
    seq_length = data_cfg.get("seq_length", 256)
    train_ratio = data_cfg.get("train_ratio", 0.7)
    val_ratio = data_cfg.get("val_ratio", 0.15)
    csv_name = data_cfg.get("csv_name", "Turn On.csv")

    try:
        train_ds = IGTBDataset(
            data_root=data_root,
            seq_length=seq_length,
            period="train",
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            csv_name=csv_name,
            seed=seed,
            normalize=data_cfg.get("normalize", True),
        )
        test_ds = IGTBDataset(
            data_root=data_root,
            seq_length=seq_length,
            period="test",
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            csv_name=csv_name,
            seed=seed,
            normalize=True,
            norm_stats=train_ds.global_norm_stats,
        )
    except FileNotFoundError as e:
        print("IGBT data not found:", e)
        return 1

    print(f"[INFO] IGBT RUL eval v2: train={len(train_ds)}, test={len(test_ds)}, "
          f"seq_length={seq_length}, feature_size={train_ds.feature_size}")

    # Rebuild degradation, noise net, and diffusion sampler
    deg_model, noise_net, diffusion_model = _load_trained_degradation_and_noise_net(
        root=root,
        cfg=cfg,
        exp_name=args.exp_name,
        device=device,
    )

    # Update feature_size in sampler and noise net now that we know it
    feature_size = train_ds.feature_size
    diffusion_model.feature_size = feature_size  # type: ignore[assignment]
    noise_net.feature_size = feature_size        # type: ignore[attr-defined]

    # Beta schedule for forward diffusion (same as in training)
    timesteps = cfg.get("diffusion", {}).get("timesteps", 200)
    betas = get_cosine_beta_schedule(timesteps, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    n_samples = max(1, int(args.n_samples))
    q = float(args.interval_quantile)
    threshold = float(args.threshold)

    all_rul_samples = []  # list of arrays, one per test window

    for batch in tqdm(test_loader, desc=f"IGBT RUL/UQ v2 ({args.exp_name})"):
        x, t_pts, m, _ = batch
        x = x.to(device)           # (1, T, F)
        t_pts = t_pts.to(device)   # (1, T)
        m = m.to(device)           # (1, T, F)

        T_len = x.shape[1]
        rul_samples = []

        for _ in range(n_samples):
            with torch.no_grad():
                B = x.shape[0]
                tau = torch.randint(0, timesteps, (B,), device=device)
                x_t, eps = forward_diffusion(x, tau, alphas_cumprod, device=device)
                eps_pred = noise_net(x_t, tau)
                alpha_bar = alphas_cumprod[tau].view(-1, 1, 1)
                x0_pred = (x_t - torch.sqrt(1 - alpha_bar + 1e-8) * eps_pred) / (torch.sqrt(alpha_bar) + 1e-8)
                x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

                # Use predicted clean trajectory as input to MBD sampler
                mask = torch.ones_like(x0_pred, dtype=torch.bool)
                half = T_len // 2
                mask[:, half:, :] = False

                imputed = diffusion_model.sample(
                    x0_pred.squeeze(0),
                    t_pts.squeeze(0),
                    mask.squeeze(0),
                    clip_denoised=True,
                )

            imputed_np = imputed.detach().cpu().numpy()  # (T,F)
            vce_norm = imputed_np[:, 0]
            rul = compute_rul_from_trajectory(vce_norm, threshold=threshold)
            rul_samples.append(rul)

        rul_samples = np.asarray(rul_samples, dtype=float)
        all_rul_samples.append(rul_samples)

    if not all_rul_samples:
        print("[WARN] No RUL samples collected; check dataset and config.")
        return 1

    all_rul_samples_arr = np.stack(all_rul_samples, axis=0)  # (N_test, n_samples)
    N_test = all_rul_samples_arr.shape[0]

    lower_q = (1.0 - q) / 2.0
    upper_q = 1.0 - lower_q
    lower = np.quantile(all_rul_samples_arr, lower_q, axis=1)
    upper = np.quantile(all_rul_samples_arr, upper_q, axis=1)
    widths = upper - lower

    uq_report = {
        "exp_name": args.exp_name,
        "n_test": int(N_test),
        "n_samples": int(n_samples),
        "interval_quantile": q,
        "mean_interval_width": float(widths.mean()),
        "std_interval_width": float(widths.std()),
        "note": (
            "True RUL labels for IGBT are not yet defined in the project; "
            "this report summarizes prediction-interval widths only."
        ),
    }

    save_dir = out_cfg.get("save_dir", "results")
    if not os.path.isabs(save_dir):
        save_dir = os.path.normpath(os.path.join(root, save_dir))
    os.makedirs(save_dir, exist_ok=True)

    out_path = os.path.join(save_dir, f"{args.exp_name}_rul_uq_igbt.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(uq_report, f, indent=2, ensure_ascii=False)

    print("[INFO] IGBT RUL/UQ v2 evaluation finished.")
    print(json.dumps(uq_report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

