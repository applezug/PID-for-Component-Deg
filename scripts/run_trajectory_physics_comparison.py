"""
Compare trajectory-level physics loss: baseline (no diffusion train) vs diffusion+PINN.
Expects imputed_trajectory_*.npy (or trajectory_*.npy) in --baseline-dir and --pinn-dir.
See docs/REMAINING_WORK.md Section 5.2.
"""
import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.cmapss_dataset import CMAPSSDataset
from models.physics import CompressorPhysicsNorm


def fit_coeff_norm(train_ds, n_samples=20) -> float:
    data = np.vstack([train_ds.sequences[i] for i in range(min(n_samples, len(train_ds)))])
    if train_ds.normalize and getattr(train_ds, "global_norm_stats", None):
        min_v, max_v = train_ds.global_norm_stats
        data_norm = 2.0 * (data - min_v) / (max_v - min_v + 1e-8) - 1.0
    else:
        data_norm = data
    P30 = data_norm[:, 2].ravel()
    Nc = data_norm[:, 3].ravel()
    x = ((Nc + 1) / 2) ** 2
    c = np.sum(P30 * x) / (np.sum(x * x) + 1e-8)
    return float(c)


def load_trajectories(dir_path: Path, pattern: str = "imputed_trajectory_*.npy"):
    files = sorted(dir_path.glob(pattern))
    if not files:
        files = sorted(dir_path.glob("trajectory_*.npy"))
    if not files:
        return []
    return [np.load(f).astype(np.float32) for f in files]


def main():
    parser = argparse.ArgumentParser(description="Baseline vs diffusion+PINN trajectory physics comparison")
    parser.add_argument("--baseline-dir", type=str, default="results/baseline_trajectories")
    parser.add_argument("--pinn-dir", type=str, default="results/diffusion_pinn_trajectories")
    parser.add_argument("--out", type=str, default="results/trajectory_physics_comparison.json")
    parser.add_argument("--data-root", type=str, default="data/cmapss")
    parser.add_argument("--dataset", type=str, default="FD001")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    os.chdir(root)

    baseline_dir = root / args.baseline_dir
    pinn_dir = root / args.pinn_dir
    out_path = root / args.out

    baseline_trajs = load_trajectories(baseline_dir)
    pinn_trajs = load_trajectories(pinn_dir)

    if not baseline_trajs or not pinn_trajs:
        msg = (
            f"缺少轨迹文件。baseline_dir={baseline_dir} 找到 {len(baseline_trajs)} 条，"
            f"pinn_dir={pinn_dir} 找到 {len(pinn_trajs)} 条。"
            "请先运行: (1) run_hpc --trajectory-subdir baseline_trajectories, "
            "(2) run_hpc --diffusion-train --trajectory-subdir diffusion_pinn_trajectories。"
        )
        print(msg)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"error": msg, "baseline_count": len(baseline_trajs), "pinn_count": len(pinn_trajs)}, f, indent=2, ensure_ascii=False)
        return

    n_common = min(len(baseline_trajs), len(pinn_trajs))
    baseline_trajs = baseline_trajs[:n_common]
    pinn_trajs = pinn_trajs[:n_common]

    train_ds = CMAPSSDataset(
        data_root=args.data_root,
        dataset=args.dataset,
        seq_length=256,
        period="train",
        train_ratio=0.7,
        val_ratio=0.15,
        seed=42,
        normalize=True,
    )
    coeff_norm = fit_coeff_norm(train_ds)
    physics = CompressorPhysicsNorm(
        speed_pressure_coeff_norm=coeff_norm,
        eta_norm_bounds=(-1.5, 1.5),
        device="cpu",
    )

    records = []
    for i in range(n_common):
        t_b = torch.from_numpy(baseline_trajs[i]).float().unsqueeze(0)
        t_p = torch.from_numpy(pinn_trajs[i]).float().unsqueeze(0)
        loss_b = physics(t_b, condition={}).item()
        loss_p = physics(t_p, condition={}).item()
        records.append({"traj_id": i, "baseline_physics_loss": loss_b, "pinn_physics_loss": loss_p})

    baseline_losses = [r["baseline_physics_loss"] for r in records]
    pinn_losses = [r["pinn_physics_loss"] for r in records]

    summary = {
        "baseline_mean": float(np.mean(baseline_losses)),
        "baseline_std": float(np.std(baseline_losses)),
        "pinn_mean": float(np.mean(pinn_losses)),
        "pinn_std": float(np.std(pinn_losses)),
        "n_trajectories": n_common,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_traj": records}, f, indent=2, ensure_ascii=False)

    print(f"Trajectory physics comparison saved to {out_path}")
    print(f"Baseline physics loss: mean={summary['baseline_mean']:.4f}, std={summary['baseline_std']:.4f}")
    print(f"Diffusion+PINN physics loss: mean={summary['pinn_mean']:.4f}, std={summary['pinn_std']:.4f}")


if __name__ == "__main__":
    main()
