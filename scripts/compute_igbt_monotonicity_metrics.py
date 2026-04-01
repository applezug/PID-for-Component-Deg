"""
Compute monotonicity loss and violation ratios for improved IGBT experiments
with different physics weights (lambda_physics).

This script is a lightweight analysis tool for IGBT experiments with different
physics weights. It:

- Iterates over a set of exp_names (default: igbt_lambda0_full, igbt_lambda01_full,
  igbt_lambda05_full).
- For each exp_name, loads a small number of generated trajectories
  (imputed_trajectory_*.npy). If these files do not exist, the user should
  first generate them by re-running the corresponding run_igbt.py experiment
  with the same config.
- Uses the current IGBTPhysics implementation to compute:
  - monotonicity loss L_mono
  - monotonicity_violation_ratio
- Aggregates mean and std for each exp_name and writes a summary JSON to:
  results/igbt_lambda_monotonicity_summary.json

Usage (from repository root):

    python scripts/compute_igbt_monotonicity_metrics.py

By default it looks for trajectories named:
    results/{exp_name}_imputed_trajectory_*.npy

You can adjust N_TRAJ_PER_EXP below if you want to analyze more (or fewer)
trajectories per experiment.
"""

import os
import glob
import json
from typing import Dict, List

import numpy as np
import torch

from models.physics.igbt_physics import IGBTPhysics


EXP_NAMES = [
    "igbt_lambda0_full",
    "igbt_lambda01_full",
    "igbt_lambda05_full",
]

RESULTS_DIR = "results"
N_TRAJ_PER_EXP = 5  # maximum number of trajectories per exp_name to analyze


def compute_metrics_for_exp(exp_name: str, device: torch.device) -> Dict[str, float]:
    """
    For a given exp_name, load up to N_TRAJ_PER_EXP imputed trajectories from
    results/{exp_name}_imputed_trajectory_*.npy and compute:
      - mean / std of monotonicity loss
      - mean / std of monotonicity_violation_ratio
    using the current IGBTPhysics implementation.
    """
    pattern = os.path.join(RESULTS_DIR, f"{exp_name}_imputed_trajectory_*.npy")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"[WARN] No imputed trajectories found for {exp_name} (pattern: {pattern}). "
              f"Skip this experiment or generate trajectories first.")
        return {}

    files = files[:N_TRAJ_PER_EXP]
    print(f"[INFO] {exp_name}: found {len(files)} trajectories for monotonicity analysis.")

    physics = IGBTPhysics(device=str(device))
    physics.to(device)

    mono_losses: List[float] = []
    mono_viols: List[float] = []

    for fpath in files:
        arr = np.load(fpath)
        if arr.ndim == 2:
            # (T,F) -> (1,T,F)
            traj = torch.from_numpy(arr).float().unsqueeze(0).to(device)
        elif arr.ndim == 3:
            traj = torch.from_numpy(arr).float().to(device)
        else:
            print(f"[WARN] {exp_name}: unexpected trajectory shape {arr.shape} in {fpath}, skip.")
            continue

        with torch.no_grad():
            # Total physics loss is mono + optional terms; here we only want the mono component.
            # Recompute mono explicitly on the given trajectory.
            mono_loss = physics._monotonicity_loss(traj).item()  # type: ignore[attr-defined]
            ratios = physics.violation_ratios(traj)
            mono_violation = float(ratios.get("monotonicity_violation_ratio", 0.0))

        mono_losses.append(float(mono_loss))
        mono_viols.append(mono_violation)

    if not mono_losses:
        return {}

    mono_losses_arr = np.asarray(mono_losses, dtype=float)
    mono_viols_arr = np.asarray(mono_viols, dtype=float)

    return {
        "mono_loss_mean": float(mono_losses_arr.mean()),
        "mono_loss_std": float(mono_losses_arr.std()),
        "mono_violation_ratio_mean": float(mono_viols_arr.mean()),
        "mono_violation_ratio_std": float(mono_viols_arr.std()),
        "n_traj": int(len(mono_losses_arr)),
    }


def main() -> int:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    summary: Dict[str, Dict[str, float]] = {}
    for exp_name in EXP_NAMES:
        metrics = compute_metrics_for_exp(exp_name, device)
        if metrics:
            summary[exp_name] = metrics

    if not summary:
        print("[WARN] No metrics computed; summary is empty. "
              "Check that imputed trajectories exist in results/.")
        return 1

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "igbt_lambda_monotonicity_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Monotonicity summary written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

