# IGBT cross-domain artifacts (reviewer pack)

This folder supports **verification of the cross-domain claim** in the manuscript **Conclusion** and **§4.10 (IGBT)**: the same MBD + physics-informed backbone applies after swapping the data loader and the physics module (compressor → IGBT).

## What is *not* here

- **Raw NASA IGBT traces** (download separately; see repository `DATA.md`).
- **Neural checkpoints** (`.pt` / `.pth`) — large and environment-specific; reviewers can regenerate with sufficient GPU time.

## What *is* here

| File | Use |
|------|-----|
| `igbt_conclusion_metrics.json` | **Primary numeric cross-check** for Conclusion / Table 9–class reporting: monotonicity loss `L_mono`, relative reduction vs baseline, violation ratios across λ∈{0,0.1,0.5} (full-physics IGBT runs). |
| `igbt_lambda_ablation_training_summary.json` | Training-side summary (best validation loss, early-stop epoch) for the three **full** λ configs, for consistency with §4.10 narrative on trade-offs. |
| `igbt_physics_metrics_sample.json` | Small **post-hoc** physics-loss aggregate from one local export (`physics_loss_mean` / `std` on generated trajectories); scale may differ from full-physics diagnostic block in the paper — use `igbt_conclusion_metrics.json` for conclusion-level numbers. |

## How reviewers can reproduce (full numerical path)

1. Obtain IGBT data under `data/NASA IGBT/` (layout in `DATA.md`).
2. Train each configuration (example for λ=0):

   ```bash
   python experiments/run_igbt.py --config config/igbt_lambda0_full.yaml --diffusion-train --epochs 200
   ```

   Repeat with `igbt_lambda01_full.yaml` and `igbt_lambda05_full.yaml`.
3. After imputed trajectories exist under `results/` (`*_imputed_trajectory_*.npy`), aggregate monotonicity statistics:

   ```bash
   python scripts/compute_igbt_monotonicity_metrics.py
   ```

   This writes `results/igbt_lambda_monotonicity_summary.json` (not shipped; regenerate locally).
4. Optional RUL / interval diagnostics (no ground-truth RUL on IGBT in the public recipe; manuscript describes distributional / UQ-style reporting):

   ```bash
   python experiments/run_igbt_rul_eval_v2.py --config config/igbt_lambda0_full.yaml --exp-name igbt_lambda0_full --n-samples 20 --interval-quantile 0.9
   ```

## Interpretation note (matches manuscript)

`L_mono` is a **soft** monotonicity penalty. **Low `L_mono` does not imply zero violation ratio**; the paper reports both (~99.98% reduction in mean `L_mono` vs the unconstrained baseline pathway, and ~20.9% mean violation ratio with substantial trajectory-to-trajectory variability). See `igbt_conclusion_metrics.json` → `interpretation`.
