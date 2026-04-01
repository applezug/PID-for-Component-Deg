# Reproducibility map (reviewers)

This file links **manuscript claims** to **files and commands** in this repository. No personal or institutional paths are required.

## C-MAPSS (main experiments)

| Topic | Verify without raw data | Full rerun |
|--------|-------------------------|------------|
| Multi-dataset RUL (τ=0.65) | `artifacts/paper/multi_dataset_rul_report.json` | `experiments/run_multi_dataset.py` after placing data (`DATA.md`) |
| λ ablation FD001/FD002 | `artifacts/paper/lambda_comparison_FD001.json`, `lambda_comparison_FD002.json` | `scripts/run_lambda_comparison.py` per dataset |
| Physics consistency (Fig. 3 class) | `hpc_cmapss_metrics.json`, `trajectory_physics_comparison.json` | `experiments/run_hpc.py` + trajectory comparison scripts |
| Training curve (Fig. 2 class) | `hpc_cmapss_history.json` | Enable `training.diffusion_train: true` in YAML and rerun `run_hpc.py` |
| PICP summary (Fig. 5 class) | `picp_multi_dataset_summary.json` | `scripts/run_picp_multi_dataset.py` |

Figure regeneration (no C-MAPSS files):

```bash
python scripts/generate_figure2_training_validation_loss.py --no-default
python scripts/generate_figure3_physics_consistency.py --no-default
python scripts/generate_figure4_lambda_ablation.py --no-default
python scripts/generate_figure5_uncertainty.py --no-default
```

## IGBT (Conclusion / cross-domain)

The Conclusion stresses **portability** of the framework (swap dataset + physics module) and reports **~99.98% reduction in mean monotonicity loss** vs the baseline pathway, with **~20.9%** mean monotonicity violation ratio (soft constraint).

| Claim | Primary artifact | Regenerate |
|--------|------------------|------------|
| Headline IGBT numbers (Table 9 class) | `artifacts/paper/igbt/igbt_conclusion_metrics.json` | Train `igbt_lambda*_full` then `scripts/compute_igbt_monotonicity_metrics.py` → `results/igbt_lambda_monotonicity_summary.json` |
| λ effect on training difficulty | `artifacts/paper/igbt/igbt_lambda_ablation_training_summary.json` | `experiments/run_igbt.py --config config/igbt_lambda*_full.yaml --diffusion-train` |
| Narrative + file index | `artifacts/paper/igbt/README.md` | — |

**Important:** IGBT raw data are **not** included. Reviewers following the full path must download the NASA IGBT package and place it under `data/NASA IGBT/` as described in `DATA.md`.

## Parallel internal projects (clarification)

Historical parallel checkouts (multi-full, lambda-full, time-stamped clones) existed only to avoid overwriting `results/`. **This repository is the single public integration point**; the paper-aligned defaults are `config/hpc_cmapss_paper.yaml` and the JSON under `artifacts/paper/`.
