# Physics-informed diffusion for component degradation & RUL (public code)

Minimal open-science release accompanying **MCA (*Processes*) / PGDP** and **AIS (component-level degradation)** submissions: model-based diffusion with compressor (or other) physics constraints, C-MAPSS RUL evaluation, optional IGBT cross-domain experiments.

**No personal data** are included. **`artifacts/paper/`** holds **anonymised JSON** so reviewers can **regenerate main figures** without running long GPU jobs.

## Repository layout

| Path | Purpose |
|------|---------|
| `experiments/` | `run_hpc.py`, `run_rul_eval.py`, `run_multi_dataset.py`, IGBT runners, ablations |
| `models/`, `datasets/`, `utils/` | Core implementation |
| `config/hpc_cmapss_paper.yaml` | Paper-aligned settings (τ=0.65, linear MBD, seed 42, `diffusion_train: false` for dt-off / fast RUL path) |
| `artifacts/paper/` | Reference metrics & summaries for figures / table checks |
| `artifacts/paper/igbt/` | **IGBT cross-domain** numbers for Conclusion / Table 9–class claims + reviewer workflow |
| `REPRODUCIBILITY.md` | Maps manuscript claims → artifacts → commands |
| `NOTICE.md` | Copyright, originality statement, data notice, academic citation |
| `scripts/` | Figure export, λ sweep helper, PICP multi-dataset script |

### IGBT (important for Conclusion)

The manuscript uses IGBT as **cross-domain evidence** (same backbone; replace data loader + physics module). Reviewers should read:

1. **`artifacts/paper/igbt/README.md`** — what is included, what is not, and the exact reproduction chain.  
2. **`artifacts/paper/igbt/igbt_conclusion_metrics.json`** — headline values: **~99.98%** mean `L_mono` reduction vs baseline pathway, **~20.9%** mean monotonicity violation ratio (soft constraint), stable across λ∈{0,0.1,0.5} in the reported full-physics runs.  
3. **`artifacts/paper/igbt/igbt_lambda_ablation_training_summary.json`** — training val-loss trade-off as λ increases.

Raw IGBT data and checkpoints are **not** redistributed; see **`DATA.md`**.

## Quick start (figures only, no C-MAPSS files needed)

```bash
pip install -r requirements.txt
python scripts/generate_figure2_training_validation_loss.py --no-default
python scripts/generate_figure3_physics_consistency.py --no-default
python scripts/generate_figure4_lambda_ablation.py --no-default
python scripts/generate_figure5_uncertainty.py --no-default
```

Outputs go to `figures/` (gitignored; regenerate after clone).

## Full pipeline (requires `data/cmapss/`)

See **`DATA.md`**. Then from the repo root:

```bash
# Single subset HPC + physics metrics (adjust dataset in YAML or override flags if supported)
python experiments/run_hpc.py --config config/hpc_cmapss_paper.yaml

# RUL on one subset (example: FD001, full test set)
python experiments/run_rul_eval.py --config config/hpc_cmapss_paper.yaml --dataset FD001 --n_test 0

# Four subsets (writes multi_dataset_rul_report.json when configured in run_multi_dataset)
python experiments/run_multi_dataset.py --config config/hpc_cmapss_paper.yaml --n_test 0
```

λ ablation (mutates `lambda_physics` in the chosen YAML; backs up to `*.bak_lambda`):

```bash
python scripts/run_lambda_comparison.py --config config/hpc_cmapss_paper.yaml --dataset FD001 --n_test 0
python scripts/run_lambda_comparison.py --config config/hpc_cmapss_paper.yaml --dataset FD002 --n_test 0
```

Enable **diffusion training** in YAML (`training.diffusion_train: true`) for the full training-time story; wall time increases substantially.

### IGBT pipeline (requires `data/NASA IGBT/`)

```bash
# Example: full-physics λ=0 run (long; GPU recommended)
python experiments/run_igbt.py --config config/igbt_lambda0_full.yaml --diffusion-train --epochs 200

# After imputed trajectories exist under results/
python scripts/compute_igbt_monotonicity_metrics.py

# Optional: interval / sample-based diagnostics (see experiments/run_igbt_rul_eval_v2.py --help)
python experiments/run_igbt_rul_eval_v2.py --config config/igbt_lambda0_full.yaml --exp-name igbt_lambda0_full --n-samples 20 --interval-quantile 0.9
```

## Citation

If you use this code for your research, please cite the corresponding article(s) and this archived version:

- **AIS paper** (under review): Q. Liu, Y. Di, S. Feng, X. Meng, Z. Chen, H. Cui, T. Wang, "Interpretable Physics-Informed Diffusion for Component-Level Degradation Modeling and RUL Prediction", *Advanced Intelligent Systems* (2026).  
- **MCA paper** (under review): Q. Liu, Y. Di, X. Meng, Z. Wang, Z. Xie, H. Cui, "Prior-Guided Diffusion Processes: A Unified Framework for Knowledge-Informed Generative Modeling", *Mathematical and Computational Applications* (2026).  

**Code archive (this version):** [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19438016.svg)](https://doi.org/10.5281/zenodo.19438016)  
For review, you may also refer to the repository URL and commit hash; the Zenodo DOI provides a permanent, citable snapshot. Full repository-level wording is in **`NOTICE.md`**.

## License and copyright

**MIT** — see [`LICENSE`](LICENSE). Copyright, originality, data boundaries, and citation guidance: see [`NOTICE.md`](NOTICE.md).

---

**中文摘要**：本仓库为期刊公开查验用的**最小代码与匿名结果包**；不含个人信息与原始 C-MAPSS/IGBT 数据。论文级图表可仅用 `artifacts/paper/` 中的 JSON 复现；完整数值需自行下载 NASA 数据后按上文命令重跑。**结论中 IGBT 跨域表述**请优先对照 `artifacts/paper/igbt/igbt_conclusion_metrics.json` 与 `igbt/README.md`，并按 `REPRODUCIBILITY.md` 映射至重现实验命令。
