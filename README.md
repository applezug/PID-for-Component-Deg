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
| `scripts/` | Figure export, λ sweep helper, PICP multi-dataset script |

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

## Citation

Cite the corresponding **MCA** and/or **AIS** article once published. Until DOIs are available, point reviewers to this repository URL and commit hash.

## License

MIT — see `LICENSE`.

---

**中文摘要**：本仓库为期刊公开查验用的**最小代码与匿名结果包**；不含个人信息与原始 C-MAPSS/IGBT 数据。论文级图表可仅用 `artifacts/paper/` 中的 JSON 复现；完整数值需自行下载 NASA 数据后按上文命令重跑。
