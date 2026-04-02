# Data (not shipped in this repository)

This package **does not** redistribute NASA or third-party datasets. Obtain them under their original terms. Repository copyright and redistribution boundaries: [`NOTICE.md`](NOTICE.md).

## NASA C-MAPSS (turbofan)

- Place extracted `train_FD00x.txt`, `test_FD00x.txt`, and `RUL_FD00x.txt` under:
  - `data/cmapss/FD001/`, `data/cmapss/FD002/`, …  
- Official documentation: NASA C-MAPSS Data Set Documentation (see e.g. NTRS citation in the paper reference list).

## NASA IGBT accelerated aging (cross-domain / Conclusion)

This dataset supports **§4.10 and the Conclusion** (portability of the MBD + physics framework). It is **required** only if reviewers rerun IGBT experiments; headline numbers for desk checks are in `artifacts/paper/igbt/`.

- Download from the NASA Prognostics Data Repository (IGBT accelerated aging dataset).
- Recommended layout: `data/NASA IGBT/` with subfolders as expected by `datasets/igbt_dataset.py` (see `config/igbt.yaml` and `config/igbt_lambda0_full.yaml`).
- After training, monotonicity aggregation: `python scripts/compute_igbt_monotonicity_metrics.py` (writes `results/igbt_lambda_monotonicity_summary.json`).

## Privacy

No personal or institutional filesystem paths are required. Use relative paths only (`data/...`).
