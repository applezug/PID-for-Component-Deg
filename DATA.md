# Data (not shipped in this repository)

This package **does not** redistribute NASA or third-party datasets. Obtain them under their original terms.

## NASA C-MAPSS (turbofan)

- Place extracted `train_FD00x.txt`, `test_FD00x.txt`, and `RUL_FD00x.txt` under:
  - `data/cmapss/FD001/`, `data/cmapss/FD002/`, …  
- Official documentation: NASA C-MAPSS Data Set Documentation (see e.g. NTRS citation in the paper reference list).

## NASA IGBT accelerated aging (optional cross-domain experiments)

- Download from the NASA Prognostics Data Repository (IGBT accelerated aging dataset).
- Recommended layout: `data/NASA IGBT/` with subfolders as expected by `datasets/igbt_dataset.py` (see `config/igbt.yaml`).

## Privacy

No personal or institutional filesystem paths are required. Use relative paths only (`data/...`).
