# Paper reference artifacts (non-identifying)

These JSON files support **figure regeneration** and **numeric cross-checks** for the open-science package. They contain **no** names, affiliations, or machine paths.

| File | Role |
|------|------|
| `multi_dataset_rul_report.json` | C-MAPSS FD001–FD004 RUL summary (τ=0.65, linear MBD, λ=0); aligns with AIS / component-degradation manuscript Table 2 class results. |
| `lambda_comparison_FD001.json`, `lambda_comparison_FD002.json` | Full λ∈{0,0.1,0.5,1.0} RUL metrics for Figure 4 / Table 5 class reporting. |
| `hpc_cmapss_metrics.json` | Physics-consistency aggregates (violation ratios, mean physics loss). |
| `trajectory_physics_comparison.json` | Per-trajectory physics losses for Figure 3 panel (a). |
| `hpc_cmapss_history.json` | Training/validation loss curve for Figure 2. |
| `picp_multi_dataset_summary.json` | PICP / interval-width summary for Figure 5 (FD002–FD004). |

To **recompute from code**, place NASA C-MAPSS files under `data/cmapss/` (see repository `DATA.md`), install dependencies, then run the commands in the root `README.md`.
