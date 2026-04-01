#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate Figure 5 (uncertainty quantification): (a) PICP across subsets, (b) mean interval width.

Data policy:
  - Prefer `artifacts/paper/picp_multi_dataset_summary.json`, then `results/picp_multi_dataset_summary.json`.
  - Optional fall back: `uncertainty_metrics.json` if you add one under results/.

Outputs (in this repo):
  - figures/figure5_uncertainty_YYYYMMDD_NN.png/.pdf
  - figures/figure5_uncertainty.png/.pdf (default; can be disabled via --no-default)
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"
FIG5_BASE = "figure5_uncertainty"
ARTIFACTS_PAPER = ROOT / "artifacts" / "paper"


def next_versioned_png(ext: str = "png") -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    tag = datetime.now().strftime("%Y%m%d")
    i = 1
    while True:
        p = FIG_DIR / f"{FIG5_BASE}_{tag}_{i:02d}.{ext}"
        if not p.exists():
            return p
        i += 1


def find_first_existing(paths: list[Path]) -> Path | None:
    for p in paths:
        if p.is_file():
            return p
    return None


def load_picp_multi_summary(path: Path) -> tuple[list[float], list[float], list[str]]:
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)

    labels = ["FD002", "FD003", "FD004", "Aggregate"]
    picp = [0.0, 0.0, 0.0, 0.0]
    widths = [0.0, 0.0, 0.0, 0.0]

    datasets = d.get("datasets", {})
    for i, key in enumerate(["FD002", "FD003", "FD004"]):
        if key in datasets:
            picp[i] = float(datasets[key].get("picp", 0.0))
            widths[i] = float(datasets[key].get("mean_interval_width", 0.0))

    agg = d.get("aggregate", {})
    picp[3] = float(agg.get("picp_overall", agg.get("picp", 0.0)))
    widths[3] = float(agg.get("mean_interval_width_avg", 0.0))
    return picp, widths, labels


def main() -> None:
    p = argparse.ArgumentParser(description="Export Figure 5 (uncertainty).")
    p.add_argument("--no-default", action="store_true", help="Do not overwrite figures/figure5_uncertainty.png/.pdf.")
    p.add_argument("--output-png", type=str, default=None, help="Exact PNG output path (override auto).")
    args = p.parse_args()

    # Prefer the same exported summary.
    picp_summary = find_first_existing(
        [
            ARTIFACTS_PAPER / "picp_multi_dataset_summary.json",
            RESULTS_DIR / "picp_multi_dataset_summary.json",
        ]
    )
    if picp_summary is None:
        raise SystemExit(
            "No picp_multi_dataset_summary.json (place under artifacts/paper/ or run PICP export to results/)."
        )

    picp, widths, labels = load_picp_multi_summary(picp_summary)

    # ---- Plot (aligned with make_results_figures.make_figure5 styling) ----
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(8, 4))
    x = np.arange(len(labels))
    width_bar = 0.6

    # (a) PICP
    ax_a.set_title("(a) PICP (prediction interval coverage)")
    ax_a.set_ylabel("Prediction interval coverage probability (PICP)")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(labels)
    # 展示名义覆盖率参考线（0.9），避免 PICP=0 时参考线不可见。
    ax_a.set_ylim(0, 1.0)
    ax_a.axhline(0.9, color="gray", linestyle="--", linewidth=1, label="Nominal coverage (90%)")
    ax_a.scatter(x, picp, color="#1E88E5", s=60, zorder=3)
    for xi, yi in zip(x, picp):
        ax_a.text(xi, yi + 0.02, f"{yi:.1f}", ha="center", va="bottom", fontsize=9)
    ax_a.grid(axis="y", alpha=0.3)
    ax_a.legend(loc="upper right", fontsize=8)

    # (b) Interval width
    ax_b.set_title("(b) Interval width")
    ax_b.set_ylabel("Mean interval width (normalised to max_rul=125 cycles)")
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(labels)
    ax_b.set_ylim(0, 0.8)
    bars = ax_b.bar(x, widths, width_bar, color="#FF9800", edgecolor="black", linewidth=0.8)
    for xi, w in zip(x, widths):
        ax_b.text(xi, w + 0.02, f"{w:.2f}", ha="center", va="bottom", fontsize=9)
    ax_b.grid(axis="y", alpha=0.3)

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    norm_note = "Mean interval widths are normalised to the RUL range (max_rul=125 cycles)."
    if all(abs(v) < 1e-12 for v in picp):
        foot = f"*All PICP values are zero; intervals are under-dispersed. {norm_note} See Section 4.6 for discussion.*"
    else:
        foot = f"*PICP and interval widths are computed from prediction intervals. {norm_note} (see Section 4.6).*"
    fig.text(0.5, 0.02, foot, ha="center", va="bottom", fontsize=7, style="italic")

    # ---- Save ----
    versioned_png = Path(args.output_png).resolve() if args.output_png else next_versioned_png("png")
    versioned_pdf = Path(str(versioned_png).replace(".png", ".pdf"))
    fig.savefig(versioned_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(versioned_pdf, dpi=300, bbox_inches="tight", facecolor="white")
    print("Data source:", picp_summary)
    print("Saved (versioned PNG):", versioned_png)
    print("Saved (versioned PDF):", versioned_pdf)

    if not args.no_default:
        d_png = FIG_DIR / f"{FIG5_BASE}.png"
        d_pdf = FIG_DIR / f"{FIG5_BASE}.pdf"
        fig.savefig(d_png, dpi=300, bbox_inches="tight", facecolor="white")
        fig.savefig(d_pdf, dpi=300, bbox_inches="tight", facecolor="white")
        print("Saved (default PNG):", d_png)
        print("Saved (default PDF):", d_pdf)

    plt.close(fig)


if __name__ == "__main__":
    main()

