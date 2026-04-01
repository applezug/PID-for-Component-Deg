#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Figure 3: physics consistency (box/scatter + violation ratios).

Data policy (same idea as Figure 1 cross-project refresh):
  - trajectory_physics_comparison.json:
      prefer first available among:
      1) <repo>/results
      2) V1.3.1/results
      3) V1.3_lambdafull/results
  - hpc_cmapss_metrics.json:
      prefer first available among:
      1) V1.3.1/results
      2) V1.3_lambdafull/results
      3) <repo>/results

Outputs:
  - figures/figure3_physics_consistency_YYYYMMDD_NN.png
  - figures/figure3_physics_consistency.png (unless --no-default)
  - figures/figure3_physics_consistency_YYYYMMDD_NN.pdf
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
FIG3_BASE = "figure3_physics_consistency"
ARTIFACTS_PAPER = ROOT / "artifacts" / "paper"


def next_figure3_versioned(ext: str = "png") -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    tag = datetime.now().strftime("%Y%m%d")
    i = 1
    while True:
        p = FIG_DIR / f"{FIG3_BASE}_{tag}_{i:02d}.{ext}"
        if not p.exists():
            return p
        i += 1


def first_existing(candidates: list[Path]) -> Path | None:
    for p in candidates:
        if p.is_file():
            return p
    return None


def load_comparison(path: Path) -> tuple[list[float], list[float]]:
    with open(path, "r", encoding="utf-8") as f:
        comp = json.load(f)
    per = comp.get("per_traj", [])
    baseline = [float(p["baseline_physics_loss"]) for p in per]
    trained = [float(p["pinn_physics_loss"]) for p in per]
    if not baseline or not trained:
        raise ValueError(f"per_traj empty in {path}")
    return baseline, trained


def load_metrics(path: Path) -> tuple[float, float]:
    with open(path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    eff = float(metrics["efficiency_violation_ratio"]) * 100.0
    mono = float(metrics["monotonicity_violation_ratio"]) * 100.0
    return eff, mono


def plot_fig3(baseline_loss: list[float], trained_loss: list[float], eff_viol: float, mono_viol: float) -> plt.Figure:
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    np.random.seed(42)
    data_to_plot = [baseline_loss, trained_loss]
    ax1.boxplot(
        data_to_plot,
        positions=[1, 2],
        widths=0.5,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(facecolor="#E3F2FD", color="#1976D2"),
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(color="#1976D2"),
        capprops=dict(color="#1976D2"),
    )
    for i, data in enumerate(data_to_plot):
        x = np.random.normal(i + 1, 0.04, size=len(data))
        c = "#1976D2" if i == 0 else "#FF9800"
        lab = "Baseline" if i == 0 else "Trained"
        ax1.scatter(x, data, alpha=0.8, color=c, s=24, zorder=3, label=lab)
    ax1.set_xticks([1, 2])
    ax1.set_xticklabels(["Baseline (MBD only)", "After diffusion_train"])
    ax1.set_ylabel("Physics loss")
    ax1.set_title("(a) Physics loss comparison")
    ax1.set_ylim(0.20, 0.40)
    ax1.set_yticks([0.20, 0.25, 0.30, 0.35, 0.40])
    ax1.grid(axis="y", alpha=0.3)
    ax1.legend(loc="upper right", fontsize=8)

    categories = ["Efficiency", "Monotonicity"]
    values = [eff_viol, mono_viol]
    colors = ["#1E88E5", "#FB8C00"]
    bars = ax2.bar(categories, values, color=colors, edgecolor="black", linewidth=0.8, width=0.6)
    ax2.set_ylabel("Violation ratio (%)")
    ax2.set_title("(b) Violation ratios (trained model)")
    for bar, val in zip(bars, values):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax2.set_ylim(0, 50)
    ax2.grid(axis="y", alpha=0.3)
    ax2.text(
        0.5,
        -0.12,
        "*Baseline ratios not available; only trained model shown.*\n"
        "*Physics loss data from earlier trajectory evaluation; violation ratios from latest configuration (V1.3.1).*",
        transform=ax2.transAxes,
        ha="center",
        va="top",
        fontsize=7,
        style="italic",
    )

    plt.tight_layout()
    return fig


def main() -> None:
    p = argparse.ArgumentParser(description="Export Figure 3 (physics consistency).")
    p.add_argument("--comparison-json", type=str, default=None, help="Override trajectory_physics_comparison.json path.")
    p.add_argument("--metrics-json", type=str, default=None, help="Override hpc_cmapss_metrics.json path.")
    p.add_argument("--no-default", action="store_true", help="Do not overwrite figure3_physics_consistency.png/.pdf.")
    p.add_argument("--output-png", type=str, default=None, help="Exact PNG path (overrides auto versioned name).")
    args = p.parse_args()

    comp_path = (
        Path(args.comparison_json).resolve()
        if args.comparison_json
        else first_existing(
            [
                ARTIFACTS_PAPER / "trajectory_physics_comparison.json",
                RESULTS_DIR / "trajectory_physics_comparison.json",
            ]
        )
    )
    metrics_path = (
        Path(args.metrics_json).resolve()
        if args.metrics_json
        else first_existing(
            [
                ARTIFACTS_PAPER / "hpc_cmapss_metrics.json",
                RESULTS_DIR / "hpc_cmapss_metrics.json",
            ]
        )
    )

    if comp_path is None or not comp_path.is_file():
        raise SystemExit("No usable trajectory_physics_comparison.json (try artifacts/paper/ or results/).")
    if metrics_path is None or not metrics_path.is_file():
        raise SystemExit("No usable hpc_cmapss_metrics.json (try artifacts/paper/ or results/).")

    baseline_loss, trained_loss = load_comparison(comp_path)
    eff_viol, mono_viol = load_metrics(metrics_path)
    fig = plot_fig3(baseline_loss, trained_loss, eff_viol, mono_viol)

    versioned_png = Path(args.output_png).resolve() if args.output_png else next_figure3_versioned("png")
    versioned_pdf = Path(str(versioned_png).replace(".png", ".pdf"))
    fig.savefig(versioned_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(versioned_pdf, dpi=300, bbox_inches="tight", facecolor="white")
    print("Data source comparison:", comp_path)
    print("Data source metrics:", metrics_path)
    print("Saved (versioned PNG):", versioned_png)
    print("Saved (versioned PDF):", versioned_pdf)

    if not args.no_default:
        d_png = FIG_DIR / f"{FIG3_BASE}.png"
        d_pdf = FIG_DIR / f"{FIG3_BASE}.pdf"
        fig.savefig(d_png, dpi=300, bbox_inches="tight", facecolor="white")
        fig.savefig(d_pdf, dpi=300, bbox_inches="tight", facecolor="white")
        print("Saved (default PNG):", d_png)
        print("Saved (default PDF):", d_pdf)
    plt.close(fig)


if __name__ == "__main__":
    main()

