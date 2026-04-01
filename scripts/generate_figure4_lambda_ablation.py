#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Figure 4: λ ablation (FD001 / FD002) from bundled lambda_comparison JSON files.

Default inputs (public repo, no local machine paths):
  artifacts/paper/lambda_comparison_FD001.json
  artifacts/paper/lambda_comparison_FD002.json

Outputs:
  figures/figure4_lambda_ablation_YYYYMMDD_NN.png/.pdf
  figures/figure4_lambda_ablation.png/.pdf (unless --no-default)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
FIG4_BASE = "figure4_lambda_ablation"
ARTIFACTS_PAPER = ROOT / "artifacts" / "paper"


@dataclass(frozen=True)
class Metric:
    rul_rmse: float
    rul_phm_score: float
    n_test: int
    failure_threshold: float | None = None
    dataset: str | None = None


def load_metric_dict(d: dict) -> Metric:
    return Metric(
        rul_rmse=float(d["rul_rmse"]),
        rul_phm_score=float(d["rul_phm_score"]),
        n_test=int(d.get("n_test", 0)),
        failure_threshold=d.get("failure_threshold", None),
        dataset=d.get("dataset", None),
    )


def load_lambda_file(path: Path) -> tuple[np.ndarray, list[Metric]]:
    with open(path, "r", encoding="utf-8") as f:
        blob = json.load(f)
    lambdas = [float(x) for x in blob["lambdas"]]
    metrics_map = blob["metrics"]
    metrics: list[Metric] = []
    for lam in lambdas:
        key = str(lam) if str(lam) in metrics_map else f"{lam:.1f}".rstrip("0").rstrip(".")
        if key not in metrics_map:
            key = f"{lam}"
        metrics.append(load_metric_dict(metrics_map[key]))
    return np.array(lambdas, dtype=float), metrics


def next_figure4_versioned(ext: str = "png") -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    tag = datetime.now().strftime("%Y%m%d")
    i = 1
    while True:
        p = FIG_DIR / f"{FIG4_BASE}_{tag}_{i:02d}.{ext}"
        if not p.exists():
            return p
        i += 1


def main() -> None:
    p = argparse.ArgumentParser(description="Export Figure 4 (lambda ablation).")
    p.add_argument("--fd001-json", type=str, default=None)
    p.add_argument("--fd002-json", type=str, default=None)
    p.add_argument("--no-default", action="store_true")
    p.add_argument("--output-png", type=str, default=None)
    args = p.parse_args()

    fd001_path = Path(args.fd001_json).resolve() if args.fd001_json else ARTIFACTS_PAPER / "lambda_comparison_FD001.json"
    fd002_path = Path(args.fd002_json).resolve() if args.fd002_json else ARTIFACTS_PAPER / "lambda_comparison_FD002.json"
    for path, label in [(fd001_path, "FD001"), (fd002_path, "FD002")]:
        if not path.is_file():
            raise SystemExit(f"Missing {label} lambda JSON: {path}")

    lam_fd001, m_fd001 = load_lambda_file(fd001_path)
    lam_fd002, m_fd002 = load_lambda_file(fd002_path)
    if not np.allclose(lam_fd001, lam_fd002):
        raise SystemExit("FD001 and FD002 lambda grids differ; align JSON files before plotting.")

    fd001_rmse = np.array([m.rul_rmse for m in m_fd001], dtype=float)
    fd001_phm = np.array([m.rul_phm_score for m in m_fd001], dtype=float)
    fd002_rmse = np.array([m.rul_rmse for m in m_fd002], dtype=float)
    fd002_phm = np.array([m.rul_phm_score for m in m_fd002], dtype=float)
    lambdas = lam_fd001

    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(8, 4))
    tick_labels = ["0", "0.1", "0.5", "1.0"]

    ax_a.set_title("(a) FD001")
    ax_a.set_xlabel("λ")
    ax_a.set_xticks(lambdas)
    ax_a.set_xticklabels(tick_labels)
    ax_a.set_ylabel("RUL RMSE [cycles]", color="#1976D2")
    ymin, ymax = float(fd001_rmse.min()), float(fd001_rmse.max())
    pad = max(1.0, (ymax - ymin) * 0.15)
    ax_a.set_ylim(max(0.0, ymin - pad), ymax + pad)
    ax_a.grid(axis="y", alpha=0.3)
    ax_a.axhline(y=fd001_rmse[0], color="#1976D2", linestyle="--", linewidth=1.5)
    ax_a.scatter(lambdas, fd001_rmse, color="#1976D2", s=40, zorder=3)
    ax_a.tick_params(axis="y", labelcolor="#1976D2")
    ax_a2 = ax_a.twinx()
    ax_a2.set_ylabel("PHM Score", color="#FF9800")
    ax_a2.plot(lambdas, fd001_phm, color="#FF9800", linestyle="--", linewidth=1.5)
    ax_a2.scatter(lambdas, fd001_phm, color="#FF9800", marker="s", s=40, zorder=3)
    ax_a2.tick_params(axis="y", labelcolor="#FF9800")
    ax_a2.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    lines_a = ax_a.get_lines() + ax_a2.get_lines()
    ax_a.legend(lines_a[:2], ["RMSE (FD001)", "PHM Score (FD001)"], loc="upper right", fontsize=8)

    ax_b.set_title("(b) FD002")
    ax_b.set_xlabel("λ")
    ax_b.set_xticks(lambdas)
    ax_b.set_xticklabels(tick_labels)
    ax_b.set_ylabel("RUL RMSE [cycles]", color="#1976D2")
    ymin, ymax = float(fd002_rmse.min()), float(fd002_rmse.max())
    pad = max(1.0, (ymax - ymin) * 0.15)
    ax_b.set_ylim(max(0.0, ymin - pad), ymax + pad)
    ax_b.grid(axis="y", alpha=0.3)
    ax_b.axhline(y=fd002_rmse[0], color="#1976D2", linestyle="--", linewidth=1.5)
    ax_b.scatter(lambdas, fd002_rmse, color="#1976D2", s=40, zorder=3)
    ax_b.tick_params(axis="y", labelcolor="#1976D2")
    ax_b2 = ax_b.twinx()
    ax_b2.set_ylabel("PHM Score", color="#FF9800")
    ax_b2.plot(lambdas, fd002_phm, color="#FF9800", linestyle="--", linewidth=1.5)
    ax_b2.scatter(lambdas, fd002_phm, color="#FF9800", marker="s", s=40, zorder=3)
    ax_b2.tick_params(axis="y", labelcolor="#FF9800")
    ax_b2.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    lines_b = ax_b.get_lines() + ax_b2.get_lines()
    ax_b.legend(lines_b[:2], ["RMSE (FD002)", "PHM Score (FD002)"], loc="upper right", fontsize=8)

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    fig.text(0.5, 0.02, "λ tested: 0, 0.1, 0.5, 1.0", ha="center", va="bottom", fontsize=8)

    versioned_png = Path(args.output_png).resolve() if args.output_png else next_figure4_versioned("png")
    versioned_pdf = Path(str(versioned_png).replace(".png", ".pdf"))
    fig.savefig(versioned_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(versioned_pdf, dpi=300, bbox_inches="tight", facecolor="white")
    print("Saved versioned PNG:", versioned_png)
    print("Data: FD001 <=", fd001_path)
    print("Data: FD002 <=", fd002_path)

    if not args.no_default:
        d_png = FIG_DIR / f"{FIG4_BASE}.png"
        d_pdf = FIG_DIR / f"{FIG4_BASE}.pdf"
        fig.savefig(d_png, dpi=300, bbox_inches="tight", facecolor="white")
        fig.savefig(d_pdf, dpi=300, bbox_inches="tight", facecolor="white")
        print("Saved default PNG:", d_png)

    plt.close(fig)


if __name__ == "__main__":
    main()
