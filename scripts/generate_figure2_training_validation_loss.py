#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Figure 2: training / validation loss vs epoch from results/hpc_cmapss_history.json.

Data priority (latest complete curve among project clones):
  - Default: <repo>/results/hpc_cmapss_history.json
  - --history-json: explicit file

Note: supply a non-empty train_loss/val_loss history (see artifacts/paper/ or your own run).

Outputs:
  - figures/figure2_training_validation_loss_YYYYMMDD_NN.png
  - figures/figure2_training_validation_loss_YYYYMMDD_NN.pdf (journal requirement)
  - figures/figure2_training_validation_loss.png (unless --no-default)
  - figures/figure2_training_validation_loss.pdf (unless --no-default)

Usage:
  python scripts/generate_figure2_training_validation_loss.py
  python scripts/generate_figure2_training_validation_loss.py --history-json path/to/hpc_cmapss_history.json
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
FIG2_BASE = "figure2_training_validation_loss"
ARTIFACTS_PAPER = ROOT / "artifacts" / "paper"


def next_figure2_versioned_png() -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    tag = datetime.now().strftime("%Y%m%d")
    i = 1
    while True:
        p = FIG_DIR / f"{FIG2_BASE}_{tag}_{i:02d}.png"
        if not p.exists():
            return p
        i += 1


def load_history(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with open(path, "r", encoding="utf-8") as f:
        hist = json.load(f)
    train_loss = np.asarray(hist.get("train_loss") or [], dtype=np.float64)
    val_loss = np.asarray(hist.get("val_loss") or [], dtype=np.float64)
    return train_loss, val_loss


def plot_figure2(train_loss: np.ndarray, val_loss: np.ndarray) -> plt.Figure:
    n = min(len(train_loss), len(val_loss))
    if n == 0:
        raise ValueError("train_loss and val_loss must be non-empty")
    train_loss = train_loss[:n]
    val_loss = val_loss[:n]
    epochs = np.arange(1, n + 1)
    n_epochs = n

    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(
        epochs,
        train_loss,
        label="Training loss",
        color="#1E88E5",
        linewidth=2.5,
        marker="o",
        markevery=max(1, n_epochs // 11),
        markersize=3,
    )
    ax.plot(
        epochs,
        val_loss,
        label="Validation loss",
        color="#FB8C00",
        linewidth=2.5,
        linestyle="--",
        marker="s",
        markevery=max(1, n_epochs // 11),
        markersize=3,
    )

    ax.set_title("Training and validation loss", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Loss", fontsize=11)
    ax.grid(alpha=0.25, color="#DDDDDD")
    ax.tick_params(axis="both", direction="in")
    ax.set_xlim(0, max(epochs) * 1.02)

    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 1.05)
    ymax = ax.get_ylim()[1]

    x_stab = min(80, n_epochs)
    x_end = n_epochs
    label_y = ymax * 0.95

    ax.axvline(x=x_stab, color="#B0BEC5", linestyle="--", linewidth=1.2)
    stab_label = f"~{x_stab} epochs\n(stabilization)" if x_stab < 80 else "~80 epochs\n(stabilization)"
    ax.text(
        x_stab,
        label_y,
        stab_label,
        ha="center",
        va="top",
        fontsize=8,
        color="#455A64",
    )

    ax.axvline(x=x_end, color="#616161", linestyle=":", linewidth=1.3)
    ax.text(
        x_end,
        label_y,
        f"{n_epochs} epochs\n(convergence)",
        ha="right",
        va="top",
        fontsize=8,
        color="#424242",
    )

    arrow_y = ymin + (ymax - ymin) * 0.55
    if x_end > x_stab + 3:
        ax.annotate(
            "",
            xy=(x_end, arrow_y),
            xytext=(x_stab, arrow_y),
            arrowprops=dict(arrowstyle="<->", color="#9E9E9E", linewidth=1),
        )
        text_center_x = (x_stab + x_end) / 2 - 2.5
        ax.text(
            text_center_x,
            arrow_y + (ymax - ymin) * 0.02,
            "training plateaued\n(converged over last 20%\nepochs)",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#616161",
        )

    idx_tip = min(14, n_epochs - 1)
    tip_x = float(epochs[idx_tip])
    ann_x = max(8.0, min(28.0, n_epochs * 0.25))
    ax.annotate(
        "initial steep decline",
        xy=(tip_x, float(train_loss[idx_tip])),
        xytext=(ann_x, label_y - (ymax - ymin) * 0.01),
        arrowprops=dict(arrowstyle="->", color="#757575", linewidth=1),
        fontsize=8,
        color="#424242",
        va="top",
        ha="left",
    )

    ax.legend(
        loc="lower left",
        fontsize=10,
        frameon=True,
        framealpha=0.8,
        facecolor="white",
        edgecolor="#B0BEC5",
    )

    plt.tight_layout()
    return fig


def main() -> None:
    p = argparse.ArgumentParser(description="Export Figure 2 (train/val loss).")
    p.add_argument(
        "--history-json",
        type=str,
        default=None,
        help="Path to hpc_cmapss_history.json (default: <repo>/results/hpc_cmapss_history.json).",
    )
    p.add_argument(
        "--no-default",
        action="store_true",
        help="Do not overwrite figure2_training_validation_loss.(png/pdf) (only write date-stamped versions).",
    )
    p.add_argument("--output-png", type=str, default=None, help="Exact PNG path (overrides versioned name).")
    args = p.parse_args()

    if args.history_json:
        hist_path = Path(args.history_json).resolve()
    else:
        cand = ARTIFACTS_PAPER / "hpc_cmapss_history.json"
        hist_path = cand if cand.is_file() else RESULTS_DIR / "hpc_cmapss_history.json"
    if not hist_path.is_file():
        raise SystemExit(f"History file not found: {hist_path}")

    train_loss, val_loss = load_history(hist_path)
    if train_loss.size == 0 or val_loss.size == 0:
        raise SystemExit(
            f"train_loss/val_loss empty in {hist_path}. "
            "Run experiments with diffusion_train enabled or supply a valid history JSON."
        )

    fig = plot_figure2(train_loss, val_loss)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    versioned_png = Path(args.output_png).resolve() if args.output_png else next_figure2_versioned_png()
    versioned_pdf = versioned_png.with_suffix(".pdf")
    fig.savefig(versioned_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(versioned_pdf, bbox_inches="tight", facecolor="white")
    print("Saved (versioned PNG):", versioned_png)
    print("Saved (versioned PDF):", versioned_pdf)
    if not args.no_default:
        default_p = FIG_DIR / f"{FIG2_BASE}.png"
        default_pdf = FIG_DIR / f"{FIG2_BASE}.pdf"
        fig.savefig(default_p, dpi=300, bbox_inches="tight", facecolor="white")
        fig.savefig(default_pdf, bbox_inches="tight", facecolor="white")
        print("Saved (default PNG):", default_p)
        print("Saved (default PDF):", default_pdf)
    plt.close(fig)


if __name__ == "__main__":
    main()
