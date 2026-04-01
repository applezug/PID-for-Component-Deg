#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate Figure 1 using real project data (config, history, RUL, metrics, C-MAPSS sample).
Optionally use real MBD-generated trajectories (imputed_trajectory_*.npy) for the Output box.
Result data (see results/optional_work_report.md, run_trajectory_physics_comparison.py):
  - results/diffusion_pinn_trajectories/  MBD+PINN 生成轨迹（推荐用于 Figure1）
  - results/baseline_trajectories/         MBD 基线生成轨迹
Usage:
  python scripts/generate_figure1_real_data.py
  python scripts/generate_figure1_real_data.py --use-generated-dir results/diffusion_pinn_trajectories
Output:
  figures/figure1_real_data_YYYYMMDD_NN.png (versioned)
  figures/figure1_real_data.png (default entry; use --no-default to skip)
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
CONFIG_DIR = ROOT / "config"
RESULTS_DIR = ROOT / "results"
DATA_DIR = ROOT / "data" / "cmapss"
FIG1_BASE = "figure1_real_data"


def next_figure1_versioned_png() -> Path:
    """figures/figure1_real_data_YYYYMMDD_NN.png — NN starts at 01, increments if file exists."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    tag = datetime.now().strftime("%Y%m%d")
    i = 1
    while True:
        p = FIG_DIR / f"{FIG1_BASE}_{tag}_{i:02d}.png"
        if not p.exists():
            return p
        i += 1


def load_yaml_simple(path: Path) -> dict:
    """Load YAML config (prefer PyYAML for nested dicts)."""
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def load_config() -> dict:
    cfg_path = CONFIG_DIR / "hpc_cmapss.yaml"
    if not cfg_path.exists():
        return {}
    return load_yaml_simple(cfg_path)


def load_observed_sequence(seq_length: int = 256, sensor_indices=None) -> np.ndarray:
    """Load one C-MAPSS sequence (first unit, first seq_length rows), normalized to [-1,1]. Shape (T, 4)."""
    if sensor_indices is None:
        sensor_indices = [2, 3, 7, 14]  # T24, T30, P30, Nc
    train_path = DATA_DIR / "train_FD001.txt"
    if not train_path.exists():
        return None
    try:
        data = np.loadtxt(train_path)
        # columns: 0=unit, 1=time, 2..=sensors
        units = np.unique(data[:, 0])
        u = int(units[0])
        block = data[data[:, 0] == u]
        block = block[np.argsort(block[:, 1])]
        if len(block) < seq_length:
            return None
        x = block[:seq_length, sensor_indices].astype(np.float64)
        min_v = x.min(axis=0)
        max_v = x.max(axis=0)
        span = max_v - min_v
        span[span < 1e-8] = 1.0
        x_norm = 2.0 * (x - min_v) / span - 1.0
        return x_norm.astype(np.float32)
    except Exception:
        return None


def load_generated_trajectory(trajectory_dir: Path, channel: int = 1, max_samples: int = 5) -> tuple:
    """
    Load MBD-generated trajectory from imputed_trajectory_*.npy in trajectory_dir.
    Returns (gen_mean, gen_lower, gen_upper, sample_trajs) where sample_trajs is a list of up to
    max_samples arrays normalized to [0,1] for thin overlay; or (None, None, None, []) if not found.
    """
    if trajectory_dir is None or not trajectory_dir.exists():
        return None, None, None, []
    files = sorted(trajectory_dir.glob("imputed_trajectory_*.npy"))
    if not files:
        return None, None, None, []
    try:
        trajs = []
        for f in files[:20]:
            arr = np.load(f)
            if arr.ndim == 2 and arr.shape[1] >= 4:
                trajs.append(arr[:, channel])
        if not trajs:
            return None, None, None, []
        trajs = np.stack(trajs)
        gen_mean = np.mean(trajs, axis=0)
        gen_std = np.std(trajs, axis=0)
        half_width = np.maximum(1.645 * gen_std, 0.04)
        gen_lower = gen_mean - half_width
        gen_upper = gen_mean + half_width
        v_min, v_max = gen_mean.min(), gen_mean.max()
        span = v_max - v_min
        if span < 1e-8:
            span = 1.0
        gen_mean_n = (gen_mean - v_min) / span
        gen_lower_n = np.clip((gen_lower - v_min) / span, 0, 1)
        gen_upper_n = np.clip((gen_upper - v_min) / span, 0, 1)
        sample_trajs = []
        for i in range(min(max_samples, len(trajs))):
            s = (trajs[i] - v_min) / span
            s = np.clip(s, 0, 1)
            sample_trajs.append(s)
        return gen_mean_n, gen_lower_n, gen_upper_n, sample_trajs
    except Exception:
        return None, None, None, []


def draw_poly_arrow(ax, points, color="#555555", lw=1.2):
    for (x_start, y_start), (x_end, y_end) in zip(points[:-1], points[1:]):
        ax.plot([x_start, x_end], [y_start, y_end], color=color, linewidth=lw)
    x_start, y_start = points[-2]
    x_end, y_end = points[-1]
    ax.annotate("", xy=(x_end, y_end), xytext=(x_start, y_start),
                arrowprops=dict(arrowstyle="->", lw=lw, color=color))


def main(use_generated_dir: Path | None = None, write_default: bool = True, output_png: Path | None = None):
    cfg = load_config()
    data_cfg = cfg.get("data", {}) if isinstance(cfg.get("data"), dict) else {}
    diff_cfg = cfg.get("diffusion", {}) if isinstance(cfg.get("diffusion"), dict) else {}
    phys_cfg = cfg.get("physics", {}) if isinstance(cfg.get("physics"), dict) else {}
    rul_cfg = cfg.get("rul", {}) if isinstance(cfg.get("rul"), dict) else {}
    train_cfg = cfg.get("training", {}) if isinstance(cfg.get("training"), dict) else {}

    seq_length = data_cfg.get("seq_length", 256) if isinstance(data_cfg, dict) else 256
    sensor_idx = data_cfg.get("sensor_indices", [2, 3, 7, 14]) if isinstance(data_cfg, dict) else [2, 3, 7, 14]
    timesteps = diff_cfg.get("timesteps", 200) if isinstance(diff_cfg, dict) else 200
    sampling_timesteps = diff_cfg.get("sampling_timesteps", 50) if isinstance(diff_cfg, dict) else 50
    Nsample = diff_cfg.get("Nsample", 1024) if isinstance(diff_cfg, dict) else 1024
    lambda_physics = phys_cfg.get("lambda_physics", 0.5) if isinstance(phys_cfg, dict) else 0.5
    failure_threshold = rul_cfg.get("failure_threshold", 0.8) if isinstance(rul_cfg, dict) else 0.8
    lr = train_cfg.get("learning_rate", 1e-4) if isinstance(train_cfg, dict) else 1e-4
    patience = train_cfg.get("patience", 30) if isinstance(train_cfg, dict) else 30

    rul_report_path = RESULTS_DIR / "multi_dataset_rul_report.json"
    rul_data = {}
    if rul_report_path.exists():
        with open(rul_report_path, "r", encoding="utf-8") as f:
            rul_data = json.load(f)
    summary = rul_data.get("summary", {})
    fd001 = summary.get("FD001", {})
    rul_rmse_fd001 = float(fd001.get("rul_rmse", 64.62))
    n_test_fd001 = int(fd001.get("n_test", 100))
    if fd001.get("failure_threshold") is not None:
        failure_threshold = float(fd001["failure_threshold"])

    metrics_path = RESULTS_DIR / "hpc_cmapss_metrics.json"
    physics_loss_mean = 0.29
    eff_viol = 0.04
    mono_viol = 0.46
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            m = json.load(f)
        physics_loss_mean = m.get("physics_loss_mean", physics_loss_mean)
        eff_viol = m.get("efficiency_violation_ratio", eff_viol)
        mono_viol = m.get("monotonicity_violation_ratio", mono_viol)

    obs_seq = load_observed_sequence(seq_length=seq_length, sensor_indices=sensor_idx)
    if obs_seq is not None:
        # Keep C-MAPSS in [-1,1] so all 4 channels share one scale (like figure1-1)
        obs_traj_plot = obs_seq.astype(np.float64)
    else:
        # Fallback: synthetic degradation curves in [-1,1] so display matches figure1-1 style
        t = np.linspace(0, 1, seq_length)
        obs_traj_plot = np.column_stack([
            0.75 - 0.35 * t + 0.04 * np.sin(4 * np.pi * t),
            0.7 - 0.4 * t + 0.05 * np.sin(3 * np.pi * t),
            0.65 - 0.3 * t + 0.06 * np.sin(5 * np.pi * t),
            0.8 - 0.45 * t + 0.03 * np.sin(2 * np.pi * t),
        ])
        for ch in range(4):
            v = obs_traj_plot[:, ch]
            v = (v - v.min()) / (v.max() - v.min() + 1e-8)
            obs_traj_plot[:, ch] = 2.0 * v - 1.0

    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["font.size"] = 10

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")

    input_bg = Rectangle((0.4, 0.4), 4.2, 7.0, facecolor="#E6F0FA", alpha=0.35, zorder=0)
    core_bg = Rectangle((5.2, 0.4), 4.2, 7.0, facecolor="#F0F7F0", alpha=0.25, zorder=0)
    output_bg = Rectangle((10.0, 0.4), 3.6, 7.0, facecolor="#FFF8E7", alpha=0.35, zorder=0)
    ax.add_patch(input_bg)
    ax.add_patch(core_bg)
    ax.add_patch(output_bg)
    core_box_left, core_box_bottom = 5.2, 0.4
    core_box_w, core_box_h = 4.2, 7.0
    core_dashed = Rectangle((core_box_left, core_box_bottom), core_box_w, core_box_h,
                             fill=False, linestyle="--", edgecolor="#555", linewidth=1.2, zorder=1)
    ax.add_patch(core_dashed)

    ax.text(7, 7.7, "Physics-informed diffusion for component-level degradation and RUL prediction",
            ha="center", va="center", fontsize=12, weight="bold")
    ax.axhline(y=7.4, xmin=0.08, xmax=0.92, color="gray", linestyle="-", linewidth=0.6)
    ax.text(2.5, 7.1, "Input", ha="center", va="center", fontsize=11, weight="bold")
    ax.text(2.5, 6.82, "Component-level (HPC)", ha="center", va="center", fontsize=8, color="gray", style="italic")
    ax.text(7.3, 7.1, "Core", ha="center", va="center", fontsize=11, weight="bold")
    ax.text(11.8, 7.1, "Output", ha="center", va="center", fontsize=11, weight="bold")

    pad = 0.5
    box_r = 0.12

    # ========== 左侧：真实观测轨迹 ==========
    char_h = 0.08
    obs_xy = (0.8, 5.0 - 5 * char_h)
    obs_w, obs_h = 3.4, 1.4
    rect_obs = FancyBboxPatch(obs_xy, obs_w, obs_h, boxstyle=f"round,pad=0.15,rounding_size={box_r}",
                              facecolor="white", edgecolor="#333", linewidth=0.8)
    ax.add_patch(rect_obs)
    ax.text(obs_xy[0] + obs_w / 2, obs_xy[1] + obs_h + 0.2, "Observed trajectories", ha="center", va="bottom", fontsize=10, style="italic")
    # 与 figure1-1 一致：横轴 60 点 + 轻微显示噪声，使波峰波谷更明显
    n_display = 60
    n_pts = obs_traj_plot.shape[0]
    if n_pts > n_display:
        idx = np.linspace(0, n_pts - 1, n_display, dtype=int)
        obs_display = obs_traj_plot[idx]
    else:
        obs_display = obs_traj_plot
        n_display = n_pts
    np.random.seed(42)
    noise_scale = 0.04
    obs_display = obs_display + noise_scale * np.random.randn(*obs_display.shape)
    x_curve = np.linspace(obs_xy[0] + 0.25, obs_xy[0] + obs_w - 0.25, n_display)
    band_height = obs_h * 0.6
    y_base = obs_xy[1] + 0.25
    colors = ["#1E88E5", "#D84315", "#2E7D32", "#6A1B9A"]
    labels = ["T24", "T30", "P30", "Nc"]
    for ch in range(4):
        v = (obs_display[:, ch] + 1.0) / 2.0
        v = np.clip(v, 0, 1)
        y_plot = y_base + v * band_height
        ax.plot(x_curve, y_plot, color=colors[ch], linewidth=1.5, alpha=0.8)
    legend_handles = [Line2D([0], [0], color=colors[i], lw=2, label=labels[i]) for i in range(4)]
    xlim, ylim = (0, 14), (0, 8)
    char_w = 0.08
    leg_x_ax = (obs_xy[0] + obs_w - 0.08 + 2 * char_w) / (xlim[1] - xlim[0])
    leg_y_ax = (obs_xy[1] + obs_h - 0.12 + 2 * char_w) / (ylim[1] - ylim[0])
    ax.legend(handles=legend_handles, loc="upper right", fontsize=7,
              bbox_to_anchor=(leg_x_ax, leg_y_ax), frameon=True, framealpha=0.95)
    ax.text(obs_xy[0] + obs_w / 2, obs_xy[1] + 0.12, f"T24, T30, P30, Nc  (T={seq_length})", ha="center", va="bottom", fontsize=8, color="gray")
    ax.text(obs_xy[0] + obs_w / 2, obs_xy[1] + 0.02, "Normalized to [-1, 1]", ha="center", va="bottom", fontsize=7, color="gray", style="italic")

    # Mask M：借鉴 figure1-1 设计——横线基线、单块深色条、标签在上、双向箭头
    mask_xy = (0.8, 3.0 - 2 * char_h)
    mask_w, mask_h = 3.4, 1.0
    rect_mask = FancyBboxPatch(mask_xy, mask_w, mask_h, boxstyle=f"round,pad=0.2,rounding_size={box_r}",
                               facecolor="#F0F0F0", edgecolor="black", linewidth=1)
    ax.add_patch(rect_mask)
    mask_title_y = 3.0 + mask_h + 0.1 + char_h - 0.6 * char_h
    ax.text(mask_xy[0] + mask_w / 2, mask_title_y, "Mask M", ha="center", va="bottom", fontsize=10, style="italic")
    mask_bar_y = mask_xy[1] + mask_h / 2
    bar_left, bar_right = mask_xy[0] + 0.2, mask_xy[0] + mask_w - 0.2
    bar_w = bar_right - bar_left
    ax.plot([bar_left, bar_right], [mask_bar_y, mask_bar_y], color="black", linewidth=2)
    ax.add_patch(Rectangle((mask_xy[0] + 0.2 + bar_w * 0.5, mask_bar_y - 0.1), bar_w * 0.5, 0.2,
                           facecolor="#444444", alpha=0.8))
    ax.text(mask_xy[0] + 0.2 + bar_w * 0.25, mask_bar_y + 0.22, "observed", ha="center", fontsize=8)
    ax.text(mask_xy[0] + 0.2 + bar_w * 0.75, mask_bar_y + 0.22, "future (masked)", ha="center", fontsize=8)
    ax.annotate("", xy=(bar_right, mask_bar_y - 0.3), xytext=(bar_left, mask_bar_y - 0.3),
                arrowprops=dict(arrowstyle="<->", color="gray", lw=1))
    ax.text(mask_xy[0] + mask_w / 2, mask_bar_y - 0.45, f"t = 0 … {seq_length} cycles", ha="center", fontsize=8)
    ax.text(2.5, 0.65, "Sec. 2.1", fontsize=7, color="gray", ha="center")

    # ========== 中间核心区：真实参数 ==========
    core_x, core_y = 5.6, 0.9
    core_w, core_h = 3.8, 5.5
    mbd_xy = (core_x + 0.2, core_y + core_h - 1.08 - pad)
    mbd_w, mbd_h = core_w - 0.4, 1.08
    rect_mbd = FancyBboxPatch(mbd_xy, mbd_w, mbd_h, boxstyle=f"round,pad=0.12,rounding_size={box_r}",
                              facecolor="#D5E8D4", edgecolor="#2E7D32", linewidth=0.8)
    ax.add_patch(rect_mbd)
    ax.text(mbd_xy[0] + mbd_w / 2, mbd_xy[1] + mbd_h - 0.18, "MBD", ha="center", va="top", fontsize=10, weight="bold")
    ax.text(mbd_xy[0] + 0.06, mbd_xy[1] + mbd_h - 0.06, "forward", fontsize=6, ha="left", va="top", color="gray")
    ax.text(mbd_xy[0] + mbd_w - 0.06, mbd_xy[1] + mbd_h - 0.06, "reverse", fontsize=6, ha="right", va="top", color="gray")
    # 四种退化模型水平子框：linear | exponential | power_law | etc.
    n_sub = 4
    sub_margin = 0.08
    sub_gap = 0.07
    sub_w = (mbd_w - 2 * sub_margin - (n_sub - 1) * sub_gap) / n_sub
    sub_h = 0.42
    sub_y = mbd_xy[1] + mbd_h - 0.18 - 0.08 - sub_h - 2 * char_h
    sub_style = f"round,pad=0.04,rounding_size=0.06"
    deg_boxes = [
        ("linear", r"$y=at+b$"),
        ("exponential", r"$y=y_0 e^{-\lambda t}$"),
        ("power_law", r"$y\propto t^{-n}$"),
        ("etc.", ""),
    ]
    for k, (label, formula) in enumerate(deg_boxes):
        sx = mbd_xy[0] + sub_margin + k * (sub_w + sub_gap)
        rect_sub = FancyBboxPatch((sx, sub_y), sub_w, sub_h, boxstyle=sub_style,
                                  facecolor="white", edgecolor="#2E7D32", linewidth=0.5)
        ax.add_patch(rect_sub)
        if label == "etc.":
            ax.text(sx + sub_w / 2, sub_y + sub_h / 2 - 0.02, "etc.", ha="center", va="center", fontsize=6, color="gray", style="italic")
        else:
            ax.text(sx + sub_w / 2, sub_y + sub_h - 0.08, label, ha="center", va="top", fontsize=6, style="italic")
            ax.text(sx + sub_w / 2, sub_y + 0.12, formula, ha="center", va="bottom", fontsize=6)
    # Clarify that multiple degradation priors are shown for extensibility, but in this work we adopt the linear prior.
    ax.text(
        mbd_xy[0] + mbd_w / 2,
        mbd_xy[1] + 0.06,
        "MBD priors: multiple supported;\nthis work uses linear",
        ha="center",
        va="bottom",
        fontsize=5.2,
        color="#2E7D32",
    )
    ax.text(mbd_xy[0] + mbd_w / 2, mbd_xy[1] + 0.03, f"T={timesteps}  sampling={sampling_timesteps}  Nsample={Nsample}", ha="center", fontsize=6)

    pinn_w, pinn_h = core_w - 0.4, 1.08
    pinn_xy = (core_x + 0.2, mbd_xy[1] - pinn_h - pad)
    rect_pinn = FancyBboxPatch(pinn_xy, pinn_w, pinn_h, boxstyle=f"round,pad=0.12,rounding_size={box_r}",
                               facecolor="#FFE6CC", edgecolor="#E65100", linewidth=0.8)
    ax.add_patch(rect_pinn)
    ax.text(pinn_xy[0] + pinn_w / 2, pinn_xy[1] + pinn_h - 0.18, "PINN", ha="center", va="top", fontsize=10, weight="bold")
    cw = 0.08
    # 四通道子框：T24 | T30 | P30 | Nc，水平排列、与 MBD 一致间距
    n_pinn_sub = 4
    pinn_sub_margin = 0.08
    pinn_sub_gap = 0.07
    pinn_sub_w = (pinn_w - 2 * pinn_sub_margin - (n_pinn_sub - 1) * pinn_sub_gap) / n_pinn_sub
    pinn_sub_h = 0.42
    pinn_sub_y = pinn_xy[1] + pinn_h - 0.18 - 0.08 - pinn_sub_h - 2 * cw
    pinn_sub_style = "round,pad=0.04,rounding_size=0.06"
    pinn_boxes = [
        ("T24", r"$\eta\propto (T_{30}{-}T_{24})$"),
        ("T30", "non-decr."),
        ("P30", r"$P_{30}\propto N_c^2$"),
        ("Nc", r"speed"),
    ]
    for k, (label, formula) in enumerate(pinn_boxes):
        psx = pinn_xy[0] + pinn_sub_margin + k * (pinn_sub_w + pinn_sub_gap)
        rect_psub = FancyBboxPatch((psx, pinn_sub_y), pinn_sub_w, pinn_sub_h, boxstyle=pinn_sub_style,
                                   facecolor="white", edgecolor="#E65100", linewidth=0.5)
        ax.add_patch(rect_psub)
        ax.text(psx + pinn_sub_w / 2, pinn_sub_y + pinn_sub_h - 0.08, label, ha="center", va="top", fontsize=6, style="italic")
        ax.text(psx + pinn_sub_w / 2, pinn_sub_y + 0.12, formula, ha="center", va="bottom", fontsize=6)
    ax.text(pinn_xy[0] + pinn_w / 2, pinn_xy[1] + 0.04, r"$L_{\mathrm{phys}}=L_{\mathrm{speed}}+L_{\mathrm{eff}}+L_{\mathrm{mono}}$", ha="center", fontsize=6, color="gray")

    loss_w, loss_h = core_w - 0.4, 0.6
    loss_xy = (core_x + 0.2, core_y + 0.25)
    rect_loss = FancyBboxPatch(loss_xy, loss_w, loss_h, boxstyle=f"round,pad=0.1,rounding_size={box_r}",
                               facecolor="#F5F5F5", edgecolor="#555", linewidth=0.8)
    ax.add_patch(rect_loss)
    ax.text(loss_xy[0] + loss_w / 2, loss_xy[1] + loss_h / 2 + 0.08, r"$L = L_{\mathrm{diff}} + \lambda L_{\mathrm{phys}}$", ha="center", fontsize=9)
    ax.text(loss_xy[0] + loss_w / 2, loss_xy[1] + 0.12, f"λ={lambda_physics}", ha="center", fontsize=8, color="gray")

    # MBD -> PINN: arrow entirely in gap; MBD contact 0.2 char down, PINN contact 0.1 char up
    core_cx = core_x + core_w / 2
    gap = 0.10
    y_below_mbd = mbd_xy[1] - gap - 0.2 * cw
    y_above_pinn = pinn_xy[1] + pinn_h + gap + 0.1 * cw
    draw_poly_arrow(ax, [(core_cx, y_below_mbd), (core_cx, y_above_pinn)], lw=1.0)
    ax.text(core_cx + 0.22, (y_below_mbd + y_above_pinn) / 2, "physics\nconstraint", fontsize=7, color="gray", ha="left", va="center")
    # PINN -> Loss: arrow entirely in gap; PINN contact 0.1 char down（加粗）
    y_below_pinn = pinn_xy[1] - gap - 0.1 * cw
    y_above_loss = loss_xy[1] + loss_h + gap
    draw_poly_arrow(ax, [(core_cx, y_below_pinn), (core_cx, y_above_loss)], lw=1.6)
    ax.text(core_cx + 0.22, (y_below_pinn + y_above_loss) / 2, "loss", fontsize=7, color="gray", ha="left", va="center")
    ax.text(7.3, 0.65, "Sec. 2.2–2.3", fontsize=7, color="gray", ha="center")

    # ========== 右侧：真实 RUL 与指标 ==========
    out_x = 10.4
    out_w = 3.0
    gen_xy = (out_x, 5.0)
    gen_w, gen_h = out_w, 1.4
    rect_gen = FancyBboxPatch(gen_xy, gen_w, gen_h, boxstyle=f"round,pad=0.15,rounding_size={box_r}",
                              facecolor="#FFF9C4", edgecolor="#333", linewidth=0.8)
    ax.add_patch(rect_gen)
    ax.text(gen_xy[0] + gen_w / 2, gen_xy[1] + gen_h + 0.2, "Generated trajectory", ha="center", va="bottom", fontsize=10, style="italic")
    gen_mean, gen_lower, gen_upper, gen_sample_trajs = load_generated_trajectory(use_generated_dir, channel=1, max_samples=5)
    if gen_mean is not None:
        n_gen = len(gen_mean)
        xg = np.linspace(gen_xy[0] + 0.25, gen_xy[0] + gen_w - 0.25, n_gen)
        for samp in gen_sample_trajs:
            if len(samp) == n_gen:
                ax.plot(xg, gen_xy[1] + 0.25 + samp * gen_h * 0.6, color="gray", linewidth=0.6, alpha=0.35)
        ax.fill_between(xg, gen_xy[1] + 0.25 + gen_lower * gen_h * 0.6, gen_xy[1] + 0.25 + gen_upper * gen_h * 0.6,
                        color="#B3E5FC", alpha=0.5)
        ax.plot(xg, gen_xy[1] + 0.25 + gen_mean * gen_h * 0.6, color="#1E88E5", linewidth=1.5)
    elif obs_seq is not None and obs_traj_plot is not None:
        gen_mean = (obs_traj_plot[:, 1] + 1.0) / 2.0
        gen_t = np.linspace(0, 1, len(gen_mean))
        gen_band = 0.08 * (1 + 0.3 * np.sin(gen_t * 6))
        xg = np.linspace(gen_xy[0] + 0.25, gen_xy[0] + gen_w - 0.25, len(gen_mean))
        ax.fill_between(xg, gen_xy[1] + 0.25 + (gen_mean - gen_band) * gen_h * 0.6,
                        gen_xy[1] + 0.25 + (gen_mean + gen_band) * gen_h * 0.6, color="#B3E5FC", alpha=0.5)
        ax.plot(xg, gen_xy[1] + 0.25 + gen_mean * gen_h * 0.6, color="#1E88E5", linewidth=1.5)
    else:
        # Fallback: generated trajectory as declining trend + small variation (not a straight line)
        xg = np.linspace(gen_xy[0] + 0.25, gen_xy[0] + gen_w - 0.25, 40)
        t = np.linspace(0, 1, 40)
        yg = 0.7 - 0.4 * t + 0.05 * np.sin(4 * np.pi * t)
        ax.fill_between(xg, gen_xy[1] + 0.25 + (yg - 0.08) * gen_h * 0.6, gen_xy[1] + 0.25 + (yg + 0.08) * gen_h * 0.6,
                        color="#B3E5FC", alpha=0.5)
        ax.plot(xg, gen_xy[1] + 0.25 + yg * gen_h * 0.6, color="#1E88E5", linewidth=1.5)
    ax.text(gen_xy[0] + gen_w / 2, gen_xy[1] - 0.1, f"90% PI  (t = 0 … {seq_length})", ha="center", fontsize=7, color="gray")

    rul_xy = (gen_xy[0], 3.2)
    rul_w, rul_h = out_w, 1.05
    rect_rul = FancyBboxPatch(rul_xy, rul_w, rul_h, boxstyle=f"round,pad=0.12,rounding_size={box_r}",
                              facecolor="#FFE0B2", edgecolor="#E65100", linewidth=0.8)
    ax.add_patch(rect_rul)
    ax.text(rul_xy[0] + rul_w / 2, rul_xy[1] + rul_h + 0.2, "RUL estimation", ha="center", va="bottom", fontsize=10, style="italic")
    ax.text(rul_xy[0] + rul_w / 2, rul_xy[1] + rul_h - 0.10, "(rule: crossing threshold → RUL)", ha="center", va="top", fontsize=6, color="gray")
    # 示意图：横轴=时间（漫长过程），每时间点对应一分布（竖向正态），随时间的推移整体下移，最终大部分低于阈值
    rx0, rx1 = rul_xy[0] + 0.22, rul_xy[0] + rul_w - 0.22
    ry_top, ry_bot = rul_xy[1] + 0.56, rul_xy[1] + 0.24
    ry_thresh = rul_xy[1] + 0.38
    n_slices = 9
    x_centers = np.linspace(rx0, rx1, n_slices)
    t_frac = np.linspace(0, 1, n_slices) ** 0.7
    mu_centers = ry_top - (ry_top - ry_bot) * t_frac
    sigma_bell = 0.045
    amp_bell = 0.14
    n_pts_bell = 40
    for i in range(n_slices):
        mu = mu_centers[i]
        y_b = np.linspace(mu - 2 * sigma_bell, mu + 2 * sigma_bell, n_pts_bell)
        pdf = np.exp(-0.5 * ((y_b - mu) / sigma_bell) ** 2)
        pdf = pdf / (pdf.max() + 1e-8)
        x_left = x_centers[i] - amp_bell * pdf
        x_right = x_centers[i] + amp_bell * pdf
        ax.fill_betweenx(y_b, x_left, x_right, facecolor="#E8D5B7", alpha=0.85, edgecolor="#333", linewidth=0.5)
    ax.plot([rx0, rx1], [ry_thresh, ry_thresh], color="red", linestyle="--", linewidth=1.2)
    ax.text(rul_xy[0] + rul_w / 2, rul_xy[1] + 0.10, f"threshold 1−τ (τ={failure_threshold})", ha="center", fontsize=7)

    unc_w, unc_h = out_w, 1.12
    out_gap = 0.75
    unc_title_y = rul_xy[1] - out_gap + 0.2
    unc_subtitle_y = rul_xy[1] - out_gap + 0.06
    unc_xy = (gen_xy[0], rul_xy[1] - unc_h - out_gap - 1.5 * char_h)
    rect_unc = FancyBboxPatch(unc_xy, unc_w, unc_h, boxstyle=f"round,pad=0.12,rounding_size={box_r}",
                              facecolor="#F8BBD9", edgecolor="#C2185B", linewidth=0.8)
    ax.add_patch(rect_unc)
    ax.text(unc_xy[0] + unc_w / 2, unc_title_y, "RUL prediction", ha="center", va="bottom", fontsize=10, style="italic")
    ax.text(unc_xy[0] + unc_w / 2, unc_subtitle_y, "(point est. & uncertainty)", ha="center", va="bottom", fontsize=6, color="gray")
    ax.text(
        unc_xy[0] + unc_w / 2,
        unc_xy[1] + 0.66,
        f"RUL RMSE ≈ {rul_rmse_fd001:.1f} cycles (FD001, n={n_test_fd001})",
        ha="center",
        fontsize=8,
    )
    ax.text(unc_xy[0] + unc_w / 2, unc_xy[1] + 0.48, "90% PI (n_samples=20)", ha="center", fontsize=7)
    int_lo = max(0, int(round(rul_rmse_fd001 - 30)))
    int_hi = int(round(rul_rmse_fd001 + 30))
    ax.text(
        unc_xy[0] + unc_w / 2,
        unc_xy[1] + 0.28,
        f"e.g. [{int_lo}, {int_hi}] cycles (schematic; not actual PI)  L_phys≈{physics_loss_mean:.2f}",
        ha="center",
        fontsize=5.6,
        color="gray",
    )
    ax.text(11.8, 0.65, "Sec. 2.4", fontsize=7, color="gray", ha="center")

    caption_y = 0.35 - 0.8 * 0.08
    ax.text(2.5, caption_y + 0.06, "Training", fontsize=8, ha="center", va="center", weight="bold")
    ax.text(2.5, caption_y - 0.08, r"min $L_{\mathrm{diff}} + \lambda L_{\mathrm{phys}}$  (Adam, lr=" + f"{lr}, patience={patience})", fontsize=7, ha="center", va="center")
    ax.text(7.3, caption_y, "Core: MBD + PINN, composite loss L", fontsize=8, ha="center", va="center")
    ax.text(11.8, caption_y + 0.06, "Inference", fontsize=8, ha="center", va="center", weight="bold")
    ax.text(
        11.8,
        caption_y - 0.08,
        f"MBD sampling (Nsample={Nsample}); physics loss post-hoc; expert intervention (adjust λ based on violation ratios)",
        fontsize=5.8,
        ha="center",
        va="center",
    )

    # Left: line starts OUTSIDE Observed/Mask; line ends at Core dashed box left border.
    obs_right = obs_xy[0] + obs_w
    mask_right = mask_xy[0] + mask_w
    gap_out = 0.18
    x_start_obs = obs_right + gap_out
    x_start_mask = mask_right + gap_out
    x_merge = (x_start_obs + core_box_left) / 2
    mid_in_y = mbd_xy[1] + mbd_h / 2
    char_w = 0.08
    draw_poly_arrow(ax, [
        (x_start_obs, obs_xy[1] + obs_h / 2),
        (x_merge, obs_xy[1] + obs_h / 2),
        (x_merge, mid_in_y),
        (core_box_left, mid_in_y),
    ], lw=1.2)
    draw_poly_arrow(ax, [
        (x_start_mask, mask_xy[1] + mask_h / 2),
        (x_merge, mask_xy[1] + mask_h / 2),
        (x_merge, mid_in_y),
        (core_box_left, mid_in_y),
    ], lw=1.0)
    ax.text(x_merge + 0.08 - char_w - 5 * char_w, mid_in_y + 0.28 + 0.20, "condition\n→ MBD", fontsize=7, color="gray", ha="left", va="center")

    # Right: line starts at Core dashed box right border; split in gap; right connection points 2.5 chars left of box
    core_box_right = core_box_left + core_box_w
    output_left = 10.0
    x_split = (core_box_right + output_left) / 2
    core_mid_y = core_y + core_h / 2
    gen_left = gen_xy[0]
    rul_left = rul_xy[0]
    unc_left = unc_xy[0]
    out_connect_dx = -2.2 * char_w
    # Trunk: from Core dashed box right edge (do not enter Output)
    ax.plot([core_box_right, x_split], [core_mid_y, core_mid_y], color="#555555", linewidth=1.2)
    # Branches: vertical in gap, then horizontal to connection point (2.5 chars left of each box)
    draw_poly_arrow(ax, [
        (x_split, core_mid_y),
        (x_split, gen_xy[1] + gen_h / 2),
        (gen_left + out_connect_dx, gen_xy[1] + gen_h / 2),
    ], lw=1.0)
    ax.text(x_split + 0.12, (core_mid_y + gen_xy[1] + gen_h / 2) / 2 - 2 * char_w, "generate", fontsize=7, color="gray", ha="left", va="center")
    draw_poly_arrow(ax, [
        (x_split, core_mid_y),
        (x_split, rul_xy[1] + rul_h / 2),
        (rul_left + out_connect_dx, rul_xy[1] + rul_h / 2),
    ], lw=1.0)
    draw_poly_arrow(ax, [
        (x_split, core_mid_y),
        (x_split, unc_xy[1] + unc_h / 2),
        (unc_left + out_connect_dx, unc_xy[1] + unc_h / 2),
    ], lw=1.0)

    data_src = "C-MAPSS FD001 (real)" if obs_seq is not None else "C-MAPSS (config)"
    ax.text(0.5, 7.85, data_src, fontsize=7, color="gray", style="italic")
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Journal插图要求：同一版本号需同时输出 PNG（位图）和 PDF（矢量）。
    # 这里以 PNG 的自动/用户指定版本 stem 为准，PDF 仅替换后缀生成。
    versioned_png = output_png if output_png is not None else next_figure1_versioned_png()
    versioned_pdf = versioned_png.with_suffix(".pdf")
    fig.savefig(versioned_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(versioned_pdf, bbox_inches="tight", facecolor="white")
    print("Saved (versioned PNG):", versioned_png)
    print("Saved (versioned PDF):", versioned_pdf)

    default_png = FIG_DIR / f"{FIG1_BASE}.png"
    default_pdf = FIG_DIR / f"{FIG1_BASE}.pdf"
    if write_default:
        fig.savefig(default_png, dpi=300, bbox_inches="tight", facecolor="white")
        fig.savefig(default_pdf, bbox_inches="tight", facecolor="white")
        print("Saved (default PNG):", default_png)
        print("Saved (default PDF):", default_pdf)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Figure 1 with optional real generated trajectory data.")
    parser.add_argument(
        "--use-generated-dir",
        type=str,
        default=None,
        help="Directory containing imputed_trajectory_*.npy (e.g. results or results/trajectories). If set, Generated trajectory box uses these data.",
    )
    parser.add_argument(
        "--no-default",
        action="store_true",
        help="Do not overwrite figures/figure1_real_data.(png/pdf) (only write date-stamped versions).",
    )
    parser.add_argument(
        "--output-png",
        type=str,
        default=None,
        help="Full path to PNG output (overrides auto figure1_real_data_YYYYMMDD_NN.png).",
    )
    args = parser.parse_args()
    use_dir = (ROOT / args.use_generated_dir).resolve() if args.use_generated_dir else None
    out_png = Path(args.output_png).resolve() if args.output_png else None
    main(use_generated_dir=use_dir, write_default=not args.no_default, output_png=out_png)
