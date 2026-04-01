"""
三种物理损失策略对比实验：
1. 策略1: 反归一化后使用物理量纲
2. 策略2: 归一化空间的新约束形式
3. 策略3: 原物理公式 + λ=1e-6 缩放

在同一批轨迹上评估各策略的物理损失，保存结果并生成对比图。
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 非 GUI 后端，适合脚本运行
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from datasets.cmapss_dataset import CMAPSSDataset
from models.physics import CompressorPhysics, CompressorPhysicsDenorm, CompressorPhysicsNorm


def fit_speed_pressure_coeff(train_ds) -> float:
    """从健康段拟合 P30 ~ c * Nc^2 (物理空间)。train_ds.sequences 为原始物理值"""
    data = np.vstack([train_ds.sequences[i] for i in range(min(20, len(train_ds)))])
    p30, nc = data[:, 2], data[:, 3]
    nc2 = nc ** 2
    c = np.sum(p30 * nc2) / (np.sum(nc2) + 1e-8)
    return float(c)


def fit_coeff_norm(train_ds) -> float:
    """从归一化数据拟合 P30_norm ~ c * (Nc_norm+1)^2"""
    data = np.vstack([train_ds.sequences[i] for i in range(min(20, len(train_ds)))])
    if train_ds.normalize and train_ds.norm_stats is not None:
        min_v, max_v = train_ds.norm_stats
        data_norm = 2.0 * (data - min_v) / (max_v - min_v + 1e-8) - 1.0
    else:
        data_norm = data
    P30 = data_norm[:, 2].ravel()
    Nc = data_norm[:, 3].ravel()
    x = ((Nc + 1) / 2) ** 2
    c = np.sum(P30 * x) / (np.sum(x * x) + 1e-8)
    return float(c)


def main():
    root = Path(__file__).resolve().parent.parent
    os.chdir(root)

    results_dir = root / "results"
    assets_dir = root / "assets"
    assets_dir.mkdir(exist_ok=True)

    # 加载数据与轨迹
    train_ds = CMAPSSDataset(
        data_root="data/cmapss",
        dataset="FD001",
        seq_length=256,
        period="train",
        train_ratio=0.7,
        val_ratio=0.15,
        seed=42,
        normalize=True,
    )
    val_ds = CMAPSSDataset(
        data_root="data/cmapss",
        dataset="FD001",
        seq_length=256,
        period="val",
        train_ratio=0.7,
        val_ratio=0.15,
        seed=42,
        normalize=True,
        norm_stats=train_ds.global_norm_stats,
    )

    # 加载已生成的轨迹（若存在，为归一化数据），否则用验证集归一化序列
    traj_files = sorted(results_dir.glob("trajectory_*.npy"))
    if not traj_files:
        print("未找到 trajectory_*.npy，使用验证集前5条作为测试轨迹（归一化）")
        trajectories = []
        for i in range(min(5, len(val_ds))):
            x, _, _, _ = val_ds[i]
            trajectories.append(x.numpy())
    else:
        trajectories = [np.load(f).astype(np.float32) for f in traj_files[:5]]
        print(f"加载 {len(trajectories)} 条轨迹 from results/")

    # 拟合系数
    coeff_phys = fit_speed_pressure_coeff(train_ds)
    coeff_norm = fit_coeff_norm(train_ds)
    norm_stats = train_ds.global_norm_stats

    # 三种策略
    physics_s1 = CompressorPhysicsDenorm(
        norm_stats=norm_stats,
        speed_pressure_coeff=coeff_phys,
        efficiency_bounds=(0.7, 0.9),
        device="cpu",
    )
    physics_s2 = CompressorPhysicsNorm(
        speed_pressure_coeff_norm=coeff_norm,
        eta_norm_bounds=(-1.5, 1.5),
        device="cpu",
    )
    physics_s3 = CompressorPhysics(
        speed_pressure_coeff=coeff_phys,
        efficiency_bounds=(0.7, 0.9),
        device="cpu",
    )
    lambda_s3 = 1e-6

    # 评估
    records = []
    for i, traj in enumerate(trajectories):
        t = torch.from_numpy(traj).float().unsqueeze(0)
        l1 = physics_s1(t, condition={}).item()
        l2 = physics_s2(t, condition={}).item()
        l3_raw = physics_s3(t, condition={}).item()
        l3_scaled = lambda_s3 * l3_raw
        records.append({
            "traj_id": i,
            "S1_denorm": l1,
            "S2_norm": l2,
            "S3_raw": l3_raw,
            "S3_scaled": l3_scaled,
        })

    df = pd.DataFrame(records)
    summary = {
        "strategy_1_denorm_mean": float(df["S1_denorm"].mean()),
        "strategy_1_denorm_std": float(df["S1_denorm"].std()),
        "strategy_2_norm_mean": float(df["S2_norm"].mean()),
        "strategy_2_norm_std": float(df["S2_norm"].std()),
        "strategy_3_raw_mean": float(df["S3_raw"].mean()),
        "strategy_3_scaled_mean": float(df["S3_scaled"].mean()),
        "lambda_physics": lambda_s3,
    }

    print("策略对比结果:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    # 保存
    out_path = results_dir / "physics_strategy_comparison.json"
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "per_traj": records}, f, indent=2, ensure_ascii=False)
    df.to_csv(results_dir / "physics_strategy_comparison.csv", index=False)
    print(f"\n结果已保存至 {out_path}")

    # 可视化
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 左图：各策略均值的柱状图（对数刻度，因量级差异大）
    strategies = ["策略1\n反归一化", "策略2\n归一化空间", "策略3\nλ×原物理"]
    means = [
        max(summary["strategy_1_denorm_mean"], 1e-10),
        max(summary["strategy_2_norm_mean"], 1e-10),
        max(summary["strategy_3_scaled_mean"], 1e-10),
    ]
    stds = [
        summary["strategy_1_denorm_std"],
        summary["strategy_2_norm_std"],
        0,
    ]
    colors = ["#2ecc71", "#3498db", "#e74c3c"]
    x_pos = np.arange(len(strategies))
    bars = axes[0].bar(x_pos, means, yerr=stds, capsize=5, color=colors, edgecolor="k")
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(strategies)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("物理损失 (log)")
    axes[0].set_title("三种策略物理损失对比（同一批轨迹）")
    axes[0].grid(axis="y", alpha=0.3)
    for b, v in zip(bars, means):
        axes[0].text(b.get_x() + b.get_width()/2, b.get_height() * 1.2, f"{v:.2e}", ha="center", fontsize=9)

    # 右图：各轨迹上的损失对比（折线，对数刻度）
    x = df["traj_id"].values
    axes[1].semilogy(x, df["S1_denorm"] + 1e-10, "o-", label="策略1 反归一化", color=colors[0])
    axes[1].semilogy(x, df["S2_norm"] + 1e-10, "s-", label="策略2 归一化空间", color=colors[1])
    axes[1].semilogy(x, df["S3_scaled"] + 1e-10, "^-", label=f"策略3 λ={lambda_s3}×原物理", color=colors[2])
    axes[1].set_xlabel("轨迹编号")
    axes[1].set_ylabel("物理损失 (log)")
    axes[1].set_title("各轨迹上三种策略损失")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    date_str = datetime.now().strftime("%Y-%m-%d")
    fig_name = f"physics_strategy_comparison_{date_str}_1.png"
    plt.savefig(assets_dir / fig_name, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"对比图已保存至 assets/{fig_name}")


if __name__ == "__main__":
    main()
