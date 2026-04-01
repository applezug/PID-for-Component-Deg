#!/usr/bin/env python
"""
专家干预低成本补充实验：模拟“依据可解释信息（违反比例）加大物理权重”的在线调整。

流程：
  - 先以较小 λ（如 0.1）训练并生成少量轨迹，记录违反比例与物理损失；
  - 再以较大 λ（如 0.5）重新训练并生成，记录同样指标；
  - 对比“调整前/后”，说明专家可依据可解释指标进行参数干预。

用法（本地低成本）：
  python scripts/run_expert_intervention_experiment.py --epochs 25 --n-gen 2
  python scripts/run_expert_intervention_experiment.py   # 默认 epochs=30, n_gen=2
"""
import os
import sys
import json
import re
import subprocess
import shutil
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config" / "hpc_cmapss.yaml"
CONFIG_BAK = ROOT / "config" / "hpc_cmapss.yaml.bak_expert"
METRICS_FILE = ROOT / "results" / "hpc_cmapss_metrics.json"
OUT_FILE = ROOT / "results" / "expert_intervention_experiment.json"
PY = sys.executable

os.chdir(ROOT)


def set_lambda(lam: float) -> None:
    text = CONFIG.read_text(encoding="utf-8")
    text = re.sub(r"lambda_physics:\s*[\d.]+", f"lambda_physics: {lam}", text)
    CONFIG.write_text(text, encoding="utf-8")


def read_metrics():
    if not METRICS_FILE.exists():
        return None
    with open(METRICS_FILE, encoding="utf-8") as f:
        return json.load(f)


def main():
    import argparse
    p = argparse.ArgumentParser(description="专家干预补充实验：λ 调整前后可解释指标对比")
    p.add_argument("--epochs", type=int, default=30, help="扩散训练轮数（低成本用 25–30）")
    p.add_argument("--n-gen", type=int, default=2, help="每次生成的轨迹条数")
    p.add_argument("--lambda-before", type=float, default=0.1, help="模拟“调整前”的物理权重")
    p.add_argument("--lambda-after", type=float, default=0.5, help="模拟“专家加大物理权重”后的 λ")
    args = p.parse_args()

    if not CONFIG.exists():
        print(f"Config not found: {CONFIG}")
        return 1

    # 备份配置
    if CONFIG_BAK.exists():
        CONFIG_BAK.unlink()
    shutil.copy(CONFIG, CONFIG_BAK)

    before_metrics = None
    after_metrics = None

    try:
        # ---------- 调整前：小 λ ----------
        set_lambda(args.lambda_before)
        print(f"\n>>> 阶段 1/2：lambda_physics={args.lambda_before}（调整前）")
        cmd = [
            PY, "experiments/run_hpc.py", "--config", str(CONFIG),
            "--diffusion-train", "--epochs", str(args.epochs), "--n-gen", str(args.n_gen),
        ]
        r1 = subprocess.run(cmd)
        if r1.returncode != 0:
            print("run_hpc (before) failed.")
            before_metrics = {"error": "run_hpc failed"}
        else:
            before_metrics = read_metrics()

        # ---------- 调整后：大 λ（专家依据可解释信息加大物理权重）----------
        set_lambda(args.lambda_after)
        print(f"\n>>> 阶段 2/2：lambda_physics={args.lambda_after}（专家加大物理权重后）")
        r2 = subprocess.run([
            PY, "experiments/run_hpc.py", "--config", str(CONFIG),
            "--diffusion-train", "--epochs", str(args.epochs), "--n-gen", str(args.n_gen),
        ])
        if r2.returncode != 0:
            print("run_hpc (after) failed.")
            after_metrics = {"error": "run_hpc failed"}
        else:
            after_metrics = read_metrics()

    finally:
        shutil.copy(CONFIG_BAK, CONFIG)
        if CONFIG_BAK.exists():
            CONFIG_BAK.unlink()

    # ---------- 汇总 ----------
    report = {
        "timestamp": datetime.now().isoformat(),
        "description": "专家干预补充实验：依据可解释信息（物理违反比例）加大物理权重 λ 后的前后对比",
        "params": {
            "epochs": args.epochs,
            "n_gen": args.n_gen,
            "lambda_before": args.lambda_before,
            "lambda_after": args.lambda_after,
        },
        "before": before_metrics,
        "after": after_metrics,
    }

    # 可解释指标对比小结
    if before_metrics and after_metrics and "error" not in before_metrics and "error" not in after_metrics:
        eff_b = before_metrics.get("efficiency_violation_ratio")
        eff_a = after_metrics.get("efficiency_violation_ratio")
        mono_b = before_metrics.get("monotonicity_violation_ratio")
        mono_a = after_metrics.get("monotonicity_violation_ratio")
        ploss_b = before_metrics.get("physics_loss_mean")
        ploss_a = after_metrics.get("physics_loss_mean")
        summary = {
            "efficiency_violation_ratio": {"before": eff_b, "after": eff_a},
            "monotonicity_violation_ratio": {"before": mono_b, "after": mono_a},
            "physics_loss_mean": {"before": ploss_b, "after": ploss_a},
        }
        report["summary"] = summary
        report["conclusion"] = (
            "当依据可解释信息（如违反比例）加大物理权重后，生成轨迹的物理损失与违反比例预期下降，"
            "体现专家可依据可解释指标进行在线参数干预。"
        )
        print("\n--- 可解释指标对比 ---")
        print(f"  efficiency_violation_ratio:  {eff_b} -> {eff_a}")
        print(f"  monotonicity_violation_ratio: {mono_b} -> {mono_a}")
        print(f"  physics_loss_mean:           {ploss_b} -> {ploss_a}")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {OUT_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
