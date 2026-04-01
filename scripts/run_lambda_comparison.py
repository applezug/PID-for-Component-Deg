#!/usr/bin/env python
"""
Run HPC + RUL evaluation for different physics.lambda_physics values (0, 0.1, 0.5, 1.0).
λ=0 为基线（无物理约束）；保存到 results/lambda_comparison.json。
"""
import os
import sys
import json
import subprocess
import shutil
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
os.chdir(ROOT)


def set_lambda(config_path: Path, lam: float):
    text = config_path.read_text(encoding="utf-8")
    import re
    text = re.sub(r"lambda_physics:\s*[\d.]+", f"lambda_physics: {lam}", text)
    config_path.write_text(text, encoding="utf-8")


def main():
    import argparse
    p = argparse.ArgumentParser(description="Lambda ablation: RUL for λ=0,0.1,0.5,1.0")
    p.add_argument(
        "--config",
        type=str,
        default="config/hpc_cmapss_paper.yaml",
        help="YAML config to mutate (default: paper-aligned τ=0.65).",
    )
    p.add_argument("--n_test", type=int, default=5, help="评估发动机数，0=全部")
    p.add_argument("--dataset", type=str, default="FD001", help="评估数据集，如 FD001 或 FD002（论文 E3 建议 FD002）")
    p.add_argument("--diffusion-train", action="store_true", help="若设置，每次 run_hpc 均带扩散训练（耗时长）")
    args = p.parse_args()
    config_rel = Path(args.config)
    CONFIG = (ROOT / config_rel).resolve() if not config_rel.is_absolute() else config_rel
    CONFIG_BAK = CONFIG.with_suffix(CONFIG.suffix + ".bak_lambda")
    n_test = args.n_test
    dataset = args.dataset
    diffusion_train = getattr(args, "diffusion_train", False)

    # λ=0 基线（无物理）；0.1, 0.5, 1.0 为有物理约束
    lambdas = [0.0, 0.1, 0.5, 1.0]
    results = {}
    if not CONFIG.is_file():
        raise SystemExit(f"Config not found: {CONFIG}")
    if CONFIG_BAK.exists():
        CONFIG_BAK.unlink()
    shutil.copy(CONFIG, CONFIG_BAK)

    for lam in lambdas:
        set_lambda(CONFIG, lam)
        print(f"--- lambda_physics={lam} ---")
        cmd = [PY, "experiments/run_hpc.py", "--config", str(CONFIG)]
        if diffusion_train:
            cmd.append("--diffusion-train")
        r1 = subprocess.run(cmd)
        if r1.returncode != 0:
            results[str(lam)] = {"error": "run_hpc failed"}
            continue
        r2 = subprocess.run([
            PY, "experiments/run_rul_eval.py", "--config", str(CONFIG),
            "--dataset", dataset, "--n_test", str(n_test)
        ])
        if r2.returncode != 0:
            results[str(lam)] = {"error": "run_rul_eval failed"}
            continue
        import yaml
        with open(CONFIG, encoding="utf-8") as f:
            exp_name = yaml.safe_load(f).get("output", {}).get("exp_name", "hpc_cmapss_paper")
        path = ROOT / "results" / f"{exp_name}_rul_metrics_{dataset}.json"
        if not path.exists():
            path = ROOT / "results" / f"{exp_name}_rul_metrics_{dataset}_linear.json"
        if path.exists():
            with open(path, encoding="utf-8") as f:
                results[str(lam)] = json.load(f)
        else:
            results[str(lam)] = {"error": "no metrics file"}

    shutil.copy(CONFIG_BAK, CONFIG)
    out = {
        "timestamp": datetime.now().isoformat(),
        "dataset": dataset,
        "n_test": n_test,
        "lambdas": lambdas,
        "metrics": results,
    }
    out_path = ROOT / "results" / f"lambda_comparison_{dataset}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
