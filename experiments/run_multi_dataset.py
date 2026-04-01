"""
多数据集 RUL 评估实验（FD001-FD004）

依次在各数据集上运行 RUL 评估，汇总 RMSE、PHM Score，生成对比报告。
支持 --ngpu 2 时双卡并行：FD001+FD002 在 GPU0，FD003+FD004 在 GPU1，加快全量评估。
Usage: python experiments/run_multi_dataset.py --config config/hpc_cmapss_paper.yaml --n_test 10
       python experiments/run_multi_dataset.py --config config/hpc_cmapss_paper.yaml --n_test 0 --ngpu 2
"""

import os
import sys
import json
import argparse
import subprocess
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from datetime import datetime

from utils.io_utils import load_yaml_config


def _run_one_dataset(ds: str, config: str, n_test: int, root: Path, gpu_id: int) -> tuple:
    """在指定 GPU 上跑一个数据集的 RUL 评估，返回 (ds, returncode)。"""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    cmd = [
        sys.executable, "experiments/run_rul_eval.py",
        "--config", config,
        "--dataset", ds,
        "--n_test", str(n_test),
    ]
    ret = subprocess.run(cmd, env=env, cwd=str(root))
    return (ds, ret.returncode)


def _resolve_metrics_path(results_dir: Path, exp_name: str, ds: str, deg_type: str) -> Optional[Path]:
    """
    Resolve metrics file path robustly:
    1) Prefer current degradation model metrics (e.g., *_linear.json)
    2) Fallback to legacy unsuffixed file
    3) If both exist, pick the newest by mtime to avoid stale reads
    """
    candidates = []
    preferred = results_dir / f"{exp_name}_rul_metrics_{ds}_{deg_type}.json"
    legacy = results_dir / f"{exp_name}_rul_metrics_{ds}.json"
    linear = results_dir / f"{exp_name}_rul_metrics_{ds}_linear.json"
    for p in [preferred, linear, legacy]:
        if p.exists() and p not in candidates:
            candidates.append(p)
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/hpc_cmapss_paper.yaml")
    parser.add_argument("--n_test", type=int, default=10, help="每数据集评估的发动机数（0=全部）")
    parser.add_argument("--datasets", type=str, default="FD001,FD002,FD003,FD004")
    parser.add_argument("--ngpu", type=int, default=1, help="并行 GPU 数，2 时 FD001+FD002 在 GPU0、FD003+FD004 在 GPU1")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    os.chdir(root)

    cfg = load_yaml_config(args.config)
    exp_name = cfg.get("output", {}).get("exp_name", "hpc_cmapss")
    deg_type = cfg.get("diffusion", {}).get("degradation_model", "linear").lower()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    results_dir = root / "results"
    results_dir.mkdir(exist_ok=True)

    summary = {}
    ngpu = max(1, min(args.ngpu, len(datasets)))

    if ngpu >= 2 and len(datasets) >= 2:
        # 双卡：按负载均衡分组（一大一小搭配），避免一侧明显多于另一侧
        # 默认 FD001,FD002,FD003,FD004 对应 100,259,100,248 → GPU0: FD001+FD004 (348), GPU1: FD002+FD003 (359)
        if len(datasets) == 4 and datasets == ["FD001", "FD002", "FD003", "FD004"]:
            group0, group1 = [datasets[0], datasets[3]], [datasets[1], datasets[2]]  # (FD001,FD004) vs (FD002,FD003)
        else:
            # 通用：4 个时 (0,3) vs (1,2) 均衡；否则前半/后半
            if len(datasets) == 4:
                group0, group1 = [datasets[0], datasets[3]], [datasets[1], datasets[2]]
            else:
                mid = (len(datasets) + 1) // 2
                group0, group1 = datasets[:mid], datasets[mid:]
        tasks = [(ds, 0) for ds in group0] + [(ds, 1) for ds in group1]
        print(f"[multi_dataset] 双卡并行（负载均衡）: GPU0 {group0}, GPU1 {group1}")
        with ThreadPoolExecutor(max_workers=2) as ex:
            futures = [ex.submit(_run_one_dataset, ds, args.config, args.n_test, root, gpu_id) for ds, gpu_id in tasks]
            done = 0
            for fut in as_completed(futures):
                ds, ret = fut.result()
                done += 1
                print(f"  [{done}/{len(datasets)}] {ds} 完成, returncode={ret}")
                if ret != 0:
                    summary[ds] = {"error": "run_rul_eval failed", "rul_rmse": None, "rul_phm_score": None}
                else:
                    metrics_path = _resolve_metrics_path(results_dir, exp_name, ds, deg_type)
                    if metrics_path.exists():
                        with open(metrics_path, encoding="utf-8") as f:
                            summary[ds] = json.load(f)
                    else:
                        summary[ds] = {"error": "metrics not found"}
    else:
        for i, ds in enumerate(datasets):
            print(f"\n{'='*50}\n[{i+1}/{len(datasets)}] 评估数据集: {ds}\n{'='*50}")
            cmd = [
                sys.executable, "experiments/run_rul_eval.py",
                "--config", args.config,
                "--dataset", ds,
                "--n_test", str(args.n_test),
            ]
            ret = subprocess.run(cmd)
            if ret.returncode != 0:
                summary[ds] = {"error": "run_rul_eval failed", "rul_rmse": None, "rul_phm_score": None}
                continue

            metrics_path = _resolve_metrics_path(results_dir, exp_name, ds, deg_type)
            if metrics_path and metrics_path.exists():
                with open(metrics_path, encoding="utf-8") as f:
                    summary[ds] = json.load(f)
            else:
                summary[ds] = {"error": "metrics not found"}

    # 汇总报告
    print("\n" + "=" * 60)
    print("多数据集 RUL 评估汇总")
    print("=" * 60)
    table = []
    for ds, m in summary.items():
        if "error" in m:
            print(f"  {ds}: {m['error']}")
            continue
        rmse = m.get("rul_rmse")
        phm = m.get("rul_phm_score")
        n = m.get("n_test")
        table.append({"dataset": ds, "RMSE": rmse, "PHM_Score": phm, "n_test": n})
        print(f"  {ds}: RMSE={rmse:.4f}, PHM={phm:.4f}, n_test={n}")

    report = {
        "timestamp": datetime.now().isoformat(),
        "config": args.config,
        "n_test_per_dataset": args.n_test,
        "summary": summary,
    }
    report_path = results_dir / "multi_dataset_rul_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n汇总报告已保存至 {report_path}")


if __name__ == "__main__":
    main()
