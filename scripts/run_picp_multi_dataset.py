#!/usr/bin/env python
"""
多数据集 PICP 不确定性量化：在 FD002、FD003、FD004（及可选 FD001）上运行
run_rul_eval --n_samples 20 --interval_quantile 0.9 --n_test 0，
汇总 PICP、平均区间宽度、覆盖台数，便于论文支撑结论。
支持 --ngpu 2 时双卡并行，加快全量评估。

用法（项目根目录）：
  python scripts/run_picp_multi_dataset.py
  python scripts/run_picp_multi_dataset.py --datasets FD002 FD003 FD004 --n_samples 20 --ngpu 2
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
CONFIG = ROOT / "config" / "hpc_cmapss_paper.yaml"


def _run_one_picp(ds: str, config: str, n_samples: int, interval_quantile: float, root: Path, gpu_id: int) -> tuple:
    """在指定 GPU 上跑一个数据集的 PICP 评估，返回 (ds, returncode)。"""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    ret = subprocess.run([
        sys.executable, "experiments/run_rul_eval.py",
        "--config", config,
        "--dataset", ds,
        "--n_test", "0",
        "--n_samples", str(n_samples),
        "--interval_quantile", str(interval_quantile),
    ], env=env, cwd=str(root))
    return (ds, ret.returncode)


def _resolve_metrics_path(ds: str, exp_name: str, deg_type: str) -> Optional[Path]:
    """Prefer current degradation model metrics; fallback to legacy; choose newest if multiple."""
    candidates = []
    preferred = RESULTS / f"{exp_name}_rul_metrics_{ds}_{deg_type}.json"
    legacy = RESULTS / f"{exp_name}_rul_metrics_{ds}.json"
    linear = RESULTS / f"{exp_name}_rul_metrics_{ds}_linear.json"
    for p in [preferred, linear, legacy]:
        if p.exists() and p not in candidates:
            candidates.append(p)
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _collect_picp_result(ds: str, exp_name: str, deg_type: str) -> dict:
    """从 results 读取该数据集的 PICP 结果，返回 summary["datasets"][ds] 形态或 error dict。"""
    detail_path = RESULTS / f"{exp_name}_picp_detail_{ds}.json"
    metrics_path = _resolve_metrics_path(ds, exp_name, deg_type)
    if detail_path.exists():
        with open(detail_path, encoding="utf-8") as f:
            d = json.load(f)
        return {
            "picp": d["picp"],
            "mean_interval_width": d["mean_interval_width"],
            "n_covered": d["n_covered"],
            "n_total": d["n_total"],
            "n_samples": d.get("n_samples", 20),
            "interval_quantile": d["interval_quantile"],
        }
    if metrics_path.exists():
        with open(metrics_path, encoding="utf-8") as f:
            m = json.load(f)
        if "rul_picp" in m:
            return {
                "picp": m["rul_picp"],
                "mean_interval_width": m["rul_mean_interval_width"],
                "n_covered": m.get("rul_n_covered"),
                "n_total": m.get("rul_n_total", m["n_test"]),
                "n_samples": 20,
                "interval_quantile": m.get("rul_interval_quantile", 0.9),
            }
        return {"error": "no PICP in metrics (run with n_samples>1)"}
    return {"error": "no detail or metrics file found"}


def main():
    p = argparse.ArgumentParser(description="Multi-dataset PICP (uncertainty quantification)")
    p.add_argument("--config", type=str, default=str(CONFIG))
    p.add_argument("--datasets", nargs="+", default=["FD002", "FD003", "FD004"],
                   help="Datasets to evaluate (FD001 has only 1 test engine)")
    p.add_argument("--n_samples", type=int, default=20)
    p.add_argument("--interval_quantile", type=float, default=0.9)
    p.add_argument("--exp_name", type=str, default=None, help="From config if not set")
    p.add_argument("--ngpu", type=int, default=1, help="并行 GPU 数，2 时两数据集同时跑")
    args = p.parse_args()
    os.chdir(ROOT)
    exp_name = args.exp_name or "hpc_cmapss"
    # run_rul_eval 默认从 config diffusion.degradation_model 读取，默认线性
    deg_type = "linear"
    try:
        import yaml
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        deg_type = (cfg.get("diffusion", {}).get("degradation_model", "linear") or "linear").lower()
    except Exception:
        pass
    summary = {"datasets": {}, "aggregate": None}
    all_covered = []
    all_total = []
    all_widths = []
    datasets = list(args.datasets)
    ngpu = max(1, min(args.ngpu, len(datasets)))

    if ngpu >= 2 and len(datasets) >= 2:
        # 双卡：按负载均衡分组。默认 FD002,FD003,FD004 对应 259,100,248 → GPU0: FD003+FD004 (348), GPU1: FD002 (259)
        if len(datasets) == 3 and set(datasets) == {"FD002", "FD003", "FD004"}:
            group0, group1 = ["FD003", "FD004"], ["FD002"]  # 348 vs 259
        else:
            # 通用：按顺序前半 GPU0、后半 GPU1（可后续按需改为按规模分组）
            mid = (len(datasets) + 1) // 2
            group0, group1 = datasets[:mid], datasets[mid:]
        print(f"[PICP] 双卡并行（负载均衡）: GPU0 {group0}, GPU1 {group1}")

        def run_group(ds_list, gpu_id):
            out = []
            for ds in ds_list:
                _, ret = _run_one_picp(ds, args.config, args.n_samples, args.interval_quantile, ROOT, gpu_id)
                out.append((ds, ret))
            return out

        with ThreadPoolExecutor(max_workers=2) as ex:
            f0 = ex.submit(run_group, group0, 0)
            f1 = ex.submit(run_group, group1, 1)
            done = 0
            for ds, ret in f0.result() + f1.result():
                done += 1
                print(f"  [{done}/{len(datasets)}] {ds} 完成, returncode={ret}")
                if ret != 0:
                    summary["datasets"][ds] = {"error": "run_rul_eval failed"}
                else:
                    summary["datasets"][ds] = _collect_picp_result(ds, exp_name, deg_type)
        for ds in summary["datasets"]:
            v = summary["datasets"][ds]
            if "error" not in v and "n_covered" in v:
                all_covered.append(v["n_covered"])
                all_total.append(v["n_total"])
                for _ in [1]:
                    detail_path = RESULTS / f"{exp_name}_picp_detail_{ds}.json"
                    if detail_path.exists():
                        with open(detail_path, encoding="utf-8") as f:
                            d = json.load(f)
                        all_widths.extend(e.get("interval_width") for e in d.get("per_engine", []) if e.get("interval_width") is not None)
    else:
        for i, ds in enumerate(datasets):
            print(f"\n--- [{i+1}/{len(datasets)}] PICP: {ds} (n_samples={args.n_samples}) ---")
            ret = subprocess.run([
                sys.executable, "experiments/run_rul_eval.py",
                "--config", args.config,
                "--dataset", ds,
                "--n_test", "0",
                "--n_samples", str(args.n_samples),
                "--interval_quantile", str(args.interval_quantile),
            ], cwd=str(ROOT))
            if ret.returncode != 0:
                summary["datasets"][ds] = {"error": "run_rul_eval failed"}
                continue

            summary["datasets"][ds] = _collect_picp_result(ds, exp_name, deg_type)
            v = summary["datasets"][ds]
            if "error" not in v and "n_covered" in v:
                all_covered.append(v["n_covered"])
                all_total.append(v["n_total"])
                detail_path = RESULTS / f"{exp_name}_picp_detail_{ds}.json"
                if detail_path.exists():
                    with open(detail_path, encoding="utf-8") as f:
                        d = json.load(f)
                    for e in d.get("per_engine", []):
                        if e.get("interval_width") is not None:
                            all_widths.append(e["interval_width"])

    if all_covered and all_total:
        total_covered = sum(all_covered)
        total_engines = sum(all_total)
        summary["aggregate"] = {
            "picp_overall": total_covered / total_engines if total_engines else 0,
            "n_covered_total": total_covered,
            "n_total": total_engines,
            "mean_interval_width_avg": sum(all_widths) / len(all_widths) if all_widths else None,
        }

    out_path = RESULTS / "picp_multi_dataset_summary.json"
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n--- PICP multi-dataset summary ---")
    for ds, v in summary["datasets"].items():
        if "error" in v:
            print(f"  {ds}: {v['error']}")
        else:
            print(f"  {ds}: PICP={v['picp']:.4f} ({v.get('n_covered')}/{v.get('n_total')} covered), mean_width={v.get('mean_interval_width'):.4f}")
    if summary.get("aggregate"):
        a = summary["aggregate"]
        print(f"  Overall: PICP={a['picp_overall']:.4f} ({a['n_covered_total']}/{a['n_total']} engines)")
    print(f"Summary saved to {out_path}")


if __name__ == "__main__":
    main()
