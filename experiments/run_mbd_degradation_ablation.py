"""
MBD 退化模型消融：在 FD001–FD004 上对比 linear / exponential / power_law 的 RUL RMSE。

支持双卡并行：--ngpu 2 时两路任务分别跑在 GPU0/GPU1，可缩短总时间。

Usage:
  python experiments/run_mbd_degradation_ablation.py --config config/hpc_cmapss_paper.yaml --n_test 0 --ngpu 2

可选:
  --n_test 0    全量测试（每数据集全部发动机，耗时长）
  --n_test 20   每数据集 20 台，用于快速验证
  --ngpu 2      使用 2 块 GPU 并行（任务级），默认 2；1 则单卡顺序执行
  --datasets FD001,FD002  只跑部分数据集
  --models linear,exponential  只跑部分退化模型（默认三模型都跑）
  --models exponential,power_law  只跑后两模型时，会与已有 mbd_degradation_ablation_report.json 合并，保留已跑完的 linear，避免重跑
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.io_utils import load_yaml_config


def _run_one_task(deg: str, ds: str, config: str, n_test: int, root: Path, gpu_id: int) -> int:
    """在指定 GPU 上跑一次 run_rul_eval，返回 returncode。"""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    cmd = [
        sys.executable, "experiments/run_rul_eval.py",
        "--config", config,
        "--dataset", ds,
        "--degradation_model", deg,
        "--n_test", str(n_test),
    ]
    ret = subprocess.run(cmd, env=env, cwd=str(root))
    return ret.returncode


def main():
    parser = argparse.ArgumentParser(description="MBD degradation model ablation: linear vs exponential vs power_law")
    parser.add_argument("--config", type=str, default="config/hpc_cmapss_paper.yaml")
    parser.add_argument("--n_test", type=int, default=20, help="每数据集评估发动机数，0=全部")
    parser.add_argument("--datasets", type=str, default="FD001,FD002,FD003,FD004")
    parser.add_argument("--models", type=str, default="linear,exponential,power_law")
    parser.add_argument("--ngpu", type=int, default=2, help="并行 GPU 数（任务级），1=单卡顺序，2=双卡并行")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    os.chdir(root)

    cfg = load_yaml_config(args.config)
    exp_name = cfg.get("output", {}).get("exp_name", "hpc_cmapss")
    results_dir = root / "results"
    results_dir.mkdir(exist_ok=True)

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    # 任务列表：(deg, ds)，顺序固定便于与 GPU 轮询对应
    tasks = [(deg, ds) for deg in models for ds in datasets]
    n_tasks = len(tasks)
    ngpu = max(1, min(args.ngpu, n_tasks))

    if ngpu >= 2:
        print(f"[MBD ablation] 双卡并行: {n_tasks} 个任务, GPU 0/1 轮询 (--ngpu {ngpu})")
    else:
        print(f"[MBD ablation] 单卡顺序: {n_tasks} 个任务")

    summary = {deg: {} for deg in models}
    completed = 0
    with ThreadPoolExecutor(max_workers=ngpu) as executor:
        futures = {}
        for i, (deg, ds) in enumerate(tasks):
            gpu_id = i % ngpu
            fut = executor.submit(_run_one_task, deg, ds, args.config, args.n_test, root, gpu_id)
            futures[fut] = (deg, ds)
        for fut in as_completed(futures):
            deg, ds = futures[fut]
            completed += 1
            try:
                ret = fut.result()
                if ret != 0:
                    summary[deg][ds] = {"error": "run_rul_eval failed", "rul_rmse": None, "rul_phm_score": None}
                else:
                    metrics_path = results_dir / f"{exp_name}_rul_metrics_{ds}_{deg}.json"
                    if metrics_path.exists():
                        with open(metrics_path, encoding="utf-8") as f:
                            summary[deg][ds] = json.load(f)
                    else:
                        summary[deg][ds] = {"error": "metrics file not found"}
            except Exception as e:
                summary[deg][ds] = {"error": str(e), "rul_rmse": None, "rul_phm_score": None}
            print(f"  [{completed}/{n_tasks}] {deg} @ {ds} done.")

    # 与已有报告合并（只跑部分模型时保留已有 linear 等）
    report_path = results_dir / "mbd_degradation_ablation_report.json"
    summary_for_report = dict(summary)
    if report_path.exists():
        try:
            with open(report_path, encoding="utf-8") as f:
                existing = json.load(f)
            existing_summary = existing.get("summary") or {}
            for deg, per_ds in existing_summary.items():
                if deg not in summary_for_report:
                    summary_for_report[deg] = per_ds
            print(f"\n已合并已有报告中的 {list(existing_summary.keys())}，本次运行 {models}。")
        except Exception as e:
            print(f"\n读取已有报告失败 ({e})，仅写入本次运行结果。")

    all_models = sorted(summary_for_report.keys())

    # 打印表格（含合并后的全部模型）
    print("\n" + "=" * 70)
    print("MBD 退化模型消融：RUL RMSE 汇总")
    print("=" * 70)
    for ds in datasets:
        print(f"\n  {ds}:")
        for deg in all_models:
            m = summary_for_report.get(deg, {}).get(ds, {})
            if "error" in m:
                print(f"    {deg}: {m['error']}")
            else:
                rmse = m.get("rul_rmse")
                phm = m.get("rul_phm_score")
                n = m.get("n_test")
                print(f"    {deg}: RMSE={rmse:.4f}, PHM={phm:.4f}, n_test={n}")

    report = {
        "timestamp": datetime.now().isoformat(),
        "config": args.config,
        "n_test_per_dataset": args.n_test,
        "ngpu": ngpu,
        "degradation_models": all_models,
        "datasets": datasets,
        "summary": summary_for_report,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n报告已保存: {report_path}")


if __name__ == "__main__":
    main()
