#!/usr/bin/env python
"""
每步实验完成后的结果校验（用于服务器端按步骤执行时的通过判定）。

用法（在项目根目录执行）：
  python scripts/verify_step_result.py check
  python scripts/verify_step_result.py mbd-quick
  python scripts/verify_step_result.py mbd-full
  python scripts/verify_step_result.py multi-quick
  python scripts/verify_step_result.py multi-full
  python scripts/verify_step_result.py picp-quick
  python scripts/verify_step_result.py picp-full
  python scripts/verify_step_result.py lambda-quick
  python scripts/verify_step_result.py lambda-full

退出码：0 表示通过，非 0 表示未通过（并打印原因）。
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
EXPECTED_N = {"FD001": 100, "FD002": 259, "FD003": 100, "FD004": 248}


def ok(msg):
    print(f"[OK] {msg}")
    return True


def fail(msg):
    print(f"[FAIL] {msg}", file=sys.stderr)
    return False


def verify_check():
    """Step 0: 四数据集测试序列数与预期一致。"""
    sys.path.insert(0, str(ROOT))
    from utils.rul_estimator import load_test_sequences_last_window
    data_root = str(ROOT / "data" / "cmapss")
    sensor_indices = [2, 3, 7, 14]
    all_ok = True
    for ds in ["FD001", "FD002", "FD003", "FD004"]:
        try:
            seqs, _, _ = load_test_sequences_last_window(
                data_root, ds, 256, sensor_indices
            )
            n = len(seqs)
            exp = EXPECTED_N[ds]
            if n != exp:
                all_ok = fail(f"check: {ds} sequences={n}, expected={exp}")
            else:
                ok(f"check: {ds} sequences={n}")
        except FileNotFoundError as e:
            all_ok = fail(f"check: {ds} data not found: {e}")
        except Exception as e:
            all_ok = fail(f"check: {ds} error: {e}")
    return all_ok


def verify_mbd_quick():
    """MBD 消融快速验证：12 条记录，FD001 n_test=20。"""
    p = RESULTS / "mbd_degradation_ablation_report.json"
    if not p.is_file():
        return fail("mbd-quick: mbd_degradation_ablation_report.json not found")
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    summary = data.get("summary") or {}
    count = 0
    for model in ["linear", "exponential", "power_law"]:
        if model not in summary:
            return fail(f"mbd-quick: missing model {model}")
        for ds in ["FD001", "FD002", "FD003", "FD004"]:
            if ds not in summary[model]:
                return fail(f"mbd-quick: missing {model}/{ds}")
            ent = summary[model][ds]
            if "rul_rmse" not in ent or "n_test" not in ent:
                return fail(f"mbd-quick: {model}/{ds} missing rul_rmse or n_test")
            n_test = ent["n_test"]
            if ds == "FD001" and n_test != 20:
                return fail(f"mbd-quick: FD001 n_test={n_test}, expected 20")
            count += 1
    if count != 12:
        return fail(f"mbd-quick: expected 12 entries, got {count}")
    return ok("mbd-quick: 12 entries, FD001 n_test=20")


def verify_mbd_full():
    """MBD 消融全量：12 条记录，各数据集 n_test 为全量。"""
    p = RESULTS / "mbd_degradation_ablation_report.json"
    if not p.is_file():
        return fail("mbd-full: mbd_degradation_ablation_report.json not found")
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    summary = data.get("summary") or {}
    for model in ["linear", "exponential", "power_law"]:
        if model not in summary:
            return fail(f"mbd-full: missing model {model}")
        for ds in ["FD001", "FD002", "FD003", "FD004"]:
            if ds not in summary[model]:
                return fail(f"mbd-full: missing {model}/{ds}")
            n_test = summary[model][ds].get("n_test")
            exp = EXPECTED_N[ds]
            if n_test != exp:
                return fail(f"mbd-full: {model}/{ds} n_test={n_test}, expected {exp}")
    return ok("mbd-full: all n_test match full expected (100,259,100,248)")


def verify_multi_quick():
    """多数据集 RUL 快速验证：四数据集均有 rul_rmse、n_test>=1。"""
    p = RESULTS / "multi_dataset_rul_report.json"
    if not p.is_file():
        return fail("multi-quick: multi_dataset_rul_report.json not found")
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    summary = data.get("summary") or {}
    for ds in ["FD001", "FD002", "FD003", "FD004"]:
        if ds not in summary:
            return fail(f"multi-quick: missing {ds}")
        ent = summary[ds]
        if "rul_rmse" not in ent or "n_test" not in ent:
            return fail(f"multi-quick: {ds} missing rul_rmse or n_test")
        if ent["n_test"] < 1:
            return fail(f"multi-quick: {ds} n_test={ent['n_test']}")
    return ok("multi-quick: FD001–FD004 present, n_test>=1")


def verify_multi_full():
    """多数据集 RUL 全量：n_test 为 100, 259, 100, 248。"""
    p = RESULTS / "multi_dataset_rul_report.json"
    if not p.is_file():
        return fail("multi-full: multi_dataset_rul_report.json not found")
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    summary = data.get("summary") or {}
    for ds in ["FD001", "FD002", "FD003", "FD004"]:
        if ds not in summary:
            return fail(f"multi-full: missing {ds}")
        n_test = summary[ds].get("n_test")
        exp = EXPECTED_N[ds]
        if n_test != exp:
            return fail(f"multi-full: {ds} n_test={n_test}, expected {exp}")
    return ok("multi-full: n_test 100,259,100,248")


def verify_picp_quick():
    """PICP 快速验证：有 PICP 相关输出即可。"""
    p = RESULTS / "picp_multi_dataset_summary.json"
    if p.is_file():
        return ok("picp-quick: picp_multi_dataset_summary.json exists")
    for ds in ["FD002", "FD003"]:
        q = RESULTS / f"hpc_cmapss_picp_detail_{ds}.json"
        if q.is_file():
            return ok(f"picp-quick: found PICP detail for {ds}")
    return fail("picp-quick: no picp_multi_dataset_summary.json or *_picp_detail_*.json")


def verify_picp_full():
    """PICP 全量：summary 含 FD002/FD003/FD004。"""
    p = RESULTS / "picp_multi_dataset_summary.json"
    if not p.is_file():
        return fail("picp-full: picp_multi_dataset_summary.json not found")
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    datasets = data.get("datasets") or {}
    for ds in ["FD002", "FD003", "FD004"]:
        if ds not in datasets:
            return fail(f"picp-full: missing {ds}")
        if "picp" not in datasets[ds] or "mean_interval_width" not in datasets[ds]:
            return fail(f"picp-full: {ds} missing picp or mean_interval_width")
    return ok("picp-full: FD002/FD003/FD004 present")


def verify_lambda_quick():
    """λ 消融快速验证：FD001 的 lambda 对比 JSON 存在且有数据。"""
    for name in ("lambda_comparison_FD001.json", "lambda_comparison.json"):
        p = RESULTS / name
        if not p.is_file():
            continue
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        metrics = data.get("metrics") or {}
        lambdas = data.get("lambdas") or []
        if len(lambdas) < 2:
            return fail(f"lambda-quick: {name} has <2 lambdas")
        for k in metrics:
            if "rul_rmse" not in metrics[k]:
                return fail(f"lambda-quick: {name} metrics[{k}] missing rul_rmse")
        return ok(f"lambda-quick: {name} exists with metrics")
    return fail("lambda-quick: no lambda_comparison_FD001.json or lambda_comparison.json")


def verify_lambda_full():
    """λ 消融全量：FD001 的 n_test=100。"""
    for name in ("lambda_comparison_FD001.json", "lambda_comparison.json"):
        p = RESULTS / name
        if not p.is_file():
            continue
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        metrics = data.get("metrics") or {}
        if not metrics:
            return fail(f"lambda-full: {name} has no metrics")
        first = next(iter(metrics.values()))
        n_test = first.get("n_test")
        if n_test != EXPECTED_N["FD001"]:
            return fail(f"lambda-full: FD001 n_test={n_test}, expected {EXPECTED_N['FD001']}")
        return ok(f"lambda-full: {name} FD001 n_test=100")
    return fail("lambda-full: no lambda_comparison_FD001.json or lambda_comparison.json")


def main():
    parser = argparse.ArgumentParser(description="Verify result of a single experiment step")
    parser.add_argument("step", choices=[
        "check", "mbd-quick", "mbd-full", "multi-quick", "multi-full",
        "picp-quick", "picp-full", "lambda-quick", "lambda-full"
    ], help="Step name to verify")
    args = parser.parse_args()
    runners = {
        "check": verify_check,
        "mbd-quick": verify_mbd_quick,
        "mbd-full": verify_mbd_full,
        "multi-quick": verify_multi_quick,
        "multi-full": verify_multi_full,
        "picp-quick": verify_picp_quick,
        "picp-full": verify_picp_full,
        "lambda-quick": verify_lambda_quick,
        "lambda-full": verify_lambda_full,
    }
    ok_result = runners[args.step]()
    return 0 if ok_result else 1


if __name__ == "__main__":
    sys.exit(main())
