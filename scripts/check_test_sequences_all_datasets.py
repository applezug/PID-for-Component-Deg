#!/usr/bin/env python
"""
核查 FD001–FD004 测试集在 load_test_sequences_last_window 下的序列数量。

修复前：仅保留 len(x) >= seq_length(256) 的发动机，FD001 仅 1 台，FD002/003/004 也偏少。
修复后：不足 256 的发动机左填充，四数据集应分别为 100、259、100、248（C-MAPSS 标准）。

用法（项目根目录）：
  python scripts/check_test_sequences_all_datasets.py
  python scripts/check_test_sequences_all_datasets.py --data_root data/cmapss --seq_length 256
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.rul_estimator import load_test_sequences_last_window


# C-MAPSS 标准测试集发动机数（用于对比）
EXPECTED = {"FD001": 100, "FD002": 259, "FD003": 100, "FD004": 248}


def main():
    p = argparse.ArgumentParser(description="Check test sequence counts for FD001–FD004")
    p.add_argument("--data_root", type=str, default="data/cmapss")
    p.add_argument("--seq_length", type=int, default=256)
    p.add_argument("--sensor_indices", type=str, default="2,3,7,14")
    args = p.parse_args()
    sensor_indices = [int(x) for x in args.sensor_indices.split(",")]

    print(f"data_root={args.data_root}, seq_length={args.seq_length}")
    print("dataset | sequences | expected | status")
    print("--------|-----------|----------|--------")

    for ds in ["FD001", "FD002", "FD003", "FD004"]:
        try:
            seqs, _, _ = load_test_sequences_last_window(
                args.data_root, ds, args.seq_length, sensor_indices
            )
            n = len(seqs)
            exp = EXPECTED.get(ds, "?")
            status = "OK" if (exp != "?" and n == exp) else ("WARN" if n < exp else "OK")
            print(f"  {ds}   |    {n:3d}     |   {exp}     | {status}")
        except FileNotFoundError as e:
            print(f"  {ds}   |   (skip)   |   {EXPECTED.get(ds,'?')}     | FILE NOT FOUND")
        except Exception as e:
            print(f"  {ds}   |   (error)  |   {EXPECTED.get(ds,'?')}     | {e}")

    print("\n若修复已部署，FD001–FD004 的 sequences 应分别等于 expected。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
