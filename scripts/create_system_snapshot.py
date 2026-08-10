# -*- coding: utf-8 -*-
"""Step 1.7: 生成初始 System Snapshot。

把当前实验系统状态固化为 caches/<data_name>/snapshots/<snapshot_id>.json。
相同状态重复运行得到相同 snapshot_id（见 hyperrag/system_snapshot）。

用法（最终验收命令）::

    python scripts/create_system_snapshot.py --data-name neurology_chunk1000
"""
import argparse
import sys
from pathlib import Path

# 让脚本直接 import 项目根目录下的 hyperrag 包。
sys.path.append(str(Path(__file__).resolve().parent.parent))

from hyperrag.experiment_contract import load_contract
from hyperrag.system_snapshot import build_system_snapshot, write_snapshot


def main():
    parser = argparse.ArgumentParser(
        description="生成实验系统状态快照 (system snapshot)"
    )
    parser.add_argument(
        "--data-name",
        type=str,
        default="neurology_chunk1000",
        help="数据集名；默认 working_dir = caches/<data_name> 且 "
             "输出到 caches/<data_name>/snapshots/",
    )
    parser.add_argument(
        "--contract",
        type=str,
        default=None,
        help="契约 YAML 路径；默认加载 docs 下冻结契约。",
    )
    parser.add_argument(
        "--calibration",
        type=str,
        default=None,
        help="tokenizer 校准报告路径；默认 caches/calibration/tokenizer_calibration.json。",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="快照输出目录；默认 caches/<data_name>/snapshots。",
    )
    parser.add_argument(
        "--default-seed",
        type=int,
        default=42,
        help="固定路径默认 seed（写入 snapshot.rules，供复现）。",
    )
    args = parser.parse_args()

    contract = load_contract(args.contract)
    wd = Path("caches") / args.data_name
    out_dir = Path(args.out_dir) if args.out_dir else (wd / "snapshots")

    snapshot = build_system_snapshot(
        contract,
        args.data_name,
        working_dir=wd,
        calibration_path=args.calibration,
        default_seed=args.default_seed,
    )
    path = write_snapshot(snapshot, out_dir)

    print(f"snapshot_id = {snapshot['snapshot_id']}")
    print(f"git_commit_sha = {snapshot['git_commit_sha']}")
    print(f"contract_version = {snapshot['contract_version']}")
    print(f"written -> {path}")


if __name__ == "__main__":
    main()
