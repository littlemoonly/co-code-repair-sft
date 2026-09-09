# 功能：筛选真实代码修复候选，剔除代码完全相同或长度变化异常大的 FAIL→AC pair。
# 核心逻辑：读取 pair 指向的前后代码，计算相对字符数变化，并分别写出保留和剔除的元数据。

import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Filter FAIL-to-AC repair pairs.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/repair_pairs_raw/metadata/pairs.jsonl"),
        help="Input pair metadata JSONL.",
    )
    parser.add_argument(
        "--files-root",
        type=Path,
        default=Path("data/repair_pairs_raw/files"),
        help="Root directory containing the buggy and fixed code files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/repair_pairs_raw/metadata/pairs_length_filtered.jsonl"),
        help="Output JSONL for retained pairs.",
    )
    parser.add_argument(
        "--rejected-output",
        type=Path,
        default=Path("data/repair_pairs_raw/metadata/pairs_length_rejected.jsonl"),
        help="Output JSONL for rejected pairs and their rejection reasons.",
    )
    parser.add_argument(
        "--max-length-change-ratio",
        type=float,
        default=0.5,
        help="Maximum abs(after-before)/max(before,after); default: 0.5.",
    )
    return parser.parse_args()


def write_jsonl_line(file_obj, item):
    file_obj.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.rejected_output.parent.mkdir(parents=True, exist_ok=True)

    counts = Counter()
    with (
        args.input.open(encoding="utf-8") as input_file,
        args.output.open("w", encoding="utf-8") as output_file,
        args.rejected_output.open("w", encoding="utf-8") as rejected_file,
    ):
        for line in input_file:
            if not line.strip():
                continue

            pair = json.loads(line)
            before = (args.files_root / pair["buggy_path"]).read_text(encoding="utf-8")
            after = (args.files_root / pair["fixed_path"]).read_text(encoding="utf-8")
            before_length = len(before)
            after_length = len(after)
            length_change_ratio = abs(after_length - before_length) / max(
                before_length, after_length, 1
            )

            reason = None
            if before == after:
                reason = "identical_code"
            elif length_change_ratio > args.max_length_change_ratio:
                reason = "abnormal_length_change"

            if reason is None:
                write_jsonl_line(output_file, pair)
                counts["kept"] += 1
            else:
                rejected = {
                    **pair,
                    "filter_reason": reason,
                    "before_length": before_length,
                    "after_length": after_length,
                    "length_change_ratio": round(length_change_ratio, 6),
                }
                write_jsonl_line(rejected_file, rejected)
                counts[reason] += 1

    total = sum(counts.values())
    print(f"total: {total}")
    print(f"kept: {counts['kept']}")
    print(f"identical_code: {counts['identical_code']}")
    print(f"abnormal_length_change: {counts['abnormal_length_change']}")


if __name__ == "__main__":
    main()
