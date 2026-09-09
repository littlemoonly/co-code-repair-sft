# 功能：将 testbench 无法可靠映射且 pair 数量不超过阈值的题目移入 deprecated。
# 核心逻辑：复用验证脚本的映射规则统计目标题目，从 catalog 删除并追加到 deprecated。

import argparse
import json
from collections import Counter
from pathlib import Path

from ..verification.verify_repair_pairs_with_testbench import (
    load_archive_problems,
    load_jsonl,
    resolve_problem_mappings,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Deprecate unmapped low-volume problems.")
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("data/problem_catalog.jsonl"),
    )
    parser.add_argument(
        "--deprecated",
        type=Path,
        default=Path("data/problem_deprecated.jsonl"),
    )
    parser.add_argument(
        "--pairs",
        type=Path,
        default=Path("data/repair_pairs_raw/metadata/pairs.jsonl"),
    )
    parser.add_argument(
        "--testbench-archive",
        type=Path,
        default=Path("testbench/co-problem-set-all-branches.tar.gz"),
    )
    parser.add_argument("--max-pairs", type=int, default=200)
    return parser.parse_args()


def write_jsonl(path, rows):
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")


def main():
    args = parse_args()
    catalog = load_jsonl(args.catalog)
    pairs = load_jsonl(args.pairs)
    pair_counts = Counter(pair["problem_id"] for pair in pairs)
    mappings = resolve_problem_mappings(
        catalog,
        load_archive_problems(args.testbench_archive),
        threshold=0.8,
        margin=0.02,
    )

    deprecated_ids = {
        problem_id
        for problem_id, count in pair_counts.items()
        if mappings.get(problem_id, {}).get("status") != "mapped"
        and count <= args.max_pairs
    }
    retained = [row for row in catalog if row["id"] not in deprecated_ids]
    moved = [row for row in catalog if row["id"] in deprecated_ids]

    existing = load_jsonl(args.deprecated) if args.deprecated.exists() else []
    deprecated_by_id = {row["id"]: row for row in existing}
    deprecated_by_id.update({row["id"]: row for row in moved})

    write_jsonl(args.catalog, retained)
    write_jsonl(args.deprecated, deprecated_by_id.values())

    print(f"moved: {len(moved)}")
    print(f"catalog_remaining: {len(retained)}")
    print(f"deprecated_total: {len(deprecated_by_id)}")
    print("kept_unmapped_over_threshold:")
    for problem_id, count in sorted(pair_counts.items()):
        if mappings.get(problem_id, {}).get("status") != "mapped" and count > args.max_pairs:
            print(f"  {problem_id}: {count}")


if __name__ == "__main__":
    main()
