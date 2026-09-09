# 功能：对已验证的 Verilog 修复代码对执行硬规则过滤、多维软评分和按分数采样。
# 核心逻辑：明显错误或课程范围外的样本直接拒绝，其余样本计算 C/P/D/I 与训练权重，
# 再保留高分样本并对其他样本加权无放回采样，同时输出低分人工复核池。

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random

from download.filtering.filter_repair_pairs_by_edit_distance import (
    levenshtein_distance,
    remove_newlines,
)


WEIGHTS = {"correctness": 0.41, "purity": 0.29,
           "diversity": 0.18, "informativeness": 0.12}


def load_jsonl(path):
    with Path(path).open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalized_distance(before, after):
    before = remove_newlines(before)
    after = remove_newlines(after)
    return levenshtein_distance(before, after) / max(len(before), len(after), 1)


def hard_rejection(pair, before, after, distance, hard_max_distance, catalog_problem_ids):
    """只删除能够确定错误或几乎完全重写的样本。"""
    if pair.get("valid_repair") is False or pair.get("verification") == "invalid":
        return "invalid_fail_to_ac_label"
    if pair["problem_id"] not in catalog_problem_ids:
        return "out_of_scope_problem"
    if not before.strip() or not after.strip():
        return "empty_code"
    if before == after:
        return "identical_code"
    if distance > hard_max_distance:
        return "near_total_rewrite"
    return None


def diversity_scores(rows):
    """题目越稀缺，D 越高；平方根缩放避免过度放大极小类别。"""
    frequencies = Counter(row["problem_id"] for row in rows)
    minimum = min(frequencies.values())
    return {problem_id: math.sqrt(minimum / count)
            for problem_id, count in frequencies.items()}


def score_pair(pair, distance, diversity):
    if pair.get("verification") == "testbench_verified" or pair.get("valid_repair") is True:
        correctness = 1.0
    elif pair.get("verification") == "manual_checked":
        correctness = 0.8
    else:
        correctness = 0.3
    purity = max(0.0, 1.0 - distance)
    # 小于 5% 的修改可能过于简单；大修改则因 purity 下降而自动降权。
    informativeness = min(1.0, distance / 0.05) * purity
    signals = {
        "correctness": correctness,
        "purity": purity,
        "diversity": diversity,
        "informativeness": informativeness,
    }
    quality_score = sum(WEIGHTS[name] * value for name, value in signals.items())
    if quality_score >= 0.75:
        tier, training_weight = "high", 1.0
    elif quality_score >= 0.50:
        tier, training_weight = "medium", 0.5
    else:
        tier, training_weight = "low", 0.2
    return {
        **pair,
        "quality_signals": {name: round(value, 6) for name, value in signals.items()},
        "quality_score": round(quality_score, 6),
        "quality_tier": tier,
        "training_weight": round(training_weight * correctness, 3),
        "needs_review": correctness < 1.0 or tier == "low",
        "normalized_edit_distance": round(distance, 6),
    }


def score_rows(pairs, files_root, catalog_problem_ids, hard_max_distance):
    candidates = []
    rejected = []
    for pair in pairs:
        before = (files_root / pair["buggy_path"]).read_text(encoding="utf-8")
        after = (files_root / pair["fixed_path"]).read_text(encoding="utf-8")
        distance = normalized_distance(before, after)
        reason = hard_rejection(
            pair, before, after, distance, hard_max_distance, catalog_problem_ids
        )
        if reason:
            rejected.append({**pair, "filter_reason": reason,
                             "normalized_edit_distance": round(distance, 6)})
        else:
            candidates.append((pair, distance))

    diversity = diversity_scores([pair for pair, _ in candidates]) if candidates else {}
    scored = [score_pair(pair, distance, diversity[pair["problem_id"]])
              for pair, distance in candidates]
    return scored, rejected


def weighted_sample(rows, keep_ratio, seed):
    """高分层优先保留，其余按 Q(x) 加权无放回采样。"""
    target = round(len(rows) * keep_ratio)
    high = sorted((row for row in rows if row["quality_tier"] == "high"),
                  key=lambda row: row["quality_score"], reverse=True)
    rest = [row for row in rows if row["quality_tier"] != "high"]
    rng = random.Random(seed)
    rest.sort(key=lambda row: rng.random() ** (1 / max(row["quality_score"], 1e-6)),
              reverse=True)
    selected = (high + rest)[:target]
    return sorted(selected, key=lambda row: str(row["pair_id"]))


def parse_args():
    parser = argparse.ArgumentParser(description="多维质量评分与分层采样")
    base = Path("data/repair_pairs_raw/metadata")
    parser.add_argument("--input", type=Path, default=base / "pairs_judge_verified.jsonl")
    parser.add_argument("--files-root", type=Path, default=Path("data/repair_pairs_raw/files"))
    parser.add_argument("--target-catalog", type=Path, default=Path("data/problem_catalog.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=base / "quality_sampling")
    parser.add_argument("--keep-ratio", type=float, default=0.8)
    parser.add_argument("--hard-max-distance", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0 < args.keep_ratio <= 1:
        raise SystemExit("--keep-ratio 必须在 (0, 1] 内")
    catalog_problem_ids = {int(row["id"]) for row in load_jsonl(args.target_catalog)}
    scored, rejected = score_rows(
        load_jsonl(args.input), args.files_root, catalog_problem_ids, args.hard_max_distance
    )
    selected = weighted_sample(scored, args.keep_ratio, args.seed)
    review = [row for row in scored if row["needs_review"]]

    write_jsonl(args.output_dir / "scored.jsonl", scored)
    write_jsonl(args.output_dir / "selected.jsonl", selected)
    write_jsonl(args.output_dir / "rejected.jsonl", rejected)
    write_jsonl(args.output_dir / "manual_review.jsonl", review)
    summary = {
        "input": len(scored) + len(rejected), "hard_rejected": len(rejected),
        "selected": len(selected), "manual_review": len(review), "weights": WEIGHTS,
        "tiers": dict(Counter(row["quality_tier"] for row in scored)),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
