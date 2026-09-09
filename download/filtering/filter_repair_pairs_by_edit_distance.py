# 功能：计算 FAIL→AC 代码对的 normalized edit distance，并剔除修改量过大的样本。
# 核心逻辑：移除换行符后计算字符级 Levenshtein 距离，默认用 Q3 + 1.5×IQR 作为异常值阈值。

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter repair pairs by normalized edit distance."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/repair_pairs_raw/metadata/pairs_judge_verified.jsonl"),
        help="Input pair metadata JSONL.",
    )
    parser.add_argument(
        "--files-root",
        type=Path,
        default=Path("data/repair_pairs_raw/files"),
        help="Root directory containing buggy and fixed code files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/repair_pairs_raw/metadata/pairs_edit_distance_filtered.jsonl"
        ),
        help="Output JSONL for retained pairs.",
    )
    parser.add_argument(
        "--rejected-output",
        type=Path,
        default=Path(
            "data/repair_pairs_raw/metadata/pairs_edit_distance_rejected.jsonl"
        ),
        help="Output JSONL for rejected pairs.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path(
            "data/repair_pairs_raw/metadata/edit_distance_summary.json"
        ),
        help="Output JSON file containing distribution statistics.",
    )
    parser.add_argument(
        "--max-distance",
        type=float,
        help="Explicit threshold in [0, 1]; defaults to the IQR upper fence.",
    )
    parser.add_argument(
        "--iqr-multiplier",
        type=float,
        default=1.5,
        help="IQR multiplier used to derive the threshold; default: 1.5.",
    )
    return parser.parse_args()


def remove_newlines(code):
    return code.replace("\r\n", "").replace("\n", "").replace("\r", "")


def trim_common_edges(left, right):
    prefix = 0
    shared_length = min(len(left), len(right))
    while prefix < shared_length and left[prefix] == right[prefix]:
        prefix += 1

    left = left[prefix:]
    right = right[prefix:]
    suffix = 0
    shared_length = min(len(left), len(right))
    while suffix < shared_length and left[-suffix - 1] == right[-suffix - 1]:
        suffix += 1

    if suffix:
        return left[:-suffix], right[:-suffix]
    return left, right


def levenshtein_distance(left, right):
    left, right = trim_common_edges(left, right)
    if len(left) > len(right):
        left, right = right, left
    if not left:
        return len(right)

    character_masks = {}
    for index, character in enumerate(left):
        character_masks[character] = character_masks.get(character, 0) | (1 << index)

    positive = ~0
    negative = 0
    distance = len(left)
    highest_bit = 1 << (len(left) - 1)

    for character in right:
        matches = character_masks.get(character, 0)
        combined = matches | negative
        horizontal = (((matches & positive) + positive) ^ positive) | matches
        positive_horizontal = negative | ~(horizontal | positive)
        negative_horizontal = positive & horizontal

        if positive_horizontal & highest_bit:
            distance += 1
        elif negative_horizontal & highest_bit:
            distance -= 1

        positive_horizontal = (positive_horizontal << 1) | 1
        negative_horizontal <<= 1
        positive = negative_horizontal | ~(combined | positive_horizontal)
        negative = positive_horizontal & combined

    return distance


def percentile(sorted_values, fraction):
    position = (len(sorted_values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def load_pairs(input_path, files_root):
    scored_pairs = []
    with input_path.open(encoding="utf-8") as input_file:
        for line in input_file:
            if not line.strip():
                continue

            pair = json.loads(line)
            before = remove_newlines(
                (files_root / pair["buggy_path"]).read_text(encoding="utf-8")
            )
            after = remove_newlines(
                (files_root / pair["fixed_path"]).read_text(encoding="utf-8")
            )
            edit_distance = levenshtein_distance(before, after)
            normalized_distance = edit_distance / max(len(before), len(after), 1)
            scored_pairs.append(
                {
                    **pair,
                    "edit_distance": edit_distance,
                    "normalized_edit_distance": round(normalized_distance, 6),
                }
            )
    return scored_pairs


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    scored_pairs = load_pairs(args.input, args.files_root)
    distances = sorted(pair["normalized_edit_distance"] for pair in scored_pairs)

    quantiles = {
        "min": percentile(distances, 0),
        "p25": percentile(distances, 0.25),
        "p50": percentile(distances, 0.5),
        "p75": percentile(distances, 0.75),
        "p90": percentile(distances, 0.9),
        "p95": percentile(distances, 0.95),
        "p99": percentile(distances, 0.99),
        "max": percentile(distances, 1),
    }
    iqr = quantiles["p75"] - quantiles["p25"]
    iqr_upper_fence = min(1.0, quantiles["p75"] + args.iqr_multiplier * iqr)
    threshold = args.max_distance if args.max_distance is not None else iqr_upper_fence

    kept = [
        pair for pair in scored_pairs if pair["normalized_edit_distance"] <= threshold
    ]
    rejected = [
        {**pair, "filter_reason": "excessive_edit_distance"}
        for pair in scored_pairs
        if pair["normalized_edit_distance"] > threshold
    ]

    write_jsonl(args.output, kept)
    write_jsonl(args.rejected_output, rejected)

    summary = {
        "total": len(scored_pairs),
        "kept": len(kept),
        "rejected": len(rejected),
        "normalization": "remove CRLF, LF, and CR characters",
        "normalized_edit_distance": "levenshtein_distance / max(before_length, after_length)",
        "distribution": {key: round(value, 6) for key, value in quantiles.items()},
        "mean": round(sum(distances) / len(distances), 6),
        "iqr_multiplier": args.iqr_multiplier,
        "iqr_upper_fence": round(iqr_upper_fence, 6),
        "threshold": round(threshold, 6),
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
