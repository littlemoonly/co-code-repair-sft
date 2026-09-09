# 功能：删除已生成数据集中的 Verilog 注释，并剔除无效样本。
# 核心逻辑：清理 buggy/fixed 代码，重新计算差异和哈希后写回各数据切分。

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
from pathlib import Path
import re


SPLITS = ("train", "dev", "test")
EXCLUDED_PAIR_IDS = {
    "27_1030617_1030632",
    "398_522103_522159",
    "399_663165_663166",
}


def strip_comments(code: str) -> str:
    result = []
    index = 0
    state = "code"

    while index < len(code):
        char = code[index]
        following = code[index + 1] if index + 1 < len(code) else ""

        if state == "code":
            if char == '"':
                result.append(char)
                state = "string"
            elif char == "\\":
                result.append(char)
                state = "escaped_identifier"
            elif char == "/" and following == "/":
                index += 1
                state = "line_comment"
            elif char == "/" and following == "*":
                result.append(" ")
                index += 1
                state = "block_comment"
            else:
                result.append(char)
        elif state == "string":
            result.append(char)
            if char == "\\" and following:
                result.append(following)
                index += 1
            elif char == '"':
                state = "code"
        elif state == "escaped_identifier":
            result.append(char)
            if char.isspace():
                state = "code"
        elif state == "line_comment":
            if char == "\n":
                result.append(char)
                state = "code"
        elif state == "block_comment":
            if char == "\n":
                result.append(char)
            elif char == "*" and following == "/":
                index += 1
                state = "code"

        index += 1

    return "\n".join(line.rstrip() for line in "".join(result).split("\n"))


def normalized_hash(code: str) -> str:
    normalized = re.sub(r"\s+", "", code)
    return hashlib.sha256(normalized.encode()).hexdigest()


def changed_lines(buggy_code: str, fixed_code: str) -> int:
    buggy = [line.strip() for line in buggy_code.splitlines() if line.strip()]
    fixed = [line.strip() for line in fixed_code.splitlines() if line.strip()]
    return sum(
        max(buggy_end - buggy_start, fixed_end - fixed_start)
        for operation, buggy_start, buggy_end, fixed_start, fixed_end
        in difflib.SequenceMatcher(None, buggy, fixed).get_opcodes()
        if operation != "equal"
    )


def clean_row(row: dict) -> tuple[dict, int]:
    before = len(row["buggy_code"]) + len(row["fixed_code"])
    row["buggy_code"] = strip_comments(row["buggy_code"])
    row["fixed_code"] = strip_comments(row["fixed_code"])

    row["buggy_line_count"] = len(row["buggy_code"].splitlines())
    row["fixed_line_count"] = len(row["fixed_code"].splitlines())
    row["changed_lines"] = changed_lines(row["buggy_code"], row["fixed_code"])
    row["buggy_normalized_sha256"] = normalized_hash(row["buggy_code"])
    row["fixed_normalized_sha256"] = normalized_hash(row["fixed_code"])

    after = len(row["buggy_code"]) + len(row["fixed_code"])
    return row, before - after


def clean_file(input_path: Path, output_path: Path) -> tuple[int, int, int]:
    kept = 0
    dropped = 0
    removed_characters = 0
    partial = output_path.with_name(output_path.name + ".part")

    with input_path.open(encoding="utf-8") as source, partial.open(
        "w", encoding="utf-8"
    ) as destination:
        for line in source:
            if not line.strip():
                continue
            row, removed = clean_row(json.loads(line))
            removed_characters += removed
            if (
                row["pair_id"] in EXCLUDED_PAIR_IDS
                or "\ufffd" in row["buggy_code"]
                or "\ufffd" in row["fixed_code"]
            ):
                dropped += 1
                continue
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1

    partial.replace(output_path)
    return kept, dropped, removed_characters


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path, default=project_root / "data" / "dataset"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=project_root / "data" / "dataset"
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        kept, dropped, removed = clean_file(
            args.input_dir / f"{split}.jsonl",
            args.output_dir / f"{split}.jsonl",
        )
        print(
            f"{split}: kept={kept}, dropped={dropped}, "
            f"removed_characters={removed}"
        )


if __name__ == "__main__":
    main()
