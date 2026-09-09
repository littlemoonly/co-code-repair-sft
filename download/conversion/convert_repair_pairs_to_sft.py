# 将 Verilog 修复 pair 转换为标准 chat SFT JSONL 格式。
#
# 核心逻辑：根据 problem_id 从题目目录读取 description，再根据 pair 中的
# buggy_path 和 fixed_path 读取错误代码与修复代码，组装 system/user/assistant 消息。

import argparse
import json
from pathlib import Path


SYSTEM_PROMPT = (
    "You are an expert Verilog code repair assistant. Fix the given buggy "
    "Verilog code according to the problem specification. Return only the "
    "complete corrected Verilog code without explanations."
)


def load_descriptions(path):
    descriptions = {}
    with path.open(encoding="utf-8") as source:
        for line in source:
            problem = json.loads(line)
            descriptions[problem["id"]] = problem["description"]
    return descriptions


def convert_pair(pair, descriptions, files_dir):
    buggy_code = (files_dir / pair["buggy_path"]).read_text(encoding="utf-8")
    fixed_code = (files_dir / pair["fixed_path"]).read_text(encoding="utf-8")
    problem_description = descriptions[pair["problem_id"]]

    return {
        "pair_id": pair["pair_id"],
        "problem_id": pair["problem_id"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Problem specification:\n{problem_description}\n\n"
                    f"Buggy Verilog code:\n```verilog\n{buggy_code}\n```"
                ),
            },
            {"role": "assistant", "content": fixed_code},
        ],
    }


def main():
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Convert repair pairs to chat SFT JSONL.")
    parser.add_argument(
        "--input",
        type=Path,
        default=project_root / "data/repair_pairs_raw/metadata/splits/train.jsonl",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=project_root / "data/problem_catalog.jsonl",
    )
    parser.add_argument(
        "--files-dir",
        type=Path,
        default=project_root / "data/repair_pairs_raw/files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "data/repair_pairs_sft/train.jsonl",
    )
    args = parser.parse_args()

    descriptions = load_descriptions(args.catalog)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with args.input.open(encoding="utf-8") as source, args.output.open(
        "w", encoding="utf-8"
    ) as destination:
        for line in source:
            pair = json.loads(line)
            sft_pair = convert_pair(pair, descriptions, args.files_dir)
            destination.write(json.dumps(sft_pair, ensure_ascii=False) + "\n")
            count += 1

    print(f"Converted {count} pairs to {args.output}")


if __name__ == "__main__":
    main()
