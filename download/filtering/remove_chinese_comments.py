# 功能：原地删除 repair pair 的 Verilog 文件中所有包含中文字符的注释。
# 核心逻辑：按字符串、转义标识符、行注释和块注释解析代码，只删除含中文的注释并保留英文注释。

import argparse
import re
from pathlib import Path


CHINESE_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def parse_args():
    parser = argparse.ArgumentParser(description="Remove Chinese Verilog comments.")
    parser.add_argument(
        "--files-root",
        type=Path,
        default=Path("data/repair_pairs_raw/files"),
        help="Directory containing normalized .v files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report changes without writing files.",
    )
    return parser.parse_args()


def contains_chinese(text):
    return CHINESE_PATTERN.search(text) is not None


def strip_chinese_comments(code):
    result = []
    index = 0
    state = "code"
    removed = 0

    while index < len(code):
        char = code[index]
        following = code[index + 1] if index + 1 < len(code) else ""

        if state == "code":
            if char == '"':
                result.append(char)
                state = "string"
                index += 1
            elif char == "\\":
                result.append(char)
                state = "escaped_identifier"
                index += 1
            elif char == "/" and following == "/":
                end = code.find("\n", index)
                end = len(code) if end == -1 else end
                comment = code[index:end]
                if contains_chinese(comment):
                    removed += 1
                else:
                    result.append(comment)
                index = end
            elif char == "/" and following == "*":
                end = code.find("*/", index + 2)
                end = len(code) if end == -1 else end + 2
                comment = code[index:end]
                if contains_chinese(comment):
                    result.append(" " + "\n" * comment.count("\n"))
                    removed += 1
                else:
                    result.append(comment)
                index = end
            else:
                result.append(char)
                index += 1
        elif state == "string":
            result.append(char)
            if char == "\\" and following:
                result.append(following)
                index += 2
            else:
                if char == '"':
                    state = "code"
                index += 1
        elif state == "escaped_identifier":
            result.append(char)
            index += 1
            if char.isspace():
                state = "code"

    cleaned = "\n".join(line.rstrip(" \t") for line in "".join(result).split("\n"))
    cleaned = cleaned.strip("\n")
    return cleaned + "\n" if cleaned else "", removed


def main():
    args = parse_args()
    total_files = 0
    changed_files = 0
    removed_comments = 0

    for path in sorted(args.files_root.rglob("*.v")):
        total_files += 1
        original = path.read_text(encoding="utf-8")
        cleaned, removed = strip_chinese_comments(original)
        removed_comments += removed

        if cleaned != original:
            changed_files += 1
            if not args.dry_run:
                path.write_text(cleaned, encoding="utf-8", newline="\n")

    print(f"total_files: {total_files}")
    print(f"changed_files: {changed_files}")
    print(f"removed_comments: {removed_comments}")
    print(f"dry_run: {args.dry_run}")


if __name__ == "__main__":
    main()
