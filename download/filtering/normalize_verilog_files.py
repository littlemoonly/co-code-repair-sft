# 功能：原地规范化 repair pair 中的 Verilog 文件，并删除文件开头的 Xilinx 模板注释。
# 核心逻辑：统一换行符、清理行尾空白和首尾空行，再保证文件末尾恰好保留一个换行。

import argparse
from pathlib import Path


HEADER_MARKERS = (
    "// Company:",
    "// Create Date:",
    "// Module Name:",
    "// Additional Comments:",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Normalize raw Verilog pair files.")
    parser.add_argument(
        "--files-root",
        type=Path,
        default=Path("data/repair_pairs_raw/files"),
        help="Directory containing raw .v files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report changes without writing files.",
    )
    return parser.parse_args()


def is_separator(line):
    return len(line) >= 20 and set(line) == {"/"}


def remove_one_template_header(lines):
    for start in range(min(10, len(lines))):
        if not is_separator(lines[start]):
            continue

        for end in range(start + 1, min(start + 40, len(lines))):
            if not is_separator(lines[end]):
                continue

            header = lines[start + 1 : end]
            if all(any(line.startswith(marker) for line in header) for marker in HEADER_MARKERS):
                return lines[:start] + lines[end + 1 :], True
            break

    return lines, False


def remove_template_headers(lines):
    removed = 0
    while True:
        lines, found = remove_one_template_header(lines)
        if not found:
            return lines, removed
        removed += 1


def normalize_code(code):
    code = code.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip(" \t") for line in code.split("\n")]
    lines, removed_headers = remove_template_headers(lines)

    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()

    normalized = "\n".join(lines) + "\n" if lines else ""
    return normalized, removed_headers


def decode_code(content):
    try:
        return content.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        try:
            return content.decode("gb18030"), "gb18030"
        except UnicodeDecodeError:
            return content.decode("utf-8", errors="replace"), "utf-8-replaced"


def main():
    args = parse_args()
    total = 0
    changed = 0
    removed_headers = 0
    converted_from_gb18030 = 0
    repaired_invalid_utf8 = 0

    for path in sorted(args.files_root.rglob("*.v")):
        total += 1
        content = path.read_bytes()
        original, encoding = decode_code(content)
        normalized, removed_count = normalize_code(original)
        removed_headers += removed_count
        converted_from_gb18030 += encoding == "gb18030"
        repaired_invalid_utf8 += encoding == "utf-8-replaced"

        normalized_content = normalized.encode("utf-8")
        if normalized_content != content:
            changed += 1
            if not args.dry_run:
                path.write_bytes(normalized_content)

    print(f"total: {total}")
    print(f"changed: {changed}")
    print(f"removed_template_headers: {removed_headers}")
    print(f"converted_from_gb18030: {converted_from_gb18030}")
    print(f"repaired_invalid_utf8: {repaired_invalid_utf8}")
    print(f"dry_run: {args.dry_run}")


if __name__ == "__main__":
    main()
