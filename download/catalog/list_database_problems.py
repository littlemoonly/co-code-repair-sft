# 功能：列出数据库中包含可下载学生提交的题目。
# 核心逻辑：聚合提交数量、学生数和结果分布，并按指定格式输出。

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

from ..common import config


PROBLEM_SQL = r"""
WITH attachments AS (
    SELECT
        r.problem_id,
        r.edx_username,
        r.judge_result,
        r.course_code,
        r.submitted_at,
        COALESCE(
            substring(lower(f.filename) from '(\.[^.]+)$'),
            '<none>'
        ) AS extension
    FROM judge_problemjudgerecord r
    JOIN judge_usersubmittedfile f ON f.id = r.attachment_id
),
submission_stats AS (
    SELECT
        problem_id,
        count(*) AS submission_count,
        count(*) FILTER (
            WHERE extension IN ('.v', '.sv')
        ) AS verilog_file_count,
        count(*) FILTER (
            WHERE extension IN ('.zip', '.rar', '.7z')
        ) AS archive_file_count,
        count(DISTINCT edx_username) FILTER (
            WHERE extension IN ('.v', '.sv')
        ) AS verilog_student_count,
        count(*) FILTER (
            WHERE extension IN ('.v', '.sv') AND judge_result = 0
        ) AS verilog_result_0_count,
        count(*) FILTER (
            WHERE extension IN ('.v', '.sv') AND judge_result = 1
        ) AS verilog_result_1_count,
        array_agg(DISTINCT extension ORDER BY extension) AS file_extensions,
        array_agg(DISTINCT course_code ORDER BY course_code) FILTER (
            WHERE course_code IS NOT NULL AND course_code <> ''
        ) AS course_codes,
        min(submitted_at) AS first_submission,
        max(submitted_at) AS last_submission
    FROM attachments
    GROUP BY problem_id
),
testcase_stats AS (
    SELECT problem_id, count(*) AS test_case_count
    FROM judge_problem_test_cases
    GROUP BY problem_id
)
SELECT
    p.id,
    p.name,
    p.type,
    p.description,
    s.submission_count,
    s.verilog_file_count,
    s.archive_file_count,
    s.submission_count - s.verilog_file_count - s.archive_file_count
        AS other_file_count,
    s.verilog_student_count,
    s.verilog_result_0_count,
    s.verilog_result_1_count,
    COALESCE(t.test_case_count, 0) AS test_case_count,
    s.file_extensions,
    COALESCE(s.course_codes, ARRAY[]::varchar[]) AS course_codes,
    s.first_submission,
    s.last_submission
FROM judge_problem p
JOIN submission_stats s ON s.problem_id = p.id
LEFT JOIN testcase_stats t ON t.problem_id = p.id
WHERE s.verilog_file_count >= %s
ORDER BY s.verilog_file_count DESC, s.submission_count DESC, p.id
"""


TABLE_FIELDS = (
    "id",
    "name",
    "type",
    "verilog_file_count",
    "verilog_student_count",
    "verilog_result_0_count",
    "verilog_result_1_count",
    "test_case_count",
    "archive_file_count",
    "course_count",
    "has_description",
    "first_submission",
    "last_submission",
    "description_preview",
)


def load_problems(min_verilog_files: int, limit: int | None) -> list[dict]:
    cursor = config.connect_db()
    connection = cursor.connection
    cursor.execute(PROBLEM_SQL, (min_verilog_files,))
    fields = [column.name for column in cursor.description]
    rows = [dict(zip(fields, row)) for row in cursor]
    cursor.close()
    connection.close()
    for row in rows:
        row["description"] = clean_optional_text(row["description"])
    return rows[:limit] if limit is not None else rows


def clean_optional_text(value: str | None) -> str | None:
    if value is None or value.strip().lower() in {"", "null", "none"}:
        return None
    return value


def compact_text(value: str | None, length: int = 100) -> str:
    text = re.sub(r"\s+", " ", value or "").strip()
    return text if len(text) <= length else text[: length - 1] + "…"


def table_row(problem: dict) -> dict:
    return {
        "id": problem["id"],
        "name": compact_text(problem["name"]),
        "type": problem["type"],
        "verilog_file_count": problem["verilog_file_count"],
        "verilog_student_count": problem["verilog_student_count"],
        "verilog_result_0_count": problem["verilog_result_0_count"],
        "verilog_result_1_count": problem["verilog_result_1_count"],
        "test_case_count": problem["test_case_count"],
        "archive_file_count": problem["archive_file_count"],
        "course_count": len(problem["course_codes"]),
        "has_description": bool(problem["description"]),
        "first_submission": problem["first_submission"],
        "last_submission": problem["last_submission"],
        "description_preview": compact_text(problem["description"]),
    }


def render_table(problems: list[dict]) -> str:
    lines = ["\t".join(TABLE_FIELDS)]
    for problem in problems:
        row = table_row(problem)
        lines.append(
            "\t".join(
                str(row[field]).replace("\t", " ") for field in TABLE_FIELDS
            )
        )
    return "\n".join(lines) + "\n"


def render_jsonl(problems: list[dict]) -> str:
    return "".join(
        json.dumps(
            {
                key: value
                for key, value in {
                    "id": problem["id"],
                    "name": problem["name"],
                    "description": problem["description"],
                }.items()
                if value is not None
            },
            ensure_ascii=False,
        )
        + "\n"
        for problem in problems
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--min-verilog-files",
        type=int,
        default=1,
        help="Minimum number of .v/.sv submissions; use 0 for all problems with attachments.",
    )
    parser.add_argument(
        "--format",
        choices=("table", "jsonl"),
        default="table",
        help="table prints a compact TSV; jsonl includes complete problem text.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.min_verilog_files < 0:
        raise SystemExit("--min-verilog-files cannot be negative")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive")
    problems = load_problems(args.min_verilog_files, args.limit)
    content = (
        render_table(problems)
        if args.format == "table"
        else render_jsonl(problems)
    )

    if args.output is None:
        sys.stdout.write(content)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding="utf-8")
        print(f"output={args.output.resolve()}", file=sys.stderr)
    print(f"problems={len(problems)}", file=sys.stderr)


if __name__ == "__main__":
    main()
