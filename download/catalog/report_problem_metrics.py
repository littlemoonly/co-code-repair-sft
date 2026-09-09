# 功能：统计题目目录中各题目的训练数据指标。
# 核心逻辑：查询提交、修复候选和测试点数量，与题目目录合并后输出 TSV。

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..common import config


METRICS_SQL = r"""
WITH source_submissions AS (
    -- 只统计目录内题目的 Verilog/SystemVerilog 提交。
    SELECT
        r.id,
        r.problem_id,
        r.edx_username,
        r.course_code,
        r.judge_result,
        r.submitted_at
    FROM judge_problemjudgerecord r
    JOIN judge_usersubmittedfile f ON f.id = r.attachment_id
    WHERE r.problem_id = ANY(%s)
      AND lower(f.filename) LIKE ANY(ARRAY['%%.v', '%%.sv'])
),
ordered_submissions AS (
    -- 对同一学生、同一题目、同一课程的提交按时间排序，
    -- previous_result 表示紧邻的上一次提交结果。
    SELECT
        *,
        lag(judge_result) OVER (
            PARTITION BY edx_username, problem_id, course_code
            ORDER BY submitted_at, id
        ) AS previous_result
    FROM source_submissions
),
submission_stats AS (
    SELECT
        problem_id,
        count(*) AS source_submissions,
        count(DISTINCT edx_username) AS students,
        count(*) FILTER (WHERE judge_result = 0) AS result_0,
        count(*) FILTER (WHERE judge_result = 1) AS result_1,
        count(*) FILTER (WHERE judge_result NOT IN (0, 1)) AS other_results,
        count(DISTINCT course_code) FILTER (
            WHERE course_code IS NOT NULL AND course_code <> ''
        ) AS courses,
        min(submitted_at) AS first_submission,
        max(submitted_at) AS last_submission
    FROM source_submissions
    GROUP BY problem_id
),
repair_stats AS (
    -- 当前数据库中 1 表示失败、0 表示通过。
    -- 同一个学生/题目/课程即使多次出现 1 -> 0，也只计为一个候选组；
    -- 后续构造数据时可从该组中选择最后一次相邻的失败和通过代码。
    SELECT
        problem_id,
        count(DISTINCT (edx_username, COALESCE(course_code, '')))
            AS repair_candidate_groups
    FROM ordered_submissions
    WHERE previous_result = 1 AND judge_result = 0
    GROUP BY problem_id
),
testcase_stats AS (
    SELECT problem_id, count(*) AS test_cases
    FROM judge_problem_test_cases
    WHERE problem_id = ANY(%s)
    GROUP BY problem_id
)
SELECT
    s.problem_id,
    s.source_submissions,
    s.students,
    s.result_0,
    s.result_1,
    s.other_results,
    COALESCE(r.repair_candidate_groups, 0) AS repair_candidate_groups,
    COALESCE(t.test_cases, 0) AS test_cases,
    s.courses,
    s.first_submission,
    s.last_submission
FROM submission_stats s
LEFT JOIN repair_stats r ON r.problem_id = s.problem_id
LEFT JOIN testcase_stats t ON t.problem_id = s.problem_id
ORDER BY s.source_submissions DESC, s.problem_id
"""


FIELDS = (
    "problem_id",
    "name",
    "has_description",
    "description_chars",
    "source_submissions",
    "students",
    "result_0",
    "result_1",
    "other_results",
    "result_0_rate",
    "repair_candidate_groups",
    "test_cases",
    "ready_for_pairing",
    "courses",
    "first_submission",
    "last_submission",
)


def load_catalog(path: Path) -> dict[int, dict]:
    with path.open(encoding="utf-8") as file_obj:
        rows = [json.loads(line) for line in file_obj if line.strip()]
    return {row["id"]: row for row in rows}


def query_metrics(problem_ids: list[int]) -> list[dict]:
    cursor = config.connect_db()
    connection = cursor.connection
    cursor.execute(METRICS_SQL, (problem_ids, problem_ids))
    fields = [column.name for column in cursor.description]
    rows = [dict(zip(fields, row)) for row in cursor]
    cursor.close()
    connection.close()
    return rows


def build_rows(catalog: dict[int, dict], metrics: list[dict]) -> list[dict]:
    rows = []
    for metric in metrics:
        problem = catalog[metric["problem_id"]]
        description = problem.get("description") or ""
        total = metric["source_submissions"]
        metric.update(
            name=problem["name"],
            has_description=bool(description),
            description_chars=len(description),
            result_0_rate=round(metric["result_0"] / total, 4),
            # 这里只检查题面、标签和测试点等基础条件。
            # True 不代表代码对已经通过编译和仿真验证。
            ready_for_pairing=(
                bool(description)
                and metric["result_0"] > 0
                and metric["result_1"] > 0
                and metric["repair_candidate_groups"] > 0
                and metric["test_cases"] > 0
            ),
        )
        rows.append(metric)
    return rows


def write_tsv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        file_obj.write("\t".join(FIELDS) + "\n")
        for row in rows:
            values = []
            for field in FIELDS:
                value = row[field]
                if hasattr(value, "isoformat"):
                    value = value.isoformat()
                values.append(str(value).replace("\t", " ").replace("\n", " "))
            file_obj.write("\t".join(values) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=project_root / "data" / "problem_catalog.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "data" / "problem_metrics.tsv",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    catalog = load_catalog(args.catalog)
    rows = build_rows(catalog, query_metrics(sorted(catalog)))
    write_tsv(args.output, rows)
    print(f"problems={len(rows)}")
    print(f"source_submissions={sum(row['source_submissions'] for row in rows)}")
    print(
        "repair_candidate_groups="
        f"{sum(row['repair_candidate_groups'] for row in rows)}"
    )
    print(f"output={args.output.resolve()}")


if __name__ == "__main__":
    main()
