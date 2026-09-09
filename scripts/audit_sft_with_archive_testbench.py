#!/usr/bin/env python3
# 使用 verify_repair_pairs_with_testbench.py 的原始判题实现重新审计 SFT 数据。
# 本脚本只负责从 messages 提取 buggy/fixed、缓存相同代码结果并保存全部 pair；
# testbench 映射、兼容清洗、iverilog/vvp 命令和输出比对均直接复用原验证脚本。
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import subprocess
import threading
import time

import verify_repair_pairs_with_testbench as verifier


ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def extract_code(content):
    blocks = re.findall(r"```(?:verilog|systemverilog|sv)?\s*\n(.*?)```", content, re.S | re.I)
    return (("\n\n".join(blocks) if blocks else content.strip()).strip() + "\n")


def load_rows(data_dir):
    rows = []
    for path in sorted(data_dir.glob("*.jsonl")):
        for line_number, line in enumerate(path.open(encoding="utf-8"), 1):
            row = json.loads(line)
            rows.append((path.stem, line_number, row))
    return rows


def summarize(records, total, started):
    by_split = defaultdict(Counter)
    categories = Counter()
    for row in records:
        categories[row["category"]] += 1
        by_split[row["split"]]["total"] += 1
        by_split[row["split"]][row["category"]] += 1
    return {
        "total": total,
        "completed": len(records),
        "complete": len(records) == total,
        "elapsed_seconds": round(time.monotonic() - started, 1),
        "categories": dict(categories),
        "by_split": dict(by_split),
    }


def main():
    parser = argparse.ArgumentParser(description="按原归档验证器审计 SFT train/dev/test")
    parser.add_argument("--data", type=Path, default=ROOT / "data/repair_pairs_sft")
    parser.add_argument("--catalog", type=Path, default=ROOT / "data/problem_catalog.jsonl")
    parser.add_argument("--archive", type=Path, default=ROOT / "testbench/co-problem-set-all-branches.tar.gz")
    parser.add_argument("--local-testbench-root", type=Path, default=ROOT / "testbench/generated/problem")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/archive_config_audit")
    parser.add_argument("--wall-timeout", type=int, default=10)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    output = args.output.resolve()
    files = sorted(args.data.glob("*.jsonl"))
    config = {
        "data_sha256": {str(path.resolve()): sha256(path) for path in files},
        "catalog_sha256": sha256(args.catalog),
        "archive_sha256": sha256(args.archive),
        "verifier_sha256": sha256(ROOT / "scripts/verify_repair_pairs_with_testbench.py"),
        "adapter_sha256": sha256(__file__),
        "wall_timeout": args.wall_timeout,
        "jobs": args.jobs,
        "compile_command": "iverilog -g2005 -Wall -o <executable> <submission> <timeout> <testbench_sources>",
        "simulation_command": "vvp <executable>",
        "iverilog_version": subprocess.run(
            ["iverilog", "-V"], capture_output=True, text=True
        ).stdout.splitlines()[0],
    }
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "run_config.json"
    if config_path.exists():
        if not args.resume or json.loads(config_path.read_text()) != config:
            parser.error("已有运行；续跑需 --resume 且数据、脚本、归档和配置完全一致")
    else:
        write_json(config_path, config)

    rows = load_rows(args.data)
    catalog = verifier.load_jsonl(args.catalog)
    archive_problems = verifier.load_archive_problems(args.archive)
    mappings = verifier.resolve_problem_mappings(catalog, archive_problems, 0.8, 0.02)
    problem_ids = {row["problem_id"] for _, _, row in rows}
    problem_names = {
        mapping["directory"]
        for problem_id, mapping in mappings.items()
        if problem_id in problem_ids and mapping["status"] == "mapped"
    }
    testbenches, locations = verifier.load_testbenches(args.archive, problem_names)
    local = verifier.load_local_testbenches(args.local_testbench_root, catalog)
    for cases in [*testbenches.values(), *local.values()]:
        for case in cases:
            case.timeout = min(case.timeout, args.wall_timeout)

    def cases_for(problem_id):
        mapping = mappings.get(problem_id, {"status": "not_found"})
        name = mapping.get("directory")
        local_cases = local.get(problem_id, [])
        if problem_id in verifier.LOCAL_TESTBENCH_OVERRIDES and local_cases:
            return local_cases, "local_override"
        if local_cases:
            return testbenches.get(name, []) + local_cases, "archive_plus_local"
        if mapping["status"] != "mapped":
            return [], mapping["status"]
        if len(locations.get(name, set())) > 1:
            return [], "ambiguous_testbench_branch"
        return testbenches.get(name, []), "archive"

    case_manifest = {
        str(pid): {"source": cases_for(pid)[1], "testcases": [case.name for case in cases_for(pid)[0]]}
        for pid in sorted(problem_ids)
    }
    write_json(output / "testcase_manifest.json", case_manifest)
    missing = [pid for pid, value in case_manifest.items() if not value["testcases"]]
    if missing:
        parser.error(f"缺少可验证 testbench: {missing}")

    cache_dir = output / "code_cache"
    cache_dir.mkdir(exist_ok=True)
    cache_locks = defaultdict(threading.Lock)

    def judge_code(problem_id, code):
        # 原始代码键配合互斥锁，避免并发审计相同提交时重复写缓存。
        key = hashlib.sha256((str(problem_id) + "\0" + code).encode()).hexdigest()
        folder = cache_dir / str(problem_id) / key
        result_path = folder / "result.json"
        with cache_locks[(problem_id, key)]:
            if result_path.exists():
                return json.loads(result_path.read_text())
            folder.mkdir(parents=True, exist_ok=True)
            source = folder / "submission.v"
            source.write_text(code)
            cases, _ = cases_for(problem_id)
            result = verifier.judge_submission(source, cases, problem_id)
            result["code_file"] = str(source.relative_to(output))
            write_json(result_path, result)
            return result

    def audit(row_info):
        split, line_number, row = row_info
        messages = row["messages"]
        buggy = extract_code(messages[1]["content"].split("Buggy Verilog code:", 1)[1])
        fixed = extract_code(messages[2]["content"])
        buggy_result = judge_code(row["problem_id"], buggy)
        fixed_result = judge_code(row["problem_id"], fixed)
        valid = buggy_result["status"] == "FAIL" and fixed_result["status"] == "AC"
        if valid:
            category = "valid_fail_to_ac"
        elif buggy_result["status"] == "AC" and fixed_result["status"] == "AC":
            category = "both_ac"
        elif buggy_result["status"] == "FAIL" and fixed_result["status"] == "FAIL":
            category = "both_fail"
        else:
            category = "buggy_ac_fixed_fail"
        return {
            "split": split,
            "line": line_number,
            "pair_id": row["pair_id"],
            "problem_id": row["problem_id"],
            "category": category,
            "valid_repair": valid,
            "buggy": buggy_result,
            "fixed": fixed_result,
        }

    started = time.monotonic()
    records = []
    results_path = output / "results.jsonl"
    completed = set()
    if args.resume and results_path.exists():
        records = [json.loads(line) for line in results_path.read_text().splitlines()]
        completed = {(row["split"], row["line"], row["pair_id"]) for row in records}
    mode = "a" if completed else "w"
    pending = [row for row in rows if (row[0], row[1], row[2]["pair_id"]) not in completed]
    with results_path.open(mode, encoding="utf-8") as stream:
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            for record in executor.map(audit, pending):
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                records.append(record)
                if len(records) % 100 == 0 or len(records) == len(rows):
                    summary = summarize(records, len(rows), started)
                    write_json(output / "summary.json", summary)
                    print(json.dumps({"completed": len(records), "categories": summary["categories"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
