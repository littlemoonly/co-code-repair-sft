# 功能：用题目 testbench 验证 repair pair，只有 buggy=FAIL 且 fixed=AC 的 pair 才会保留。
# 核心逻辑：优先按题目 ID 读取本地新增测试，否则从归档映射测试，再用 iverilog/vvp 编译仿真并比对输出。

import argparse
import io
import json
import re
import subprocess
import tarfile
import tempfile
import zipfile
import yaml
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path


LOCAL_TESTBENCH_OVERRIDES = {326}


@dataclass
class Testcase:
    name: str
    files: dict[str, bytes]
    standard: str
    output_regex: str | None
    timeout: int
    simulating_time: str


def parse_args():
    parser = argparse.ArgumentParser(description="Verify FAIL-to-AC repair pairs.")
    parser.add_argument(
        "--pairs",
        type=Path,
        default=Path("data/repair_pairs_raw/metadata/pairs.jsonl"),
    )
    parser.add_argument(
        "--files-root",
        type=Path,
        default=Path("data/repair_pairs_raw/files"),
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("data/problem_catalog.jsonl"),
    )
    parser.add_argument(
        "--testbench-archive",
        type=Path,
        default=Path("testbench/co-problem-set-all-branches.tar.gz"),
    )
    parser.add_argument(
        "--local-testbench-root",
        type=Path,
        default=Path("testbench/generated/problem"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/repair_pairs_raw/metadata/pairs_judge_verified.jsonl"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("data/repair_pairs_raw/metadata/judge_verification.jsonl"),
    )
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--wall-timeout", type=int, default=10)
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--problem-ids", type=int, nargs="+")
    parser.add_argument("--exact-name-only", action="store_true")
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--semantic-threshold", type=float, default=0.8)
    parser.add_argument("--semantic-margin", type=float, default=0.02)
    return parser.parse_args()


def load_jsonl(path):
    with path.open(encoding="utf-8") as file_obj:
        return [json.loads(line) for line in file_obj if line.strip()]


def decode_text(content):
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        return content.decode("gb18030", errors="replace")


def find_zip_file(files, filename):
    for name, content in files.items():
        if Path(name).name.lower() == filename:
            return content
    return None


def parse_testcase(member_name, content, yaml_content):
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        files = {
            name: archive.read(name)
            for name in archive.namelist()
            if not name.endswith("/")
        }

    argv_content = find_zip_file(files, "argv.txt")
    standard_content = find_zip_file(files, "standard.txt")
    verilog_files = {
        Path(name).name: data
        for name, data in files.items()
        if Path(name).suffix.lower() in {".v", ".sv"}
    }
    if standard_content is None or not verilog_files:
        return None

    if argv_content is not None:
        config = json.loads(decode_text(argv_content))
        universal = config.get("universal", {})
    else:
        config = yaml.safe_load(decode_text(yaml_content)) or {}
        universal = (config.get("judge_parameter") or {}).get("universal", {})
    return Testcase(
        name=Path(member_name).parents[1].name,
        files={Path(name).name: data for name, data in files.items()},
        standard=decode_text(standard_content).replace("\r\n", "\n").replace("\r", "\n"),
        output_regex=universal.get("output_regex"),
        timeout=int(universal.get("timeout", 90)),
        simulating_time=universal.get("simulating_time", "1000ns"),
    )


def normalize_text(text):
    return re.sub(r"[^0-9a-z\u3400-\u9fff]+", "", text.lower())


def description_similarity(left, right):
    left = normalize_text(left)
    right = normalize_text(right)
    if not left or not right:
        return 0.0
    left_grams = {left[index : index + 5] for index in range(len(left) - 4)}
    right_grams = {right[index : index + 5] for index in range(len(right) - 4)}
    if not left_grams or not right_grams:
        return 0.0
    return len(left_grams & right_grams) / min(len(left_grams), len(right_grams))


def load_archive_problems(archive_path):
    problems = {}
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        testcase_locations = {
            (Path(member.name).parts[0], Path(member.name).parts[2])
            for member in members
            if member.isfile()
            and len(Path(member.name).parts) >= 7
            and Path(member.name).parts[1] == "problem"
            and Path(member.name).parts[-2] == "attachment"
            and Path(member.name).suffix.lower() == ".zip"
        }
        for member in members:
            parts = Path(member.name).parts
            location = (parts[0], parts[2]) if len(parts) >= 3 else None
            if (
                member.isfile()
                and len(parts) == 4
                and parts[1] == "problem"
                and parts[-1] == "description.md"
                and location in testcase_locations
            ):
                problems[location] = decode_text(archive.extractfile(member).read())
        for location in testcase_locations:
            problems.setdefault(location, "")
    return problems


def resolve_problem_mappings(catalog_rows, archive_problems, threshold, margin):
    mappings = {}
    for row in catalog_rows:
        exact = [location for location in archive_problems if location[1] == row["name"]]
        if len(exact) == 1:
            branch, directory = exact[0]
            mappings[row["id"]] = {
                "status": "mapped",
                "method": "exact_name",
                "branch": branch,
                "directory": directory,
            }
            continue
        if len(exact) > 1:
            mappings[row["id"]] = {"status": "ambiguous"}
            continue

        description = row.get("description") or ""
        candidates = []
        for (branch, directory), archive_description in archive_problems.items():
            description_score = description_similarity(description, archive_description)
            name_score = SequenceMatcher(
                None, normalize_text(row["name"]), normalize_text(directory)
            ).ratio()
            combined_score = 0.9 * description_score + 0.1 * name_score
            candidates.append(
                (
                    combined_score,
                    description_score,
                    branch,
                    directory,
                    normalize_text(archive_description),
                )
            )
        candidates.sort(reverse=True)
        best = candidates[0] if candidates else None
        distinct_candidates = (
            [candidate for candidate in candidates[1:] if candidate[4] != best[4]]
            if best
            else []
        )
        second_score = distinct_candidates[0][0] if distinct_candidates else 0.0

        if (
            best
            and best[1] >= threshold
            and best[0] - second_score >= margin
        ):
            mappings[row["id"]] = {
                "status": "mapped",
                "method": "semantic_description",
                "branch": best[2],
                "directory": best[3],
                "description_similarity": round(best[1], 6),
            }
        else:
            mappings[row["id"]] = {"status": "not_found"}
    return mappings


def load_testbenches(archive_path, problem_names):
    locations = {}
    members_by_problem = {}

    with tarfile.open(archive_path, "r:gz") as archive:
        zip_members = []
        members = archive.getmembers()
        yaml_members = {
            member.name: member
            for member in members
            if member.isfile() and Path(member.name).name == "testcase.yaml"
        }
        for member in members:
            parts = Path(member.name).parts
            if (
                member.isfile()
                and len(parts) >= 7
                and parts[1] == "problem"
                and parts[2] in problem_names
                and parts[-2] == "attachment"
                and parts[-1].lower().endswith(".zip")
            ):
                problem_name = parts[2]
                locations.setdefault(problem_name, set()).add(parts[0])
                zip_members.append((problem_name, member))

        unique_names = {
            name for name, branches in locations.items() if len(branches) == 1
        }
        for problem_name, member in zip_members:
            if problem_name not in unique_names:
                continue
            yaml_name = str(Path(member.name).parents[1] / "testcase.yaml")
            yaml_member = yaml_members.get(yaml_name)
            yaml_content = archive.extractfile(yaml_member).read() if yaml_member else b"{}"
            testcase = parse_testcase(
                member.name,
                archive.extractfile(member).read(),
                yaml_content,
            )
            if testcase is not None:
                members_by_problem.setdefault(problem_name, []).append(testcase)

    return members_by_problem, locations


def load_local_testbenches(root, catalog_rows):
    testbenches = {}
    names_to_ids = {row["name"]: row["id"] for row in catalog_rows}
    if not root.exists():
        return testbenches

    for problem_dir in root.iterdir():
        problem_id = names_to_ids.get(problem_dir.name)
        if problem_id is None:
            continue
        for zip_path in problem_dir.glob("testcase/*/attachment/*.zip"):
            yaml_path = zip_path.parents[1] / "testcase.yaml"
            yaml_content = yaml_path.read_bytes() if yaml_path.exists() else b"{}"
            testcase = parse_testcase(
                str(zip_path),
                zip_path.read_bytes(),
                yaml_content,
            )
            if testcase is not None:
                testbenches.setdefault(problem_id, []).append(testcase)
    return testbenches


def normalized_output(text, output_regex):
    if output_regex:
        return [
            " ".join(match.group(0).split())
            for match in re.finditer(output_regex, text, re.MULTILINE)
        ]
    return [line.strip() for line in text.splitlines() if line.strip()]


def simulation_time_ns(value):
    match = re.fullmatch(r"([0-9.]+)\s*(s|ms|us|ns|ps|fs)", value)
    number = float(match.group(1))
    factors = {"s": 1e9, "ms": 1e6, "us": 1e3, "ns": 1, "ps": 1e-3, "fs": 1e-6}
    return number * factors[match.group(2)]


def compatible_submission(code_path, problem_id):
    code = code_path.read_text(encoding="utf-8")
    if problem_id == 24:
        code = re.sub(r"\bmodule\s+string\b", "module expr", code, count=1)
    return re.sub(r";[ \t]*;", ";", code)


def run_testcase(code_path, testcase, problem_id):
    with tempfile.TemporaryDirectory() as temp_dir:
        work_dir = Path(temp_dir)
        submission_path = work_dir / "submission.v"
        submission_path.write_text(
            compatible_submission(code_path, problem_id),
            encoding="utf-8",
        )

        timeout_path = work_dir / "judge_timeout.v"
        timeout_path.write_text(
            "`timescale 1ns / 1ps\n"
            "module __judge_timeout;\n"
            f"initial begin #{simulation_time_ns(testcase.simulating_time)}; $finish; end\n"
            "endmodule\n",
            encoding="utf-8",
        )
        source_paths = [submission_path, timeout_path]
        for filename, content in testcase.files.items():
            path = work_dir / filename
            path.write_bytes(content)
            if path.suffix.lower() in {".v", ".sv"}:
                source_paths.append(path)

        executable = work_dir / "simulation.out"
        compile_result = subprocess.run(
            ["iverilog", "-g2005", "-Wall", "-o", str(executable), *map(str, source_paths)],
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if compile_result.returncode != 0:
            return False, "compile", compile_result.stderr[-1000:]
        if re.search(r"warning: Port \d+ .* expects \d+ bits, got \d+", compile_result.stderr):
            return False, "interface", compile_result.stderr[-1000:]

        try:
            simulation = subprocess.run(
                ["vvp", str(executable)],
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=testcase.timeout,
            )
        except subprocess.TimeoutExpired:
            return False, "timeout", f"simulation exceeded {testcase.timeout}s"

        if simulation.returncode != 0:
            return False, "simulation", simulation.stderr[-1000:]

        actual = normalized_output(simulation.stdout, testcase.output_regex)
        expected = normalized_output(testcase.standard, testcase.output_regex)
        if actual != expected:
            detail = f"expected={expected[:20]!r}, actual={actual[:20]!r}"
            return False, "wrong_answer", detail
        return True, "passed", ""


def judge_submission(code_path, testcases, problem_id):
    passed = 0
    for testcase in testcases:
        success, stage, detail = run_testcase(code_path, testcase, problem_id)
        if not success:
            return {
                "status": "FAIL",
                "passed_testcases": passed,
                "total_testcases": len(testcases),
                "failed_testcase": testcase.name,
                "failure_stage": stage,
                "detail": detail,
            }
        passed += 1
    return {
        "status": "AC",
        "passed_testcases": passed,
        "total_testcases": len(testcases),
    }


def verify_pair(pair, files_root, mappings, testbenches, locations, local_testbenches):
    problem_id = pair["problem_id"]
    mapping = mappings.get(problem_id, {"status": "not_found"})
    problem_name = mapping.get("directory")
    local_testcases = local_testbenches.get(problem_id, [])
    if problem_id in LOCAL_TESTBENCH_OVERRIDES and local_testcases:
        testcases = local_testcases
    else:
        testcases = testbenches.get(problem_name, []) + local_testcases

    if local_testcases:
        buggy = judge_submission(files_root / pair["buggy_path"], testcases, problem_id)
        fixed = judge_submission(files_root / pair["fixed_path"], testcases, problem_id)
        valid_repair = buggy["status"] == "FAIL" and fixed["status"] == "AC"
        return pair, {
            "status": "VERIFIED",
            "problem_name": problem_name or f"problem_{problem_id}",
            "mapping": {"status": "mapped", "method": "local_problem_id"},
            "valid_repair": valid_repair,
            "buggy": buggy,
            "fixed": fixed,
        }

    if mapping["status"] == "ambiguous":
        return pair, {"status": "UNVERIFIED", "reason": "ambiguous_problem_mapping"}
    if mapping["status"] != "mapped":
        return pair, {"status": "UNVERIFIED", "reason": "testbench_not_found"}
    if len(locations.get(problem_name, set())) > 1:
        return pair, {"status": "UNVERIFIED", "reason": "ambiguous_testbench_branch"}
    if not testcases:
        return pair, {"status": "UNVERIFIED", "reason": "testbench_not_found"}

    buggy = judge_submission(files_root / pair["buggy_path"], testcases, problem_id)
    fixed = judge_submission(files_root / pair["fixed_path"], testcases, problem_id)
    valid_repair = buggy["status"] == "FAIL" and fixed["status"] == "AC"
    report = {
        "status": "VERIFIED",
        "problem_name": problem_name,
        "mapping": mapping,
        "valid_repair": valid_repair,
        "buggy": buggy,
        "fixed": fixed,
    }
    return pair, report


def main():
    args = parse_args()
    pairs = load_jsonl(args.pairs)
    if args.problem_ids:
        problem_ids = set(args.problem_ids)
        pairs = [pair for pair in pairs if pair["problem_id"] in problem_ids]

    catalog_rows = load_jsonl(args.catalog)
    archive_problems = load_archive_problems(args.testbench_archive)
    mappings = resolve_problem_mappings(
        catalog_rows,
        archive_problems,
        args.semantic_threshold,
        args.semantic_margin,
    )
    if args.exact_name_only:
        pairs = [
            pair
            for pair in pairs
            if mappings.get(pair["problem_id"], {}).get("method") == "exact_name"
        ]
    if args.max_pairs is not None:
        pairs = pairs[: args.max_pairs]
    problem_ids = {pair["problem_id"] for pair in pairs}
    problem_names = {
        mapping["directory"]
        for problem_id, mapping in mappings.items()
        if problem_id in problem_ids and mapping["status"] == "mapped"
    }
    testbenches, locations = load_testbenches(args.testbench_archive, problem_names)
    local_testbenches = load_local_testbenches(args.local_testbench_root, catalog_rows)
    for testcases in [*testbenches.values(), *local_testbenches.values()]:
        for testcase in testcases:
            testcase.timeout = min(testcase.timeout, args.wall_timeout)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    counts = {"valid": 0, "invalid": 0, "unverified": 0}
    existing_pair_ids = (
        {row["pair_id"] for row in load_jsonl(args.output)}
        if args.append and args.output.exists()
        else set()
    )

    def verify(pair):
        return verify_pair(
            pair,
            args.files_root,
            mappings,
            testbenches,
            locations,
            local_testbenches,
        )

    with (
        args.output.open("a" if args.append else "w", encoding="utf-8") as output_file,
        args.report.open("a" if args.append else "w", encoding="utf-8") as report_file,
        ThreadPoolExecutor(max_workers=args.jobs) as executor,
    ):
        for index, (pair, report) in enumerate(executor.map(verify, pairs), 1):
            report_row = {"pair_id": pair["pair_id"], "problem_id": pair["problem_id"], **report}
            report_file.write(json.dumps(report_row, ensure_ascii=False) + "\n")

            if report.get("valid_repair"):
                if pair["pair_id"] not in existing_pair_ids:
                    verified_pair = {**pair, "verification": "testbench_verified"}
                    output_file.write(json.dumps(verified_pair, ensure_ascii=False) + "\n")
                    existing_pair_ids.add(pair["pair_id"])
                counts["valid"] += 1
            elif report["status"] == "UNVERIFIED":
                counts["unverified"] += 1
            else:
                counts["invalid"] += 1

            if index % 100 == 0:
                print(f"processed: {index}/{len(pairs)}")

    print(f"total: {len(pairs)}")
    print(f"valid: {counts['valid']}")
    print(f"invalid: {counts['invalid']}")
    print(f"unverified: {counts['unverified']}")


if __name__ == "__main__":
    main()
