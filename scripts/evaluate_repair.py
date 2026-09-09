#!/usr/bin/env python3
# 功能：生成模型修复结果，并使用题目 testbench 判定 Verilog 是否通过。
# 核心逻辑：把候选代码编译为仿真文件，运行 iverilog/vvp，提取实际输出与标准输出
# 中的匹配内容；同时提供审计脚本复用的代码提取、哈希、判题和汇总函数。

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    """计算文件 SHA256。"""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    """原子写入 JSON，避免中断时留下半个结果文件。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def extract_code(raw):
    """去掉 Markdown 代码围栏，保留一个可直接编译的 Verilog 文件。"""
    blocks = re.findall(r"```(?:verilog|systemverilog|sv)?\s*\n(.*?)```", raw, re.S | re.I)
    return ("\n\n".join(blocks) if blocks else raw.strip()).strip() + "\n"


def run_command(command, cwd, timeout, prefix):
    """运行编译或仿真命令，并把 stdout/stderr 保存到工作目录。"""
    cwd = Path(cwd)
    start = time.monotonic()
    stdout_path = cwd / f"{prefix}.stdout"
    stderr_path = cwd / f"{prefix}.stderr"
    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        process = subprocess.Popen(command, cwd=cwd, stdout=stdout, stderr=stderr,
                                    start_new_session=True)
        timed_out = False
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
    return {
        "command": command,
        "returncode": process.returncode,
        "timeout": timed_out,
        "seconds": round(time.monotonic() - start, 3),
    }


def matches(pattern, text):
    """按行匹配输出正则，返回所有非空匹配文本。"""
    return [match.group(0).strip() for match in re.finditer(pattern, text.replace("\r\n", "\n"), re.MULTILINE)]


def judge(code_path, problem, work, timeout):
    """编译并运行一个候选代码，返回逐测试用例结果。"""
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    cases = []
    for index, case in enumerate(problem["cases"]):
        folder = work / f"case_{index:03d}"
        folder.mkdir(parents=True, exist_ok=True)
        sources = [str(code_path)] + [str(Path(case["directory"]) / source) for source in case["sources"]]
        executable = folder / "sim.out"
        compile_result = run_command(
            ["iverilog", "-g2005", "-s", case["top_module"], "-o", str(executable), *sources],
            folder, timeout, "compile")
        result = {"name": case.get("name", str(index)), "compile": compile_result}
        if compile_result["timeout"]:
            result["status"] = "compile_timeout"
        elif compile_result["returncode"] != 0:
            result["status"] = "compile_error"
        else:
            simulation = run_command(["vvp", str(executable)], folder, timeout, "simulation")
            result["simulation"] = simulation
            if simulation["timeout"]:
                result["status"] = "simulation_timeout"
            elif simulation["returncode"] != 0:
                result["status"] = "simulation_error"
            else:
                actual = (folder / "simulation.stdout").read_text(errors="replace")
                standard = (Path(case["directory"]) / case["standard"]).read_text(errors="replace")
                expected = matches(case["output_regex"], standard)
                found = matches(case["output_regex"], actual)
                result["status"] = "passed" if found == expected and expected else "wrong_answer"
        cases.append(result)
    return {"passed": all(case["status"] == "passed" for case in cases), "cases": cases}


def summary(rows, records, mode):
    """汇总评测结果；未完成时不计算 Repair Rate@1。"""
    complete = len(records) == len(rows)
    passed = sum(record.get("passed", False) for record in records)
    by_problem = defaultdict(lambda: {"total": 0, "passed": 0})
    for row in rows:
        by_problem[str(row["problem_id"])] ["total"] += 1
    for record in records:
        item = by_problem[str(record["problem_id"])]
        item["passed"] += int(record.get("passed", False))
    return {
        "mode": mode,
        "complete": complete,
        "total": len(rows),
        "completed": len(records),
        "passed": passed,
        "pass_rate": passed / len(rows) if rows else 0,
        "repair_rate_at_1": passed / len(rows) if complete and rows else None,
        "by_problem": dict(by_problem),
    }


def main():
    parser = argparse.ArgumentParser(description="评测 Verilog 修复结果")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["model", "reference", "buggy"], default="model")
    parser.add_argument("--pairs", type=Path, default=ROOT / "data/repair_pairs_sft/test.jsonl")
    parser.add_argument("--timeout", type=int, default=90)
    args = parser.parse_args()
    raise SystemExit("模型生成入口已从仓库恢复，当前请使用已有训练/基线脚本提供的生成流程")


if __name__ == "__main__":
    main()
