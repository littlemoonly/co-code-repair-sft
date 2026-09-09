#!/usr/bin/env python3
"""Run the untuned Qwen baseline and judge it with the archive audit testbench."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import audit_sft_with_archive_testbench as audit
import verify_repair_pairs_with_testbench as verifier


ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def model_manifest(model_dir):
    result = {}
    for path in sorted(model_dir.glob("*")):
        if not path.is_file():
            continue
        # Hash configuration/tokenizer files. Avoid reading all 15 GB of weights on every resume.
        result[path.name] = (
            {"bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
            if path.suffix == ".safetensors"
            else {"sha256": sha256(path)}
        )
    return result


def make_judge(rows, args):
    catalog = verifier.load_jsonl(args.catalog)
    archive_problems = verifier.load_archive_problems(args.archive)
    mappings = verifier.resolve_problem_mappings(catalog, archive_problems, 0.8, 0.02)
    problem_ids = {row["problem_id"] for row in rows}
    problem_names = {
        mapping["directory"]
        for problem_id, mapping in mappings.items()
        if problem_id in problem_ids and mapping["status"] == "mapped"
    }
    archive_cases, locations = verifier.load_testbenches(args.archive, problem_names)
    local_cases = verifier.load_local_testbenches(args.local_testbench_root, catalog)
    for cases in [*archive_cases.values(), *local_cases.values()]:
        for case in cases:
            case.timeout = min(case.timeout, args.wall_timeout)

    def cases_for(problem_id):
        mapping = mappings.get(problem_id, {"status": "not_found"})
        directory = mapping.get("directory")
        added = local_cases.get(problem_id, [])
        if problem_id in verifier.LOCAL_TESTBENCH_OVERRIDES and added:
            return added, "local_override"
        if added:
            return archive_cases.get(directory, []) + added, "archive_plus_local"
        if mapping["status"] != "mapped":
            return [], mapping["status"]
        if len(locations.get(directory, set())) > 1:
            return [], "ambiguous_testbench_branch"
        return archive_cases.get(directory, []), "archive"

    manifest = {
        str(problem_id): {
            "source": cases_for(problem_id)[1],
            "testcases": [case.name for case in cases_for(problem_id)[0]],
        }
        for problem_id in sorted(problem_ids)
    }
    missing = [problem_id for problem_id, item in manifest.items() if not item["testcases"]]
    if missing:
        raise ValueError(f"缺少可验证 testbench: {missing}")

    def judge(problem_id, code_path):
        cases, _ = cases_for(problem_id)
        return verifier.judge_submission(code_path, cases, problem_id)

    return manifest, judge


def summarize(rows, records, started):
    status_counts = Counter(record["test_result"]["status"] for record in records)
    by_problem = defaultdict(lambda: {"total": 0, "completed": 0, "repaired": 0})
    for row in rows:
        by_problem[str(row["problem_id"])]["total"] += 1
    for record in records:
        group = by_problem[str(record["problem_id"])]
        group["completed"] += 1
        group["repaired"] += record["test_result"]["status"] == "AC"
    for group in by_problem.values():
        group["repair_rate"] = group["repaired"] / group["total"]
    complete = len(records) == len(rows)
    repaired = status_counts["AC"]
    return {
        "total": len(rows),
        "completed": len(records),
        "complete": complete,
        "repaired": repaired,
        "repair_rate": repaired / len(rows) if complete else None,
        "running_repair_rate": repaired / len(records) if records else 0.0,
        "status_counts": dict(status_counts),
        "elapsed_seconds": round(time.monotonic() - started, 1),
        "by_problem": dict(by_problem),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="使用未经微调的 Qwen2.5-Coder-7B-Instruct 生成，并按 archive audit 标准计算 Repair Rate@1。"
    )
    parser.add_argument("--test", type=Path, default=ROOT / "data/repair_pairs_sft/test.jsonl")
    parser.add_argument("--model", type=Path, default=ROOT / "models/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--catalog", type=Path, default=ROOT / "data/problem_catalog.jsonl")
    parser.add_argument("--archive", type=Path, default=ROOT / "testbench/co-problem-set-all-branches.tar.gz")
    parser.add_argument("--local-testbench-root", type=Path, default=ROOT / "testbench/generated/problem")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/qwen25_coder_7b_baseline_archive")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--judge-jobs", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wall-timeout", type=int, default=10)
    parser.add_argument("--limit", type=int, help="仅供冒烟测试；正式 baseline 不要设置")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size < 1 or args.judge_jobs < 1 or args.max_new_tokens < 1:
        raise SystemExit("batch-size、judge-jobs 和 max-new-tokens 必须为正数")
    for binary in ("iverilog", "vvp"):
        if not shutil.which(binary):
            raise SystemExit(f"缺少 {binary}，请先安装 Icarus Verilog")

    rows = load_jsonl(args.test)
    if args.limit is not None:
        if args.limit < 1:
            raise SystemExit("--limit 必须为正数")
        rows = rows[: args.limit]
    if not rows or len({row["pair_id"] for row in rows}) != len(rows):
        raise SystemExit("test set 为空或 pair_id 重复")
    for row in rows:
        if [message["role"] for message in row["messages"]] != ["system", "user", "assistant"]:
            raise SystemExit(f"messages 结构错误: {row['pair_id']}")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest, judge = make_judge(rows, args)
    write_json(output / "testcase_manifest.json", manifest)

    versions = {}
    for package in ("torch", "transformers"):
        versions[package] = importlib.metadata.version(package)
    generation_settings = {
        "do_sample": False,
        "num_beams": 1,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "dtype": "bfloat16",
        "batch_size": args.batch_size,
    }
    config = {
        "test": str(args.test.resolve()),
        "test_sha256": sha256(args.test),
        "pair_ids": [row["pair_id"] for row in rows],
        "catalog_sha256": sha256(args.catalog),
        "archive_sha256": sha256(args.archive),
        "verifier_sha256": sha256(ROOT / "scripts/verify_repair_pairs_with_testbench.py"),
        "audit_adapter_sha256": sha256(ROOT / "scripts/audit_sft_with_archive_testbench.py"),
        "runner_sha256": sha256(__file__),
        "model": str(args.model.resolve()),
        "model_files": model_manifest(args.model),
        "generation": generation_settings,
        "wall_timeout": args.wall_timeout,
        "judge_jobs": args.judge_jobs,
        "versions": versions,
        "iverilog_version": subprocess.run(
            ["iverilog", "-V"], capture_output=True, text=True
        ).stdout.splitlines()[0],
    }
    config_path = output / "run_config.json"
    if config_path.exists():
        if not args.resume or json.loads(config_path.read_text()) != config:
            raise SystemExit("输出目录已有不同运行；续跑需使用 --resume 且所有配置完全一致")
    else:
        write_json(config_path, config)

    results_path = output / "results.jsonl"
    records = load_jsonl(results_path) if args.resume and results_path.exists() else []
    if len({record["pair_id"] for record in records}) != len(records):
        raise SystemExit("已有 results.jsonl 包含重复 pair_id")
    done = {record["pair_id"] for record in records}
    expected_ids = set(config["pair_ids"])
    if not done <= expected_ids:
        raise SystemExit("已有 results.jsonl 含当前 test set 之外的 pair_id")

    pending = [(index, row) for index, row in enumerate(rows) if row["pair_id"] not in done]
    started = time.monotonic()
    if pending:
        # OMP_NUM_THREADS=0 makes libgomp complain and is never useful here.
        if os.environ.get("OMP_NUM_THREADS") == "0":
            os.environ["OMP_NUM_THREADS"] = "1"
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

        if not torch.cuda.is_available():
            raise SystemExit("未检测到 CUDA GPU")
        set_seed(args.seed)
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, local_files_only=True
        ).to("cuda").eval()
        eos_ids = model.generation_config.eos_token_id
        eos_ids = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids or [])

        mode = "a" if records else "w"
        with results_path.open(mode, encoding="utf-8") as stream:
            for offset in range(0, len(pending), args.batch_size):
                batch = pending[offset : offset + args.batch_size]
                prompts = [
                    tokenizer.apply_chat_template(
                        row["messages"][:2], tokenize=False, add_generation_prompt=True
                    )
                    for _, row in batch
                ]
                encoded = tokenizer(prompts, add_special_tokens=False, padding=True, return_tensors="pt")
                input_tokens = encoded["attention_mask"].sum(dim=1).tolist()
                padded_length = encoded["input_ids"].shape[1]
                if padded_length + args.max_new_tokens > model.config.max_position_embeddings:
                    raise ValueError(f"batch 中 prompt 超过上下文限制: {[row['pair_id'] for _, row in batch]}")
                encoded = encoded.to("cuda")
                generation_started = time.monotonic()
                with torch.inference_mode():
                    generated = model.generate(
                        **encoded,
                        do_sample=False,
                        num_beams=1,
                        max_new_tokens=args.max_new_tokens,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=model.generation_config.eos_token_id,
                    )[:, padded_length:]
                generation_seconds = time.monotonic() - generation_started

                generated_items = []
                for item, prompt, prompt_tokens, token_tensor in zip(batch, prompts, input_tokens, generated):
                    index, row = item
                    token_ids = token_tensor.tolist()
                    output_length = next(
                        (position + 1 for position, token_id in enumerate(token_ids) if token_id in eos_ids),
                        len(token_ids),
                    )
                    raw_output = tokenizer.decode(token_ids[:output_length], skip_special_tokens=True)
                    sample = output / "samples" / f"{index:05d}_{row['pair_id']}"
                    sample.mkdir(parents=True, exist_ok=True)
                    write_json(sample / "prompt_messages.json", row["messages"][:2])
                    (sample / "prompt.txt").write_text(prompt, encoding="utf-8")
                    (sample / "model_output.txt").write_text(raw_output, encoding="utf-8")
                    code_path = sample / "repair.v"
                    code_path.write_text(audit.extract_code(raw_output), encoding="utf-8")
                    generation_record = {
                        "pair_id": row["pair_id"],
                        "problem_id": row["problem_id"],
                        "model_output": raw_output,
                        "input_tokens": int(prompt_tokens),
                        "output_tokens": output_length,
                        "hit_max_new_tokens": output_length == args.max_new_tokens,
                        "batch_generation_seconds": round(generation_seconds, 3),
                        "code_file": str(code_path.relative_to(output)),
                    }
                    write_json(sample / "generation.json", generation_record)
                    generated_items.append((index, row, sample, code_path, generation_record))

                with ThreadPoolExecutor(max_workers=args.judge_jobs) as executor:
                    outcomes = list(
                        executor.map(lambda item: judge(item[1]["problem_id"], item[3]), generated_items)
                    )
                for item, outcome in zip(generated_items, outcomes):
                    index, row, sample, _, generation_record = item
                    record = {**generation_record, "test_result": outcome}
                    write_json(sample / "result.json", record)
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stream.flush()
                    records.append(record)
                    write_json(output / "summary.json", summarize(rows, records, started))
                    print(
                        f"[{len(records)}/{len(rows)}] {row['pair_id']}: {outcome['status']}",
                        flush=True,
                    )

    order = {pair_id: index for index, pair_id in enumerate(config["pair_ids"])}
    records.sort(key=lambda record: order[record["pair_id"]])
    results_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    report = summarize(rows, records, started)
    write_json(output / "summary.json", report)
    print(json.dumps({key: value for key, value in report.items() if key != "by_problem"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
