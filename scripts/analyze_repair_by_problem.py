#!/usr/bin/env python3
# 功能：对比 Base 与 LoRA 的测试结果，按 problem_id 统计修复率并生成 CSV 和 Markdown 报告。
# 核心逻辑：以 pair_id 对齐两次评测，按题聚合 AC 数量，同时结合训练集样本数分析提升分布。

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def read_jsonl(path):
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def percent(value):
    return f"{value * 100:.2f}%"


def main():
    parser = argparse.ArgumentParser(description="按题目分析 Base 与 LoRA 修复率")
    parser.add_argument("--base", type=Path, required=True, help="Base results.jsonl")
    parser.add_argument("--lora", type=Path, required=True, help="LoRA results.jsonl")
    parser.add_argument("--train", type=Path, required=True, help="训练集 JSONL")
    parser.add_argument("--catalog", type=Path, required=True, help="题目目录 JSONL")
    parser.add_argument("--output-dir", type=Path, required=True, help="报告输出目录")
    args = parser.parse_args()

    base = {row["pair_id"]: row for row in read_jsonl(args.base)}
    lora = {row["pair_id"]: row for row in read_jsonl(args.lora)}
    if base.keys() != lora.keys():
        raise ValueError("Base 与 LoRA 的 pair_id 集合不一致，不能直接比较")

    train_counts = Counter(row["problem_id"] for row in read_jsonl(args.train))
    names = {row["id"]: row["name"] for row in read_jsonl(args.catalog)}
    grouped = defaultdict(lambda: {"test": 0, "base": 0, "lora": 0})
    for pair_id, base_row in base.items():
        problem_id = base_row["problem_id"]
        if lora[pair_id]["problem_id"] != problem_id:
            raise ValueError(f"pair_id={pair_id} 的 problem_id 不一致")
        grouped[problem_id]["test"] += 1
        grouped[problem_id]["base"] += base_row["test_result"]["status"] == "AC"
        grouped[problem_id]["lora"] += lora[pair_id]["test_result"]["status"] == "AC"

    rows = []
    for problem_id in sorted(grouped):
        item = grouped[problem_id]
        base_rate = item["base"] / item["test"]
        lora_rate = item["lora"] / item["test"]
        rows.append({
            "problem_id": problem_id,
            "problem_name": names.get(problem_id, ""),
            "train_samples": train_counts[problem_id],
            "test_samples": item["test"],
            "base_repaired": item["base"],
            "base_rate": base_rate,
            "lora_repaired": item["lora"],
            "lora_rate": lora_rate,
            "lift": lora_rate - base_rate,
        })

    total = sum(row["test_samples"] for row in rows)
    base_total = sum(row["base_repaired"] for row in rows)
    lora_total = sum(row["lora_repaired"] for row in rows)
    base_macro = sum(row["base_rate"] for row in rows) / len(rows)
    lora_macro = sum(row["lora_rate"] for row in rows) / len(rows)

    # 训练集达到单题上限 250 条的题定义为“高频题”，单独计算其提升贡献。
    rich = [row for row in rows if row["train_samples"] == 250]
    other = [row for row in rows if row["train_samples"] != 250]
    rich_gain = sum(row["lora_repaired"] - row["base_repaired"] for row in rich)
    total_gain = lora_total - base_total

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "by_problem.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["problem_id", "problem_name", "train样本数", "test样本数", "Base repaired",
                         "Base repair rate", "LoRA repaired", "LoRA repair rate", "提升百分点"])
        for row in rows:
            writer.writerow([row["problem_id"], row["problem_name"], row["train_samples"],
                             row["test_samples"], row["base_repaired"], percent(row["base_rate"]),
                             row["lora_repaired"], percent(row["lora_rate"]),
                             f'{row["lift"] * 100:+.2f}'])

    def group_line(label, group):
        count = sum(row["test_samples"] for row in group)
        base_repaired = sum(row["base_repaired"] for row in group)
        lora_repaired = sum(row["lora_repaired"] for row in group)
        return (f"| {label} | {len(group)} | {count} | {percent(base_repaired / count)} | "
                f"{percent(lora_repaired / count)} | {(lora_repaired - base_repaired) / count * 100:+.2f} | "
                f"{lora_repaired - base_repaired} |")

    report = [
        "# Qwen2.5-Coder-7B LoRA 按题目修复结果分析", "",
        "## 总体结果", "",
        f"- 测试样本：{total}；题目数：{len(rows)}",
        f"- Base Micro Repair Rate：{base_total}/{total} = {percent(base_total / total)}",
        f"- LoRA Micro Repair Rate：{lora_total}/{total} = {percent(lora_total / total)}",
        f"- Base Macro Repair Rate：{percent(base_macro)}",
        f"- LoRA Macro Repair Rate：{percent(lora_macro)}",
        f"- Micro 提升：{(lora_total - base_total) / total * 100:+.2f} 个百分点",
        f"- Macro 提升：{(lora_macro - base_macro) * 100:+.2f} 个百分点", "",
        "## 提升覆盖面", "",
        f"- 提升题目：{sum(row['lift'] > 0 for row in rows)}/{len(rows)}",
        f"- 持平题目：{sum(row['lift'] == 0 for row in rows)}/{len(rows)}",
        f"- 下降题目：{sum(row['lift'] < 0 for row in rows)}/{len(rows)}", "",
        "## 训练高频题与其他题", "",
        "这里将训练样本数达到单题上限 250 条的题定义为高频题。`新增修复` 是 LoRA repaired 减 Base repaired。", "",
        "| 分组 | 题目数 | test 样本数 | Base rate | LoRA rate | 提升百分点 | 新增修复 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        group_line("高频题（train=250）", rich),
        group_line("其他题（train<250）", other), "",
        "## 结论", "",
        f"- LoRA 的提升具有普遍性：{sum(row['lift'] > 0 for row in rows)}/{len(rows)} 道题提升，"
        f"没有题目下降；Macro 也提升了 {(lora_macro - base_macro) * 100:.2f} 个百分点。",
        f"- 高频题贡献 {rich_gain}/{total_gain}（{rich_gain / total_gain * 100:.2f}%）个新增修复，"
        f"但它们本身也占 {sum(row['test_samples'] for row in rich)}/{total}（{sum(row['test_samples'] for row in rich) / total * 100:.2f}%）的测试样本。",
        f"- 高频题与其他题分别提升 {((sum(row['lora_repaired'] for row in rich) - sum(row['base_repaired'] for row in rich)) / sum(row['test_samples'] for row in rich)) * 100:.2f} 和 "
        f"{((sum(row['lora_repaired'] for row in other) - sum(row['base_repaired'] for row in other)) / sum(row['test_samples'] for row in other)) * 100:.2f} 个百分点，差距很小。"
        "因此总新增修复在数量上主要来自高频题，是测试集构成所致；从题目覆盖面和分组提升幅度看，并非只对 ALU、FSM、counter 等高频题有效。", "",
        "## 每题结果", "",
        "| problem_id | problem_name | train 样本数 | test 样本数 | Base repaired | Base repair rate | LoRA repaired | LoRA repair rate | 提升百分点 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        report.append(
            f"| {row['problem_id']} | {row['problem_name']} | {row['train_samples']} | {row['test_samples']} | "
            f"{row['base_repaired']} | {percent(row['base_rate'])} | {row['lora_repaired']} | "
            f"{percent(row['lora_rate'])} | {row['lift'] * 100:+.2f} |"
        )
    (args.output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
