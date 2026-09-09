#!/usr/bin/env python3
# 统计 SFT 数据经过 Qwen tokenizer 后的输入、assistant 输出和完整对话长度分布。
# 核心逻辑：用 Qwen chat template 计算 prompt/完整对话 token 数，单独编码 assistant
# 回复；再根据分位数和候选 max_seq_length 计算覆盖率，并列出异常长样本。

import argparse
import csv
import json
import math
from pathlib import Path


def load_jsonl(path):
    """读取 JSONL 数据。"""
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def token_count(tokenizer, text):
    """不添加额外 special token，统计普通文本的 token 数。"""
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def chat_token_count(tokenizer, messages, add_generation_prompt):
    """按 tokenizer 的 chat template 统计对话 token 数。"""
    tokens = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
    )
    return len(tokens["input_ids"])


def percentile(values, percent):
    """使用线性插值计算分位数，避免额外依赖 numpy。"""
    if not values:
        return 0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def round_up(value, multiple):
    """将长度向上取整到指定倍数，方便直接作为 max_seq_length。"""
    return math.ceil(value / multiple) * multiple


def summarize(values, percentiles):
    return {
        "count": len(values),
        "min": min(values) if values else 0,
        "mean": round(sum(values) / len(values), 2) if values else 0,
        "max": max(values) if values else 0,
        "percentiles": {
            f"p{str(percentile_value).replace('.', '_')}": round(percentile(values, percentile_value), 2)
            for percentile_value in percentiles
        },
    }


def get_messages(row):
    messages = row.get("messages") or row.get("conversation")
    if not messages:
        raise ValueError("样本缺少 messages/conversation 字段")
    return messages


def analyze(rows, tokenizer, max_seq_lengths, anomaly_percentile, round_multiple):
    records = []
    for index, row in enumerate(rows):
        messages = get_messages(row)
        assistant_indexes = [i for i, item in enumerate(messages) if item.get("role") == "assistant"]
        if not assistant_indexes:
            raise ValueError(f"第 {index + 1} 条样本没有 assistant 消息")

        # SFT 通常只训练最后一个 assistant 回复；prompt 是它之前的上下文。
        assistant_index = assistant_indexes[-1]
        prompt_messages = messages[:assistant_index]
        assistant_text = messages[assistant_index].get("content", "")
        input_tokens = chat_token_count(tokenizer, prompt_messages, add_generation_prompt=True)
        assistant_tokens = token_count(tokenizer, assistant_text)
        full_tokens = chat_token_count(tokenizer, messages, add_generation_prompt=False)

        records.append({
            "index": index,
            "pair_id": row.get("pair_id"),
            "problem_id": row.get("problem_id"),
            "input_tokens": input_tokens,
            "assistant_tokens": assistant_tokens,
            "full_tokens": full_tokens,
        })

    stats = {
        name: summarize([record[name] for record in records], [50, 90, 95, 99, 99.5])
        for name in ("input_tokens", "assistant_tokens", "full_tokens")
    }
    full_values = [record["full_tokens"] for record in records]
    anomaly_threshold = percentile(full_values, anomaly_percentile)
    recommended = round_up(percentile(full_values, 99), round_multiple)

    for record in records:
        record["is_anomaly"] = record["full_tokens"] >= anomaly_threshold
        record["exceeds_max_seq_length"] = {
            str(length): record["full_tokens"] > length for length in max_seq_lengths
        }

    coverage = {
        str(length): {
            "covered": sum(record["full_tokens"] <= length for record in records),
            "total": len(records),
            "coverage_percent": round(
                100 * sum(record["full_tokens"] <= length for record in records) / len(records), 2
            ) if records else 0,
        }
        for length in max_seq_lengths
    }
    return records, {
        "stats": stats,
        "anomaly_percentile": anomaly_percentile,
        "anomaly_threshold_tokens": round(anomaly_threshold, 2),
        "anomaly_count": sum(record["is_anomaly"] for record in records),
        "recommended_max_seq_length_p99": recommended,
        "round_multiple": round_multiple,
        "coverage_by_max_seq_length": coverage,
    }


def write_csv(path, records):
    fields = list(records[0]) if records else [
        "index", "pair_id", "problem_id", "input_tokens", "assistant_tokens",
        "full_tokens", "is_anomaly", "exceeds_max_seq_length",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser(description="统计 SFT 数据的 Qwen token 长度分布")
    parser.add_argument("--data", type=Path, required=True, help="输入 JSONL 文件")
    parser.add_argument("--model", type=Path, required=True, help="Qwen tokenizer 目录或 Hugging Face 模型名")
    parser.add_argument("--output", type=Path, default=Path("token_length_stats"), help="输出目录")
    parser.add_argument(
        "--max-seq-lengths", type=int, nargs="+", default=[2048, 4096, 8192],
        help="要评估覆盖率的候选 max_seq_length，默认：2048 4096 8192",
    )
    parser.add_argument("--anomaly-percentile", type=float, default=99, help="异常阈值分位数，默认 p99")
    parser.add_argument("--round-multiple", type=int, default=128, help="推荐长度向上取整倍数")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.model.exists())
    rows = load_jsonl(args.data)
    records, summary = analyze(
        rows, tokenizer, args.max_seq_lengths, args.anomaly_percentile, args.round_multiple
    )
    summary.update({"data": str(args.data), "model": str(args.model), "sample_count": len(rows)})

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(args.output / "token_lengths.csv", records)
    (args.output / "anomalies.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records if record["is_anomaly"]),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"异常样本明细：{args.output / 'anomalies.jsonl'}")


if __name__ == "__main__":
    main()
