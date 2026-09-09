#!/usr/bin/env python3
# Qwen2.5-Coder-7B-Instruct + LoRA Verilog 修复 SFT。
# 核心逻辑：按 Qwen chat template 编码 system/user 作为 prompt，只把最后一个
# assistant 的 fixed Verilog 编码为标签，其余位置设为 -100；训练结束后调用
# 现有 archive audit/verifier 判题链路，在 test 集上计算 Repair Rate@1，并同步保存 W&B/本地日志。

import argparse
import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import TrainerCallback


ROOT = Path(__file__).resolve().parents[1]
MAX_SEQ_LENGTH = 8192


def run_archive_test(model_path, adapter_path, rows, output, catalog, archive,
                     local_testbench_root, wall_timeout, judge_jobs, gen_batch_size=16,
                     generate_missing=True):
    """先批量生成，再并行运行 archive audit verifier；支持复用已生成样本。"""
    import run_qwen_baseline_archive as archive_runner
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    judge_args = argparse.Namespace(
        catalog=catalog, archive=archive, local_testbench_root=local_testbench_root,
        wall_timeout=wall_timeout,
    )
    manifest, judge = archive_runner.make_judge(rows, judge_args)
    output.mkdir(parents=True, exist_ok=True)
    (output / "testcase_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    generated_items = []
    pending = []
    for index, row in enumerate(rows):
        sample = output / "samples" / f"{index:05d}_{row['pair_id']}"
        code_path = sample / "repair.v"
        model_output = sample / "model_output.txt"
        if code_path.exists() and model_output.exists():
            generated_items.append((row, code_path, model_output.read_text(encoding="utf-8"), None))
        else:
            pending.append((index, row))

    if pending and generate_missing:
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16, attn_implementation="flash_attention_2",
            local_files_only=True,
        ).to("cuda").eval()
        model = PeftModel.from_pretrained(model, adapter_path, local_files_only=True).eval()
        eos_ids = model.generation_config.eos_token_id
        eos_ids = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids or [])
        # 相近长度放入同一 batch，减少左侧 padding 和 KV cache 浪费。
        pending_with_prompts = []
        for item in pending:
            prompt = tokenizer.apply_chat_template(
                item[1]["messages"][:2], tokenize=False, add_generation_prompt=True
            )
            prompt_length = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
            pending_with_prompts.append((prompt_length, item, prompt))
        pending_with_prompts.sort(key=lambda x: x[0])

        offset = 0
        current_batch_size = gen_batch_size
        while offset < len(pending_with_prompts):
            batch_items = pending_with_prompts[offset:offset + current_batch_size]
            batch = [item for _, item, _ in batch_items]
            prompts = [prompt for _, _, prompt in batch_items]
            encoded = tokenizer(
                prompts, add_special_tokens=False, padding=True, return_tensors="pt"
            ).to("cuda")
            padded_length = encoded["input_ids"].shape[1]
            try:
                with torch.inference_mode():
                    generated = model.generate(
                        **encoded, do_sample=False, num_beams=1, max_new_tokens=4096,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=model.generation_config.eos_token_id,
                    )[:, padded_length:]
            except torch.OutOfMemoryError:
                if current_batch_size == 1:
                    raise
                current_batch_size = max(1, current_batch_size // 2)
                torch.cuda.empty_cache()
                print(f"CUDA OOM，生成 batch size 降为 {current_batch_size}", flush=True)
                continue

            for (index, row), prompt, token_tensor in zip(batch, prompts, generated):
                token_ids = token_tensor.tolist()
                output_length = next(
                    (i + 1 for i, token_id in enumerate(token_ids) if token_id in eos_ids),
                    len(token_ids),
                )
                raw_output = tokenizer.decode(token_ids[:output_length], skip_special_tokens=True)
                sample = output / "samples" / f"{index:05d}_{row['pair_id']}"
                sample.mkdir(parents=True, exist_ok=True)
                (sample / "prompt.txt").write_text(prompt, encoding="utf-8")
                (sample / "model_output.txt").write_text(raw_output, encoding="utf-8")
                code_path = sample / "repair.v"
                code_path.write_text(archive_runner.audit.extract_code(raw_output), encoding="utf-8")
                generated_items.append((row, code_path, raw_output, output_length))
            offset += len(batch)
            print(f"generated {offset}/{len(pending_with_prompts)} (batch={current_batch_size})", flush=True)

        # GPU 阶段结束后释放模型，避免后续纯 CPU 判题长期占用显存。
        del model
        torch.cuda.empty_cache()

    generated_items.sort(key=lambda item: [row["pair_id"] for row in rows].index(item[0]["pair_id"]))
    with ThreadPoolExecutor(max_workers=judge_jobs) as executor:
        outcomes = list(executor.map(lambda item: judge(item[0]["problem_id"], item[1]), generated_items))
    results = []
    for item, outcome in zip(generated_items, outcomes):
        row, code_path, raw_output, output_length = item
        results.append({
            "pair_id": row["pair_id"], "problem_id": row["problem_id"],
            "model_output": raw_output, "output_tokens": output_length,
            "code_file": str(code_path.relative_to(output)), "test_result": outcome,
        })
        print(f"judged {len(results)}/{len(rows)}: {row['pair_id']} {outcome['status']}", flush=True)

    (output / "results.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results), encoding="utf-8"
    )
    repaired = sum(row["test_result"]["status"] == "AC" for row in results)
    report = {
        "total": len(rows), "completed": len(results), "complete": len(results) == len(rows),
        "repaired": repaired,
        "repair_rate_at_1": repaired / len(rows) if len(results) == len(rows) else None,
    }
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def read_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class RepairDataset(Dataset):
    """把 messages 转成 prompt + assistant labels，并过滤超长样本。"""

    def __init__(self, rows, tokenizer, max_seq_length):
        self.items = []
        self.filtered_pair_ids = []
        for row in rows:
            messages = row["messages"]
            assistant_index = next(i for i, m in enumerate(messages) if m["role"] == "assistant")
            prompt_ids = tokenizer.apply_chat_template(
                messages[:assistant_index], tokenize=True, add_generation_prompt=True
            )
            if hasattr(prompt_ids, "__getitem__") and not isinstance(prompt_ids, list):
                prompt_ids = prompt_ids["input_ids"]
            answer_ids = tokenizer(
                messages[assistant_index]["content"], add_special_tokens=False
            )["input_ids"]
            if tokenizer.eos_token_id is not None:
                answer_ids = answer_ids + [tokenizer.eos_token_id]
            input_ids = prompt_ids + answer_ids
            if len(input_ids) > max_seq_length:
                self.filtered_pair_ids.append(row.get("pair_id"))
                continue
            self.items.append({
                "input_ids": input_ids,
                "labels": [-100] * len(prompt_ids) + answer_ids,
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class SFTCollator:
    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    def __call__(self, features):
        input_ids = pad_sequence(
            [torch.tensor(x["input_ids"], dtype=torch.long) for x in features],
            batch_first=True, padding_value=self.pad_id,
        )
        labels = pad_sequence(
            [torch.tensor(x["labels"], dtype=torch.long) for x in features],
            batch_first=True, padding_value=-100,
        )
        return {"input_ids": input_ids, "labels": labels,
                "attention_mask": input_ids.ne(self.pad_id)}


class MetricsCallback(TrainerCallback):
    """将 Trainer 日志补充 GPU 显存，并写成简洁 JSONL。"""

    def __init__(self, path, wandb_run):
        self.path = path
        self.wandb_run = wandb_run

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        record = {"step": state.global_step, **logs}
        if state.epoch is not None:
            record["epoch"] = round(state.epoch, 4)
        if torch.cuda.is_available():
            record["gpu_memory_allocated_gb"] = round(torch.cuda.memory_allocated() / 2**30, 3)
            record["gpu_memory_reserved_gb"] = round(torch.cuda.memory_reserved() / 2**30, 3)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        if self.wandb_run:
            self.wandb_run.log(record, step=state.global_step)


def build_args(parser):
    parser.add_argument("--model", type=Path, default=ROOT / "models/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--train", type=Path, default=ROOT / "data/repair_pairs_sft/train.jsonl")
    parser.add_argument("--dev", type=Path, default=ROOT / "data/repair_pairs_sft/dev.jsonl")
    parser.add_argument("--test", type=Path, default=ROOT / "data/repair_pairs_sft/test.jsonl")
    parser.add_argument("--catalog", type=Path, default=ROOT / "data/problem_catalog.jsonl")
    parser.add_argument("--archive", type=Path, default=ROOT / "testbench/co-problem-set-all-branches.tar.gz")
    parser.add_argument("--local-testbench-root", type=Path,
                        default=ROOT / "testbench/generated/problem")
    parser.add_argument("--wall-timeout", type=int, default=10)
    parser.add_argument("--judge-jobs", type=int, default=8)
    parser.add_argument("--gen-batch-size", type=int, default=16)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/qwen25_coder_7b_lora_sft")
    parser.add_argument("--wandb-project", default="verilog-repair-sft")
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--skip-test", action="store_true", help="只训练，不运行最终 test 评测")
    parser.add_argument("--eval-only", action="store_true", help="使用已有 adapter，只重新生成/判题 test")
    parser.add_argument("--judge-only", action="store_true", help="只判题已落盘的 repair.v，不加载模型")
    return parser


def main():
    parser = build_args(argparse.ArgumentParser(description="Qwen Verilog 修复 LoRA SFT"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoints = args.output / "checkpoints"
    adapter_dir = args.output / "adapter"
    metrics_path = args.output / "metrics.jsonl"

    if args.eval_only:
        if not adapter_dir.exists():
            parser.error(f"adapter 不存在: {adapter_dir}")
        report = run_archive_test(
            args.model, adapter_dir, read_jsonl(args.test), args.output / "test_eval",
            args.catalog, args.archive, args.local_testbench_root,
            args.wall_timeout, args.judge_jobs, args.gen_batch_size,
            generate_missing=not args.judge_only,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    import wandb
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                              TrainingArguments, set_seed)

    set_seed(42)
    if not torch.cuda.is_available():
        parser.error("需要 CUDA GPU 才能训练 7B 模型")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_rows, dev_rows = read_jsonl(args.train), read_jsonl(args.dev)
    train_data = RepairDataset(train_rows, tokenizer, MAX_SEQ_LENGTH)
    dev_data = RepairDataset(dev_rows, tokenizer, MAX_SEQ_LENGTH)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",
        local_files_only=True,
    )
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ))
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    # transformers 5.x 仅保留 warmup_steps；按 effective batch size 换算 3% warmup。
    steps_per_epoch = math.ceil(len(train_data) / (2 * 16))
    total_steps = steps_per_epoch * 2
    warmup_steps = max(1, math.ceil(total_steps * 0.03))
    training_args = TrainingArguments(
        output_dir=str(checkpoints), num_train_epochs=2, learning_rate=1e-4,
        lr_scheduler_type="cosine", warmup_steps=warmup_steps, weight_decay=0.01,
        max_grad_norm=1.0, per_device_train_batch_size=2, gradient_accumulation_steps=16,
        per_device_eval_batch_size=2, bf16=True, tf32=True, gradient_checkpointing=True,
        optim="adamw_torch", adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-8,
        eval_strategy="epoch", save_strategy="epoch", logging_strategy="steps", logging_steps=10,
        report_to=[], seed=42, data_seed=42, remove_unused_columns=False,
        save_total_limit=2, load_best_model_at_end=False,
    )
    config = {
        "model": str(args.model), "max_seq_length": MAX_SEQ_LENGTH, "packing": False,
        "train_samples": len(train_data), "dev_samples": len(dev_data),
        "warmup_ratio": 0.03, "warmup_steps": warmup_steps, "total_steps": total_steps,
        "filtered_train_pair_ids": train_data.filtered_pair_ids,
        "filtered_dev_pair_ids": dev_data.filtered_pair_ids,
        "lora": {"r": 16, "alpha": 32, "dropout": 0.05, "bias": "none",
                 "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]},
        "training_args": training_args.to_dict(),
    }
    (args.output / "training_args.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )

    run = wandb.init(project=args.wandb_project, name=args.wandb_run_name,
                     config=config, dir=str(args.output))
    callback = MetricsCallback(metrics_path, run)
    trainer = Trainer(model=model, args=training_args, train_dataset=train_data,
                      eval_dataset=dev_data, data_collator=SFTCollator(tokenizer),
                      processing_class=tokenizer, callbacks=[callback])
    trainer.train()
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    final = {"adapter": str(adapter_dir), "train_samples": len(train_data), "dev_samples": len(dev_data)}
    if not args.skip_test:
        result_dir = args.output / "test_eval"
        test_rows = read_jsonl(args.test)
        report = run_archive_test(
            args.model, adapter_dir, test_rows, result_dir, args.catalog, args.archive,
            args.local_testbench_root, args.wall_timeout, args.judge_jobs, args.gen_batch_size,
        )
        final["repair_rate_at_1"] = report["repair_rate_at_1"]
        wandb.log({"test/repair_rate_at_1": final["repair_rate_at_1"]})
    with metrics_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"final": final}, ensure_ascii=False) + "\n")
    (args.output / "final_metrics.json").write_text(
        json.dumps(final, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    wandb.finish()
    print(json.dumps(final, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
