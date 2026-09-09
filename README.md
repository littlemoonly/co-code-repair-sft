<div align="center">

# Verilog 代码智能修复模型

**💻 面向 BUAA CS 专业课的 Verilog 代码纠错模型**

**实现在保留学生原始结构的同时，完成必要正确的修复。🎉**

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.8-EE4C2C?logo=pytorch&logoColor=white)
![Model](https://img.shields.io/badge/Base-Qwen2.5--Coder--7B-6C5CE7)
<!-- ![Tests](https://img.shields.io/badge/Tests-5%20passed-brightgreen) -->

</div>

## 项目概述

通用代码大模型在修复 Verilog 作业时，容易把“纠错”变成“重写”：虽然输出可能可编译，却破坏了学生原有结构，也不利于教学分析。本项目基于 **Qwen2.5-Coder-7B-Instruct** 进行 LoRA 监督微调，让模型从真实的 `FAIL → AC` 提交对中学习最小必要修改，并通过真实 testbench 编译、仿真和判题。

### 核心特性

- **结构保持修复**：输入题目描述与 buggy Verilog，输出完整修复代码，减少无关改写。
- **真实教学数据**：提供 train/dev/test 三个 SFT 切分，共 6,969 个代码对，按学生划分以降低数据泄漏风险。
- **参数高效训练**：基于 PEFT LoRA，仅训练注意力和 MLP 投影层的低秩适配参数。
- **可复现评测**：固定贪心解码、随机种子、测试集和判题资源，保留生成、编译及仿真日志。
- **完整数据流水线**：覆盖清洗、编辑距离过滤、testbench 验证、SFT 转换、训练与结果分析。

工作流：`学生提交 → FAIL→AC 配对与清洗 → Chat SFT → LoRA 训练 → 生成修复 → Icarus Verilog 判题`。

## 实验结果

固定测试集包含 **737 个样本、47 道题**(题目与代码相关数据尚未上传)。每个样本只生成一个候选，且通过该题全部测试用例才记为修复成功。

| 模型 | 修复数 | Repair Rate@1 |
| --- | ---: | ---: |
| Qwen2.5-Coder-7B-Instruct | 209 / 737 | 28.36% |
| LoRA SFT | **494 / 737** | **67.03%** |

LoRA 相比基线提升 **38.45 个百分点**。

## 安装

### 环境要求

- Linux、Python 3.10+
- 支持 BF16 的 NVIDIA GPU 与 CUDA
- `iverilog`、`vvp`
- 本地 Qwen2.5-Coder-7B-Instruct 权重

```bash
sudo apt-get update
sudo apt-get install -y iverilog

python -m venv .venv
source .venv/bin/activate
pip install torch
pip install transformers peft accelerate wandb pyyaml
pip install flash-attn --no-build-isolation

huggingface-cli download Qwen/Qwen2.5-Coder-7B-Instruct \
  --local-dir models/Qwen2.5-Coder-7B-Instruct
```


## 快速开始

先在一分钟内完成环境与判题链路自检：

```bash
python -m unittest discover -s tests -v
```

运行 3 个样本的基线冒烟评测：

```bash
python scripts/run_qwen_baseline_archive_split.py \
  --limit 3 \
  --output runs/baseline_smoke
```

评测支持生成与判题分阶段执行；显存不足时会自动降低生成 batch size。

## 训练与评测

### 1. 审计训练数据（可选）

```bash
python scripts/prepare_testbenches.py --all-splits \
  --output testbench/audit_prepared
```

### 2. 训练 LoRA

```bash
wandb login
python scripts/train_qwen_lora_sft.py \
  --output runs/qwen25_coder_7b_lora_sft
```

默认配置为 2 epochs、最大序列长度 8192、BF16、FlashAttention 2、gradient checkpointing；LoRA 使用 `r=16`、`alpha=32`、`dropout=0.05`，只对 assistant 回复计算 loss。训练完成后自动在 test 集计算 Repair Rate@1，可用 `--skip-test` 跳过。

### 3. 独立评测与续跑

```bash
# 使用已有 adapter 重新生成并判题
python scripts/train_qwen_lora_sft.py --eval-only \
  --output runs/qwen25_coder_7b_lora_sft

# 基线中断后续跑
python scripts/run_qwen_baseline_archive_split.py --resume \
  --output runs/qwen25_coder_7b_baseline_archive
```

主要产物包括 `results.jsonl`、`summary.json`、`testcase_manifest.json`、逐样本 `repair.v` 以及编译/仿真日志。详细脚本索引见 [`scripts/README.md`](scripts/README.md)，原始数据处理流程见 [`download/README.md`](download/README.md)。

## 数据格式

每条 JSONL 包含 `pair_id`、`problem_id` 和三轮 `messages`：system 定义任务，user 提供题目与 buggy 代码，assistant 提供 fixed 代码。Train/Dev/Test 分别有 5,413/819/737 条，覆盖 58/51/47 道题。

## Roadmap

- [ ] 增加更多课程题目与边界测试用例
- [ ] 提供统一依赖文件和一键训练配置
- [ ] 增加最小编辑距离、语法正确率等辅助指标
- [ ] 对比更多代码模型与参数高效微调方法

## 贡献

欢迎提交 Issue 或 Pull Request。贡献前请运行完整测试，并确保新增 Python 脚本包含功能说明、核心逻辑说明和适量中文注释。涉及数据或指标的改动，请同时提供数据来源、划分方式和可复现实验命令。

## License

本仓库当前未附带开源许可证。在许可证明确前，代码默认保留所有权利，不应视为已获得复制、修改或分发授权。
