# scripts 目录说明

脚本按工作流分为四组，建议从上到下执行：

## 数据与测试资源

- `prepare_testbenches.py`：从题目归档整理固定 testbench。
- `verify_repair_pairs_with_testbench.py`：验证单批 repair pair。
- `evaluate_repair.py`：提供公共的代码提取、判题、输出匹配和结果汇总函数。

## 数据审计

- `audit_repair_pairs.py`：审计 train/dev/test 中 buggy 与 fixed 的通过情况。
- `report_pair_audit.py`：把审计缓存整理成报告和 CSV。
- `audit_sft_with_archive_testbench.py`：按归档验证 SFT 数据。

## 模型训练与评测

- `train_qwen_lora_sft.py`：训练 Qwen LoRA SFT，并进行测试集评测。
- `run_qwen_baseline_archive.py`：运行 Qwen 基线并判题。
- `run_qwen_baseline_archive_split.py`：分阶段运行生成与判题，适合显存紧张场景。
- `compare_repair.py`：比较基线和 LoRA 的结果。

## 结果分析

- `analyze_repair_by_problem.py`：按题目统计修复率。
- `analyze_sft_token_lengths.py`：统计 SFT 样本 token 长度。

脚本均从项目根目录执行，例如：

```bash
python scripts/prepare_testbenches.py
python scripts/audit_repair_pairs.py --output runs/pair_audit_full --timeout 10
```
