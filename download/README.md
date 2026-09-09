# 数据处理脚本

`download` 目录负责从原始数据生成可训练、可验证的 repair pair 数据，按阶段分为四组：

- `catalog/`：题目目录、数据库题目清单和题目统计。
- `filtering/`：代码规范化、注释清理、pair 筛选和编辑距离过滤。
- `verification/`：使用 testbench 验证 buggy/fixed 代码对。
- `conversion/`：把验证后的 pair 转换为 SFT JSONL，并按学生划分数据集。

建议流程：

```text
catalog -> filtering -> verification -> conversion
```

脚本中的相对路径默认以项目根目录为基准，建议在项目根目录执行。例如：

```bash
python -m download.filtering.filter_repair_pairs
python -m download.conversion.convert_repair_pairs_to_sft
```

验证器的统一实现位于 `verification/verify_repair_pairs_with_testbench.py`。
`scripts/verify_repair_pairs_with_testbench.py` 仅用于兼容已有命令和导入路径。
