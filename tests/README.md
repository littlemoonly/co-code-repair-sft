# 测试说明

测试使用 Python 标准库 `unittest`，按功能分为三组：

- `test_evaluate_repair.py`：测试代码块提取、输出匹配、评测汇总，以及真实 Icarus 编译/仿真结果。
- `test_compare_repair.py`：测试基线与 LoRA 结果的配对比较，以及评测条件不一致时的拒绝逻辑。
- `test_audit_repair_pairs.py`：测试审计缓存的超时限制，避免复用超过当前时限的旧成功结果。

在项目根目录运行全部测试：

```bash
python -m unittest discover -s tests -v
```

其中 `test_evaluate_repair.py` 需要系统已安装 `iverilog` 和 `vvp`。
