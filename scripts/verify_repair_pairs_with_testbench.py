#!/usr/bin/env python3
# 功能：保留旧的 scripts/ 调用路径，转发到 download/verification 中的统一实现。
# 核心逻辑：通过兼容导入复用唯一的验证器，避免两份相同代码长期出现差异。

from download.verification.verify_repair_pairs_with_testbench import *
from download.verification.verify_repair_pairs_with_testbench import main


if __name__ == "__main__":
    main()
