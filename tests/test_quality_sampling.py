# 功能：验证多维质量评分中的标签过滤、课程范围过滤和稀缺性计算。
# 核心逻辑：构造小型代码对，检查错误标签与域外题目被删除，稀缺题目获得更高分数。

import tempfile
from pathlib import Path
import unittest

from download.filtering.score_and_sample_repair_pairs import score_rows, weighted_sample


class QualitySamplingTest(unittest.TestCase):
    def test_hard_filter_and_soft_signals(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            codes = {
                "a.v": "module a; assign x = 0; endmodule\n",
                "b.v": "module a; assign x = 1; endmodule\n",
                "c.v": "module b; assign y = 0; endmodule\n",
                "d.v": "module b; assign y = 1; endmodule\n",
            }
            for name, code in codes.items():
                (root / name).write_text(code, encoding="utf-8")
            pairs = [
                {"pair_id": "common-1", "problem_id": 1, "buggy_path": "a.v",
                 "fixed_path": "b.v", "verification": "testbench_verified"},
                {"pair_id": "common-2", "problem_id": 1, "buggy_path": "a.v",
                 "fixed_path": "b.v", "verification": "testbench_verified"},
                {"pair_id": "rare", "problem_id": 2, "buggy_path": "c.v",
                 "fixed_path": "d.v", "verification": "testbench_verified"},
                {"pair_id": "invalid", "problem_id": 3, "buggy_path": "a.v",
                 "fixed_path": "b.v", "verification": "invalid", "valid_repair": False},
                {"pair_id": "unverified", "problem_id": 4, "buggy_path": "a.v",
                 "fixed_path": "b.v"},
                {"pair_id": "out-of-scope", "problem_id": 5, "buggy_path": "a.v",
                 "fixed_path": "b.v", "verification": "testbench_verified"},
            ]
            scored, rejected = score_rows(pairs, root, {1, 2, 3, 4}, 0.9)
            by_id = {row["pair_id"]: row for row in scored}

            self.assertEqual(rejected[0]["filter_reason"], "invalid_fail_to_ac_label")
            self.assertEqual(rejected[1]["filter_reason"], "out_of_scope_problem")
            self.assertGreater(by_id["rare"]["quality_signals"]["diversity"],
                               by_id["common-1"]["quality_signals"]["diversity"])
            self.assertTrue(by_id["unverified"]["needs_review"])
            self.assertLess(by_id["unverified"]["training_weight"],
                            by_id["common-1"]["training_weight"])
            self.assertEqual(len(weighted_sample(scored, 0.5, 42)), 2)


if __name__ == "__main__":
    unittest.main()
