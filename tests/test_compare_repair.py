# 用独立构造的两组评测记录验证配对比较及条件不一致时的拒绝行为。
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/compare_repair.py'


class ComparisonTest(unittest.TestCase):
    def test_paired_changes_and_condition_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = {'test_sha256': 'test', 'pair_ids': ['a', 'b', 'c'], 'manifest_sha256': 'manifest',
                      'resources': {}, 'script_sha256': 'script', 'generation': {'max_new_tokens': 4096},
                      'timeout': 90, 'iverilog_version': 'test', 'versions': {}, 'model': '/base',
                      'model_files': {}, 'mode': 'model', 'adapter': None}
            for name, passed in [('base', [True, False, False]), ('lora', [False, True, True])]:
                folder = root / name
                folder.mkdir()
                current = dict(config, adapter=None if name == 'base' else '/adapter')
                (folder / 'run_config.json').write_text(json.dumps(current))
                (folder / 'summary.json').write_text(json.dumps({'complete': True}))
                (folder / 'results.jsonl').write_text(''.join(json.dumps({'pair_id': pid, 'passed': value}) + '\n'
                                                            for pid, value in zip(config['pair_ids'], passed)))
            command = [sys.executable, str(SCRIPT), str(root / 'base'), str(root / 'lora')]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(result.stdout)
            self.assertEqual(report['improved_pair_ids'], ['b', 'c'])
            self.assertEqual(report['regressed_pair_ids'], ['a'])
            self.assertAlmostEqual(report['delta_percentage_points'], 100 / 3)
            config['timeout'] = 10
            (root / 'lora/run_config.json').write_text(json.dumps(config))
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('timeout', result.stderr)


if __name__ == '__main__':
    unittest.main()
