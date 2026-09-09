# 验证较长时限的审计缓存不会把超过当前时限的成功结果直接算作通过。
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from audit_repair_pairs import check_code, code_hash


class CacheTest(unittest.TestCase):
    def test_cache_time_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            code = 'module dut; endmodule\n'
            old = root / 'old'
            path = old / 'codes/1' / code_hash(code)
            path.mkdir(parents=True)
            (path / 'code.v').write_text(code)
            previous = {'status': 'passed', 'passed': True, 'total_cases': 1, 'tested_cases': 1,
                        'code_file': str((path / 'code.v').relative_to(old)),
                        'cases': [{'status': 'passed', 'compile': {'timeout': False, 'seconds': .1},
                                   'simulation': {'timeout': False, 'seconds': 20}}]}
            (path / 'result.json').write_text(json.dumps(previous))
            outcome = {'passed': False, 'cases': [{'status': 'simulation_timeout'}]}
            with patch('audit_repair_pairs.judge', return_value=outcome) as judge:
                result = check_code('1', code, {'cases': [{}]}, root / 'new', 10, old)
                judge.assert_called_once()
                self.assertFalse(result['passed'])
            previous['cases'][0]['simulation']['seconds'] = .2
            (path / 'result.json').write_text(json.dumps(previous))
            with patch('audit_repair_pairs.judge') as judge:
                result = check_code('1', code, {'cases': [{}]}, root / 'fast', 10, old)
                judge.assert_not_called()
                self.assertTrue(result['passed'])
                self.assertEqual((root / 'fast' / result['code_file']).resolve(), path / 'code.v')


if __name__ == '__main__':
    unittest.main()
