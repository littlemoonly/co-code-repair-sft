# 用真实 Icarus 验证成功、错误输出、编译失败和超时，防止把仿真退出当成通过。
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from evaluate_repair import extract_code, judge, matches, summary


class EvaluationTest(unittest.TestCase):
    def test_extract_multiple_modules(self):
        self.assertEqual(extract_code('```verilog\nmodule a; endmodule\n```\n```verilog\nmodule b; endmodule\n```'),
                         'module a; endmodule\n\n\nmodule b; endmodule\n')

    def test_no_empty_matches_or_partial_rate(self):
        self.assertEqual(matches('[01]\\n', 'ISim\r\n1\r\n0\r\n'), ['1', '0'])
        self.assertEqual(matches(r'^[0-9]\n', '0\r\n1\r\n2\r\n'), ['0', '1', '2'])
        report = summary([{'problem_id': 1}, {'problem_id': 1}],
                         [{'problem_id': 1, 'status': 'passed', 'passed': True}], 'model')
        self.assertIsNone(report['repair_rate_at_1'])
        self.assertEqual(report['total'], 2)

    def test_real_simulator(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = root / 'fixture'
            fixture.mkdir()
            (fixture / 'tb.v').write_text('`timescale 1ns/1ps\nmodule tb; wire out; dut d(out); initial begin #1; $display("RESULT=%b",out); #1; $finish; end endmodule')
            (fixture / 'standard.txt').write_text('RESULT=1\n')
            problem = {'cases': [{'name': 'fixture', 'directory': str(fixture), 'sources': ['tb.v'],
                                 'standard': 'standard.txt', 'top_module': 'tb', 'simulating_time': '100ns',
                                 'timeout': 1, 'output_regex': 'RESULT=[01]'}]}
            candidates = [('module dut(output out); assign out=1; endmodule', 'passed'),
                          ('module dut(output out); assign out=0; endmodule', 'wrong_answer'),
                          ('this is invalid', 'compile_error'),
                          ('module dut(output out); initial forever begin end endmodule', 'simulation_timeout')]
            for index, (code, expected) in enumerate(candidates):
                candidate = root / f'candidate{index}.v'
                candidate.write_text(code)
                result = judge(candidate, problem, root / f'run{index}', 1)
                self.assertEqual(result['cases'][0]['status'], expected)
                self.assertEqual(result['passed'], expected == 'passed')


if __name__ == '__main__':
    unittest.main()
