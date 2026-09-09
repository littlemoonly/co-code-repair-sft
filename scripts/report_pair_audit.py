#!/usr/bin/env python3
# 将完整代码对审计结果整理成中文报告与 CSV，补充题目明确的源码禁用字符检查。
# 只读取原始数据和审计缓存；报告和派生表写入审计输出目录，不改动代码对。
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import re

from evaluate_repair import ROOT, digest, extract_code, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    summary = json.loads((output / 'summary.json').read_text())
    config = json.loads((output / 'run_config.json').read_text())
    if not summary['complete']:
        parser.error('审计尚未完成')
    records = [json.loads(line) for line in (output / 'results.jsonl').read_text().splitlines()]
    if len(records) != summary['total']:
        parser.error('结果条数与汇总不一致')
    constraints = []
    for name, expected_hash in config['data_sha256'].items():
        source = Path(name)
        if digest(source) != expected_hash:
            parser.error(f'审计后数据发生变化: {source}')
        for line, text in enumerate(source.read_text().splitlines(), 1):
            row = json.loads(text)
            # 这两道题在题面及 testcase.yaml 中明确禁止源码出现这些字符。
            pattern = {326: r'[<>]', 397: r'[%*]'}.get(row['problem_id'])
            if pattern is None:
                continue
            for side, code in [('buggy', extract_code(row['messages'][1]['content'].split('Buggy Verilog code:', 1)[1])),
                               ('fixed', extract_code(row['messages'][2]['content']))]:
                if re.search(pattern, code):
                    constraints.append({'split': source.stem, 'line': line, 'pair_id': row['pair_id'],
                                        'problem_id': row['problem_id'], 'side': side,
                                        'forbidden_characters': sorted(set(re.findall(pattern, code)))})
    write_json(output / 'source_constraint_findings.json', constraints)
    fields = ['split', 'line', 'pair_id', 'problem_id', 'category', 'issues',
              'buggy_status', 'fixed_status', 'buggy_code', 'fixed_code']
    with (output / 'anomalies.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in records:
            if row['issues'] or row['category'] != 'valid_fail_to_pass':
                flat = {k: row[k] for k in fields[:5]}
                flat['issues'] = ','.join(row['issues'])
                for side in ['buggy', 'fixed']:
                    flat[f'{side}_status'] = row[side]['status']
                    flat[f'{side}_code'] = row[side]['code_file']
                writer.writerow(flat)
    lines = ['# 代码对全量审计', '',
             f'共检查 {len(records)} 对。按当前 Icarus Verilog-2005 与 testbench 判定，不代表所有输入上的形式化正确性。', '',
             '| 数据集 | 总数 | FAIL→PASS | 双方通过 | 双方失败 | PASS→FAIL | 超时待确认 |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    keys = ['total', 'valid_fail_to_pass', 'both_passed', 'both_failed', 'pass_to_fail', 'timeout_unresolved']
    for split in ['train', 'dev', 'test']:
        counts = summary['by_split'][split]
        lines.append('| ' + split + ' | ' + ' | '.join(str(counts.get(k, 0)) for k in keys) + ' |')
    lines += ['', '## fixed 未通过的样本', '', '| 数据集 | pair_id | 状态 | 代码 |', '| --- | --- | --- | --- |']
    failed = [r for r in records if not r['fixed']['passed']]
    for row in failed:
        path = row['fixed']['code_file']
        lines.append(f'| {row["split"]} | {row["pair_id"]} | {row["fixed"]["status"]} | [code.v]({path}) |')
    diagnostics_path = output / 'fixed_diagnostics.json'
    if diagnostics_path.exists():
        lines += ['', '诊断副本验证（未修改原始数据）：', '']
        for result in json.loads(diagnostics_path.read_text()):
            status = '全部通过' if result['passed'] else '仍未全部通过'
            lines.append(f'- `{result["pair_id"]}`：{result["change"]}后，{status}。')
    lines += ['', f'fixed 通过 {sum(r["fixed"]["passed"] for r in records)}/{len(records)}。', '',
              '## buggy 也通过的题目分布', '', '| problem_id | 数量 |', '| --- | ---: |']
    for pid, count in Counter(r['problem_id'] for r in records if r['buggy']['passed']).most_common():
        lines.append(f'| {pid} | {count} |')
    lines += ['', '## 边界与复核', '',
              f'- 当前全量筛查墙钟上限为 {config["timeout"]} 秒；较早的兼容结果来自 90 秒审计，原始命令、时间和日志均保留。',
              '- timeout_unresolved 不作为已证明代码错误的结论，需在更宽裕的资源/时限下复核。',
              '- both_passed 表示当前用例未区分代码对，不能直接断言 buggy 无错误。',
              f'- 源码禁用字符命中 {len(constraints)} 条，fixed 命中 {sum(r["side"] == "fixed" for r in constraints)} 条，详见 source_constraint_findings.json。上表为仿真结果，源码约束另列。',
              '- 模块名、数据内容未由本次审计修改。',
              '- 评测匹配已改为逐行正则模式，旧脚本只比较部分锚定输出的历史结果不能继续使用。',
              '', '异常 CSV：`anomalies.csv`；逐对完整结果：`results.jsonl`；用例日志在对应 code.v 同目录的 test_*/case_000 下。', '']
    (output / 'report.md').write_text('\n'.join(lines))
    print(output / 'report.md')


if __name__ == '__main__':
    main()
