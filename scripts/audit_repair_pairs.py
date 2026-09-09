#!/usr/bin/env python3
# 审计 train/dev/test 的代码对：buggy 应失败，fixed 应通过对应题目的全部用例。
# 复用评测脚本的 Icarus 判题规则，串行执行并按题目和代码哈希缓存；遇到首个
# 失败即可确定该代码未全部通过。保存逐对结果、异常清单和日志，不修改原始数据。
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

from evaluate_repair import ROOT, digest, extract_code, judge, matches, write_json


def code_hash(code):
    return hashlib.sha256(code.encode()).hexdigest()


def check_code(pid, code, problem, output, timeout, reuse=None):
    work = output / 'codes' / pid / code_hash(code)
    result_path = work / 'result.json'
    if result_path.exists():
        return json.loads(result_path.read_text())
    work.mkdir(parents=True, exist_ok=True)
    # 可复用较宽松超时下的结果，但成功结果的每个阶段都必须在当前时限内。
    # 原先运行到更长超时仍未结束的代码，当前也记为超时，保留原始证据路径。
    previous = reuse / 'codes' / pid / code_hash(code) / 'result.json' if reuse else None
    if previous and previous.exists():
        saved = json.loads(previous.read_text())
        phases = [c[k] for c in saved['cases'] for k in ['compile', 'simulation'] if k in c]
        if all(p['timeout'] or p['seconds'] <= timeout for p in phases):
            saved['code_file'] = os.path.relpath(reuse / saved['code_file'], output)
            saved['reused_from'] = str(previous)
            write_json(result_path, saved)
            return saved
    code_path = work / 'code.v'
    code_path.write_text(code)
    cases = []
    for index, case in enumerate(problem['cases']):
        # 审计仅需确定是否全部通过，失败后的其他用例不再运行。
        outcome = judge(code_path, {'cases': [case]}, work / f'test_{index:03d}', timeout)
        cases.extend(outcome['cases'])
        if not outcome['passed']:
            break
    status = cases[-1]['status'] if cases else 'missing_testbench'
    result = {'status': status, 'passed': status == 'passed', 'cases': cases,
              'tested_cases': len(cases), 'total_cases': len(problem['cases']),
              'code_file': str(code_path.relative_to(output))}
    write_json(result_path, result)
    return result


def build_summary(records, total, started):
    by_split = defaultdict(Counter)
    by_problem = defaultdict(Counter)
    issues = Counter()
    for row in records:
        for group in [by_split[row['split']], by_problem[str(row['problem_id'])]]:
            group['total'] += 1
            group[row['category']] += 1
        issues.update(row['issues'])
    return {'total': total, 'completed': len(records), 'complete': len(records) == total,
            'elapsed_seconds': round(time.monotonic() - started, 1),
            'categories': dict(Counter(r['category'] for r in records)),
            'issues': dict(issues), 'by_split': dict(by_split), 'by_problem': dict(by_problem)}


def main():
    parser = argparse.ArgumentParser(description='用真实 testbench 串行检查所有 buggy/fixed 代码对')
    parser.add_argument('--data', type=Path, default=ROOT / 'data/repair_pairs_sft')
    parser.add_argument('--manifest', type=Path, default=ROOT / 'testbench/audit_prepared/manifest.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/pair_audit')
    parser.add_argument('--timeout', type=float, default=90)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--reuse', type=Path, help='复用数据和判题规则相同、超时不短于当前设置的审计缓存')
    args = parser.parse_args()
    output = args.output.resolve()
    files = sorted(args.data.glob('*.jsonl'))
    manifest = json.loads(args.manifest.read_text())
    rows = [(file.stem, n, json.loads(line)) for file in files
            for n, line in enumerate(file.read_text().splitlines(), 1)]
    resources = {}
    for _, _, row in rows:
        pid = str(row['problem_id'])
        if pid not in manifest or not manifest[pid]['cases']:
            parser.error(f'缺少测试用例: problem_id={pid}')
    for problem in manifest.values():
        for case in problem['cases']:
            folder = ROOT / case['directory']
            for file in sorted(folder.rglob('*')):
                if file.is_file() and file.suffix != '.zip':
                    resources[str(file.relative_to(ROOT))] = digest(file)
            if not matches(case['output_regex'], (folder / case['standard']).read_text(errors='replace')):
                parser.error(f'空标准输出匹配: {folder}')
    config = {'data_sha256': {str(p.resolve()): digest(p) for p in files},
              'manifest_sha256': digest(args.manifest), 'resources': resources,
              'audit_script_sha256': digest(__file__), 'judge_script_sha256': digest(ROOT / 'scripts/evaluate_repair.py'),
              'timeout': args.timeout,
              'iverilog_version': subprocess.run(['iverilog', '-V'], capture_output=True, text=True).stdout.splitlines()[0]}
    reuse = args.reuse.resolve() if args.reuse else None
    if reuse:
        previous = json.loads((reuse / 'run_config.json').read_text())
        for key in ['data_sha256', 'manifest_sha256', 'resources', 'judge_script_sha256', 'iverilog_version']:
            if previous[key] != config[key]:
                parser.error(f'缓存条件不一致: {key}')
        if previous['timeout'] < args.timeout:
            parser.error('源缓存的超时必须不短于当前设置')
        config['reuse'] = {'directory': str(reuse), 'config_sha256': digest(reuse / 'run_config.json')}
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / 'run_config.json'
    if config_path.exists():
        if not args.resume or json.loads(config_path.read_text()) != config:
            parser.error('已有运行：使用 --resume 且数据、资源和脚本必须一致，或另选输出目录。')
    else:
        write_json(config_path, config)
    started = time.monotonic()
    records = []
    seen_pairs = Counter(row['pair_id'] for _, _, row in rows)
    # 持续写入结果；中断后依靠已原子落盘的代码级缓存恢复。
    with (output / 'results.jsonl').open('w') as full, (output / 'anomalies.jsonl').open('w') as bad:
        for split, line, row in rows:
            pid = str(row['problem_id'])
            messages = row['messages']
            if [m['role'] for m in messages] != ['system', 'user', 'assistant']:
                raise ValueError(f'messages 结构异常: {split}:{line}')
            buggy = extract_code(messages[1]['content'].split('Buggy Verilog code:', 1)[1])
            fixed = extract_code(messages[2]['content'])
            outcomes = {side: check_code(pid, code, manifest[pid], output, args.timeout, reuse)
                        for side, code in [('buggy', buggy), ('fixed', fixed)]}
            issues = []
            if buggy == fixed:
                issues.append('identical_code')
            if seen_pairs[row['pair_id']] > 1:
                issues.append('duplicate_pair_id')
            for side, outcome in outcomes.items():
                if 'timeout' in outcome['status']:
                    issues.append(f'{side}_timeout')
            if outcomes['buggy']['passed']:
                issues.append('buggy_passed')
            if not outcomes['fixed']['passed']:
                issues.append('fixed_not_passed')
            if any('timeout' in o['status'] for o in outcomes.values()):
                category = 'timeout_unresolved'
            elif outcomes['fixed']['passed']:
                category = 'both_passed' if outcomes['buggy']['passed'] else 'valid_fail_to_pass'
            else:
                category = 'pass_to_fail' if outcomes['buggy']['passed'] else 'both_failed'
            expected = re.search(r'^Module:\s*(\w+)', messages[1]['content'], re.M)
            record = {'split': split, 'line': line, 'pair_id': row['pair_id'], 'problem_id': row['problem_id'],
                      'category': category, 'issues': issues, 'expected_module': expected[1] if expected else None,
                      'modules': {side: re.findall(r'\bmodule\s+([A-Za-z_$][\w$]*)', code)
                                  for side, code in [('buggy', buggy), ('fixed', fixed)]},
                      **{side: {key: outcome[key] for key in ['status', 'passed', 'tested_cases', 'total_cases', 'code_file']}
                         for side, outcome in outcomes.items()}}
            encoded = json.dumps(record, ensure_ascii=False) + '\n'
            full.write(encoded)
            full.flush()
            if issues or category != 'valid_fail_to_pass':
                bad.write(encoded)
                bad.flush()
            records.append(record)
            if len(records) % 50 == 0 or len(records) == len(rows):
                report = build_summary(records, len(rows), started)
                write_json(output / 'summary.json', report)
                print(json.dumps({k: report[k] for k in ['completed', 'total', 'elapsed_seconds', 'categories']}, ensure_ascii=False), flush=True)
    print(f'结果: {output / "summary.json"}', flush=True)


if __name__ == '__main__':
    main()
