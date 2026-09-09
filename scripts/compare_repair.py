#!/usr/bin/env python3
# 比较基线和 LoRA 的完整评测结果；先核对测试集、用例、生成和判题条件，
# 再统计 Repair Rate@1 及逐样本改善/退化，拒绝比较未完成的运行。
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('baseline', type=Path)
    parser.add_argument('finetuned', type=Path)
    args = parser.parse_args()
    configs = [json.loads((p / 'run_config.json').read_text()) for p in [args.baseline, args.finetuned]]
    for key in ['test_sha256', 'pair_ids', 'manifest_sha256', 'resources', 'script_sha256', 'generation', 'timeout', 'iverilog_version']:
        if configs[0][key] != configs[1][key]:
            parser.error(f'评测条件不一致: {key}')
    if configs[0]['adapter'] is not None or configs[1]['adapter'] is None:
        parser.error('需要无 adapter 的基线和加载 adapter 的微调结果')
    if configs[0]['model'] != configs[1]['model']:
        parser.error('base model 路径不一致')
    base_prefix = configs[0]['model'] + '/'
    base_files = [{k: v for k, v in c['model_files'].items() if k.startswith(base_prefix)} for c in configs]
    if base_files[0] != base_files[1]:
        parser.error('base model 文件信息不一致')
    for package in ['torch', 'transformers']:
        if configs[0]['versions'].get(package) != configs[1]['versions'].get(package):
            parser.error(f'版本不一致: {package}')
    records = []
    for path, config in zip([args.baseline, args.finetuned], configs):
        report = json.loads((path / 'summary.json').read_text())
        if config['mode'] != 'model' or not report['complete']:
            parser.error(f'不是完整模型评测: {path}')
        data = [json.loads(line) for line in (path / 'results.jsonl').read_text().splitlines()]
        if [r['pair_id'] for r in data] != config['pair_ids']:
            parser.error(f'结果样本不匹配: {path}')
        records.append({r['pair_id']: r for r in data})
    before, after = records
    improved = [pid for pid in before if not before[pid]['passed'] and after[pid]['passed']]
    regressed = [pid for pid in before if before[pid]['passed'] and not after[pid]['passed']]
    rates = [sum(r['passed'] for r in data.values()) / len(data) for data in records]
    print(json.dumps({'total': len(before), 'baseline_repair_rate_at_1': rates[0],
                      'finetuned_repair_rate_at_1': rates[1], 'delta_percentage_points': 100 * (rates[1] - rates[0]),
                      'improved_pair_ids': improved, 'regressed_pair_ids': regressed}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
