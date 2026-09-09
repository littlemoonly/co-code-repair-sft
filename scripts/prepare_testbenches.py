#!/usr/bin/env python3
# 为固定测试集整理测试资源：按 catalog 题名从对应年份分支提取测试用例，
# 合并 generated 补充用例，优先采用 testcase.yaml 的判题参数，生成显式映射。
import argparse
import io
import json
from pathlib import Path
import shutil
import tarfile
import zipfile
import yaml

ROOT = Path(__file__).resolve().parents[1]


def prepare(output, all_splits=False):
    files = sorted((ROOT / 'data/repair_pairs_sft').glob('*.jsonl')) if all_splits else [ROOT / 'data/repair_pairs_sft/test.jsonl']
    rows = [json.loads(line) for file in files for line in file.read_text().splitlines()]
    catalog = {str(x['id']): x['name'] for x in map(json.loads, (ROOT / 'data/problem_catalog.jsonl').read_text().splitlines())}
    names = {catalog[str(r['problem_id'])] for r in rows}
    output.mkdir(parents=True, exist_ok=True)
    # 历年新增题在年份分支，其余题只取 main，避免重复同名用例。
    branches = {name: ('2023-main' if name.endswith('_2023') else
                       '2024-main' if name in {'P1_different_number', 'P1_eat_meal', 'P1_sit_in_rows',
                                              'P1_password_check', 'P1_inverted_order', 'P1_spend_a_day'} else 'main')
                for name in names}
    with tarfile.open(ROOT / 'testbench/co-problem-set-all-branches.tar.gz') as archive:
        for member in archive:
            parts = Path(member.name).parts
            if len(parts) < 4 or parts[1] != 'problem' or parts[2] not in names or parts[0] != branches[parts[2]] or not member.isfile():
                continue
            relative = Path(*parts[2:])
            if '..' in relative.parts:
                raise ValueError(member.name)
            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.extractfile(member).read())
    for source in (ROOT / 'testbench/generated/problem').iterdir():
        if source.name in names:
            shutil.copytree(source, output / source.name, dirs_exist_ok=True)
    manifest = {}
    for pid in sorted({str(r['problem_id']) for r in rows}, key=int):
        cases = []
        for config in sorted((output / catalog[pid]).glob('testcase/*/testcase.yaml')):
            attachment = config.parent / 'attachment'
            if not attachment.exists():
                attachment = config.parent / 'standard'
            for zipped in sorted(attachment.glob('*.zip')):
                with zipfile.ZipFile(io.BytesIO(zipped.read_bytes())) as z:
                    for name in z.namelist():
                        if '..' in Path(name).parts or Path(name).is_absolute():
                            raise ValueError(name)
                        if not name.endswith('/'):
                            dest = attachment / name
                            # 已提供的明文文件优先于 zip。
                            if not dest.exists():
                                dest.parent.mkdir(parents=True, exist_ok=True)
                                dest.write_bytes(z.read(name))
            params = yaml.safe_load(config.read_text())['judge_parameter']['universal']
            # Icarus 的负数 %d 不带前导空格，ISim 标准输出有空格；仅放宽行首空白。
            if pid == '326' and config.parent.name == 'P1_L1_find_max_1':
                params['output_regex'] = r'(?m)^ *-?[0-9]+\n'
            standards = list(attachment.rglob('standard.txt'))
            sources = sorted(p for p in attachment.rglob('*.v')
                             if '__MACOSX' not in p.parts and not p.name.startswith('._'))
            if not standards or not sources:
                raise ValueError(f'缺少 standard.txt 或 Verilog 文件: {config}')
            cases.append({'name': config.parent.name, 'directory': str(attachment.relative_to(ROOT)),
                          'standard': str(standards[0].relative_to(attachment)),
                          'sources': [str(p.relative_to(attachment)) for p in sources], **params})
        manifest[pid] = {'name': catalog[pid], 'cases': cases}
    path = output / 'manifest.json'
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    missing = [pid for pid, p in manifest.items() if not p['cases']]
    print(f'{path}: {len(manifest)} problems, {sum(len(p["cases"]) for p in manifest.values())} cases; missing={missing}')
    if missing:
        raise SystemExit('测试覆盖不完整，请补全 manifest 后再运行评测。')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'testbench/prepared')
    parser.add_argument('--all-splits', action='store_true', help='整理 train/dev/test 全部题目的用例')
    args = parser.parse_args()
    prepare(args.output.resolve(), args.all_splits)
