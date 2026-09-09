# 将修复 pair 数据划分为 train / val / test。
#
# 核心逻辑：
# 1. 读取 pairs_edit_distance_filtered.jsonl。
# 2. 按 student_id 随机划分学生：80% train、10% val、10% test，
#    保证同一学生的所有 pair 只属于一个数据集，避免数据泄漏。
# 3. 仅对训练集按 problem_id 做采样，每题最多保留 250 个 pair，
#    减弱少数高频题目对 SFT 训练的主导作用。
# 4. validation 和 test 不做 cap，保持原始数据分布。
#
# 输出：
#   data/repair_pairs_raw/metadata/splits/train.jsonl
#   data/repair_pairs_raw/metadata/splits/val.jsonl
#   data/repair_pairs_raw/metadata/splits/test.jsonl

import json
import random
from collections import defaultdict
from pathlib import Path

INPUT_PATH = Path("data/repair_pairs_raw/metadata/pairs_edit_distance_filtered.jsonl")
OUTPUT_DIR = Path("data/repair_pairs_raw/metadata/splits")

SEED = 42
TRAIN_RATIO = 0.85
VAL_RATIO = 0.075
TRAIN_CAP_PER_PROBLEM = 250


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def cap_by_problem(data, cap, rng):
    """每个题目随机最多保留 cap 条训练样本。"""
    groups = defaultdict(list)
    for item in data:
        groups[item["problem_id"]].append(item)

    result = []
    for pairs in groups.values():
        pairs.sort(key=lambda x: x["pair_id"])
        rng.shuffle(pairs)
        result.extend(pairs[:cap])

    rng.shuffle(result)
    return result


def main():
    rng = random.Random(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_jsonl(INPUT_PATH)

    # 按学生划分，而不是直接按 pair 划分
    students = sorted({item["student_id"] for item in data})
    rng.shuffle(students)

    n = len(students)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    train_students = set(students[:train_end])
    val_students = set(students[train_end:val_end])
    test_students = set(students[val_end:])

    train = [x for x in data if x["student_id"] in train_students]
    val = [x for x in data if x["student_id"] in val_students]
    test = [x for x in data if x["student_id"] in test_students]

    # 只对训练集限制每题最大样本数
    train_before_cap = len(train)
    train = cap_by_problem(train, TRAIN_CAP_PER_PROBLEM, rng)

    save_jsonl(train, OUTPUT_DIR / "train.jsonl")
    save_jsonl(val, OUTPUT_DIR / "val.jsonl")
    save_jsonl(test, OUTPUT_DIR / "test.jsonl")

    print(f"Total pairs: {len(data)}")
    print(f"Students: {len(students)}")
    print(f"Train: {train_before_cap} -> {len(train)} after cap")
    print(f"Val:   {len(val)}")
    print(f"Test:  {len(test)}")
    print(f"Train problems: {len({x['problem_id'] for x in train})}")
    print(f"Val problems:   {len({x['problem_id'] for x in val})}")
    print(f"Test problems:  {len({x['problem_id'] for x in test})}")


if __name__ == "__main__":
    main()
