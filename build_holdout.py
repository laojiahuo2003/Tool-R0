#!/usr/bin/env python3
"""造一个固定的 solver held-out 评测集，之后每一轮都用同一批题。

为什么需要：step3 每轮训的是当轮 generator 新出的题，直接比"这轮 reward"比不了——
题目换了。要比"solver 有没有变强"，必须有一批从今以后再也不变的题。

选集的来源和排除规则（查过 wandb run，不是猜的）：
  * iter1 的 step3 只跑过一次，训的是 iter1_data_lr1e4_ckpt50_tv0.7.json
    （wandb/offline-run-20261014_181257-9bign72c，18:12 起、18:26 收尾）。
  * 所以 iter1_data_lr1e4_ckpt50_tv0.001.json 的 verified pool 从没被任何 solver
    训过，可以当评测集。
  * 但要剔掉 tv0.7 教材里那 100 条——它们的题干在 tv0.001 pool 里有 416 条重叠，
    那 100 条是 iter1 solver 的训练数据，混进来等于送分。

分层抽样而不是顺序取前 N 条：verified pool 按 solver_agreement 排序过，
前 N 条会全挤在同一个档，评测集就没有难度跨度、分不出好坏。
"""
import argparse
import collections
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from step2_select_curriculum import build_solver_question  # noqa: E402

# agreement 在 k_verify=10 下是 0.1 的整数倍。按"还有多少余量"划档，每档有上限：
# 档位故意不均等——verified pool 里 agreement=1.0 的有 342 条（69%），那种题 solver
# 已经全会，测不出好坏，只留 10 条当"别退化"的哨兵。step2 的 drop_agreement_ge 1.0
# 本来就把它们从训练集里剔了，评测集的主体力也该放在有余量的档上。
BANDS = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 1.0), (1.0, 1.01)]
BAND_CAP = [40, 50, 50, 30, 20, 10]


def band_of(a):
    for i, (lo, hi) in enumerate(BANDS):
        if lo <= a < hi:
            return i
    return len(BANDS) - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="iter1_data_lr1e4_ckpt50_tv0.001.json.verified.json")
    ap.add_argument("--trained", action="append",
                    default=["iter1_data_lr1e4_ckpt50_tv0.7.json"],
                    help="已被某个 solver 训过的教材，要排除（可重复传）")
    ap.add_argument("--out", default="holdout_solver.json")
    ap.add_argument("--seed", type=int, default=20261015)
    args = ap.parse_args()

    pool = json.load(open(os.path.join(HERE, args.pool)))
    trained_q = set()
    for p in args.trained:
        d = json.load(open(os.path.join(HERE, p)))
        trained_q |= set(x["question"] for x in d)
    print("pool %d 条；要排除的已训练题干 %d 条" % (len(pool), len(trained_q)))

    buckets = {i: [] for i in range(len(BANDS))}
    skipped_trained = 0
    for it in pool:
        q = build_solver_question(it["question"], it["tools"])
        if q in trained_q:
            skipped_trained += 1
            continue
        a = it.get("meta", {}).get("solver_agreement")
        if a is None:
            continue
        buckets[band_of(a)].append({
            "question": q,
            "answer": it["calls"],
            "domain": it.get("meta", {}).get("domain", "unknown"),
            "solver_agreement": a,
        })
    print("排除已训练 %d 条" % skipped_trained)
    for i, (lo, hi) in enumerate(BANDS):
        print("  档 [%.2f,%.2f): %d 条" % (lo, hi, len(buckets[i])))

    rng = random.Random(args.seed)
    out = []
    for i, (lo, hi) in enumerate(BANDS):
        b = buckets[i][:]
        rng.shuffle(b)
        out.extend(b[:BAND_CAP[i]])
    rng.shuffle(out)

    path = os.path.join(HERE, args.out)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    print("\n写出 %d 条 -> %s" % (len(out), path))
    print("域覆盖 %d 个" % len(set(x["domain"] for x in out)))
    print("agreement 均值 %.3f，分布 %s" % (
        sum(x["solver_agreement"] for x in out) / len(out),
        dict(sorted(collections.Counter(
            round(x["solver_agreement"], 1) for x in out).items()))))
    # 分档报一遍，之后每轮都按同样的档看"涨在哪里"
    print("\n各档进入评测集的条数：")
    for i, (lo, hi) in enumerate(BANDS):
        n = sum(1 for x in out if lo <= x["solver_agreement"] < hi)
        print("  [%.2f,%.2f): %d" % (lo, hi, n))


if __name__ == "__main__":
    main()
