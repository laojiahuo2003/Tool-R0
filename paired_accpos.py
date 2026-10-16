#!/usr/bin/env python3
"""acc_pos 的配对检验：sol_acc_pos = 拿到任意部分分（acc > 0）的比例。

为什么要有这个文件：`paired_test.py --metric acc_pos` **是坏的**，而且坏得很安静——
逐条 json 里没有 `acc_pos` 这个字段（只有 `acc`），`x.get("acc_pos")` 返回 None，
`num()` 把 None 当 0.0，于是两边均值都报 0.0000、"两者都非零 0"、p=1.0000、
判定"与 0 分不开，算持平"。看着像一次正常的检验，其实是 0 和 0 比。

`paired_test.py --metric acc` 也拿不到它：那行的"均值"打的是连续均值（sol_acc），
而 McNemar 分支里的 `ma = (both + only_a)/n` 本来正是 acc_pos，却在分档分支里被丢掉没打印。
只有"两者都非零 %d"那一行能反推出 acc_pos = (both + only_a) / n。

所以这里显式地对 `acc > 0` 做配对检验。McNemar 在这里是**正确**的检验
（acc_pos 天生是二值的），不需要置换检验。

用法：
  python paired_accpos.py --a iter1_solver --b iter2_solver
"""
import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EV = os.path.join(HERE, "evals")


def load_solver(tag):
    p = os.path.join(EV, tag + ".json")
    if not os.path.exists(p):
        sys.exit("没有 %s" % p)
    d = json.load(open(p))
    rows = d.get("solver")
    if rows is None:
        sys.exit("%s 没有 solver 的逐条结果" % p)
    return rows


def binom_two_sided(k, n):
    """精确二项检验双侧 p 值（p=0.5）。n=0 时返回 1。"""
    if n == 0:
        return 1.0

    def logc(a, b):
        return math.lgamma(a + 1) - math.lgamma(b + 1) - math.lgamma(a - b + 1)

    logp = lambda i: logc(n, i) - n * math.log(2)
    obs = logp(k)
    tot = 0.0
    for i in range(n + 1):
        lp = logp(i)
        if lp <= obs + 1e-12:
            tot += math.exp(lp)
    return min(1.0, tot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    A, B = load_solver(args.a), load_solver(args.b)
    if len(A) != len(B):
        sys.exit("题数不一样（%d vs %d）" % (len(A), len(B)))

    # 和 paired_test.py 一致：按 question 对齐
    mb = {x["question"]: x for x in B}
    pairs = [(x, mb[x["question"]]) for x in A if x["question"] in mb]
    if len(pairs) < len(A):
        print("[warn] 只配上 %d/%d 条" % (len(pairs), len(A)))

    both = only_a = only_b = neither = 0
    for x, y in pairs:
        oa, ob = x["acc"] > 0, y["acc"] > 0
        if oa and ob:
            both += 1
        elif oa and not ob:
            only_a += 1
        elif (not oa) and ob:
            only_b += 1
        else:
            neither += 1

    n = len(pairs)
    pa = (both + only_a) / n
    pb = (both + only_b) / n
    p = binom_two_sided(only_b, only_a + only_b)

    print("acc_pos 配对检验 %s -> %s  n=%d" % (args.a, args.b, n))
    print("  %s acc_pos %.4f    %s acc_pos %.4f    差 %+.4f（%+.1f 题）"
          % (args.a, pa, args.b, pb, pb - pa, (pb - pa) * n))
    print("  两者都拿到分 %d   两者都零分 %d" % (both, neither))
    print("  只有 %s 拿到分 %d（退步）   只有 %s 拿到分 %d（进步）" % (args.a, only_a, args.b, only_b))
    print("  McNemar 精确双侧 p = %.4f" % p)
    if pb < pa - 1e-12 and p < args.alpha:
        print("  判定：*** 退化 ***")
        sys.exit(1)
    if pb > pa + 1e-12 and p < args.alpha:
        print("  判定：上升（p<%.2f）" % args.alpha)
    else:
        extra = "，方向为正但样本量不够" if pb > pa + 1e-12 else ""
        print("  判定：与 0 分不开，算持平%s" % extra)


if __name__ == "__main__":
    main()
