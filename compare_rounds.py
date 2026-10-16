#!/usr/bin/env python3
"""把 iter_metrics.jsonl 摆成一张多轮对照表。

这就是"多轮有没有用"的答案表。每一行是一个 checkpoint，列是同一批题上测出来的指标。
判断标准只有一条：同一列，后面的行不能低于前面的行（generator 看 gen_four_tag /
gen_pv_ok，solver 看 sol_acc / sol_name_hit）。

用法：python compare_rounds.py [--cols gen_four_tag,gen_pv_ok]
"""
import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, "iter_metrics.jsonl")

DEFAULT_COLS = ["gen_four_tag", "gen_pv_ok", "sol_fmt", "sol_acc",
                "sol_acc_pos", "sol_name_hit"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cols", default=",".join(DEFAULT_COLS))
    ap.add_argument("--path", default=PATH)
    args = ap.parse_args()
    cols = [c for c in args.cols.split(",") if c]

    if not os.path.exists(args.path):
        print("还没有 %s" % args.path, file=sys.stderr)
        sys.exit(1)
    rows = [json.loads(l) for l in open(args.path) if l.strip()]
    if not rows:
        print("%s 是空的" % args.path, file=sys.stderr)
        sys.exit(1)

    # 同一个 tag 重测要覆盖旧的。eval_iter.py 是追加写账本的（open(METRICS,"a")），
    # 而"用更好的条件重测同一个 checkpoint"是计划里的固定步骤（iter1_solver 一开始是
    # 未设种测的，要设种重测）。不去重的话表里会出现两行 iter1_solver，单调性检查
    # 拿着旧值当中间点，"升了又降回去"根本说不清是哪一次测的。
    # 取最后一次：tag 的含义是"这个 checkpoint 目前最好的测量值"。
    dedup = {}
    for r in rows:
        dedup[r["tag"]] = r
    if len(dedup) != len(rows):
        print("[注] 账本里有重复 tag，保留最后一次测量（%d 行 -> %d 行）"
              % (len(rows), len(dedup)))
    rows = list(dedup.values())

    # 只显示有数据的列
    present = [c for c in cols if any(c in r for r in rows)]
    w = max(22, max(len(r["tag"]) for r in rows) + 2)
    print("%-*s %s" % (w, "tag", " ".join("%18s" % c.replace("gen_", "g.").replace("sol_", "s.")
                                          for c in present)))
    print("-" * (w + 19 * len(present)))
    for r in rows:
        cells = []
        for c in present:
            v = r.get(c)
            cells.append("%18s" % ("-" if v is None else "%.4f" % v))
        print("%-*s %s" % (w, r["tag"], " ".join(cells)))

    # 逐列判断单调性。_rep<N> 是同一个 checkpoint 再测一遍，用来量评测自身噪声的
    # （iter1_solver_rep2），它不是一轮，两次测量差多少本来就是随机的——把它算进
    # "后面的行不能低于前面的行"里，等于用噪声判自己的方法对不对，会误报回落。
    def is_round(tag):
        return re.search(r"_rep\d+$", tag) is None

    print()
    for c in present:
        seq = [(r["tag"], r[c]) for r in rows
               if isinstance(r.get(c), float) and is_round(r["tag"])]
        if len(seq) < 2:
            continue
        drops = [((ta, a), (tb, b)) for (ta, a), (tb, b) in zip(seq, seq[1:]) if b < a - 1e-9]
        if not drops:
            print("  %-18s 单调不降  (%s -> %s)" % (c, seq[0][0], seq[-1][0]))
        else:
            print("  %-18s 有回落: %s" % (
                c, "; ".join("%s %.4f -> %s %.4f" % (ta, a, tb, b)
                             for (ta, a), (tb, b) in drops)))

    # solver 的分档明细
    print("\nsolver accuracy 按评测集难度分档（看涨在哪里）：")
    bands = None
    for r in rows:
        if "sol_acc_by_band" not in r:
            continue
        d = r["sol_acc_by_band"]
        if bands is None:
            bands = list(d.keys())
        print("  %-*s %s" % (w, r["tag"],
                             " ".join("%s=%s" % (k, "-" if d.get(k) is None else "%.3f" % d[k])
                                      for k in bands)))


if __name__ == "__main__":
    main()
