#!/usr/bin/env python3
""" correctness 门：确认某一轮的教材没有和固定评测集重题。

为什么必须有这个检查：solver 的分数是在 `holdout_solver.json` 上量的，而 step3 训的
是当轮 generator 新出的题。两者同源（都是那个 generator 产的），所以**有可能撞题**。
一旦撞了，"held-out 比 in-sample 涨得多所以是泛化"这个结论就是假的——那一轮等于
在评测集上训过。这种事日志里看不出来，只有逐条比对能发现。

用法：
  python check_holdout_clean.py <curriculum.json> [更多.json]
  python check_holdout_clean.py iter2_data.json

判定：有重题就非零退出，并把重题打出来。没重题退出 0。
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
HOLDOUT = os.path.join(HERE, "holdout_solver.json")


def norm(q):
    # 题面在落盘前后可能被 strip 过，比对前统一一下，别把纯空白差当成两道题
    return " ".join(q.split())


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    hold = json.load(open(HOLDOUT))
    hq = {norm(d["question"]) for d in hold}
    print("固定评测集：%d 条" % len(hold))

    bad = 0
    missing = 0
    for path in sys.argv[1:]:
        try:
            data = json.load(open(path))
        except Exception as e:
            # 读不到和撞题是两种故障，必须分开报。原来这里也走 bad += 1，
            # 于是"路径写错 / 文件还没生成"会被说成"教材污染了评测集"——
            # 一个错的诊断比没有诊断更糟：它会让人去查教材，而问题在路径上。
            # 仍然非零退出（教材缺失对这一轮同样是致命的），但话说清楚。
            print("  %-46s 读不到：%s" % (path, e))
            missing += 1
            continue
        tq = {norm(d["question"]) for d in data}
        ov = hq & tq
        status = "干净" if not ov else "!!! %d 条重题" % len(ov)
        print("  %-46s %4d 条  %s" % (path, len(data), status))
        for q in sorted(ov)[:3]:
            print("      重题：%s..." % q[:100])
        if ov:
            bad += 1

    if bad:
        print("\n有 %d 个文件的教材污染了固定评测集。这一轮的 sol_* 分数不能当 held-out 用，"
              "要先换评测集或换教材再比。" % bad)
        sys.exit(1)
    if missing:
        print("\n有 %d 个文件读不到（路径写错或还没生成）。"
              "这不是污染，是教材本身没找到——先确认路径，别去查重题。" % missing)
        sys.exit(2)
    print("\n全部干净：这些教材没有一条出现在固定评测集里。")


if __name__ == "__main__":
    main()
