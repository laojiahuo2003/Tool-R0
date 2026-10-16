#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""held-out solver 主判据：条件 acc = mean(acc | name_hit)。

为什么单独一个脚本：round4/round4b 的**主判据是条件 acc > 0.7450，不是 sol_acc**。
sol_acc 会被点名率拖着走（点名率从 0.79 到 0.84 只靠"敢答"也能涨），
条件 acc 才回答"点名之后参数对不对"——而那正是口径修复要修的东西。

分解式（已用 iter2->iter3 实测闭合，见下）：
    acc = 命中支 + 未中支
    命中支 = nh * cond          命中题的总贡献
    未中支 = (1-nh) * cond_miss 未命中题的总贡献
    命中支内部再分：
        nh 贡献    = (nh_new - nh_old) * cond_old
        cond 贡献  = nh_new * (cond_new - cond_old)
    acc 在未命中题上不为 0：奖励是稠密的（0.2*名 + 0.3*键名F1 + 0.5*键值），
    名错仍可能拿到键名/键值分，所以必须显式算 cond_miss，不能当 0。

用法: .venv/bin/python cond_acc_check.py iter3_solver iter4_solver
"""
import json
import os
import sys

EV = "evals"
HOLDOUT = "holdout_solver.json"
MAIN_THRESHOLD = 0.7450  # PLAN.md 事前钉的；iter3 基线 0.7315


def load(tag):
    p = os.path.join(EV, tag + ".json")
    if not os.path.exists(p):
        return None
    return json.load(open(p)).get("solver")


def stats(ev):
    hit = [r for r in ev if r.get("name_hit")]
    miss = [r for r in ev if not r.get("name_hit")]
    nh = len(hit) / len(ev)
    cond = sum(r["acc"] for r in hit) / len(hit) if hit else float("nan")
    cmiss = sum(r["acc"] for r in miss) / len(miss) if miss else float("nan")
    return {
        "n": len(ev), "nhit": len(hit),
        "acc": sum(r["acc"] for r in ev) / len(ev),
        "nh": nh, "cond": cond, "cmiss": cmiss,
        "hit_block": nh * cond, "miss_block": (1 - nh) * cmiss,
    }


def main():
    tags = sys.argv[1:]
    if len(tags) != 2:
        print("用法: cond_acc_check.py <旧轮tag> <新轮tag>")
        return 2
    evs = [load(t) for t in tags]
    if any(e is None for e in evs):
        for t, e in zip(tags, evs):
            if e is None:
                print("!!! 缺 evals/%s.json" % t)
        return 2
    a, b = stats(evs[0]), stats(evs[1])

    print("固定评测集 %d 题（两个 tag 必须同一套题）" % a["n"])
    if a["n"] != b["n"]:
        print("!!! 题数不等（%d vs %d），不可比" % (a["n"], b["n"]))
        return 2

    print("\n%-18s %8s %8s %10s %10s" % ("", "acc", "点名率", "条件acc", "未中acc"))
    for t, s in zip(tags, (a, b)):
        print("%-18s %8.4f %8.4f %10.4f %10.4f" % (t, s["acc"], s["nh"], s["cond"], s["cmiss"]))

    d_acc = b["acc"] - a["acc"]
    d_hit = b["hit_block"] - a["hit_block"]
    d_miss = b["miss_block"] - a["miss_block"]
    d_cond = b["cond"] - a["cond"]
    nh_part = (b["nh"] - a["nh"]) * a["cond"]
    cond_part = b["nh"] * d_cond

    print("\n=== %s -> %s ===" % (tags[0], tags[1]))
    print("  acc      %+.4f = 命中支 %+.4f + 未中支 %+.4f   (闭合误差 %.2e)"
          % (d_acc, d_hit, d_miss, abs(d_acc - d_hit - d_miss)))
    print("  点名率   %.4f -> %.4f（%+d/%d 道）"
          % (a["nh"], b["nh"], b["nhit"] - a["nhit"], b["n"]))
    print("  条件acc  %.4f -> %.4f（%+.4f）" % (a["cond"], b["cond"], d_cond))
    if abs(d_hit) > 1e-9:
        print("  命中支 %+.4f 拆开：点名率贡献 %+.4f（%.0f%%）+ 条件acc贡献 %+.4f（%.0f%%）"
              % (d_hit, nh_part, 100 * nh_part / d_hit, cond_part, 100 * cond_part / d_hit))
    print("  * 分解式acc = 命中支 + 未中支 是恒等式，不是模型；份额比例才是结论")

    print("\n=== 主判据：条件 acc > %.4f ===" % MAIN_THRESHOLD)
    ok = b["cond"] > MAIN_THRESHOLD
    print("  %s 条件 acc %.4f %s %.4f  ==> %s"
          % (tags[1], b["cond"], ">" if ok else "<=", MAIN_THRESHOLD,
             "过" if ok else "不过"))
    if ok:
        print("  超出基线 %.4f（基线 %.4f）" % (b["cond"] - MAIN_THRESHOLD, a["cond"]))
    else:
        print("  距基线 %.4f 还差 %.4f" % (a["cond"], MAIN_THRESHOLD - b["cond"]))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
