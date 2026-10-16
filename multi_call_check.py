#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多调用子集（len(gold calls) >= 2）的 held-out 表现。

为什么单独一个脚本：这是"可收复失地"最大的单块。iter3 上只有 7 道，
点名率 1/7、acc 均值 0.2786；对总 acc 的拖累 (1.0-0.2786)*7/135 = 0.0374，
比一整轮的增量还大。而 `tau_verify=0.2` 把多调用题按 95.3% 剔掉
（单调用只 10.8%），所以训练分布 0.4% 多调用 vs 评测集 5.2%——
这一块基本没被教过。

手算过的坑（所以这里全做成断言）：
  1. 分档表里有两个 n=7，多调用集和 0.85-1.0 档**不是同一批题**
     （多调用在低 agreement 档，0.85-1.0 全是单调用）。别混。
  2. 必须按 question 文本对齐，不能按下标——两个 json 的题序不一定一致。
  3. acc 是逐条奖励均值，name_hit 是布尔；点名率 = mean(name_hit)。

用法: .venv/bin/python multi_call_check.py iter3_solver iter4_solver
"""
import json
import os
import sys

EV = "evals"
HOLDOUT = "holdout_solver.json"


def calls_of(item):
    a = item.get("answer")
    if isinstance(a, str):
        try:
            a = json.loads(a)
        except Exception:
            return []
    return a if isinstance(a, list) else []


def nkeys(item):
    return sum(len(c.get("arguments", {}) or {}) for c in calls_of(item))


def load_eval(tag):
    p = os.path.join(EV, tag + ".json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    return d.get("solver")


def main():
    tags = sys.argv[1:] or ["iter3_solver"]
    hold = json.load(open(HOLDOUT))
    n_all = len(hold)
    multi_idx = [i for i, r in enumerate(hold) if len(calls_of(r)) >= 2]
    print("固定评测集 %d 题，其中多调用 %d 道（%.1f%%）"
          % (n_all, len(multi_idx), 100.0 * len(multi_idx) / n_all))

    rows = {}
    for tag in tags:
        ev = load_eval(tag)
        if ev is None:
            print("  [缺] evals/%s.json" % tag)
            continue
        by_q = {r["question"]: r for r in ev}
        if len(by_q) != len(ev):
            print("  !!! %s 里有重题，对齐不可靠" % tag)
            continue
        miss = [i for i in multi_idx if hold[i]["question"] not in by_q]
        if miss:
            print("  !!! %s 有 %d 道多调用题对不上" % (tag, len(miss)))
            continue
        sub = [by_q[hold[i]["question"]] for i in multi_idx]
        allr = [by_q[r["question"]] for r in hold]
        rows[tag] = {
            "n": len(sub),
            "name_hit": sum(1 for r in sub if r.get("name_hit")) / len(sub) if sub else 0.0,
            "acc": sum(r["acc"] for r in sub) / len(sub) if sub else 0.0,
            "acc_all": sum(r["acc"] for r in allr) / len(allr),
        }
        r = rows[tag]
        drag = (1.0 - r["acc"]) * len(sub) / n_all
        print("\n%s" % tag)
        print("  多调用 %d 道：点名率 %.3f  acc %.4f" % (r["n"], r["name_hit"], r["acc"]))
        print("  全集 acc %.4f；多调用对总 acc 的拖累 %.4f" % (r["acc_all"], drag))
        if len(sub) <= 10:
            print("  逐条：" + "  ".join(
                "c%d/ag%.1f/%s/%.2f" % (len(calls_of(hold[i])),
                                         by_q[hold[i]["question"]]["solver_agreement"],
                                         "Y" if by_q[hold[i]["question"]].get("name_hit") else "n",
                                         by_q[hold[i]["question"]]["acc"])
                for i in multi_idx))

    if len(rows) >= 2:
        a, b = list(rows.items())[-2], list(rows.items())[-1]
        print("\n=== %s -> %s ===" % (a[0], b[0]))
        print("  多调用点名率 %.3f -> %.3f（%+d/%d 道）"
              % (a[1]["name_hit"], b[1]["name_hit"],
                 round(b[1]["name_hit"] * b[1]["n"] - a[1]["name_hit"] * a[1]["n"]), b[1]["n"]))
        print("  多调用 acc   %.4f -> %.4f（%+.4f）" % (a[1]["acc"], b[1]["acc"], b[1]["acc"] - a[1]["acc"]))
        print("  全集 acc     %.4f -> %.4f（%+.4f）" % (a[1]["acc_all"], b[1]["acc_all"], b[1]["acc_all"] - a[1]["acc_all"]))
        # 多调用块对总 acc 的理论贡献
        contrib = (b[1]["acc"] - a[1]["acc"]) * b[1]["n"] / n_all
        print("  多调用块对总 acc 的贡献 %+.4f（占总增量 %s）"
              % (contrib, ("%.0f%%" % (100 * contrib / (b[1]["acc_all"] - a[1]["acc_all"])))
                 if abs(b[1]["acc_all"] - a[1]["acc_all"]) > 1e-9 else "n/a（总 acc 没动）"))
        print("  * n=%d，方向可读，幅度不可当统计结论" % b[1]["n"])


if __name__ == "__main__":
    main()
