#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""教材构成的跨轮对比：多调用比例、3+键比例、按全等口径的难度分布。

为什么单独一个脚本：round4 暴露了一件事——generator 的多调用产出在涨
（iter3 7.5% → round4 10.1%），但 verify 门按 98.3% 剔掉，教材里一条不剩。
光看终选教材的条数看不出这个，必须把"原始产出 → verify 后 → 终选"三段并排。

手算过的坑（所以这里全做成断言/兜底）：
  1. intermediate 文件的字段是 `calls`，终选教材是 `answer`——
     只用 `answer` 会把原始产出全解析成空，算出"0 条多调用"的假结论。
     我第一版就是这么错的。
  2. `meta` 在 intermediate（原始）里没有 solver_agreement*，
     只有 verified 之后才有。没有就别分档，别填 0 假装有。

用法: .venv/bin/python curriculum_composition.py 3 4 4b
"""
import json
import os
import sys

BASE = "qwen2.5-0.5b-instruct-tool-r0"


def calls_of(it):
    a = it.get("calls")
    if a is None:
        a = it.get("answer")
    if isinstance(a, str):
        try:
            a = json.loads(a)
        except Exception:
            return []
    return a if isinstance(a, list) else []


def nkeys(it):
    return sum(len(c.get("arguments", {}) or {}) for c in calls_of(it))


def band(it):
    f = it.get("meta", {}).get("solver_agreement_full")
    if f is None:
        return None
    if f < 0.4:
        return "hard"
    if f < 0.8:
        return "medium"
    return "easy"


def stat(round_tag):
    """round_tag: '3' / '4' / '4b' -> 前缀 iter3 / iter4 / iter3b"""
    pre = "iter3b" if round_tag == "4b" else "iter" + round_tag
    out = {}
    # 三个阶段的文件名不统一，这是踩过的坑：
    #   round4/iter3 的 verify 输出是 <pre>_data.json.intermediate.verified.json
    #   round4b 的输出是          iter3b_data.json.verified.json（没有 .intermediate）
    #   round4b 的"原始产出"其实就是 iter3 的 intermediate（它复用 iter3 的题，不重新生成）
    # 所以逐个候选路径试，别拼一个就算。拼错时的失败模式是静默显示"文件不在"。
    def find(cands):
        for p in cands:
            if os.path.exists(p):
                return p
        return None

    raw_pre = "iter3" if round_tag == "4b" else pre
    cands = {
        "原始产出": [os.path.join(BASE, raw_pre + "_data.json.intermediate.json")],
        "verify 后": [os.path.join(BASE, pre + "_data.json.intermediate.verified.json"),
                      os.path.join(BASE, pre + "_data.json.verified.json")],
        "终选教材": [os.path.join(BASE, pre + "_data.json")],
    }
    out = {}
    for label in ("原始产出", "verify 后", "终选教材"):
        p = find(cands[label])
        if p is None:
            out[label] = None
            continue
        d = json.load(open(p))
        items = d if isinstance(d, list) else d.get("items", [])
        bands = {}
        for r in items:
            b = band(r)
            if b:
                bands[b] = bands.get(b, 0) + 1
        out[label] = {
            "n": len(items),
            "mc": sum(1 for r in items if len(calls_of(r)) >= 2),
            "k3": sum(1 for r in items if nkeys(r) >= 3),
            "bands": bands,
            "path": p,
        }
    return out


def main():
    tags = sys.argv[1:] or ["3", "4"]
    allst = {t: stat(t) for t in tags}

    hdr = "%-12s %-12s %6s %14s %12s" % ("轮次", "阶段", "条数", "多调用", "3+键")
    print(hdr)
    print("-" * len(hdr))
    for t in tags:
        for label in ("原始产出", "verify 后", "终选教材"):
            s = allst[t][label]
            if s is None:
                print("%-12s %-12s %s" % (t, label, "（文件不在）"))
                continue
            print("%-12s %-12s %6d %8d（%4.1f%%） %4d（%4.1f%%）"
                  % (t, label, s["n"], s["mc"], 100.0 * s["mc"] / s["n"],
                     s["k3"], 100.0 * s["k3"] / s["n"]))
        print()

    print("按全等口径的难度分布（只有 verify 后的文件才有 meta.solver_agreement_full）：")
    for t in tags:
        for label in ("verify 后", "终选教材"):
            s = allst[t][label]
            if s is None or not s["bands"]:
                continue
            b = s["bands"]
            tot = sum(b.values())
            print("  %-4s %-10s hard %4d / medium %4d / easy %4d"
                  "  （占比 %d/%d/%d）"
                  % (t, label, b.get("hard", 0), b.get("medium", 0), b.get("easy", 0),
                     100 * b.get("hard", 0) // tot,
                     100 * b.get("medium", 0) // tot,
                     100 * b.get("easy", 0) // tot))

    # verify 门对多调用做了什么
    print("\nverify 门对多调用条的剔除率（原始产出 -> verify 后）：")
    for t in tags:
        a, b = allst[t]["原始产出"], allst[t]["verify 后"]
        if a is None or b is None:
            continue
        if a["mc"]:
            print("  %-4s 多调用 %d -> %d，剔掉 %.1f%%"
                  % (t, a["mc"], b["mc"], 100.0 * (a["mc"] - b["mc"]) / a["mc"]))
        else:
            print("  %-4s 原始产出里没有多调用条，无从计算" % t)
    print("  * 单调用条的剔除率对照：")
    for t in tags:
        a, b = allst[t]["原始产出"], allst[t]["verify 后"]
        if a is None or b is None:
            continue
        sa, sb = a["n"] - a["mc"], b["n"] - b["mc"]
        if sa:
            print("      %-4s 单调用 %d -> %d，剔掉 %.1f%%"
                  % (t, sa, sb, 100.0 * (sa - sb) / sa))


if __name__ == "__main__":
    main()
