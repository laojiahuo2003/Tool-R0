#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""训练日志里，多调用题的实际遭遇：第二个 call 出现几次、拿到多少奖励、advantage 正负。

为什么看这个而不是推算：`completions/completions_*.parquet` 是 TRL `--log_completions`
落的盘，每行一个 prompt 的一条 completion，带 format/accuracy reward 和 advantage。
它直接回答"这一轮训练里，2-call 行为到底有没有被强化过"——
比从 P(2 calls) 推算硬，也不用重新跑 GPU。

**这是 round4b 结论的核心证据**（PLAN.md 第 7 节）：
round4b 全量 400 槽里 8 个 completion 吐了 2 个 call，
**8 个全落在单调用题上、0 个落在多调用题上、7 个负优势**。
所以 5 条多调用教材不是"太少所以没用"，是被 95 条单调用教材**主动压制**。

注意：TRL 每个 prompt 只记 1 条 completion（num_generations=4 但不全记），
所以这里数的是下界，不是全集。但"0/20 个多调用槽出现 2-call"这个零
本身已经很说明问题。

用法: .venv/bin/python mc_training_signal.py [iter3b_solver|iter3_solver|iter4_solver]
"""
import glob
import json
import os
import sys
import collections

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import pyarrow.parquet as pq  # noqa: E402

from step2_genverify import (  # noqa: E402
    extract_tool_call_answer_block, clean_json_blob, coerce_json)

BASE = "qwen2.5-0.5b-instruct-tool-r0"


def ncalls(txt):
    """一条 completion 里实际吐了几个 call（0 = 没解析出来/格式就不对）。"""
    blk = extract_tool_call_answer_block(txt)
    if blk is None:
        return 0
    p = coerce_json(clean_json_blob(blk))
    if isinstance(p, list):
        return len(p)
    if isinstance(p, dict):
        return 1
    return 0


def calls_of(it):
    """教材/池文件两个字段都要试：verified 池存 `calls`，终选教材存 `answer`。
    只用 `answer` 会把池文件全解析成空列表，静默得出"0 条"的假结论。"""
    a = it.get("calls")
    if a is None:
        a = it.get("answer")
    if isinstance(a, str):
        try:
            a = json.loads(a)
        except Exception:
            return []
    return a if isinstance(a, list) else []


def data_path_for(ckpt):
    """ckpt 名 -> 教材文件。iter3b_solver -> iter3b_data.json。"""
    tag = ckpt.split("_")[0]
    return os.path.join(BASE, tag + "_data.json")


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "iter3b_solver"
    dp = data_path_for(ckpt)
    if not os.path.exists(dp):
        print("缺教材文件 %s" % dp)
        return 2
    cur = json.load(open(dp))
    items = cur if isinstance(cur, list) else cur.get("items", [])
    mc_q = set(r["question"] for r in items if len(calls_of(r)) >= 2)
    print("教材多调用题 %d 条（来自 %s）" % (len(mc_q), dp))

    files = sorted(glob.glob(os.path.join(BASE, ckpt, "completions", "completions_*.parquet")),
                   key=lambda p: int(p.split("_")[-1].split(".")[0]))
    if not files:
        print("缺 %s/%s/completions/ 下的 parquet" % (BASE, ckpt))
        return 2
    print("parquet %d 个（= %d 步）" % (len(files), len(files)))

    rows = []
    for f in files:
        step = int(os.path.basename(f).split("_")[-1].split(".")[0])
        dd = pq.read_table(f).to_pandas()
        for _, r in dd.iterrows():
            is_mc = any(q in r["prompt"] or q[:60] in r["prompt"] for q in mc_q)
            rows.append({"step": step, "mc": is_mc, "k": ncalls(r["completion"]),
                         "fmt": r["format_reward_func"],
                         "acc": r["accuracy_reward_func"], "adv": r["advantage"]})

    mc = [r for r in rows if r["mc"]]
    print("\n多调用题的 prompt 槽: %d 个" % len(mc))
    print("  被记到的 completion 里 call 个数分布: %s"
          % dict(collections.Counter(r["k"] for r in mc)))
    print("  其中 >=2 个 call 的: %d 个  <-- 这个数是关键" % sum(1 for r in mc if r["k"] >= 2))
    one = [r for r in mc if r["k"] == 1]
    if one:
        print("  只吐 1 个 call 的 %d 个：acc 均值 %.4f，advantage>0 的 %d 个"
              % (len(one), sum(r["acc"] for r in one) / len(one),
                 sum(1 for r in one if r["adv"] > 0)))

    print("\n全量 %d 槽：call 个数分布 %s"
          % (len(rows), dict(collections.Counter(r["k"] for r in rows))))
    two = [r for r in rows if r["k"] >= 2]
    print("\n全量里吐了 >=2 个 call 的 completion 共 %d 个：" % len(two))
    for r in two:
        print("  step %2d  calls=%d  fmt=%.2f  acc=%.4f  advantage=%+.4f  %s"
              % (r["step"], r["k"], r["fmt"], r["acc"], r["adv"],
                 "多调用题" if r["mc"] else "单调用题"))
    print("\n  advantage>0（被强化）: %d 个   advantage<0（被压制）: %d 个"
          % (sum(1 for r in two if r["adv"] > 0), sum(1 for r in two if r["adv"] < 0)))
    print("  落在多调用题上的: %d / %d" % (sum(1 for r in two if r["mc"]), len(two)))
    print("\n  解读：2-call 行为只在单调用题上出现，而单调用题 gold 只有 1 条，")
    print("  多吐一个 call 按 extra=1 被 penalty=0.8 打折 => 奖励更低 => 负优势。")
    print("  即这个行为在大部份教材上被主动压制，而在该被奖励的多调用题上一次都没出现。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
