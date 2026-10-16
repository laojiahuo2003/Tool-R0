#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多调用计分的三个性质（零 GPU，纯函数回归测试）。

为什么值得固化成测试：round4b 的结论「solver 见过但学不会」要靠"多调用题
到底有没有梯度"来定性。我最初的假设是 greedy 配对有顺序偏差（pred 被强行配给
gt[0]），实测**被推翻了**——`if r > best_r` 严格大于、best_r 初始 0.0，
所以零奖励的 pred 永不被占用。这个性质不写下来，下次还会按错的假设推理。

性质 1（上界，已确认）：2 条 gold 只吐 1 个 call => reward 恒 <= 0.5
   所以评测里 acc > 0.5 就是"这一条吐了 >=2 个 call"的充分必要条件。
   这是 /tmp/twocall_traj.py 那个全轮轨迹表的判据基础。
性质 2（顺序稳健，已确认）：同一个 pred 配哪个 gold 与 gold 书写顺序无关。
性质 3（有梯度，已确认）：吐第二个 call 能把 0.5 抬到 1.0，梯度空间 0.5 真实存在。
   所以"学不会"不是奖励看不见多调用，别再去改奖励函数。

用法: .venv/bin/python test_multicall_reward.py   退出码 0 = 全过
"""
import sys

sys.path.insert(0, ".")
from rewards_solver import compute_accuracy_score


def C(name, **args):
    return {"name": name, "arguments": args}


def run():
    fails = []
    gt = [C("search", q="rain"), C("lookup", city="paris")]

    # ---- 性质 1：单 call 上界 ----
    for label, pred, expect in [
        ("答对 gt[0]", [C("search", q="rain")], 0.5),
        ("答对 gt[1]", [C("lookup", city="paris")], 0.5),
        ("两个都对", [C("search", q="rain"), C("lookup", city="paris")], 1.0),
        ("多吐一个", [C("search", q="rain"), C("lookup", city="paris"),
                    C("search", q="x")], 0.8),
    ]:
        r, d = compute_accuracy_score(pred, gt)
        ok = abs(r - expect) < 1e-9
        print("  [%s] %-10s reward=%.4f（预期 %.4f）base=%.4f extra=%d penalty=%.2f"
              % ("PASS" if ok else "FAIL", label, r, expect,
                 d["base_score"], d["extra_calls"], d["extra_call_penalty"]))
        if not ok:
            fails.append("性质1:" + label)

    # ---- 性质 2：gold 书写顺序不影响单 call 的得分 ----
    a, _ = compute_accuracy_score([C("lookup", city="paris")],
                                  [C("search", q="rain"), C("lookup", city="paris")])
    b, _ = compute_accuracy_score([C("lookup", city="paris")],
                                  [C("lookup", city="paris"), C("search", q="rain")])
    ok = abs(a - b) < 1e-9 and abs(a - 0.5) < 1e-9
    print("  [%s] 顺序稳健   gold=[s,l] -> %.4f / gold=[l,s] -> %.4f（都应 0.5）"
          % ("PASS" if ok else "FAIL", a, b))
    if not ok:
        fails.append("性质2")

    # 零奖励 pred 不得占用名额，把能配的那个挤掉。
    # 注意别把期望值算错：gt[0] 配不上时贡献 0 但仍占分母，
    # 所以"2 条里完美答对 1 条"就是 1.0/2 = 0.5，不是 0.75。
    # 这里真正要验的是 reward > 0——若 lookup 被 wrong 挤掉，结果会是 0。
    c, _ = compute_accuracy_score([C("wrong", x=1), C("lookup", city="paris")], gt)
    ok = abs(c - 0.5) < 1e-9
    print("  [%s] 零奖励不占位 pred=[wrong,lookup] -> %.4f（预期 0.5；若是 0 说明被挤掉）"
          % ("PASS" if ok else "FAIL", c))
    if not ok:
        fails.append("性质2b")

    # ---- 性质 3：第二个 call 的梯度空间 ----
    one, _ = compute_accuracy_score([C("search", q="rain")], gt)
    two, _ = compute_accuracy_score([C("search", q="rain"), C("lookup", city="paris")], gt)
    ok = two - one > 0.4
    print("  [%s] 梯度空间   1 call %.4f -> 2 calls %.4f，差 %.4f"
          % ("PASS" if ok else "FAIL", one, two, two - one))
    if not ok:
        fails.append("性质3")

    print()
    if fails:
        print("未过 %d 项：%s" % (len(fails), fails))
        return 1
    print("三项性质全过。多调用题有 0.5 的梯度空间，且配对与 gold 顺序无关。")
    print("=> round4b「见过但学不会」不能归因于奖励函数看不见多调用。")
    return 0


if __name__ == "__main__":
    sys.exit(run())
