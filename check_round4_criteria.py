#!/usr/bin/env python3
"""round4 的事前判据，机械核对。

为什么单独写一个脚本而不是人眼看 report_funnel.py 的输出：
那四个阈值是**在看到 round4 的数字之前**写进 PLAN.md 的（「所以 round4 的事前判据
（现在钉住，不许赛后改）」，2026-10-16 00:48）。它的全部价值在于"不看着数字重新解释它"。
所以这里把阈值写成字面量，跑完只打 PASS/FAIL 和"这个结果意味着什么"——
不重新推理、不给"虽然超了但是……"留出口子。想改阈值，改 PLAN.md 并写清为什么，
那是一次可追溯的修正；在这里顺手放宽，是一次看不见的作弊。

用法：.venv/bin/python check_round4_criteria.py [ITER]     默认 4
退出码 = 未过判据条数，方便串在 post_round.sh 里。
"""
import sys

import report_funnel as RF

# ---- 事前阈值。来源：PLAN.md「所以 round4 的事前判据（现在钉住，不许赛后改）」----
# 改这里等于改事前判据，请在 PLAN.md 里留一条记录再改。
CRITERIA = [
    ("剔饱和数", "n_saturated", 656, ("lt", 360),
     "> 360 说明修复没生效（口径没切过去，或切了但没起作用）"),
    ("教材池", "hard_pool_post_total", 296, ("gt", 590),
     "< 592 说明口径没切过去（硬下界是 592）"),
    ("hard 池", "pool_post.hard", 30, ("between", 30, 60),
     "若 > 91 说明我对'难度是自指的'这个判断错了——那是好消息，但要认"),
    ("medium 池", "pool_post.medium", 93, ("gt", 150),
     "恢复项主要该落在这里；不涨说明回来的样本没进 medium 档"),
]

# iter3 的产出端规模，用来判断"教材池变小"是不是被 generator 减产混淆。
ITER3_POOL_PRE = 952


def get(d, path):
    for k in path.split("."):
        if d is None:
            return None
        d = d.get(k)
    return d


def judge(val, rule):
    op = rule[0]
    if val is None:
        return None, "无数值（日志或中间文件缺失）"
    if op == "lt":
        return val < rule[1], "要求 < %d" % rule[1]
    if op == "gt":
        return val > rule[1], "要求 > %d" % rule[1]
    if op == "between":
        return rule[1] <= val <= rule[2], "要求在 [%d, %d]" % (rule[1], rule[2])
    raise AssertionError("未知规则 %r" % (rule,))


def main():
    it = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    res = RF.report(it)

    # n_saturated 是日志里的；日志缺失时从"剔前-剔后"反推，两处必须一致。
    if res.get("n_saturated") is None and res.get("n_pool_pre") is not None:
        res["n_saturated"] = res["n_pool_pre"] - res["n_pool_post"]
        print("  （日志里没找到 [saturation] 行，用剔前 %d - 剔后 %d 反推 = %d）"
              % (res["n_pool_pre"], res["n_pool_post"], res["n_saturated"]))
    res["hard_pool_post_total"] = res.get("n_pool_post")

    print("=" * 78)
    print("round %d 事前判据核对（阈值来自 PLAN.md，看到数字之前钉的）" % it)
    print("=" * 78)

    n_fail = 0
    for name, path, base, rule, meaning in CRITERIA:
        val = get(res, path)
        ok, req = judge(val, rule)
        if ok is None:
            verdict = "?? 无法判定"
        elif ok:
            verdict = "PASS"
        else:
            verdict = "FAIL"
            n_fail += 1
        got = "缺失" if val is None else str(val)
        print("  %-10s iter3=%-5s round%-2d=%-7s %-14s %s"
              % (name, base, it, got, verdict, req))
        print("             %s" % meaning)

    # ---- 一个不在事前判据里、但会污染判据的混淆，必须单独说 ----
    pre = res.get("n_pool_pre")
    if pre is not None and pre < ITER3_POOL_PRE * 0.8:
        print()
        print("  !!! 混淆提醒：这一轮 verify 只活了 %d 条，iter3 是 %d。"
              % (pre, ITER3_POOL_PRE))
        print("      产出端缩了 %d 条，'教材池'和'hard 池'变小可能只是题变少，"
              % (ITER3_POOL_PRE - pre))
        print("      不是口径或自指的问题。上面的 FAIL 要按这个折扣读。")

    print()
    print("未过判据 %d 条" % n_fail)
    return n_fail


if __name__ == "__main__":
    sys.exit(main())
