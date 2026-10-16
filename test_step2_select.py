#!/usr/bin/env python3
"""step2_select_curriculum.py 的单元测试（不需要 GPU，纯逻辑）。

跑法：.venv/bin/python test_step2_select.py

覆盖 2026-10-15 新加/改动的三块：
  1. assign_difficulty_from_agreement —— 阈值边界、缺值兜底
  2. split_quota_by_remaining          —— 全局目标分摊到小配额域（原来 hard 名额会被 round 成 0）
  3. drop_saturated                    —— 已有的饱和过滤，回归保护
"""

import sys

from step2_select_curriculum import (
    Item,
    assign_difficulty_from_agreement,
    difficulty_targets,
    drop_saturated,
    load_items,
    split_quota_by_remaining,
)

FAILS = []


def check(name, got, want):
    if got == want:
        print("  ok   %s" % name)
    else:
        print("  FAIL %s: got %r want %r" % (name, got, want))
        FAILS.append(name)


def mk(agr):
    it = Item(question="q", tools=[], calls=[], meta={})
    it.solver_agreement = agr
    return it


print("assign_difficulty_from_agreement")
# k_verify=10 时 agreement 只能取 0.1 的整数倍，边界按 <hard_lt / >=easy_ge 划
items = [mk(v / 10.0) for v in range(0, 11)]
assign_difficulty_from_agreement(items, 0.4, 0.8, "medium")
check("11 个离散值的分档",
      [it.difficulty for it in items],
      ["hard"] * 4 + ["medium"] * 4 + ["easy"] * 3)   # 0-.3 hard, .4-.7 medium, .8-1.0 easy
check("judge_raw 记下依据",
      [it.judge_raw for it in items[:3]],
      ["agreement=0.00", "agreement=0.10", "agreement=0.20"])
check("有命中率时不算 fallback",
      [it.judge_used_fallback for it in items], [False] * 11)

# 缺 agreement：不能猜成 0 或 1，按 default_diff 兜底并标记
missing = [mk(None)]
n_have, n_missing = assign_difficulty_from_agreement(missing, 0.4, 0.8, "hard")
check("缺值按 default_diff", missing[0].difficulty, "hard")
check("缺值记 fallback", missing[0].judge_used_fallback, True)
check("缺值 judge_raw", missing[0].judge_raw, "agreement=missing")
check("返回值 (有, 缺)", (n_have, n_missing), (0, 1))

# 混在一起时两个计数都要对
mixed = [mk(0.1), mk(None), mk(0.9), mk(None)]
n_have, n_missing = assign_difficulty_from_agreement(mixed, 0.4, 0.8, "medium")
check("混合输入计数", (n_have, n_missing), (2, 2))
check("混合输入分档", [it.difficulty for it in mixed],
      ["hard", "medium", "easy", "medium"])

# 阈值收紧/放宽都要生效。注意语义是 a < hard_lt 算 hard、a >= easy_ge 算 easy，
# 所以 0.3 在 hard_lt=0.3 时落在 medium（0.3 < 0.3 为假），这里用 0.29/0.70 真正压边界。
tight = [mk(0.29), mk(0.5), mk(0.70)]
assign_difficulty_from_agreement(tight, 0.3, 0.7, "medium")
check("自定义阈值", [it.difficulty for it in tight], ["hard", "medium", "easy"])
# 恰好等于阈值的那侧归属
edge = [mk(0.3), mk(0.7)]
assign_difficulty_from_agreement(edge, 0.3, 0.7, "medium")
check("等于 hard_lt 归 medium，等于 easy_ge 归 easy",
      [it.difficulty for it in edge], ["medium", "easy"])

print("\nsplit_quota_by_remaining")
# 核心回归：q=3 + mix 20/50/30 时，老代码 difficulty_targets(3, mix) 给 hard=0
check("老代码在 q=3 上 hard=0（这就是要修的 bug）",
      difficulty_targets(3, {"easy": 0.2, "medium": 0.5, "hard": 0.3})["hard"], 0)
tgt = difficulty_targets(100, {"easy": 0.2, "medium": 0.5, "hard": 0.3})
check("全局目标", tgt, {"easy": 20, "medium": 50, "hard": 30})

# 第一域（q=3）按剩余 20/50/30 分摊，必须拿到 hard 名额
first = split_quota_by_remaining(3, dict(tgt))
check("q=3 也能分到 hard", first["hard"] >= 1, True)
check("q=3 分摊总和为 3", sum(first.values()), 3)

# 把 32 个域（100 个名额）依次分摊完，总和必须精确等于全局目标
remaining = dict(tgt)
total_taken = {"easy": 0, "medium": 0, "hard": 0}
quotas = [4] * 4 + [3] * 28          # 实测：4 个域 q=4，28 个域 q=3
for q in quotas:
    t = split_quota_by_remaining(q, remaining)
    for d in t:
        total_taken[d] += t[d]
        remaining[d] -= t[d]
check("32 域分摊后总和", sum(total_taken.values()), 100)
check("各档不超全局目标",
      all(total_taken[d] <= tgt[d] for d in tgt), True)
check("hard 档拿满", total_taken["hard"], 30)

# 某档需求已达成后，名额要流向还有缺口的档，不能继续硬塞
r = {"easy": 0, "medium": 50, "hard": 0}
t = split_quota_by_remaining(3, r)
check("只补还有缺口的档", (t["easy"], t["hard"]), (0, 0))
check("名额都给了 medium", t["medium"], 3)

# 全都没缺口（池子见底之类的）时均摊，别把名额浪费掉
t = split_quota_by_remaining(5, {"easy": 0, "medium": 0, "hard": 0})
check("无缺口时均摊", t, {"easy": 2, "medium": 2, "hard": 1})
check("q=0 返回全 0", split_quota_by_remaining(0, tgt),
      {"easy": 0, "medium": 0, "hard": 0})

# 极端配比：只要 hard
t = split_quota_by_remaining(3, {"easy": 0, "medium": 0, "hard": 30})
check("单档需求", t, {"easy": 0, "medium": 0, "hard": 3})

print("\ndrop_saturated（回归）")
its = [mk(1.0), mk(0.9), mk(None), mk(0.0)]
kept, dropped = drop_saturated(its, 1.0)
check("只剔 1.0", (len(kept), dropped), (3, 1))
check("缺值保留", any(it.solver_agreement is None for it in kept), True)
kept, dropped = drop_saturated(its, 0.8)
check("阈值 0.8", (len(kept), dropped), (2, 2))
kept, dropped = drop_saturated(its, 1.01)
check("阈值 >1 关闭", (len(kept), dropped), (4, 0))
check("空输入", drop_saturated([], 1.0), ([], 0))

print("\n%s  %d 个失败" % ("全部通过" if not FAILS else "有失败", len(FAILS)))
sys.exit(1 if FAILS else 0)
