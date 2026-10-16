#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""round4b 六项事前判据的机械核对。

阈值全部抄自 PLAN.md「推荐的验证实验：round 4b」那张表（2374-2385 行），
写死成字面量——价值全在于"不看着数字重新解释它"。

为什么不让手算：这类 join（教材的 answer / holdout 的 gold calls / 评测逐条 acc）
我手算过三遍、前两遍都错（分解式不成立 + 把不同批次当成同一批题）。
所以这里每一步都 assert 数量、assert 题目文本对齐，不靠顺序假设。

用法: .venv/bin/python check_round4b_criteria.py
退出码 = 未过判据条数。
"""
import json
import os
import sys

BASE = "qwen2.5-0.5b-instruct-tool-r0"
ITER3_DATA = os.path.join(BASE, "iter3_data.json")                    # iter3 教材（对照）
B_DATA = os.path.join(BASE, "iter3b_data.json")                       # round4b 教材
B_VERIFIED = os.path.join(BASE, "iter3b_data.json.verified.json")     # tau=0 后的池
HOLDOUT = "holdout_solver.json"
EVAL_B = "evals/iter3b_solver.json"
EVAL_3 = "evals/iter3_solver.json"

# ---- 事前阈值（PLAN.md:2374-2381，看到 round4b 数字之前钉的）----
EXP_KEPT = 1149          # tau=0 应该 1149 条全进
EXP_MULTI_FRAC = 0.075   # 池里多调用占比回到原始分布
EXP_MULTI_N = 5          # 教材多调用条数 >=5
EXP_KEYS3_N = 18         # 教材 3+ 键条数 >=18
ACC_FLOOR = 0.6835       # held-out sol_acc > 0.6627 + 0.0208
MULTI_NAMEHIT_FLOOR = 0.30   # held-out 多调用题点名率 > 0.143


def load(p):
    if not os.path.exists(p):
        print("  [缺] %s 不在" % p)
        return None
    return json.load(open(p))


def calls_of(item):
    """一条教材/holdout 的工具调用列表。

    两个字段都要试，这是踩过的坑：**verified 池文件存的是 `calls`，
    终选教材存的是 `answer`**。只用 `answer` 会把池文件全解析成空列表，
    于是"池里多调用占比"静默算出 0.0% 并报 FAIL——而真实值是 86/1149 = 7.5%。
    失败模式是"看起来合理地报了个 FAIL"，比崩溃更难发现。
    """
    a = item.get("calls")
    if a is None:
        a = item.get("answer")
    if isinstance(a, str):
        try:
            a = json.loads(a)
        except Exception:
            return []
    return a if isinstance(a, list) else []


def nkeys(item):
    return sum(len(c.get("arguments", {}) or {}) for c in calls_of(item))


def ncalls(item):
    return len(calls_of(item))


def main():
    fails = []
    print("=" * 74)
    print("round4b 事前判据核对（阈值来自 PLAN.md:2374-2385，不许赛后改）")
    print("=" * 74)

    # ---------- 1) verify 留存 ----------
    print("\n[1] verify 留存（tau=0 应当全进）")
    pool = load(B_VERIFIED)
    if pool is None:
        print("    round4b 还没跑，无从核对。")
        return 0
    kept = len(pool)
    print("    留存 %d 条 / 预期 %d" % (kept, EXP_KEPT))
    if kept < 1100:
        print("    !!! 远低于预期——tau=0 下不该还有门。看 round4b.verify.log 的 early-drop")
        fails.append("verify 留存")

    # ---------- 2) 池里多调用占比 ----------
    multi = [r for r in pool if ncalls(r) >= 2]
    frac = len(multi) / len(pool) if pool else 0.0
    print("\n[2] 池里多调用占比（回到原始生成分布）")
    print("    实测 %.1f%% (%d/%d) / 预期 %.1f%%" % (frac * 100, len(multi), len(pool), EXP_MULTI_FRAC * 100))
    if frac < 0.05:
        print("    !!! 多调用仍被压着，tau 不是唯一的门")
        fails.append("池多调用占比")

    # ---------- 3) 4) 教材构成 ----------
    cur = load(B_DATA)
    base = load(ITER3_DATA)
    if cur is not None:
        c_multi = [r for r in cur if ncalls(r) >= 2]
        c_keys3 = [r for r in cur if nkeys(r) >= 3]
        b_multi = [r for r in base if ncalls(r) >= 2] if base else []
        b_keys3 = [r for r in base if nkeys(r) >= 3] if base else []
        print("\n[3] 教材多调用条数（iter3 只有 %d 条）" % len(b_multi))
        print("    round4b %d 条 / 预期 >=%d" % (len(c_multi), EXP_MULTI_N))
        if len(c_multi) < EXP_MULTI_N:
            print("    !!! 教材里多调用还是太少——选材那一支仍把它滤掉了")
            fails.append("教材多调用条数")

        print("\n[4] 教材 3+ 键条数（iter3 是 %d 条）" % len(b_keys3))
        print("    round4b %d 条 / 预期 >=%d" % (len(c_keys3), EXP_KEYS3_N))
        if len(c_keys3) < EXP_KEYS3_N:
            print("    !!! 参数键数没上来")
            fails.append("教材 3+ 键条数")

    # ---------- 5) held-out sol_acc ----------
    print("\n[5] held-out sol_acc（对照 iter3_solver 的 0.6627）")
    ev = load(EVAL_B)
    if ev is not None:
        accs = [r["acc"] for r in ev["solver"] if "acc" in r]
        m = sum(accs) / len(accs) if accs else 0.0
        print("    实测 %.4f (n=%d) / 预期 >%.4f" % (m, len(accs), ACC_FLOOR))
        if m <= ACC_FLOOR:
            print("    !!! 没过线。按 PLAN.md:2385，先看是不是零梯度题拖累，"
                  "再决定要不要取中间 tau=0.1")
            fails.append("held-out sol_acc")
    else:
        print("    还没评。")

    # ---------- 6) held-out 多调用题点名率 ----------
    print("\n[6] held-out 多调用题点名率（iter3 是 1/7 = 0.143，n=7 极小）")
    if ev is not None:
        h = load(HOLDOUT)
        if h is None:
            print("    holdout 不在，无法切多调用子集")
        else:
            # 按题目文本对齐，不假设顺序；数量必须一致
            by_q = {r["question"]: r for r in ev["solver"]}
            miss = [q for q in (r["question"] for r in h) if q not in by_q]
            if miss:
                print("    !!! %d 道 holdout 题在评测里找不到——对齐失败，不往下算" % len(miss))
                fails.append("多调用点名率对齐")
            else:
                sub = [by_q[r["question"]] for r in h if ncalls(r) >= 2]
                if not sub:
                    print("    holdout 里没有多调用题？")
                else:
                    hits = sum(1 for r in sub if r.get("name_hit"))
                    rate = hits / len(sub)
                    print("    实测 %d/%d = %.3f / 预期 >%.2f" % (hits, len(sub), rate, MULTI_NAMEHIT_FLOOR))
                    if rate <= MULTI_NAMEHIT_FLOOR:
                        print("    !!! 多调用点名率没动")
                        fails.append("多调用点名率")

    # ---------- 分叉判定（PLAN.md:2383-2385 的两个分支）----------
    print("\n" + "=" * 74)
    print("分叉判定（PLAN.md:2383-2385）")
    print("=" * 74)
    if cur is not None and ev is not None and "by_q" in dir():
        try:
            sub = [by_q[r["question"]] for r in h if ncalls(r) >= 2]
            hits = sum(1 for r in sub if r.get("name_hit"))
            rate = hits / len(sub) if sub else 0.0
            multi_up = len(c_multi) >= EXP_MULTI_N
            accs = [r["acc"] for r in ev["solver"] if "acc" in r]
            m = sum(accs) / len(accs) if accs else 0.0
            if multi_up and rate <= MULTI_NAMEHIT_FLOOR:
                print("  教材多调用上去了(%d 条)但多调用点名率仍 %.3f" % (len(c_multi), rate))
                print("  => 「solver 见过但学不会」。问题在 step3 的训练信号")
                print("     （怎么给多调用题算奖励），不在选材。")
                print()
                print("  [事后实测的归因，2026-10-16 04:00，不是预注册判定]")
                print("  上面括号里'怎么给多调用题算奖励'这个猜测**已被证伪**：")
                print("  test_multicall_reward.py 实测多调用题有 0.5 的梯度空间")
                print("  （只吐1个call <=0.5，吐2个都对 =1.0），奖励看得见。")
                print("  真机制是**被95条单调用教材主动压制**：mc_training_signal.py")
                print("  读到 round4b 的400个训练槽里8个completion吐了2个call，")
                print("  8个全落在单调用题上、0个在多调用题上、7/8负优势")
                print("  （多吐一个触发 extra=1 -> penalty=0.8）。净梯度为负。")
                print("  两条预注册假设也都测掉了：")
                print("   - '零梯度题拖累训练' 机制不可能（scale_rewards 默认 group）")
                print("   - 'GRPO采不到第二个call' 证伪（多样本 P=6.55%/4.17%，非零）")
                print("  => 该做 SFT 冷启动教输出契约，不是继续调 step2 选材。")
                print("     详见 PLAN.md 第 7 节与总结论第 3 节。")
            elif m < 0.6627:
                print("  held-out sol_acc 从 0.6627 掉到 %.4f" % m)
                print("  => 按 PLAN.md:2385：放进来的零梯度题拖累了训练，取中间 tau=0.1 再测。")
            elif m > ACC_FLOOR and rate > MULTI_NAMEHIT_FLOOR:
                print("  acc %.4f 越线、多调用点名率 %.3f 也越线" % (m, rate))
                print("  => tau_verify 是增益递减的根因，这条证据成立。")
            else:
                print("  acc %.4f、多调用点名率 %.3f、教材多调用 %d 条——落在预设分支之外，"
                      "照实汇报，别硬套。" % (m, rate, len(c_multi)))
        except NameError:
            print("  （数据不全，跳过）")

    print("\n未过判据 %d 条：%s" % (len(fails), fails or "无"))
    return len(fails)


if __name__ == "__main__":
    sys.exit(main())
