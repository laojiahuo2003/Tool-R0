#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""跨轮比 step3 的"教材有多少条真的在产梯度"。

为什么看这个而不是只看 reward：reward 均值会被已经学会的题抬高，
而 `frac_reward_zero_std` 是 TRL 直接打的"本步有多少个 prompt 组内奖励零方差"——
零方差 = GRPO 优势恒 0 = 这一步对这些题白训。它衡量教材的**信息量**，
不衡量 solver 的绝对水平，所以可以跨轮直接比。

用法: .venv/bin/python step3_signal.py 2 3 4 4b
"""
import re
import sys


def stats(path):
    try:
        raw = open(path, "rb").read().decode("utf-8", "replace")
    except OSError:
        return None
    raw = raw.replace("\r", "\n")
    zs = [float(m) for m in re.findall(r"'frac_reward_zero_std': ([0-9.]+)", raw)]
    acc = [float(m) for m in
           re.findall(r"'rewards/accuracy_reward_func/mean': ([0-9.]+)", raw)]
    accsd = [float(m) for m in re
             .findall(r"'rewards/accuracy_reward_func/std': ([0-9.]+)", raw)]
    rew = [float(m) for m in re.findall(r"'reward': ([0-9.]+)", raw)]
    if not zs:
        return None
    n = len(zs)
    return {
        "steps": n,
        "zero_std_mean": sum(zs) / n,
        "zero_std_nonzero_steps": 100.0 * sum(1 for x in zs if x > 0) / n,
        "acc_mean": (sum(acc) / len(acc)) if acc else float("nan"),
        "acc_std_mean": (sum(accsd) / len(accsd)) if accsd else float("nan"),
        "reward_mean": (sum(rew) / len(rew)) if rew else float("nan"),
    }


def logpath(tag):
    """轮次标签 -> 日志路径。

    标签是**字符串不是整数**：round4b 的日志叫 round4b.step3.log，
    而 `int("4b")` 直接 ValueError。第一版写成 int()，
    于是"4b 这一轮的零方差"这个数根本取不出来——
    而它正是 round4b 判据里"先看是不是零梯度题拖累"要用的那个数。
    """
    return "round%s.step3.log" % tag


def main():
    tags = sys.argv[1:] or ["2", "3", "4"]
    print("=" * 78)
    print("step3 训练信号跨轮对比（零方差占比越低 = 教材信息量越高）")
    print("=" * 78)
    print("%-6s %-7s %-12s %-12s %-12s %-12s %-10s"
          % ("轮次", "步数", "零方差均值", "有零方差步%", "acc均值", "acc组内std", "reward"))
    rows = []
    for r in tags:
        s = stats(logpath(r))
        if s is None:
            print("%-6s %s" % (r, "(无日志或无该字段)"))
            continue
        rows.append((r, s))
        print("%-6s %-7d %-12.4f %-12.1f %-12.4f %-12.4f %-10.4f"
              % (r, s["steps"], s["zero_std_mean"], s["zero_std_nonzero_steps"],
                 s["acc_mean"], s["acc_std_mean"], s["reward_mean"]))
    if len(rows) >= 2:
        a, b = rows[-2], rows[-1]
        d = b[1]["zero_std_mean"] - a[1]["zero_std_mean"]
        print("\n  round%s -> round%s 零方差均值变化 %+.4f" % (a[0], b[0], d))
        if d < -0.02:
            print("  => 教材信息量上升（更少的组在空转）")
        elif d > 0.02:
            print("  => 教材信息量下降")
        else:
            print("  => 基本持平（|变化| < 0.02）")


if __name__ == "__main__":
    main()
