#!/usr/bin/env python3
"""两个 checkpoint 在同一批题上的配对检验。

为什么必须配对：同样的题，这一轮做对、上一轮做错，这种"翻转"才是提升的证据。
比聚合均值的差会把方差浪费掉——n=135 时聚合差 3pp 可能在噪声里，而配对之后
只有真正翻转的那几条算进检验，灵敏度高得多。

用法：
  python paired_test.py --a iter1_solver --b iter2_solver --field solver --metric acc
  python paired_test.py --a iter1_gen --b iter2_gen --field gen --metric four_tag
  python paired_test.py --a iter1_gen --b iter2_gen --field gen --metric pv_ok

判定口径（事前定，避免事后挑好看的）：
  p < 0.05 且方向为"b 不差于 a" -> 这一轮在这个指标上算上升
  p >= 0.05                    -> 持平，不算上升也不算退化
  方向为退且 p < 0.05           -> 退化了，要定位

**连续指标用哪个 p（2026-10-15 定，在看到 iter2 结果之前）**：
`acc` 是 accuracy_reward 的**连续值**（分档实测 0.31~0.63，不是 0/1），McNemar 没法
直接用它——下面的 `ok()` 会把它二值化成 `>0`，于是报出来的"成功率"其实是
`sol_acc_pos`（拿到任意部分分的比例），不是 `sol_acc`（奖励均值），而且幅度信息全丢。
所以对连续值指标，**以"连续配对检验"那一行的 p 为准**，McNemar 那行只当稳健性参照。
`fmt` / `name_hit` / `four_tag` / `pv_ok` 天生是 0/1，两行一致，以 McNemar 为准。

**n=135 的检出下限（也算在看见结果之前）**：McNemar 精确检验在 135 配对题上很弱——
翻盘 5 题且全好（5/0）p=0.0625 都不显著，6/1 是 0.1250，8/2 是 0.1094，12/4 是 0.0768。
只有 6/0（0.0312）或净好 ≥10 题且比值约 3:1 才够显著。所以"方向为正但 p≥0.05"
是这个样本量下的**常态**，那种情况报"方向为正、不显著"，不许叫上升。
"""
import argparse
import json
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EV = os.path.join(HERE, "evals")

# 置换检验的种子固定：p 值必须可复现，不然"显著/不显著"会随跑变。
PERM_SEED = 20261015
PERM_ITERS = 20000


def load(tag):
    p = os.path.join(EV, tag + ".json")
    if not os.path.exists(p):
        sys.exit("没有 %s——这个 tag 是用旧版 eval_iter.py 跑的，没有逐条结果" % p)
    return json.load(open(p))


def binom_two_sided(k, n):
    """精确二项检验双侧 p 值（p=0.5）。n=0 时返回 1。"""
    if n == 0:
        return 1.0
    # 用对数避免阶乘溢出
    def logc(a, b):
        return (math.lgamma(a + 1) - math.lgamma(b + 1) - math.lgamma(a - b + 1))
    logp = lambda i: logc(n, i) - n * math.log(2)
    obs = logp(k)
    tot = 0.0
    for i in range(n + 1):
        lp = logp(i)
        if lp <= obs + 1e-12:
            tot += math.exp(lp)
    return min(1.0, tot)


def paired_perm_p(diffs, seed=PERM_SEED, iters=PERM_ITERS):
    """配对符号翻转置换检验，双侧。H0：配对差均值为 0。

    为什么不用 McNemar： McNemar 只吃二值。而 `fmt` 实测取值是 {0.0, 0.3, 1.0}
    （114/11/10），`acc` 有 21 个不同取值——都是分档奖励。把它们按 >0 压成 0/1 之后，
    "格式从 0.3 修到 1.0"那 11 条变成"两者都对"，一个 discordant 都不产生，
    格式真变好了也检不出来。这里直接对配对差做符号翻转，检验的是差均值，
    幅度信息不丢，二值和连续指标都能用。

    种子写死：p 值必须可复现，否则"显著/不显著"会随跑变，那就没法做事前口径。
    返回 (p, 蒙特卡洛标准误)。+1 平滑避免报出 p=0。
    """
    n = len(diffs)
    if n == 0:
        return 1.0, 0.0
    obs = sum(diffs) / n
    if abs(obs) < 1e-15:
        return 1.0, 0.0
    rng = random.Random(seed)
    cnt = 0
    for _ in range(iters):
        s = 0.0
        for d in diffs:
            s += d if rng.random() < 0.5 else -d
        if abs(s / n) >= abs(obs) - 1e-15:
            cnt += 1
    p = (cnt + 1) / (iters + 1)
    se = math.sqrt(max(p * (1 - p), 1e-12) / iters)
    return p, se


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="前一格的 tag")
    ap.add_argument("--b", required=True, help="后一格的 tag")
    ap.add_argument("--field", choices=["gen", "solver"], required=True)
    ap.add_argument("--metric", required=True,
                    help="gen: four_tag / pv_ok；solver: fmt / acc / name_hit")
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    A, B = load(args.a), load(args.b)
    la, lb = A.get(args.field), B.get(args.field)
    if la is None or lb is None:
        sys.exit("%s 或 %s 没有 %s 的逐条结果" % (args.a, args.b, args.field))
    if len(la) != len(lb):
        sys.exit("两条记录的题数不一样（%d vs %d），没法配对" % (len(la), len(lb)))

    # 对齐：solver 按 question，gen 按下标（spec 列表由固定 seed 决定，顺序一致）
    if args.field == "solver":
        mb = {x["question"]: x for x in lb}
        pairs = [(x, mb[x["question"]]) for x in la if x["question"] in mb]
    else:
        pairs = list(zip(la, lb))
    if len(pairs) < len(la):
        print("[warn] 只配上 %d/%d 条" % (len(pairs), len(la)))

    def ok(x):
        v = x.get(args.metric)
        return bool(v) if not isinstance(v, float) else v > 0

    only_a = only_b = both = neither = 0
    for x, y in pairs:
        oa, ob = ok(x), ok(y)
        if oa and ob:
            both += 1
        elif oa and not ob:
            only_a += 1          # a 对 b 错 -> 退步
        elif (not oa) and ob:
            only_b += 1          # a 错 b 对 -> 进步
        else:
            neither += 1

    n = len(pairs)
    ma = (both + only_a) / n
    mb = (both + only_b) / n
    p = binom_two_sided(only_b, only_a + only_b)

    # 连续量：同一个指标的真实取值（不是二值化后的）。用来判定它是不是分档奖励，
    # 以及给置换检验供数据。
    def num(x):
        v = x.get(args.metric)
        if v is None or v is False:
            return 0.0
        if v is True:
            return 1.0
        return float(v)

    va = [num(x) for x, _ in pairs]
    vb = [num(y) for _, y in pairs]
    n_val = len(set(va) | set(vb))
    graded = n_val > 2          # 取值超过 {0,1} 就是分档奖励，McNemar 会丢幅度

    print("配对检验 %s -> %s  指标 %s/%s  n=%d" % (args.a, args.b, args.field, args.metric, n))
    if graded:
        print("  [注意] %s 有 %d 个不同取值，是分档奖励不是 0/1——McNemar 把它按 >0 压扁了，" % (args.metric, n_val))
        print("         所以下面的\" McNemar 判定\"只当稳健性参照，主检验是连续配对检验")
    print("  %s 均值 %.4f    %s 均值 %.4f    差 %+.4f" % (args.a, sum(va) / n, args.b, sum(vb) / n, sum(vb) / n - sum(va) / n))
    print("  两者都非零 %d   两者都零 %d" % (both, neither))
    print("  只有 %s 非零 %d（退步）   只有 %s 非零 %d（进步）" % (args.a, only_a, args.b, only_b))
    print("  McNemar 精确双侧 p = %.4f%s" % (p, "（二值化，仅参照）" if graded else ""))

    # 事前定好的口径：多值指标以连续配对检验为主。
    if graded:
        pp, se = paired_perm_p([b - a for a, b in zip(va, vb)])
        print("  连续配对检验：符号翻转置换 p = %.4f（SE %.4f，%d 次，种子 %d）"
              % (pp, se, PERM_ITERS, PERM_SEED))
        p_main, mean_a, mean_b = pp, sum(va) / n, sum(vb) / n
    else:
        p_main, mean_a, mean_b = p, ma, mb

    if mean_b < mean_a - 1e-12 and p_main < args.alpha:
        print("  判定：*** 退化 ***")
        sys.exit(1)
    if mean_b > mean_a + 1e-12 and p_main < args.alpha:
        print("  判定：上升（p<%.2f）" % args.alpha)
    else:
        # n=135 上 McNemar 的检出下限很高（5/0 全好都 p=0.0625）。方向为正但不显著
        # 是这个样本量下的常态，照实说，不包装成"趋势向好"。
        extra = "，方向为正但样本量不够（n=%d 的 McNemar 要净翻 ≥10 题才够显著）" % n if mean_b > mean_a + 1e-12 else ""
        print("  判定：与 0 分不开，算持平%s" % extra)


if __name__ == "__main__":
    main()
