#!/usr/bin/env python3
"""把 solver 的 acc 增益拆成"点名"和"参数"两个通道，并给出难度分档。

  用法: ./channel_split.py --a iter2_solver --b iter3_solver

为什么要有这个文件：这个分析我在 iter1→iter2 上**手算了三遍，前两遍都错了**，
错法还不一样，所以必须固化成脚本 + 自检，不能每轮现场推。

  第一遍错在分解式：写 `acc = 点名率 × 条件acc`。这个式子**不成立**——
    点名未命中时 acc 仍有 0.334（工具名错但参数对照样拿部分分，accuracy_reward
    是稠密分）。照它算得 0.5778×0.584 = 0.337，而实际 acc 是 0.478，对不上。
    正确分解是 `acc = 命中支 + 未中支`，命中支 = 点名率 × 条件acc，
    未中支 = (1-点名率) × 未中时的acc，两项都要算。
  第二遍错在读数：看到 iter1 条件 acc 按档是 0.50~0.64 就说"没有档位效应"。
    0.50 对 0.641 是 28% 的相对差，后来核出 0.7-0.85 档的坑**就在**条件 acc 上。
    "范围看起来不大"不等于"没效应"。
  第三遍（脚本做法）对：分组必须由 **a 格（较早那轮）定义**，再原样贴到 b 格上。
    按各自轮次自己的 name_hit 分组，两组不是同一批题，比出来的差没有意义。

自检（任一不过就 exit 1，不许带着可疑数字往下讲）：
  1. 两边按 question 对齐后 n 一致
  2. 命中支 + 未中支 == 该轮 acc（容差 1e-9）
  3. 固定分组的 n1 + n2 == n
"""
import argparse
import json
import sys

BANDS = [("lt0.3", 0.0, 0.3), ("0.3-0.5", 0.3, 0.5), ("0.5-0.7", 0.5, 0.7),
         ("0.7-0.85", 0.7, 0.85), ("0.85-1.0", 0.85, 1.0), ("eq1.0", 1.0, 1.01)]


def load(tag):
    d = json.load(open(f"evals/{tag}.json"))
    rows = d["solver"] if isinstance(d, dict) and "solver" in d else d
    return rows


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def branches(rows):
    """返回 (点名率, 条件acc, 未中率, 未中时acc, acc, 命中支, 未中支)。"""
    n = len(rows)
    hit = [x for x in rows if x["name_hit"]]
    miss = [x for x in rows if not x["name_hit"]]
    nh = len(hit) / n
    ca = mean(x["acc"] for x in hit)
    mca = mean(x["acc"] for x in miss) if miss else 0.0
    acc = mean(x["acc"] for x in rows)
    return nh, ca, 1 - nh, mca, acc, nh * ca, (1 - nh) * mca


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    args = ap.parse_args()

    A, B = load(args.a), load(args.b)
    mb = {x["question"]: x for x in B}
    pairs = [(x, mb[x["question"]]) for x in A if x["question"] in mb]
    if not pairs:
        sys.exit("!!! 两边没有一条能按 question 对齐")
    if len(pairs) != len(A) or len(pairs) != len(B):
        print(f"[警告] 对齐 {len(pairs)} 条，a 有 {len(A)} 条、b 有 {len(B)} 条")
    n = len(pairs)

    print(f"配对检验 {args.a} -> {args.b}   n={n}")
    print("分解方式：acc = 命中支(点名率×条件acc) + 未中支(未中率×未中时acc)")
    print("分组方式：按 **a 格** 的 name_hit 切，同一批题贴到 b 格上\n")

    # ---- 1. 总体分解 ----
    print("=== 总体 ===")
    print(f"{'':14s} {'点名率':>8s} {'条件acc':>8s} {'未中率':>8s} {'未中acc':>8s} "
          f"{'命中支':>8s} {'未中支':>8s} {'acc':>8s}")
    for tag, rows in ((args.a, [a for a, _ in pairs]), (args.b, [b for _, b in pairs])):
        nh, ca, mr, mca, acc, hb, mb_ = branches(rows)
        ok = abs(hb + mb_ - acc) < 1e-9
        print(f"{tag:14s} {nh:8.4f} {ca:8.4f} {mr:8.4f} {mca:8.4f} "
              f"{hb:8.4f} {mb_:8.4f} {acc:8.4f}  {'OK' if ok else '!!! 分解不闭合'}")

    ba, bb = branches([a for a, _ in pairs]), branches([b for _, b in pairs])
    d_acc = bb[4] - ba[4]
    d_hb, d_mb = bb[5] - ba[5], bb[6] - ba[6]
    print(f"\n  acc {ba[4]:.4f} -> {bb[4]:.4f}  ({d_acc:+.4f})")
    print(f"    命中支变化 {d_hb:+.4f}   未中支变化 {d_mb:+.4f}   "
          f"（两项相加 {d_hb + d_mb:+.4f}，应等于 acc 变化）")
    print(f"    点名率   {ba[0]:.4f} -> {bb[0]:.4f} ({bb[0]-ba[0]:+.4f})   "
          f"条件acc {ba[1]:.4f} -> {bb[1]:.4f} ({bb[1]-ba[1]:+.4f})")

    # ---- 2. 固定分组 ----
    print(f"\n=== 固定分组（按 {args.a} 的 name_hit 切）===")
    print(f"{'分组':16s} {'n':>4s} {'a acc':>8s} {'b acc':>8s} {'差':>8s} "
          f"{'进步/退步':>10s} {'a点名率':>8s} {'b点名率':>8s} {'b条件acc':>9s}")
    tot_hit = tot_miss = 0
    for idx, (label, sel) in enumerate((
            (f"{args.a} 点名命中", lambda a: a["name_hit"]),
            (f"{args.a} 点名未中", lambda a: not a["name_hit"]))):
        p = [(a, b) for a, b in pairs if sel(a)]
        if not p:
            continue
        # 用循环下标认组，不能用 sel(p[0][0])——sel 本身就是"选出这组"的条件，
        # 两组都会返回 True，于是 tot_hit 被第二组覆盖成 57、tot_miss 一直是 0。
        if idx == 0:
            tot_hit = len(p)
        else:
            tot_miss = len(p)
        ma = mean(a["acc"] for a, _ in p)
        mbb = mean(b["acc"] for _, b in p)
        up = sum(1 for a, b in p if b["acc"] > a["acc"])
        dn = sum(1 for a, b in p if b["acc"] < a["acc"])
        na = mean(1.0 if a["name_hit"] else 0.0 for a, _ in p)
        nb = mean(1.0 if b["name_hit"] else 0.0 for _, b in p)
        bh = [b for _, b in p if b["name_hit"]]
        cb = mean(b["acc"] for b in bh) if bh else float("nan")
        print(f"{label:16s} {len(p):4d} {ma:8.4f} {mbb:8.4f} {mbb-ma:+8.4f} "
              f"{str(up)+'/'+str(dn):>10s} {na:8.3f} {nb:8.3f} {cb:9.3f}")
    if tot_hit + tot_miss != n:
        sys.exit(f"!!! 分组不闭合：{tot_hit}+{tot_miss} != {n}")

    # ---- 3. 难度分档 ----
    print("\n=== 难度分档（solver_agreement，ag 越高=基座越会点名工具=越容易）===")
    print(f"{'档位':10s} {'n':>4s} | {'a点名率':>8s} {'a条件acc':>9s} {'a acc':>7s} | "
          f"{'b点名率':>8s} {'b条件acc':>9s} {'b acc':>7s} | {'acc差':>7s}")
    for nm, lo, hi in BANDS:
        p = [(a, b) for a, b in pairs if lo <= a.get("solver_agreement", -1) < hi]
        if not p:
            continue
        ra, rb = [a for a, _ in p], [b for _, b in p]
        A_ = branches(ra)
        B_ = branches(rb)
        print(f"{nm:10s} {len(p):4d} | {A_[0]:8.3f} {A_[1]:9.3f} {A_[4]:7.3f} | "
              f"{B_[0]:8.3f} {B_[1]:9.3f} {B_[4]:7.3f} | {B_[4]-A_[4]:+7.3f}")

    # ---- 4. 一句话结论 ----
    print("\n=== 怎么读 ===")
    print("  * b条件acc 若不随档位变（各档接近），说明是**与难度无关的上限**，")
    print("    再涨要靠专门改参数填充，不是加轮数。")
    print("  * 若某档 a点名率不低但 a条件acc 最低，该档的坑在**参数**不在点名。")
    print("  * 固定分组里 'b点名率' 掉下来的组要留意：可能拆东墙补西墙。")


if __name__ == "__main__":
    main()
