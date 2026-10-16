#!/usr/bin/env python3
"""从 train.log 里把每一步的指标抠出来，回答"这一轮到底学没学到东西"。

为什么需要：iter2 那轮 9 小时只跑 17 步，最后是拿"reward 对步数回归斜率 t=+1.14"
下的结论，而那些数是从日志里一条条抠的。每轮都要做同样的对比，所以固化成脚本。

一个坑先说：日志里嵌了工具菜单的 ASCII 表格，表格的列名就叫 Exception/Error:，
所以崩没崩不能靠 grep 'Error'，要加 `grep -v '│'`。这个脚本按 `{'loss'` 起头的
dict 解析，天然不受表格影响。

用法：
  python parse_trainlog.py round2.log
  python parse_trainlog.py qwen2.5-0.5b-instruct-tool-r0/iter1_generator_rl_lr1e4/train.log
"""
import argparse
import ast
import json
import re
import sys

# step1（generator）和 step3（solver）想看的不是同一组键
GEN_KEYS = ["reward", "reward_std", "kl", "grad_norm", "step_time",
            "rewards/reward_validity_accuracy/mean",
            "rewards/reward_format_accuracy/mean",
            "rewards/reward_curriculum/mean",
            "completions/mean_length", "completions/clipped_ratio",
            "frac_reward_zero_std"]
SOLVER_KEYS = ["reward", "reward_std", "kl", "grad_norm", "step_time",
               "rewards/format_reward_func/mean",
               "rewards/accuracy_reward_func/mean",
               "completions/mean_length", "frac_reward_zero_std"]


def parse(path):
    txt = open(path, errors="replace").read()
    recs = []
    for m in re.finditer(r"\{'loss'.*?\}(?=\s|$)", txt, re.S):
        try:
            d = ast.literal_eval(m.group(0).replace("\n", " "))
        except Exception:
            continue
        if isinstance(d, dict) and "grad_norm" in d:
            recs.append(d)
    return recs


def slope_t(vals):
    """加权线性回归斜率 + t 值。步时和 reward 都是自相关的，t 值只用来判断
    "和 0 分不分得开"，不当作效应量。"""
    n = len(vals)
    if n < 4:
        return None, None
    xs = list(range(n))
    mx = sum(xs) / n
    my = sum(vals) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return None, None
    b = sum((x - mx) * (y - my) for x, y in zip(xs, vals)) / sxx
    resid = [y - (my + b * (x - mx)) for x, y in zip(xs, vals)]
    sse = sum(r * r for r in resid)
    se2 = sse / (n - 2) / sxx if n > 2 else 0.0
    if se2 <= 0:
        return b, None
    se = se2 ** 0.5
    return b, (b / se if se else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--kind", choices=["gen", "solver"], default="gen")
    ap.add_argument("--head", type=int, default=8, help="前多少步")
    ap.add_argument("--tail", type=int, default=8, help="后多少步")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    recs = parse(args.log)
    if not recs:
        print("没解析出任何 step 记录（文件里没有 {'loss' 开头的 dict？）", file=sys.stderr)
        sys.exit(1)
    keys = GEN_KEYS if args.kind == "gen" else SOLVER_KEYS
    keys = [k for k in keys if k in recs[0]]
    print("%s: %d 步（%s）" % (args.log, len(recs), args.kind))
    print("%-46s %10s %10s %10s %8s" % ("指标", "前%d均值" % args.head,
                                        "后%d均值" % args.tail, "斜率/步", "t"))
    out = {"log": args.log, "kind": args.kind, "n_steps": len(recs), "metrics": {}}
    for k in keys:
        vals = [r.get(k) for r in recs if isinstance(r.get(k), (int, float))]
        if len(vals) < len(recs):
            continue
        h = vals[:args.head]
        t = vals[-args.tail:]
        b, tv = slope_t(vals)
        print("%-46s %10.4f %10.4f %10.5f %8s" % (
            k, sum(h) / len(h), sum(t) / len(t), b if b is not None else float("nan"),
            ("%.2f" % tv) if tv is not None else "-"))
        out["metrics"][k] = {
            "head_mean": sum(h) / len(h), "tail_mean": sum(t) / len(t),
            "slope": b, "t": tv,
            "first": vals[0], "last": vals[-1],
        }
    st = [r.get("step_time") for r in recs if isinstance(r.get("step_time"), (int, float))]
    if st:
        print("\nstep_time: 首 %.0fs 末 %.0fs 中位 %.0fs 总 %.1f 小时"
              % (st[0], st[-1], sorted(st)[len(st) // 2], sum(st) / 3600))
        out["step_time"] = {"first": st[0], "last": st[-1],
                            "median": sorted(st)[len(st) // 2],
                            "total_hours": sum(st) / 3600}
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)
        print("-> %s" % args.json)


if __name__ == "__main__":
    main()
