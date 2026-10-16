#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多调用题的多样本探针：solver 到底采不采得到第二个 call？

为什么需要它：单次评测的"7 道里 0 道 acc>0.5"不算铁证——若真实概率 p=0.05，
7 次全没中的概率是 0.70，跟 p=0 无法区分。而这个问题是 round4b 结论的核心：
「solver 见过但学不会」有两种完全不同的解释，修法相反。

  (a) 探索不到：4 个采样从来没出现过 2 个 call => GRPO 无从强化
      （奖励里 0.5 -> 1.0 的梯度空间是有的，但采样碰不到）
  (b) 学不动：采样里有 2 个 call，但奖励信号不足以强化它

test_multicall_reward.py 已经排掉第三种解释"奖励看不见多调用"（有 0.5 梯度空间）。
这里分辨 (a) 和 (b)：数 N 个采样里出现 >=2 个 call 的比例。

设种方式：每个样本 seed+i，与 eval_iter 的 SOLVER_SEED 同源但逐样本偏移，
既可复现又能拿到 N 个独立样本（同一个 seed 重复采样会拿到同一个结果）。

用法: .venv/bin/python probe_multicall.py iter3_solver iter3b_solver [N]
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

import eval_iter as EI  # noqa: E402  复用同一套 load/prompt/解码条件
from step2_genverify import (  # noqa: E402
    SYSTEM_PROMPT_SOLVER, extract_tool_call_answer_block, clean_json_blob)
from rewards_solver import accuracy_reward_func  # noqa: E402

HOLDOUT = EI.HOLDOUT
N_DEFAULT = 24
BASE = "qwen2.5-0.5b-instruct-tool-r0"


def coerce(x):
    from step2_genverify import coerce_json
    return coerce_json(x)


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


def probe(path, n=N_DEFAULT):
    data = json.load(open(HOLDOUT))
    mc = [d for d in data if len(calls_of(d)) >= 2]
    tok, model = EI.load(path)
    print("  checkpoint: %s" % path)
    print("  多调用题 %d 道，每题 %d 个采样（temp=%.1f，与评测同条件）"
          % (len(mc), n, EI.SOLVER_TEMP))

    out = []
    for d in mc:
        msgs = [{"role": "system", "content": SYSTEM_PROMPT_SOLVER},
                {"role": "user", "content": d["question"]}]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = tok([prompt], return_tensors="pt", padding=True,
                  truncation=True, max_length=4096).to("cuda")

        ncalls_list, rewards = [], []
        for i in range(n):
            torch.manual_seed(EI.SOLVER_SEED + i)
            with torch.no_grad():
                o = model.generate(
                    **enc, max_new_tokens=EI.SOLVER_MAX_NEW,
                    do_sample=True, temperature=EI.SOLVER_TEMP, top_p=0.95,
                    pad_token_id=tok.pad_token_id)
            txt = tok.decode(o[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
            blk = extract_tool_call_answer_block(txt)
            pred = coerce(clean_json_blob(blk)) if blk is not None else None
            k = len(pred) if isinstance(pred, list) else (1 if isinstance(pred, dict) else 0)
            ncalls_list.append(k)
            import contextlib, io
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                r = accuracy_reward_func(
                    prompts=[msgs], completions=[[{"content": txt}]], answer=[d["answer"]])
            rewards.append(r[0])
        ge2 = sum(1 for k in ncalls_list if k >= 2)
        out.append({
            "question": d["question"][:52],
            "gold": len(calls_of(d)),
            "n": n, "ge2": ge2,
            "p_ge2": ge2 / n,
            "hist": {k: ncalls_list.count(k) for k in sorted(set(ncalls_list))},
            "reward_max": max(rewards),
            "reward_mean": sum(rewards) / len(rewards),
            "n_reward_gt_half": sum(1 for r in rewards if r > 0.5),
        })
    del model
    EI.free()
    return out


def resolve(a):
    """checkpoint 名 -> 路径。

    坑：不能按"参数里有没有 /"决定要不要拼 BASE——`iter3_solver/checkpoint-50`
    带斜杠但仍是相对 BASE 的。正确做法是先按原样找，找不到再拼 BASE 试。
    """
    for p in (a, os.path.join(BASE, a)):
        if os.path.exists(p):
            return p
    return None


def main():
    n = N_DEFAULT
    rest = []
    for a in sys.argv[1:]:
        if a.isdigit():
            n = int(a)
        else:
            rest.append(a)
    if not rest:
        print("用法: .venv/bin/python probe_multicall.py <ckpt> [<ckpt2>] [N]")
        return 2
    allres = {}
    for a in rest:
        p = resolve(a)
        if p is None:
            print("缺 %s（试过原样和 %s/ 两种）" % (a, BASE)); continue
        print("=" * 78)
        allres[a] = probe(p, n)
        print()
        for r in allres[a]:
            print("  gold=%d  %s..." % (r["gold"], r["question"]))
            print("        >=2 call: %2d/%d = %.3f   输出个数分布 %s"
                  % (r["ge2"], r["n"], r["p_ge2"], r["hist"]))
            print("        reward 均值 %.3f  最大 %.3f  >0.5 的样本 %d 个"
                  % (r["reward_mean"], r["reward_max"], r["n_reward_gt_half"]))
        tot = sum(r["ge2"] for r in allres[a])
        den = sum(r["n"] for r in allres[a])
        print("  ---- %s 合计：%d/%d = %.4f 的采样吐了 >=2 个 call"
              % (a, tot, den, tot / den if den else 0))
        print()

    if len(allres) >= 2:
        tags = list(allres)
        print("=" * 78)
        print("跨 checkpoint 对比：P(吐 >=2 个 call)")
        print("=" * 78)
        for t in tags:
            rs = allres[t]
            tot = sum(r["ge2"] for r in rs)
            den = sum(r["n"] for r in rs)
            print("  %-18s %d/%d = %.4f" % (t, tot, den, tot / den if den else 0))
        print()
        print("  判读：")
        print("    全都 ~0        => 探索不到，GRPO 无从强化（解释 a）")
        print("    明显 >0 但 acc 不涨 => 学不动，奖励/优化有问题（解释 b）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
