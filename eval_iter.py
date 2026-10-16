#!/usr/bin/env python3
"""每一轮共用的评测台：同一批题、同一个 seed，测 generator 和 solver。

为什么要有这个文件：step3 每轮训的是当轮 generator 新出的题，题目换了就不能直接比
reward。要多轮对照，必须有一份从第一轮之后再也不变的题，和一个从不改变的 spec 集。
这两个集合在 build_holdout.py 里定死，这里只负责按固定流程跑分。

输出追加到 iter_metrics.jsonl，一行一个 checkpoint，字段：
  gen_four_tag / gen_pv_ok        generator 的四标签齐全率 / parse_and_validate 通过率
  gen_tags_<t>                    单个标签的发出率
  sol_fmt / sol_acc / sol_acc_pos  solver 的 format_reward / accuracy_reward 均值 /
                                  accuracy_reward>0 的比例（这三项就是 step3 的训练奖励）
  sol_name_hit                    单次采样里工具名与 gold 精确匹配的比例，
                                  与 step2_genverify 的 solver_agreement 同一定义

跑法：
  python eval_iter.py --tag iter1_gen --gen_ckpt <path>
  python eval_iter.py --tag iter1_solver --solver_ckpt <path>
  python eval_iter.py --tag base --gen_ckpt Qwen/Qwen2.5-0.5B-Instruct --solver_ckpt Qwen/Qwen2.5-0.5B-Instruct
"""
import argparse
import contextlib
import io
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# ---- 固定评测条件。改这里等于让历史分数全部作废，所以别改。----
GEN_SEED = 20261015
GEN_N = 192            # 192 条 spec：四标签率在 90% 附近时，±3.5% 的 1σ
GEN_TEMP = 0.8         # 与 smoke_test_generator / step2_gen 一致
GEN_MAX_NEW = 2048     # SFT 集 p99=2389，但 RL 后 mean 已降到 ~1350 token，2048 够
SOLVER_TEMP = 0.7      # 与 step2_genverify 默认 temp_verify 一致
SOLVER_MAX_NEW = 1024  # 与 step2_genverify 的 max_tokens_verify 一致
# solver 评测也要固定种子。temp 0.7 是采样解码，不设种时 CUDA RNG 按进程启动时的
# 熵播种，同一个 checkpoint 每跑一遍分数都不一样——那样"多轮对照"就少了一半意义：
# iter1 和 iter2 的差里混进了两套独立采样的差，而不是同一个随机条件下的差。
# generator 那边一直设了种（eval_generator 开头两行），solver 这边漏了，补上。
SOLVER_SEED = 20261016
HOLDOUT = os.path.join(HERE, "holdout_solver.json")
METRICS = os.path.join(HERE, "iter_metrics.jsonl")

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from step2_gen import SYSTEM_PROMPT_GENERATOR_TEMPLATE, USER_PROMPT_GENERATOR, parse_and_validate  # noqa: E402
from step1_generator import sample_spec  # noqa: E402
from rewards_generator import _parse_completion  # noqa: E402
from step2_genverify import (  # noqa: E402
    SYSTEM_PROMPT_SOLVER,
    extract_tool_call_answer_block,
    clean_json_blob,
    coerce_json,
    normalize_call_names_only,
)

TAGS = ("think", "question", "available_tools", "tool_call_answer")


def load(path):
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    # 批量生成必须左 padding：右 padding 会把 pad token 塞在 prompt 之后，模型真的会把
    # 它们当上下文读，生成的标签数会凭空变化（smoke_test_generator.py 里实测踩过）。
    tok.padding_side = "left"
    return tok, model


def free():
    torch.cuda.empty_cache()


def eval_generator(path, n=GEN_N, seed=GEN_SEED):
    random.seed(seed)
    torch.manual_seed(seed)
    specs = [sample_spec() for _ in range(n)]

    tok, model = load(path)
    prompts = []
    for s in specs:
        sp = SYSTEM_PROMPT_GENERATOR_TEMPLATE.format(**s)
        msgs = [{"role": "system", "content": sp},
                {"role": "user", "content": USER_PROMPT_GENERATOR}]
        prompts.append(tok.apply_chat_template(msgs, tokenize=False,
                                               add_generation_prompt=True))

    four = pv = 0
    per_tag = {t: 0 for t in TAGS}
    reasons = {}
    lens = []
    per_item = []
    bs = 8
    for i in range(0, len(prompts), bs):
        chunk = prompts[i:i + bs]
        enc = tok(chunk, return_tensors="pt", padding=True,
                  truncation=True, max_length=2048).to("cuda")
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=GEN_MAX_NEW,
                do_sample=True, temperature=GEN_TEMP, top_p=0.95,
                pad_token_id=tok.pad_token_id)
        for j, o in enumerate(out):
            text = tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True)
            p = _parse_completion(text)
            present = [t for t in TAGS if p["present"].get(t)]
            for t in present:
                per_tag[t] += 1
            ok4 = len(present) == 4
            if ok4:
                four += 1
            _, why = parse_and_validate(text)
            if why == "ok":
                pv += 1
            else:
                reasons[why] = reasons.get(why, 0) + 1
            lens.append(len(text))
            per_item.append({"spec": specs[i + j], "four_tag": ok4,
                             "pv_ok": why == "ok", "pv_why": why,
                             "n_chars": len(text)})
        print("  [gen %d/%d] four_tag=%d pv_ok=%d"
              % (min(i + bs, len(prompts)), len(prompts), four, pv), flush=True)

    del model
    free()
    return {
        "gen_n": len(prompts),
        "gen_four_tag": four / len(prompts),
        "gen_pv_ok": pv / len(prompts),
        "gen_tags": {t: per_tag[t] / len(prompts) for t in TAGS},
        "gen_pv_reasons": dict(sorted(reasons.items(), key=lambda x: -x[1])),
        "gen_mean_chars": sum(lens) / len(lens),
        "_per_item": per_item,
    }


def eval_solver(path, seed=SOLVER_SEED):
    random.seed(seed)
    torch.manual_seed(seed)
    data = json.load(open(HOLDOUT))
    tok, model = load(path)

    prompt_msgs, prompts = [], []
    for d in data:
        msgs = [{"role": "system", "content": SYSTEM_PROMPT_SOLVER},
                {"role": "user", "content": d["question"]}]
        prompt_msgs.append(msgs)
        prompts.append(tok.apply_chat_template(msgs, tokenize=False,
                                               add_generation_prompt=True))
    answers = [d["answer"] for d in data]

    completions = []
    bs = 16
    for i in range(0, len(prompts), bs):
        chunk = prompts[i:i + bs]
        enc = tok(chunk, return_tensors="pt", padding=True,
                  truncation=True, max_length=4096).to("cuda")
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=SOLVER_MAX_NEW,
                do_sample=True, temperature=SOLVER_TEMP, top_p=0.95,
                pad_token_id=tok.pad_token_id)
        for j, o in enumerate(out):
            completions.append([{"content": tok.decode(
                o[enc["input_ids"].shape[1]:], skip_special_tokens=True)}])
        print("  [solver %d/%d]" % (min(i + bs, len(prompts)), len(prompts)), flush=True)

    del model
    free()

    # 奖励函数会打印一大堆调试块，评测时不需要，吞掉。
    # prompts 必须原样传：两个奖励函数内部都是 zip(prompts, completions, strict=True)，
    # 传 None 会直接抛异常。
    from rewards_solver import format_reward_func, accuracy_reward_func
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fmts = format_reward_func(prompts=prompt_msgs, completions=completions)
        accs = accuracy_reward_func(prompts=prompt_msgs, completions=completions,
                                    answer=answers)

    name_hit = 0
    per_item = []
    for idx, (c, gold) in enumerate(zip(completions, answers)):
        txt = extract_tool_call_answer_block(c[0]["content"])
        hit = False
        if txt is not None:
            pred = coerce_json(clean_json_blob(txt))
            if pred is not None and normalize_call_names_only(pred) == normalize_call_names_only(gold):
                hit = True
        if hit:
            name_hit += 1
        per_item.append({"question": data[idx]["question"],
                         "domain": data[idx]["domain"],
                         "solver_agreement": data[idx]["solver_agreement"],
                         "fmt": fmts[idx], "acc": accs[idx],
                         "name_hit": hit})

    n = len(data)
    # 分档看涨在哪里：评测集按 agreement 分过档，这里按同一个口径汇总
    bands = {"lt0.3": 0, "0.3-0.5": 0, "0.5-0.7": 0, "0.7-0.85": 0,
             "0.85-1.0": 0, "eq1.0": 0}
    band_n = dict(bands)
    for d, a in zip(data, accs):
        ag = d["solver_agreement"]
        k = ("eq1.0" if ag >= 1.0 else "lt0.3" if ag < 0.3 else
             "0.3-0.5" if ag < 0.5 else "0.5-0.7" if ag < 0.7 else
             "0.7-0.85" if ag < 0.85 else "0.85-1.0")
        bands[k] += a
        band_n[k] += 1

    return {
        "sol_n": n,
        "sol_fmt": sum(fmts) / n,
        "sol_acc": sum(accs) / n,
        "sol_acc_pos": sum(1 for a in accs if a > 0) / n,
        "sol_name_hit": name_hit / n,
        "sol_acc_by_band": {k: (bands[k] / band_n[k] if band_n[k] else None)
                            for k in band_n},
        "sol_band_n": band_n,
        "_per_item": per_item,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="这一行的名字，比如 iter1_gen / iter2_solver")
    ap.add_argument("--gen_ckpt", default=None)
    ap.add_argument("--solver_ckpt", default=None)
    ap.add_argument("--note", default="")
    args = ap.parse_args()
    if not args.gen_ckpt and not args.solver_ckpt:
        ap.error("--gen_ckpt 和 --solver_ckpt 至少要给一个")

    row = {"tag": args.tag, "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
           "note": args.note}
    t0 = time.time()
    per_item = {}
    if args.gen_ckpt:
        print("[eval] generator %s" % args.gen_ckpt, flush=True)
        row["gen_ckpt"] = args.gen_ckpt
        r = eval_generator(args.gen_ckpt)
        per_item["gen"] = r.pop("_per_item")
        row.update(r)
    if args.solver_ckpt:
        print("[eval] solver %s" % args.solver_ckpt, flush=True)
        row["solver_ckpt"] = args.solver_ckpt
        r = eval_solver(args.solver_ckpt)
        per_item["solver"] = r.pop("_per_item")
        row.update(r)
    row["eval_seconds"] = round(time.time() - t0, 1)

    with open(METRICS, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # 逐条结果单独存一份。多轮对照要的是配对检验（同样的题，这一轮对上一轮对不对），
    # 只有聚合值做不了 McNemar——聚合值的方差把配对带来的方差抵消全浪费了。
    ev_dir = os.path.join(HERE, "evals")
    os.makedirs(ev_dir, exist_ok=True)
    with open(os.path.join(ev_dir, args.tag + ".json"), "w", encoding="utf-8") as f:
        json.dump(per_item, f, ensure_ascii=False)

    print("\n=== %s ===" % args.tag)
    for k in ("gen_four_tag", "gen_pv_ok", "sol_fmt", "sol_acc",
              "sol_acc_pos", "sol_name_hit", "eval_seconds"):
        if k in row:
            v = row[k]
            print("  %-16s %s" % (k, ("%.4f" % v) if isinstance(v, float) else v))
    print("  -> %s" % METRICS)


if __name__ == "__main__":
    main()
