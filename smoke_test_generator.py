#!/usr/bin/env python3
"""SFT 冷启动是否真的成功：不猜 loss，直接采样看标签发出率。

为什么需要这个脚本：SFT 的 loss 低只说明"背下来了"，不说明"采样时会用"。
Tool-R0 的 generator 任务是从四个标量凭空发明整套任务，唯一硬指标是
  1. 四个标签（<think>/question/available_tools/tool_call_answer）是否都在
  2. 产出的字符串能不能过 step2_gen.parse_and_validate
这两件事才是 reward 能否从 0 爬起来的开关。

用法：
    python smoke_test_generator.py Qwen/Qwen2.5-0.5B-Instruct           # 基线
    python smoke_test_generator.py ./qwen2.5-0.5b-instruct-tool-r0/iter1_generator_sft
"""
import argparse
import json
import os
import random
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")


def _require_repo_venv():
    """提前确认跑在本仓库的 venv 里。

    step2_gen 顶层 import vllm，而这个脚本只需要它的 parse_and_validate 和
    两个 prompt 常量。跑在裸 python 或别的 venv 上会以一个和本脚本毫无关系的
    ModuleNotFoundError 挂掉（实测踩过），很难看出真正原因是 venv 选错了。
    这里先拦一道，报人话。
    """
    here = os.path.dirname(os.path.abspath(__file__))
    venv = os.path.join(here, ".venv")
    if not os.path.isdir(venv):
        return
    if os.path.realpath(sys.prefix) != os.path.realpath(venv):
        sys.exit("venv 不对：请先激活本仓库的环境\n"
                 f"    source {venv}/bin/activate\n"
                 f"当前解释器是 {sys.executable}")


_require_repo_venv()

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from step2_gen import (
    SYSTEM_PROMPT_GENERATOR_TEMPLATE,
    USER_PROMPT_GENERATOR,
    parse_and_validate,
)
from step1_generator import sample_spec
from rewards_generator import _parse_completion

TAGS = ("think", "question", "available_tools", "tool_call_answer")


def render(spec):
    sp = SYSTEM_PROMPT_GENERATOR_TEMPLATE.format(**spec)
    msgs = [{"role": "system", "content": sp},
            {"role": "user", "content": USER_PROMPT_GENERATOR}]
    return sp, msgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_path")
    ap.add_argument("--n", type=int, default=24, help="采样多少条")
    ap.add_argument("--seed", type=int, default=1234,
                    help="固定 seed：基线与 SFT 后必须用同一批 prompt 才可比")
    ap.add_argument("--temperature", type=float, default=0.8,
                    help="与 step2_gen 采样温度一致，别用贪心——贪心看不出分布")
    ap.add_argument("--max-new-tokens", type=int, default=1600)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    specs = [sample_spec() for _ in range(args.n)]

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, trust_remote_code=True
    ).cuda().eval()
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    # 批量生成必须左 padding：右 padding 会把 pad token 塞在 prompt 之后，
    # 模型真的会把它们当上下文读，生成的标签数会凭空变化（实测踩过）。
    tok.padding_side = "left"

    prompts, specs_used = [], []
    for s in specs:
        _, msgs = render(s)
        prompts.append(tok.apply_chat_template(msgs, tokenize=False,
                                               add_generation_prompt=True))
        specs_used.append(s)

    all_ok = all_present = all_pv = 0
    per_tag = {t: 0 for t in TAGS}
    reasons = {}
    examples = []
    bs = 6
    for i in range(0, len(prompts), bs):
        chunk = prompts[i:i + bs]
        enc = tok(chunk, return_tensors="pt", padding=True,
                  truncation=True, max_length=2048).to("cuda")
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=args.max_new_tokens,
                do_sample=True, temperature=args.temperature,
                top_p=0.95, pad_token_id=tok.pad_token_id,
            )
        for j, o in enumerate(out):
            text = tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True)
            spec = specs_used[i + j]
            p = _parse_completion(text)
            present = [t for t in TAGS if p["present"].get(t)]
            for t in present:
                per_tag[t] += 1
            if len(present) == 4:
                all_present += 1
            _, why = parse_and_validate(text)
            if why == "ok":
                all_pv += 1
            else:
                reasons[why] = reasons.get(why, 0) + 1
            if len(examples) < 3 and why != "ok":
                examples.append((spec, why, text[:400]))
            all_ok += 1
        print(f"  [{min(i + bs, len(prompts))}/{len(prompts)}] "
              f"tags4={all_present} pv_ok={all_pv}", flush=True)

    n = all_ok
    print(f"\n=== {args.model_path}  n={n} temp={args.temperature} ===")
    print(f"  四标签齐全      {all_present:>3}/{n} = {all_present / n:.1%}")
    print(f"  parse_and_validate ok {all_pv:>3}/{n} = {all_pv / n:.1%}")
    print("  单标签发出率：", {t: f"{per_tag[t] / n:.0%}" for t in TAGS})
    if reasons:
        print("  pv 拒绝原因：", dict(sorted(reasons.items(), key=lambda x: -x[1])))
    for spec, why, txt in examples:
        print(f"\n  --- {why} (menu={spec['tool_menu_size']} calls={spec['num_calls']}) ---")
        print("  " + txt.replace("\n", "\n  ")[:400])


if __name__ == "__main__":
    main()
