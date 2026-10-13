#!/usr/bin/env python3
"""用强教师模型批量生成 generator 的 SFT 热启动数据。

为什么要这个脚本：没热启动过的 generator 在 RL 阶段几乎产生不出规定的四段标签格式，
奖励信号常年为 0，GRPO 没有梯度。先用一个强教师模型产出一批"格式 100% 合格、
任务 100% 有效"的样例做一次 SFT，把 generator 先拽到能写出正确格式的位置，
再进 RL，才有意义。

两个"一致"是本脚本的核心：
  1. 输入分布一致 —— prompt 模板与 control-spec 采样完全复用 step1 训练时的
     step1_generator（SYSTEM_PROMPT_GENERATOR_TEMPLATE / sample_spec），
     不另写一份，避免和训练分布漂移。
  2. 判定标准一致 —— 只用训练时那套 reward 逻辑（rewards_generator 的
     格式分项与有效性分项）做硬过滤，要求格式满分、有效性满分，因此留下的
     样本在 RL 里都是"满分正例"。

用法：
    export TEACHER_API_KEY=sk-xxxx
    python gen_sft_data.py --n 300 --workers 8

先只看配比、不调 API：
    python gen_sft_data.py --n 300 --dry-run
"""

import argparse
import json
import os
import random
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed


def _require_repo_venv():
    """提前确认跑在本仓库的 venv 里。

    本脚本只用 step1_generator 里的纯 Python 常量与采样函数，但那个模块顶层
    import 了 trl/torch。跑在别的 venv 上会以一个和本脚本毫无关系的错误挂掉
    （实测：search-zero 的 venv 里 trl 1.13 撞上 torch 的 FSDPModule 导入失败），
    很难看出真正原因是 venv 选错了。这里先拦一道，报人话。
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

from openai import OpenAI

# 复用 spec 采样：单一数据源，不重复定义。
from step1_generator import sample_spec, DOMAIN_WEIGHTS
# prompt 模板以 step2_gen 为准：SFT 出来的模型最终是在 step2_gen 里产数据的，
# 训练它的 prompt 必须和那里逐字相同。step1_generator 本地那份和 step2_gen
# 曾有缩进和措辞两处漂移（"Generate a new" vs "Generate one new"），已统一。
from step2_gen import (
    parse_and_validate,
    SYSTEM_PROMPT_GENERATOR_TEMPLATE,
    USER_PROMPT_GENERATOR,
)
# 复用训练侧的奖励判定：留下的样本必须在这套标准下拿满分。
from rewards_generator import (
    _parse_completion,
    _json_loads_relaxed,
    _question_gate_mask,
    reward_tags_parsed,
    reward_tools_valid_json,
    reward_gold_valid_json,
    reward_gold_tool_exists,
    reward_gold_args_valid,
    reward_gold_values_valid,
)

DEFAULT_BASE_URL = "https://llm-eaqzvjv93oadnpx3.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3.7-flash"

# 各分项在训练里的权重（与 rewards_generator 中的常量保持一致）。
A_TAGS = A_TOOLS = A_GOLD = 1.0 / 3.0
A_EXISTS = A_ARGS = 0.4
A_VALUES = 0.2

_print_lock = threading.Lock()


def build_system_prompt(spec):
    return SYSTEM_PROMPT_GENERATOR_TEMPLATE.format(**spec)


def canonical_target(blocks):
    """把四段内容按 prompt 要求的顺序重排成规范的 assistant 目标串。

    教师偶尔会在前面加寒暄、或用代码块包裹，这里只取各段内容重新拼装，
    既不改变内容，又保证目标串严格符合训练时的格式约定。

    think 段必须原样包进 <think> / </think>：reward_tags_parsed 数的是这四个
    标签是否都在（REQUIRED_GENERATOR_TAGS），少一个格式分就只有 3/4，
    fmt 掉到 0.917，整条样本会被自己的满分标准判掉。
    """
    return (
        "<think>\n" + blocks["think"].strip() + "\n</think>\n\n"
        "<question>\n" + blocks["question"].strip() + "\n</question>\n"
        "<available_tools>\n" + blocks["available_tools"].strip() + "\n</available_tools>\n"
        "<tool_call_answer>\n" + blocks["tool_call_answer"].strip() + "\n</tool_call_answer>"
    )


def validate(text, spec):
    """用训练时的 reward 逻辑严格校验一条样本。

    返回 (ok, reason, info)；info 仅在 ok 时有意义。
    """
    parsed = _parse_completion(text)
    present = parsed["present"]
    missing = [t for t in ("think", "question", "available_tools", "tool_call_answer")
               if not present.get(t)]
    if missing:
        return False, "missing_tags:" + ",".join(missing), None

    target = canonical_target(parsed["blocks"])

    # 全部判定都跑在最终要训练的目标串上，确保"存下来的就是满分的"。
    comps = [[{"content": target}]]
    r1 = reward_tags_parsed(None, comps)[0]
    r2 = reward_tools_valid_json(None, comps)[0]
    r3 = reward_gold_valid_json(None, comps)[0]
    fmt = A_TAGS * r1 + A_TOOLS * r2 + A_GOLD * r3
    if fmt < 1.0 - 1e-9:
        return False, f"format_score={fmt:.3f}", None

    v1 = reward_gold_tool_exists(None, comps)[0]
    v2 = reward_gold_args_valid(None, comps)[0]
    v3 = reward_gold_values_valid(None, comps)[0]
    val = A_EXISTS * v1 + A_ARGS * v2 + A_VALUES * v3
    if val < 1.0 - 1e-9:
        return False, f"validity_score={val:.3f}", None

    if _question_gate_mask(comps)[0] != 1.0:
        return False, "question_gated", None

    # ---- 下游闸门：step2_gen 实际用来决定样本生死的那一道 ----
    # reward 系列只覆盖格式与 gold 一致性，step2_gen.parse_and_validate 还查
    # contains_cjk、question 长度/模板化、tool schema 完整性等。SFT 数据若在
    # 这里过关、在 step2_gen 被拒，等于教模型一种下游立刻丢掉的行为。
    # 注意判在 canonical target 上：那才是模型真正会吐出的东西。
    _s, pv_reason = parse_and_validate(target)
    if pv_reason != "ok":
        return False, "pv:" + pv_reason, None

    # ---- 额外的 spec 贴合度检查（prompt 里明确要求，教师必须满足）----
    p = _parse_completion(target)
    tools = p["tools"] or []
    if len(tools) != spec["tool_menu_size"]:
        return False, f"tool_menu_size={len(tools)}!={spec['tool_menu_size']}", None

    gold_obj = _json_loads_relaxed(p["blocks"]["tool_call_answer"])
    if not isinstance(gold_obj, list):
        return False, "gold_not_a_list", None
    if len(gold_obj) != spec["num_calls"]:
        return False, f"num_calls={len(gold_obj)}!={spec['num_calls']}", None

    # 参数必须是扁平标量（prompt 规则 5）。
    for call in gold_obj:
        args = call.get("arguments", {}) if isinstance(call, dict) else {}
        for v in args.values():
            if isinstance(v, (list, dict)) or v is None:
                return False, "non_scalar_argument", None

    if spec["context_type"] == "multi_turn":
        q = p["blocks"]["question"]
        has_conv = bool(re.search(r"(?i)\bconversation\b", q)) or \
                   bool(re.search(r"(?im)^\s*(user|agent|assistant)\s*:", q))
        if not has_conv:
            return False, "multi_turn_shape_missing", None

    info = {
        "format": round(fmt, 4),
        "validity": round(val, 4),
        "num_tools": len(tools),
        "num_calls": len(gold_obj),
        "target": target,
    }
    return True, "ok", info


def call_teacher(client, model, system_prompt, temperature, max_tokens,
                 extra_body=None, api_retries=3):
    """调用教师模型；网络/限流类错误做指数退避重试。"""
    last_err = None
    for attempt in range(api_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": USER_PROMPT_GENERATOR},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body=extra_body,
            )
            return resp.choices[0].message.content or "", None
        except Exception as e:  # noqa: BLE001 - 需要把任何 API 异常都挡在这一层
            last_err = e
            time.sleep(min(2 ** attempt, 8))
    return None, str(last_err)


def gen_one(client, args, spec):
    """对单个 spec 反复尝试，直到拿到合格样本或用完次数。"""
    system_prompt = build_system_prompt(spec)
    last_reason = "no_attempt"
    for _ in range(args.max_attempts):
        text, err = call_teacher(
            client, args.model, system_prompt, args.temperature, args.max_tokens,
            extra_body=args.extra_body,
        )
        if text is None:
            last_reason = f"api_error:{err}"
            continue
        ok, reason, info = validate(text, spec)
        if ok:
            return {"ok": True, "spec": spec, "info": info, "system_prompt": system_prompt}
        last_reason = reason
    return {"ok": False, "spec": spec, "reason": last_reason}


def histogram(specs):
    return {
        "domain": Counter(s["domain"] for s in specs),
        "context_type": Counter(s["context_type"] for s in specs),
        "num_calls": Counter(s["num_calls"] for s in specs),
        "tool_menu_size": Counter(s["tool_menu_size"] for s in specs),
    }


def print_histogram(title, h, total):
    print(f"\n=== {title} (n={total}) ===")
    for key in ("context_type", "num_calls", "tool_menu_size"):
        parts = "  ".join(f"{k}:{v}" for k, v in sorted(h[key].items()))
        print(f"  {key:16s} {parts}")
    print("  domain:")
    for k, v in sorted(h["domain"].items()):
        print(f"    {k:22s} {v:4d}  ({v / max(total, 1):.1%})")


def main():
    ap = argparse.ArgumentParser(description="生成 generator 的 SFT 热启动数据")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--api-key", default=os.environ.get("TEACHER_API_KEY")
                    or os.environ.get("OPENAI_API_KEY"))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--n", type=int, default=300, help="目标合格样本数")
    ap.add_argument("--out", default="sft_generator_data.jsonl")
    ap.add_argument("--rejected-out", default="sft_generator_data.rejected.jsonl")
    ap.add_argument("--summary-out", default="sft_generator_data.summary.json")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-attempts", type=int, default=4, help="每个 spec 最多重试几次")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-tokens", type=int, default=4096,
                    help="实测单条 1000-1900 token；开思考时约 4100。4096 是安全下限")
    ap.add_argument("--thinking", action="store_true",
                    help="允许教师走独立 reasoning_content。默认关闭：快约 4 倍、"
                         "省约 4 倍 token，且模型会把思考写进 <think> 块里（正是 SFT 目标）")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--dry-run", action="store_true", help="只打印配比与示例 prompt，不调 API")
    args = ap.parse_args()

    random.seed(args.seed)
    # qwen3.7-flash 是思考模型：默认关掉独立 reasoning，否则 reasoning_content
    # 会吃掉 max_tokens 且 content 里的 <think> 只剩个空壳。
    args.extra_body = None if args.thinking else {"enable_thinking": False}
    specs = [sample_spec() for _ in range(args.n)]
    target_hist = histogram(specs)

    if args.dry_run:
        print_histogram("目标配比（与 step1 训练分布一致）", target_hist, len(specs))
        print("\n=== 示例 system prompt ===")
        print(build_system_prompt(specs[0]))
        print("\n=== user prompt ===")
        print(USER_PROMPT_GENERATOR)
        return

    if not args.api_key:
        sys.exit("缺少 API key：设置环境变量 TEACHER_API_KEY，或传 --api-key。")

    client = OpenAI(base_url=args.base_url, api_key=args.api_key, timeout=args.timeout)

    accepted, rejected = [], []
    counter = {"done": 0}
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(gen_one, client, args, s) for s in specs]
        for fut in as_completed(futures):
            res = fut.result()
            if res["ok"]:
                accepted.append(res)
            else:
                rejected.append(res)
            counter["done"] += 1
            with _print_lock:
                if counter["done"] % 10 == 0 or counter["done"] == len(specs):
                    rate = counter["done"] / max(time.time() - t0, 1e-9)
                    print(f"[{counter['done']}/{len(specs)}] "
                          f"accepted={len(accepted)} rejected={len(rejected)} "
                          f"({rate:.2f}/s)", flush=True)

    with open(args.out, "w", encoding="utf-8") as f:
        for r in accepted:
            f.write(json.dumps({
                "messages": [
                    {"role": "system", "content": r["system_prompt"]},
                    {"role": "user", "content": USER_PROMPT_GENERATOR},
                    {"role": "assistant", "content": r["info"]["target"]},
                ],
                "spec": r["spec"],
                "scores": {"format": r["info"]["format"], "validity": r["info"]["validity"]},
            }, ensure_ascii=False) + "\n")

    if rejected:
        with open(args.rejected_out, "w", encoding="utf-8") as f:
            for r in rejected:
                f.write(json.dumps({"spec": r["spec"], "reason": r["reason"]},
                                   ensure_ascii=False) + "\n")

    acc_hist = histogram([r["spec"] for r in accepted])
    summary = {
        "model": args.model,
        "requested": args.n,
        "accepted": len(accepted),
        "rejected": len(rejected),
        "accept_rate": round(len(accepted) / max(args.n, 1), 4),
        "target_distribution": {k: dict(v) for k, v in target_hist.items()},
        "accepted_distribution": {k: dict(v) for k, v in acc_hist.items()},
        "reject_reasons": dict(Counter(r["reason"].split(":")[0] for r in rejected)),
    }
    with open(args.summary_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print_histogram("目标配比", target_hist, len(specs))
    print_histogram("实际合格样本配比", acc_hist, len(accepted))
    print(f"\n合格 {len(accepted)}/{args.n}（接受率 {summary['accept_rate']:.1%}）")
    print(f"写出：{args.out} / 摘要：{args.summary_out}"
          + (f" / 拒绝样本：{args.rejected_out}" if rejected else ""))
    if rejected:
        print("主要拒绝原因：", summary["reject_reasons"])


if __name__ == "__main__":
    main()