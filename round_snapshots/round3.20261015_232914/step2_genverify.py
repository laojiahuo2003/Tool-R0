#!/usr/bin/env python3
"""
step2_genverify.py

Post-verification script:
- Reads intermediate JSON produced by gen.py (list of dicts containing question/tools/calls/raw/meta).
- Uses a solver model to re-solve each sample K times.
- Keeps a sample iff solver exact-match agreement >= tau.
- Writes filtered intermediate JSON (same schema as input).
- Writes a human-readable report txt with stats for this iteration.

This does NOT modify gen.py.
"""

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


_PLACEHOLDER_PATTERNS = [
    r"\bthe generated user question here\b",
    r"\buser question must be from the specified domain\b",
    r"\bgenerate one new tool-calling task now\b",
    r"\bgenerate a new tool-calling task now\b",
    r"\bcontrol spec\b",
    r"\brules to satisfy\b",
    r"\bavailable_tools\b",
    r"\btool_call_answer\b",
    r"\bprivate reasoning\b",
    r"\[the private reasoning here\]",
    r"\[.*generated user question.*\]",
]

_PLACEHOLDER_RE = re.compile("|".join(_PLACEHOLDER_PATTERNS), re.IGNORECASE | re.DOTALL)

def is_placeholder_question(q: str) -> bool:
    if not q or not isinstance(q, str):
        return True
    qq = q.strip()
    if len(qq) < 8:
        return True
    return _PLACEHOLDER_RE.search(qq) is not None



TAG_PATTERNS = {
    "tool_call_answer": re.compile(r"<tool_call_answer>(.*?)</tool_call_answer>", re.DOTALL | re.IGNORECASE),
}


def extract_tool_call_answer_block(text: str) -> Optional[str]:
    if not text:
        return None

    m = re.search(r"<tool_call_answer>(.*?)</tool_call_answer>", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()

    m = re.search(r"<tool_call_answer>\s*(.*)$", text, re.DOTALL | re.IGNORECASE)
    if m:
        tail = m.group(1).strip()
        if "[" in tail:
            return tail

    return None



def clean_json_blob(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    i, j = s.find("["), s.rfind("]")
    if i != -1 and j != -1 and j > i:
        s = s[i:j+1]
    return s.strip()



def coerce_json(s: str) -> Optional[Any]:
    try:
        return json.loads(s)
    except Exception:
        return None


def normalize_call_json(obj: Any) -> Optional[str]:
    """
    Canonicalize calls for exact-match comparison:
      - must be list[dict{name:str, arguments:dict}]
      - sort argument keys
      - dump with sort_keys=True
    """
    if not isinstance(obj, list):
        return None
    norm = []
    for c in obj:
        if not isinstance(c, dict):
            return None
        name = c.get("name")
        args = c.get("arguments")
        if not isinstance(name, str) or not isinstance(args, dict):
            return None
        norm.append({"name": name, "arguments": dict(sorted(args.items(), key=lambda x: x[0]))})
    return json.dumps(norm, ensure_ascii=False, sort_keys=True)


def normalize_call_names_only(obj: Any) -> Optional[str]:
    """
    Canonicalize only the sequence of tool names.
    Keeps call order (important if num_calls can be 2).
    """
    if not isinstance(obj, list):
        return None
    names = []
    for c in obj:
        if not isinstance(c, dict):
            return None
        n = c.get("name")
        if not isinstance(n, str):
            return None
        names.append(n)
    return json.dumps(names, ensure_ascii=False)



def gold_norm(calls: List[Dict[str, Any]]) -> str:
    return normalize_call_json(calls) or ""


_TAG_OPEN = "<tool_" + "call_answer>"
_TAG_CLOSE = "</tool_" + "call_answer>"

# The base model never emits the tag from the bare description above. Measured on
# the 18 samples step2_gen produced (n=3 draws each): tag emission 11%, gold
# name-match 0%. Appending two worked examples lifts that to 35% / 11% per draw,
# which is enough to clear tau_verify=0.20 at k=10 for roughly a third of the
# samples. Without this every sample dies on missing_tag and iter1_data.json is
# always empty, so step3 never trains and no solver checkpoint is ever written.
# Demos use tool names absent from the generator's domains so a model that merely
# parrots them is distinguishable from one that answers.
_FEWSHOT_DEMOS = (
    "\n\nFollow these examples exactly.\n\n"
    "Example 1\nUser request:\nCancel my 3pm dentist appointment on Friday.\n\n"
    "Available tools (JSON):\n"
    '[{"name": "list_appointments", "description": "List appointments for a date", '
    '"parameters": {"type": "object", "properties": {"date": {"type": "string", '
    '"description": "Date YYYY-MM-DD"}}, "required": ["date"]}}, '
    '{"name": "cancel_appointment", "description": "Cancel an appointment by id", '
    '"parameters": {"type": "object", "properties": {"appointment_id": {"type": "string"}}, '
    '"required": ["appointment_id"]}}]\n\n'
    "Assistant:\n"
    "<think>I need the appointment's id before I can cancel it.</think>\n"
    + _TAG_OPEN
    + '[{"name": "list_appointments", "arguments": {"date": "2024-06-14"}}, '
      '{"name": "cancel_appointment", "arguments": {"appointment_id": "appt_3pm_friday"}}]'
    + _TAG_CLOSE
    + "\n\nExample 2\nUser request:\nHow many open tickets does the payments team have right now?\n\n"
    "Available tools (JSON):\n"
    '[{"name": "list_teams", "description": "List teams", "parameters": {"type": "object", '
    '"properties": {}, "required": []}}, '
    '{"name": "count_tickets", "description": "Count tickets for a team", '
    '"parameters": {"type": "object", "properties": {"team": {"type": "string"}, '
    '"status": {"type": "string"}}, "required": ["team"]}}]\n\n'
    "Assistant:\n"
    "<think>I need the payments team id before I can count its tickets.</think>\n"
    + _TAG_OPEN
    + '[{"name": "list_teams", "arguments": {}}, '
      '{"name": "count_tickets", "arguments": {"team": "payments", "status": "open"}}]'
    + _TAG_CLOSE
)

SYSTEM_PROMPT_SOLVER = (
    "A conversation between user and tool-calling assistant. The user asks a question, and the assistant uses tools to solve it. The "
    "assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
    "The reasoning process and answer are enclosed within <think></think> and " + _TAG_OPEN + _TAG_CLOSE + " tags, i.e., <think>\nThis is my "
    "reasoning.\n</think>\n" + _TAG_OPEN + "[{\"name\": \"<tool_name>\", \"arguments\": {\"arg1\": \"value\", \"arg2\": \"value2\", ...}}, ...]" + _TAG_CLOSE + "."
    + _FEWSHOT_DEMOS
)

SOLVER_USER_TMPL = """User request:
{question}

Available tools (JSON):
{tools_json}
"""

def build_solver_prompt(question, tools, tokenizer):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_SOLVER},
        {"role": "user", "content": SOLVER_USER_TMPL.format(
            question=question,
            tools_json=json.dumps(tools, ensure_ascii=False),
        )},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def solver_agreement_batch(
    llm_solver: LLM,
    prompts: List[str],
    gold_calls_list: List[List[Dict[str, Any]]],
    k: int,
    temperature: float,
    max_tokens: int,
    chunk_size: int,
) -> Tuple[List[float], List[str], List[Optional[Any]], List[str]]:
    """
    Returns:
      - agreement rates
      - failure reason (best-effort)
      - first valid parsed prediction per prompt (for debugging/reporting)
    """
    sp = SamplingParams(
        temperature=temperature,
        top_p=1.0 if temperature == 0.0 else 0.95,
        max_tokens=max_tokens,
        n=k,
        stop=["</tool_call_answer>"],
        include_stop_str_in_output=True,
    )

    gold_norms = [normalize_call_names_only(gc) or "" for gc in gold_calls_list]

    rates: List[float] = [0.0] * len(prompts)
    reasons: List[str] = ["ok"] * len(prompts)
    preds: List[Optional[Any]] = [None] * len(prompts)
    raw_samples: List[str] = [""] * len(prompts) 

    for start in range(0, len(prompts), chunk_size):
        end = min(start + chunk_size, len(prompts))
        print(f"[verify] processing {start}-{end}/{len(prompts)}", file=sys.stderr)  
        req_outs = llm_solver.generate(prompts[start:end], sp)

        for j, req in enumerate(req_outs):
            idx = start + j
            g = gold_norms[idx]
            ok = 0
            local = Counter()

            first_valid_pred = None

            if req.outputs:
                raw_samples[idx] = req.outputs[0].text[:500]

            for o in req.outputs:
                txt = extract_tool_call_answer_block(o.text)
                if txt is None:
                    local["missing_tag"] += 1
                    continue

                txt = clean_json_blob(txt)
                pred = coerce_json(txt)
                if pred is None:
                    local["json_parse_fail"] += 1
                    continue

                if first_valid_pred is None:
                    first_valid_pred = pred

                pn = normalize_call_names_only(pred)
                if pn is None:
                    local["bad_call_schema"] += 1
                    continue

                if pn == g:
                    ok += 1
                else:
                    local["mismatch"] += 1

            rates[idx] = ok / float(k)
            preds[idx] = first_valid_pred

            if ok < k and local:
                reasons[idx] = local.most_common(1)[0][0]

    return rates, reasons, preds, raw_samples


def main():
    t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--solver_model", type=str, required=True)
    ap.add_argument("--in_intermediate_json", type=str, required=True,
                    help="Input JSON from gen.py (intermediate).")
    ap.add_argument("--out_intermediate_json", type=str, required=True,
                    help="Filtered intermediate JSON (same schema) for probe.py.")
    ap.add_argument("--report_txt", type=str, required=True,
                    help="Write drop statistics report here.")

    ap.add_argument("--k_verify", type=int, default=6)
    ap.add_argument("--tau_verify", type=float, default=0.5)
    ap.add_argument("--temp_verify", type=float, default=0.0)
    ap.add_argument("--max_tokens_verify", type=int, default=1024)
    ap.add_argument("--verify_batch_size", type=int, default=32)

    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    ap.add_argument("--max_model_len", type=int, default=4096)

    ap.add_argument("--report_examples", type=int, default=5)

    args = ap.parse_args()

    print(f"[cfg] solver_model={args.solver_model}", file=sys.stderr)
    print(f"[cfg] in={args.in_intermediate_json}", file=sys.stderr)
    print(f"[cfg] out={args.out_intermediate_json}", file=sys.stderr)
    print(f"[cfg] report={args.report_txt}", file=sys.stderr)
    print(f"[cfg] k={args.k_verify} tau={args.tau_verify} temp={args.temp_verify}", file=sys.stderr)

    llm_solver = LLM(
        model=args.solver_model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enforce_eager=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.solver_model, trust_remote_code=True)

    with open(args.in_intermediate_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list (as produced by gen.py).")

    prompts: List[str] = []
    gold_calls_list: List[List[Dict[str, Any]]] = []
    kept_input_rows: List[Dict[str, Any]] = []

    dropped_early = []
    drop_early_reasons = Counter()

    for d in data:
        q = d.get("question", "")
        if is_placeholder_question(q):
            meta = d.get("meta", {})
            if not isinstance(meta, dict):
                meta = {}
            meta["solver_agreement"] = 0.0
            meta["solver_disagree_reason"] = "placeholder_question"
            d["meta"] = meta

            dropped_early.append(d)
            drop_early_reasons["placeholder_question"] += 1
            continue

        kept_input_rows.append(d)
        prompts.append(build_solver_prompt(d["question"], d["tools"], tokenizer))
        gold_calls_list.append(d["calls"])

    print(
        f"[verify] loaded {len(data)} samples. early_drop={len(dropped_early)} to avoid placeholder compute. verifying {len(kept_input_rows)}...",
        file=sys.stderr,
    )


    rates, reasons, preds, raw_samples = solver_agreement_batch(
        llm_solver=llm_solver,
        prompts=prompts,
        gold_calls_list=gold_calls_list,
        k=args.k_verify,
        temperature=args.temp_verify,
        max_tokens=args.max_tokens_verify,
        chunk_size=args.verify_batch_size,
    )

    kept = []
    dropped = []
    drop_reasons = Counter()

    for d, r, rsn, pred, raw in zip(kept_input_rows, rates, reasons, preds, raw_samples):
        meta = d.get("meta", {})
        if not isinstance(meta, dict):
            meta = {}
        meta["solver_agreement"] = r
        meta["solver_disagree_reason"] = rsn
        meta["solver_predicted_calls"] = pred
        meta["solver_raw_output"] = raw
        d["meta"] = meta

        if r >= args.tau_verify:
            kept.append(d)
        else:
            dropped.append(d)
            drop_reasons[rsn] += 1
    drop_reasons.update(drop_early_reasons)

    os.makedirs(os.path.dirname(args.out_intermediate_json) or ".", exist_ok=True)
    with open(args.out_intermediate_json, "w", encoding="utf-8") as f:
        json.dump(kept, f, ensure_ascii=False, indent=2)

    os.makedirs(os.path.dirname(args.report_txt) or ".", exist_ok=True)
    with open(args.report_txt, "w", encoding="utf-8") as rf:
        rf.write("step2_genverify.py report\n")
        rf.write(f"input:  {args.in_intermediate_json}\n")
        rf.write(f"output: {args.out_intermediate_json}\n")
        rf.write(f"solver_model: {args.solver_model}\n")
        rf.write(f"k_verify: {args.k_verify}\n")
        rf.write(f"tau_verify: {args.tau_verify}\n")
        rf.write(f"temp_verify: {args.temp_verify}\n")
        rf.write(f"max_tokens_verify: {args.max_tokens_verify}\n")
        rf.write(f"verify_batch_size: {args.verify_batch_size}\n")
        rf.write("\n")
        rf.write(f"TOTAL:   {len(data)}\n")
        rf.write(f"KEPT:    {len(kept)}\n")
        rf.write(f"DROPPED: {len(dropped)}\n")
        rf.write("\nDrop reasons (top):\n")
        for k, v in drop_reasons.most_common(20):
            rf.write(f"  - {k}: {v}\n")

        rf.write("\nAgreement distribution:\n")
        bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        for i in range(len(bins) - 1):
            count = sum(1 for r in rates if bins[i] <= r < bins[i+1])
            rf.write(f"  [{bins[i]:.1f}, {bins[i+1]:.1f}): {count}\n")
        count_perfect = sum(1 for r in rates if r == 1.0)
        rf.write(f"  [1.0, 1.0]: {count_perfect}\n")

        kept_domains = Counter(d.get("meta", {}).get("gen_spec", {}).get("domain", "unknown") for d in kept)

        all_dropped = dropped + dropped_early
        dropped_domains = Counter(d.get("meta", {}).get("gen_spec", {}).get("domain", "unknown") for d in all_dropped)

        
        rf.write("\nDomain distribution (kept):\n")
        for domain, count in kept_domains.most_common():
            rf.write(f"  - {domain}: {count}\n")
        
        rf.write("\nDomain distribution (dropped):\n")
        for domain, count in dropped_domains.most_common():
            rf.write(f"  - {domain}: {count}\n")

        if args.report_examples > 0 and dropped:
            rf.write("\n--- Dropped examples (truncated) ---\n")
            for i, d in enumerate(dropped[: args.report_examples]):
                rf.write(f"\n[{i+1}] solver_agreement={d.get('meta', {}).get('solver_agreement'):.3f} "
                         f"reason={d.get('meta', {}).get('solver_disagree_reason')}\n")
                rf.write("QUESTION:\n")
                rf.write((d.get("question", "")[:600] + "\n") if d.get("question") else "\n")
                rf.write("AVAILABLE TOOLS:\n")
                rf.write(json.dumps(d.get("tools", []), ensure_ascii=False)[:600] + "\n")
                rf.write("GOLD CALLS:\n")
                rf.write(json.dumps(d.get("calls", []), ensure_ascii=False)[:600] + "\n")
                rf.write("PREDICTED CALLS:\n")
                rf.write(json.dumps(d.get("meta", {}).get("solver_predicted_calls", []), ensure_ascii=False)[:600] + "\n")
                rf.write("\nSOLVER RAW OUTPUT:\n")
                rf.write(d.get("meta", {}).get("solver_raw_output", "N/A") + "\n")

    dt = time.time() - t0
    print(f"[out] wrote kept={len(kept)} dropped={len(dropped)} to {args.out_intermediate_json}", file=sys.stderr)
    print(f"[out] report saved to {args.report_txt}", file=sys.stderr)
    print(f"[time] {dt:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
