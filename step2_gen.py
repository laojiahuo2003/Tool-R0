#!/usr/bin/env python3
"""
gen.py

Generator script: generates candidates, validates, filters, and deduplicates.
Outputs intermediate JSON file for probe.py to process.

Full-weights checkpoints expected (HF-compatible dirs).
"""

from transformers import AutoTokenizer


import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from vllm import LLM, SamplingParams

from question_quality import question_quality_reason


DOMAINS = [
    # --- Primary Functional Domains (Explicitly listed in docs) ---
    "finance",              # Stock trading, mortgage calculation, banking APIs
    "travel",               # Flight booking, hotel reservation, airport info
    "math",                 # Algebra, statistics, calculator functions
    "sports",               # Sports scores (e.g., Soccer/Football, NBA)
    "weather",              # Real-time weather data
    "system",               # File System (ls, cd, cat), OS commands
    "database",             # SQL queries (SELECT, INSERT, etc.)
    "vehicle_control",      # Car status, engine control, EV charging
    "communication",        # Messaging (Slack, Email, SMS)
    "entertainment",        # Movies (TMDB), Music (Spotify)
    "retail_ecommerce",     # Inventory management, order status, product search
    "scheduling",           # Calendar management, meeting booking
    "cloud_infrastructure", # VM management, AWS/Cloud resource handling
    "geolocation",          # Maps, routing, distance estimation
    
    # --- Agentic & Technical Domains (Live Updates) ---
    "web_search",           # Multi-hop reasoning, internet search (SerpAPI)
    "memory_management",    # Key-value store, vector database, recursive summarization
    "programming",          # Code execution, debugging helpers
    "iot",                  # Internet of Things (smart home device control)
    "social_media",         # Twitter/X, Reddit API interactions
    "logistics",            # Shipping tracking, supply chain
    "real_estate",          # Property listing, housing data
    
    # --- Specialized & "Live" API Domains ---
    "food_ordering",        # Restaurant delivery services
    "healthcare",           # Basic medical info (via public APIs)
    "education",            # Course searching, academic references
    "productivity",         # Note-taking (Notion), task management
    "insurance",            # Purchase insurance, policy checks
    "cybersecurity",        # Basic security checks, auth verification
    "legal",                # Regulatory data lookup
    "government",           # Public service APIs
    "news",                 # News aggregation
    "translation",          # Language translation services
    "utilities",            # Energy usage, utility billing
    "customer_support"      # FAQ retrieval, ticket creation
]

DOMAIN_WEIGHTS = {
    "finance": 0.03125,
    "healthcare": 0.03125,
    "productivity": 0.03125,
    "retail_ecommerce": 0.03125,
    "scheduling": 0.03125,
    "database": 0.03125,
    "cloud_infrastructure": 0.03125,
    "system": 0.03125,
    "programming": 0.03125,
    "geolocation": 0.03125,
    "logistics": 0.03125,
    "communication": 0.03125,
    "iot": 0.03125,
    "cybersecurity": 0.03125,
    "insurance": 0.03125,
    "legal": 0.03125,
    "news": 0.03125,
    "weather": 0.03125,
    "sports": 0.03125,
    "entertainment": 0.03125,
    "education": 0.03125,
    "real_estate": 0.03125,
    "food_ordering": 0.03125,
    "translation": 0.03125,
    "utilities": 0.03125,
    "government": 0.03125,
    "memory_management": 0.03125,
    "web_search": 0.03125,
    "social_media": 0.03125,
    "math": 0.03125,
    "vehicle_control": 0.03125,
    "travel": 0.03125,
}

# DOMAIN_WEIGHTS = {
#     # higher weight on precision/routing friendly eval-like domains
#     "finance": 0.10,
#     "healthcare": 0.08,
#     "productivity": 0.08,
#     "retail_ecommerce": 0.08,
#     "scheduling": 0.07,
#     "database": 0.07,
#     "cloud_infrastructure": 0.07,
#     "system": 0.06,
#     "programming": 0.06,
#     "geolocation": 0.05,
#     "logistics": 0.05,
#     "communication": 0.05,
#     "iot": 0.04,
#     "cybersecurity": 0.04,
#     "insurance": 0.03,
#     "legal": 0.03,
#     "news": 0.03,
#     "weather": 0.03,
#     "sports": 0.02,
#     "entertainment": 0.02,
#     "education": 0.02,
#     "real_estate": 0.02,
#     "food_ordering": 0.02,
#     "translation": 0.02,
#     "utilities": 0.02,
#     "government": 0.01,
#     "memory_management": 0.01,
#     "web_search": 0.01,
#     "social_media": 0.01,
#     "math": 0.01,
#     "vehicle_control": 0.01,
#     "travel": 0.02,  # intentionally low
# }


def weighted_choice(weight_dict):
    items = list(weight_dict.items())
    keys, weights = zip(*items)
    return random.choices(keys, weights=weights, k=1)[0]

def sample_spec():
    context_type = random.choices(["single_turn", "multi_turn"], weights=[0.9, 0.1], k=1)[0]

    if context_type == "multi_turn":
        num_calls = 1
    else:
        num_calls = random.choices([1, 2], weights=[0.8, 0.2], k=1)[0]

    if num_calls > 1:
        tool_menu_size = random.choices([3, 4, 5], weights=[0.3, 0.4, 0.3], k=1)[0]
    else:
        bucket = random.choices(["SMALL5", "LARGE"], weights=[0.4, 0.6], k=1)[0]
        tool_menu_size = random.randint(2, 4) if bucket == "SMALL5" else random.randint(5, 8)

    domain = weighted_choice(DOMAIN_WEIGHTS)

    return {
        "domain": domain,
        "num_calls": num_calls,
        "tool_menu_size": tool_menu_size,
        "context_type": context_type,
    }

_PLACEHOLDER_PATTERNS = [
    r"\bthe generated user question here\b",
    r"\buser question must be from the specified domain\b",
    r"\bgenerate one new tool-calling task now\b",
    r"\bgenerate a new tool-calling task now\b",
    r"\bcontrol spec\b",
    r"\brules to satisfy\b",
    r"\bthen, without revealing your reasoning\b",
    r"\[the private reasoning here\]",
    r"\[.*generated user question.*\]",
    r"\[.*user question.*here.*\]",
]

_PLACEHOLDER_RE = re.compile("|".join(_PLACEHOLDER_PATTERNS), re.IGNORECASE | re.DOTALL)

def is_placeholder_text(text: str) -> bool:
    if not text or not text.strip():
        return True
    if len(text.strip()) < 8:
        return True
    return _PLACEHOLDER_RE.search(text) is not None


CJK_RE = re.compile(
    r"[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF\U00020000-\U0002CEAF]"
)

def contains_cjk(text: str) -> bool:
    return bool(text) and (CJK_RE.search(text) is not None)


def is_json_primitive(x: Any) -> bool:
    return x is None or isinstance(x, (str, int, float, bool))

def has_nested_args(args: Any) -> bool:
    """
    True if args is not a flat dict[str -> primitive].
    Rejects lists/dicts nested anywhere (including list of primitives).
    """
    if not isinstance(args, dict):
        return True
    for _, v in args.items():
        if not is_json_primitive(v):
            return True
    return False

PRIMITIVE_TYPES = {
    "string", "str",
    "integer", "int",
    "number", "float", "double",
    "boolean", "bool",
    "date", "datetime",
}

def is_allowed_param_type(t: Any) -> bool:
    if not isinstance(t, str):
        return False
    tt = t.strip().lower()
    if any(k in tt for k in ["list", "array", "dict", "dictionary", "object", "map", "json"]):
        return False
    return tt in PRIMITIVE_TYPES


TAG_PATTERNS = {
    "question": re.compile(r"<question>(.*?)</question>", re.DOTALL | re.IGNORECASE),
    "available_tools": re.compile(r"<available_tools>(.*?)</available_tools>", re.DOTALL | re.IGNORECASE),
    "tool_call_answer": re.compile(r"<tool_call_answer>(.*?)</tool_call_answer>", re.DOTALL | re.IGNORECASE),
}

def extract_tag(text: str, tag: str) -> Optional[str]:
    m = TAG_PATTERNS[tag].search(text)
    if not m:
        return None
    return m.group(1).strip()


def is_bad_tool_name(name: str) -> bool:
    if not name:
        return True
    if len(name) <= 2 and name.isalpha():
        return True
    if name.lower() in {"tool", "function", "api", "call"}:
        return True
    return False


def coerce_json(s: str) -> Optional[Any]:
    try:
        return json.loads(s)
    except Exception:
        return None


def validate_tool_schema(tools: Any) -> Tuple[bool, str]:
    if not isinstance(tools, list) or len(tools) == 0:
        return False, "tools_not_list_or_empty"

    seen = set()
    for t in tools:
        if not isinstance(t, dict):
            return False, "tool_not_object"

        name = t.get("name")
        desc = t.get("description")
        params = t.get("parameters")
        
        req = t.get("required", [])
        if req is not None:
            if not isinstance(req, list) or any(not isinstance(x, str) for x in req):
                return False, "tool_required_not_list_of_strings"


        if not isinstance(name, str) or is_bad_tool_name(name):
            return False, "bad_tool_name"
        if name in seen:
            return False, "duplicate_tool_name"
        seen.add(name)

        if not isinstance(desc, str) or len(desc.strip()) < 5:
            return False, "tool_missing_or_short_description"

        if not isinstance(params, dict) or len(params) == 0:
            return False, "tool_missing_parameters"

        if "properties" in params or params.get("type") == "object":
            props = params.get("properties")
            if props is None or not isinstance(props, dict):
                return False, "tool_parameters_missing_properties"
            if len(props) == 0:
                return False, "tool_parameters_empty_properties"
                
            for _, spec in props.items():
                if not isinstance(spec, dict):
                    return False, "tool_parameters_bad_property_spec"
                if "type" not in spec or not is_allowed_param_type(spec.get("type")):
                    return False, "tool_parameters_nonprimitive_type"
        else:
            for k, v in params.items():
                if not isinstance(k, str):
                    return False, "tool_parameters_bad_arg_name"

                if isinstance(v, str):
                    if not is_allowed_param_type(v):
                        return False, "tool_parameters_nonprimitive_type"
                    continue

                if not isinstance(v, dict):
                    return False, "tool_parameters_bad_arg_spec"
                if "type" not in v:
                    return False, "tool_parameters_missing_arg_type"
                if not is_allowed_param_type(v.get("type")):
                    return False, "tool_parameters_nonprimitive_type"
    return True, "ok"


def validate_calls(calls: Any, tools: List[Dict[str, Any]]) -> Tuple[bool, str]:
    if not isinstance(calls, list) or len(calls) == 0:
        return False, "calls_not_list_or_empty"
    toolmap = {t["name"]: t for t in tools}
    for c in calls:
        if not isinstance(c, dict):
            return False, "call_not_object"
        name = c.get("name")
        args = c.get("arguments")
        if not isinstance(name, str) or name not in toolmap:
            return False, "call_tool_name_unknown"
        if not isinstance(args, dict):
            return False, "call_args_not_object"

        if has_nested_args(args):
            return False, "call_nested_arguments"

        tool = toolmap[name]
        schema = tool.get("parameters", {})

        required = tool.get("required", schema.get("required", []))

        if required is None:
            required = []

        if not isinstance(required, list):
            return False, "required_field_malformed"

        for r in required:
            if not isinstance(r, str):
                return False, "required_field_malformed"
            if r not in args:
                return False, "call_missing_required_arg"

        if len(required) > 0 and len(args) == 0:
            return False, "call_empty_arguments"



    return True, "ok"


def normalize_tools_schema(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    norm_tools = []
    for t in tools:
        t = dict(t)
        params = t.get("parameters", {})

        if isinstance(params, dict) and ("properties" in params or params.get("type") == "object"):
            if "required" not in params:
                params["required"] = []
            t["parameters"] = params
            norm_tools.append(t)
            continue

        if isinstance(params, dict):
            props = {}
            for arg, spec in params.items():
                if isinstance(spec, str):
                    props[arg] = {"type": spec}
                elif isinstance(spec, dict):
                    props[arg] = {k: v for k, v in spec.items() if k in {"type", "description", "enum"}}
                else:
                    continue
            if props:
                tool_required = t.get("required", [])
                t["parameters"] = {
                    "type": "object",
                    "properties": props,
                    "required": tool_required if isinstance(tool_required, list) else []
                }
                norm_tools.append(t)
            continue

        norm_tools.append(t)

    return norm_tools


def non_triviality_heuristic(question: str, calls: List[Dict[str, Any]]) -> Tuple[bool, str]:
    # Shared gate first: meta-instructions, tool-naming questions and leaked
    # transcripts. Keeping this in step2_gen alone let the generator earn format
    # reward during training for questions step2 then discarded.
    reason = question_quality_reason(question)
    if reason is not None:
        return False, reason
    q = question.lower().strip()
    if len(q) < 25:
        return False, "question_too_short"
    banned = ["capital of", "define ", "explain ", "summarize ", "write a poem", "tell me a joke"]
    if any(b in q for b in banned):
        return False, "likely_no_tool_needed"
    if len(calls) == 1 and len(calls[0].get("arguments", {})) <= 1:
        if any(x in q for x in ["what is", "who is", "when is", "where is"]):
            return False, "too_trivial_single_call"
    return True, "ok"


def normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\d", "0", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def tool_signature(tools: List[Dict[str, Any]]) -> str:
    parts = []
    for t in sorted(tools, key=lambda x: x["name"]):
        props = t.get("parameters", {}).get("properties", {})
        keys = sorted(list(props.keys())) if isinstance(props, dict) else []
        parts.append(f'{t["name"]}({",".join(keys)})')
    return "|".join(parts)


def call_signature(calls: List[Dict[str, Any]]) -> str:
    parts = []
    for c in calls:
        keys = sorted(list(c.get("arguments", {}).keys()))
        parts.append(f'{c.get("name")}({",".join(keys)})')
    return "->".join(parts)


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def fingerprints(question: str, tools: List[Dict[str, Any]], calls: List[Dict[str, Any]]) -> Dict[str, str]:
    qn = normalize_text(question)
    ts = tool_signature(tools)
    cs = call_signature(calls)
    return {
        "q_fp": sha1(qn),
        "tool_fp": sha1(ts),
        "call_fp": sha1(cs),
        "combo_fp": sha1(qn + "||" + ts + "||" + cs),
    }


SYSTEM_PROMPT_GENERATOR_TEMPLATE = """You are an expert task generator for tool-calling agents.

FIRST, in your private scratch-pad, reason step-by-step to design a realistic, non-trivial task that cannot be solved without correctly calling one or sometimes multiple tools.

CONTROL SPEC (MUST FOLLOW EXACTLY):
- Domain: {domain}
- Context type: {context_type}  (single_turn or multi_turn)
- Number of available tools: {tool_menu_size} (<available_tools>)
- Number of gold tool calls: {num_calls} (<tool_call_answer>)

RULES TO SATISFY THE SPEC:
1) You MUST output exactly {tool_menu_size} tools in <available_tools>.
2) You MUST output exactly {num_calls} tool calls (JSON list length) in <tool_call_answer>.
3) Domain must be {domain}. Do not drift into other domains.
4) If context_type=multi_turn, embed a short conversation in <question> like: "# Conversation\\nUser: ...\\nAgent: ...\\nUser: ...\\nAgent: ..."
5) Tool arguments must be flat primitives only (no lists, no nested objects).
6) The function values (<value1>, <value2>, ...) MUST be present inside user question (<question>...</question>), otherwise agent cannot solve the task.

THEN, without revealing your reasoning, output the following four blocks in the exact format, NOTHING ELSE:

<think>
Your private reasoning here.
</think>

<question>
Write a natural user question (no bullet points, no meta-instructions, no placeholders).
It must be a natural question, be in domain "{domain}", and mention the exact argument values that appear in <tool_call_answer>.
</question>

<available_tools>
A JSON list of tools. Each tool MUST include: "name", "description", and "parameters". "parameters" MUST be a JSON object mapping param_name -> {{ "type": "...", "description": "..." }} and OPTIONALLY include top-level "required": ["param1", ...] which can be empty list if no required parameters.
[
    {{
        "name": "<tool_name>",
        "description": "<short description>",
        "parameters": {{
        "<param1>": {{"type": "<param1_type>", "description": "<param1_description>"}},
        "<param2>": {{"type": "<param2_type>", "description": "<param2_description>"}},
        ...
        }},
        "required": [<param1>, ...],
    }},
    ...
]
</available_tools>

<tool_call_answer>
[
{{\"name\": \"<tool_name>\", \"arguments\": {{\"<param>\": <value>, ...}}}}
]
</tool_call_answer>"""


USER_PROMPT_GENERATOR = "Generate one new tool-calling task now. Follow the CONTROL SPEC exactly and remember to format the output exactly as instructed."

@dataclass
class Sample:
    question: str
    tools: List[Dict[str, Any]]
    calls: List[Dict[str, Any]]
    raw: str
    meta: Dict[str, Any]


def parse_and_validate(raw: str) -> Tuple[Optional[Sample], str]:
    if contains_cjk(raw):
        return None, "contains_cjk"

    if is_placeholder_text(raw) and ("[the private reasoning here]" in raw.lower() or "[the generated user question here]" in raw.lower()):
        return None, "placeholder_or_prompt_echo"

    
    q = extract_tag(raw, "question")
    t = extract_tag(raw, "available_tools")
    a = extract_tag(raw, "tool_call_answer")
    if q is None or t is None or a is None:
        return None, "missing_tags"
    
    if is_placeholder_text(q):
        return None, "placeholder_question"

    tools = coerce_json(t)
    if tools is None:
        return None, "tools_json_parse_fail"
    ok, reason = validate_tool_schema(tools)
    if not ok:
        return None, reason
    
    tools = normalize_tools_schema(tools)

    calls = coerce_json(a)
    if calls is None:
        return None, "calls_json_parse_fail"
    ok, reason = validate_calls(calls, tools)
    if not ok:
        return None, reason

    ok, reason = non_triviality_heuristic(q, calls)
    if not ok:
        return None, reason

    fps = fingerprints(q, tools, calls)
    s = Sample(question=q, tools=tools, calls=calls, raw=raw, meta={"fingerprints": fps})
    return s, "ok"


def simple_quality_score(s: Sample) -> float:
    score = 0.0
    score += min(1.0, len(s.question) / 220.0)
    score += min(1.0, len(s.calls) / 3.0)
    score += min(1.0, len(s.tools) / 4.0)
    if re.search(r"\b(dummy|placeholder|lorem ipsum)\b", s.question.lower()):
        score -= 1.0
    return float(max(0.0, score))


def sample_to_dict(s: Sample) -> Dict[str, Any]:
    """Convert Sample to dictionary for JSON serialization."""
    return {
        "question": s.question,
        "tools": s.tools,
        "calls": s.calls,
        "raw": s.raw,
        "meta": s.meta,
    }


def main():
    start_time = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--generator_model", type=str, required=True)
    ap.add_argument("--out_intermediate_json", type=str, required=True,
                    help="Output path for intermediate JSON file (will be read by probe.py)")
    ap.add_argument("--n_generate", type=int, default=20000)
    ap.add_argument("--max_tokens_gen", type=int, default=1200)
    ap.add_argument("--temp_gen", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=13)

    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    ap.add_argument("--max_model_len", type=int, default=4096)

    ap.add_argument("--dedup_by", type=str, default="combo_fp",
                    choices=["q_fp", "tool_fp", "call_fp", "combo_fp"])
    ap.add_argument("--keep_per_fingerprint", type=int, default=1)

    args = ap.parse_args()
    random.seed(args.seed)

    print(f"[cfg] generator_model={args.generator_model}", file=sys.stderr)
    print(f"[cfg] out_intermediate_json={args.out_intermediate_json}", file=sys.stderr)

    llm_gen = LLM(
        model=args.generator_model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enforce_eager=True,
    )

    tok = AutoTokenizer.from_pretrained(args.generator_model, trust_remote_code=True)

    batch_size = 256
    candidates: List[Sample] = []
    rejected = 0

    sp_gen = SamplingParams(
        temperature=args.temp_gen,
        max_tokens=args.max_tokens_gen,
        n=1,
    )

    total = args.n_generate
    n_batches = math.ceil(total / batch_size)
    print(f"[gen] generating {total} samples in {n_batches} batches ...", file=sys.stderr)

    rejection_reasons = {}
    sample_raw_outputs = []
    rejected_raws: List[str] = []

    for bi in range(n_batches):
        cur = min(batch_size, total - bi * batch_size)
        specs = [sample_spec() for _ in range(cur)]
        prompts = []

        for spec in specs:
            sys_prompt = SYSTEM_PROMPT_GENERATOR_TEMPLATE.format(**spec)
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": USER_PROMPT_GENERATOR},
            ]
            full_prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(full_prompt)

        outs = llm_gen.generate(prompts, sp_gen)

        for o, spec, prompt in zip(outs, specs, prompts):
            raw = o.outputs[0].text if o.outputs else ""
            if not raw.strip():
                rejected += 1
                reason = "empty_output"
                rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
                rejected_raws.append(raw)
                if len(sample_raw_outputs) < 3:
                    sample_raw_outputs.append(f"[EMPTY OUTPUT] prompt_len={len(prompt)}")
                continue
            
            s, reason = parse_and_validate(raw)
            if s is None:
                rejected += 1
                rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
                rejected_raws.append(raw)

                if reason in {"placeholder_question", "placeholder_or_prompt_echo"} and len(sample_raw_outputs) < 3:
                    sample_raw_outputs.append(raw[:500])

                if len(sample_raw_outputs) < 3:
                    sample_raw_outputs.append(raw[:500])
                continue

            s.meta["gen_spec"] = spec
            s.meta["domain"] = spec["domain"]
            s.meta["quality_score"] = simple_quality_score(s)
            candidates.append(s)

        if (bi + 1) % 10 == 0:
            print(f"[gen] batch {bi+1}/{n_batches} kept={len(candidates)} rejected={rejected}", file=sys.stderr)


    print(f"[gen] done. kept={len(candidates)} rejected={rejected}", file=sys.stderr)
    top = sorted(rejection_reasons.items(), key=lambda x: x[1], reverse=True)[:20]
    print("[gen] rejection reasons (top):", file=sys.stderr)
    for k, v in top:
        print(f"  - {k}: {v}", file=sys.stderr)


    if rejected_raws:
        rejected_path = f"{args.out_intermediate_json}.rejected.txt"
        try:
            with open(rejected_path, "w", encoding="utf-8") as rf:
                rf.write("\n\n\n".join(rejected_raws))
            print(
                f"[gen] wrote {len(rejected_raws)} rejected samples to {rejected_path}",
                file=sys.stderr,
            )
        except Exception as e:
            print(
                f"[gen] failed to write rejected samples to {rejected_path}: {e}",
                file=sys.stderr,
            )

    if len(candidates) == 0:
        print(f"[gen] Rejection reasons: {rejection_reasons}", file=sys.stderr)
        if sample_raw_outputs:
            print(f"[gen] Sample raw outputs (first 500 chars each):", file=sys.stderr)
            for i, raw in enumerate(sample_raw_outputs):
                print(f"[gen] Sample {i+1}:\n{raw}\n---", file=sys.stderr)
        
        if rejection_reasons.get("empty_output", 0) > 0:
            print(f"[gen] WARNING: {rejection_reasons['empty_output']} samples had empty outputs!", file=sys.stderr)
            print(f"[gen] This suggests the model is not generating text. Possible causes:", file=sys.stderr)
            print(f"[gen]   1. Model needs chat template formatting (try adding chat template)", file=sys.stderr)
            print(f"[gen]   2. Model is not compatible with vLLM", file=sys.stderr)
            print(f"[gen]   3. max_tokens_gen is too low or model is hitting stop sequences", file=sys.stderr)
            print(f"[gen]   4. Model checkpoint is corrupted or not loaded correctly", file=sys.stderr)
        
        raise RuntimeError("No valid samples survived gating. Check generator format/prompt.")

    key = args.dedup_by
    keep_k = args.keep_per_fingerprint
    groups: Dict[str, List[Sample]] = {}
    for s in candidates:
        fp = s.meta["fingerprints"][key]
        groups.setdefault(fp, []).append(s)

    deduped: List[Sample] = []
    for fp, lst in groups.items():
        lst.sort(key=lambda x: x.meta.get("quality_score", 0.0), reverse=True)
        deduped.extend(lst[:keep_k])

    print(f"[dedup] {len(candidates)} -> {len(deduped)} (by {key}, keep {keep_k}/fp)", file=sys.stderr)

    domain_counts = {}
    for s in deduped:
        d = s.meta.get("gen_spec", {}).get("domain", "unknown")
        domain_counts[d] = domain_counts.get(d, 0) + 1
    sorted_domains = sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)
    print(f"[dedup] domain distribution:", file=sys.stderr)
    for domain, count in sorted_domains:
        print(f"  - {domain}: {count} ({100*count/len(deduped):.1f}%)", file=sys.stderr)

    os.makedirs(os.path.dirname(args.out_intermediate_json) or ".", exist_ok=True)
    output_data = [sample_to_dict(s) for s in deduped]
    with open(args.out_intermediate_json, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"[out] wrote {len(deduped)} samples to {args.out_intermediate_json}", file=sys.stderr)

    

    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = elapsed % 60
    if hours > 0:
        print(f"[time] total time: {hours}h {minutes}m {seconds:.1f}s", file=sys.stderr)
    elif minutes > 0:
        print(f"[time] total time: {minutes}m {seconds:.1f}s", file=sys.stderr)
    else:
        print(f"[time] total time: {seconds:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
