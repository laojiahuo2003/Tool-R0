
#!/usr/bin/env python3
"""
judge_select_3lvl.py

Ultra-robust LLM judge for 3-level difficulty: easy/medium/hard only.

Key guarantees:
- Judge is prompted to output ONLY one token: easy|medium|hard.
- Parser never expects JSON. It extracts the first occurrence of easy|medium|hard.
- If parsing fails, it falls back to "medium" (configurable).
- No KeyError crashes on malformed samples; malformed rows are skipped.
- Domain quotas computed FIRST (availability-aware) then selection within each domain.
- If not enough samples for a domain/difficulty, pads with any available level.
- Optional confidence blending with meta["p_mode"]/["mode_frac"] is optional and safe.

Input JSON: list of dicts, expected keys (best-effort):
  - question: str
  - tools: list[dict]
  - calls: list[dict]
  - meta: dict (optional), should include meta.gen_spec.domain

Output JSON: list of selected dicts including:
  - question, tools, calls
  - domain, difficulty
  - judge_raw (optional) and judge_used_fallback (bool)
  - combined_confidence (optional if enabled)
"""

import argparse
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


_PLACEHOLDER_PATTERNS = [
    r"\bthe generated user question here\b",
    r"\buser question must be from the specified domain\b",
    r"\bgenerate .* tool-calling task\b",
    r"\bcontrol spec\b",
    r"\brules to satisfy\b",
    r"\bavailable_tools\b",
    r"\btool_call_answer\b",
    r"\bprivate reasoning\b",
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






DIFF_RE = re.compile(r"\b(easy|medium|hard)\b", re.IGNORECASE)

def clamp01(x: float) -> float:
    try:
        x = float(x)
    except Exception:
        return 0.0
    if math.isnan(x) or math.isinf(x):
        return 0.0
    return max(0.0, min(1.0, x))

def safe_get_domain(meta: Dict[str, Any]) -> str:
    try:
        d = meta.get("gen_spec", {}).get("domain", "unknown")
        if isinstance(d, str) and d.strip():
            return d.strip()
    except Exception:
        pass
    return "unknown"

def safe_get_mode_frac(meta: Dict[str, Any]) -> Optional[float]:
    try:
        for k in ("p_mode", "mode_frac", "mode_fraction"):
            if k in meta:
                return clamp01(meta.get(k))
    except Exception:
        pass
    return None

def parse_difficulty_token(text: str, default: str = "medium") -> Tuple[str, bool]:
    """
    Extract first occurrence of easy|medium|hard from model output.
    Returns: (difficulty, used_fallback)
    """
    if not text or not isinstance(text, str):
        return default, True
    m = DIFF_RE.search(text)
    if not m:
        return default, True
    return m.group(1).lower(), False



@dataclass
class Item:
    question: str
    tools: List[Dict[str, Any]]
    calls: List[Dict[str, Any]]
    meta: Dict[str, Any]

    domain: str = "unknown"
    difficulty: str = "medium"
    judge_raw: str = ""
    judge_used_fallback: bool = False

    combined_confidence: Optional[float] = None


def load_items(path: str) -> List[Item]:
    """
    Ultra-safe loader: never crashes on malformed rows, skips them.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        return []

    items: List[Item] = []
    for d in data:
        try:
            if not isinstance(d, dict):
                continue
            q = d.get("question", None)
            tools = d.get("tools", None)
            calls = d.get("calls", None)
            meta = d.get("meta", {}) if isinstance(d.get("meta", {}), dict) else {}

            if is_placeholder_question(q):
                continue

            if "gen_spec" in d and isinstance(d["gen_spec"], dict):
                meta.setdefault("gen_spec", d["gen_spec"])

            if not isinstance(q, str) or not isinstance(tools, list) or not isinstance(calls, list):
                continue

            it = Item(question=q, tools=tools, calls=calls, meta=meta)
            it.domain = safe_get_domain(meta)
            items.append(it)
        except Exception:
            continue

    return items



JUDGE_SYSTEM = (
    "You are a strict difficulty judge for tool-calling tasks.\n"
    "You MUST output ONLY one word: easy, medium, or hard.\n"
    "No punctuation. No extra words. No explanations.\n"
)

JUDGE_USER_TMPL = """User request:
{question}

Available tools (JSON):
{tools_json}

Gold tool call (JSON):
{calls_json}

Decide the difficulty for a solver to produce the correct tool call(s).
Output ONLY one word: easy, medium, or hard.
"""

def build_judge_prompts(items: List[Item], tokenizer: AutoTokenizer) -> List[str]:
    prompts: List[str] = []
    for it in items:
        msg_user = JUDGE_USER_TMPL.format(
            question=it.question,
            tools_json=json.dumps(it.tools, ensure_ascii=False),
            calls_json=json.dumps(it.calls, ensure_ascii=False),
        )
        messages = [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": msg_user},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)
    return prompts


def run_judge(
    llm: LLM,
    tokenizer: AutoTokenizer,
    items: List[Item],
    batch_size: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    default_diff: str = "medium",
) -> None:
    """
    Runs judge and fills item.difficulty.
    Never crashes due to output mismatch: parser always returns a valid label.
    """
    if not items:
        return

    prompts = build_judge_prompts(items, tokenizer)
    sp = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        n=1,
        stop=["\n", "</s>", "assistant:", "Assistant:"],
        include_stop_str_in_output=False,
    )

    for start in range(0, len(prompts), batch_size):
        end = min(start + batch_size, len(prompts))
        chunk = prompts[start:end]

        try:
            req_outs = llm.generate(chunk, sp)
        except Exception:
            for i in range(start, end):
                items[i].difficulty = default_diff
                items[i].judge_used_fallback = True
                items[i].judge_raw = ""
            continue

        for j, req in enumerate(req_outs):
            idx = start + j
            raw = ""
            try:
                raw = req.outputs[0].text if req.outputs else ""
            except Exception:
                raw = ""

            diff, used_fb = parse_difficulty_token(raw, default=default_diff)
            items[idx].difficulty = diff
            items[idx].judge_used_fallback = used_fb
            items[idx].judge_raw = raw[:200] 



def normalize_weights(dw: Dict[str, float]) -> Dict[str, float]:
    s = sum(float(v) for v in dw.values()) if dw else 0.0
    if s <= 0:
        return {}
    return {k: float(v) / s for k, v in dw.items()}

def compute_domain_quotas(
    items: List[Item],
    n_final: int,
    domain_weights: Dict[str, float],
    seed: int,
) -> Dict[str, int]:
    """
    First compute desired quotas by weights, but cap by availability.
    Redistribute leftover to domains with remaining capacity.
    """
    rng = random.Random(seed)

    by_domain = defaultdict(list)
    for it in items:
        by_domain[it.domain].append(it)

    dw = normalize_weights(domain_weights)
    if not dw:
        doms = list(by_domain.keys())
        if not doms:
            return {}
        base = n_final // len(doms)
        quotas = {d: min(base, len(by_domain[d])) for d in doms}
        allocated = sum(quotas.values())
        rem = n_final - allocated
        doms_sorted = sorted(doms, key=lambda d: len(by_domain[d]) - quotas[d], reverse=True)
        for d in doms_sorted:
            if rem <= 0:
                break
            headroom = len(by_domain[d]) - quotas[d]
            take = min(headroom, rem)
            quotas[d] += take
            rem -= take
        return {d: q for d, q in quotas.items() if q > 0}

    observed = set(by_domain.keys())
    dw_obs = {d: w for d, w in dw.items() if d in observed}
    for d in observed:
        if d not in dw_obs:
            dw_obs[d] = 0.0

    dw_obs = normalize_weights(dw_obs) if sum(dw_obs.values()) > 0 else {d: 1.0 / len(observed) for d in observed}

    quotas: Dict[str, int] = {}
    allocated = 0
    doms_sorted = sorted(dw_obs.keys(), key=lambda d: dw_obs[d], reverse=True)

    for d in doms_sorted:
        if allocated >= n_final:
            quotas[d] = 0
            continue
        want = int(round(n_final * dw_obs[d]))
        cap = len(by_domain[d])
        q = max(0, min(want, cap, n_final - allocated))
        quotas[d] = q
        allocated += q

    rem = n_final - allocated
    if rem > 0:
        headroom = [(d, len(by_domain[d]) - quotas.get(d, 0)) for d in doms_sorted]
        headroom = [(d, h) for d, h in headroom if h > 0]
        headroom.sort(key=lambda x: (dw_obs.get(x[0], 0.0), x[1]), reverse=True)

        i = 0
        while rem > 0 and headroom:
            d, h = headroom[i % len(headroom)]
            if quotas[d] < len(by_domain[d]):
                quotas[d] += 1
                rem -= 1
            i += 1
            if i > 10 * (len(headroom) + 1) and all(quotas[x[0]] >= len(by_domain[x[0]]) for x in headroom):
                break

    return {d: q for d, q in quotas.items() if q > 0}



def normalize_mix(mix: Dict[str, float]) -> Dict[str, float]:
    m = {k: float(mix.get(k, 0.0)) for k in ("easy", "medium", "hard")}
    s = sum(m.values())
    if s <= 0:
        return {"easy": 0.2, "medium": 0.5, "hard": 0.3}
    return {k: v / s for k, v in m.items()}

def difficulty_targets(n: int, mix: Dict[str, float]) -> Dict[str, int]:
    mix = normalize_mix(mix)
    e = int(round(n * mix["easy"]))
    m = int(round(n * mix["medium"]))
    e = max(0, min(e, n))
    m = max(0, min(m, n - e))
    h = n - e - m
    return {"easy": e, "medium": m, "hard": h}



def compute_combined_confidence(
    it: Item,
    enable: bool,
    alpha: float,
) -> Optional[float]:
    """
    If enabled, combine:
      judge_conf  (we don't have one in token-only judge, so we treat as 1.0 if not fallback else 0.5)
      mode_frac   (if present in meta)
    combined = alpha * judge_conf + (1-alpha) * mode_frac
    """
    if not enable:
        return None

    judge_conf = 0.5 if it.judge_used_fallback else 1.0
    mode_frac = safe_get_mode_frac(it.meta)
    if mode_frac is None:
        return clamp01(judge_conf)

    a = clamp01(alpha)
    return clamp01(a * judge_conf + (1.0 - a) * mode_frac)



def select_domain_then_difficulty(
    items: List[Item],
    n_final: int,
    domain_weights: Dict[str, float],
    mix: Dict[str, float],
    seed: int,
    prefer_high_conf: bool,
    enable_conf_blend: bool,
    conf_alpha: float,
) -> List[Item]:
    rng = random.Random(seed)

    if n_final <= 0:
        return []
    if not items:
        return []

    for it in items:
        it.combined_confidence = compute_combined_confidence(it, enable_conf_blend, conf_alpha)

    def score(it: Item) -> float:
        if enable_conf_blend and it.combined_confidence is not None:
            return float(it.combined_confidence)
        return 0.0

    by_domain = defaultdict(list)
    for it in items:
        by_domain[it.domain].append(it)

    quotas = compute_domain_quotas(items, n_final, domain_weights, seed=seed)

    selected: List[Item] = []
    used_ids = set()

    for dom, q in quotas.items():
        if q <= 0:
            continue
        pool = list(by_domain.get(dom, []))
        if not pool:
            continue

        rng.shuffle(pool)

        by_diff = {"easy": [], "medium": [], "hard": []}
        for it in pool:
            by_diff.get(it.difficulty, by_diff["medium"]).append(it)

        if prefer_high_conf and enable_conf_blend:
            for d in by_diff:
                by_diff[d].sort(key=score, reverse=True)

        tgt = difficulty_targets(q, mix)
        dom_sel: List[Item] = []

        for d in ("easy", "medium", "hard"):
            need = tgt[d]
            if need <= 0:
                continue
            take = []
            for it in by_diff[d]:
                if id(it) in used_ids:
                    continue
                take.append(it)
                if len(take) >= need:
                    break
            dom_sel.extend(take)
            for it in take:
                used_ids.add(id(it))

        if len(dom_sel) < q:
            rest = [it for it in pool if id(it) not in used_ids]
            if prefer_high_conf and enable_conf_blend:
                rest.sort(key=score, reverse=True)
            take = rest[: (q - len(dom_sel))]
            dom_sel.extend(take)
            for it in take:
                used_ids.add(id(it))

        selected.extend(dom_sel[:q])

    if len(selected) < n_final:
        rest_all = [it for it in items if id(it) not in used_ids]
        rng.shuffle(rest_all)
        if prefer_high_conf and enable_conf_blend:
            rest_all.sort(key=score, reverse=True)
        need = n_final - len(selected)
        selected.extend(rest_all[:need])

    return selected[:n_final]



def build_solver_question(question: str, tools: List[Dict[str, Any]]) -> str:
    # Delegated rather than restated, and dumped without indent to match
    # step2_genverify.build_solver_prompt exactly. This local copy used indent=2
    # while verify probed the solver with the compact dump, so step3 trained the
    # solver on a prompt that differed from the one it was later evaluated with.
    from step2_genverify import SOLVER_USER_TMPL
    return SOLVER_USER_TMPL.format(
        question=question,
        tools_json=json.dumps(tools, ensure_ascii=False),
    )

def write_output(path: str, selected: List[Item], enable_conf: bool, conf_alpha: float) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    out = []
    for it in selected:
        rec = {
            "question": build_solver_question(it.question, it.tools),
            "answer": it.calls,
            "difficulty": it.difficulty,
            "domain": it.domain,
            "judge_used_fallback": bool(it.judge_used_fallback),
        }
        rec["judge_raw"] = it.judge_raw

        if enable_conf:
            rec["combined_confidence"] = it.combined_confidence
            rec["conf_alpha"] = conf_alpha
            rec["mode_frac"] = safe_get_mode_frac(it.meta)

        out.append(rec)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge_model", type=str, required=True)
    ap.add_argument("--in_json", type=str, required=True)
    ap.add_argument("--out_json", type=str, required=True)
    ap.add_argument("--n_final", type=int, default=1500)

    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_tokens_judge", type=int, default=8)
    ap.add_argument("--temp_judge", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)

    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    ap.add_argument("--max_model_len", type=int, default=4096)

    ap.add_argument("--mix_easy", type=float, default=0.25)
    ap.add_argument("--mix_medium", type=float, default=0.50)
    ap.add_argument("--mix_hard", type=float, default=0.25)

    ap.add_argument("--enable_conf_blend", action="store_true")
    ap.add_argument("--conf_alpha", type=float, default=0.7)
    ap.add_argument("--prefer_high_conf", action="store_true")

    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--default_diff", type=str, default="medium", choices=["easy", "medium", "hard"])

    args = ap.parse_args()
    t0 = time.time()

    items = load_items(args.in_json)
    if not items:
        print("[err] no valid items loaded (input malformed?)", file=sys.stderr)
        write_output(args.out_json, [], args.enable_conf_blend, args.conf_alpha)
        return

    print(f"[load] {len(items)} items", file=sys.stderr)
    print("[load] domains top20:", Counter(it.domain for it in items).most_common(20), file=sys.stderr)

    llm = LLM(
        model=args.judge_model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enforce_eager=True,
    )
    tok = AutoTokenizer.from_pretrained(args.judge_model, trust_remote_code=True)

    print("[judge] labeling easy/medium/hard...", file=sys.stderr)
    run_judge(
        llm=llm,
        tokenizer=tok,
        items=items,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens_judge,
        temperature=args.temp_judge,
        top_p=args.top_p,
        seed=args.seed,
        default_diff=args.default_diff,
    )

    print("[judge] diff:", Counter(it.difficulty for it in items), file=sys.stderr)
    print("[judge] fallback_used:", sum(1 for it in items if it.judge_used_fallback), "/", len(items), file=sys.stderr)

    mix = {"easy": args.mix_easy, "medium": args.mix_medium, "hard": args.mix_hard}

    selected = select_domain_then_difficulty(
        items=items,
        n_final=min(args.n_final, len(items)),
        domain_weights=DOMAIN_WEIGHTS,
        mix=mix,
        seed=args.seed,
        prefer_high_conf=args.prefer_high_conf,
        enable_conf_blend=args.enable_conf_blend,
        conf_alpha=args.conf_alpha,
    )

    print(f"[select] selected {len(selected)}", file=sys.stderr)
    print("[select] domains top20:", Counter(it.domain for it in selected).most_common(20), file=sys.stderr)
    print("[select] diff:", Counter(it.difficulty for it in selected), file=sys.stderr)

    write_output(args.out_json, selected, args.enable_conf_blend, args.conf_alpha)

    dt = time.time() - t0
    print(f"[done] wrote -> {args.out_json} in {dt:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
