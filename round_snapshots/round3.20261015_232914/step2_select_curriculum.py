
#!/usr/bin/env python3
"""
judge_select_3lvl.py

3-level difficulty curriculum selection: easy/medium/hard only.

难度从哪来有两条路，由 --difficulty_source 选（默认 agreement）：
  - agreement：用 step2_genverify 实测写进 meta 的 solver_agreement 分档，不加载任何模型。
  - judge：加载 --judge_model 让 LLM 猜 easy/medium/hard。2026-10-15 实测这条路是坏的
    （584/584 全判 hard，配比等于没写），保留只为做对照。

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
    and (for --difficulty_source agreement) meta.solver_agreement

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

    # step2_genverify.py 写进 meta 的 solver_agreement：solver 在 k_verify 次抽样里
    # 命中 gold 的比例。1.0 = 10 次全对。GRPO 是按组算 advantage 的，组内没有奖励方差
    # 就没有梯度，所以全对的样本对 step3 是纯浪费（还会占掉 n_final 的名额）。
    solver_agreement: Optional[float] = None

    combined_confidence: Optional[float] = None


def drop_saturated(items: List[Item], threshold: float) -> Tuple[List[Item], int]:
    """剔掉 solver 已经全会（agreement 达到 threshold）的样本。

    为什么需要：GRPO 的 advantage 是按"同一个 prompt 采 G 个 completion"归一化出来的，
    组内奖励没有方差 → advantage 全 0 → 这一步对这个 prompt 的梯度是 0。
    agreement=1.0 就是 k_verify 次全对，训了也白训，还占掉 n_final 的名额。

    2026-10-14 实测：换成 step3 训过的 solver 之后，verify 留存从 563 涨到 811，
    其中 227 条（28%）agreement=1.0，而这个文件原来根本不读 solver_agreement
    （grep 过，全文没有这个词），所以 select 完全没有机制避开它们。

    threshold 默认 1.0，只剔"彻底全会"的。agreement=0.9 那种（9/10 对）组内还有方差，
    G=4 时约 34% 的组能出梯度，不算死，不用剔。想更激进就把它调低。
    threshold > 1.0 等于关掉这个过滤。
    """
    if threshold > 1.0:
        return items, 0
    kept, dropped = [], 0
    for it in items:
        if it.solver_agreement is not None and it.solver_agreement >= threshold:
            dropped += 1
            continue
        kept.append(it)
    return kept, dropped


def assign_difficulty_from_agreement(
    items: List[Item],
    hard_lt: float,
    easy_ge: float,
    default_diff: str,
) -> Tuple[int, int]:
    """用 verify 关实测的 solver_agreement 直接分档，不调 LLM judge。

    为什么换掉 judge：2026-10-15 实测，judge 不管喂 base 还是 step3 训过的 solver，
    输出都塌成单一档 —— base 是 583 hard + 1 easy，换成 ckpt50 之后 **584/584 全 hard**，
    `fallback_used: 0/584`、`judge_raw` 逐条就是字面的 `'hard'`。也就是说
    `--mix_easy .20 --mix_medium .50 --mix_hard .30` 实际执行成 **100% hard**，
    难度配比是死的，select 里的分档逻辑等于没写。LLM judge 想猜的"这题对 solver 有多难"，
    我们每一轮都直接测过了（k_verify 次抽样命中 gold 的比例），没必要再猜一遍。
    顺带省掉一次 judge 模型加载和推理，这一步从要 GPU 变成零 GPU 成本。

    agreement 被 k_verify 离散化（k=10 时只能取 0, 0.1, ..., 1.0），所以阈值取 0.35 还是
    0.40 没有区别，只有跨过 0.1 的整数倍时才变。

    默认阈值的依据（2026-10-15，ckpt50 阅卷的 811 条剔掉 227 条饱和后剩 584 条）：
        agreement < 0.4      83 条  14.2%   -> hard
        0.4 <= a < 0.8      262 条  44.9%   -> medium
        a >= 0.8            239 条  40.9%   -> easy
    对照 --mix 想要的 30/50/20：**全局每档都够，hard 甚至富余 53 条**。但 select 是先分域
    再分档，每域平均只有 18 条、配额 3，其中 31/32 个域能凑出 1 条 hard，geolocation 一条
    都没有，所以 padding 仍会把一些 medium/easy 补进来 —— 别指望最终配比精确等于 30/50/20。

    缺 agreement 的样本（比如输入不是 step2_genverify 的产物）按 default_diff 处理，并记
    `judge_used_fallback=True`，和 judge 解析失败时的语义保持一致，调用方照旧能数出来。

    返回 (有 agreement 的条数, 缺 agreement 的条数)。
    """
    n_have, n_missing = 0, 0
    for it in items:
        a = it.solver_agreement
        if a is None:
            it.difficulty = default_diff
            it.judge_used_fallback = True
            it.judge_raw = "agreement=missing"
            n_missing += 1
            continue
        n_have += 1
        if a < hard_lt:
            it.difficulty = "hard"
        elif a >= easy_ge:
            it.difficulty = "easy"
        else:
            it.difficulty = "medium"
        # judge_raw 留着是给事后查的：这条教材当初判成这个档，依据是哪个命中率。
        it.judge_raw = "agreement=%.2f" % a
        it.judge_used_fallback = False
    return n_have, n_missing


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
            # verify 关的 agreement 原样带过来；缺这个键（比如输入不是 genverify 的产物）
            # 或不是数字时保持 None，后面按"不过滤"处理，不能当成 0 或 1 猜。
            raw_agr = meta.get("solver_agreement", None)
            if isinstance(raw_agr, (int, float)) and not isinstance(raw_agr, bool):
                it.solver_agreement = float(raw_agr)
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


_LEVELS = ("easy", "medium", "hard")


def split_quota_by_remaining(q: int, remaining: Dict[str, int]) -> Dict[str, int]:
    """把本域的 q 个名额按"全局还差多少"切给各难度档，最大余数法补尾。

    为什么不能直接 difficulty_targets(q, mix)：那个函数拿**本域的小配额**去乘全局比例。
    n_final=100、32 域等权时每域只有 3~4 个名额，而 mix=20/50/30 在 q=3 上 round 出来是
    easy 1 / medium 2 / **hard 0** —— 只有 q=4 的域才轮得到 1 个 hard 名额。
    2026-10-15 实测：换成 agreement 分档之后真正的配比问题才浮出水面，最终 100 条里
    hard 只有 4 条（4%），而想要 30 条。judge 那条路因为 584/584 全判 hard，桶里只有一档
    有货，这个 rounding 一直被掩盖着。

    所以目标按全局算一次（difficulty_targets(n_final, mix)），再按剩余需求往各域分摊。
    remaining 是调用方维护的"全局还差几条"，取完就减，这样不会出现前几个域把某档吃光、
    后面域的配额明明还有却只能拿别的档。

    remaining 全为 0 时（目标已达成，或某档池子已空）退化成均摊，别把名额浪费掉。
    """
    out = {d: 0 for d in _LEVELS}
    if q <= 0:
        return out

    need = {d: max(0, int(remaining.get(d, 0))) for d in _LEVELS}
    total = sum(need.values())
    if total <= 0:
        base, extra = divmod(q, len(_LEVELS))
        for i, d in enumerate(_LEVELS):
            out[d] = base + (1 if i < extra else 0)
        return out

    raw = {d: q * need[d] / total for d in _LEVELS}
    out = {d: int(math.floor(raw[d])) for d in _LEVELS}
    left = q - sum(out.values())
    # 小数部分大的优先，并列时缺口大的优先，保证确定性
    order = sorted(_LEVELS, key=lambda d: (-(raw[d] - out[d]), -need[d]))
    i = 0
    while left > 0 and i < 10 * len(_LEVELS):
        out[order[i % len(order)]] += 1
        left -= 1
        i += 1
    return out



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

    # 难度目标按全局算一次，再分摊到各域。别在域内用 difficulty_targets(q, mix)：
    # 每域只有 3~4 个名额，round 之后 hard 档会是 0，见 split_quota_by_remaining 的注释。
    global_targets = difficulty_targets(n_final, mix)
    remaining = dict(global_targets)

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

        tgt = split_quota_by_remaining(q, remaining)
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
            # 只减真正取到的条数：这个域该档可能本来就没有那么多，
            # 缺口要留给后面的域去补，不能按 need 减否则会凭空消失。
            remaining[d] = max(0, remaining[d] - len(take))

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

        # 带出去是为了事后能查：这条教材当初 solver 的命中率是多少。
        # step3 那边如果看到某批样本 reward 一直零方差，先从这列找原因。
        if it.solver_agreement is not None:
            rec["solver_agreement"] = it.solver_agreement

        if enable_conf:
            rec["combined_confidence"] = it.combined_confidence
            rec["conf_alpha"] = conf_alpha
            rec["mode_frac"] = safe_get_mode_frac(it.meta)

        out.append(rec)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser()
    # 2026-10-15 起 judge_model 不再是必填：默认走 agreement 分档，那条路不加载任何模型。
    # 只有 --difficulty_source judge 才需要它。
    ap.add_argument("--judge_model", type=str, default=None,
                    help="Judge model path. Only required when "
                         "--difficulty_source judge.")
    ap.add_argument("--in_json", type=str, required=True)
    ap.add_argument("--out_json", type=str, required=True)
    ap.add_argument("--n_final", type=int, default=1500)

    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_tokens_judge", type=int, default=8)
    ap.add_argument("--temp_judge", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)

    ap.add_argument("--difficulty_source", type=str, default="agreement",
                    choices=["judge", "agreement"],
                    help="agreement（默认）：按 verify 关实测的 solver_agreement 分档，"
                         "零 GPU 成本。judge：老行为，加载 --judge_model 让 LLM 猜 "
                         "easy/medium/hard —— 2026-10-15 实测它 584/584 全判 hard，"
                         "等于没有配比，留着只是为了做对照。")
    ap.add_argument("--agr_hard_lt", type=float, default=0.4,
                    help="--difficulty_source agreement 时，solver_agreement 小于这个值 "
                         "算 hard。k=10 时 agreement 是 0.1 的整数倍，阈值跨不过 0.1 就没区别。")
    ap.add_argument("--agr_easy_ge", type=float, default=0.8,
                    help="--difficulty_source agreement 时，solver_agreement 大于等于这个值 "
                         "算 easy，其余算 medium。")

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
    ap.add_argument("--drop_agreement_ge", type=float, default=1.0,
                    help="Drop samples whose solver_agreement >= this (default 1.0 = "
                         "only fully-solved ones, which give step3 zero GRPO gradient). "
                         "Pass >1.0 to disable.")

    args = ap.parse_args()
    t0 = time.time()

    # 参数互斥/越界在这里就挡住，别等到跑了一半才发现分档区间是空的或者根本没传模型。
    if args.difficulty_source == "judge" and not args.judge_model:
        ap.error("--difficulty_source judge 需要 --judge_model")
    if args.agr_easy_ge <= args.agr_hard_lt:
        ap.error(f"--agr_easy_ge ({args.agr_easy_ge}) 必须大于 "
                 f"--agr_hard_lt ({args.agr_hard_lt})，否则 medium 档是空的")

    items = load_items(args.in_json)
    if not items:
        print("[err] no valid items loaded (input malformed?)", file=sys.stderr)
        write_output(args.out_json, [], args.enable_conf_blend, args.conf_alpha)
        return

    print(f"[load] {len(items)} items", file=sys.stderr)
    print("[load] domains top20:", Counter(it.domain for it in items).most_common(20), file=sys.stderr)

    # 饱和过滤放在 judge 之前：被剔的样本本来就不该进训练集，省一次 judge 推理。
    # agreement=1.0 的样本对 step3 是零梯度，见 drop_saturated 的注释。
    items, n_saturated = drop_saturated(items, args.drop_agreement_ge)
    n_with_agr = sum(1 for it in items if it.solver_agreement is not None)
    print(f"[saturation] dropped {n_saturated} already-solved samples "
          f"(agreement >= {args.drop_agreement_ge}); {len(items)} left, "
          f"{n_with_agr} of them carry an agreement value", file=sys.stderr)
    if n_saturated and len(items) < args.n_final:
        print(f"[saturation] WARNING: only {len(items)} usable items left but "
              f"--n_final={args.n_final}; the output will be short. Either lower "
              f"--drop_agreement_ge or widen the candidate pool in step2.",
              file=sys.stderr)
    if not items:
        print("[err] no valid items left after saturation filter", file=sys.stderr)
        write_output(args.out_json, [], args.enable_conf_blend, args.conf_alpha)
        return

    if args.difficulty_source == "agreement":
        n_have, n_missing = assign_difficulty_from_agreement(
            items, args.agr_hard_lt, args.agr_easy_ge, args.default_diff)
        print(f"[difficulty] 按 solver_agreement 分档（不加载 judge，零 GPU 成本）："
              f"hard <{args.agr_hard_lt} / medium [{args.agr_hard_lt},{args.agr_easy_ge}) "
              f"/ easy >={args.agr_easy_ge}；{n_have} 条有命中率，"
              f"{n_missing} 条缺失按 {args.default_diff} 兜底", file=sys.stderr)
        if n_have == 0:
            # 全都没有 agreement，硬分档只会把所有人都塞进 default_diff 一个桶里，
            # 那和老行为（judge 塌成单一档）是同一个毛病，不如直接报错说清楚。
            print("[err] --difficulty_source agreement，但输入里没有一条带 "
                  "solver_agreement。确认 --in_json 是 step2_genverify 的产物"
                  "（verified json，不是 intermediate）；要走老路就加 "
                  "--difficulty_source judge --judge_model <path>。",
                  file=sys.stderr)
            write_output(args.out_json, [], args.enable_conf_blend, args.conf_alpha)
            return
    else:
        if not args.judge_model:  # 兜底，正常已被 ap.error 挡住
            print("[err] --difficulty_source judge 但没给 --judge_model", file=sys.stderr)
            write_output(args.out_json, [], args.enable_conf_blend, args.conf_alpha)
            return

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
        print("[judge] fallback_used:",
              sum(1 for it in items if it.judge_used_fallback), "/", len(items),
              file=sys.stderr)

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
    got = Counter(it.difficulty for it in selected)
    n = len(selected) or 1
    want = normalize_mix(mix)
    # 把"想要的比例"和"实际拿到的比例"并排打出来。分域之后每域只剩 ~18 条，hard 档在个别域
    # （geolocation）是空的，padding 会拿 medium/easy 补，所以两者不会精确相等。
    # 2026-10-15 在 584 条非饱和池上量过：全局 hard 富余 53 条，但分域后 31/32 个域才凑得出
    # 1 条 hard。如果哪次看到实际配比又塌成单一档，先看这行和上面 [difficulty] 那行。
    print("[select] diff: %s" % {k: got.get(k, 0) for k in ("easy", "medium", "hard")},
          file=sys.stderr)
    print("[select] mix : 想要 easy %.0f%% / medium %.0f%% / hard %.0f%%"
          "   实际 easy %.0f%% / medium %.0f%% / hard %.0f%%" % (
              100*want["easy"], 100*want["medium"], 100*want["hard"],
              100*got.get("easy", 0)/n, 100*got.get("medium", 0)/n, 100*got.get("hard", 0)/n),
          file=sys.stderr)

    write_output(args.out_json, selected, args.enable_conf_blend, args.conf_alpha)

    dt = time.time() - t0
    print(f"[done] wrote -> {args.out_json} in {dt:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
