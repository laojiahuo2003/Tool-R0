import ast
import json
import re
import os
from typing import Any, Dict, List, Optional, Tuple

import wandb

LAMBDA_NAME = 0.2
LAMBDA_PARAM_NAMES = 0.3
LAMBDA_PARAM_VALUES = 0.5

EXTRA_CALL_PENALTY_ALPHA = 0.25

REQUIRE_THINK_TAG = True

TAG_PATTERNS = {
    "redacted_reasoning": re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE),
    "tool_call_answer": re.compile(r"<tool_call_answer>(.*?)</tool_call_answer>", re.DOTALL | re.IGNORECASE),
}

def is_main_process() -> bool:
    """Return True only on the main process in distributed setups.

    Uses the `RANK` environment variable convention adopted by torchrun,
    accelerate, and deepspeed to detect whether the current process is
    the global rank 0 worker.
    """
    return int(os.environ.get("RANK", "0")) == 0


def _sep(title: str, width: int = 88) -> None:
    """Print a visual separator banner with a centered title."""
    pad = max(0, width - len(title) - 2)
    print("\n" + "=" * width)
    print(f"= {title}" + " " * pad + "=")
    print("=" * width)


def _block(title: str, content: str) -> None:
    """Print a titled text block, used for step-by-step debugging."""
    print(f"\n--- {title} ---")
    print(content if content else "<EMPTY>")


def _to_jsonable(x):
    """Recursively convert common Python types into JSON-serializable forms.

    Normalizes mappings, sequences, sets, NumPy scalars, byte strings,
    and ellipses so that diagnostic structures can be safely passed to
    `json.dumps` without serialization errors.
    """
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, tuple):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, set):
        try:
            return sorted([_to_jsonable(v) for v in x])
        except Exception:
            return [_to_jsonable(v) for v in x]

    if x is Ellipsis:
        return "<ELLIPSIS>"
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="replace")
    try:
        import numpy as np
        if isinstance(x, (np.integer, np.floating, np.bool_)):
            return x.item()
    except Exception:
        pass

    return x


def _json_block(title: str, obj: Any) -> None:
    """Pretty-print a Python object as JSON for debugging."""
    print(f"\n--- {title} ---")
    if obj is None:
        print("<NONE>")
    else:
        print(json.dumps(_to_jsonable(obj), indent=2, ensure_ascii=False))


def _short(s: Optional[str], n: int = 5000) -> str:
    """Return a human-readable, possibly truncated preview of a string."""
    if s is None:
        return "<NONE>"
    s = s.strip()
    if len(s) <= n:
        return s
    truncated = s[:n]
    last_newline = truncated.rfind("\n")
    if last_newline > n * 0.8:
        return truncated[:last_newline] + "\n... [TRUNCATED]"
    return truncated + "\n... [TRUNCATED]"


def _last_tag(text: str, tag: str) -> Optional[str]:
    """Return the last occurrence of a tagged block from the completion text."""
    m = TAG_PATTERNS[tag].findall(text or "")
    if not m:
        return None
    return m[-1].strip()

_MISSING = object()


def _contains_ellipsis(x: Any) -> bool:
    """True if x is, or contains, a bare Ellipsis (what a truncated `...` parses to)."""
    if x is Ellipsis:
        return True
    if isinstance(x, dict):
        return any(_contains_ellipsis(k) or _contains_ellipsis(v) for k, v in x.items())
    if isinstance(x, (list, tuple)):
        return any(_contains_ellipsis(i) for i in x)
    return False


def _loads_super_relaxed(s: str, *, guard_ellipsis: bool = True) -> Optional[Any]:
    """
    Parse either:
      - valid JSON
      - python repr strings produced by str(dict/list) (single quotes)
      - strings wrapped in ```json fences

    `guard_ellipsis` drops a result containing a bare Ellipsis, which is what a
    truncated `...`, `[...]` or `{"k": ...}` parses to via ast.literal_eval. The
    check is on the parsed value, not on the raw text: "..." also occurs inside
    legitimate strings ("wait..."), and rejecting on the raw text threw away
    valid tool calls -- on the model side a correct answer scored 0, and on the
    ground-truth side it made the gold answer unparseable, so a perfect
    prediction scored 0 there too. Truncation that does not parse into an
    Ellipsis is rejected by the parsers themselves. Pass False to skip the check
    (ground truth is trusted data, not model output).
    """
    if s is None:
        return None

    s2 = s.strip()
    s2 = re.sub(r"^```(?:json)?\s*|\s*```$", "", s2, flags=re.IGNORECASE).strip()

    parsed = _MISSING
    for attempt in (
        lambda: json.loads(s2),
        lambda: ast.literal_eval(s2),
        lambda: json.loads(s2.replace("None", "null").replace("True", "true").replace("False", "false")),
    ):
        try:
            parsed = attempt()
        except Exception:
            continue
        break
    if parsed is _MISSING:
        return None

    if guard_ellipsis and _contains_ellipsis(parsed):
        return None
    return parsed

def _json_loads_relaxed(s: str) -> Optional[Any]:
    """Backward-compatible alias for `_loads_super_relaxed`."""
    return _loads_super_relaxed(s)

def _canonical_json(x: Any) -> str:
    """Serialize a Python object into a canonical, comparable JSON string."""
    return json.dumps(_to_jsonable(x), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def extract_solver_fields(completion_text: str) -> Optional[Dict[str, str]]:
    """
    Required: <tool_call_answer>
    Optional: <think> (only required if REQUIRE_THINK_TAG=True)
    """
    tool_call = _last_tag(completion_text, "tool_call_answer")
    if tool_call is None or tool_call == "":
        return None

    reasoning = _last_tag(completion_text, "redacted_reasoning")
    if REQUIRE_THINK_TAG and (reasoning is None or reasoning == ""):
        return None

    out = {"tool_call_answer": tool_call.strip()}
    if reasoning is not None and reasoning != "":
        out["redacted_reasoning"] = reasoning.strip()
    return out

def parse_ground_truth_answer(answer_obj: Any) -> Optional[List[Dict[str, Any]]]:
    """Parse a ground-truth answer into a list of raw tool-call dicts."""
    if isinstance(answer_obj, list):
        return answer_obj
    if isinstance(answer_obj, dict):
        return [answer_obj]
    if not isinstance(answer_obj, str):
        return None

    parsed = _loads_super_relaxed(answer_obj, guard_ellipsis=False)
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        return [parsed]
    return None

def normalize_tool_call(obj: Any) -> Optional[Dict[str, Any]]:
    """
    Normalize to {"name": str, "arguments": dict}
    Supports list -> first element, OpenAI-ish function wrapper, etc.
    """
    if isinstance(obj, list):
        if len(obj) == 0:
            return None
        obj = obj[0]

    if not isinstance(obj, dict):
        return None

    if "function" in obj and isinstance(obj["function"], dict):
        fn = obj["function"]
        name = fn.get("name")
        args = fn.get("arguments")
        if isinstance(args, str):
            args = _loads_super_relaxed(args)
        if isinstance(name, str) and isinstance(args, dict):
            return {"name": name, "arguments": args}

    name = obj.get("name") or obj.get("tool_name")
    if not isinstance(name, str) or not name.strip():
        return None

    args = obj.get("arguments")
    if isinstance(args, str):
        args = _loads_super_relaxed(args)
    if isinstance(args, dict):
        return {"name": name, "arguments": args}

    flat = {k: v for k, v in obj.items() if k not in ("name", "tool_name")}
    return {"name": name, "arguments": flat}

def parse_solver_tool_calls(tool_call_text: str) -> Optional[List[Dict[str, Any]]]:
    obj = _loads_super_relaxed(tool_call_text)
    if obj is None:
        return None

    if isinstance(obj, list):
        out: List[Dict[str, Any]] = []
        for item in obj:
            norm = normalize_tool_call(item)
            if norm is not None:
                out.append(norm)
        return out if out else None

    norm = normalize_tool_call(obj)
    return [norm] if norm is not None else None

_NUM_RE = re.compile(r"^\s*-?\d+(?:\.\d+)?\s*$")

def _coerce_number(x: Any) -> Optional[float]:
    """Safely coerce small numeric-like values to floats for comparison."""
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        try:
            return float(x)
        except (OverflowError, ValueError):
            return None

    if isinstance(x, str) and _NUM_RE.match(x):
        s = x.strip()
        if len(s.lstrip("+-").replace(".", "")) > 15: 
            return None
        try:
            return float(s)
        except (OverflowError, ValueError):
            return None

    return None


def robust_value_match(v1: Any, v2: Any) -> bool:
    """Compare two values using a conservative, semantics-aware heuristic."""
    if v1 == v2:
        return True

    if isinstance(v1, (int, str)) and isinstance(v2, (int, str)):
        s1 = str(v1).strip()
        s2 = str(v2).strip()
        if _NUM_RE.match(s1) and _NUM_RE.match(s2):
            return s1.lstrip("+") == s2.lstrip("+")

    n1 = _coerce_number(v1)
    n2 = _coerce_number(v2)
    if n1 is not None and n2 is not None:
        return abs(n1 - n2) < 1e-9

    if isinstance(v1, str) and isinstance(v2, str):
        s1 = " ".join(v1.strip().split())
        s2 = " ".join(v2.strip().split())
        return s1 == s2

    return _canonical_json(v1) == _canonical_json(v2)


def f1_keys(pred_keys: set, gt_keys: set) -> float:
    """Compute the F1 score between two sets of argument keys."""
    if not pred_keys and not gt_keys:
        return 1.0
    if not pred_keys or not gt_keys:
        return 0.0
    inter = len(pred_keys & gt_keys)
    if inter == 0:
        return 0.0
    prec = inter / len(pred_keys)
    rec = inter / len(gt_keys)
    return (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

def score_tool_call(predicted: Dict[str, Any], ground_truth: Dict[str, Any]) -> Tuple[float, float, float]:
    """
    Returns scores in [0,1] for (name_score, key_score, value_score).
    - key_score: F1 over keys (dense)
    - value_score: fraction of matching values over intersection keys
    """
    pred_name = (predicted.get("name") or "").strip()
    gt_name = (ground_truth.get("name") or "").strip()
    name_score = 1.0 if pred_name == gt_name and pred_name != "" else 0.0

    pred_args = predicted.get("arguments", {})
    gt_args = ground_truth.get("arguments", {})
    if not isinstance(pred_args, dict):
        pred_args = {}
    if not isinstance(gt_args, dict):
        gt_args = {}

    pred_keys = set(pred_args.keys())
    gt_keys = set(gt_args.keys())

    key_score = f1_keys(pred_keys, gt_keys)

    inter = pred_keys & gt_keys
    if not inter:
        value_score = 1.0 if (not pred_keys and not gt_keys) else 0.0
    else:
        matches = 0
        for k in inter:
            if robust_value_match(pred_args.get(k), gt_args.get(k)):
                matches += 1
        value_score = matches / len(inter)

    return name_score, key_score, value_score


def compute_accuracy_score(
    predicted_calls: List[Dict[str, Any]],
    ground_truth_calls: List[Dict[str, Any]],
):
    """
    Returns:
      final_reward ∈ [0,1]
      diagnostics dict with sub-rewards for logging
    """
    diagnostics = {
        "mean_name_score": 0.0,
        "mean_key_score": 0.0,
        "mean_value_score": 0.0,
        "mean_pair_reward": 0.0,
        "num_pred_calls": len(predicted_calls),
        "num_gt_calls": len(ground_truth_calls),
        "extra_calls": 0,
        "extra_call_penalty": 1.0,
        "base_score": 0.0,
    }

    if not ground_truth_calls:
        final = 1.0 if not predicted_calls else 0.0
        diagnostics["base_score"] = final
        return final, diagnostics

    if not predicted_calls:
        return 0.0, diagnostics

    used_pred = set()
    total = 0.0

    name_scores = []
    key_scores = []
    value_scores = []
    pair_rewards = []

    for gt in ground_truth_calls:
        best_r = 0.0
        best_i = -1
        best_tuple = (0.0, 0.0, 0.0)

        for i, pred in enumerate(predicted_calls):
            if i in used_pred:
                continue

            n_s, k_s, v_s = score_tool_call(pred, gt)
            r = (LAMBDA_NAME * n_s) + (LAMBDA_PARAM_NAMES * k_s) + (LAMBDA_PARAM_VALUES * v_s)

            if r > best_r:
                best_r = r
                best_i = i
                best_tuple = (n_s, k_s, v_s)

        if best_i != -1:
            used_pred.add(best_i)
            total += best_r

            name_scores.append(best_tuple[0])
            key_scores.append(best_tuple[1])
            value_scores.append(best_tuple[2])
            pair_rewards.append(best_r)

    base = total / len(ground_truth_calls)

    extra = max(0, len(predicted_calls) - len(ground_truth_calls))
    penalty = 1.0
    if extra > 0:
        penalty = 1.0 / (1.0 + EXTRA_CALL_PENALTY_ALPHA * extra)

    final = base * penalty

    diagnostics["mean_name_score"] = float(sum(name_scores) / max(1, len(name_scores)))
    diagnostics["mean_key_score"] = float(sum(key_scores) / max(1, len(key_scores)))
    diagnostics["mean_value_score"] = float(sum(value_scores) / max(1, len(value_scores)))
    diagnostics["mean_pair_reward"] = float(sum(pair_rewards) / max(1, len(pair_rewards)))

    diagnostics["base_score"] = float(base)
    diagnostics["extra_calls"] = int(extra)
    diagnostics["extra_call_penalty"] = float(penalty)
    diagnostics["final_accuracy_reward"] = float(final)

    return float(max(0.0, min(1.0, final))), diagnostics

    
def format_reward_func(prompts, completions, **kwargs) -> List[float]:
    """
    Graded format reward:
      +0.3 if <tool_call_answer> tag exists
      +0.3 if it parses (super-relaxed)
      +0.4 if it normalizes into at least one tool call
    """
    rewards: List[float] = []
    for prompt, completion in zip(prompts, completions, strict=True):
        comp = completion[0]["content"]

        tool_call_text = _last_tag(comp, "tool_call_answer")
        if tool_call_text is None or tool_call_text.strip() == "":
            rewards.append(0.0)
            continue

        r = 0.3
        parsed = _loads_super_relaxed(tool_call_text)
        if parsed is not None:
            r += 0.3
        calls = parse_solver_tool_calls(tool_call_text)
        if calls is not None:
            r += 0.4

        rewards.append(float(max(0.0, min(1.0, r))))
    return rewards


def accuracy_reward_func(prompts, completions, answer, **kwargs) -> List[float]:
    """
    Dense accuracy reward:
    - parses solver tool calls
    - parses GT tool calls (super-relaxed)
    - soft match (name, key-F1, values on intersection)
    - penalizes extra calls beyond GT
    """
    rewards: List[float] = []

    if not isinstance(answer, list):
        answer = [answer] if answer is not None else []

    for idx, (prompt, completion) in enumerate(zip(prompts, completions, strict=True)):
        _sep("SOLVER STEP")

        user_prompt = prompt[-1]["content"] if len(prompt) > 0 else "<EMPTY>"
        _block("Question", _short(user_prompt, n=5000))

        comp = completion[0]["content"]
        fields = extract_solver_fields(comp)
        if fields is None:
            _block("Solver Output (RAW)", _short(comp, n=5000))
            _block("Error", "Missing <tool_call_answer> (or <think> if REQUIRE_THINK_TAG=True)")
            rewards.append(0.0)
            continue

        predicted_calls = parse_solver_tool_calls(fields["tool_call_answer"])
        if predicted_calls is None:
            _block("Solver <tool_call_answer> (RAW)", _short(fields["tool_call_answer"], n=5000))
            _block("Error", "Failed to parse tool_call_answer")
            rewards.append(0.0)
            continue

        if idx >= len(answer):
            _block("Error", "Ground truth answer not available for this index")
            rewards.append(0.0)
            continue

        answer_str = answer[idx]
        if not isinstance(answer_str, str):
            answer_str = str(answer_str)

        gt_raw = parse_ground_truth_answer(answer_str)
        if gt_raw is None:
            _block("Ground Truth Answer (RAW)", _short(answer_str, n=5000))
            _block("Error", "Failed to parse ground truth answer (super-relaxed)")
            rewards.append(0.0)
            continue

        ground_truth_calls: List[Dict[str, Any]] = []
        for x in gt_raw:
            norm = normalize_tool_call(x)
            if norm is not None:
                ground_truth_calls.append(norm)

        if not ground_truth_calls:
            _json_block("Ground Truth Answer (RAW PARSED)", gt_raw)
            _block("Error", "GT parsed but could not be normalized to {name, arguments}")
            rewards.append(0.0)
            continue

        _json_block("Predicted Answer", predicted_calls)
        _json_block("Ground Truth Answer", ground_truth_calls)

        score, diag = compute_accuracy_score(predicted_calls, ground_truth_calls)

        _sep("SCORE")
        print(f"ACCURACY REWARD: {score:.3f}\n")

        step = kwargs.get("step", None)
        if is_main_process() and wandb.run is not None:

            wandb.log({
                "solver/name_score": diag["mean_name_score"],
                "solver/key_score": diag["mean_key_score"],
                "solver/value_score": diag["mean_value_score"],
                "solver/pair_reward": diag["mean_pair_reward"],

                "solver/base_score": diag["base_score"],
                "solver/extra_calls": diag["extra_calls"],
                "solver/extra_call_penalty": diag["extra_call_penalty"],
                "solver/final_accuracy_reward": diag["final_accuracy_reward"],

                "solver/num_pred_calls": diag["num_pred_calls"],
                "solver/num_gt_calls": diag["num_gt_calls"],
            }, step=step, commit=False)

        rewards.append(float(max(0.0, min(1.0, score))))

    return rewards
