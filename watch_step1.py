#!/usr/bin/env python3
"""Emit ONE line per completed step1 step, so the monitor produces 1 event
per step instead of ~7 (tqdm + 5 metrics).

Reads the block-buffered training log from stdin (already \r-translated). In
this log the tqdm progress bar and the full metric dict share one line, e.g.

  54%|...| 54/100 [4:23:37<1:55:56, 151.23s/it]{'loss': 0.0, 'grad_norm': ...}
"""
import re
import sys

PROG = re.compile(r"(\d+)/\d+ \[[0-9:]+<[0-9:]+, ([0-9.]+)s/it\]")
KV = re.compile(r"'([^']+)': ([0-9.eE+-]+)")
FAIL = ("Traceback", "CUDA out of memory", "Killed", "ABORT",
        "did not become ready", "Max retries", "RuntimeError", "OutOfMemoryError")

WANT = ("grad_norm", "completions/max_length", "completions/clipped_ratio",
        "rewards/reward_format_accuracy/mean",
        "rewards/reward_validity_accuracy/mean", "num_tokens")

cur = {}


def flush():
    global cur
    if not cur:
        return
    g = lambda k: cur.get(k) or 0.0
    print(
        "step {step:>3} | {pace:>6.0f}s/it | max_len {max_length:>5.0f} | "
        "clip {clipped_ratio:.3f} | grad {grad_norm:.3f} | fmt {fmt:.4f} | "
        "val {val:.4f}".format(
            step=cur.get("step", -1), pace=g("pace"),
            max_length=g("completions/max_length"),
            clipped_ratio=g("completions/clipped_ratio"),
            grad_norm=g("grad_norm"),
            fmt=g("rewards/reward_format_accuracy/mean"),
            val=g("rewards/reward_validity_accuracy/mean")),
        flush=True)
    cur = {}


for line in sys.stdin:
    if any(f in line for f in FAIL):
        print("!!! " + line.strip()[:200], flush=True)
        continue
    m = PROG.search(line)
    if m:
        # Each step emits a bare tqdm line first, then a tqdm+metrics line.
        # Only carry forward a step that already had its full metric block;
        # otherwise the bare line would trigger a bogus incomplete flush.
        if cur and all(k in cur for k in WANT):
            flush()
        else:
            cur = {}
        cur = {"step": int(m.group(1)), "pace": float(m.group(2))}
        for k, v in KV.findall(line):
            if k in WANT:
                cur[k] = float(v)
        if all(k in cur for k in WANT):
            flush()
    elif cur and "'grad_norm'" in line:
        # metric dict split across lines: keep filling the current step
        for k, v in KV.findall(line):
            if k in WANT:
                cur[k] = float(v)
        if all(k in cur for k in WANT):
            flush()
