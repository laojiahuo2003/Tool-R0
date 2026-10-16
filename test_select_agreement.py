#!/usr/bin/env python3
""" 集成测试：跑一遍 step2_select_curriculum.py 的 agreement 分档路径（零 GPU）。

验的是 2026-10-16 那次改口径的三件事：
  1. drop_saturated 用全等口径：名=1.0 但全等<1.0 的**必须留下**（round3 误丢 296 条那种）。
  2. 难度分档用全等口径： judge_raw 写成 agreement_full=，且档位按全等算。
  3. 向后兼容：喂 round3 那种只有名口径的老数据，不静默走新逻辑，
     要退回名口径并在 judge_raw 里留 fallback=names_only 痕。

用法：.venv/bin/python test_select_agreement.py
"""
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(HERE, ".venv", "bin", "python")
FAILS = []


def check(name, got, want):
    ok = got == want
    print(("  PASS  " if ok else "  FAIL  ") + name + f"  got={got!r} want={want!r}")
    if not ok:
        FAILS.append(name)


def mkrow(i, domain, agr, agr_full):
    return {
        "question": f"q{i} " + "x" * 20,
        "tools": [{"name": "t", "parameters": {"type": "object", "properties": {}}}],
        "calls": [{"name": "t", "arguments": {"i": i}}],
        "meta": {
            "gen_spec": {"domain": domain},
            "solver_agreement": agr,
            "solver_agreement_full": agr_full,
        },
    }


def run_select(rows, extra=()):
    d = tempfile.mkdtemp(prefix="seltest_")
    inp = os.path.join(d, "in.json")
    outp = os.path.join(d, "out.json")
    with open(inp, "w") as f:
        json.dump(rows, f)
    cmd = [PY, "step2_select_curriculum.py", "--in_json", inp, "--out_json", outp,
           "--n_final", "20", "--difficulty_source", "agreement",
           "--drop_agreement_ge", "1.0", "--mix_easy", "0.2",
           "--mix_medium", "0.5", "--mix_hard", "0.3", *extra]
    p = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
    if p.returncode != 0:
        print("  !!! select 跑挂了 rc=%d" % p.returncode)
        print(p.stdout[-2000:]); print(p.stderr[-2000:])
        sys.exit(1)
    return json.load(open(outp)), p.stdout + p.stderr


DOMAINS = ["finance", "health", "travel", "legal", "sports", "weather",
           "news", "retail", "system", "cloud", "iot", "education"]

print("=== 1. drop_saturated 用全等：名=1.0 全等=0.9 的必须活下来 ===")
rows = []
# 30 条名=1.0 全等=0.9（round3 那 296 条的类型）+ 10 条真的全等=1.0
for i in range(30):
    rows.append(mkrow(i, DOMAINS[i % len(DOMAINS)], 1.0, 0.9))
for i in range(30, 40):
    rows.append(mkrow(i, DOMAINS[i % len(DOMAINS)], 1.0, 1.0))
# 再补点各档的，好让 select 凑得出 20 条
for i in range(40, 70):
    rows.append(mkrow(i, DOMAINS[i % len(DOMAINS)], 0.6, 0.3))
out, log = run_select(rows)
# 注意这里查的是"有没有被 drop_saturated 剔"，**不是**"有没有进最终 20 条"——
# 第一版测试查错了对象：60 个候选抢 20 个名额加域配额，本来就不可能全中。
# drop_saturated 的账要看日志里的 [saturation] 行。
import re
m = re.search(r"\[saturation\] dropped (\d+) already-solved", log)
check("剔饱和数 = 10（只有全等=1.0 那 10 条）", int(m.group(1)) if m else None, 10)
m2 = re.search(r"(\d+) of them carry a full-match value", log)
check("剔完剩 60 条且都带全等值", int(m2.group(1)) if m2 else None, 60)
by_q = {r["question"].split()[0]: r for r in out}
check("真饱和（全等=1.0）的 10 条一条都没进",
      sum(1 for i in range(30, 40) if f"q{i}" in by_q), 0)

print()
print("=== 2. 分档按全等算：judge_raw 前缀是 agreement_full= ===")
rows = [mkrow(i, DOMAINS[i % len(DOMAINS)], 0.9, 0.1 * (i % 10)) for i in range(60)]
out, log = run_select(rows)
raves = [r.get("judge_raw", "") for r in out]
check("所有 judge_raw 都是 agreement_full= 开头",
      all(r.startswith("agreement_full=") for r in raves), True)
check("输出带了 solver_agreement_full 列",
      all("solver_agreement_full" in r for r in out), True)
# 档位要和全等值对得上：full<0.4 -> hard
bad = [r for r in out
       if (float(r["solver_agreement_full"]) < 0.4 and r["difficulty"] != "hard")
       or (0.4 <= float(r["solver_agreement_full"]) < 0.8 and r["difficulty"] != "medium")
       or (float(r["solver_agreement_full"]) >= 0.8 and r["difficulty"] != "easy")]
check("档位与全等值一致（没有错配）", bad, [])

print()
print("=== 3. 名口径和全等口径给出不同档位时，按全等 ===")
# 名 0.9（easy）但全等 0.2（hard）——round3 那种被算成 easy 的"名对参错"
rows = [mkrow(i, DOMAINS[i % len(DOMAINS)], 0.9, 0.2) for i in range(40)]
out, log = run_select(rows)
check("全 0.2 -> 全判 hard", {r["difficulty"] for r in out}, {"hard"})

print()
print("=== 4. 向后兼容：老数据没有 solver_agreement_full ===")
rows = []
for i in range(40):
    r = mkrow(i, DOMAINS[i % len(DOMAINS)], 0.1 * (i % 10), 0.1 * (i % 10))
    del r["meta"]["solver_agreement_full"]   # 模拟 round3 及以前的产物
    rows.append(r)
out, log = run_select(rows)
check("没炸，正常出 20 条", len(out), 20)
check("judge_raw 带 fallback=names_only 痕",
      all("fallback=names_only" in r.get("judge_raw", "") for r in out), True)
check("judge_used_fallback 为 False（不是当缺失处理）",
      all(r.get("judge_used_fallback") is False for r in out), True)
# 退回名口径后档位应和名值一致（名值 == 上面的全等值，所以可直接比）
bad = [r for r in out
       if (float(r["solver_agreement"]) < 0.4 and r["difficulty"] != "hard")
       or (0.4 <= float(r["solver_agreement"]) < 0.8 and r["difficulty"] != "medium")
       or (float(r["solver_agreement"]) >= 0.8 and r["difficulty"] != "easy")]
check("退回名口径后档位与名值一致", bad, [])

print()
print("=== 5. 全等=1.0 但名<1.0 不可能（全等是名的子集），构造一下确认不会反序 ===")
rows = [mkrow(i, DOMAINS[i % len(DOMAINS)], 0.3, 1.0) for i in range(20)]
out, log = run_select(rows)
# 这种数据本身矛盾，但不该让 select 崩；drop_saturated 按全等=1.0 全剔
check("矛盾数据不崩，输出 0 条（全被当饱和剔掉）", len(out), 0)

print()
if FAILS:
    print(f"!!! {len(FAILS)} 项不过: {FAILS}")
    sys.exit(1)
print("全部通过。")
