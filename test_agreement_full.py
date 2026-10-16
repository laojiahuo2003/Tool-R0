#!/usr/bin/env python3
""" 不依赖 GPU 的单测：验 step2_genverify 新增的"全等口径"计数语义。

为什么值得单测：这次改的是 agreement 的定义，而它同时喂给
tau_verify 门、drop_agreement_ge、难度分档三个地方。计数错一格不会报错，
只会让下一轮的漏斗 silently 变形——round3 已经因为"只看名"吃了 296 条的亏，
不能再靠"跑一遍看看"来验。

跑法：.venv/bin/python test_agreement_full.py
"""
import json
import os
import sys
from collections import Counter
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import step2_genverify as G

FAILS = []


def check(name, got, want):
    ok = got == want
    print(("  PASS  " if ok else "  FAIL  ") + name + f"  got={got!r} want={want!r}")
    if not ok:
        FAILS.append(name)


class FakeOut:
    def __init__(self, text):
        self.text = text


class FakeReq:
    def __init__(self, texts):
        self.outputs = [FakeOut(t) for t in texts]


class FakeLLM:
    """按 prompts 的顺序，依次吐出预设的 k 个样本文本。"""

    def __init__(self, script):
        self.script = script  # list[list[str]]，每个 prompt 一组
        self.calls = 0

    def generate(self, prompts, sp):
        out = []
        for p in prompts:
            out.append(FakeReq(self.script[self.calls]))
            self.calls += 1
        return out


def wrap(calls):
    """把一次预测包成 solver 的输出格式。"""
    return f"<think>thinking</think>\n<tool_call_answer>{json.dumps(calls, ensure_ascii=False)}</tool_call_answer>"


def run(prompts, golds, script, k, chunk=2):
    llm = FakeLLM(script)
    return G.solver_agreement_batch(
        llm_solver=llm, prompts=prompts, gold_calls_list=golds,
        k=k, temperature=0.7, max_tokens=64, chunk_size=chunk,
    )


print("=== 1. 名对参错：名口径该算命中，全等口径不算 ===")
gold = [[{"name": "get_user_info", "arguments": {"userId": "johnsmith"}}]]
prompts = ["q"]
# k=4：两个完全对，两个名对参错
script = [[
    wrap([{"name": "get_user_info", "arguments": {"userId": "johnsmith"}}]),
    wrap([{"name": "get_user_info", "arguments": {"userId": "john_smith"}}]),
    wrap([{"name": "get_user_info", "arguments": {"userId": "johnsmith"}}]),
    wrap([{"name": "get_user_info", "arguments": {"userId": "JOHN"}}]),
]]
r, rf, rsn, pred, raw = run(prompts, gold, script, k=4)
check("名口径 = 4/4 = 1.0", r[0], 1.0)
check("全等口径 = 2/4 = 0.5", rf[0], 0.5)
check("reason 记 args_mismatch", rsn[0], "args_mismatch")

print()
print("=== 2. round3 那个实案：名对、值差一个下下划线，ag 必须仍是 1.0 且全等 0 ===")
gold = [[{"name": "get_user_info", "arguments": {"userId": "johnsmith"}}]]
script = [[wrap([{"name": "get_user_info", "arguments": {"userId": "john_smith"}}])] * 10]
r, rf, rsn, pred, raw = run(["q"], gold, script, k=10)
check("名口径 1.0（历史行为，没变）", r[0], 1.0)
check("全等口径 0.0（新口径抓到它）", rf[0], 0.0)
# 这一条的期望值改过一次。原来是 "ok"，理由是"ok==k 就不记 reason"——
# 那正是 round3 的毛病：全点名对但参数全错，报告上看不出来，
# 于是 296 条被当成"solver 已经全会"丢掉。现在 ok_full<k 也记 reason。
check("reason 记 args_mismatch（改动后：这种必须可见）", rsn[0], "args_mismatch")

print()
print("=== 3. 参数键顺序不同：算全等（sorted keys） ===")
gold = [[{"name": "search", "arguments": {"q": "x", "limit": 5}}]]
script = [[wrap([{"name": "search", "arguments": {"limit": 5, "q": "x"}}])]]
r, rf, rsn, pred, raw = run(["q"], gold, script, k=1)
check("名口径 1.0", r[0], 1.0)
check("全等口径 1.0（键序不该影响）", rf[0], 1.0)

print()
print("=== 4. 多个 call：名字序列对但第二个参数错 -> 全等不算 ===")
gold = [[{"name": "a", "arguments": {"x": 1}},
         {"name": "b", "arguments": {"y": 2}}]]
script = [[wrap([{"name": "a", "arguments": {"x": 1}},
                 {"name": "b", "arguments": {"y": 3}}])]]
r, rf, rsn, pred, raw = run(["q"], gold, script, k=1)
check("名口径 1.0（两个名都对）", r[0], 1.0)
check("全等口径 0.0（第二个参数错）", rf[0], 0.0)

print()
print("=== 5. call 顺序不同：名口径和历史一样**不**算命中（保序） ===")
gold = [[{"name": "a", "arguments": {}}, {"name": "b", "arguments": {}}]]
script = [[wrap([{"name": "b", "arguments": {}}, {"name": "a", "arguments": {}}])]]
r, rf, rsn, pred, raw = run(["q"], gold, script, k=1)
check("名口径 0.0（保序，和改动前一致）", r[0], 0.0)
check("全等口径 0.0", rf[0], 0.0)

print()
print("=== 6. arguments 不是 dict：名命中但全等不算 ===")
gold = [[{"name": "a", "arguments": {"x": 1}}]]
script = [[wrap([{"name": "a"}])]]  # 缺 arguments
r, rf, rsn, pred, raw = run(["q"], gold, script, k=1)
check("名口径 1.0", r[0], 1.0)
check("全等口径 0.0（没给参数不算全等）", rf[0], 0.0)

print()
print("=== 7. 解析失败 / 缺 tag：两个口径都不算，且不炸 ===")
gold = [[{"name": "a", "arguments": {}}]]
script = [["no tag here", "<think></think><tool_call_answer>not json</tool_call_answer>"]]
r, rf, rsn, pred, raw = run(["q"], gold, script, k=2)
check("名口径 0.0", r[0], 0.0)
check("全等口径 0.0", rf[0], 0.0)
check("reason 是 missing_tag（top1）", rsn[0], "missing_tag")
check("first_valid_pred 仍是 None（没有可解析的）", pred[0], None)

print()
print("=== 8. chunk_size 切分：跨 chunk 的 idx 不能错位 ===")
golds = [[{"name": "t", "arguments": {"i": i}}] for i in range(5)]
prompts = [f"q{i}" for i in range(5)]
# 第 0、3 个完全对；其余名对参错
script = []
for i in range(5):
    if i in (0, 3):
        script.append([wrap([{"name": "t", "arguments": {"i": i}}])] * 3)
    else:
        script.append([wrap([{"name": "t", "arguments": {"i": 999}}])] * 3)
r, rf, rsn, pred, raw = run(prompts, golds, script, k=3, chunk=2)
check("名口径全 1.0", r, [1.0] * 5)
check("全等口径 [1,0,0,1,0]", rf, [1.0, 0.0, 0.0, 1.0, 0.0])

print()
print("=== 9. 全等 <= 名（单调性，全等是名的子集） ===")
import random
random.seed(7)
golds = [[{"name": "t", "arguments": {"i": i}}] for i in range(30)]
prompts = [f"q{i}" for i in range(30)]
script = []
for i in range(30):
    grp = []
    for _ in range(4):
        pick = random.random()
        if pick < 0.4:
            grp.append(wrap([{"name": "t", "arguments": {"i": i}}]))       # 全等
        elif pick < 0.8:
            grp.append(wrap([{"name": "t", "arguments": {"i": 999}}]))     # 名对参错
        else:
            grp.append(wrap([{"name": "WRONG", "arguments": {"i": i}}]))   # 名也错
    script.append(grp)
r, rf, rsn, pred, raw = run(prompts, golds, script, k=4, chunk=7)
check("逐条 rf <= r", all(a <= b for a, b in zip(rf, r)), True)
check("两条口径不恒等（确实抓到了差异）", any(a != b for a, b in zip(rf, r)), True)

print()
if FAILS:
    print(f"!!! {len(FAILS)} 项不过: {FAILS}")
    sys.exit(1)
print("全部通过。")
