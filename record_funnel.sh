#!/bin/bash
# 把每一轮 step2 的漏斗数字记下来，摆成一张跨轮对照表。
#
#  用法: ./record_funnel.sh <ITER> [STEPS]
#
# 为什么单独记：step2 每轮的 verify 用的是**上一轮**的 solver，所以"这一轮留存多少条"
# 衡量的其实是上一轮 solver 的判别力，不是这一轮 generator 的质量。这个混淆必须在表里
# 写清楚，不然会把"留存的变多"误读成"题出得更好"。
set -u
ITER=${1:?用法: ./record_funnel.sh <ITER> [STEPS]}
STEPS=${2:-50}
cd /home/uos/code/Tool-R0

DATA="qwen2.5-0.5b-instruct-tool-r0/iter${ITER}_data.json"
REPORT="${DATA}.verify_report.txt"
OUT="iter_funnel.jsonl"

[ -f "$REPORT" ] || { echo "!!! 没有 $REPORT，step2 还没跑完"; exit 1; }

# verify_report.txt 的头几行是 TOTAL / KEPT / DROPPED
TOTAL=$(awk '/^TOTAL:/{print $2}' "$REPORT")
KEPT=$(awk '/^KEPT:/{print $2}' "$REPORT")
DROPPED=$(awk '/^DROPPED:/{print $2}' "$REPORT")
NCUR=$(.venv/bin/python -c "import json,sys;print(len(json.load(open('$DATA'))))" 2>/dev/null || echo 0)

# 教材的难度配比（step2_select_curriculum.py 只打印不落盘，这里自己数）。
# difficulty 在输出 json 的**顶层**，不在 meta 里（实测踩过：按 meta 取会得 0/0/0）。
MIX=$(.venv/bin/python - "$DATA" <<'PY'
import json, sys, collections
d = json.load(open(sys.argv[1]))
def diff(x):
    return x.get("difficulty") or x.get("meta", {}).get("difficulty") or "?"
c = collections.Counter(diff(x) for x in d)
n = len(d) or 1
print("easy %d (%.0f%%) / medium %d (%.0f%%) / hard %d (%.0f%%)" % (
    c.get("easy", 0), 100 * c.get("easy", 0) / n,
    c.get("medium", 0), 100 * c.get("medium", 0) / n,
    c.get("hard", 0), 100 * c.get("hard", 0) / n))
PY
)

ROW=$(.venv/bin/python - "$ITER" "$TOTAL" "$KEPT" "$DROPPED" "$NCUR" "$MIX" <<'PY'
import json, sys, time
print(json.dumps({
    "iter": int(sys.argv[1]),
    "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
    "generated": int(sys.argv[2] or 0),
    "verified": int(sys.argv[3] or 0),
    "dropped": int(sys.argv[4] or 0),
    "curriculum": int(sys.argv[5] or 0),
    "mix": sys.argv[6],
    # verify 用第 ITER-1 轮的 solver，所以这一列衡量的是上一轮 solver 的判别力
    "verified_by": "iter%d solver" % (int(sys.argv[1]) - 1),
}, ensure_ascii=False))
PY
)
echo "$ROW" >> "$OUT"

echo "iter${ITER}: 生成 $TOTAL -> verify 留存 $KEPT（丢 $DROPPED）-> 教材 $NCUR 条"
echo "         配比 $MIX"
echo "         verify 用的是 iter$((ITER - 1)) 的 solver"
echo "         -> $OUT"
