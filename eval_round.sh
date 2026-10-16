#!/bin/bash
# 一轮跑完后，用**同一套固定条件**评这一轮的 generator 和 solver。
#
#  用法: ./eval_round.sh <ITER> [上轮 ITER] [STEPS，默认 50]
#
# 为什么要固定条件：step3 每轮训的是当轮 generator 新出的题，题目换了就不能直接比
# reward。多轮对照必须有一份从第一轮之后再也不变的题（holdout_solver.json，135 条）
# 和一个从不改变的 spec 集（eval_iter.py 里 seed 20261015 的 192 条）。
set -u
ITER=${1:?用法: ./eval_round.sh <ITER> [上轮 ITER]}
PREV=${2:-$((ITER - 1))}
cd /home/uos/code/Tool-R0
STEPS=${3:-50}

GEN="qwen2.5-0.5b-instruct-tool-r0/iter${ITER}_generator/checkpoint-$STEPS"
SOL="qwen2.5-0.5b-instruct-tool-r0/iter${ITER}_solver/checkpoint-$STEPS"
DATA="qwen2.5-0.5b-instruct-tool-r0/iter${ITER}_data.json"

[ -f "$GEN/model.safetensors" ]  || { echo "!!! 缺 $GEN/model.safetensors"; exit 1; }
[ -f "$SOL/model.safetensors" ]  || { echo "!!! 缺 $SOL/model.safetensors"; exit 1; }

# correctness 门：这一轮的教材不能和固定评测集重题。solver 的分数是在 holdout 上量的，
# 而教材是当轮 generator 产的——同源，真有可能撞。撞了的话 held-out 就不是 held-out，
# "泛化了"这个结论直接作废，而这种污染在日志里看不出来，只有逐条比对能发现。
# 实测 iter1 用的是干净的（0 重题），所以 iter1 的基线可用。
if [ -f "$DATA" ]; then
    .venv/bin/python check_holdout_clean.py "$DATA" || {
        echo "!!! $DATA 污染了固定评测集，先查清楚再评——不然报出去的是假 held-out 分数"
        exit 1
    }
else
    echo "[warn] 没有 $DATA，跳过评测集污染检查"
fi

LOG="/tmp/eval_iter${ITER}.log"
: > "$LOG"
echo "[eval_round] $(date +%H:%M:%S) 评 iter${ITER}" | tee -a "$LOG"

# 未合并的 adapter 也能评（eval 走 transformers 加载），但要说明白，因为 step0/step2 用不了
for m in "$GEN" "$SOL"; do
    [ -f "$m/model.safetensors" ] || echo "[warn] $m 只有 adapter，没合并" | tee -a "$LOG"
done

.venv/bin/python eval_iter.py --tag "iter${ITER}_gen" --gen_ckpt "$GEN" \
    --note "iter${ITER} step1 产物" >> "$LOG" 2>&1 || { echo "!!! iter${ITER}_gen 失败"; tail -20 "$LOG"; exit 1; }
.venv/bin/python eval_iter.py --tag "iter${ITER}_solver" --solver_ckpt "$SOL" \
    --note "iter${ITER} step3 产物" >> "$LOG" 2>&1 || { echo "!!! iter${ITER}_solver 失败"; tail -20 "$LOG"; exit 1; }

echo "[eval_round] $(date +%H:%M:%S) 评完，开始配对检验" | tee -a "$LOG"
echo | tee -a "$LOG"
for spec in "gen four_tag" "gen pv_ok" "solver fmt" "solver acc" "solver name_hit"; do
    set -- $spec
    echo "===== iter${PREV}_$1 -> iter${ITER}_$1  指标 $2 =====" | tee -a "$LOG"
    .venv/bin/python paired_test.py --a "iter${PREV}_$1" --b "iter${ITER}_$1" \
        --field "$1" --metric "$2" 2>&1 | tee -a "$LOG"
    echo | tee -a "$LOG"
done

echo "[eval_round] 多轮对照表：" | tee -a "$LOG"
.venv/bin/python compare_rounds.py 2>&1 | tee -a "$LOG"
echo "[eval_round] 全部完成 -> $LOG" | tee -a "$LOG"
