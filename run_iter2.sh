#!/bin/bash
# iter2 自举：把 iter1 的产物接上，再转一圈。
#
# 为什么单独写一个脚本而不是把 run_main.sh 的 ITERATIONS 改成 2：
#   run_main.sh 的循环按固定名字找 `${BASE_DIR}/iter1_solver/checkpoint-50` 和
#   `${BASE_DIR}/iter1_generator/checkpoint-50`，但 iter1 是手工跑的，实际产物叫
#   `iter1_solver_rl/` 和 `iter1_generator_rl_lr1e4/`（lr 那轮的名字留在了目录里）。
#   直接改 ITERATIONS=2 会在循环开头 die，而且 iter1 那段会无条件重跑一遍（6.8 小时）。
#   所以这里显式映射路径，步骤顺序与 run_main.sh 的循环完全一致。
#
# 与 iter1 的差异（都是自举该有的样子，不是新调参）：
#   step0  host 的不再是 base，而是 iter1 训出来的 solver
#   step1  generator 从 iter1 的 generator 接着训，不再从 base 起
#   step2  verify 和 judge 都用 iter1 的 solver（base 时 judge 退化成 562/563 全判 hard，
#         换成训过的 solver 之后会不会恢复，这是 iter2 主要想看的事）
#   step3  solver 从 iter1 的 solver 接着训，不再从 base 起
#
# 一次只动一个变量：run_step3.sh 仍然不传 --beta/--weight_decay（论文 Table 4 值，
# 上游漏传，留作对照实验），所以 iter2 与 iter1 在超参上完全可比。
set -u
cd /home/uos/code/Tool-R0

BASE_DIR=qwen2.5-0.5b-instruct-tool-r0
PREV_GEN="$BASE_DIR/iter1_generator_rl_lr1e4/checkpoint-50"
PREV_SOLVER="$BASE_DIR/iter1_solver_rl/checkpoint-50"
GEN2="$BASE_DIR/iter2_generator"
SOLVER2="$BASE_DIR/iter2_solver"
DATA2="$BASE_DIR/iter2_data.json"
STEPS=50
TS() { date +%H:%M:%S; }

die() { echo "!!! FATAL: $*" >&2; exit 1; }

# ---- 起跑前检查：iter1 的产物必须在，否则后面全是白跑 ----
for f in "$PREV_GEN/model.safetensors" "$PREV_SOLVER/model.safetensors"; do
    [ -f "$f" ] || die "iter1 产物缺失: $f"
done
[ -d "$GEN2" ] && die "$GEN2 已存在，先确认要不要删掉重跑"
[ -d "$SOLVER2" ] && die "$SOLVER2 已存在，先确认要不要删掉重跑"
[ -f "$DATA2" ] && die "$DATA2 已存在，先确认要不要删掉重跑"
nvidia-smi --query-gpu=memory.used --format=csv,noheader
echo "[$(TS)] iter2 起跑。gen 起点=$PREV_GEN  solver 起点=$PREV_SOLVER"

# ---------------------------------------------------------------- step0
echo "[$(TS)] >>> step0: host iter1 solver (port 5000, max-model-len 8192)"
bash run_step0.sh "$PREV_SOLVER" 5000 || die "step0: solver 服务没起来"
curl -sf http://localhost:5000/v1/models | head -c 200; echo

# ---------------------------------------------------------------- step1
echo "[$(TS)] >>> step1: 训 generator v2（从 iter1 generator 接，50 步，约 7 小时）"
bash run_step1.sh "$PREV_GEN" "$GEN2" "qwen2.5-0.5b-instruct-tool-r0-iter2-gen" "$STEPS" \
    || die "step1 failed"

GEN2_CKPT="$GEN2/checkpoint-$STEPS"
[ -f "$GEN2_CKPT/model.safetensors" ] || die "step1 没产出 $GEN2_CKPT/model.safetensors"
echo "[$(TS)] step1 OK: $GEN2_CKPT"

# 显存还给 step2。只按 api_server 的模式杀：裸 pkill python 会把机器上别的 python
# （比如编辑器的 language server）一起带走。
pkill -f "vllm.entrypoints.openai.api_server" || true
sleep 5
nvidia-smi --query-gpu=memory.used --format=csv,noheader

# ---------------------------------------------------------------- step2
# run_step2.sh 里已经带了 --drop_agreement_ge 1.0（饱和过滤），verify 用 iter1 solver。
echo "[$(TS)] >>> step2: 出题 + 阅卷 + 选教材（solver/judge 都是 iter1 solver）"
bash run_step2.sh "$GEN2_CKPT" "$PREV_SOLVER" "$DATA2" || die "step2 failed"

python - "$DATA2" <<'PY' || die "step2 产出的数据不可用"
import json, sys
d = json.load(open(sys.argv[1]))
print("[check] iter2 教材 %d 条" % len(d))
sys.exit(0 if len(d) >= 5 else 1)
PY
echo "[$(TS)] step2 OK: $DATA2"

pkill -f "vllm.entrypoints.openai.api_server" || true
sleep 5
nvidia-smi --query-gpu=memory.used --format=csv,noheader

# ---------------------------------------------------------------- step3
echo "[$(TS)] >>> step3: 训 solver v2（从 iter1 solver 接，50 步）"
bash run_step3.sh "$PREV_SOLVER" "$DATA2" "$SOLVER2" \
    "qwen2.5-0.5b-instruct-tool-r0-iter2-solver" "$STEPS" || die "step3 failed"

[ -f "$SOLVER2/checkpoint-$STEPS/model.safetensors" ] \
    || die "step3 没产出合并后的整模（TRL 对 PEFT 走 merge 分支，正常应该有）"
echo "[$(TS)] >>> iter2 完成: $SOLVER2/checkpoint-$STEPS"
