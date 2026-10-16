#!/bin/bash
# 跑一轮完整的 iter：step0(host solver) -> step1(训 generator) -> step2(出教材) -> step3(训 solver)。
#
#  用法: ./run_round.sh <ITER> [STEPS]        STEPS 默认 50
#
# 与 run_iter2.sh 的区别：那个写死了 iter2 的路径，这个按 ITER 推，iter3 起直接用。
# iter2 的起点特殊——iter1 是手工跑的，产物叫 iter1_generator_rl_lr1e4/ 和
# iter1_solver_rl/（lr 那轮的名字留在目录里了），不是 iter1_generator/。
#
# 三处为了"能无人值守跑完"加的东西：
#   1. 全流程日志落 round<ITER>.log，崩了能看到崩在哪一阶段
#   2. 每个阶段做完写 round<ITER>.<stage>.ok，重跑时跳过已完成的阶段
#      （step1 一个半小时，不该因为 step2 的一个小错从头再来）
#   3. 环境变量补齐：run_iter2.sh 一个 export 都没有，而 run_main.sh 里有
#      TRITON_CACHE_DIR / NCCL_* / VLLM_USE_V1=0 / VLLM_DISABLE_COMPILE_CACHE=1。
#      vLLM 0.8.4 配 TRL 要的 0.19+，V1 引擎实测会出问题，必须关。
set -u
cd /home/uos/code/Tool-R0
# 本仓库的绝对路径。kill_server 的身份核验要用它（见 _owns_pid）——
# 没有它那个核验会因为 want 为空而永远判否，等于没核验。
HERE=$PWD

ITER=${1:?用法: ./run_round.sh <ITER> [STEPS]}
STEPS=${2:-50}
BASE=qwen2.5-0.5b-instruct-tool-r0

# ---- 上一轮产物的路径映射 ----
if [ "$ITER" -eq 2 ]; then
    PREV_GEN="$BASE/iter1_generator_rl_lr1e4/checkpoint-50"
    PREV_SOLVER="$BASE/iter1_solver_rl/checkpoint-50"
else
    PREV_GEN="$BASE/iter$((ITER-1))_generator/checkpoint-$STEPS"
    PREV_SOLVER="$BASE/iter$((ITER-1))_solver/checkpoint-$STEPS"
fi

GEN="$BASE/iter${ITER}_generator"
SOLVER="$BASE/iter${ITER}_solver"
DATA="$BASE/iter${ITER}_data.json"
LOG="round${ITER}.log"
TS() { date +%m-%d_%H:%M:%S; }

log() { echo "[$(TS)] $*" | tee -a "$LOG"; }

# 每个阶段之间都要确认显存真的还回来了。step2 的三个脚本各自起停自己的 vLLM 实例，
# 万一有一个没退干净，下一阶段会在一个已经少了几百 MiB 的卡上跑，而 step1 对显存
# 极度敏感（见 run_step1.sh 里关于 WSL2 spill 的注释）。
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' '; }

die() { echo "!!! FATAL: $*" >&2; echo "[$(TS)] FATAL: $*" >> "$LOG"; exit 1; }

# ---- 起跑前检查 ----
[ -f "$PREV_GEN/model.safetensors" ]    || die "上一轮 generator 缺失: $PREV_GEN"
[ -f "$PREV_SOLVER/model.safetensors" ] || die "上一轮 solver 缺失: $PREV_SOLVER"

FREE=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' || echo 99999)
if [ "$((FREE))" -gt 2000 ]; then
    die "GPU 上还有 ${FREE} MiB 被占用。先确认那不是别人的服务——别人的进程不动。"
fi

export TRITON_CACHE_DIR="./triton_autotune"
mkdir -p "$TRITON_CACHE_DIR"
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export VLLM_USE_V1=0
export VLLM_DISABLE_COMPILE_CACHE=1
export WANDB_MODE=offline

log "########## round $ITER 起跑（STEPS=$STEPS）"
log "  gen 起点    = $PREV_GEN"
log "  solver 起点 = $PREV_SOLVER"
log "  本轮产物    = $GEN / $SOLVER / $DATA"

# 只认"本仓库 venv 起的进程"。两个条件缺一不可：
#   exe 解析后必须等于 .venv/bin/python 的解析路径，且 cwd 必须就是本仓库。
# 为什么不能只比 exe：uv 建的 venv 里 .venv/bin/python 是软链，解析出来是
# /home/uos/.local/share/uv/python/cpython-3.12.14-.../bin/python3.12，
# 别的仓库也用同一个 uv python 时 exe 同样匹配——那样会把别人的服务杀了。
# 为什么不能只按 *.venv/bin/python* 匹配（原来的写法）：解析后的路径里
# **不含 .venv/bin/python 这一段**，所以那个分支从来没进去过，一次都没杀成功过，
# 日志里却每次都打"不是本仓库 venv 的 python，不动它"，看着像在保护。
_owns_pid() {
    local p="$1" exe cwd want
    exe=$(readlink /proc/$p/exe 2>/dev/null || echo "")
    cwd=$(readlink /proc/$p/cwd 2>/dev/null || echo "")
    want=$(readlink -f "$HERE/.venv/bin/python" 2>/dev/null || echo "")
    [ -n "$want" ] && [ "$exe" = "$want" ] && [ "$cwd" = "$HERE" ]
}

# 打印 $1 的全部后代 pid。必须杀整棵树：vLLM 的 GPU 在 multiprocessing.spawn 起的
# worker 子进程里（API server 854859 的子进程 855019，RSS 5.65 GiB），
# 只杀 API server 的话 worker 留着继续占显存，端口也不一定放。
_tree() {
    local k
    for k in $(ps -o pid= --ppid "$1" 2>/dev/null); do
        echo "$k"
        _tree "$k"
    done
}

# 杀掉属于本仓库的整个进程树。$1 = 根 pid。
kill_tree_if_ours() {
    local p="$1" victim
    [ -n "$p" ] || return 0
    if _owns_pid "$p"; then
        for victim in "$p" $(_tree "$p"); do
            kill -TERM "$victim" 2>/dev/null || true
        done
    else
        log "!!! pid=$p 不是本仓库 venv 起的进程，不动它"
    fi
}

kill_server() {
    # 按 PID 杀，不用 pkill -f：模式匹配会误杀机器上别人的 vLLM 服务
    # （这台盒子上不止一个仓库在跑推理）。只有确认 PID 属于本仓库起的那个才杀。
    if [ -f vllm.pid ]; then
        local p
        p=$(cat vllm.pid)
        if [ -n "$p" ] && kill -0 "$p" 2>/dev/null; then
            kill_tree_if_ours "$p"
        fi
        rm -f vllm.pid
    fi
    # 兜底：只按端口认，不按全命令行模式认。但同样要验明正身——
    # 这台盒子上不止一个仓库在跑推理，万一别人的服务占了 5000，杀了它就等于
    # 动别人的进程。验不过就只报不动，然后让下面的显存检查去决定要不要停。
    local leftovers
    leftovers=$(ss -ltnp 2>/dev/null | awk '/:5000 /{print}' | grep -o 'pid=[0-9]*' | cut -d= -f2 | sort -u)
    for p in $leftovers; do
        [ -n "$p" ] || continue
        kill_tree_if_ours "$p"
    done
    sleep 8
    local u
    u=$(gpu_used)
    log "kill_server 后 GPU 已用 ${u} MiB"
    if [ -n "$u" ] && [ "$((u))" -gt 2500 ]; then
        die "收摊后 GPU 还占着 ${u} MiB。有进程没退干净——先查清楚再跑下一阶段，别在缺显存的卡上起 step1"
    fi
}

# ---------------------------------------------------------------- step0 + step1
if [ -f "round${ITER}.step1.ok" ]; then
    log ">>> step1 已有 round${ITER}.step1.ok，跳过"
else
    log ">>> step0: host 上一轮的 solver (port 5000)"
    bash run_step0.sh "$PREV_SOLVER" 5000 >> "$LOG" 2>&1 || die "step0: solver 服务没起来"
    curl -sf http://localhost:5000/v1/models | head -c 200 >> "$LOG" 2>&1

    log ">>> step1: 训 generator v$ITER（从上一轮接，$STEPS 步）"
    # 单独一份日志：parse_trainlog.py 是按 {'loss' 开头的 dict 解析 step 记录的，
    # step1 和 step3 的记录混在一个文件里会分不开是哪一阶段的。
    bash run_step1.sh "$PREV_GEN" "$GEN" \
        "qwen2.5-0.5b-instruct-tool-r0-iter${ITER}-gen" "$STEPS" \
        > "round${ITER}.step1.log" 2>&1 || die "step1 failed（细节看 round${ITER}.step1.log）"

    [ -f "$GEN/checkpoint-$STEPS/model.safetensors" ] \
        || die "step1 没产出 $GEN/checkpoint-$STEPS/model.safetensors"
    touch "round${ITER}.step1.ok"
    log ">>> step1 OK: $GEN/checkpoint-$STEPS"
fi
GEN_CKPT="$GEN/checkpoint-$STEPS"

kill_server

# ---------------------------------------------------------------- step2
if [ -f "round${ITER}.step2.ok" ]; then
    log ">>> step2 已有 round${ITER}.step2.ok，跳过"
else
    log ">>> step2: 出题 + 阅卷 + 选教材（verify 用上一轮 solver，难度按 agreement 分档）"
    bash run_step2.sh "$GEN_CKPT" "$PREV_SOLVER" "$DATA" \
        > "round${ITER}.step2.log" 2>&1 || die "step2 failed（细节看 round${ITER}.step2.log）"

    N=$(python - "$DATA" <<'PY'
import json, sys
try:
    print(len(json.load(open(sys.argv[1]))))
except Exception:
    print(0)
PY
)
    log ">>> step2 OK: $DATA（$N 条）"
    [ "$N" -ge 5 ] || die "教材只有 $N 条，不够训 step3"
    touch "round${ITER}.step2.ok"
fi

kill_server

# ---------------------------------------------------------------- step3
if [ -f "round${ITER}.step3.ok" ]; then
    log ">>> step3 已有 round${ITER}.step3.ok，跳过"
else
    log ">>> step3: 训 solver v$ITER（从上一轮接，$STEPS 步）"
    bash run_step3.sh "$PREV_SOLVER" "$DATA" "$SOLVER" \
        "qwen2.5-0.5b-instruct-tool-r0-iter${ITER}-solver" "$STEPS" \
        > "round${ITER}.step3.log" 2>&1 || die "step3 failed（细节看 round${ITER}.step3.log）"

    # step3 结束时会把 LoRA 合并回整模（TRL 对 PEFT 走 merge 分支），
    # 所以这里认 model.safetensors；adapter 形态兜底，万一 TRL 改了行为。
    if [ -f "$SOLVER/checkpoint-$STEPS/model.safetensors" ]; then
        :
    elif [ -f "$SOLVER/checkpoint-$STEPS/adapter_config.json" ]; then
        log "!!! 注意：$SOLVER/checkpoint-$STEPS 只有 adapter，没合并。step0/step2 用 vLLM 加载会失败"
        die "step3 产出的是未合并 adapter"
    else
        die "step3 没产出 $SOLVER/checkpoint-$STEPS"
    fi
    touch "round${ITER}.step3.ok"
    log ">>> step3 OK: $SOLVER/checkpoint-$STEPS"
fi

log "########## round $ITER 完成"
log "  generator: $GEN_CKPT"
log "  solver   : $SOLVER/checkpoint-$STEPS"
log "  教材     : $DATA"
nvidia-smi --query-gpu=memory.used --format=csv,noheader >> "$LOG" 2>&1 || true
echo "ROUND_${ITER}_DONE" > "round${ITER}.status"
# 放锁。launch_round.sh 靠这个锁防止起第二个训练进程，跑完了必须放掉，
# 否则下一轮会被"已经在跑"挡住。
rm -f "round${ITER}.launch.lock"
