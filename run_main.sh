#!/bin/bash

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
# Generator 的 RL 起点。默认等于 BASE_MODEL；做过 SFT 热启动就把它指到
# run_sft.sh 产出的合并 checkpoint（<OUTPUT_DIR>/checkpoint-<N>），
# 这样只有第 1 轮的 generator 从热启动模型起，solver 仍从 BASE_MODEL 起。
GEN_INIT_MODEL="${GEN_INIT_MODEL:-$BASE_MODEL}"
ABBREVIATION="${ABBREVIATION:-qwen2.5-3b-instruct-tool-r0}"
ITERATIONS=1
# 4 步时每个 optimizer step 只有 2 个不同 prompt（generation_batch_size=8 completions
# ÷ num_generations 4），全程只见过 8 个 prompt，3/4 的步 grad_norm=0、frac_reward_zero_std=1.0，
# generator 实质没动，step2 只能 kept 7/300——所以步数不能太少。这里取 50：与论文每轮的
# 50 步、以及本文件的 TRAIN_STEPS_SOLVER=50 对齐，50 步 ≈ 100 个 prompt（N_SPECS=100 时
# 约 1 epoch，每步约 2 个不同 prompt），时间比 100 步减半，先把闭环跑通再谈加量。
# Trainer 会在 max_steps 处存一份最终 checkpoint，所以 50 能被 run_step1.sh 的
# save_steps 20 整除与否都不影响 checkpoint-50 的存在（SFT 那轮 73 步存出 checkpoint-73 即证）。
TRAIN_STEPS_GEN=50
TRAIN_STEPS_SOLVER=50
BASE_DIR="./${ABBREVIATION}"

# Activate this repo's venv if it exists. On boxes that already ship a working
# ROCm/CUDA stack system-wide (e.g. a bare MI300X image) there is no .venv, and a
# hard `source` would abort the run. When .venv does exist (the original dev box)
# we still want it, otherwise bare `python` resolves to some unrelated venv.
VENV="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.venv"
[ -f "$VENV/bin/activate" ] && source "$VENV/bin/activate"

# Offline so a run does not stall on a wandb login prompt.
export WANDB_MODE=offline
export WANDB_PROJECT="self-play-$ABBREVIATION"

echo "----------------------------------------------------------------"
echo "Starting Self-Play (Iter 1 explicit + Loop for 2..$ITERATIONS)"
echo "Base Model: $BASE_MODEL"
echo "----------------------------------------------------------------"

export TRITON_CACHE_DIR="./triton_autotune"
mkdir -p "$TRITON_CACHE_DIR"

export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

export VLLM_USE_V1=0
export VLLM_DISABLE_COMPILE_CACHE=1

# ---------------------------------------------------------------- assertions
# Every stage must leave a real artifact behind. Without these checks the script
# printed "Self-Play Complete." even when step2 wrote [] and step3 never trained,
# which made a dead run look like a finished one.
die() { echo "!!! FATAL: $*" >&2; exit 1; }

require_dir()  { [ -d "$1" ] || die "$2 not found at $1"; }
require_file() { [ -f "$1" ] || die "$2 not found at $1"; }

json_rows() {
    python - "$1" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    print(len(d) if isinstance(d, (list, dict)) else 0)
except Exception:
    print(0)
PY
}

require_solver_ckpt() {
    require_dir "$1/checkpoint-${TRAIN_STEPS_SOLVER}" "solver checkpoint"
    require_file "$1/checkpoint-${TRAIN_STEPS_SOLVER}/adapter_config.json" \
                 "solver adapter config"
}
# ---------------------------------------------------------------------------


echo ">>> STARTING ITERATION 1 (Initialization)"

GEN_V1_DIR="${BASE_DIR}/iter1_generator"
SOLVER_V1_DIR="${BASE_DIR}/iter1_solver"
DATA_V1_JSON="${BASE_DIR}/iter1_data.json"

echo "[Iter 1] Step 0: Hosting Base Model (Solver v0)..."
bash run_step0.sh "$BASE_MODEL" 5000

echo "[Iter 1] Step 1: Training Generator v1..."
bash run_step1.sh \
    "$GEN_INIT_MODEL" \
    "$GEN_V1_DIR" \
    "${ABBREVIATION}-iter1-gen" \
    "$TRAIN_STEPS_GEN" || die "step1 failed"

GEN_V1_CKPT="${GEN_V1_DIR}/checkpoint-${TRAIN_STEPS_GEN}"

# step1 merges the LoRA adapter back into the base weights, so the checkpoint is
# a plain model dir with safetensors in it, not an adapter dir.
require_dir  "$GEN_V1_CKPT" "generator checkpoint"
require_file "$GEN_V1_CKPT/model.safetensors" "merged generator weights"
echo "[Iter 1] generator checkpoint OK: $GEN_V1_CKPT"

# Free the card for step2. Scoped to the api_server pattern on purpose: a bare
# `pkill python` here would also take down unrelated python processes on the box.
pkill -f "vllm.entrypoints.openai.api_server" || true
sleep 5
# 看一眼显存是否真的释放。CUDA 机器用 nvidia-smi，ROCm 机器用 rocm-smi，两个都没有就跳过。
nvidia-smi 2>/dev/null || rocm-smi 2>/dev/null || true


export VLLM_WORKER_MULTIPROC_METHOD=spawn
echo "[Iter 1] Step 2: Generating Data..."
bash run_step2.sh \
    "$GEN_V1_CKPT" \
    "$BASE_MODEL" \
    "$DATA_V1_JSON" || die "step2 failed"

DATA_ROWS=$(json_rows "$DATA_V1_JSON")
echo "[Iter 1] dataset rows = $DATA_ROWS"
[ "$DATA_ROWS" -ge 5 ] || die "dataset $DATA_V1_JSON has only $DATA_ROWS rows; refusing to train step3 on it"

pkill -f "vllm.entrypoints.openai.api_server" || true
sleep 5

echo "[Iter 1] Step 3: Training Solver v1..."
bash run_step3.sh \
    "$BASE_MODEL" \
    "$DATA_V1_JSON" \
    "$SOLVER_V1_DIR" \
    "${ABBREVIATION}-iter1-solver" \
    "$TRAIN_STEPS_SOLVER" || die "step3 failed"

require_solver_ckpt "$SOLVER_V1_DIR"
echo ">>> FINISHED ITERATION 1"


for (( i=2; i<=ITERATIONS; i++ ))
do
    prev=$((i-1))
    echo ">>> STARTING ITERATION $i (Derived from v$prev)"

    PREV_SOLVER_DIR="${BASE_DIR}/iter${prev}_solver"
    PREV_SOLVER_CKPT="${PREV_SOLVER_DIR}/checkpoint-${TRAIN_STEPS_SOLVER}"
    PREV_GEN_DIR="${BASE_DIR}/iter${prev}_generator"
    PREV_GEN_CKPT="${PREV_GEN_DIR}/checkpoint-${TRAIN_STEPS_GEN}"


    CURR_GEN_DIR="${BASE_DIR}/iter${i}_generator"
    CURR_SOLVER_DIR="${BASE_DIR}/iter${i}_solver"
    CURR_DATA_JSON="${BASE_DIR}/iter${i}_data.json"

    if [ ! -d "$PREV_SOLVER_CKPT" ]; then
        die "Previous solver checkpoint not found at $PREV_SOLVER_CKPT"
    fi
    if [ ! -d "$PREV_GEN_CKPT" ]; then
        die "Previous generator checkpoint not found at $PREV_GEN_CKPT"
    fi


    echo "[Iter $i] Step 0: Hosting Solver v$prev..."
    bash run_step0.sh "$PREV_SOLVER_CKPT" 5000


    echo "[Iter $i] Step 1: Training Generator v$i..."
    bash run_step1.sh \
        "$PREV_GEN_CKPT" \
        "$CURR_GEN_DIR" \
        "${ABBREVIATION}-iter${i}-gen" \
        "$TRAIN_STEPS_GEN"


    CURR_GEN_CKPT="${CURR_GEN_DIR}/checkpoint-${TRAIN_STEPS_GEN}"

    pkill -f "vllm.entrypoints.openai.api_server" || true
    sleep 5

    echo "[Iter $i] Step 2: Generating Data..."
    bash run_step2.sh \
        "$CURR_GEN_CKPT" \
        "$PREV_SOLVER_CKPT" \
        "$CURR_DATA_JSON"

    pkill -f "vllm.entrypoints.openai.api_server" || true
    sleep 5

    echo "[Iter $i] Step 3: Training Solver v$i..."
    bash run_step3.sh \
        "$PREV_SOLVER_CKPT" \
        "$CURR_DATA_JSON" \
        "$CURR_SOLVER_DIR" \
        "${ABBREVIATION}-iter${i}-solver" \
        "$TRAIN_STEPS_SOLVER" || die "step3 (iter $i) failed"

    require_solver_ckpt "$CURR_SOLVER_DIR"
    echo ">>> FINISHED ITERATION $i"
done

FINAL_SOLVER="${BASE_DIR}/iter${ITERATIONS}_solver/checkpoint-${TRAIN_STEPS_SOLVER}"
require_solver_ckpt "${BASE_DIR}/iter${ITERATIONS}_solver"

echo "================================================================"
echo "Self-Play Complete."
echo "Model: $FINAL_SOLVER"
echo "================================================================"
