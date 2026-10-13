#!/bin/bash

# Activate this repo's venv if it exists (see run_main.sh for why it is optional).
VENV="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.venv"
[ -f "$VENV/bin/activate" ] && source "$VENV/bin/activate"

MODEL_PATH=${1:-${BASE_MODEL:-Qwen/Qwen2.5-3B-Instruct}}
PORT=${2:-5000}

# NOTE: --served-model-name must stay "solver" and the port must stay 5000.
# rewards_generator.py falls back to base_url=http://localhost:5000/v1 and
# model="solver" when those keys are absent from the dataset columns.
#
# --max-model-len 用 8192 而不是 4096：奖励逻辑每次向 solver 声明
# MAX_TOKENS_SOLVER=2048 个输出 token，而 vLLM 在生成前会先校验「输入 + 声明的输出
# 上限」是否超过 max-model-len，超了直接回 400，整轮训练就断在那里。
# 生成器吐出的任务描述可以长到两千多 token，叠上 solver 的系统提示后输入轻松越过
# 2048，4096 的顶一破就是 4097。实测 step1 跑到第 23 步就是这么挂的。
# 8192 留足余量，3B 模型多这几千 token 的 KV cache 几乎不占显存。
echo "Starting vLLM on port $PORT with model: $MODEL_PATH"

export VLLM_USE_V1=0

CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --served-model-name solver \
  --port "$PORT" \
  --enforce-eager \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.30 > vllm_server.log 2>&1 &

echo "Waiting for vLLM to initialize..."
# Poll instead of a fixed sleep: the reward path has no error handling around
# this endpoint, so a half-started server turns into an APIConnectionError
# part-way through step1.
for i in $(seq 1 60); do
    if curl -sf "http://localhost:${PORT}/v1/models" 2>/dev/null | grep -q "solver"; then
        echo "vLLM is ready (after ~$((i * 10))s)."
        exit 0
    fi
    sleep 10
done

echo "ERROR: vLLM did not become ready within 600s. Check vllm_server.log" >&2
exit 1
