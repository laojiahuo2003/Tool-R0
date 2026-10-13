#!/bin/bash

# Activate this repo's venv explicitly. Bare `python` otherwise resolves to
# whatever venv the calling shell has active (search-zero/.venv on this box),
# which has no vllm and a torch too old for TRL.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.venv/bin/activate"

MODEL_PATH=${1:-"Qwen/Qwen2.5-0.5B-Instruct"}
PORT=${2:-5000}

# NOTE: --served-model-name must stay "solver" and the port must stay 5000.
# rewards_generator.py falls back to base_url=http://localhost:5000/v1 and
# model="solver" when those keys are absent from the dataset columns.
echo "Starting vLLM on port $PORT with model: $MODEL_PATH"

export VLLM_USE_V1=0

CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --served-model-name solver \
  --port "$PORT" \
  --enforce-eager \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
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
