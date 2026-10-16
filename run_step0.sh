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

# --gpu-memory-utilization 取 0.35，是从 iter1 的 vllm_server.log 实测倒推的，不是拍的。
# 那轮用 0.30，vLLM 自己报了账（10.00 GiB 卡）：
#   10.00 × 0.30 = 3.00 GiB 预算
#     model weights        0.93 GiB
#     non_torch_memory     0.05 GiB
#     PyTorch activation   1.40 GiB   <- 这一项是 profile 时的峰值，vLLM 先扣掉它
#     KV cache             0.62 GiB
#   最大并发 6.62×（8192 token/请求）
# 也就是说 0.93+0.05+1.40 = 2.38 GiB 是**地板**，低于它 KV cache 直接是负数，vLLM
# 起不来。地板对应 0.238，所以 0.20 这个值是不行的（改过又改回来，记下来免得再犯）。
#
# 取 0.35 而不是 0.30：预算 3.50 GiB，KV cache 1.12 GiB，并发 ~12×。奖励路径每一步
# 要并行问 solver（日志里 Running: 8 reqs 是常态），0.30 的 6.62× 会让请求排队。
# 多出来的 0.5 GiB 从 step1 那里借——step1 的峰值见 run_step1.sh 里 B=1 的 4.23 GiB，
# 10.24 - 3.50 = 6.74 GiB 还够，且比 iter1（solver 3.00 + 训练 12.62 = 15.62 GiB，
# 超物理显存 50%，全部静默 spill 到宿主内存）宽松得多。
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --served-model-name solver \
  --port "$PORT" \
  --enforce-eager \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.35 > vllm_server.log 2>&1 &

# 记下 PID，让调用方按 PID 收摊。用 pkill -f "vllm.entrypoints.openai.api_server"
# 收摊会误杀机器上别人的 vLLM 服务（这台盒子上不止一个仓库在跑），按 PID 杀不会。
echo $! > vllm.pid
echo "vLLM pid=$(cat vllm.pid) 日志=vllm_server.log"

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
