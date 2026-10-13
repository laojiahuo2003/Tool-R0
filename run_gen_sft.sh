#!/bin/bash
# 用阿里云 MaaS 的 qwen3.7-flash 当教师，批量产 generator 的 SFT 热启动数据。
#
# key 只从环境变量读，不写进任何文件、不以命令行参数出现（ps 里看不到）：
#   export TEACHER_API_KEY=sk-xxxx
#   bash run_gen_sft.sh [n] [workers]
#
# 产物默认 sft_generator_data.jsonl，可直接喂给 run_sft.sh。
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 只有存在 .venv 时才激活（远程裸机没有 venv，直接走系统 python）。
if [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi

if [[ -z "${TEACHER_API_KEY:-}" ]]; then
    echo "缺少 TEACHER_API_KEY：export TEACHER_API_KEY=sk-xxxx" >&2
    exit 1
fi

N=${1:-1200}
WORKERS=${2:-12}

# 默认关掉独立 reasoning（--thinking 才开）：快约 4 倍、省约 4 倍 token，
# 且模型会把思考写进 <think> 块里，而那正是 SFT 要学的目标格式。
python gen_sft_data.py \
    --n "$N" \
    --workers "$WORKERS" \
    --out sft_generator_data.jsonl \
    --rejected-out sft_generator_data.rejected.jsonl \
    --summary-out sft_generator_data.summary.json
