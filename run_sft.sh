#!/bin/bash
# Generator 的 SFT 热启动。产物是一个"完整权重"目录（LoRA 已合并），
# 可以直接当作 step1 GRPO 的起始模型，也可以直接被 step2 的 vLLM 加载。
#
# 典型链路：gen_sft_data.py 产数据 -> run_sft.sh 热启动 -> run_step1.sh 进 RL
#
# 用法：
#   bash run_sft.sh \
#       Qwen/Qwen2.5-3B-Instruct \
#       ./qwen2.5-3b-instruct-tool-r0/iter1_generator_sft \
#       iter1-gen-sft \
#       ./sft_generator_data.jsonl \
#       3

# 显式激活本仓库的 venv（若存在）：否则裸 python 会落到调用 shell 当前激活的别的 venv。
# 远程裸机没有 .venv 时跳过这一句，直接用系统 python。
VENV="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.venv"
[ -f "$VENV/bin/activate" ] && source "$VENV/bin/activate"

MODEL_PATH=$1
OUTPUT_DIR=$2
RUN_NAME=$3
DATA_PATH=${4:-./sft_generator_data.jsonl}
EPOCHS=${5:-3}

export CUDA_VISIBLE_DEVICES=0
# offline 以免卡在 wandb 登录。
export WANDB_MODE=offline

# max_length 4096 而不是 3072：实测 prompt 恒为 566 token，target 969-1773，
# total 最大 2341。3072 只剩 ~730 token 余量，menu=8 + calls=2 的长尾样本会顶到，
# 而 completion_only_loss 下被截掉的正好是尾部的 </think> —— 那恰是最要学的标签。
# 0.5B + packing False，4096 的显存代价可以忽略。

python sft_generator.py \
    --model_name_or_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --run_name "$RUN_NAME" \
    --sft_data_path "$DATA_PATH" \
    --num_train_epochs "$EPOCHS" \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --lr_scheduler_type cosine \
    --bf16 True \
    --max_length 4096 \
    --completion_only_loss True \
    --packing False \
    --logging_steps 1 \
    --save_steps 50 \
    --save_total_limit 2 \
    --report_to wandb \
    --use_peft \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05