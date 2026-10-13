#!/bin/bash

# Activate this repo's venv explicitly. Bare `python` otherwise resolves to
# whatever venv the calling shell has active (search-zero/.venv on this box),
# which has no vllm and a torch too old for TRL.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.venv/bin/activate"

MODEL_PATH=$1
DATA_PATH=$2
OUTPUT_DIR=$3
RUN_NAME=$4
STEPS=$5

# Single GPU + LoRA: no DeepSpeed (see run_step1.sh).
export CUDA_VISIBLE_DEVICES=0

python step3_solver.py \
    --model_name_or_path "$MODEL_PATH" \
    --generated_data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --max_steps $STEPS \
    --save_steps 50 \
    --save_total_limit 2 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-6 \
    --dtype bfloat16 \
    --max_completion_length 2048 \
    --log_completions \
    --num_generations 4 \
    --remove_unused_columns False \
    --loss_type grpo \
    --use_peft \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --logging_steps 1 \
    --run_name "$RUN_NAME" \
    --report_to wandb
