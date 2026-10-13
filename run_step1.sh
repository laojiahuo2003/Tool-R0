#!/bin/bash

# Activate this repo's venv explicitly. Bare `python` otherwise resolves to
# whatever venv the calling shell has active (search-zero/.venv on this box),
# which has no vllm and a torch too old for TRL.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.venv/bin/activate"

MODEL_PATH=$1
OUTPUT_DIR=$2
RUN_NAME=$3
STEPS=$4
# Number of distinct task specs in the generator's training set. The paper uses
# 2000 (run_step1.sh upstream); this repo was shipped at 100, which means the
# generator only ever saw ~16% of the reachable spec space and saw each spec 4x
# over 100 steps. 2000 is the paper-faithful value.
N_SPECS=${5:-100}

# Single GPU + LoRA: the trainable params are small enough that DeepSpeed ZeRO-3
# buys nothing (there is no second device to shard across), so run plain python.
export CUDA_VISIBLE_DEVICES=0
# max_completion_length 2560 而不是作者的 2048，是本次唯一偏离作者配置的地方，
# 2026-10-13 实测后改的：第一步 clipped_ratio=0.3125，31% 的 completion 撞 2048 顶。
# 撞顶的样本缺闭合标签，tags_parsed/tools_json/gold_json 全 0、question gate 也过不去，
# 等于白跑。（注意：撞顶样本集中在长 completion 这件事，不要顺手推成"num_calls=2
# 才是难任务"——实测 num_calls 和 reward 相关性只有 -0.021，真正压 reward 的是
# tool_menu_size，见 memory 的 tool-r0-sft-coldstart-baseline。）
#
# 为什么不是探针说的"碰不到顶"：step1_generator.py 没设 temperature/top_p，走 TRL
# GRPOConfig 默认 temperature=1.0，比 smoke_test_generator.py 的 0.8 热得多；
# 且那个探针 --max-new-tokens 1600，结构上看不见 1600 以上的截断。两个因素叠一起，
# 探针注定测不出这个问题。以后测生成长度必须让 max_new_tokens 大于被检查的顶。
#
# 步耗时不是不变，是 +10%：2048 下 326.9s/it，2560 下前 7 步 326.9→360.7→403.3
# →362.8→367.4→378.2，稳态 ~365-380s。因为撞顶样本不是"免费"的——它照样生成
# 2560 个 token，只是没闭合标签；省下的撞顶样本省下的 token 抵不掉剩余样本多吐的。
# 换来的东西：clipped_ratio 从 31% 降到 ~27%（前 3 步看着像 12%，是 n=16 的噪声，
# 后 6 步 0.25-0.375 才是稳态），那些样本现在能拿格式分。
# offline 以免卡在 wandb 登录：这个环境没有 API key，而 report_to wandb 会在
# trainer 初始化时就调 maybe_login，直接以 UsageError 挂掉，根本跑不到第一步。
# run_sft.sh 早就加了这一行，step1 漏了（实测踩过）。
export WANDB_MODE=offline

python step1_generator.py \
    --model_name_or_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --max_steps $STEPS \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 1 \
    --run_name "$RUN_NAME" \
    --number_of_generated_data "$N_SPECS" \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-6 \
    --dtype bfloat16 \
    --max_completion_length 2560 \
    --log_completions \
    --num_generations 4 \
    --remove_unused_columns False \
    --loss_type grpo \
    --use_peft \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --report_to wandb
