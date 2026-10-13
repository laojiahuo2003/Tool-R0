#!/bin/bash

# Activate this repo's venv if it exists (see run_main.sh for why it is optional).
VENV="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.venv"
[ -f "$VENV/bin/activate" ] && source "$VENV/bin/activate"

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
#
# 上游原版（emrecanacikgoz/Tool-R0）的 run_step1.sh 是 3 卡 accelerate launch +
# configs/deepseed_zero3.yaml 做**全参数微调**，不传 --use_peft。我们改成单卡 LoRA，
# 这是本脚本最大的一处偏离，直接影响 learning_rate 该怎么取（见下面 lr 那段）。
# 上游原值备查：CUDA_VISIBLE_DEVICES=0,1,2、accelerate --num_processes 3、
# per_device_train_batch_size 2、grad_accum 4、save_steps 2、num_generations 4、
# loss_type grpo、lr 1e-6、max_completion_length 2048。upstream run_main.sh 是
# BASE_MODEL=Qwen2.5-1.5B-Instruct、ITERATIONS=5、GEN/SOLVER 各 50 步，
# generator 从裸 base 起（无 SFT）。
export CUDA_VISIBLE_DEVICES=0

# ---- 与论文 Table 4 / 附录 C.1 对齐的超参（2026-10-13 补） ----
# weight_decay 1e-2 和 --beta 1e-2（论文的 λKL）论文 Table 4 明写，但上游原版
# run_step1.sh 两项都没传，TRL GRPOConfig 默认都是 0.0。这里取论文值：
# KL 项把策略锚在参考模型（SFT checkpoint）上，没有它策略可以任意漂移。
# max_completion_length 取论文附录 C.1 的 4096（"set to 4096 tokens, which
# accommodates full task specifications including tool menus and gold tool
# calls"），不取上游原版代码的 2048。两处冲突时选论文：撞顶是我们**唯一直接测量到
# 在伤害 reward 的因素**，而 lr 只是理论上的存疑（见下面 lr 那段，位移量已经够）。
# 2048 下实测 clipped_ratio=0.31，2560 下降到 ~0.27，都意味着四分之一的 completion
# 撞顶后缺闭合标签，tags_parsed/tools_json/gold_json 全 0、question gate 也过不去，
# 整组白跑。更要紧的是 GRPO 的 advantage 是组内归一化的：撞顶样本 reward=0 会把
# 组均值压低、自己拿到负 advantage，等于**主动教模型"别写长"**，和"写出完整 spec"
# 这个目标直接冲突。SFT 训练集实测 p99=2389、max=3783，只有 4096 能基本盖住。
#
# 为什么不是探针说的"碰不到顶"：step1_generator.py 没设 temperature/top_p，走 TRL
# GRPOConfig 默认 temperature=1.0，比 smoke_test_generator.py 的 0.8 热得多；
# 且那个探针 --max-new-tokens 1600，结构上看不见 1600 以上的截断。两个因素叠一起，
# 探针注定测不出这个问题。以后测生成长度必须让 max_new_tokens 大于被检查的顶。
#
# 撞顶样本集中在长 completion，但别顺手推成"num_calls=2 才是难任务"——实测
# num_calls 和 reward 相关性只有 -0.021，真正压 reward 的是 tool_menu_size
# （corr(menu,validity)=-0.627、corr(menu,curriculum)=-0.562），
# 因为 menu 越大 <available_tools> 的 JSON 越长，0.5B 写错类型/漏闭合的概率越高。
# 上面那两个相关系数是在 2560 那轮量的。
#
# 4096 有个连带要求：solver 服务必须用 --max-model-len 8192 起，不能用 4096。
# 奖励逻辑每次向 solver 声明 MAX_TOKENS_SOLVER=2048 个输出 token，vLLM 生成前会
# 校验「输入 + 声明的输出上限」是否超过 max-model-len，超了直接回 400。
# generator 的产出就是 solver 的输入，2560 时输入 ~2700、加 2048 也才 4750，4096 的顶
# 勉强够；4096 时输入能到 4200，一破就是 4097。实测 step1 第 23 步死于这个。
# run_step0.sh 已改成 8192，**起跑前要确认跑着的 solver 是 8192 那个**
# （ps 看 --max-model-len，旧进程是 4096 起的，不会自己升级）。
#
# learning_rate：论文 Table 4 是 1e-6（AdamW 固定），上游代码也是 1e-6。但那是给
# **全参数微调**定的值。我们改成了 LoRA r=16，而 LoRA 的惯例学习率比全参数高约两个
# 数量级（1e-4~5e-4），因为 LoRA 只有 lora_B 收得到有效梯度（lora_B 初始化成 0，
# ΔW=B·A 一开始就是 0），lr=1e-6 下实测 15 步后相对位移仅 ~0.006%，100 步也才 0.04%
# ——四标签发出率前 7 步 81/112=72.3%、后 8 步 93/128=72.7%，线性趋势 t=-0.51 不显著，
# 等于没学。论文用全参数+1e-6 能从 12.5% 四标签的裸模型做到 0.5B benchmark 15.47→30.57，
# 说明 1e-6 不是绝对值太小，是不适配 LoRA。这里取 1e-5 是折中：比全参数的 1e-6 高 10 倍，
# 比 LoRA 惯例的 1e-4 低 10 倍，先看它能不能把发出率推离 72%。
#
# save_steps 20 而不是上游的 2：上游每个 checkpoint 都是全参数模型，存 25 份放不下；
# 我们存的是合并后的 0.5B（~1G/份），20 步一份 + save_total_limit 3 能在同一次运行里
# 对比"到底有没有在动"，从而区分"lr 不够"和"步数不够"——这两个原因在单点采样下分不开。
# offline 以免卡在 wandb 登录：这个环境没有 API key，而 report_to wandb 会在
# trainer 初始化时就调 maybe_login，直接以 UsageError 挂掉，根本跑不到第一步。
# run_sft.sh 早就加了这一行，step1 漏了（实测踩过）。
export WANDB_MODE=offline

python step1_generator.py \
    --model_name_or_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --max_steps $STEPS \
    --save_steps 20 \
    --save_total_limit 3 \
    --logging_steps 1 \
    --run_name "$RUN_NAME" \
    --number_of_generated_data "$N_SPECS" \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --weight_decay 0.01 \
    --beta 0.01 \
    --dtype bfloat16 \
    --max_completion_length 4096 \
    --log_completions \
    --num_generations 4 \
    --remove_unused_columns False \
    --loss_type grpo \
    --use_peft \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --report_to wandb
