#!/bin/bash

# Activate this repo's venv if it exists (see run_main.sh for why it is optional).
VENV="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.venv"
[ -f "$VENV/bin/activate" ] && source "$VENV/bin/activate"

MODEL_PATH=$1
DATA_PATH=$2
OUTPUT_DIR=$3
RUN_NAME=$4
STEPS=$5

# Single GPU + LoRA: no DeepSpeed (see run_step1.sh).
export CUDA_VISIBLE_DEVICES=0

# 2026-10-14 补：这一行原来是漏的，而脚本里有 --report_to wandb。
# 本机没有 API key（~/.netrc 不存在、WANDB_API_KEY 未设置、wandb.api.api_key 返回
# None），TRL 在 trainer 初始化时就调 wandb.init(mode="online")，实测直接以
# UsageError: No API key configured 挂掉，跑不到第一步。
# run_step1.sh 和 run_sft.sh 都早有这一行（run_step1.sh 的注释写明了"实测踩过"），
# step3 漏了。离线模式照样写本地 run 目录，只是不上传。
export WANDB_MODE=offline

# ---- learning_rate：1e-6 -> 1e-4（2026-10-14 改，原值没有任何注释说明来历）----
# 论文 Table 4 是 1e-6（AdamW 固定），上游原版代码也是 1e-6，但那是给**全参数微调**定的。
# 我们和 step1 一样是 LoRA r=16、target_modules 没传（PEFT 默认 q_proj/v_proj）。
# LoRA 只有 lora_B 收得到有效梯度（lora_B 初始化成 0，ΔW=B·A 一开始就是 0），
# 梯度天生小一个量级，lr 必须比全参数高约两个数量级。
#
# 这不是推理，是 step1 上已经付过两轮代价量出来的：
#   lr=1e-5  实测 24 步：全局 ||ΔW||_F/||W||_F = 0.0047%，494M 个元素里只有 0.47%
#            真的变了，ΔW 峰值只有 bf16 同量级 1 ULP 的 0.57~3.73 倍。四标签率
#            前 12 步 76.0% -> 后 12 步 72.4%，斜率 t=-1.11，与"完全没动"区分不开。
#            => 24 步白跑。
#   lr=1e-4  实测 50 步：位移 0.2797%，四标签齐全率 73.5% -> 89.75%（cluster-robust
#            t 从 +3.20 到 +6.76，四路奖励全显著上升）。=> 有效。
# AdamW 把每步更新幅度归一化到 ≈lr，与任务无关；1e-6 外推位移 ~0.0028%、
# ~99.5% 的更新低于 1 bf16 ULP（q/v 权重量级 0.02~0.1，该量级 1 ULP 是 1.2e-4~4.9e-4，
# 比 1e-6 大 1~2 个数量级）。"更新落不了地"不随"solver 任务奖励更密"而改变——物理上
# 就表示不出来。所以别先按 1e-6 跑一遍"看看"。
#
# 只改这一个变量。仍待定、这次**没动**的两项（都是论文 Table 4 值而上游原版没传）：
#   --beta 0.01（λKL）：不传则 TRL 默认 0.0，策略没有锚在参考模型上，100 条数据
#     × 约 100 步可能漂移。step1 传了。
#   --weight_decay 0.01：论文值，对 LoRA 影响小。
# 一次只改一个，所以这两项留给你们决定，别和 lr 混在一起改。

python step3_solver.py \
    --model_name_or_path "$MODEL_PATH" \
    --generated_data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --max_steps $STEPS \
    --save_steps 50 \
    --save_total_limit 2 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --beta 0.01 \
    --weight_decay 0.01 \
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
# 2026-10-15 补 --beta 0.01 --weight_decay 0.01：论文 Table 4 明写，上游原版没传，
# 于是 iter1 的 step3 是在 TRL 默认 beta=0.0（无 KL 锚）下跑的。上面注释里"留待决定"
# 的那两项，这次决定补上，理由：
#   * beta 是 λKL，把策略锚在参考模型（上一轮的 solver）上。solver 每轮从上一轮接着训，
#     没有锚的话 100 条数据 × 50 步足够让它漂走，而"多轮上升"要的恰恰是不漂。
#   * step1 一直都传这两个值，step3 不传是纯粹的不一致，不是有意的对照。
#   * 位移量实测：step1 在 beta=0.01 下 50 步位移 0.2797%，四标签率 73.5%→89.75%，
#     说明 0.01 这个量级不会把学习摁死，不是"加了就什么都学不到"的那种大值。
# 代价：iter1 的 step3 是在 beta=0 下跑的，所以 iter1→iter2 的 solver 指标不可直接比。
# iter2 起两边一致，iter2→iter3→iter4 是干净的多轮对照。
