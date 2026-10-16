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
# **（2026-10-14 实测修正：lr=1e-4 满 50 步的结果表明"教模型别写长"这个担心是错的。
# 模型确实学会了写短（mean 7917→5411 字符，p90 15660→8159，clipped_ratio 0.125→~0，
# 最后一次非零在 step 48），但四标签齐全率同时从 73.5% 涨到 89.75%，内容没变差。
# 写短省下的 token 全花在闭合标签上了，两个目标不冲突。）**
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
# ΔW=B·A 一开始就是 0），梯度天生就小一个量级，lr 再按全参数给就等于没学。
#
# 这个结论是实测逼出来的，不是推的。2026-10-13 那轮 lr=1e-5 跑了 24 步（机器 02:30
# 干净关机打断，不是崩溃），四路证据全都指向"没动"：
#   1) 权重位移：合并后 checkpoint-20 对比 SFT 基座 checkpoint-219，全局
#      ||ΔW||_F/||W||_F = 0.0047%，q_proj/v_proj 单项 ~0.03%，494M 个元素里只有
#      231 万（0.47%）真的变了。ΔW 峰值只有 bf16 在该量级 1 ULP 的 0.57~3.73 倍
#      ——等于在 bf16 的噪声地板上训练。
#   2) 四标签发出率：全程 285/384 = 74.2%，前 12 步 76.0% → 后 12 步 72.4%，
#      加权线性回归斜率 -0.0031/步、t=-1.11，与"完全没动"区分不开。
#   3) smoke test（同 seed 1234、n=24、temp 0.8）：RL ckpt-20 是 62.5% 四标签 /
#      37.5% pv_ok，SFT 基线 75.0% / 54.2%，z=-0.93 / -1.16 都在噪声内。
#   4) grad_norm 0.105~0.179 一直不为 0、kl ~0.001，说明优化器在工作，只是步子太小。
# 注意 (2)(3) 这两个数是"没变好"，不是"变差了"——n=24 下 62.5% vs 75.0% 不显著。
#
# 所以这次取 1e-4：LoRA 惯例区间的下沿，比 1e-5 高 10 倍。AdamW 把每步更新幅度归一化
# 到 ≈lr，位移量近似 lr×步数 的线性关系，10 倍 lr 预期把位移从 0.03% 推到 ~0.3%。
# 只改这一个变量：LoRA 目标模块（现在只有 q_proj/v_proj，是 PEFT 默认，没显式指定）
# 是容量问题，lr 没修好之前容量根本测不出来，不要一起改。如果 1e-4 跑满 50 步位移
# 到了 ~0.3% 但四标签率还是不动，下一个该动的是 target_modules（加 k/o/gate/up/down），
# 不是继续加步数或加 lr。
#
# save_steps 20 而不是上游的 2：上游每个 checkpoint 都是全参数模型，存 25 份放不下；
# 我们存的是合并后的 0.5B（~1G/份），20 步一份 + save_total_limit 3 能在同一次运行里
# 对比"到底有没有在动"，从而区分"lr 不够"和"步数不够"——这两个原因在单点采样下分不开。
# offline 以免卡在 wandb 登录：这个环境没有 API key，而 report_to wandb 会在
# trainer 初始化时就调 maybe_login，直接以 UsageError 挂掉，根本跑不到第一步。
# run_sft.sh 早就加了这一行，step1 漏了（实测踩过）。
export WANDB_MODE=offline

# ---- per_device_train_batch_size 4 / accum 4  ->  1 / 16（2026-10-15 改）----
# 优化器步完全等价：每个 step 还是 4 prompt × num_generations 4 = 16 条 completion、
# 还是每 16 条反传一次，梯度求和的项一个不多一个不少。改的只是"这 16 条分几批过
# 前向/反向"。
#
# 为什么必须改：TRL 的 _get_per_token_logps_and_entropies（grpo_trainer.py:1452-1560）
# 把 logits_to_keep 设成整个 completion 长度，还额外调 entropy_from_logits，于是物化
# [16, 1200, 151936] 的 logits 外加 softmax/exp 拷贝。这台机器是 10 GiB 的 3080，
# 实测 B=16 峰值 12.62 GiB、B=4 降到 4.23 GiB，而 step1 跑着的时候 solver server
# （--gpu-memory-utilization 0.30）还占着约 3 GiB，只剩 6.5 GiB——必超。
#
# 超了不会报 OOM。WSL2/dxgkrnl 在 CUDA 分配超过物理显存时默默 spill 到宿主内存，
# PyTorch 自己都报 "56.08 GiB allocated / 47.76 GiB peak"。日志里一个 OOM 都看不到，
# 只表现为同样的活越来越慢：iter1 的 step1 是 480s/步（比硬件实测基线 626.9 tok/s
# 高 15 倍），iter2 恶化到 2759s/步。非单调的振荡也是它——spill 到哪段内存是随机的。
# 阅卷阶段不受影响，因为那阶段不走 logprob 路径。
#
# 别用 expandable_segments:True 绕：这台 WSL2 上直接 CUDA driver error: device not ready。
#
# 顺带把 run_step0.sh 的 solver server 从 0.30 降到 0.20，给训练多留 1 GiB。

python step1_generator.py \
    --model_name_or_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --max_steps $STEPS \
    --save_steps 20 \
    --save_total_limit 3 \
    --logging_steps 1 \
    --run_name "$RUN_NAME" \
    --number_of_generated_data "$N_SPECS" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 1e-4 \
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
