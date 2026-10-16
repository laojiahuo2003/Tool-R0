#!/bin/bash
# step2 漏斗双温对照：同一个 generator checkpoint、同一份 1500 个候选，只改 --temp_verify。
#
#   用法: ./run_step2_funnel.sh <GEN_CKPT> <SOLVER_NAME> <TAG> [TEMP_VERIFY...]
#         TEMP_VERIFY 默认 "0.7 0.001"
#
# 为什么共享 step2_gen 的产物：temp_verify 只作用于 step2_genverify.py（估计
# p_success 的 solver 采样温度），对 step2_gen.py 没有任何影响。所以候选池可以只
# 生成一次，两次 verify 吃同一份输入——漏斗差异就能完全归因到 temp_verify，
# 不用混入生成采样的方差。原来 run_step2.sh 是三个阶段串成一条，
# 每改一个参数就要把 1500 次生成重跑一遍。
#
# 已知限制：step2_genverify.py 没有 --seed，SamplingParams 也不设种子，
# 所以 temp=0.7 那一轮是单次随机抽样、不可复现；temp=0.001 接近确定性。
# 对比时这一条要算进结论里。step2_select_curriculum.py 有 --seed 13，是确定的。

set -u
VENV="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.venv"
[ -f "$VENV/bin/activate" ] && source "$VENV/bin/activate"

GEN_CKPT=$1
SOLVER_NAME=$2
TAG=$3
shift 3
TEMPS=("$@")
[ ${#TEMPS[@]} -eq 0 ] && TEMPS=(0.7 0.001)

export CUDA_VISIBLE_DEVICES=0

SHARED="/tmp/step2_${TAG}.intermediate.json"

echo "########## step2_gen（只跑一次，候选池共享）"
if [ -f "$SHARED" ]; then
    echo "  复用已有 $SHARED"
else
    python step2_gen.py \
        --generator_model "$GEN_CKPT" \
        --out_intermediate_json "$SHARED" \
        --n_generate 1500 \
        --max_tokens_gen 4096 \
        --tensor_parallel_size 1 || exit 1
fi

for TV in "${TEMPS[@]}"; do
    OUT="iter1_data_${TAG}_tv${TV}.json"
    VERIFIED="${OUT}.verified.json"
    REPORT="${OUT}.verify_report.txt"
    echo ""
    echo "########## temp_verify=${TV}  -> ${OUT}"

    python step2_genverify.py \
        --solver_model "$SOLVER_NAME" \
        --in_intermediate_json "$SHARED" \
        --out_intermediate_json "$VERIFIED" \
        --report_txt "$REPORT" \
        --k_verify 10 \
        --tau_verify 0.20 \
        --temp_verify "$TV" \
        --max_tokens_verify 1024 \
        --verify_batch_size 32 \
        --tensor_parallel_size 1 || { echo "verify 失败 tv=$TV"; continue; }

    python step2_select_curriculum.py \
        --judge_model "$SOLVER_NAME" \
        --in_json "$VERIFIED" \
        --out_json "$OUT" \
        --n_final 100 \
        --tensor_parallel_size 1 \
        --gpu_memory_utilization 0.90 \
        --max_model_len 4096 \
        --batch_size 32 \
        --temp_judge 0.0 \
        --max_tokens_judge 10 \
        --mix_easy 0.20 \
        --mix_medium 0.50 \
        --mix_hard 0.30 \
        --seed 13 \
        --default_diff medium \
        --drop_agreement_ge 1.0 \
        --difficulty_source agreement \
        --judge_model "$SOLVER_NAME" || { echo "select 失败 tv=$TV"; continue; }
# --difficulty_source agreement（2026-10-15 起的默认）：难度用 verify 关实测的
# solver_agreement 分档，不再让 LLM judge 猜。老路实测 584/584 全判 hard，
# --mix_* 等于没写；新路在 584 条非饱和池上是 hard 14.2% / medium 44.9% / easy 40.9%，
# 且这一步不加载模型。--judge_model 只在加 --difficulty_source judge 做对照时才会被读。
# --drop_agreement_ge 1.0：剔掉 solver_agreement=1.0 的样本（对 step3 零梯度），
# 理由与实测数字见 run_step2.sh 里同一行的注释。
done

echo ""
echo "########## 全部完成"
