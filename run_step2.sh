#!/bin/bash

# Activate this repo's venv if it exists (see run_main.sh for why it is optional).
VENV="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.venv"
[ -f "$VENV/bin/activate" ] && source "$VENV/bin/activate"

GEN_CKPT=$1
SOLVER_NAME=$2
OUT_JSON=$3

# Single GPU: all three stages share the one card, and they run sequentially as
# separate processes, so each releases its memory before the next starts.
export CUDA_VISIBLE_DEVICES=0

INTERMEDIATE_JSON="${OUT_JSON}.intermediate.json"
VERIFIED_JSON="${OUT_JSON}.intermediate.verified.json"
VERIFY_REPORT="${OUT_JSON}.verify_report.txt"

python step2_gen.py \
    --generator_model "$GEN_CKPT" \
    --out_intermediate_json "$INTERMEDIATE_JSON" \
    --n_generate 1500 \
    --max_tokens_gen 4096 \
    --tensor_parallel_size 1


python step2_genverify.py \
    --solver_model "$SOLVER_NAME" \
    --in_intermediate_json "$INTERMEDIATE_JSON" \
    --out_intermediate_json "$VERIFIED_JSON" \
    --report_txt "$VERIFY_REPORT" \
    --k_verify 10 \
    --tau_verify 0.20 \
    --temp_verify 0.7 \
    --max_tokens_verify 1024 \
    --verify_batch_size 32 \
    --tensor_parallel_size 1


python step2_select_curriculum.py \
    --in_json "$VERIFIED_JSON" \
    --out_json "$OUT_JSON" \
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
    --judge_model "$SOLVER_NAME"
# 2026-10-15：难度判据从 LLM judge 换成了 verify 关实测的 solver_agreement 分档。
#   老行为（--difficulty_source judge）实测是坏的：同一个 811 条只换 --judge_model，
#   base 判出 583 hard + 1 easy，换成 step3 训过的 solver 之后 584/584 全 hard，
#   fallback_used 0/584、judge_raw 逐条就是字面的 'hard'。也就是说上面
#   --mix_easy .20 --mix_medium .50 --mix_hard .30 实际执行成 100% hard，
#   分档逻辑等于没写。LLM judge 想猜的东西我们每轮都直接测了，没必要再猜一遍。
#   新默认值 agreement 分档：hard <0.4 / medium [0.4,0.8) / easy >=0.8，
#   584 条非饱和池上是 14.2% / 44.9% / 40.9%，且这一步不再加载模型，零 GPU 成本。
# --judge_model 只在想跑对照时才有用（加 --difficulty_source judge），留着不删是为了
#   一键 A/B；agreement 这条路不会去读它。
# --drop_agreement_ge 1.0 剔掉 solver_agreement=1.0（k_verify 次全对）的样本。GRPO 按组算
#   advantage，组内奖励没有方差就没有梯度，这种样本训了白训还占掉 n_final 的名额。
#   实测 verify 留存 563→811 里 227 条（28%）是全对的；不过滤最终 100 条教材有 27 条零梯度。
#   想要旧行为传 1.01。它跑在分档之前，所以 easy 档实际是 [0.8, 1.0)。
