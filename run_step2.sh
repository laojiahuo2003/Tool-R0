#!/bin/bash

# Activate this repo's venv explicitly. Bare `python` otherwise resolves to
# whatever venv the calling shell has active (search-zero/.venv on this box),
# which has no vllm and a torch too old for TRL.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.venv/bin/activate"

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
    --judge_model "$SOLVER_NAME" \
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
    --default_diff medium
