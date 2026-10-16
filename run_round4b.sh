#!/bin/bash
# round4b：隔离 tau_verify 这一个变量，验证"多轮增益递减是不是它造成的"。
#
#  用法: ./run_round4b.sh          # 必须在 round4 收摊之后手工起
#
# 为什么要单独一个脚本而不是并进 run_round.sh：round4b 不是一轮正常的 iter，
# 它不重训 generator（generator 仍用 iter3 的），solver 仍从 iter3_solver 接。
# 唯一变量是 tau_verify 0.2 -> 0.0。这样"课程变简单了"和"generator 换了"两个混淆
# 都被排除，round4 检验口径修复、round4b 检验难度地板，两条独立证据。
#
# 数据来源：iter3 的原始产出池 iter3_data.json.intermediate.json（1149 条）已在盘上，
# 不需要重新生成，也不需要 GPU 之外的任何新东西。
#
# 绝不自动起：round4 还在跑，两者都要 GPU。起之前先确认 round4.step3.ok 在。
set -u
cd /home/uos/code/Tool-R0
HERE=$PWD
BASE=qwen2.5-0.5b-instruct-tool-r0
TS() { date +%m-%d_%H:%M:%S; }
log() { echo "[$(TS)] $*" | tee -a round4b.log; }
die() { echo "!!! FATAL: $*" >&2; echo "[$(TS)] FATAL: $*" >> round4b.log; exit 1; }

# ---- 前置：round4 必须已经收摊 ----
# 不是形式主义：两个吃卡的活同时跑，会互相把对方挤到 WSL2 主机内存里，
# 而那种失败不报 OOM，只是 step_time 悄悄翻倍——round3 已经为这个付过代价。
[ -f round4.step3.ok ] || die "round4.step3.ok 不在，round4 还没收摊。等它跑完再起这个。"

# ---- 前置：卡必须真的空着 ----
# 2026-10-16 02:48 的教训：原来这里只查 "GPU 显存 < 2000 MiB"。round4 收摊后
# post_round 的 gen 评测正在跑，占 1900 MiB，刚好低于阈值，守卫放行，
# round4b 的 verify 起来和 eval 抢卡。后果不是 OOM，是两边一起变慢
# （gen 评测从 12.5 条/分掉到 2.75 条/分）——正是那个不报错的故障模式。
# 已 kill 掉，但根子是守卫问错了问题。
#
# 正确的问法不是"显存少不少"，是"有没有进程真的占着卡"。
# --query-compute-apps 直接给占卡进程的 PID，不用 pgrep
# （pgrep -f 会命中我自己的 bash -c 包装进程，见 memory pgrep-f-matches-wrapper-not-target）。
APPS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
       | tr -d ' " ' | grep -v '^$' || true)
if [ -n "$APPS" ]; then
    {
        echo "!!! FATAL: GPU 上还有进程在跑，不是空卡。占卡的 PID："
        for p in $APPS; do
            [ -d "/proc/$p" ] && echo "    pid=$p  $(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | cut -c1-100)"
        done
        echo "!!! 等它们收摊再起 round4b。别人的进程不要动。"
    } >&2
    echo "[$(TS)] FATAL: GPU 非空（PID: $(echo $APPS)），拒绝起跑" >> round4b.log
    exit 1
fi
# 兜底：compute-apps 为空但显存仍被占（例如只持有显存没有 compute context 的进程）
FREE=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' || echo 99999)
if [ "$((FREE))" -gt 800 ]; then
    die "GPU 显存还占着 ${FREE} MiB 但没有 compute 进程。先查清楚是什么，别硬起。"
fi
echo "[$(TS)] 前置检查过：round4 已收摊，GPU 空（compute-apps 无进程，显存 ${FREE} MiB）"

export WANDB_MODE=offline
export VLLM_USE_V1=0
export VLLM_DISABLE_COMPILE_CACHE=1
export TRITON_CACHE_DIR="./triton_autotune"; mkdir -p "$TRITON_CACHE_DIR"

RAW="$BASE/iter3_data.json.intermediate.json"
VOUT="$BASE/iter3b_data.json.verified.json"
VREPORT="$BASE/iter3b_data.json.verify_report.txt"
DATA="$BASE/iter3b_data.json"
SOLVER_OUT="$BASE/iter3b_solver"
PREV_SOLVER="$BASE/iter3_solver/checkpoint-50"

[ -f "$RAW" ] || die "找不到 iter3 原始产出池 $RAW"
[ -f "$PREV_SOLVER/model.safetensors" ] || die "找不到 $PREV_SOLVER/model.safetensors"

log "########## round4b 起跑：唯一变量是 tau_verify 0.2 -> 0.0"
log "  原始池     = $RAW（$(.venv/bin/python -c "import json;print(len(json.load(open('$RAW'))))") 条）"
log "  solver 起点 = $PREV_SOLVER（与 iter3 同一起点，不重训 generator）"

# ---- 1) verify：只换 tau，其余与 run_step2.sh 逐字相同 ----
log ">>> 1/6 verify，tau_verify=0.0（k/temp/max_tokens/batch 全部不变）"
.venv/bin/python step2_genverify.py \
    --solver_model "$PREV_SOLVER" \
    --in_intermediate_json "$RAW" \
    --out_intermediate_json "$VOUT" \
    --report_txt "$VREPORT" \
    --k_verify 10 \
    --tau_verify 0.0 \
    --temp_verify 0.7 \
    --max_tokens_verify 1024 \
    --verify_batch_size 32 \
    --tensor_parallel_size 1 > round4b.verify.log 2>&1 \
    || die "verify 失败，细节看 round4b.verify.log"

KEPT=$(.venv/bin/python -c "import json;print(len(json.load(open('$VOUT'))))")
log ">>> verify 留存 $KEPT 条（iter3 同池在 tau=0.2 下是 952 条）"

# 事前判据：tau=0 应该 1149 条全进。少了说明还有别的门，先查再往下走。
[ "$KEPT" -ge 1100 ] || log "!!! 注意：留存 $KEPT 条，预期 ~1149。可能有 early-drop，看 round4b.verify.log"

# ---- 2) select：参数与 run_step2.sh 逐字相同，只换 in_json ----
log ">>> 2/6 选教材（mix 20/50/30、seed 13、drop_agreement_ge 1.0，全部不变）"
.venv/bin/python step2_select_curriculum.py \
    --in_json "$VOUT" \
    --out_json "$DATA" \
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
    --judge_model "$PREV_SOLVER" > round4b.select.log 2>&1 \
    || die "select 失败，细节看 round4b.select.log"

N=$(.venv/bin/python -c "import json;print(len(json.load(open('$DATA'))))")
log ">>> 教材 $N 条"

# ---- 3) 污染门：必须的，不是可选的 ----
# 这批教材来自 iter3 原始池里那 197 条从未进过课程的题，和 iter3 的教材不是同一批。
# holdout 是从更早的池子里抽的，所以**完全有可能撞**。撞了的话 iter3b 的 held-out
# 分数就不是 held-out，而这个实验的全部意义就在那个分数上。
log ">>> 3/6 污染门：教材 vs 固定评测集 holdout_solver.json"
.venv/bin/python check_holdout_clean.py "$DATA"
CRC=$?
if [ "$CRC" -eq 0 ]; then
    log ">>> 污染门过"
elif [ "$CRC" -eq 2 ]; then
    die "$DATA 读不到（rc=2）。这不是撞题，先查路径。"
else
    die "教材和固定评测集撞题（rc=$CRC）。撞了这一轮的数字不能用。"
fi

# ---- 4) 训 solver：从 iter3_solver 接，lr/beta/步数全部与 run_step3.sh 相同 ----
log ">>> 4/6 训 iter3b_solver（从 iter3_solver 接，50 步）"
bash run_step3.sh "$PREV_SOLVER" "$DATA" "$SOLVER_OUT" \
    "qwen2.5-0.5b-instruct-tool-r0-iter3b-solver" 50 \
    > round4b.step3.log 2>&1 || die "step3 失败，细节看 round4b.step3.log"
[ -f "$SOLVER_OUT/checkpoint-50/model.safetensors" ] \
    || die "step3 没产出 $SOLVER_OUT/checkpoint-50/model.safetensors"
log ">>> step3 OK: $SOLVER_OUT/checkpoint-50"

# ---- 5) 评测：只评 solver，generator 没动所以不用重评 ----
# 用 iter3b_solver 这个 tag，paired_test / paired_accpos / channel_split 都按 tag 读
# evals/<tag>.json，所以能直接和 iter3_solver 配对。
log ">>> 5/6 评 iter3b_solver，并与 iter3_solver 配对"
.venv/bin/python eval_iter.py --tag "iter3b_solver" \
    --solver_ckpt "$SOLVER_OUT/checkpoint-50" \
    --note "round4b: tau_verify 0.0 教材，从 iter3_solver 接" \
    > round4b.eval.log 2>&1 || die "eval_iter 失败，细节看 round4b.eval.log"

# 事前判据自动核对。为什么自动跑而不是等我想起来：这六个阈值写在
# PLAN.md:2374-2385，价值全在于"不看着数字重新解释它"。人手跑就会变成
# "我记得的时候才跑"，而这类分析我手算过三遍、前两遍都错。
# 算法已在 iter3 现有数据上验过：教材多调用 1 条 / 3+键 13 条 / sol_acc 0.6627 /
# 多调用点名率 1/7=0.143，与记录一致。退出码 = 未过条数，不阻断后面的配对检验。
log ">>> 6/6 事前判据核对（阈值看到数字之前钉的）"
# PIPESTATUS[0] 不是 $?：管道后 $? 取的是最后一个命令（tee）的退出码，恒 0，
# 那样"未过几条"永远是 0，等于没核对。同一行上不许再用 $?。
.venv/bin/python check_round4b_criteria.py 2>&1 | tee -a round4b.log
CRC=${PIPESTATUS[0]}
log ">>> 事前判据未过 $CRC 条"

# 主判据：条件 acc > 0.7450。为什么不能只看 sol_acc——sol_acc 会被点名率拖着走
# （点名率 0.79->0.84 只靠"敢答"也能涨），条件 acc = mean(acc | name_hit) 才回答
# "点名之后参数对不对"，而那正是 tau_verify 这道门该修的东西。
# 分解式已用 iter2->iter3 实测闭合：+0.0247 = 命中支 +0.0423 + 未中支 -0.0176。
# 退出码 1 = "主判据不过"，是结论不是失败；本脚本只有 set -u 没有 set -e，不会被打断。
.venv/bin/python cond_acc_check.py iter3_solver iter3b_solver 2>&1 | tee -a round4b.log
CAC=${PIPESTATUS[0]}
if [ "$CAC" -eq 2 ]; then
    die "cond_acc_check 自己失败了（数据缺失/题数不等），主判据没判成"
elif [ "$CAC" -eq 0 ]; then
    log ">>> 主判据过：iter3b 条件 acc > 0.7450"
else
    log ">>> 主判据不过：iter3b 条件 acc <= 0.7450（iter3 基线 0.7315）"
fi

# 配对检验只评 solver 那三项（generator 没动，重评是浪费 30 分钟 gen 评测）。
# 指标名与 eval_round.sh 逐字一致：fmt / acc / name_hit。
for m in fmt acc name_hit; do
    echo "===== iter3_solver -> iter3b_solver  指标 $m =====" | tee -a round4b.log
    .venv/bin/python paired_test.py --a iter3_solver --b iter3b_solver \
        --field solver --metric "$m" 2>&1 | tee -a round4b.log
    echo | tee -a round4b.log
done
.venv/bin/python paired_accpos.py --a iter3_solver --b iter3b_solver 2>&1 | tee -a round4b.log
.venv/bin/python channel_split.py --a iter3_solver --b iter3b_solver 2>&1 | tee -a round4b.log

log "########## round4b 完成"
log "  判据见 PLAN.md「推荐的验证实验：round 4b」那张表"
log "  汇报前跑：.venv/bin/python report_funnel.py 3 4；.venv/bin/python compare_rounds.py"
