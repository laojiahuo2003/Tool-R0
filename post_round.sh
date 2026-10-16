#!/usr/bin/env bash
# 一轮收摊后的固定顺序（通用版，round 3 起用）。
#
#  用法: ./post_round.sh <ITER> [PREV]        PREV 默认 ITER-1
#
# 和 post_round2.sh 的区别：那个里有三步是 round 2 专属的，round 3 起不再需要——
#   1. 备份未设种的 iter1_solver  —— 只有 iter1 有过"补种子之前测的数"
#   2. 设种重测 iter1_solver      —— 同上，且它已经是 iter2 的 a 格了
#   3. iter1_solver_rep2 量噪声底 —— 已量过，结论是 0（PLAN.md「评测完全可复现」）
# 这三个在 round 2 跑完就一次性做完了，重做只会覆盖掉唯一的未设种留证。
#
# 保留的部分：漏斗、污染门、评测、配对检验。污染门放在最前面是因为
# 它不过的话后面所有 sol_* 都不是 held-out，评了也白评。
set -uo pipefail
cd "$(dirname "$0")"
ITER=${1:?用法: ./post_round.sh <ITER> [PREV]}
PREV=${2:-$((ITER - 1))}
STEPS=50
BASE=qwen2.5-0.5b-instruct-tool-r0
DATA="$BASE/iter${ITER}_data.json"

say() { echo; echo "########## $* ##########"; }

# ---- 前置：这一轮必须真的跑完了 ----
# round${ITER}.step3.ok 不在就说明训练还在进行，现在跑评测会去抢显存。
if [ ! -f "round${ITER}.step3.ok" ]; then
    echo "!!! round${ITER}.step3.ok 不在，round ${ITER} 还没收摊。现在跑会去抢正在训练的显存。"
    exit 1
fi

say "1/5 记漏斗（纯 CPU，读 step2 的 verify 报告）"
bash record_funnel.sh "$ITER" "$STEPS" || echo "  [警告] record_funnel 没成功，不影响后面的评测"

say "1.5/5 round4 事前判据核对（阈值是看到数字之前钉的，不许赛后改）"
# 为什么自动跑：这四个阈值写在 PLAN.md「所以 round4 的事前判据（现在钉住，不许赛后改）」，
# 价值全在于"不看着数字重新解释它"。人手跑就会变成"我记得的时候才跑"，
# 而这类分析我手算过三遍、前两遍都错。退出码 = 未过判据条数，非 0 不阻断后面的评测
# ——判据不过是诊断，不是门；但它必须被看见，所以打出来。
.venv/bin/python check_round4_criteria.py "$ITER" || echo "  [警告] 有事前判据未过，见上。不阻断评测，但汇报时必须连这一起说"

say "2/5 污染门：iter${ITER} 教材 vs 固定评测集"
# eval_round.sh 自己也会查一遍，但这里先查：不过就停，省得白等 30 分钟 gen 评测。
.venv/bin/python check_holdout_clean.py "$DATA"
rc=$?
if [ $rc -eq 2 ]; then
    # 读不到和撞题是两种故障。run_round.sh 里的门已经按退出码分开报过了，
    # 这里对齐同一套说法——否则一个路径打错的故障会被说成"教材污染了评测集"，
    # 而那时候 solver 已经训完，人会被指去查重题，问题却在路径上。
    echo "  !!! $DATA 读不到（rc=2）。这不是撞题，先查路径。"
    exit 1
fi
if [ $rc -ne 0 ]; then
    echo "  !!! 有重题，rc=$rc。先查清楚再评——不然 sol_* 不是 held-out"
    exit 1
fi
echo "  0 重题，rc=0"

say "3/5 评测 iter${ITER}，并和 iter${PREV} 配对检验"
# eval_round.sh 内部：gen 评测（~30min）+ solver（~2min）+ 5 项配对检验 + compare_rounds
bash eval_round.sh "$ITER" "$PREV" "$STEPS"

say "4/5 补跑 sol_acc_pos（eval_round.sh 的 5 项里没有它）"
# 为什么单独补：acc_pos 是"拿到任意部分分的比例"，比 acc（奖励均值）更接近
# "这题到底会不会做"，而且 PLAN 的事前口径里它是独立一行。
# eval_round.sh 写死了 5 个指标，加它就得改那个脚本——不改，在这里补跑更安全。
#
# 为什么用 paired_accpos.py 而不是 paired_test.py --metric acc_pos：
# 逐条 json 里**没有 acc_pos 这个字段**（只有 acc），那么写会两边都报 0.0000、
# p=1.0000「与 0 分不开」，看着像正常检验其实是 0 比 0。已实测踩过。
# 而 paired_test.py --metric acc 那行打的是连续均值，McNemar 分支里算出来的
# acc_pos 在分档分支里被丢掉没打印——所以只能单独算。
.venv/bin/python paired_accpos.py --a "iter${PREV}_solver" --b "iter${ITER}_solver"

say "5/5 分通道拆解：增益来自点名还是参数（round 3 起的事前判据）"
# 为什么要自动跑：PLAN.md「round 3 的分通道预期」那条事前判据是
# "name_hit 涨幅该 < iter2 的 +0.2148，acc 若还在涨应主要来自条件 acc"。
# 不自动跑就会变成"我记得的时候才跑"，而这类分析我手算过三遍、
# 前两遍都错（分解式不成立 + 把 0.50~0.64 读成没有档位效应）。
# 脚本自带闭合自检：命中支+未中支必须等于 acc、两组 n 必须加起来等于总数，
# 不过就 exit 1，不许带着可疑数字汇报。
.venv/bin/python channel_split.py --a "iter${PREV}_solver" --b "iter${ITER}_solver"

say "收摊完成：iter${ITER} vs iter${PREV}。判据见 PLAN.md「round 3 起跑前」那一节"
echo "  汇报前先跑：.venv/bin/python compare_rounds.py"
