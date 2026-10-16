#!/usr/bin/env bash
# round 2 收摊后的固定顺序。顺序在 PLAN.md 里是事前写死的，不要随手换、不要跳步。
#
# 为什么要这个脚本而不是到时候手敲：这里有一步会覆盖 evals/iter1_solver.json
# （设种重测），而覆盖之前必须先确认未设种的那份已经备份。手敲容易在备份之前
# 就先跑了重测，把唯一的未设种证据冲掉——那个备份是"未设种/设种差多少"的唯一凭据。
set -uo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
BASE=qwen2.5-0.5b-instruct-tool-r0
ITER=2

say() { echo; echo "########## $* ##########"; }

# ---- 前置：round 2 必须真的跑完了 ----
if [ ! -f "round${ITER}.step3.ok" ]; then
    echo "!!! round${ITER}.step3.ok 不在，round 2 还没收摊。现在跑会去抢正在训练的显存。"
    exit 1
fi

say "1/6 记漏斗（纯 CPU，读 step2 的 verify 报告）"
bash record_funnel.sh "$ITER" 50 || echo "  [警告] record_funnel 没成功，不影响后面的评测"

say "2/6 未设种的 iter1_solver 备份"
# 只在没有备份时才做。绝不用 cp 无条件覆盖：一旦第 3 步跑过，evals/iter1_solver.json
# 就是设种后的值，再 cp 一遍会把未设种那份冲掉，而且没有第二份。
if [ -f evals/iter1_solver_unseeded.json ]; then
    echo "  备份已存在（$(stat -c%y evals/iter1_solver_unseeded.json | cut -c1-19)），不覆盖"
    echo "  条数：$($PY -c 'import json;print(len(json.load(open("evals/iter1_solver_unseeded.json"))))')"
else
    if [ -f evals/iter1_solver.json ]; then
        cp evals/iter1_solver.json evals/iter1_solver_unseeded.json
        echo "  已备份 evals/iter1_solver.json -> evals/iter1_solver_unseeded.json"
    else
        echo "  [警告] 两份都不在，跳过备份（配对检验仍然能跑，只是少了未设种的对照）"
    fi
fi

say "3/6 设种重测 iter1_solver（eval_round.sh 的配对检验拿它当 a 格）"
# 注意：不能写 SOLVER_SEED=xxx python eval_iter.py——那个变量没人读，是空操作。
# 设种之所以生效是因为 eval_iter.py 的 eval_solver 里有 random.seed + torch.manual_seed。
"$PY" eval_iter.py --tag iter1_solver \
    --solver_ckpt "$BASE/iter1_solver_rl/checkpoint-50" \
    --note "设种重测（代码里的 SOLVER_SEED=20261016）" || exit 1

say "4/6 同 ckpt 同种子再测一遍 = 评测自身噪声底"
# 这一步不是走过场：iter1_solver 和 iter1_solver_rep2 的差就是评测噪声。
# 比这个差还小的"上升"一律不算上升。配对检验的 p 值要对着它看。
"$PY" eval_iter.py --tag iter1_solver_rep2 \
    --solver_ckpt "$BASE/iter1_solver_rl/checkpoint-50" \
    --note "同 ckpt 同种子再测一遍，量评测自身噪声底" || exit 1

say "5/6 设种重测 base_solver（把账本补全）"
"$PY" eval_iter.py --tag base_solver \
    --solver_ckpt Qwen/Qwen2.5-0.5B-Instruct \
    --note "设种重测" || exit 1

say "6/6 查 iter2 教材有没有和 holdout 重题（污染门）"
"$PY" check_holdout_clean.py "$BASE/iter${ITER}_data.json"
rc=$?
if [ $rc -ne 0 ]; then
    echo "  !!! 有重题，rc=$rc。先查清楚再跑 eval_round.sh——否则 sol_* 不是 held-out"
    exit 1
fi
echo "  0 重题，rc=0"

say "评测 iter2 并和 iter1 配对检验"
bash eval_round.sh "$ITER" 1 50

say "收摊完成。接下来：修 run_round.sh 的 kill_server 守卫 + 清 port 5000，再跑 round 3"
echo "  （守卫为什么必须修、为什么不能现在修，见 PLAN.md「round 3 起跑前必须先做两件事」）"
