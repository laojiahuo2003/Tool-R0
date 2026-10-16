#!/bin/bash
# round4 收摊后的一次性分析汇总。post_round.sh 已经跑了漏斗/判据/评测/配对/分通道，
# 这里补四件它不跑的，并按固定顺序摆好，免得汇报时漏项或顺序乱了读错。
#
#  用法: ./report_round4.sh        # 只在 round4.step3.ok + evals/iter4_solver.json 都在之后跑
set -u
cd "$(dirname "$0")"

[ -f evals/iter4_solver.json ] || { echo "!!! evals/iter4_solver.json 不在，评测还没跑完"; exit 1; }

echo "################################################################"
echo "# 1/5 漏斗 + 事前判据（阈值看到数字之前钉的）"
echo "################################################################"
.venv/bin/python report_funnel.py 2 3 4
echo
.venv/bin/python check_round4_criteria.py 4
echo "   （退出码 = 未过判据条数；2 条不过是我的判据定错了，见 PLAN.md 02:39 那节）"

echo
echo "################################################################"
echo "# 2/5 step3 训练信号跨轮（零方差占比 = 教材信息量）"
echo "################################################################"
.venv/bin/python step3_signal.py 2 3 4

echo
echo "################################################################"
echo "# 3/5 held-out 主判据：条件 acc > 0.7450（不是 sol_acc）"
echo #   sol_acc 会被点名率拖着走；条件 acc 才回答"点名之后参数对不对"。
echo #   分解式已用 iter2->iter3 实测闭合：+0.0247 = 命中支 +0.0423 + 未中支 -0.0176
echo "################################################################"
for pair in "base_solver iter1_solver" "iter1_solver iter2_solver" \
            "iter2_solver iter3_solver" "iter3_solver iter4_solver"; do
    .venv/bin/python cond_acc_check.py $pair
    echo "   （退出码 $? = 主判据是否过）"
    echo
done

echo
echo "################################################################"
echo "# 4/5 多调用子集（可收复失地最大的单块）"
echo "################################################################"
.venv/bin/python multi_call_check.py base_solver iter1_solver iter2_solver iter3_solver iter4_solver

echo
echo "################################################################"
echo "# 5/5 跨轮总表"
echo "################################################################"
.venv/bin/python compare_rounds.py
