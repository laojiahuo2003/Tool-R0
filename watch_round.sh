#!/bin/bash
# 看守一整轮：step1 之后接着看 step2 / step3。每分钟报一次状态。
#
#  用法: ./watch_round.sh <ITER> [上限分钟，默认 200]
#
# 与 watch_step1.sh 的分工：那个盯的是 step1 每步的 step_time（用来判断 WSL2 spill
# 有没有修好，是最早能发现问题的信号）；这个盯的是"这一轮整体走到哪了、有没有崩"。
# step2 / step3 没有逐步指标可看，只能看阶段标记和进程存活。
#
# 覆盖的失败签名：Traceback / CUDA out of memory / CUDA error / FATAL /
#   stepN failed / Killed / 进程消失 / 阶段标记长时间不动
set -u
ITER=${1:?用法: ./watch_round.sh <ITER> [分钟]}
MAXMIN=${2:-200}
cd /home/uos/code/Tool-R0

CTL="round${ITER}.log"
STAGES="step1 step2 step3"

[ -f "$CTL" ] || { echo "没有 $CTL"; exit 1; }

START=$(date +%s)
DEADLINE=$((START + MAXMIN * 60))
LAST_LINES=-1
STALL=0
ANNOUNCED=""

echo "[round-watch] 开始看守 round ${ITER}（上限 ${MAXMIN} 分钟）"

# 阶段完成标记（round<ITER>.<stage>.ok）的数量，0->3
done_stages() {
    local n=0 s
    for s in $STAGES; do [ -f "round${ITER}.${s}.ok" ] && n=$((n + 1)); done
    echo "$n"
}

while [ "$(date +%s)" -lt "$DEADLINE" ]; do
    # 整轮完成
    if [ -f "round${ITER}.status" ]; then
        echo "[round-watch] round ${ITER} 完成："
        tail -8 "$CTL"
        exit 0
    fi

    # 失败签名。模式要写得"只有真崩才会命中"，不能图省事用裸词：
    #   * "Traceback" 单独一个词会被生成内容撞上（generator 吐的题里带 Python 代码），
    #     必须连头部一起匹配 "Traceback (most recent call last)"。
    #   * "Killed" 同理，OOM killer 打的是独立一行，用 ^Killed 锚住。
    #   * grep -v '│' 必须加：train.log 里嵌的工具菜单 ASCII 表列名叫 Exception/Error:，
    #     不加会把正常的训练记录误判成崩溃（这个误报踩过一次）。
    BAD=$(grep -a -h -E "Traceback \(most recent call last\)|CUDA out of memory|CUDA error|!!! FATAL:|step[123] failed|^Killed" \
            "$CTL" "round${ITER}.step1.log" "round${ITER}.step2.log" "round${ITER}.step3.log" \
            2>/dev/null | grep -v '│' | tail -3)
    if [ -n "$BAD" ]; then
        echo "[round-watch] !!! 失败签名:"
        echo "$BAD"
        exit 3
    fi

    # 驱动进程还在不在
    if ! pgrep -f "run_round.sh ${ITER}\b" > /dev/null 2>&1; then
        echo "[round-watch] !!! run_round.sh ${ITER} 进程不在了"
        tail -15 "$CTL"
        exit 2
    fi

    N=$(wc -l < "$CTL")
    if [ "$N" = "$LAST_LINES" ]; then
        STALL=$((STALL + 1))
    else
        STALL=0
        LAST_LINES=$N
    fi
    # 15 分钟控制台不长，说明这个阶段可能卡死了。
    #
    # 阈值不能按"控制台多少分钟没新行"单条件判——step2 的 verify 阶段整段不往
    # round<ITER>.log 写东西（iter1 实测 verify 1079 条 × k=10 花了 857s，round2 通过
    # 格式的候选更多、只会更久），step3 的 50 步同样只在步边界刷一行。所以"15 分钟
    # 没新行"是这两个阶段的**正常状态**，拿它当卡死等于必报。
    # 而且 watch_round.sh 一退出（exit 4）就没人接着看这一轮了，误报一次等于后半夜
    # 全程裸奔——和 watch_step1.sh 里那个坑是同一个，那边已经踩过一次。
    #
    # 真卡死的特征是 GPU 也停了：verify 在跑推理、step3 在生成，util 都在 30-95%；
    # 死锁/挂住 util 掉到 0。所以要求"15 分钟没新行" **且** "GPU 利用率 < 10%"同时成立。
    # 阶段日志的 mtime 不能当判据：step3 的 tqdm 只在步边界刷，一步之内日志完全不增长。
    if [ "$STALL" -ge 15 ]; then
        UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
        if [ -n "$UTIL" ] && [ "$UTIL" -lt 10 ] 2>/dev/null; then
            echo "[round-watch] !!! 控制台 15 分钟没有新行（停在 $N 行）且 GPU 利用率 ${UTIL}% —— 真卡死"
            nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
            for s in $STAGES; do
                f="round${ITER}.${s}.log"
                [ -f "$f" ] && echo "--- $f 末尾 ---" && tail -3 "$f" | grep -v '│'
            done
            exit 4
        fi
        echo "[round-watch] 注意：控制台 15 分钟没有新行（停在 $N 行），但 GPU 利用率 ${UTIL}%（在算，verify/step3 本来就不刷控制台）"
        STALL=0
    fi

    # 新完成的阶段要喊一声（只喊一次）
    for s in $STAGES; do
        if [ -f "round${ITER}.${s}.ok" ] && [ "${ANNOUNCED#*$s}" = "$ANNOUNCED" ]; then
            echo "[round-watch] $(date +%H:%M:%S) stage $s 完成"
            ANNOUNCED="$ANNOUNCED $s"
        fi
    done

    echo "[round-watch] $(date +%H:%M:%S) 已完成 $(done_stages)/3 阶段  控制台 $N 行"
    sleep 60
done

echo "[round-watch] 超过 ${MAXMIN} 分钟还没结束，超时退出"
exit 5
