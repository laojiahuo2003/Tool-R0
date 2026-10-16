#!/bin/bash
# post_round 4 的心跳看守。v2 设计（同 watch_round4.sh）：
# 每 HEARTBEAT 秒无条件打一行，静默 = 看守器自己有问题，不是任务没问题。
#
# 覆盖的终态不只是成功：post_round.sh 死掉、eval 崩、污染门不过、判据不过，都要喊。
cd /home/uos/code/Tool-R0
HEARTBEAT=${HEARTBEAT:-300}
PID=${PID:-1040667}
LOG=/tmp/eval_iter4.log
STATE=/tmp/post4_watch.state
: > "$STATE"

while true; do
    ts=$(date +%H:%M:%S)
    alive=否
    [ -d /proc/$PID ] && alive=是
    # eval_iter4.log 的大小和最后一条 [eval_round] 标记
    sz=0; mark="-"
    if [ -f "$LOG" ]; then
        sz=$(stat -c %s "$LOG" 2>/dev/null || echo 0)
        mark=$(grep -a '\[eval_round\]' "$LOG" 2>/dev/null | tail -1 | cut -c1-60)
    fi
    gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
    # gen 评测进度：eval_iter 打的是 "[gen 24/192] four_tag=.. pv_ok=.."，
    # 不是 tqdm，所以两条都抓。
    gen=$(grep -ao '\[gen [0-9]*/[0-9]*\][^|]*' "$LOG" 2>/dev/null | tail -1)
    sol=$(grep -ao '\[sol [0-9]*/[0-9]*\][^|]*' "$LOG" 2>/dev/null | tail -1)

    # 坏行：post_round 输出 + eval 日志；逐文件 grep，不 cat（避免末行粘连）。
    # 两个词不能图省事：
    #   '重题' 会命中成功行 "0 重题，rc=0"，'污染' 会命中分节标题 "########## 2/5 污染门：..."
    # 所以用 '有重题' / '撞题' / '读不到'，并排除 # 开头的标题行和 tqdm 表格线 │。
    for f in /tmp/post_round4.log "$LOG"; do
        [ -f "$f" ] || continue
        while IFS= read -r line; do
            [ -z "$line" ] && continue
            grep -qxF "$line" "$STATE" 2>/dev/null && continue
            echo "$line" >> "$STATE"
            echo "!!! 坏行[$f]: $line"
        done < <(grep -aE 'Traceback|CUDA out of memory|Killed|FATAL|!!!|有重题|撞题|读不到（rc=2）|失败' "$f" 2>/dev/null \
                 | grep -av '^#\|│' | tail -4)
    done

    echo "[$ts] 心跳 post_round=$alive GPU=${gpu}MiB eval日志=${sz}B | $mark ${gen:+| $gen}${sol:+ | $sol}"

    # 收摊标记
    if grep -aq '收摊完成：iter4 vs iter3' /tmp/post_round4.log 2>/dev/null; then
        echo "[$ts] post_round 收摊，看守结束"; break
    fi
    if [ "$alive" = "否" ]; then
        echo "[$ts] !!! post_round 死了但没收摊标记——查 /tmp/post_round4.log 尾部"
        break
    fi
    sleep "$HEARTBEAT"
done
