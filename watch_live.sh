#!/bin/bash
# 把一轮的看守器输出和 round<ITER>.log 的新增行搬到眼前，跑到本轮结束为止。
#
#  用法: ./watch_live.sh <ITER> [总分钟数，默认 260]
#
# 为什么不用 tail -f：这一轮要跑两三小时，超过 Monitor 单次 30 分钟上限，得能反复重新武装；
# 而且 tail -f 不会自己结束，本轮跑完了它还挂着。这里轮询 + "上次看到第几行"去重，
# 既不漏也不重，结束条件就是 run_round.sh 写的 round<ITER>.status。
#
# 覆盖三类信号，缺一类就等于"出事但没人喊"：
#   1. 阶段推进（>>> / OK / DONE）
#   2. 看守器自己的结论（watch] 开头的行：步时、RSS、spill 签名、卡死、超时）
#   3. 崩溃签名 —— 不看看守器脸色，直接扫 round<ITER>.step1.log / .step2.log / .step3.log，
#      免得"看守器自己卡住"和"训练卡住"分不清
set -u
ITER=${1:?用法: ./watch_live.sh <ITER> [分钟]}
MAXMIN=${2:-260}
cd /home/uos/code/Tool-R0

S1="/tmp/watch${ITER}.log"
SR="/tmp/watch${ITER}_round.log"
HB="/tmp/watch${ITER}.hb"
CTL="round${ITER}.log"
STATUS="round${ITER}.status"

KEEP='watch\]|>>>|完成|ROUND_|OK|DONE|失败|Traceback \(most recent call last\)|CUDA out of memory|CUDA error|!!! FATAL:|step[123] failed|^Killed|spill|卡死|超时|非零退出|提前退'

declare -A SEEN=()
declare -A LASTLINE=()
declare -A CRASHDONE=()
WDEAD=""          # 看守器死亡只报一次，否则每分钟重播
END=$(( $(date +%s) + MAXMIN * 60 ))

# 起跑时先把"现在已经有多少行"记下来，之后只搬新增的。
# 不这么做的话，Monitor 单次只有 30 分钟上限、每半小时就得重新武装一次，而每次重挂
# 都会把从 16:44 到现在整段历史重播一遍——watch_step1.sh 每分钟写一行，重挂一次就是
# 几十条旧进度涌进来，真正该看的（步数变了、RSS 变了、出事了）直接被埋掉。
# 崩溃签名不依赖这个去重（它扫全文件），所以漏不了出事。
for f in "$S1" "$SR" "$CTL"; do
    [ -f "$f" ] && SEEN[$f]=$(wc -l < "$f" 2>/dev/null || echo 0)
done

while [ "$(date +%s)" -lt "$END" ]; do
    # 结束条件以 run_round.sh 自己写的状态文件为准
    if [ -f "$STATUS" ] && grep -q "ROUND_${ITER}_DONE" "$STATUS" 2>/dev/null; then
        echo "[live] ROUND_${ITER} 全部结束 $(date +%H:%M:%S)"
        exit 0
    fi

    # 看守器自己死没死。这一步是必需的：这个脚本是"搬运行看守器的输出"，而运行中的
    # 看守器退出后 /tmp/watch<ITER>.log 就再也不长了——那时上面的搬运循环一条新行都没有，
    # 看起来和"这一分钟没事发生"一模一样。round 2 实测踩过：watch_step1.sh 16:59 被旧版
    # 单条件卡死判据误杀（两条件修复当时还没落盘），之后 3 小时零事件，我以为一直在看，
    # 其实早就瞎了。所以这里独立查一遍。
    #
    # 用心跳文件而不是 pgrep 认进程名：`pgrep -f "watch_step1.sh 2"` 会命中任何命令行里
    # 含这串文本的进程——包括我自己这条探查命令（实测命中过自己的 bash -c 包装，于是永远
    # 返回"活着"）。要 /proc 逐个剥 argv 才靠谱，不值得。心跳文件没这个歧义：
    # watch_step1.sh 每分钟 touch 一次，停了就是死了。
    if [ -z "$WDEAD" ]; then
        if [ -f "round${ITER}.step1.ok" ]; then
            # step1 完了，理论上 launch_round.sh 该接上 watch_round.sh
            if [ ! -f "/tmp/watch${ITER}_round.log" ] \
               || [ $(( $(date +%s) - $(stat -c %Y "/tmp/watch${ITER}_round.log" 2>/dev/null || echo 0) )) -gt 300 ]; then
                WDEAD=1
                echo "[live] !!! step1 已完成但 watch_round.sh 没有心跳——step2/step3 现在没有看守器"
                echo "[live]     处置：nohup setsid bash watch_round.sh ${ITER} 200 >> /tmp/watch${ITER}_round.log 2>&1 &"
            fi
        elif [ -f "$HB" ]; then
            AGE=$(( $(date +%s) - $(stat -c %Y "$HB" 2>/dev/null || echo 0) ))
            if [ "$AGE" -gt 300 ]; then
                WDEAD=1
                echo "[live] !!! watch_step1.sh 心跳停了 ${AGE}s（它每分钟写一次）——step 级看守（步时/RSS/reward）已断"
                echo "[live]     训练不受影响（run_round.sh 自己管），但看不了步级信号了"
                # 330 和 launch_round.sh 里的上限保持一致：round2 step1 实测 194.6 分钟，
                # 200 分钟只余 5.4 分钟，round3 慢一点就会超时退出、丢掉后面的看守链。
                echo "[live]     处置：nohup setsid bash watch_step1.sh ${ITER} 330 >> ${S1} 2>&1 &"
                tail -5 "$S1" | grep -v '│'
            fi
        fi
    fi

    for f in "$S1" "$SR" "$CTL"; do
        [ -f "$f" ] || continue
        n=$(wc -l < "$f" 2>/dev/null || echo 0)
        last=${SEEN[$f]:-0}
        [ "$n" -gt "$last" ] || continue
        new=$(tail -n +$((last + 1)) "$f" 2>/dev/null | head -n $((n - last)))
        SEEN[$f]=$n
        # watch_step1.sh 每分钟都写一行，步数没变时除了时间戳其余一模一样。原样搬过来
        # 就是一分钟一条重复通知，把真正该看的（步数变了、RSS 变了、出事了）淹掉。
        # 所以比对前先把 "[watch] HH:MM:SS " 这个前缀去掉，剩下的才算内容签名。
        # （直接比整行没用：时间戳每行都不同，等于永远不重复。）
        while IFS= read -r line; do
            keep=$(echo "$line" | grep -aE "$KEEP" | grep -av '│')
            [ -z "$keep" ] && continue
            sig=$(echo "$keep" | sed 's/^\(\[watch\]\) [0-9][0-9]:[0-9][0-9]:[0-9][0-9] /\1 /')
            [ "$sig" = "${LASTLINE[$f]:-}" ] && continue
            LASTLINE[$f]=$sig
            echo "[$(basename "$f" .log)] $keep"
        done <<< "$new"
    done

    # 崩溃签名单独扫：三个阶段的日志直接看，不依赖看守器还活着。
    # 每个日志只报一次——崩了之后这一轮不会写 round<ITER>.status，watch_live.sh 会一直
    # 转到 MAXMIN，不记"报过了"就是每 30 秒重播一遍同一个 Traceback。
    for st in step1 step2 step3; do
        L="round${ITER}.${st}.log"
        [ -f "$L" ] || continue
        [ -n "${CRASHDONE[$L]:-}" ] && continue
        B=$(grep -aE "Traceback \(most recent call last\)|CUDA out of memory|CUDA error|!!! FATAL:|step[123] failed|^Killed" "$L" 2>/dev/null | grep -av '│' | tail -2)
        [ -z "$B" ] && continue
        CRASHDONE[$L]=1
        echo "[live] !!! ${st} 出现失败签名:"$'\n'"$B"
    done

    sleep 30
done

echo "[live] ${MAXMIN} 分钟到了本轮仍未结束，需要重新武装"
exit 5
