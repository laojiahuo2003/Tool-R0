#!/bin/bash
# round4 看守器 v2：自带心跳。
#
# v1 为什么不能用：grep 模式看着该命中（02:22:14 round4.log 就写了 "step1 OK"），
# 结果 30 分钟零事件。只转发"匹配行"的看守器分不清"没匹配"和"自己死了"——
# round2 已经为这个假监控过 3 小时。
#
# v2 的修法：每 HEARTBEAT 秒无条件打一行状态（run_round 活不活 / 哪些 ok / 进度到哪），
# 再加一个"新出现的坏行"通道（计数文件去重）。静默超过一个心跳周期 = 看守器自己有问题。
#
# v2 干跑时抓到的两个自己的 bug，别再犯：
#   - `cat round4*.log` 会把 round4b.log 也 glob 进来，里面 02:01 那条 FATAL 是
#     测 run_round4b.sh 守卫时的残留，会被当成 round4 崩了。已改名 round4b.guardtest.log。
#   - cat 拼文件时末行没换行会把两行粘成一条，所以改成逐文件 grep，不 cat。
cd /home/uos/code/Tool-R0
HEARTBEAT=${HEARTBEAT:-300}
STATE=/tmp/round4_watch.state
: > "$STATE"

while true; do
    ts=$(date +%H:%M:%S)
    ok=""
    for f in round4.step1.ok round4.step2.ok round4.step3.ok; do
        [ -f "$f" ] && ok="$ok ${f#round4.}"
    done
    alive=否
    [ -d /proc/1003176 ] && alive=是
    prog=$(tr '\r' '\n' < round4.step2.log 2>/dev/null \
           | grep -ao 'processing [0-9]*-[0-9]*/[0-9]*' | tail -1)
    s3=-
    if [ -f round4.step3.log ]; then
        s3=$(tr '\r' '\n' < round4.step3.log 2>/dev/null \
             | grep -ao '[0-9]*/50 \[[0-9:]*<[0-9:]*' | tail -1)
    fi
    gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')

    # 坏行：逐文件 grep（不 cat，避免末行粘连），只扫 round4 自己的三个日志。
    # '│' 过滤掉 tqdm/工具菜单 ASCII 表里列名叫 Exception/Error 的误报。
    for f in round4.log round4.step1.log round4.step2.log round4.step3.log; do
        [ -f "$f" ] || continue
        while IFS= read -r line; do
            [ -z "$line" ] && continue
            grep -qxF "$line" "$STATE" 2>/dev/null && continue
            echo "$line" >> "$STATE"
            echo "!!! 坏行[$f]: $line"
        done < <(grep -aE 'Traceback|CUDA out of memory|Killed|FATAL|step[123] failed|有重题' "$f" 2>/dev/null \
                 | grep -av '│' | tail -5)
    done

    echo "[$ts] 心跳 run_round=$alive ok:${ok:-无} GPU=${gpu}MiB ${prog:+| verify $prog} | step3 ${s3:-无}"

    [ -f round4.step3.ok ] && { echo "[$ts] round4.step3.ok 出现，round4 收摊，看守结束"; break; }
    # run_round 死了且没有 step3.ok = 这一轮没跑完，必须喊
    if [ "$alive" = "否" ] && [ ! -f round4.step3.ok ]; then
        echo "[$ts] !!! run_round 死了但 round4.step3.ok 不在——round4 没跑完"
        break
    fi
    sleep "$HEARTBEAT"
done
