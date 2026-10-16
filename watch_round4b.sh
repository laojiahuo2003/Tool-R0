#!/bin/bash
# round4b 的心跳看守。设计要点（每个都是踩过的坑）：
#   1. 每 HEARTBEAT 秒无条件发一行——纯 grep 看守分不清"没坏行"和"我死了"，
#      round4 为此假监控 3 小时（memory: watcher-death-is-indistinguishable-from-quiet）。
#   2. 逐文件 grep，不用 cat——cat 会把没有结尾换行的文件和下一份粘成一行，
#      我自己的 dry-run 就抓到过这个（round4b.guardtest.log 没有尾换行）。
#   3. 坏行过滤 '│'——train.log 里嵌的工具菜单 ASCII 表列名叫 Exception/Error:
#      （memory: tool-r0-log-grep-false-positives）。
#   4. '重题'/'污染' 这种词不能用：会命中成功行 "0 重题，rc=0" 和
#      分节标题 "########## 2/5 污染门：..."。用 '有重题'/'撞题'。
#   5. 只认 round4b.log 里的 "round4b 完成"，不靠进程退出——进程可能被外部杀掉。
set -u
cd "$(dirname "$0")"
PID="${PID:-1047895}"
HEARTBEAT="${HEARTBEAT:-300}"
STATE=/tmp/round4b_watch.state
[ -f "$STATE" ] || : > "$STATE"

TS() { date '+%H:%M:%S'; }

scan_bad() {
    for f in round4b.log round4b.verify.log round4b.select.log round4b.step3.log round4b.eval.log; do
        [ -f "$f" ] || continue
        while IFS= read -r line; do
            [ -z "$line" ] && continue
            grep -qxF "$line" "$STATE" 2>/dev/null && continue
            echo "$line" >> "$STATE"
            echo "!!! 坏行[$f]: $line"
        done < <(grep -aE 'Traceback|CUDA out of memory|Killed|FATAL|!!!|有重题|撞题|读不到（rc=2）|失败' "$f" 2>/dev/null \
                 | grep -av '^#\|│' | tail -4)
    done
}

while true; do
    alive=是; ls /proc/$PID >/dev/null 2>&1 || alive=否
    vf=$(grep -aoE '\[verify\] processing [0-9]+-[0-9]+/[0-9]+' round4b.verify.log 2>/dev/null | tail -1 | grep -oE '[0-9]+-[0-9]+/[0-9]+')
    sf=$(grep -aoE '\[[0-9]+/[0-9]+\]' round4b.step3.log 2>/dev/null | tail -1)
    ef=$(grep -aoE '\[sol [0-9]+/[0-9]+\]|\[gen [0-9]+/[0-9]+\]' round4b.eval.log 2>/dev/null | tail -1)
    echo "[$(TS)] 心跳 driver=$alive verify=$vf step3=$sf eval=$ef | $(tail -1 round4b.log 2>/dev/null | cut -c1-70)"
    scan_bad
    if grep -aq 'round4b 完成' round4b.log 2>/dev/null; then
        echo "[$(TS)] round4b.log 出现 'round4b 完成'，看守结束"
        break
    fi
    if [ "$alive" = "否" ]; then
        echo "[$(TS)] driver 进程 $PID 不在了，看守结束（上面没有 'round4b 完成' 就是没跑完）"
        break
    fi
    sleep "$HEARTBEAT"
done
