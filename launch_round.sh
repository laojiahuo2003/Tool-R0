#!/bin/bash
# 起一轮 run_round.sh，并把看守器挂上。
#
#  用法: ./launch_round.sh <ITER> [STEPS]
#
# 为什么单独一个文件：run_round.sh 自己要跑两三小时，得放进后台；而看守器
# （watch_step1.sh）要等 round<ITER>.step1.log 出现才能启动，不然它第一句就退出。
# 两件事之间有先后，写在一个脚本里才不会出现"后台任务起了但看守没挂上"。
set -u
ITER=${1:?用法: ./launch_round.sh <ITER> [STEPS]}
STEPS=${2:-50}
cd /home/uos/code/Tool-R0

DRIVER="/tmp/round${ITER}_driver.log"
WATCH="/tmp/watch${ITER}.log"

# ---- 起跑前再查一次：别人的进程不碰，但自己的残留得认出来 ----
U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
if [ -n "$U" ] && [ "$((U))" -gt 2000 ]; then
    echo "!!! GPU 上还有 ${U} MiB。先查清楚是谁占的，不是本仓库起的就别动。" | tee -a "$DRIVER"
    exit 1
fi
# 重复起跑检查用锁文件，不用 pgrep -f。pgrep 的模式匹配有个绕不开的坑：只要启动命令
# 的命令行里碰巧含同样的文本（比如在 shell -c 的包装里），就会把自己算成"已经在跑"，
# 于是永远起不了；而换任何写法都得先猜中这次的外层包装长什么样。锁文件没这个问题。
LOCK="round${ITER}.launch.lock"
if [ -f "$LOCK" ]; then
    OLD=$(cat "$LOCK" 2>/dev/null)
    if [ -n "$OLD" ] && kill -0 "$OLD" 2>/dev/null; then
        echo "!!! round ${ITER} 的启动器还在跑（pid=$OLD），不重复起" | tee -a "$DRIVER"
        exit 1
    fi
    echo "[launch] 发现陈旧锁文件（pid=$OLD 已不在），清掉继续" | tee -a "$DRIVER"
    rm -f "$LOCK"
fi
echo $$ > "$LOCK"
# 注意：锁里最终放的是 run_round.sh 的 PID，不是本脚本的。看守器可能因为超时先退，
# 那时候 run_round.sh 还在训，锁必须依然有效，否则重跑一次会起第二个训练进程。
# 所以不设 EXIT trap 删锁，由 run_round.sh 自己跑完时删（见它末尾的 rm -f）。

echo "[launch] $(date +%H:%M:%S) 起 round ${ITER}（STEPS=${STEPS}），driver 日志 ${DRIVER}" | tee -a "$DRIVER"
# 起跑前先把"这一轮跑的是哪一版代码"钉在磁盘上。round3 期间 run_round.sh 被改过一次
# （mtime 20:33:55）而内容无从恢复，iter2->iter3 的对照里只好永远留一行"一次未知编辑"。
# snapshot_round.sh 缺文件就退出、SHA256 不过就退出，所以它失败时下面不会静默起跑。
if bash snapshot_round.sh "$ITER" >> "$DRIVER" 2>&1; then
    echo "[launch] 代码快照已存 round_snapshots/" | tee -a "$DRIVER"
else
    echo "[launch] !!! 快照失败，仍然起跑（快照是留证不是正确性门），但这一轮的对照会带残留未知项" | tee -a "$DRIVER"
fi
bash run_round.sh "$ITER" "$STEPS" > "$DRIVER" 2>&1 &
RP=$!
echo "$RP" > "$LOCK"
echo "[launch] run_round.sh pid=$RP（已写进 $LOCK）" | tee -a "$DRIVER"

# 等 step1 的日志文件出现（run_round.sh 里 step0 要先起 vLLM，约一两分钟）
for _ in $(seq 1 60); do
    [ -f "round${ITER}.step1.log" ] && break
    kill -0 "$RP" 2>/dev/null || { echo "[launch] !!! run_round.sh 提前退了" | tee -a "$DRIVER"; tail -20 "$DRIVER"; exit 1; }
    sleep 5
done
[ -f "round${ITER}.step1.log" ] || { echo "[launch] !!! 等了 5 分钟也没有 round${ITER}.step1.log" | tee -a "$DRIVER"; tail -20 "$DRIVER"; exit 1; }

echo "[launch] 挂 step1 看守器 -> ${WATCH}" | tee -a "$DRIVER"
# watch_step1.sh 自己会在 step1 结束 / 出事 / 超时时退出
#
# 上限为什么从 200 改成 330：round2 的 step1 实测跑了 194.6 分钟
# （round2.log：16:44:44 step1 起 -> 19:59:21 step1 OK）。200 分钟只剩 5.4 分钟余量，
# round3 只要慢 6 分钟，这里就 exit 5，下面 rc!=0 分支不接 round 看守器——
# round 2 那个"看守空档"就是这么来的（那次是别的原因，但结果一样：
# 训练在跑、我以为有人在看）。看守器晚退没有代价（step1 真的跑完时它靠
# grep "step1 OK" 正常 exit 0），早退纯亏。
bash watch_step1.sh "$ITER" 330 > "$WATCH" 2>&1
RC=$?
echo "[launch] step1 看守器退出 rc=${RC}" | tee -a "$DRIVER"

# rc!=0 说明 step1 看守器自己出事了（崩了/卡死/超时），**但整轮仍然要接着看**。
# round3 实测的教训：watch_step1.sh 因一个竞态（训练进程先退、run_round.sh 后写
# "step1 OK"）误报 exit 2，这里于是不接 round 看守器，step2/step3 全程没人看——
# 是我 23:35 发现 /tmp/watch3_round.log 不存在才手工补挂的。
# 那个竞态已经在 watch_step1.sh 里修掉（完成标记提前查 + 进程消失给宽限期），
# 但这里仍然改成无条件接挂：真崩了 run_round.sh 会 die 退出，watch_round.sh 有
# "run_round.sh 进程不在了 -> exit 2" 的分支，几秒内就会报完，
# 代价为零；而漏挂一次是整轮裸奔，代价极大。两边不对称，所以选永远接挂。
if [ "$RC" -ne 0 ]; then
    echo "[launch] step1 看守器非零退出（rc=${RC}），仍然接 round 看守器。细节看 ${WATCH}" | tee -a "$DRIVER"
fi

# step1 完了接着看守 step2 / step3。这两个阶段没有逐步指标，只看阶段标记和失败签名。
echo "[launch] 挂 round 看守器（step2/step3）" | tee -a "$DRIVER"
bash watch_round.sh "$ITER" 200 > "/tmp/watch${ITER}_round.log" 2>&1
RC=$?
echo "[launch] round 看守器退出 rc=${RC}，run_round.sh 可能还在跑（pid=$RP）" | tee -a "$DRIVER"
exit "$RC"
