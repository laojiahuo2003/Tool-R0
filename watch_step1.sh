#!/bin/bash
# 看守某一轮的 step1 训练：每分钟报一次进度，出事立刻喊。
#
#  用法: ./watch_step1.sh <ITER> [超时分钟数，默认 200]
#
# 为什么不用 tail -f | grep：step 记录是跟在 tqdm 进度条同一行后面的，grep 单行匹配
# 抓不到；而且崩掉的签名（Traceback / FATAL / CUDA error）和正常的 step 记录混在
# 一起，需要分别处理。这里每分钟解析一次，两种信号都报。
#
# 覆盖的失败签名（缺一个就等于"崩了但看守没喊"）：
#   Traceback / CUDA out of memory / CUDA error / RuntimeError / FATAL /
#   step1 failed / killed / 进程消失
set -u
ITER=${1:?用法: ./watch_step1.sh <ITER> [分钟]}
MAXMIN=${2:-200}
# 进程消失后的宽限期（秒）。要覆盖 merge + model.safetensors 校验这段，
# 实测 round3 这段约 2-10 秒，但慢的时候（大 checkpoint / 磁盘忙）可能到一分钟以上。
# 取 180 是因为"多等 3 分钟"的代价远小于"误判导致 step2/step3 没人看"。
GRACE=${GRACE:-180}
LOG="round${ITER}.step1.log"
CTL="round${ITER}.log"

[ -f "$LOG" ] || { echo "没有 $LOG"; exit 1; }

START=$(date +%s)
DEADLINE=$((START + MAXMIN * 60))
LAST_N=-1
STALL=0

echo "[watch] 开始看守 $LOG（上限 ${MAXMIN} 分钟）"
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
    # 已完成 round 的标记。**必须放在进程检查之前**——
    # run_round.sh 的顺序是：训练进程退出 -> merge LoRA -> 校验 model.safetensors
    # -> 才写 "step1 OK"。中间有几十秒到一两分钟，进程已经没了但标记还没落盘。
    # round3 实测就在这里误判：23:35:44 报"进程不在了"exit 2，
    # 而 "step1 OK" 23:35:46 才写。launch_round.sh 见 rc!=0 就不接 round 看守器，
    # step2/step3 于是没人看（这次是我手工补挂的，见 PLAN.md 23:35 那条）。
    if grep -q "step1 OK" "$CTL" 2>/dev/null; then
        echo "[watch] step1 完成"
        exit 0
    fi

    # 训练进程还在不在
    if ! pgrep -f "step1_generator.py" > /dev/null 2>/dev/null; then
        # 进程不在了不等于失败，先给宽限期等完成标记（理由同上一段）。
        echo "[watch] step1_generator.py 进程不在了，等 ${GRACE}s 看是否只是收尾中"
        for _ in $(seq 1 $((GRACE / 5))); do
            grep -q "step1 OK" "$CTL" 2>/dev/null && {
                echo "[watch] step1 完成（宽限期内）"; exit 0; }
            sleep 5
        done
        echo "[watch] !!! 宽限期 ${GRACE}s 过完仍无 step1 OK，判定为训练进程消失"
        tail -25 "$LOG" | grep -v '│' | tail -20
        exit 2
    fi

    # 失败签名。裸词 "Traceback" / "Killed" 会被 generator 吐出来的题面撞上
    # （题里带 Python 代码），所以 Traceback 必须连 Python 的头部一起匹配，
    # Killed 用 ^ 锚成独立一行（OOM killer 的写法）。grep -v '│' 是另一回事：
    # train.log 里嵌的工具菜单 ASCII 表列名叫 Exception/Error:，不过滤会把正常记录误判成崩。
    BAD=$(grep -a -E "Traceback \(most recent call last\)|CUDA out of memory|CUDA error|!!! FATAL:|step1 failed|^Killed" "$LOG" 2>/dev/null | grep -v '│' | tail -3)
    if [ -n "$BAD" ]; then
        echo "[watch] !!! 发现失败签名:"
        echo "$BAD"
        exit 3
    fi

    # 解析 step 记录报进度
    LINE=$(.venv/bin/python - "$LOG" <<'PY'
import ast, re, sys
txt = open(sys.argv[1], errors="replace").read()
recs = []
for m in re.finditer(r"\{'loss'.*?\}(?=\s|$)", txt, re.S):
    try:
        d = ast.literal_eval(m.group(0).replace("\n", " "))
    except Exception:
        continue
    if isinstance(d, dict) and "grad_norm" in d:
        recs.append(d)
if not recs:
    print("0 - - - - - -")
    sys.exit()
r = recs[-1]
st = [x.get("step_time") for x in recs if isinstance(x.get("step_time"), (int, float))]
med = sorted(st)[len(st) // 2] if st else -1
# step_time 的最小/最大也要：spill 的特征是同一份配置下步时来回大幅振荡
# （iter1 中位 484s，iter2 同样配置跑到 2759s，配置没变，变的是 spill 到哪段内存）。
print("%d %.0f %.0f %.0f %.4f %.4f %.4f" % (len(recs), med,
      min(st) if st else -1, max(st) if st else -1,
      r.get("reward", -1),
      r.get("rewards/reward_validity_accuracy/mean", -1),
      r.get("completions/mean_length", -1)))
PY
)
    N=$(echo "$LINE" | awk '{print $1}')
    if [ "$N" = "$LAST_N" ]; then
        STALL=$((STALL + 1))
    else
        STALL=0
        LAST_N=$N
    fi
    # 连续 N 分钟步数不长，说明卡住了。
    #
    # 阈值不能按"步数没变多少分钟"单条件判：iter1 单步最长 635 秒，round2 第 3 步
    # 实测 426 秒，而慢是正常的（生成长度主导步时）。原来写 6 分钟，等于在最坏单步
    # 之内就喊卡死——round2 step3 跑到 7 分钟时误报过一次，把看守链整个断了。
    # 真卡死的特征是 GPU 也停了：慢步骤在生成，util 30-95%；死锁/挂住 util 掉到 0。
    # 所以要"步数 12 分钟没动" **且** "GPU 利用率低于 10%"两个条件同时成立。
    # （日志大小不能当判据：实测一步之内日志完全不增长，tqdm 只在步边界刷。）
    if [ "$STALL" -ge 12 ]; then
        UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
        if [ -n "$UTIL" ] && [ "$UTIL" -lt 10 ] 2>/dev/null; then
            echo "[watch] !!! 步数停在 $N 已 12 分钟且 GPU 利用率 ${UTIL}% —— 真卡死"
            nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
            exit 4
        fi
        echo "[watch] 注意：步数停在 $N 已 12 分钟，但 GPU 利用率 ${UTIL}%（在算，只是这一步特别慢）"
        STALL=0
    fi

    # WSL2 spill 探针：训练进程的 RSS。GPU 只有 10 GiB，PyTorch 报 56 GiB 那轮就是
    # spill 到宿主内存——日志里一个 OOM 都没有，只看步时会把"配置错了"和"本来就慢"
    # 混在一起。0.5B+LoRA+激活的正常 RSS 在 4-8 GiB，冲过 15 GiB 就是在 spill。
    #
    # 选 PID 必须按可执行文件认，不能图省事用 pgrep -f "step1_generator.py" | head -1：
    # 那个模式会同时命中包住它的 bash -c 包装进程（包装进程的命令行里含着同样的文本），
    # 而 head -1 取到的是 PID 更小的那个，往往是包装进程——它的 RSS 只有几 MB，
    # 于是这个探针永远读 0，spill 中止条件等于没写。实测踩过（round 2 起跑时）。
    RSS="-"
    for p in $(pgrep -f "step1_generator.py" 2>/dev/null); do
        exe=$(readlink /proc/$p/exe 2>/dev/null || echo "")
        case "$exe" in
            *python*) RSS=$(awk '/^VmRSS/{print int($2/1048576)}' /proc/$p/status 2>/dev/null || echo "-"); break ;;
        esac
    done

    echo "[watch] $(date +%H:%M:%S) 已完成 $N 步  step_time中位 $(echo "$LINE" | awk '{print $2}')s  [min $(echo "$LINE" | awk '{print $3}')s max $(echo "$LINE" | awk '{print $4}')s]  RSS ${RSS}GiB  reward $(echo "$LINE" | awk '{print $5}')  validity $(echo "$LINE" | awk '{print $6}')"

    # 心跳。watch_live.sh 靠它判断"这个看守器还活着"——不能用 pgrep 认进程名，
    # 那个模式会命中命令行里含同样文本的任何进程（包括别人写的探查命令），
    # 实测会让存活检查永远返回真。心跳文件没这个歧义：停了就是死了。
    # round 2 为此瞎过 3 小时：watch_step1.sh 16:59 被旧版单条件卡死判据误杀，
    # 之后 /tmp/watch2.log 再也不长，而 watch_live.sh 只在有新行时才搬东西，
    # 于是"看守器死了"和"这一分钟没事发生"输出完全一样。
    touch "/tmp/watch${ITER}.hb"

    # 中止判据：光慢不算证据（generation 占大头，本来就慢），必须是 spill 签名——
    # 步时超过 400s 且宿主内存被吃进去。两个条件同时满足才停，避免把正常慢步骤误杀。
    MED=$(echo "$LINE" | awk '{print $2}')
    if [ "$N" -ge 5 ] && [ "$MED" != "-" ] && [ "${MED%.*}" -gt 400 ] 2>/dev/null \
       && [ "$RSS" != "-" ] && [ "$RSS" -gt 15 ]; then
        echo "[watch] !!! 步时中位 ${MED}s 且 RSS ${RSS}GiB —— WSL2 spill 签名，B=1 没修好"
        echo "[watch]     处置：先停，不带着疑问跑下一轮"
        exit 6
    fi
    sleep 60
done

echo "[watch] 超过 ${MAXMIN} 分钟还没结束，超时退出"
exit 5
