#!/usr/bin/env bash
# 每轮起跑前，把"这一轮到底跑的是哪一版代码"钉在磁盘上。
#
#  用法: ./snapshot_round.sh <ITER>
#
# 为什么要这个文件：2026-10-15 round3 期间，run_round.sh 在 20:33:55 被改过一次，
# 改动内容无从恢复，于是 iter2→iter3 的对照里永远留了一行"一次未知编辑"——
# 不是不能跑，是**汇报时必须单独列一行、不能算干净**。这个文件就是让那一行消失。
#
# 为什么单独一个文件而不并进 launch_round.sh：round3 正在跑，
# launch_round.sh / run_round.sh 是被那个 bash 按保存的字节偏移读着的，
# 此刻编辑它们会损坏执行。等 round3 收摊再并进去（ PLAN.md 记了这条前置任务）。
#
# 快照内容：编排脚本 + 四个阶段的启动脚本 + 训练/选题的 python + 奖励函数。
# 这 12 个文件覆盖了两个已识别的混淆源（export 段、难度判据参数）的全部落点。
set -uo pipefail
cd "$(dirname "$0")"
ITER=${1:?用法: ./snapshot_round.sh <ITER>}

SNAP="round_snapshots/round${ITER}.$(date +%Y%m%d_%H%M%S)"
mkdir -p "$SNAP" || exit 1

# 一个都不能少。少一个就退出——半份快照比没有快照更坏，因为它看着像有。
FILES="run_round.sh launch_round.sh run_step0.sh run_step1.sh run_step2.sh run_step3.sh \
step1_generator.py step2_genverify.py step2_select_curriculum.py step3_solver.py \
rewards_generator.py rewards_solver.py \
check_holdout_clean.py"
# check_holdout_clean.py 是 round4 新加进 run_round.sh 的污染门要调的脚本。
# 不把它存进来，还原快照复现 round4 时会得到一个"门在、但门要调的检查器不在"的半份，
# 而那正是这个清单开头说的"少一个就退出"要防的情况。
missing=0
for f in $FILES; do
    if [ ! -f "$f" ]; then echo "!!! 缺 $f"; missing=1; fi
done
[ "$missing" -eq 0 ] || { echo "快照未生成：有文件缺失"; exit 1; }

cp $FILES "$SNAP"/ || exit 1

# 除了内容，还要把"哪些文件此刻正在被改"这件事本身记下来：
# mtime 是事后反推"这一轮跑的是哪一版"的唯一凭据。
: > "$SNAP/MTIMES"
for f in $FILES; do
    printf '%s  %s\n' "$(stat -c '%y' "$f")" "$f" >> "$SNAP/MTIMES"
done

sha256sum $FILES > "$SNAP/SHA256SUMS" || exit 1

echo "快照 -> $SNAP"
echo "  文件数 $(ls "$SNAP" | grep -vcE 'MTIMES|SHA256SUMS')"
echo "  自检："
sha256sum -c "$SNAP/SHA256SUMS" --quiet && echo "    SHA256SUMS 校验通过" || {
    echo "    !!! 校验不过"; exit 1; }
