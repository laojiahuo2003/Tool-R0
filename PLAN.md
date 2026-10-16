# 多轮自举：计划与验收标准

目标（用户原话）：**能够复现这一套，是正确的，多轮是可以有用的。**

拆成三件可判真伪的事：

| # | 目标 | 怎么判 |
|---|---|---|
| 1 | 复现整套 | step0→1→2→3 四步各产出真实 artifact，无人工干预跑完一轮 |
| 2 | 是正确的 | 每轮产出的教材配比真的是 20/50/30、solver 真的在学（位移+奖励双证据）、指标是在**固定评测集**上量的，不是换题换出来的 |
| 3 | 多轮有用 | 同一列指标，iter3 ≥ iter2 ≥ iter1，且不是噪声 |

## 固定评测条件（`eval_iter.py` 顶部常量，改这里等于历史分数全部作废）

- **generator**：192 条 spec，`random.seed(20261015)`，temp 0.8，max_new 2048，左 padding
  指标 `gen_four_tag` / `gen_pv_ok`。prompt 实测 566-576 token，不会截断。
- **solver**：`holdout_solver.json` 135 条，temp 0.7，max_new 1024，`seed=20261016`
  指标 `sol_fmt` / `sol_acc` / `sol_acc_pos` / `sol_name_hit`，后三项就是 step3 的训练奖励。
- 评测集来源与排除规则见 `build_holdout.py` 顶部注释（iter1 的 step3 只训过 tv0.7 那份）。

**已实测验证（不是注释里说说）**：`check_holdout_clean.py` 逐条比对，
iter1 step3 真正用的 `iter1_data_lr1e4_ckpt50_tv0.7.json`（100 条）与 holdout **0 重题**，
所以 iter1 的 sol_* 分数是货真价实的 held-out。同源的
`..._tv0.001.json` 有 24 条重题——那是 holdout 的来源池，iter1 没在它上面训过，不算污染。
每一轮出完教材都要重跑这个检查，`eval_round.sh` 里已经接成门。

**2026-10-15 补的坑**：`eval_solver` 原先没设种，temp 0.7 采样解码下同一个 checkpoint
每跑一遍分数都不一样（generator 那边一直有种子）。已在 `eval_solver` 开头补
`random.seed` + `torch.manual_seed`（常量 `SOLVER_SEED=20261016`，第 44 行）。
代价：`iter1_solver` 和 `base_solver` 那两行是**未设种时**测的，要重测一次才对得上
（solver 评测只要 86s/232s，便宜）。重测前不要引用旧的 sol_* 数值。
注意这个种子是**代码里的常量不是环境变量**，"设种重测"就是用当前代码原样重跑，
写 `SOLVER_SEED=xxx python eval_iter.py` 是空操作。

### 固定评测集还有多少上升空间（按 iter1_solver 逐条结果分档实测）

这决定了"多轮上升"该看哪一列，先量出来，免得到 iter4 把没空间误判成没学会：

| 指标 | 已打满（该档 1.000）的题数 | 各档现状 | 结论 |
|---|---|---|---|
| `sol_fmt` | **36/135 = 27%**（全是 agreement ≥0.7） | 0.81 ~ 1.00 | 理论上限约 0.98。涨到 0.95+ 就该换指标看，不是没学会 |
| `sol_name_hit` | 0 | 0.44 ~ 0.90 | 每条档都有空间 |
| `sol_acc`（accuracy_reward 均值） | 0 | 0.31 ~ 0.63 | 每条档都有空间 |

所以多轮的 solver 信号主要来自 `acc` / `name_hit` 两列，`sol_fmt` 当 sanity check 看。
（分档数字是小样本：agreement 0.9 那档 n=7，acc_reward 0.31 比 0.8 档的 0.63 还低，
这是噪声不是规律，别逐档解释。）

## iter1 基线（已量，`iter1_step1_baseline_summary.json`）

generator step1，50 步，6.8 小时：

| 指标 | 前8步 | 后8步 | 斜率/步 | t |
|---|---|---|---|---|
| reward | 0.7790 | 1.2622 | +0.0113 | **+5.60** |
| reward_validity_accuracy | 0.1797 | 0.3859 | +0.0052 | +5.42 |
| reward_format_accuracy | 0.5534 | 0.7643 | +0.0048 | +5.22 |
| reward_curriculum | 0.0459 | 0.1120 | +0.0014 | +3.18 |
| completions/mean_length | 1942 | 1263 | -15.5 | -7.28 |
| step_time 中位 | 484s | | | |

这轮是有效学习，不是噪声。iter2 的 generator 要从 `reward≈1.26 / validity≈0.39` 这个起点继续往上。

## round2 step1 跑完了（20:00 收工，3.2 小时），和 iter1 逐行对照

先说最要紧的一条：**round2 的起点 reward 1.2083 ≈ iter1 的终点 1.2622**（差 0.054，就是
"接着上一轮的 checkpoint 继续训"该有的样子），然后它自己又涨了 **+0.4049** 到 1.6132，
t=4.30。iter1 从 0.7790 涨到 1.2622 是 +0.4832。两轮加起来 +0.834。
**"多轮有没有用"在训练日志这一层就已经看得到了**，不用等评测。

| 指标 | iter1 前8 | iter1 后8 | iter1 斜率 | iter1 t | r2 前8 | r2 后8 | r2 斜率 | r2 t |
|---|---|---|---|---|---|---|---|---|
| reward | 0.7790 | 1.2622 | +0.0113 | +5.60 | 1.2083 | 1.6132 | +0.0091 | **+4.30** |
| reward_validity_accuracy | 0.1797 | 0.3859 | +0.0052 | +5.42 | 0.3391 | 0.5453 | +0.0044 | **+4.36** |
| reward_format_accuracy | 0.5534 | 0.7643 | +0.0048 | +5.22 | 0.7533 | 0.8600 | +0.0029 | **+3.66** |
| reward_curriculum | 0.0459 | 0.1120 | +0.0014 | +3.18 | 0.1160 | 0.2079 | +0.0018 | **+2.76** |
| completions/mean_length | 1942 | 1263 | −15.5 | −7.28 | 1292 | 1039 | −5.14 | −2.84 |
| completions/clipped_ratio | 0.1406 | 0.0078 | −0.0029 | −5.02 | 0.0156 | **0.0000** | −0.0003 | −1.25 |
| kl | 0.0016 | 0.0068 | +0.00012 | +9.13 | 0.0027 | 0.0066 | +0.00007 | +3.92 |
| step_time 中位 | 484s | | | | **234s** | | | |

**三个可判定的分量仍然全部显著**（t = 4.36 / 3.66 / 2.76），所以 generator 没有到顶——
我跑之前怀疑过 plateau，这条怀疑被数据否掉了。斜率比 iter1 小是**起点高、剩余空间小**，
不是学不动：round2 起点已经站在 iter1 终点的位置上。

**但有一个要提前说的信号**：`clipped_ratio` 在 round2 起点就只有 0.0156、终点已经是 0.0000。
iter1 的 validity 涨幅有很大一块来自"学会写短、不再被 4096 截顶"（mean_length 1942→1263、
clipped 0.14→0.008）——这个机制在 round2 已经用完了。round2 的 validity 还涨了 +0.206，
说明有别的来源，但**round3 不该再指望这条**，斜率大概率继续变小。这不是失败，是自举的
正常形态：每一轮的绝对水平更高，可涨的越来越少。

> **22:22 更正：上面这段的"0.0156 → 0.0000"是 OLS 拟合端点，不是实测值，而且 t=-1.25 不显著。**
> 实测 round2 的 `completions/clipped_ratio`：50 步里只有 **7 步非零**（步 3/4/15/16/17/37/42），
> 取值**全是 0.0625**，按 10 步窗口是 `0.0125 / 0.01875 / 0.0000 / 0.00625 / 0.00625`——
> **末窗是 0.00625，不是 0**，且没有单调趋势。
> 我重算 OLS：斜率 −0.000267/步、t=−1.25、拟合端点 步1=0.0153 / 步50=0.0022。
> **"从 0.0156 掉到 0.0000"这个说法把一条不显著的拟合线读成了实测现象**，
> 还在它上面建了"截顶机制用完"的因果故事。
> 结论层面（round3 斜率大概率继续变小）靠的是"起点更高、剩余空间更小"，**不靠截顶机制**，
> 所以结论仍可用，但机制那句要作废。详见下面 22:22 那条第三处自我更正。

step_time 从 484s 掉到 234s（中位），总耗时 6.8h → 3.2h，是 `--per_device_train_batch_size 1
--gradient_accumulation_steps 16` 那一版修复的功劳——**这是"能多轮跑"的前提**，
不然一轮一天，跑不了三圈。

### 训练日志之外的第一个跨轮证据：generator 的出题良率

step2 都是 `--n_generate 1500`，而 `step2_gen` 会自己把解析不出来的丢掉再写中间文件：

| | step2_gen 收到 | 写进 intermediate | 良率 |
|---|---|---|---|
| iter1 | 1500 | 1079 | 71.9% |
| round2 | 1500 | **1118** | **74.5%** |

+39 条（+3.6%）。这个数不经过任何评测集、不掺 solver，就是同一个生成器在同样批量下的
产出质量，方向和 step1 训练日志的 format/validity 上升一致。
（`step2_genverify` 那行 `early_drop=0` 是说它自己没再早丢，不是 1118 的来历——
1118 是 step2_gen 过滤后的结果。）

### round2 step2 漏斗（20:23:28 收工）

| | step2_gen 产出 | verify 留存 | 教材 | 难度配比 |
|---|---|---|---|---|
| iter1 | 1079/1500（71.9%） | 563（52.2%） | 100 | hard 99 / easy 1（judge 退化） |
| round2 | **1118/1500（74.5%）** | **918（82.1%）** | 100 | **easy 21 / medium 49 / hard 30** |

`TOTAL: 1118 / KEPT: 918 / DROPPED: 200`，select 实际配比 easy 21% / medium 49% / hard 30%
（目标 20/50/30），**难度分档第一次不是退化的**——iter1 那 99:1 就是换成 agreement 的原因。

**留存率从 52.2% 跳到 82.1% 不许读成"题出得更好"**。`record_funnel.sh` 的注释自己写明了这个
混淆：step2 的 verify 用的是**上一轮**的 solver，所以留存率衡量的是上一轮 solver 的判别力。
iter1 的 verify 用 base solver（`sol_fmt` 0.2163），round2 用 iter1 训过的（0.8689）；
而 iter1 那 68% 的丢弃**本来就是 `missing_tag`**（solver 不输出工具调用块）。
solver 学会输出格式了，missing_tag 归零，留存率就上去了。这是同一件事的两面，
不是 generator 的进步。**generator 侧的跨轮证据只有良率那一行（71.9%→74.5%）。**

### 未设种 vs 设种：同一份 checkpoint 差 12 题（20:40 实测，这就是为什么必须重测）

设种重测 `iter1_solver` 之前，账本里已经有一条 `iter1_solver`（85.7s，20:03 之前测的，
那会儿 `eval_solver` 还没补 `random.seed`）。两条放在一起，**同一个 checkpoint、
同一套 135 题、只是换了采样种子**：

| 指标 | 未设种 | 设种 | 差 | 折合题数 |
|---|---|---|---|---|
| `sol_fmt` | 0.8689 | 0.9007 | +0.0319 | +4.3 |
| `sol_acc` | 0.4453 | 0.4782 | +0.0329 | +4.4 |
| `sol_acc_pos` | 0.6370 | 0.6889 | +0.0519 | +7.0 |
| `sol_name_hit` | 0.6667 | 0.5778 | **-0.0889** | **-12.0** |

逐条看，同题两次采样给出不同结果的比例：`fmt` 32/135（24%）、`acc` 75/135（56%）、
`name_hit` 48/135（36%）。

**两个结论**：

1. **未设种那批数一个都不能用**，包括之前 PLAN 里引用过的 `base_solver` 0.2163 和
   iter1 的 0.8689。误差量级（name_hit 上 12 题）比绝大多数预期效应都大——
   拿它当 a 格，iter2 随便怎么动都"显著"。
2. `name_hit` 的抖动（-12 题）远大于 `fmt`/`acc`（+4 题）。`fmt` 和 `acc` 是
   整个输出的综合判定，倾向于把采样噪声平均掉；`name_hit` 是"工具名有没有精确对上"
   这种单点判定，一抽风就翻。所以 **`name_hit` 是最不该用未设种数据比的指标**。

（另：这两个数现在都有留证。`evals/iter1_solver_unseeded.json` 是 19:23 备份的未设种版，
`evals/iter1_solver.json` 是 20:34 覆盖后的设种版。这份对照**不可再造**——
备份只有一份，而 `post_round2.sh` 第 2 步写的是"没有备份才 cp"，所以它不会被后来的
重测冲掉。）

**设种之后噪声是 0，不是"很小"**（见下面「评测完全可复现」）。

## reward 是怎么加起来的（归因口径，2026-10-15 实测）

先说一个坑：`rewards_generator.py` 顶部写着 `W_FORMAT=0.25 / W_GOLD_VALID=0.25 / W_DIFFICULTY=0.5`，
看着像"难度权重是格式的 2 倍"。**这三个常量在整个仓库里没有任何引用，是死代码。**
用最近 6 步解三元线性方程组，实测权重是 **1.0 : 1.0 : 1.0**——

```
round2 step38: fmt 0.8229  val 0.4625  cur 0.1172
  三项之和 1.4026 == reward 1.4026
```

即 `reward = reward_format_accuracy + reward_validity_accuracy + reward_curriculum`，
三项等权相加。按 0.25/0.25/0.5 算是 0.3799，和日志差 3.7 倍。
**所以看日志别按 W_* 想比例，三项各占三分之一票。**

### 三项里哪一项能跨轮比

| 组件 | 子奖励 | 调 solver？ | 能跨轮比？ |
|---|---|---|---|
| `reward_format_accuracy` | tags_parsed / tools_valid_json / gold_valid_json | 否（纯静态解析） | **能** |
| `reward_validity_accuracy` | gold_tool_exists / gold_args_valid / gold_values_valid | 否（纯静态解析） | **能** |
| `reward_curriculum` | difficulty_raw + semantic_coherence | **是**（8 处 / 3 处） | **不能** |

curriculum 不能跨轮比的原因是结构性的：`reward_difficulty_raw` 是
`difficulty_reward_bandpass(p_success)`（p∈[0.25,0.75] 给满分，σ=0.12 高斯外扩），
而 `p_success` 是**当场调 port 5000 上的 solver 采 8 次量出来的**。每轮的 solver 都不一样
（run_main.sh:102 iter1 的 step1 host 的是 **BASE 未训**，round2 host 的是 iter1 训过的），
p_success 的分布就不同。这是自举的设计不是 bug，但意味着 **curriculum 不是"题目越好越高"的
单调指标，是"还瞄不瞄得准当前 solver 能力前沿"的跟踪指标**——solver 越强，generator 越要
重新对中，完全跟住的话这一列是平的而不是涨的。

### round2 的增益拆开（round2 步21-30 vs iter1 步43-50，即 PLAN 里的"后8"）

| 组件 | iter1 后8 | r2 步21-30 | Δ | 占总增益 | 可跨轮比 |
|---|---|---|---|---|---|
| format | 0.7643 | 0.8161 | +0.0518 | 26.5% | 能 |
| validity | 0.3859 | 0.4775 | +0.0916 | 46.9% | 能 |
| curriculum | 0.1120 | 0.1639 | +0.0520 | 26.6% | **不能** |
| **reward** | **1.2622** | **1.4576** | **+0.1953** | 100% | |
| completions/mean_length | 1262.7 | 977.0 | −285.6 | | |

=> **+0.1953 里只有 +0.1434（73.4%）是可跨轮比的内在增益**，另外 26.6% 要归到
"换了 solver"这个结构性因素上，不能算 generator 的本事。

**没有混淆的版本是轮内自己比**（同轮同 solver 同教材，步1-10 → 步21-30）：
reward +0.2256 = format +0.0661 + validity +0.1125 + curriculum +0.0469。
三个分量同向、量级和跨轮一致，机制也一致（mean_length 1222→977，写得短就不被截顶）。

结论：**round2 的 generator 提升是真的，但报告里必须写"内在增益 +0.1434（format+validity）"，
不能拿 +0.1953 说事。**

## 跑之前的改动（都已落地）

1. `run_step1.sh`：`per_device_train_batch_size 4×accum4 → 1×accum16`。
   优化器步完全等价（还是 16 条/步、每 16 条反传一次）。修的是 WSL2 静默 spill：
   TRL 的 logprob+entropy 路径物化 `[16,1200,151936]` logits，10 GiB 卡上实测峰值
   12.62 GiB，而 solver server 占 3 GiB——必超，且 WSL2 不报 OOM 只变慢（484s/步
   就是比硬件基线 626.9 tok/s 慢 15 倍）。B=1 时峰值 4.23 GiB，不溢。
2. `run_step3.sh`：补 `--beta 0.01 --weight_decay 0.01`（论文 Table 4，step1 一直传，
   step3 漏了）。βKL 把策略锚在上一轮 solver 上，没有它每轮可能漂走。
3. `run_step0.sh`：solver server `0.30 → 0.35`（KV cache 0.62→1.12 GiB，并发 6.6→12×），并写 `vllm.pid`。0.20 是错值：vLLM 的地板是 0.238，见 run_step0.sh 里的实测账。
4. `run_round.sh`（新）：按 ITER 推路径的通用轮次执行器，全程日志 `round<ITER>.log`，
   分阶段 `.ok` 标记可续跑，收摊按 PID 杀不用 `pkill -f`（会误杀别人的 vLLM）。

**代价说清楚**：iter1→iter2 不是干净对照（难度判据已换 + 这两个补丁）。

### 第三个混淆：**作废**。round2 也是 V0，和 round3 一样（22:15 用 round2.step2.log 推翻 21:34 的结论）

> 21:34 我在这节写的是"round2 跑 V1、round3 起跑 V0，所以第一个干净对照要推到 iter3→iter4"。
> **那个结论是错的，22:15 作废。** 错在哪：我拿 `run_iter2.sh` 当成了 round2 的驱动脚本，
> 而它**从来没驱动过 round2**。round2 其实是 `run_round.sh` 驱动的，那一版早就带 `VLLM_USE_V1=0`。

**怎么发现自己错的**：想核一下"round2 到底是哪个脚本跑的"，
于是拿 `round2.log` 里的文案去 grep 两个脚本：

```
$ grep -c "round .* 起跑（STEPS=" run_round.sh run_iter2.sh
run_round.sh:1
run_iter2.sh:0
```

`round2.log` 里 `>>> step2: 出题 + 阅卷 + 选教材（verify 用上一轮 solver，难度按 agreement 分档）`
这句**只存在于 `run_round.sh`**。所以 round2 是 `run_round.sh` 跑的，`run_iter2.sh`（mtime 10-14 22:45）
是个更老、本轮没用上的脚本。我 21:34 比的是两个**没有因果关系**的脚本。

**作废后重新取证，四步全部可查**：

| # | 查什么 | 结果 |
|---|---|---|
| 1 | round2 的驱动脚本是谁 | `run_round.sh`（文案唯一匹配，见上） |
| 2 | round2 实际跑的 vLLM 引擎 | **V0**。`round2.step2.log:7` 和 `:102`：`Initializing a V0 LLM engine (v0.8.4)` |
| 3 | step2 的引擎能代表 step1/step3 吗 | 能。`run_step{1,2,3}.sh` 一处 `VLLM_USE_V1` 都没有（grep -c = 0），引擎完全由外层环境决定 |
| 4 | round3 的引擎 | V0。`run_round.sh` 现版 `export VLLM_USE_V1=0`，且 `/proc/936622/environ` 实锤 |

`round2.step2.log` 是**这一轮里唯一没被覆盖的一手证据**——它开头就是 step2 自己起的
vLLM 服务横幅（19:59:32 起，两个模型各起一次），而 `vllm_server.log` 每轮被 truncate 重写、
round2 的那份已经没了。**教训：会覆盖的日志不能当证据，要找那些天然只写一次的。**

**修正后的结论**：round2 和 round3 都跑 V0，**引擎这一项没有混淆**。
所以 **iter2→iter3 在所有我能叫出名字的混淆上都是干净的**：
难度判据（agreement，round2.log 自己 echo 了）、step3 beta（0.01，
`run_step3.sh` mtime 15:53 早于 round2 step3 起跑 20:23）、引擎（V0/V0）。
唯一的残留未知项：`run_round.sh` 在 **20:33:55** 被改过一次（round2 step3 20:32:13 结束后 100 秒），
改动内容无从考证——export 段当时已经是 V0（由证据 2 反推），但那一次编辑可能动过别的地方。

**iter3→iter4 仍然更强**：不是"更干净"，而是"零残留未知"——同一版 `run_round.sh`，
中间没有编辑过。所以汇报时分两句说：
- iter2→iter3：三项已命名的混淆全清，一次未知编辑
- iter3→iter4：三项全清 + 零未知

**方法论教训（两条，都不是原来那条）**：
1. **"比脚本"之前先确认这个脚本真的在跑。** 我 21:34 比 `run_iter2.sh` vs `run_round.sh` 比得很认真，
   diff 出了 7 个 export，还追进 `vllm/envs.py` 看了默认值——整套推理都成立，
   唯一的问题是**比错了对象**。先问"哪个脚本驱动了这一轮"，再比它。
2. 会覆盖的日志（`vllm_server.log` 每轮 truncate）不能当跨轮证据；
   每轮只写一次的文件（`round{N}.step2.log` 的启动横幅）才能当证据。

## 执行顺序

- [x] 建固定评测集 `holdout_solver.json`（135 条，32 域，agreement 均值 0.535）
- [x] 建评测台 `eval_iter.py` + 对照表 `compare_rounds.py` + 日志解析 `parse_trainlog.py`
- [x] 通用轮次执行器 `run_round.sh` + step1 看守器 `watch_step1.sh`
- [x] 基线：base / iter1 的 generator 与 solver 各测一遍（**solver 两项要补测，见上**）
- [x] round 2 step1 判据存档点：步 11-20 均值 1.1995 ≥ 1.086，通过，跑完
- [x] round 2 step1 收工（20:00:21，3.2 小时）：checkpoint-50 是**合并整模**
      （`model.safetensors` 988 MB，无 adapter 残留），三项 reward 仍全部显著，见上面那张对照表
- [x] round 2 step2/step3 全程（20:00:29 → 20:32:13）
- [x] round 2 step2 收工（20:23:28，100 条教材，难度 easy 21/medium 49/hard 30）
- [x] round 2 step3 收工（20:32:13，`iter2_solver/checkpoint-50` 是合并整模）
      **ROUND_2_DONE**：generator `iter2_generator/checkpoint-50`、
      solver `iter2_solver/checkpoint-50`、教材 `iter2_data.json`
- [x] **修 `run_round.sh` 的 `kill_server` 守卫**（20:33 改完并干跑验证）。
      改成 `_owns_pid`（exe 解析匹配 **且** cwd 是本仓库，两个条件缺一不可）
      + `kill_tree_if_ours`（杀整棵树，GPU 在 `multiprocessing.spawn` 的 worker 子进程里）。
      顺手补了 `HERE=$PWD`——第一版忘了这个变量，`want` 会是空的，守卫又变成永远判否，
      **等于把刚修的那个 bug 原样再犯一次**。改完用 `sed` 把函数体从文件里抽出来干跑：
      pid 854859 认得出、2214/857656 拒绝、`HERE` 为空时倒向安全侧。
- [x] **清掉 port 5000 上遗留的 solver 服务**（round 2 收摊后它还活着：854859 + 855018/855019）。
      **20:35 已清**：用修好的 `_owns_pid` + `_tree` 逐个核验身份后 TERM 整棵树，
      854859/855018/855019 全部退出、端口释放。`ss -ltn | grep -c ':5000 '` 现在为 0。
      顺带量到它当时的 GPU 占用是 0 MiB（杀前后总量差 8 MiB，差值来自评测进程自身波动）。
- [x] **核对 round 3 链路上没有危险的 pkill**（20:46）。这台盒子上有 Trae 的 jedi
      语言服务器（pid 2200/11315），一条 `pkill python` 会把它们一起干掉。
      核对结果：`run_round.sh` 只调 `run_step{0,1,2,3}.sh`；这四个里 `run_step1/2/3.sh`
      **一处 pkill 都没有**，`run_step0.sh` 那一处是注释（第 49 行，说明为什么不按模式杀）。
      带 `pkill -f "vllm.entrypoints.openai.api_server"` 的 `run_main.sh` / `run_iter2.sh`
      **不在 round 链路里**（`run_round.sh:6,14` 只是注释里提到它们）。
      上游仓库 `run_step1.sh` 结尾那句 `pkill python` 本地版本里没有。
- [x] **round 2 收摊后的固定顺序**（`post_round2.sh`，20:32 起跑，跑完）：
      1. `bash record_funnel.sh 2 50`（纯 CPU，读 verify 报告）
      2. `cp evals/iter1_solver.json evals/iter1_solver_unseeded.json`（留证）
      3. 设种重测 `iter1_solver`（86s）——`eval_round.sh` 的配对检验要拿它当 a 格，
         不重测就等于把"未设种/设种两套独立采样"的差混进配对差里
      4. 同 ckpt 再测一遍 `iter1_solver_rep2`，**量评测自身噪声**：同一个 checkpoint
         在同一个种子下跑两遍，差值就是评测噪声底。这个数决定了 iter2 比 iter1 高多少
         才算真高——比它小的"上升"一律不算
      5. 设种重测 `base_solver`（232s），把账本补全
      6. `check_holdout_clean.py qwen2.5-0.5b-instruct-tool-r0/iter2_data.json`
      7. `bash eval_round.sh 2 1 50`（gen 评测 ~30min + solver 86s + 5 项配对检验）
      8. 按上面定死的口径判：p<0.05 才算上升；方向正但 p≥0.05 报"不显著"；
         连续指标（fmt/acc）以置换检验为准

      **step1 一收工就要手动接 round 看守器**：`launch_round.sh` 是**前台**跑
      `watch_step1.sh` 的，watch_step1.sh 16:59 被旧版单条件卡死判据误杀（非零退出），
      launch_round.sh 按 "rc!=0 → 不接 round 看守器" 直接退了，所以它不会自动接上。
      19:06 手工重启的 watch_step1.sh 补的是 step 级看守，接不上这条链。
      ```
      nohup setsid bash watch_round.sh 2 200 >> /tmp/watch2_round.log 2>&1 &
      ```
      不接的话 step2/step3（各几十分钟）处于无人看守状态，而 watch_live.sh 会在
      `round2.step1.ok` 出现后查 `/tmp/watch2_round.log` 的 mtime，>300s 就喊。

      **未设种的 `iter1_solver` 逐条结果已于 19:23 备份到 `evals/iter1_solver_unseeded.json`**
      （135 条核对过）。这个备份必须在设种重测覆盖 `evals/iter1_solver.json` 之前做，
      已经做了，后面不用再管。

      步骤 3/4/5 的确切命令（ckpt 路径已核对：`iter1_solver_rl/checkpoint-50` 是**合并整模**
      `model.safetensors`，base 用 `Qwen/Qwen2.5-0.5B-Instruct` 且 HF 缓存里已有，不会去下载）：
      ```bash
      .venv/bin/python eval_iter.py --tag iter1_solver \
          --solver_ckpt qwen2.5-0.5b-instruct-tool-r0/iter1_solver_rl/checkpoint-50 \
          --note "设种重测（eval_iter.py 的 SOLVER_SEED=20261016，20:03 核对仍是这个值）"
      .venv/bin/python eval_iter.py --tag iter1_solver_rep2 \
          --solver_ckpt qwen2.5-0.5b-instruct-tool-r0/iter1_solver_rl/checkpoint-50 \
          --note "同 ckpt 同种子再测一遍，量评测自身噪声底"
      .venv/bin/python eval_iter.py --tag base_solver \
          --solver_ckpt Qwen/Qwen2.5-0.5B-Instruct --note "设种重测"
      ```
      **别写成 `SOLVER_SEED=... python eval_iter.py`**：`SOLVER_SEED` 是 `eval_iter.py` 里的
      模块级常量（第 44 行），没有任何 `os.environ` 读取，设环境变量对这个脚本是空操作。
      设种之所以生效，是因为 20:03 之前给 `eval_solver` 补了 `random.seed` + `torch.manual_seed`
      （注释原话："generator 那边一直设了种，solver 这边漏了，补上"）。所以
      **"设种重测"= 用当前代码原样重跑**，不需要也不应该加环境变量前缀。
      `iter1_solver.json` 是补种子之前测的，这就是它"未设种"的由来，也是为什么要重测。
      `iter1_solver_rep2` 量的是设种之后剩下的非确定性（bf16 CUDA kernel 的原子加、
      batch 内归约顺序），是评测噪声的下界——配对检验的显著性要对着它看。
      **20:37 实测结果：噪声底是严格的 0，不是"很小"。**
      `iter1_solver` 和 `iter1_solver_rep2` 四个指标逐位相同——
      `sol_fmt 0.9007`、`sol_acc 0.4782`、`sol_acc_pos 0.6889`、`sol_name_hit 0.5778`
      （106.4s vs 105.7s 只差在墙上时间）。也就是说这套评测在设种之后
      **完全可复现，没有任何随机成分**。

      这件事把判读口径简化了，也把一件事的责任加重了：
      - **简化**：不再需要"比噪声底小的上升不算上升"这道筛子，因为噪声底是 0。
        任何差值都是真实的模型差异，可以直接看差值本身和 p 值。
      - **加重**：p 值小不再能归因于"运气好"，它就是真的。反过来，
        **一个看起来很大的差值如果 p 值不显著，那它一定是方向不一致**（一堆正、
        一堆负互相抵消），而不是"样本量不够"——这点必须去数进步/退步条数，不能拿
        "n=135 不够"当借口。检出下限那一节算的仍然有效，但它的含义从
        "多少才测得出"变成了"多少才方向一致"。
      - 另外要核对 `evals/iter1_solver.json` 和 `evals/iter1_solver_rep2.json` 的
        **逐条内容**是否也完全相同（聚合指标相同通常意味着逐条相同，但值得确认一次）。
        真逐位相同的话，以后做 A/B 甚至不需要配对检验的置换部分，直接比聚合值即可。
        **20:40 已核对：`["solver"]` 列表 135 条 `==` 判定为真，md5 一致。**
        聚合和逐条两层都完全可复现。
      这三步和 `eval_round.sh` **都要等 ROUND_2_DONE 之后再跑**：step2/step3 正占着卡，
      0.5B 评测虽然只要 1-2 GiB，但为省 5 分钟去抢一个几小时的运行的显存不值得，
      抢失败了要重跑的代价大得多。它们也不依赖 round 2 的产物，没有正确性理由要提前。

      **工具链已在 19:20 干跑验证过**（不是等 20:00 才发现问题）：
      - `paired_test.py` 自比（iter1_gen/four_tag、iter1_solver/fmt）都报 p=1.0000、
        进步 0 退步 0；`fmt` 正确识别出 3 个取值并改判置换检验，McNemar 标"仅参照"
      - `check_holdout_clean.py`：iter1 真正训过的 `iter1_data_lr1e4_ckpt50_tv0.7.json`
        0 重题 rc=0；同源的 `..._tv0.001.json` 24 条重题 rc=1（门真的会喊，不是装饰）
      - `compare_rounds.py`：修了两个坑（详见下）
      - `record_funnel.sh` 只需要 `iter2_data.json.verify_report.txt`，step2 跑完就有
      - `eval_round.sh` 的路径映射核对过：`DATA=qwen2.5-0.5b-instruct-tool-r0/iter2_data.json`
        正是 run_round.sh 传给 run_step2.sh 的那个，PREV 默认 `ITER-1`=1 也对
- [~] round 3 全程（**21:10:13 已起跑**，run_round.sh pid=936154，step1 21:10:57 开始，看守器已确认挂上：`/tmp/watch3.log` 有进度行、`/tmp/watch3.hb` 21:11 在写、训练进程 exe 是 uv python 所以 RSS 探针有效。起跑前 GPU 299 MiB < 2000。**22:15 进度 16/50**，step_time 中位 229s、reward 1.8401、validity 0.6500、RSS 2.04 GiB、GPU 9899 MiB——无 spill 迹象。round2 同期中位 228.2s，**两边步速可比**（这一条 22:15 才核，因为要排除"round3 慢是因为服务器占着显存"）。按 229s/步算，step1 约 **00:20** 收，加 step2 ~24min + step3 ~9min，**整轮约 00:55**。**23:29 修正 ETA：step_time 中位 145s 比 22:15 估的 229s 快，step1 实际约 23:40 收**（46/50 时 ETA 11:54）。step1 的数见下面 23:26 结案那条）
- [ ] 记 round 3 漏斗（`record_funnel.sh 3`，基线已存：iter2 = 1118→918→100，easy 21/medium 49/hard 30）
      生成阶段已出（23:43）：round3 **1149 条 / 7m19s**，round2 1118 条 / 8m31s
      —— generator 多产 2.8% 且快 14%，和 step1 里 mean_length 885 vs 1120（短 21%）一致。
      1500 个 spec 投进去只写出 1149，即生成阶段自己滤掉 ~351 条。
- [ ] 评 iter3，和 iter2 比（iter2→iter3 已确认是干净对照：难度判据、beta、引擎三项全清，只有一次未知编辑）
- [ ] 按 21:30 那条**分通道预期**对照：name_hit 涨幅该 < iter2 的 +0.2148，acc 若还在涨应主要来自条件 acc
- [ ] 汇报

## 为什么 round 2 没被 `kill_server` 那个 bug 卡住

不是运气好，是**量过**。`kill_server` 末尾有 `u > 2500 就 die`，而 round 2 的 solver 服务
（`--gpu-memory-utilization 0.35`，自己日志里写 `total 10.00GiB × 0.35 = 3.50GiB`，
weights 0.93 + non_torch 0.05 + activation 1.40 + KV 1.12）**按理该占 3.5 GiB，
超过门槛 round 2 就会在 step2 后、step3 前死掉**。

但两次独立读数都说它只占 ~615 MiB：

| 时刻 | 总显存 | 当时活着的进程 |
|---|---|---|
| 19:57 | 9961 MiB | step1 trainer + port5000 服务 |
| 19:59:21（trainer 刚退出） | **615 MiB** | 只剩 port5000 服务 |
| 20:03 | 9802 MiB | 服务 + step2_gen 的 vLLM（拿到满 9.00 GiB） |
| 20:08+ | 9822 MiB | 服务 + genverify 的 vLLM（拿到满 9.00 GiB） |

后两行是硬证据：step2_gen 和 genverify 的 vLLM 各自都拿到
`total 10.00GiB × 0.90 = 9.00GiB` 全额预算（日志原样打印），**服务还活着**。
3.5 + 9.0 = 12.5 GiB > 10 GiB，所以它当时绝不可能占着 3.5 GiB。
（它自己只启动过一次，`vllm_server.log` 里"Started server process"和那行显存分解各只有一条。）

服务为什么从 3.5 GiB 缩到 615 MiB 我没查到底，但不影响结论：
**两个 vLLM 实例在它活着的时候都能拿满预算 + 19:59:21 的直接读数，两条独立证据一致。**

**20:35 按 PID 清掉它的时候把这个数补上了**：杀之前 GPU 4495 MiB、杀之后 4487 MiB，
**差 8 MiB**——那个服务到 round 2 收摊时已经不占显存了，RSS 5.65 GiB 全是主机内存。
进程活着、端口占着、GPU 放着。所以它真正造成的危害只有一个：**占着 port 5000 不让
round 3 起新服务**，显存方面从来没构成威胁（`>2500` 那条 die 从始至终不会触发）。
整棵树已杀（854859 API server + 855018 resource_tracker + 855019 EngineCore worker），
用的是修好的 `_owns_pid` + `_tree`，杀前逐个核验过身份。

## 新增/改动的文件

| 文件 | 作用 |
|---|---|
| `build_holdout.py` | 造固定 solver 评测集（含来源与排除规则的论证） |
| `holdout_solver.json` | 固定评测集本身：135 条、32 域、agreement 均值 0.535。**难度分布是左截断的**（见下面 21:47 那条）：agreement ∈ [0.2, 1.0]，没有 ag<0.2 的题，`lt0.3` 档实际是 0.2-0.3 |
| `eval_iter.py` | 固定条件下的 generator / solver 评测台，结果追加 `iter_metrics.jsonl` |
| `eval_round.sh` | 一轮跑完后自动评 generator+solver，并连着做 5 项配对检验 |
| `paired_test.py` | McNemar 精确检验 + 配对符号翻转置换检验。多值指标（fmt 3 个取值、acc 21 个）自动改判，以置换检验为主——McNemar 的 `>0` 二值化对"0.3→1.0"完全瞎 |
| `check_holdout_clean.py` | correctness 门：确认当轮教材没和固定评测集重题（同源真会撞，撞了 held-out 就作假） |
| `iter_metrics.jsonl` | 每个 checkpoint 一行的指标账本；`evals/<tag>.json` 存逐条结果 |
| `compare_rounds.py` | 把 `iter_metrics.jsonl` 摆成多轮对照表 + 逐列判单调。**重复 tag 取最后一次**（eval_iter.py 是追加写，设种重测同一个 ckpt 会产生第二行，不去重的话"升了又降回去"说不清是哪次测的）；`_rep<N>` 行只显示不进单调性判定（那是量评测噪声的探针，用它判自己的方法等于用噪声判噪声） |
| `parse_trainlog.py` | 从 train.log 抠每步指标，出前8/后8 均值 + 斜率 + t 值 |
| `run_round.sh` | 通用轮次执行器（按 ITER 推路径、分阶段日志、可续跑） |
| `watch_step1.sh` | 看守 step1：每分钟报步时/RSS/reward，覆盖崩溃、卡死、进程消失、spill 签名；每分钟 `touch /tmp/watch<ITER>.hb` 供上层查存活 |
| `launch_round.sh` | 起 run_round.sh 并自动挂看守器（锁文件防重入，起跑前查 GPU） |
| `watch_round.sh` | 看守 step2/step3：阶段标记 + 失败签名 + 15 分钟无输出**且** GPU<10% 才算卡死 |
| `record_funnel.sh` | 记每轮 step2 漏斗（生成→留存→教材）+ 难度配比，写 `iter_funnel.jsonl` |
| `watch_live.sh` | 把看守器输出搬进来看：只报新增行 + **独立查被看守方心跳**（不然"看守器死了"和"没事发生"输出一样），结束条件是 `round<ITER>.status` |
| `run_step1.sh` | batch 4×4 → 1×16（修 WSL2 spill） |
| `run_step3.sh` | 补 `--beta 0.01 --weight_decay 0.01` |
| `run_step0.sh` | gpu-mem-util 0.30→0.35，写 `vllm.pid` |
| `post_round2.sh` | round 2 收摊后的固定顺序，写成一个脚本。**存在的理由**：第 3 步会覆盖 `evals/iter1_solver.json`（设种重测），而覆盖前必须先确认未设种那份已备份——手敲容易在备份之前就跑重测，把唯一证据冲掉。脚本只在没有备份时才 `cp`，绝不用无条件覆盖；另有 `round2.step3.ok` 前置守卫，round 2 没收摊就退出（不去抢正在训练的显存） |
| `post_round.sh` | round 3 起的通用收摊顺序：漏斗 → 污染门 → 评测+配对 → 补 `acc_pos` → 分通道拆解。比 `post_round2.sh` 少三步（备份未设种、设种重测 iter1_solver、量噪声底）——那三步是 round 2 一次性的事，round 3 起重做只会覆盖掉唯一的未设种留证。**污染门放在最前面**：它不过的话后面所有 `sol_*` 都不是 held-out，评了白评。前置守卫 `round<ITER>.step3.ok` 不在就 exit 1（训练中跑评测会抢显存），已干跑验证 |
| `paired_accpos.py` | `acc_pos` 的配对检验。**为什么必须有它**：`paired_test.py --metric acc_pos` 是坏的且坏得很安静——逐条 json 里没有 `acc_pos` 字段（只有 `acc`），`x.get()` 返回 None，`num()` 当 0.0，于是两边均值都报 0.0000、"两者都非零 0"、p=1.0000「与 0 分不开」，看着像一次正常检验其实是 0 比 0。而 `--metric acc` 那行打的是连续均值，McNemar 分支里算出的 acc_pos 在分档分支里被丢掉没打印。已用 base_solver→iter1_solver 验过（0.0519→0.6889、88 进步 2 退步、p<0.0001） |
| `channel_split.py` | 把 acc 增益拆成**点名**和**参数**两个通道 + 难度分档。**为什么必须有它**：这个分析我手算了三遍、前两遍都错——①写成 `acc=点名率×条件acc`，但点名未命中时 acc 仍有 0.334（`accuracy_reward` 是稠密分），该式子不成立；正确是 `acc=命中支+未中支`，命中支=点名率×条件acc、未中支=(1-点名率)×未中时acc。②看到 iter1 条件 acc 按档 0.50~0.64 就说"没有档位效应"，而 0.50 对 0.641 是 28% 相对差，`0.7-0.85` 档的坑就在条件 acc 上。**分组必须由 a 格（较早那轮）的 `name_hit` 定义再贴到 b 格**，按各自轮次自己的分组比出来的差不是同一批题。自带两条闭合自检（命中支+未中支==acc、两组 n 相加==总数），不过就 exit 1。已用 iter1→iter2 验过：0.5755+0.0626=0.6380、固定分组 78/57、+0.0922/+0.2523 |

## 中止条件（不停就是跑通之前不许收工，但这几条要停）

1. 任何阶段 30 分钟无日志输出且 GPU 利用率 0
2. 同一处连续崩两次且第二次的根因和第一次相同
3. 指标掉了但定位不出原因——停下来报，不带着疑问跑下一轮
4. GPU 上出现不是本仓库起的进程占着显存

## 配对检验的检出下限（事前算，2026-10-15，在看到 iter2 结果之前）

`n=135` 的 McNemar 精确检验很弱，先把"多大变化才叫上升"定死，免得上界噪声当成果：

| 翻盘（进步/退步） | p | 判定 |
|---|---|---|
| 5 / 0 | 0.0625 | 不显著 |
| 6 / 0 | 0.0312 | 显著 |
| 6 / 1 | 0.1250 | 不显著 |
| 8 / 2 | 0.1094 | 不显著 |
| 12 / 4 | 0.0768 | 不显著 |
| 15 / 5 | 0.0414 | 显著 |
| 20 / 5 | 0.0041 | 显著 |

=> **135 题上净翻盘不到 10 题，就等于没有证据**，不是"趋势向好"。
方向为正但 p≥0.05 报"方向为正、不显著"，不许叫上升。

**连续指标必须换检验（这天实测出来的）**：`fmt` 取值是 {0.0, 0.3, 1.0}（114/11/10），
`acc` 有 21 个不同取值——两个都是分档奖励。McNemar 的 `>0` 二值化会把它们压成 0/1，
合成用例实测：**11 条 fmt 从 0.3 修到 1.0，McNemar 报 p=1.0000、discordant 0**
（"两者都非零 125"），格式真变好了它完全看不见；同一份数据连续配对检验 p=0.0011。
所以 `paired_test.py` 里：多值指标（>2 个取值）自动改判，主检验用**配对符号翻转置换检验**
（固定种子 20261015、20000 次），McNemar 那行只当稳健性参照。
`name_hit` / `four_tag` / `pv_ok` 是真二值，两行一致，以 McNemar 为准。

## 剩余空间在哪（iter1 逐条结果分出来的，决定多轮信号看哪列）

generator 192 条，`gen_four_tag` 0.9323（13 条不齐）：

| 失败原因 | 条数 | 说明 |
|---|---|---|
| `missing_tags` | 10 | 四标签失败的主因，**全部是写太长被 max_new=2048 截顶**（失败条中位 8063 字符，全体中位 4349） |
| `contains_cjk` | 2 | 题面里冒中文 |
| `tools_json_parse_fail` | 1 | |

=> `gen_four_tag` 剩下 13 条里 10 条是**长度问题不是能力问题**，上限被评测条件的 max_new=2048 卡住。
iter2→iter3 在这一列大概率走平，别误判成"generator 到顶了"。

`gen_pv_ok` 只有 **0.4740**（91/192 通过，101 条失败），这才是大头，且失败是语义性的：

| 失败原因 | 条数 |
|---|---|
| `tools_json_parse_fail` | 25 |
| `tool_parameters_nonprimitive_type` | 25 |
| `tool_missing_parameters` | 18 |
| `missing_tags` | 10 |
| `too_trivial_single_call` | 6 |
| 其余 8 类 | 17 |

=> **多轮的 generator 信号主要来自 `gen_pv_ok`**（0.51 起步，翻倍空间），
`gen_four_tag` 当 sanity check 看。和 solver 侧一样：`fmt` 快满了（27% 已 1.000），
`acc` / `name_hit` 有空间——两边的信号都不在最直观的那一列上。

**待做的一次性诊断**（不改固定评测条件，另起一列）：把 GEN_MAX_NEW 从 2048 提到 4096
单独跑一遍 iter1_generator，验证那 10 条 missing_tags 到底是不是截顶。
是截顶就说明 0.93 是评测设置的上限不是模型的上限，报告里要写清楚，不能留着让人以为到顶。
`GEN_MAX_NEW` 是调用时从模块全局读的，所以在进程里改它就行，不碰 `eval_iter.py`：

```bash
.venv/bin/python -c "
import eval_iter as E
E.GEN_MAX_NEW = 4096          # 只在这一进程里生效，不影响 eval_round.sh 的固定条件
r = E.eval_generator('qwen2.5-0.5b-instruct-tool-r0/iter1_generator_rl_lr1e4/checkpoint-50')
print({k: v for k, v in r.items() if not k.startswith('_')})
"
```

## round 2 step1 存档点（判据事前写死）

iter1 的线性 lr 从 1e-4 衰到 ~2e-6，它的后 10 步是"沉降后"读数；round2 从 1e-4 冷重启，
早期在震荡。所以拿 round2 步 11-20 的 reward 均值跟 iter1 步 41-50 的 **1.2445** 比，
iter1 后 10 步 σ≈0.25 → 10 步均值 SE=0.079 → 2σ 下界 **1.086**。

- **≥ 1.086** → 冷重启正常恢复中，继续跑完
- **< 1.086** → 相对 iter1 终点统计上真实退化，停下来定位

**2026-10-15 18:14 实测：步 11-20 均值 = 1.1995，高于 1.086，判据通过，继续跑完。**
（与 iter1 终点 1.2445 差 0.045 ≈ 0.57 SE，分不开——即"恢复到了 iter1 终点水平"，
不是"超过了"。step_time 中位 237s、RSS 2 GiB，无 spill 签名。）

**19:05 补记（这一步是 round 2 最大的运维事故，记下来免得复犯）**：
`watch_step1.sh` 在 16:59 就被旧版单条件卡死判据误杀了（两条件修复 16:59 之后才落盘，
跑着的进程还是旧逻辑），而 `watch_live.sh` 只在新行出现时才搬东西——于是"看守器死了"
和"这一分钟没事发生"输出一模一样，我拿着零事件的 Monitor 以为一直在看，实际瞎了 3 小时。
训练没受影响（36/50 步、RSS 2 GiB、reward 1.39），但"我在监控"这个信念是假的。
已修：watch_step1.sh 每分钟写心跳文件 `/tmp/watch2.hb`，watch_live.sh 独立查心跳新旧
（>300s 即报），不再用 `pgrep -f` 认进程名——那个模式会命中探查命令自己（实测命中过）。
**教训：只转发新增行的看守器，必须单独回答"被看守的那端死了，我的输出会有什么不同"。**

## round 3 起跑前：iter2 判据 + round 3 预期（20:45 写，iter2 评测跑完之前）

**这一段写在看到 iter2 任何评测数之前**，目的是把"什么叫上升"钉住，免得出数之后挑好看的列。

### iter2 vs iter1 的判定口径（沿用 round 2 已定口径，不改）

a 格固定为**设种后**的 `iter1_solver`（0.9007 / 0.4782 / 0.6889 / 0.5778）
和 `iter1_gen`（four_tag 0.9323 / pv_ok 0.4740）。**不许拿未设种那版当 a 格**
（未设种 name_hit 0.6667，比设种的 0.5778 高 12 题——拿它当基线，iter2 随便怎么动都"显著"）。

| 指标 | a=iter1 | 什么算上升 | 备注 |
|---|---|---|---|
| `sol_fmt` | 0.9007 | 净翻 ≥5 题且 p<0.05 | 只剩 5 条 0.000 + 12 条 0.3，**天花板效应，不许期待大动** |
| `sol_acc` | 0.4782 | 净翻 ≥10 题且 p<0.05 | 21 个取值，看连续配对检验那行 |
| `sol_acc_pos` | 0.6889 | 净翻 ≥10 题且 p<0.05 | `eval_round.sh` 不跑这项，**要单独补跑** |
| `sol_name_hit` | 0.5778 | 净翻 ≥10 题且 p<0.05 | 57 条失败，空间最大；但也是采样最抖的指标 |
| `gen_four_tag` | 0.9323 | — | **只当 sanity check**，179/192 已通过，动不了几个 |
| `gen_pv_ok` | 0.4740 | 净翻 ≥10 题且 p<0.05 | **generator 侧的主信号**，101 条失败、语义性失败 |

- **p<0.05 才算上升**；方向为正但 p≥0.05 报"方向为正、不显著"，**不许叫上升**。
- 噪声底是 0（设种后评测完全可复现），所以**不需要"比噪声底小就不算"这道筛子**；
  但反过来说，p 值也不再能拿"运气"解释，**差值大而 p 不显著一定是方向不一致**，
  那时该做的是去数进步/退步条数，不是拿 n 不够当借口。
- 任何"上升"如果小于 round2 已经拿到的幅度（solver 三项都动了几题），要说清楚是增量不是突破。

### round 3 的预期（事前写，用来判断结果是否反常）

1. **generator 不该有 round2 那么大的涨幅**。round2 step1 涨了 +0.4049 reward。
   （~~其中相当一部分来自 `clipped_ratio` 从 0.0156 掉到 0.0000 的截顶机制~~
   —— **22:22 作废：那两个数是 OLS 拟合端点且 t=−1.25 不显著，实测 50 步里只有 7 步非零、
   末窗均值 0.00625 不是 0。机制那句不成立。**）
   真正该用的理由是：**round3 的起点在 1.50 而不是 round2 的 1.21**，
   同样的梯度下剩余空间更小。所以 reward 涨幅预期明显小于 +0.40。
   round3 step1 后 10 步 reward 若还高于 round2 终点算继续涨，
   （~~~1.61~~ 这个数我记错了，21:14 已钉的正确基线是**步 41-50 窗口均值 1.5632**、
   末步单值 1.7973，见下面「round 3 step1 的对照基线」）
   若回落到 1.2~1.3 是"震荡后重新沉降"，不是失败。
2. **`gen_four_tag` 大概率不动**（13 条失败的余量），主信号看 `gen_pv_ok`。
   iter1→round2 的 pv_ok 走向要等 iter2_gen 的数才知道；若 iter2 已经在 0.5 以上，
   round3 的空间就在收窄，报"接近上限"而不是"没涨"。
3. **solver 侧 `name_hit` 是最可能动的**（57 条失败，且 step3 每轮都训同一套 135 题的格式）。
4. **难度分档**：round2 已经是 easy 21/medium 49/hard 30，不是退化的 99:1。
   round3 若还是这个配比，说明 agreement 分档稳定，**不许因为"配比没变"说没进步**——
   分档本来就不该每轮剧变，它是教材结构不是质量指标。

### round 3 的分通道预期（21:30 补，依据 21:27 那条点名/参数分解）

这条是**可证伪的预测**，写在这就是为了结果出来后能对照，不是事后解释。

iter2 的底子（设种、135 题）：`name_hit 0.7926`、`acc 0.6380`、
条件 acc 0.726（n=107）。按 `acc = 命中支 + 未中支` 拆：

| | 命中支（点名对） | 未中支（点名错） | 合计 |
|---|---|---|---|
| iter1 | 0.5778 × 0.584 = 0.337 | 0.422 × 0.334 = 0.141 | 0.478 |
| iter2 | 0.7926 × 0.726 = 0.575 | 0.207 × 0.302 = 0.063 | 0.638 |

**预测：**
1. `name_hit` 只剩 28 题空间（1 - 0.7926），**round3 的 name_hit 涨幅应明显小于
   iter2 的 +0.2148**。若还涨 0.20 以上，说明我"点名通道快吃完了"的判断错了。
2. 若 round3 的 `sol_acc` 仍在涨，**主要应该来自条件 acc**（0.726 → 更高），
   不是点名率。对照方法：iter3 评测出来后按 21:27 那张固定分组表重算一遍。
3. **两种结果都算"循环还有用"，都不算失败**：
   - name_hit 涨幅 < 0.10 但 acc 涨幅 ≥ 0.10 → 印证"点名吃完、参数还有油"，该往下走
   - 两者都 ≥ iter2 的一半 → 循环还没到头，可以继续 round 4
4. **只有一种情况算到头**：`sol_acc` 涨幅 < 0.05 且 `sol_name_hit` 涨幅 < 0.05。
   那时该做的是改设计，**不是加轮数**。具体改哪三处、以及为什么我第一版列的两条
   （奖励给 values 加权 / 拉长 completion）是错的，见下面「round 4 的计划」里
   22:07 那次核销。

### round 3 漏斗预测：agreement 会被自己的成功抬高等（21:44）

**这条是循环的结构性限制，不是 round 3 的意外，先写下来免得看到时当成 bug。**

机制：`solver_agreement` 是用**上一轮的 solver** 测的，不是用基座。
`iter2_data.json.verify_report.txt` 头几行实证：
`solver_model: qwen2.5-0.5b-instruct-tool-r0/iter1_solver_rl/checkpoint-50`、`temp_verify: 0.7`。
而 `run_round.sh:172` 是 `bash run_step2.sh "$GEN_CKPT" "$PREV_SOLVER" "$DATA"` ——
**PREV_SOLVER 每轮都变强**。

已知的两端：
- iter2 教材（iter1_solver 阅卷）：agreement 均值 **0.532**，hard(<0.4) 30 / medium 49 / easy 21，
  实际配比 30/49/21，和想要的 30/50/20 基本吻合
- 固定评测集 135 题（**基座** + temp 0.001 测，另一套口径）：均值 0.535，hard 24%

**预测 round 3 教材的 agreement 均值会明显高于 0.532**，因为 iter2_solver 的点名率
已经 0.7926（iter1_solver 是 0.5778）。跟着来的是：
- `hard` 条数掉、`easy` 条数涨，实际配比偏离 30/50/20
- `drop_saturated`（默认剔 agreement≥1.0）会剔掉更多，留存池进一步缩小
- 域名内配额取不满时按 medium/easy 补（`step2_select_curriculum.py` 的 padding），
  **不会崩，是优雅降级**——所以现象是"配比慢慢变容易"，不是报错

**这意味着什么**：难度信号是**自指**的——solver 变强直接消灭自己的训练信号。
round 4、5 会继续这个趋势。判据上要注意：
`配比变了` 不等于 `退步`（事前口径第 4 条已经写了"不许因为配比没变说没进步"，
这里补另一半：也不许因为配比变容易就说没进步）。
真正的尽头信号是 **hard 条数趋近 0**——那时 agreement 分档就不再提供任何课程信号，
该换难度定义（比如改用参数正确率而不是点名一致率）。

**待核**：round 3 step2 跑完后用 `record_funnel.sh 3` 的配比行对照这条预测。

## round 4 的计划（22:03 写，round 3 还没跑完）

### 23:41 round3 收摊前修掉的三处运维缺陷（都不改训练设计，所以不影响"原样跑"）

趁 `watch_step1.sh` / `launch_round.sh` 都已退出、只有 `run_round.sh` 还在跑的窗口改的
（**`run_round.sh` 一个字节都没碰**，mtime 仍是 20:33:55，snapshot 里那份就是 round3 跑的版本）：

1. **`watch_step1.sh` 的竞态（round3 真的踩到了）**。原循环先查「进程还在不在」再查「step1 OK」，
   而 `run_round.sh` 的顺序是训练进程退出 → merge LoRA → 校验 `model.safetensors` → 才写 "step1 OK"。
   round3 实测 23:35:44 报「进程不在了」exit 2，而 "step1 OK" 23:35:46 才落盘——
   `launch_round.sh` 见 rc≠0 就不接 round 看守器，**step2/step3 全程没人看**
   （我 23:35 发现 `/tmp/watch3_round.log` 不存在才手工补挂）。
   修法：完成标记移到进程检查**之前**，且进程消失后给 `GRACE=180s` 宽限期再判失败。
   两条路径都实测过：进程不在+标记已在 → exit 0；进程不在+无标记 → 等满宽限期 → exit 2。
2. **`launch_round.sh` 改成无条件接 round 看守器**。原来 rc≠0 直接 `exit`，于是 step1 看守器
   一误报整轮裸奔。改成照样接挂：真崩了 `run_round.sh` 会 `die`，`watch_round.sh` 有
   "run_round.sh 进程不在了 → exit 2" 分支，几秒内报完，代价为零；
   漏挂一次的代价是整轮没人看。**两边不对称，所以选永远接挂。**
3. **`snapshot_round.sh` 调用并进 `launch_round.sh`**（起跑前、`run_round.sh` 之前）。
   快照失败不阻断起跑（它是留证不是正确性门），但会在 driver 日志里喊一声。

**这三条都是运维修复，不动任何训练超参、奖励函数、难度判据**，
所以 round4「原样跑、零残留未知」的性质不变，只是不会再出现看守空档。

**核心决定：round 4 原样跑，不改任何设计。**

理由是 round 4 是**零残留未知的那一个多轮对照**（iter2→iter3 也已经很干净，但中间有过一次未知编辑）。已知的混淆：

| 对照 | 难度判据 | step3 beta | 引擎 | 残留未知 |
|---|---|---|---|---|
| iter1→iter2 | 换了（judge→agreement） | 0 → 0.01 | 未知 | 两个已命名混淆 |
| iter2→iter3 | 一致 | 一致 | **V0 / V0** | 一次未知编辑（`run_round.sh` 20:33:55） |
| **iter3→iter4** | **一致** | **一致** | **V0 / V0** | **无** |

（引擎那行 22:15 改过：原先写 iter2→iter3 是"V1→V0"，依据是拿 `run_iter2.sh` 当 round2 的驱动脚本，
而它从没驱动过 round2。`round2.step2.log:7` 的 `Initializing a V0 LLM engine` 才是实证。
详见上面"第三个混淆：作废"那节。）

用户的目标是"多轮是可以有用的"，这句话要能站住，**至少要有一对两个变量都没动的对照**。
iter3→iter4 是零残留未知的那一对，iter2→iter3 也已经是（三项混淆全清 + 一次未知编辑）。
如果现在为了"多赚一点"去改奖励或改难度定义，就把这两对都毁了，等于用一次改动换一个数，不值。

**round 4 事前判据**（和 round 3 同一套口径，不新造）：
- `sol_acc` / `sol_name_hit` / `sol_acc_pos` / `sol_fmt` 都用 `post_round.sh 4 3`
  出的配对检验，p<0.05 才算上升，方向为正但不显著照实报
- 参照 round 3 的幅度打折预期：点名通道已吃到 0.79，
  **round 4 的 name_hit 涨幅若 < 0.10 是预期之内，不是退步**
- 分通道用 `channel_split.py --a iter3_solver --b iter4_solver`，
  重点看**条件 acc 有没有从 0.73 再往上**——那是还没枯竭的通道

**什么情况下才改设计（不改成别的）**：
`sol_acc` 涨幅 < 0.05 **且** `sol_name_hit` 涨幅 < 0.05。
那时按优先级动这三处，一次只动一处：

1. **难度定义从"点名一致率"换成"参数正确率"**（agreement 自指问题的根治办法）。
   这是三个里唯一有结构性证据的：agreement 用的是 `normalize_call_names_only`
   （`step2_genverify.py`），只 serialise 工具名、不看参数，而 solver 变强主要就强在点名，
   于是自己消灭自己的 hard 档。**尽头信号是 hard 条数趋近 0**。

   **23:45 已核成可执行的两行改动**（原来只知道"该换"，不知道换哪）：
   `normalize_call_names_only`（`step2_genverify.py:117`）函数体只取 `c.get("name")`，
   `arguments` 一个字段都不读；而 `normalize_call_json`（`:96`）做 name + 排序后 arguments 的
   全量精确匹配。`solver_agreement_batch` 里 gold 侧在 **`:231`**、pred 侧在 **`:269`**，
   两处都调 names_only。
   另外 `gold_norm`（`:136`，包着 `normalize_call_json`）**定义了但全仓库无人调用**——
   是死代码，说明"全量匹配"这条路本来就预留了、只是没接上。
   **所以这一处就是把 `:231` 和 `:269` 的 `normalize_call_names_only` 换成 `normalize_call_json`**，
   两行，不新增函数。改完 agreement 就从"点名一致率"变成"点名+参数都对的一致率"，
   hard 档不会再被点名提升自己掏空。
2. **把 `tau_verify` 从 0.2 降到 0**，重跑一次 step2 verify。
   现在 verified 池被 `step2_genverify.py:388` 左截断，最小值正好是 0.2，
   固定评测集也从同一个池抽样，所以它的 `lt0.3` 档实际是 0.2-0.3。
   "RL 在基座弱的地方涨"这个结论只在 [0.2,1.0] 验过，基座最弱那批（ag<0.2）
   的行为未知。降 tau 能让 eval 集和训练池都覆盖到真正难的题。
3. **抬高 curriculum 规模**（`n_final` 现在 100，step3 只跑 50 步）。
   100 条教材 × 50 步，等于同一批题被反复嚼。代价是 verify 时间按 k_verify=10 线性涨。

**我第一版在这里列错了两条，22:07 用日志核掉后删了**（记下来是因为错的方式很典型）：
- ~~"奖励里给 values 加权，format 项已经 0.977 接近天花板等于白占一份权重"~~
  —— **0.977 是 held-out 上 step3 solver 的 `sol_fmt`，不是 step1 generator 的 format reward**。
  step1 的 `rewards/reward_format_accuracy/mean` 实测 round2 0.7457→0.8698、round3 0.8238→0.8993
  （首6步→末6步，23:35 用 50 步完整数据重算过——我 22:07 原文写的"0.875→0.9375 / 0.817→0.854"
  **两个数都不对**，那是我从尾部几个值估的，不是首末窗。这是今天第九处自我更正，
  形状和第二处一样：把估计值写成了"实测"），
  离天花板还远且在动。format 接近常数这件事**只对 step3 成立**
  （`round2.step3.log` 的 `format_reward_func` 50 步里大多是 1.0，最低 0.7875），
  而 step3 的 reward 就是 `format_reward_func + accuracy_reward_func`（1:1，
  `step3_solver.py:84` 没传 `reward_weights`，走 TRL 默认）。
  可 format 是常数项时它对 advantage 的贡献本来就近 0——这**不是**该修的问题。
- ~~"拉长 `max_completion_length`，round3 的 max_length 均值 2251 贴到 2048 边缘"~~
  —— **上限那个数错了**：step1 的上限是 **4096**（`run_step1.sh:136`），不是 2048；
  step3 的上限才是 2048（`run_step3.sh:61`），而 `round2.step3.log` 的
  `completions/mean_length` 中位 **77.5**、最高 231，差一个数量级。
  **所以"拉长上限"这个设计改动确实不该做**（只有极少数步会撞顶，抬了也白抬），
  但我 22:07 给的理由"根本没有截断"是**错的**，22:38 在 28 步时被数据打脸：

  | | round2（n=50） | round3（n=28，22:38 时） |
  |---|---|---|
  | `max_length` 中位 | 2198 | **1604** |
  | `max_length` 最大 | 4096 | **4096** |
  | 撞到 4096 上限的步数 | **7 / 50（14%）** | **1 / 28（3.6%）** |

  **截顶是存在的，两轮都撞过 4096。** 22:07 我只看了前 15 步（那一段 max_length 最高 2165），
  就把"这 15 步没撞顶"外推成了"根本没有截顶"——**一个在检查当时为真的观察，
  被我写成了永恒的结论**。这是今天第五处自我更正，形状和前四处不同：
  不是归属错、不是对象错、不是把拟合线当数据，是**把有限样本的时刻快照当成稳定性质**。
  正确的说法是：**round3 撞顶频率是 round2 的四分之一（3.6% vs 14%）**，
  因为 round3 的 completion 明显更短。抬上限这个动作本身仍然不值得做（3.6% 的步），
  但理由要从"没有截断"改成"截断罕见且在减少"。

**明确不做的事**：不加轮数碰运气、不为了凑显著去换评测集、
不把 iter1→iter2 的涨幅当成"循环有效"的证据（它有至少两个混淆）。

**round 4 起跑前必须补的一个运维动作（22:20 记）**：**每轮快照驱动脚本。**
现在 `run_round.sh` 被改一次，之前所有轮次跑的是哪一版就**永久不可考**——
iter2→iter3 因此背上一个"20:33:55 那次编辑动了什么"的残留未知项，
而它本可以是一行 `cp` 的成本。做法（round 3 跑完之后再做，
**现在不能改 `launch_round.sh`/`run_round.sh`，它们正被 round 3 占着，bash 按字节偏移读脚本**）：

```bash
# launch_round.sh 起跑处加
SNAP="round_snapshots/round${ITER}.$(date +%Y%m%d_%H%M%S)"
mkdir -p "$SNAP" && cp run_round.sh run_step{0,1,2,3}.sh launch_round.sh \
    step1_generator.py step2_genverify.py step2_select_curriculum.py \
    step3_solver.py rewards_*.py "$SNAP"/
sha256sum "$SNAP"/* > "$SNAP/SHA256SUMS"
```

判据：快照里要有**每一轮的 export 段和难度判据参数**，因为这两项是已识别的混淆源
（引擎 V1/V0、judge/agreement）。代码常量类的混淆（`SOLVER_SEED` 那种）也一起覆盖到，
因为 `seeding-via-code-constant-not-env-var` 那类"以为设了其实没设"只有快照能查。

**23:29 已实现成一个独立文件 `snapshot_round.sh`（趁 round3 还在跑的最后 5 步写的）。**
为什么没并进 `launch_round.sh`：round3 正在跑，`launch_round.sh`/`run_round.sh`
是被那个 bash 按保存的字节偏移读着的，此刻编辑会损坏执行。**并进去是 round4 的前置任务。**
比上面那段伪代码多了三样：① 12 个文件一个不少，缺一个就退出（半份快照比没有更坏，
因为它看着像有）；② 单独记 `MTIMES`，因为 mtime 是事后反推"这一轮跑的是哪一版"的唯一凭据；
③ `sha256sum -c` 自检，不过就退出。
实测：`bash snapshot_round.sh 3` → 12 个文件 + 校验通过，
`round_snapshots/round3.20261015_232914/`。

**副作用（比预想的有价值）**：这份快照落盘于 23:29，晚于 round3 启动（20:47），
而 `run_round.sh` 的 mtime 是 **20:33:55**——正是 iter2→iter3 那笔"未知编辑"。
所以**这份快照就是 round3 实际跑的代码**，20:33:55 改了什么虽然仍不可知，
但"round3 跑的是哪一版"从此有据可查，不再是靠 mtime 猜。
round4 起这个残留未知项类彻底消失。


### round 3 起跑前修掉的两个运维隐患（20:48）

1. **`watch_step1.sh` 的超时上限从 200 分钟提到 330 分钟**（改在 `launch_round.sh:57`）。
   round2 的 step1 实测 **194.6 分钟**（`round2.log`：16:44:44 step1 起 → 19:59:21 step1 OK），
   200 分钟只剩 **5.4 分钟余量**。round3 只要慢 6 分钟，看守器就 `exit 5`，
   `launch_round.sh` 走 rc≠0 分支不接 `watch_round.sh`——训练照跑但 step2/step3 没人看，
   和 round 2 那个空档一模一样。看守器晚退没有代价（真的跑完时它靠 grep "step1 OK"
   正常 exit 0），早退纯亏，所以往宽了设。
   `watch_live.sh` 里那条"重启 watch_step1.sh"的处置提示同步改成 330。
2. **核对 round 3 链路上没有危险的 pkill**。这台盒子上有 Trae 的 jedi 语言服务器
   （pid 2200/11315），上游仓库 `run_step1.sh` 结尾那句 `pkill python` 会把它们一起干掉。
   核对结果：`run_round.sh` 只调 `run_step{0,1,2,3}.sh`，这四个里 `run_step1/2/3.sh`
   一处 pkill 都没有，`run_step0.sh` 那处是注释；带 `pkill -f "vllm...api_server"` 的
   `run_main.sh` / `run_iter2.sh` 不在 round 链路里。本地没有那句 `pkill python`。

### gen 配对检验的地基验过了（20:51）

`paired_test.py` 对 generator 是**按下标配对**（solver 才是按 question 配对）。
所以 iter1_gen / iter2_gen / iter3_gen 必须跑在同一批、同一顺序的 192 条 spec 上，
否则配对差是"题不一样"不是"模型变好了"。

验了三层，都过：
1. `eval_generator` 里 `random.seed(GEN_SEED)` 之后才 `[sample_spec() for _ in range(n)]`
   ——顺序由种子决定，不受调用时机影响
2. 两个独立进程各生成一遍，192 条 spec 的 md5 相同（`e0d5f45b54c33152`）
3. `evals/iter1_gen.json` 里 192 条的 `spec` 字段和重新生成的逐条相同、md5 相同

=> 只要 `GEN_SEED` / `GEN_N` 不改，gen 侧的跨轮配对就是同题配对。
（这两个是 PLAN 顶部"固定评测条件"里写死不许改的量。）

## iter2 结果（21:08 收齐）：六项里五项显著上升，一项持平

**a 格全部是设种后的值**（`iter1_solver` 0.9007/0.4782/0.6889/0.5778，
`iter1_gen` 0.9323/0.4740）。未设种那版一个都没用。

### 账本（`compare_rounds.py`，21:08）

| tag | g.four_tag | g.pv_ok | s.fmt | s.acc | s.acc_pos | s.name_hit |
|---|---|---|---|---|---|---|
| `base_gen` | 0.0833 | 0.0052 | - | - | - | - |
| `base_solver` | - | - | 0.1719 | 0.0285 | 0.0519 | 0.0741 |
| `iter1_gen` | 0.9323 | 0.4740 | - | - | - | - |
| `iter1_solver` | - | - | 0.9007 | 0.4782 | 0.6889 | 0.5778 |
| `iter2_gen` | **0.9531** | **0.6094** | - | - | - | - |
| `iter2_solver` | - | - | **0.9770** | **0.6380** | **0.8889** | **0.7926** |

**六列全部单调不降**（base → iter1 → iter2）。

**23:38 核过一处容易重新吵的地方**：`base_solver` 在 `iter_metrics.jsonl` 里有**两行**
（16:08「原版」未设种 fmt 0.2163 / 20:36「设种重测」fmt 0.1719），表里用的是 0.1719。
去重规则是"重复 tag 取最后一次"，而最后一次正是设种那份——**和 `iter1_solver` 同规则，是对的**，
不是挑了个差的基线。看着像"基线被改低了"，实际只是 fmt 这个 graded 指标的抽样波动
（同一次设种重测里 `sol_acc` 0.0207→0.0285、`sol_acc_pos` 0.0296→0.0519 都是涨的）。
单调性判定用哪一行都成立（0.2163 也远小于 iter1 的 0.9007），所以不影响结论。

### 配对检验 iter1 → iter2（135 题 solver / 192 条 spec gen）

| 指标 | iter1 | iter2 | 差 | 进步/退步 | p | 判定 |
|---|---|---|---|---|---|---|
| `gen_four_tag` | 0.9323 | 0.9531 | +0.0208（+4 题） | 13 / 9 | 0.5235 | **持平** |
| `gen_pv_ok` | 0.4740 | 0.6094 | +0.1354（+26 题） | 65 / 39 | 0.0138 | **上升** |
| `sol_fmt` | 0.9007 | 0.9770 | +0.0763（+10 题） | 5 / 2 | 0.0029（置换） | **上升** |
| `sol_acc` | 0.4782 | 0.6380 | +0.1598（+22 题） | 36 / 9 | <0.0001 | **上升** |
| `sol_acc_pos` | 0.6889 | 0.8889 | +0.2000（+27 题） | 36 / 9 | 0.0001 | **上升** |
| `sol_name_hit` | 0.5778 | 0.7926 | +0.2148（+29 题） | 36 / 7 | <0.0001 | **上升** |

按事前口径判：五项 `p<0.05` 且净翻 ≥10 题，算上升；
`gen_four_tag` 净翻 4 题、p=0.5235，算持平——**和事前预判一致**（179/192 已通过，
当时就写了"只当 sanity check，动不了几个"）。没有出现"方向为正但不显著"需要打折扣的情况。

`sol_fmt` 的 McNemar 报 0.4531（5/2），但它是分档奖励（4 个取值），按事前定的口径
以连续配对检验为准（p=0.0029）。这里要说明白：**fmt 只有 5 条从非零变成零以上、
2 条反向，靠 McNemar 的 discordant 计数根本检不出来**，而置换检验用上了幅度
（0.3 → 1.0 也算进步）才显著。这正是当初把 McNemar 降级成"仅参照"的原因。

### 增益来自哪一档（**结论我写反过一次，21:20 改正，改正后的版本在下面**）

`solver_agreement` 是题目固有属性（iter1/iter2 两份逐条分布完全相同，都是
{0.2:16, 0.3:16, 0.4:17, 0.5:28, 0.6:22, 0.7:13, 0.8:6, 0.9:7, 1.0:10}），档位可比。

**先说这个字段到底量什么**（我一开始理解错了，导致结论整个反了）：

| | 判定标准 | 谁测的 | 温度 |
|---|---|---|---|
| `solver_agreement` | **只比工具名序列**（`normalize_call_names_only` 把参数全丢了，`pn == g`） | **基座** `Qwen/Qwen2.5-0.5B-Instruct` | 0.001，k=10 |
| `sol_acc` | `accuracy_reward`：稠密分，比 name + key-F1 + values，还罚超出 GT 的调用 | 当轮 solver | 0.7，单次 |

出处：`iter1_data_lr1e4_ckpt50_tv0.001.json.verify_report.txt` 头两行
`solver_model: Qwen/Qwen2.5-0.5B-Instruct` / `temp_verify: 0.001`；
`step2_genverify.py:117` 的 `normalize_call_names_only` 只序列化 name。

**所以 agreement 越高 = 基座越会点名工具，不是"越难"。**
这也解释了一个 otherwise 很矛盾的现象：基座在 `eq1.0` 档 agreement=1.0（十次全中），
但它在同一档的 `sol_acc` 是 **0.000**——工具名全对、参数全错。

分档结果（配对，同一批 135 题）：

| 档位 | n | iter1 | iter2 | 差 | 进步/退步 | 符号检验 p |
|---|---|---|---|---|---|---|
| `lt0.3` | 16 | 0.460 | 0.727 | +0.266 | 8 / 2 | 0.109 |
| `0.3-0.5` | 33 | 0.441 | 0.624 | +0.184 | 17 / 3 | 0.0026 |
| `0.5-0.7` | 50 | 0.540 | 0.662 | +0.121 | 20 / 6 | 0.0094 |
| `0.7-0.85` | 19 | 0.384 | 0.647 | +0.263 | 8 / 3 | 0.227 |
| `0.85-1.0` | 7 | 0.416 | 0.395 | -0.021 | 3 / 2 | 1.000 |
| `eq1.0` | 10 | 0.542 | 0.575 | +0.033 | 2 / 2 | 1.000 |

**正确的结论：增益集中在 ag<0.85 的 118 题（合计 53 进步 / 14 退步）；
ag≥0.85 的 17 题是 5 进步 / 4 退步、p=1.0，没有进步。**
而那 17 题正是**基座已经会点名工具**的题（基座在 eq1.0 档 agreement=1.0）。
所以形状是：**RL 在基座弱的地方涨，在基座已经会的地方饱和**——这是健康的样子，
不是 bug。总量上 58 进步 / 18 退步。

**对 round 3 的含义（这条也跟着翻过来了）**：`select_curriculum` 挑的 hard 30%
是**低 agreement** 题，而低 agreement 档恰恰是训练有效的地方。
所以"按 agreement 分档选教材"这个设计**是在往对的地方使力**，
round 2 把 judge（退化成 100% hard）换成 agreement 是对的。
round 3 若同样呈现"低 agreement 涨、高 agreement 平"，就确认了这个循环的作用范围。

#### 分档解读的一个前提：评测集的难度是**左截断**的（21:47 补）

上面所有分档表都建立在一件事上，这件事之前没写明：**评测集里没有一道基座真正觉得难的题。**

证据链：
- 评测集从 `iter1_data_lr1e4_ckpt50_tv0.001.json.verified.json` 分层抽样来
  （`build_holdout.py --pool` 默认值）
- 那个池子 499 条，agreement **min = 0.2**，没有任何一条 < 0.2
- 原因是 `step2_genverify.py:388` 的 `if r >= args.tau_verify: kept`，
  而 `tau_verify: 0.2` —— **低于 0.2 的在入库前就被滤掉了**
- 池子取值分布 `{0.2:17, 0.3:21, 0.4:21, 0.5:31, 0.6:33, 0.7:14, 0.8:10, 0.9:10, 1.0:342}`，
  **69% 是 agreement=1.0**

**所以分档表的 `lt0.3` 档实际是 0.2-0.3，不是 0-0.3。**
"基座最弱的地方"在这个评测集上只到"10 次里对 2 次"，
真正"10 次全错"的题一道都没进来。

**这对结论的影响**：
- 不影响"低 agreement 档涨得多"这个观察本身（档内比较仍然有效）
- 但影响外推：**不能说"RL 在基座弱的地方涨"已经被充分验证**——
  基座最弱的那批题（ag<0.2）根本没进评测集，那边的行为是未知的
- 也不影响 `select_curriculum`：它用的是每轮新 verify 的池子，
  那个池子同样被 tau 截断（round2 的 tau 是多少要看当轮 report），
  所以"hard 档"在两个地方都是同一个截断后的区间

**要验证的话**：把 tau_verify 降到 0 重跑一遍 verify，看 ag<0.2 的题有多少、
RL 在那边还涨不涨。这是个独立的实验，不在当前 round 3 的范围里，先记着。

### `0.7-0.85` 档最低这件事：查清了，但**我第一版结论写反了**（21:24 写，21:27 改）

上面留的那个"还没解释的现象"有答案了。先记方法，因为**第一版就是方法对但结论读反**：

按 `name_hit` 拆开配对是对的，**不能用** `acc = 点名率 × 条件acc` 这个式子做分解——
点名未命中时 acc 仍有 0.334（工具名错但参数对照样拿部分分），
0.5778 × 0.584 = 0.337 ≠ 0.478。正确做法是按一个**固定分组**（iter1 的 name_hit）
各自配对比较 iter1/iter2 在同一批题上的 acc：

| 分组 | n | iter1 acc | iter2 acc | 差 | 进步/退步 | 点名率变化 |
|---|---|---|---|---|---|---|
| iter1 **点名命中** | 78 | 0.584 | 0.676 | +0.092 | 20 / 12 | 1.000 → 0.910（**降**） |
| iter1 **点名未中** | 57 | 0.334 | 0.586 | **+0.252** | 38 / 6 | 0.000 → 0.632 |

iter2 在这两组上的条件 acc 几乎一样（0.724 vs 0.730）——这点很重要，见结论 3。

**结论 1（改正后）：`0.7-0.85` 档低是"参数难"，不是"点名不稳"。**
低 ag 四档拆成两个通道：

| 档位 | n | acc | 点名率 | 条件acc |
|---|---|---|---|---|
| `lt0.3` | 16 | 0.460 | 0.500 | 0.552 |
| `0.3-0.5` | 33 | 0.441 | 0.485 | 0.541 |
| `0.5-0.7` | 50 | 0.540 | 0.600 | 0.641 |
| `0.7-0.85` | 19 | **0.384** | 0.579 | **0.500** |

极差：acc 0.156 / 点名率 0.115 / 条件acc 0.141——**两个通道各贡献一半左右，
不存在"几乎全部来自点名率"**。`0.7-0.85` 这一档点名率 0.579 是四档里第 2 高，
条件 acc 0.500 是四档里最低，所以它的坑在**参数填充**上。
（我 21:24 那版写的是"ag 只管点名、高档参数不难、坑在点名率"，**读反了**。
起因是我当时只有 iter1 的条件 acc 表，看到 0.50–0.64 就当成"没有档位效应"——
0.50 对 0.64 是 28% 的相对差，不是没效应。）

**结论 2：RL 涨得最猛的地方是 iter1 点名失败的题（+0.252 vs +0.092）。**
"在模型弱的地方涨、在已经会的地方饱和"这个形状，在题目层面也成立。
代价记下来：命中组点名率掉了 9 个点（78 题里 7 题点名退化），
round 3 要盯这个数，别变成拆东墙补西墙。

**结论 3（这条才是能指导下一步的）：iter2 的条件 acc 收敛到 ~0.73，且不随难度变。**
iter1 条件 acc 按档是 0.50–0.64 有档位效应；iter2 按档变成
0.868 / 0.722 / 0.722 / 0.680（低 ag 四档），而固定分组上两组都是 0.72–0.73。
也就是说 round 2 把"点名命中之后参数对不对"从 0.58 提到了 0.73，
**但它是个与题目难度无关的上限**——不是某类题不会，是全都卡在七成三。
结合点名率已经从 0.578 吃到 0.793：**round 4 之后再加轮，点名这条通道快没油了，
收益得往参数填充走**。具体怎么走见下面「round 4 的计划」22:07 那次核销——
我原来在这里写的"更长的 completion、或奖励里给 values 加权"**两条都不成立**
（completion 根本没被截断；format 是常数项时对 advantage 贡献本就近 0）。

### round 3 step1 的对照基线（21:14 钉，此时 round3 只有 1 步数据）

上面那条预期里我写"round2 终点 ~1.61"是**记错的**。`round2.step1.log` 逐步解出来的真实基线：

| 窗口 | round2 reward 均值 | σ |
|---|---|---|
| 步 1-10 | 1.2320 | 0.1613 |
| 步 11-20 | 1.1995 | 0.2675 |
| 步 21-30 | 1.4576 | 0.1833 |
| 步 31-40 | 1.4857 | 0.1931 |
| 步 41-50 | **1.5632** | 0.1901 |
| 全程 50 步 | 1.3876 | 0.2488 |

全程斜率 +0.00909/步（t=4.30），末步单值 1.7973，全程 min 0.6802 / max 1.8806。
分量均值：format 0.7964 / validity 0.4497 / curriculum 0.1415。

**所以"继续涨"的判据改成（同窗口比，不用单步）**：
round3 的步 N-M 窗口均值 > round2 同窗口均值，且差距超过两窗 σ 合并的 1/√N 量级
（10 步窗口约 0.08）才算这一轮在同一训练阶段更强。

**round3 已知的 1 步**（21:12:58，reward 1.2880，step_time 103s，均长 965）：
单步对照没意义——round2 自己第 1 步 1.6000、第 2-4 步就掉到 1.24/1.07/1.06。
要比就等 10 步窗口。已知的两个好兆头：
- step_time 103s vs round2 同期 146s，快了 30%（triton cache 在 round2 已经热了，
  `TRITON_CACHE_DIR=./triton_autotune` 是仓库内持久的）
- 均长 965 vs round2 第 1 步 976，没有变长——**变长才会让步时爆炸**

### 10 步时的对照（21:47 补，第一个完整窗口）

| | round2 | round3 |
|---|---|---|
| 步 1-10 reward 均值 | 1.2320 | **1.5020** |
| 步 41-50 窗口均值 | 1.5632 | 未完成 |
| 全程 50 步均值 | 1.3876 | 未完成 |

round3 逐步：1.288 1.580 1.512 1.455 1.442 1.387 1.755 1.555 1.389 1.657
round2 逐步：1.600 1.239 1.074 1.055 1.229 1.084 1.210 1.175 1.215 1.439

**round3 起点比 round2 高 0.27**，这是应该的——round3 从 `iter2_generator` 接，
round2 从 `iter1_generator` 接。**所以这 +0.27 不是本轮进步，是起点差异，不能当成
"round3 step1 比 round2 强"的证据。**

### 18 步时的对照（22:19 补，第二个窗口）

| | round2 | round3 |
|---|---|---|
| 步 1-10 reward 均值 | 1.2320 | **1.5020** |
| 步 11-20 窗口均值 | 1.1995 | **1.6540** |
| 步 21-30 窗口均值 | 1.4576 | 进行中 |
| 步 31-40 窗口均值 | 1.4857 | 未到 |
| 步 41-50 窗口均值 | 1.5632 | 未到 |

round3 步 11-20：1.671 1.807 1.372 1.601 1.329 1.840 1.612 1.722 1.742 1.843
round3 全程至今均值见 `/tmp/watch3.log`（n 每步增长）。

**同样的话再说一遍，免得 32 步后看到数好看就昏头**：步 11-18 比 round2 同期高 0.42，
这里面**绝大部分还是起点差异**（round3 从 iter2_generator 接，round2 从 iter1_generator 接）。
本轮进步的干净量仍然只有**步 41-50 窗口 vs 1.5632** 那一个。
步 11-18 已经站上 1.5632 只说明"起点高的优势还没被磨平"，
**不等于**"这一轮还在涨"——要等后 32 步的窗口均值。

step_time 中位 208s（round2 同期 228.2s），RSS 2.04 GiB，GPU 9918 MiB，无 spill 迹象。

#### 分量分解：round3 的优势摊在三个分量上，不是某一个被撬起来（22:28，n=22）

分解闭合已验证：三项相加 **1.5954** = reward 均值 **1.5954**，差 0.000000
（再次实证聚合是 1:1:1 等权相加，`W_*` 常量确实是死代码）。

| 分量 | round2 全程 | round3 至今（n=22） | round3 占比 | round2 占比 |
|---|---|---|---|---|
| format | 0.7964 | **0.8681** | 54.4% | 57.4% |
| validity | 0.4497 | **0.5597** | 35.1% | 32.4% |
| curriculum | 0.1415 | **0.1676** | 10.5% | 10.2% |

**三个分量全都比 round2 高，但占比几乎没变。** 这是"起点整体更高"该有的样子；
如果某一个分量被单独撬起来（比如 curriculum 占比翻倍），那才要怀疑是某一通道被过拟合。
目前没有这个迹象。

**顺便把 22:07 那次核销里我没分清的一层说清楚**：format 占总 reward 的 **54.4%**，
这是"对 reward **水平**的贡献"；而它对 **advantage** 的贡献要看它自己的方差——
round3 至今 format 实测在 **0.7656 ~ 0.9583** 之间动（极差 0.1927，n=22），有方差，
所以它**不是**常数项。22:07 我说"format 接近常数所以对 advantage 贡献近 0"那句
**只对 step3 成立**（`round2.step3.log` 的 `format_reward_func` 50 步里大多是 1.0），
对 step1 不成立。step1 的 format 仍有梯度信号，这就是它占 54.4% 却仍在动的原因。

本轮进步的干净量只有一个：**round3 步 41-50 窗口均值 − round2 步 41-50 窗口均值（1.5632）**。

#### 22:41：事前预期第 1 条**眼看要失败**，而我的替代解释也不成立（第六处，先记下来）

预期第 1 条说"round3 的 reward 涨幅该明显小于 round2 的 +0.4049"。
29 步时按同口径（都取前 29 步）做 OLS：

| | 截距 | 斜率/步 | t | 拟合涨幅 |
|---|---|---|---|---|
| round2 前 29 步 | 1.1369 | +0.01064 | +2.05 | +0.3085 |
| **round3 前 29 步** | **1.4399** | **+0.01239** | **+2.93** | **+0.3592** |
| round2 全程 50 步 | 1.1558 | +0.00909 | +4.30 | +0.4545 |

**round3 的斜率比 round2 更陡、t 更显著。**按这个斜率，round3 的涨幅不会小于 round2。
**这条事前预期正在被数据否掉。**

**而且我 22:22 给的那个替代解释也不成立**：我当时写"真正该用的理由是 round3 起点在 1.50
而不是 1.21，同样梯度下剩余空间更小"。
**"起点高所以斜率该更平"这句话我没有任何证据**，它只是听起来合理。
数据说反了：round3 起点高 0.30、斜率还更陡。
（一条可能的解释：round3 的截顶频率只有 round2 的四分之一，被截断的样本不再拖累梯度——
这和 22:40 发现的 clipped_ratio≡截顶 对得上。但这是**假设，不是结论**，要先验证才能写。）

**现在必须守住的边界**（免得把 generator 的进步误当成用户要的那个"多轮有用"）：
- 上面这些全是 **step1 的 generator reward**，是手段不是目的。
- 用户的验收标准是 **iter3_solver > iter2_solver**，在固定 135 题 holdout 上，
  用 `post_round.sh 3 2` 的配对检验判。generator 涨不涨、涨多快，都不直接算数。
- 所以 round3 现在"看起来很好"这件事，**一句都不能写进结论**，要等 solver 的数。

**还没结束，先不判**：29/50，剩下 21 步可能把斜率压平（round2 自己就是从
前 29 步的 0.01064 掉到全程的 0.00909）。等 50 步跑完再定这条预期算不算被否掉。

##### 23:26 结案：这条预期**通过了**，而 22:41 那个"眼看要失败"是只看前 29 步的窗口伪影（第七处自我更正）

> **这一节的表是 45/50 的快照，23:36 已有 50 步完整数据，见下一节。**
> 两节的结论一致（都通过），但数字不同：45 步时末6窗 1.7454/窗差 +0.3013/步41-45 均值 1.7046，
> 50 步时是 1.7598/+0.3157/1.7201。**报数用 50 步那份。**

45/50 时的同口径对比：

| | 首6步 | 末6步 | **窗差** | 前29步斜率 | 全程斜率 | 步41-50窗均值 |
|---|---|---|---|---|---|---|
| round2 | 1.2134 | 1.6483 | **+0.4348** | +0.01064 | +0.00909 | 1.5632 |
| round3（前45步） | 1.4441 | 1.7454 | **+0.3013** | +0.01239 | **+0.00657** | 1.7046 |

**结论：事前预期第 1 条的两半都成立。**
1. 「round3 后 10 步 reward 若还高于 round2 终点算继续涨」，钉死的基线是 1.5632 →
   round3 步 41-45 均值 **1.7046 > 1.5632（+0.1414）**。窗口内最差单步 1.5275
   也只比基线低 0.036。**通过。**
2. 「涨幅该明显小于 +0.40」→ 窗差 **+0.3013 < +0.4348**。**通过。**

**22:41 我错在哪**：我拿"前 29 步斜率"当涨幅的代理量，而**涨幅真正写下来的判据是首末 6 步窗差**。
round3 的斜率衰减比 round2 剧烈得多（0.01239 → 0.00657，衰减 47%；round2 只衰减 15%），
所以前 29 步看着更陡、全程看却是更平。**窗口选错会让"更陡"和"更平"同时成立**——
这不是数据反常，是我的口径漂移。而且 22:22 那个被我自己否掉的
「起点高 → 剩余空间小 → 涨幅小」的解释，**恰恰是数据最后支持的方向**，
我 22:41 说它"被数据说反了"也说反了。今天第七处自我更正，方向上和 22:22 一致。

**仍然守住的边界**（这条不因为预期通过而松动）：上面全是 step1 的 generator reward，
是手段不是目的。验收标准仍是 **iter3_solver > iter2_solver**，固定 135 题 holdout，
`post_round.sh 3 2` 配对检验。round3 step1 数字再漂亮，一句都不能进结论。

##### 23:36 step1 收摊，50 步完整数据（事前预期第 1 条**两半都通过**）

`round3.step1.ok` 落盘，`iter3_generator/checkpoint-50/model.safetensors` 988 MB 在位
（**merge 成功**，run_round.sh:159 那道 `die` 关过了）。同口径终值：

| | round2 | round3 | 判据 |
|---|---|---|---|
| reward 首6步 | 1.2134 | 1.4441 | |
| reward 末6步 | 1.6483 | 1.7598 | |
| **首末窗差** | **+0.4348** | **+0.3157** | 预期"明显小于 +0.40" → **过** |
| OLS 斜率 | +0.00909 (t=4.30) | +0.00566 (t=3.28) | round3 更平 |
| **步41-50 窗均值** | **1.5632** | **1.7201** | 预期"高于 1.5632 算继续涨" → **过（+0.1569）** |
| 撞顶步 | 7/50 (14%) | 1/50 (2%) | |
| `mean_length` 首6→末6 | 1353→1044 | 1103→688 | round3 更短 |

**分量分解（三个分量的增益之和精确等于 reward 窗差，这是个免费自检）**：

| 分量 | round2 首6→末6 | 增益 | round3 首6→末6 | 增益 |
|---|---|---|---|---|
| format | 0.7457→0.8698 | +0.1241 | 0.8238→0.8993 | +0.0755 |
| **validity** | 0.3458→0.5687 | **+0.2229** | 0.5042→0.6646 | **+0.1604** |
| curriculum | 0.1219→0.2097 | +0.0878 | 0.1162→0.1959 | +0.0797 |
| 合计 | | **+0.4348** | | **+0.3157** |

round3 的增益摊在三个分量上，**最大单项还是 validity（占 50.8%）**，和 round2 一样
（round2 是 51.3%）。validity 的绝对水平 round3 全程 0.60 vs round2 0.45——
**round3 的 generator 出的题更"有效"**，这是比 reward 总数更直观的一条。
格式那条按 22:07 的核销已经知道不是问题，此处 +0.0755 只是跟着涨。

**一句都不能外推**：以上全部是 generator 侧的。solver 侧的数要等 step3 + `post_round.sh 3 2`。

#### 事前预期第 1 条的"截顶机制"前提，22:22 核出来是**不成立的**（第三处自我更正）

> **22:40 再更正一次（第五处，也是今天最重要的一处）**：
> 我在 22:22 写"机制那句不成立"，**这个否决本身下过头了**。
> 截顶机制**是真的**，`clipped_ratio` 和截顶是**严格一一对应**的。详见本节末尾 22:40 那段。
> 22:22 否掉的只是我引用的**数字**（OLS 拟合端点），不是机制。

预期第 1 条写的是"round2 的 +0.4049 里相当一部分来自 `clipped_ratio` 从 0.0156 掉到 0.0000
的截顶机制，round3 没有这个机制可用"。逐窗口解完两轮的 `completions/clipped_ratio`：

| 窗口 | round2 均值 | round3 均值 |
|---|---|---|
| 步 1-10 | 0.01250 | 0.00000 |
| 步 11-20 | 0.01875 | 0.00694（n=9） |
| 步 21-30 | 0.00000 | 未到 |
| 步 31-40 | 0.00625 | 未到 |
| 步 41-50 | 0.00625 | 未到 |

**两件事和预期写的不一样：**
1. **round2 的 clipped_ratio 没有"掉到 0.0000"**。全程 7 步非零（步 3/4/15/16/17/37/42），
   值全是 0.0625，末窗均值 **0.00625**，不是 0。"0.0156→0.0000"这个数对不上任何窗口。
2. **round3 的 clipped_ratio 和 round2 同量级**，不是"没有了"。步 11 出现一次 0.0625，
   步 11-19 窗口均值 0.00694，与 round2 同窗的 0.01875 同量级。

**所以这条预期的结论（round3 涨幅该小于 round2）可能仍然对，但它给的机制是错的**——
不能再说"因为截顶机制没了"。真正该用来解释的是：**round3 的起点已经在 1.50 而不是 1.21**，
同样的梯度下可涨的空间本来就小。这是起点差异，不是机制差异。

（22:21 在 PLAN.md 里已经把这条预期的"只是部分兑现"写成"偶尔出现"，
现在看连"偶尔出现"都说过头了——round3 至今 1 次、round2 全程 7 次，频率确实更低，
但均值同量级。照实记：频率低、量级同。）

另一个要留意的：round3 前 10 步的抖动比 round2 大（min 1.288 / max 1.755，
极差 0.467；round2 同期 min 1.055 / max 1.600，极差 0.545 其实更大）。
**别读单步**，round2 自己第 1 步 1.6000、第 2-4 步就掉到 1.24/1.07/1.06。

**截顶机制的实测（21:58 写于 13 步时，22:22 在 19 步时补正）**：
21:58 我看到 round3 前 12 步里有 1 步 `clipped_ratio = 0.0625`，于是写"这条预期只是部分兑现：
不是完全没了，是偶尔出现"。**19 步时把这个判断也收紧了**：

| | round2 全程 | round3 至今（19 步） |
|---|---|---|
| 非零步数 | 7 / 50 | 1 / 19 |
| 非零时的取值 | 全是 0.0625 | 0.0625 |
| 步 11-20 窗口均值 | 0.01875 | 0.00694（n=9） |

**频率确实更低（7/50 vs 1/19），但量级相同。**"偶尔出现"这个说法对 round3 成立，
可它隐含"round2 是经常出现"，而 round2 其实也是偶尔（14% 的步）。
真正的问题是**事前预期里的"0.0156→0.0000"本身不成立**（OLS 拟合端点、t=−1.25）。
~~所以整条"截顶机制用完"的叙事从头就不该有。上面 22:22 那段已作废。~~
（22:40：这句又说过头了，见下。）

##### 22:40 第五处更正，也是今天最重要的一处：**截顶机制是真的，我 22:22 否决错了对象**

28 步时我把两轮的 `completions/clipped_ratio` 和 `completions/max_length`
**逐步对齐**比了一次，结果非常干净：

| | round2（n=50） | round3（n=28） |
|---|---|---|
| `clipped_ratio > 0` 的步 | {3, 4, 15, 16, 17, 37, 42} | {11} |
| `max_length ≥ 4090`（撞 4096 顶）的步 | {3, 4, 15, 16, 17, 37, 42} | {11} |
| 两个集合相同？ | **是** | **是** |
| 非撞顶步里 `clipped_ratio` 的最大值 | **0.0** | **0.0** |

**`clipped_ratio > 0` 与"这一步有 completion 撞了 4096 上限"是严格等价的**，
两轮都成立，且非撞顶步的 clipped_ratio 恒等于 0.0（不是"接近 0"，是精确 0）。

**所以：**
1. **"截顶驱动 clipping"这个机制是真的**，我 22:22 说"机制那句不成立"是否决错了对象——
   我否掉的只是我引用的数字（OLS 端点），机制本身站得住。
2. **`clipped_ratio` 是一个免费的截顶探测器**，不用去解 `max_length` 也知道哪步被截了。
   这件事有实用价值：round4 起想监控"generator 是不是又开始写长"，看 clipped_ratio 就够。
3. **round3 的截顶频率是 round2 的四分之一**（1/28 = 3.6% vs 7/50 = 14%），
   因为 round3 的 completion 明显更短。所以事前预期第 1 条的**结论**
   （round3 涨幅该明显小于 round2 的 +0.4049）**既有起点差异的原因、
   也有截顶机制贡献变小的原因**——两个原因都真，我 22:22 把第二个原因误杀了。

**这一处的形状和前四处不同，所以单独记**：前面是"归属错 / 对象错 / 拟合线当数据"，
这次是**"检查当时为真的快照，被我写成了永恒结论"**。
22:07 我看前 15 步 `max_length` 最高 2165，就写了"离 4096 差一半，根本没有截顶"；
到 28 步时它撞了 4096。**"我查过"不等于"查够了"**——
尤其当被查的量本身还在随训练变化时（mean_length 从 1103 一路降到 774）。

##### 23:32 用完整数据复核，一一对应比 28 步时更干净（不是新更正，是加强）

step1 跑到 47 步时重跑同一次对齐，**两轮都零反例**：

| | round2（n=50） | round3（n=47） |
|---|---|---|
| `clipped_ratio > 0` 的步 | 7 步 | 1 步 |
| 这些步的 `max_length` | **全部 = 4096** | **= 4096** |
| 这些步的 `clipped_ratio` | **全部 = 0.0625** | **= 0.0625** |
| `clipped_ratio = 0` 的步 | 43 步 | 46 步 |
| 零值步的 `max_length` 最大 | 3437 | 4009 |
| 零值步里 `max_length ≥ 4090` 的 | **0 步** | **0 步** |

比 22:40 那次多出来的一条：**截顶步的 clipped_ratio 恒等于 0.0625 = 1/16**，
`max_length` 恒等于 4096。也就是说"这一步被截了"永远是
**16 个 completion 里恰好 1 个撞顶**，不是 2 个也不是半个。
这让 clipped_ratio 从"布尔探测器"升级成**计数器**：`round(clipped_ratio × 16)` 就是撞顶条数。

**顺带修正上面第 3 条的一个措辞（第八处自我更正，而且形状和 22:22 完全一样：把对的改错了）**：
我 23:32 先写了"round3 真正短的是 mean 不是 max，所以'短所以不撞顶'这个因果是错的，别再这么写"。
**这句更正本身是错的**，完整数据摆在这：

| | round2 | round3 |
|---|---|---|
| `mean_length` 首6 → 末6 | 1353 → 1044 | 1103 → 747 |
| `mean_length` 全程均值 | 1120 | **885** |
| `max_length` 全程均值 | 2382 | **1804** |
| 非撞顶步的 `max_length` 极值 | 3437 | 4009 |
| 撞顶步数（/ 16×步数） | 7 / 800 | 1 / 768 |

**round3 在 mean 和 max 的均值上全都更短**（各低 21% / 24%），
所以 22:40 写的"round3 的 completion 明显更短 → 截顶频率低"**方向是对的，数据支持**。
我错在只看了"单条极值 4009 > 3437"这一个数就宣布因果不成立——
**一个方向的极值不能否证一个分布层面的结论**，这和 22:41 拿"前 29 步斜率"否证"全程涨幅"是同一个错误。

**但原措辞确实有一处要收紧**：不能写成确定性因果。round2 的 7 次撞顶散布在步
{3,4,15,16,17,37,42}，其中 5 次在前 17 步——而 round2 是 lr=1e-4 **冷重启**、早期方差大，
round3 是从 round2 终点接着训。所以"更短"和"更稳"两个机制都在往同一个方向推，
现有数据分不开哪个贡献多少。**正确写法是"截顶频率低，和 completion 整体变短一致"，
不是"因为变短所以不撞顶"。**




---

### 23:52 round3 漏斗预测：先把"两个混淆源"钉下来（看到数之前）

iter2 curriculum 的 agreement 基线，从 `iter2_data.json` 原文件算（不用记忆里的数）：

| | |
|---|---|
| n | 100 |
| mean solver_agreement | **0.5320** |
| min / max | 0.2 / 0.9 |
| mix easy/medium/hard | 21 / 49 / 30 |
| 分布 | {0.2:19, 0.3:11, 0.4:9, 0.5:11, 0.6:12, 0.7:17, 0.8:10, 0.9:11} |
| fallback（缺 agreement） | 0 |

`iter_funnel.jsonl` 里 iter2 那行还写着 `"verified_by": "iter1 solver"`。

**预测（跑之前写下来）**：round3 curriculum 的 mean agreement > 0.5320 且 hard < 30。

**但这个预测有两个混淆源，看到数之后不许只认第一个：**

1. **阅卷的 solver 变了**（iter1 → iter2）。iter2_solver 比 iter1_solver 更会点名工具，
   `normalize_call_names_only` 只比 `c.get("name")`，所以同一批题 agreement 会涨。
   这是已知的自指退化：solver 变强 → hard 池变小 → 训练信号变少。
2. **题目也换了**。round3 的 1149 条是 iter3_generator 新生成的，不是 iter2 那 1118 条。
   step1 的 validity reward 推着 generator 写"更合规"的题，如果它顺带把 gold call 的
   工具名写得**更规范**（更贴近 solver 见过的那种叫法），agreement 同样会涨——
   而 solver 一点都没变强。

**所以 agreement 均值上涨本身不能算"solver 进步"的证据。** 判读规则：

- round3 的 mean > 0.5320 **且** solver 侧 held-out 指标也涨 → 与两个混淆源都一致，
  这时 agreement 只当"课程确实在变简单"的运营信号看，**不当 solver 质量指标**。
- round3 的 mean > 0.5320 **但** solver 侧 held-out 没涨 → 涨的是 generator 侧
  （题变规范了），难度信号已经和 solver 能力脱钩，比自指更麻烦。
- round3 的 mean ≤ 0.5320 → 两个混淆源都没压过课程本身的难度， hardest 那条路仍然通。

（和 [[selfplay-reward-components-that-call-the-model]] 同一条：会调当前模型的量是跟踪
指标，不是质量指标。solver_agreement 就是这样一个量。）

#### 补：iter2 的"筛选前"基线（比最终 100 条更能看出分布变化）

只看最终 100 条会把 select 的配额效应混进"分布变化"。select 本来就挑难的
（30% hard 配额），所以筛选后 mean 必然低于筛选前。iter2 两个口径都算出来：

| 口径 | 条数 | mean agreement | 分布 | 分档 hard/medium/easy |
|---|---|---|---|---|
| **筛选前**（`iter2_data.json.intermediate.verified.json`） | 918 | **0.7739** | {0.2:55, 0.3:36, 0.4:47, 0.5:51, 0.6:56, 0.7:85, 0.8:112, 0.9:144, **1.0:332**} | 91 / 239 / 588 (9.9%/26.0%/64.1%) |
| 筛选后（`iter2_data.json`） | 100 | 0.5320 | {0.2:19, 0.3:11, 0.4:9, 0.5:11, 0.6:12, 0.7:17, 0.8:10, 0.9:11} | 30 / 49 / 21 |

漏斗链（iter_funnel.jsonl）：1118 生成 → tau_verify=0.2 剔掉 200 → 918 →
`drop_agreement_ge 1.0` 剔掉 **332 条（36.2%）** → 586 → select 100。

**所以 round3 要同时报两个口径**，两个基线分别是 0.7739（前）和 0.5320（后）。

**顺带一个运营风险，现在就说清楚**：iter2 的筛选前池子里 36.2% 是饱和样本。
iter2_solver 比 iter1_solver 强（这正是 iter1→iter2 的意义），所以 round3 的
`1.0` 占比大概率更高、剔饱和后剩下的池子更小。池子一小，select 的每域配额
（32 域 × 3）就凑不齐，padding 会把 medium/easy 补进来，最终 mix 就会偏离
30/50/20 —— 这**不是 select 写错了，是自指退化的正常表现**。
所以判据写成"hard < 30 且池子缩小"算退化，"hard < 30 且池子没缩小"才算真变简单。

#### 快照到底证明了什么（23:55 校准）

round3 起跑时间 `round3.log` = **21:10:13**。`run_round.sh` mtime = 20:33:55，
在起跑**之前**，所以快照抓到的是编辑后的版本，也就是 round3 真正执行的那一版。

`round_snapshots/round3.20261015_232914/` 与当前文件逐个 diff：**11/12 相同**。

唯一不同的是 `launch_round.sh`——它是我在 round3 起跑之后改的（加 snapshot 调用 +
无条件接挂看守）。它不在 round3 的执行路径里（run_round.sh 起来之后它只负责起跑和看守），
所以不影响 round3 的任何训练行为。

**所以这两件事要分开说，别混成一句：**

- ✅ 快照证明了：**round3 跑的是哪一版代码**。这一条现在有实据，不再靠 mtime 推断。
- ❌ 快照没有证明：**这一版相对于 20:33:55 之前改了什么**。那次编辑的内容无从恢复，
  iter2→iter3 的对照里这一行仍然要单独列，不能算干净。

（round 4 起 `launch_round.sh` 会在 `run_round.sh` 起跑**之前**调 snapshot_round.sh，
拿到的是真正的"起跑前"快照，顺序才对。round3 这份是起跑后补的，靠 mtime 早于起跑时间
才等效。）

#### 23:56 污染门提前过（比 post_round.sh 查得更宽）

`post_round.sh` 的污染门查的是最终 100 条（`iter3_data.json`），而 select 还没跑。
但 1149 条的生成池已经落盘，所以**现在就能查，而且查的是超集**：

```
固定评测集：135 条
  iter3_data.json.intermediate.json  1149 条  干净      rc=0
  对照 iter2_data.json.intermediate.json  1118 条  干净   rc=0
```

最终 100 条是这 1149 条的子集，子集干净 ⟸ 全集干净。所以 round3 的
`check_holdout_clean.py` 一定过，不用等。

**为什么值得单独记**：subset 干净能推出的结论比"我跑过门而且过了"强。
`drop_saturated` 和 select 都是在同一批题里挑，不会引入新题，所以这个推导是闭的。

#### 23:58 iter2→iter3 的判据上限（从 iter_metrics.jsonl 实抄）

| tag | ts | sol_acc | sol_acc_pos |
|---|---|---|---|
| base_solver（设种） | 20:36:42 | 0.0285 | 0.0519 |
| iter1_solver（设种） | 20:32:56 | 0.4782 | 0.6889 |
| iter1_solver_rep2 | 20:34:50 | 0.4782 | 0.6889 |
| **iter2_solver** | 21:05:07 | **0.6380** | **0.8889** |

（未设种那两行 base 0.0207 / iter1 0.4453 按 compare_rounds 的去重规则不留；
`iter1_solver_rep2` 与设种行完全相同 = 噪声底 0，已单独量过。）

实测算出来的增量：

| 区间 | sol_acc | sol_acc_pos | name_hit |
|---|---|---|---|
| base → iter1 | +0.4497 | +0.6370 | — |
| **iter1 → iter2** | **+0.1598** | **+0.2000** | **+0.2148** |

**所以 iter2 → iter3 的事前判据是：name_hit 涨幅 < +0.2148，acc 若还在涨应主要来自
条件 acc（0.726 → 更高）。**

**天花板账（现在就算，免得看到数再圆）**：sol_acc_pos = 0.8889 = **120/135**，
只剩 15 条没拿到任意部分分。也就是说 **iter3 的 acc_pos 涨幅上限是 +0.1111**，
比 iter1→iter2 的 +0.2000 小一半。sol_acc = 0.6380，上限 +0.3620。

**因此"多轮上升"这件事在 round3 之后必然越来越难，而且瓶颈大概率在 acc_pos 先见顶。**
这不是 round3 会不会失败的问题，是这套 self-play 的容量/任务上限问题——
判据里要分清"没涨（方法无效）"和"涨不动了（指标见顶）"：
- sol_acc 涨幅 < 0.05 **且** name_hit 涨幅 < 0.05 → 只有这一种算撞墙。
- sol_acc_pos 涨幅接近 0 但 sol_acc 还在涨 → 正常，是 acc_pos 先饱和。

---

## 00:00 round3 漏斗：预测过了，但是以退化那种方式过的

### 数字

| 口径 | iter2（iter1_solver 阅卷） | iter3（iter2_solver 阅卷） |
|---|---|---|
| 生成 | 1118 | 1149 |
| tau_verify=0.2 之后 | 918 | 952 |
| **筛选前 mean agreement** | 0.7739 | **0.9136** |
| 饱和（ag=1.0） | 332（36.2%） | **656（68.9%）** |
| 剔饱和后剩 | 586 | **296** |
| **hard 池（ag<0.4）** | 91 | **30** |
| 最终 mix e/m/h | 21 / 49 / 30 | 33 / 45 / **22** |
| 最终 mean | 0.5320 | **0.6060** |

`[select] mix: 想要 easy 20% / medium 50% / hard 30%  实际 easy 33% / medium 45% / hard 22%`

### 事前判据的判决

事前写的是：mean > 0.5320 且 hard < 30。
**实测 0.6060 > 0.5320，22 < 30。预测成立。**

但同一条事前规则里还写了后半句：
> "hard < 30 且池子缩小"算退化，"hard < 30 且池子没缩小"才算真变简单。

**池子从 586 掉到 296（−49%），hard 池从 91 掉到 30（−67%）。所以按我自己事前定的
规则，这是退化，不是"课程变简单了"。**

而且比"退化"更具体：**hard 池 30 条，配额正好 30 条，余量为零。**
iter2 时 hard 池是配额的 3 倍（91 vs 30），select 还能挑；
round3 是 1 倍，select 只能全收，再按 32 个域一分就凑不齐，
于是 padding 把 easy 补进来 —— 最终 33 个 easy（想要 20）就是这么来的。
**mix 偏离 30/50/20 不是 select 写错了，是 hard 池见底的直接读数。**

### 按这个速率，round4 会是什么样

饱和率 36.2% → 68.9%（+32.7pp），hard 池 −67%。
线性外推一轮：饱和 ~100%，**hard 池归零**。
那不是"配比难看"，是**难度信号整个消失**——select 没有任何 hard 可放，
`reward_curriculum` 的 difficulty 分量失去意义。

（外推只能当预警看，不是预测：它假设 generator 和 solver 同步变强，
而 round4 起我打算改 agreement 的匹配口径，那会打断这个趋势。见下。）

### 第二个混淆源坐实了：不是我"可能"，是结构上就是

23:52 我写"题目也换了，validity 可能顺带把工具名写得更规范"，
当时标的是猜测。现在读 `rewards_generator.py` 确认这不是猜测：

`reward_validity_accuracy`（`:1293`）= 三项加权和，第一项是
`reward_gold_tool_exists`（`:708`）：

```python
tindex = tool_index(parsed["tools"])
gold_name = parsed["gold"]["name"]
if gold_name in tindex:
    rewards.append(1.0)
```

**"gold call 的工具名在 available_tools 里"直接给 1.0。**
而 `solver_agreement` 量的是"solver 能不能吐出同一个工具名"
（`normalize_call_names_only` 只比 `c.get("name")`）。

**所以 generator 的 validity 奖励和难度判据吃的是同一个量。**
step1 的 validity 从 round2 的 0.45 涨到 round3 的 0.60，
agreement 跟着从 0.7739 涨到 0.9136 —— 这两个数同步上涨有直接机械原因，
不需要假设任何"变强"。

**必须说清楚的边界**：`gold_tool_exists` 是 agreement 的**必要不充分**条件
（工具名存在 ≠ solver 能吐出它）。所以这个机制解释了耦合的存在，
但**不能定量拆分**"agreement 涨的里边多少是 generator 写规范了、多少是 solver 变强了"。
要拆得做那个控制实验：**拿 iter2 那 1118 道原题，用 iter2_solver 重新阅卷**。
同题、只换 solver，出来的差就是 solver 单独的贡献。这个要 GPU，
round3 的 step3 正在用，列为待办。

### 一个正向印证

原定的 round4 改动是把 agreement 的匹配从 `normalize_call_names_only`
（只比工具名）换成 `normalize_call_json`（工具名 + 参数全等）。
**这会让我上面刚坐实的这条耦合变弱**：全等严格更难满足，
agreement 会被压下来，hard 池跟着回来。

改动前它是"fallback 设计变更 #1"，改动后它是**针对已坐实的退化机制的修复**。
优先级从"可选"提到"必做"。

---

## 00:07 本轮最重的发现：难度和饱和两个信号都只看工具名，不看参数

### 怎么发现的

看 `iter3_data.json.intermediate.verified.json` 第一条：

```
gold calls:              [{"name": "get_user_info", "arguments": {"userId": "johnsmith"}}]
solver_predicted_calls:  [{"name": "get_user_info", "arguments": {"userId": "john_smith"}}]
solver_agreement:        1.0
solver_disagree_reason:  'ok'
```

工具名对，参数值错（`johnsmith` vs `john_smith`）。流水线判 **agreement=1.0、reason=ok**，
于是 `drop_agreement_ge 1.0` 把这条当"solver 已经全会"**丢掉**。

### 量化（CPU 可算，两轮都算）

| | iter2 池 | iter3 池 |
|---|---|---|
| 总条数 | 918 | 952 |
| 结构不可比（gold/pred 不是规范 list-of-dict） | 2 | 0 |
| **被当饱和丢掉的（ag≥1.0）** | 331 | **656** |
| ├ 名对、**参错**（全等口径下不该丢） | 153（46.2%） | **296（45.1%）** |
| └ 名+参都对（真的该丢） | 178（53.8%） | 360（54.9%） |
| 存的那条预测：工具名命中 | 735（80.1%） | 866（**91.0%**） |
| 存的那条预测：名+参全等 | 319（34.7%） | 431（**45.3%**） |

**iter3 有 296 道题是被误丢的。这是硬下界，不是估计**，理由：

`step2_genverify.py:284-286` 是
```python
if first_valid_pred is None:
    first_valid_pred = pred
...
preds[idx] = first_valid_pred
```
存的是 k=10 里**第一个能解析成 JSON 的**样本。而 `ag=1.0` 的定义是
`ok == k`，即 10 个样本**全部**点名正确。所以存的这条必然也点名正确
（我的复算印证了：656 条里"名就没对"的是 **0** 条）。
它参数错 ⟹ 这 10 个里至少有一个参数错 ⟹ 全等口径下 ok_full < 10
⟹ agreement < 1.0 ⟹ **过不了 `drop_agreement_ge 1.0`，本该留在池子里**。

**自检**：296+360=656、153+178=331，闭合；且"ag=1.0 但名没对"= 0 条，
说明我重写的名匹配与流水线 `normalize_call_names_only` 在这 987 条上完全一致。

### 这意味着什么

**任务是要产出正确的工具调用，而 pipeline 的难度判据和饱和判据都只看到工具名。**

1. `solver_agreement` 把 solver 的能力**放大了约 2 倍**：同一条预测，
   按名算命中 91.0%，按名+参算只有 45.3%。
2. `drop_agreement_ge 1.0` 因此**系统性地扔掉最有训练价值的样本**——
   "名对参错"正是"差一点就会"的那批，梯度信息最密集。
   iter3 一轮就误扔 296 条，占剔饱和后剩余池子（296 条）的 **100%**。
3. 上一节记的"hard 池从 91 掉到 30"，**至少有相当一部分是这个口径问题造成的，
   不全是 solver 变强**。我 23:52 写的第二个混淆源（generator 侧）是真的，
   但当时不知道还有第三个、而且是更大的一个：**判据本身漏掉了参数**。

### 对 round4 的改动：从"可选"变成"第一件事"

原计划是把 agreement 的匹配从 `normalize_call_names_only` 换成
`normalize_call_json`（`step2_genverify.py:231` 和 `:269` 两处）。
现在这一条的定位完全不同了：

- 改动前：它是 fallback 设计变更 #1，"让难度信号更严格"。
- 改动后：**它修的是一个已量化的缺陷——误丢 45% 的饱和样本、
  把 solver 能力夸大 2 倍、并直接造成 hard 池见底。**

**预期效果（可以事后验）**：换成全等口径后，
① 饱和集合至少少 296 条（iter3 等价规模），池子从 296 涨到 ≥592；
② `tau_verify=0.2` 改用在全等命中率上，会比现在严得多，
   会额外掉一批"名对参错"的题 —— 这一项目前**方向不确定**，
   可能把池子又压回去，所以净效果必须实测，不能预告。

### 待办（要 GPU，round3 step3 用完）

拿 iter2 那 1118 道原题、用 iter2_solver 重新阅卷，只换 solver 不换题，
拆出"solver 单独贡献了多少 agreement 涨幅"。这是把
generator 侧 / solver 侧 / 口径侧三个混淆源分开的唯一干净实验。

### padding 是怎么发生的（按域拆）

| | iter2 | iter3 |
|---|---|---|
| 域数 | 32 | 32 |
| 每域条数 | 3×28 + 4×4 | 3×26 + 4×5 + **2×1** |
| **没有 hard 的域** | **2/32** | **12/32** |
| 每域 hard 数分布 | {0:2, 1:30} | {0:12, 1:18, 2:2} |
| 最终 hard | 30（=配额） | **22（<配额 30）** |

iter3 的 hard 池正好是 **30 条**（ag=0.2 的 16 条 + ag=0.3 的 14 条），
数量上刚好够配额。但它们**只落在 20 个域里**，12 个域一条 hard 都没有，
而 select 是先分域再分档、每域最多 3 条，所以 30 条池子只能放进 22 条。

**所以"hard 22 < 30"不是池子不够，是池子的域分布不够散。**
iter2 时 30 个域各贡献 1 条，正好铺满；iter3 铺不满。

---

## 00:20 round4 的核心修复已落地：agreement 拆成名口径和全等口径

### 改了什么

`step2_genverify.py`：
- `solver_agreement_batch` 现在返回**两个**命中率：`rates`（只比工具名，
  历史口径）和 `rates_full`（工具名 + 参数全等，`normalize_call_json`）。
- `meta` 里同时写 `solver_agreement` 和 **`solver_agreement_full`**。
- 新增 drop reason `args_mismatch`（名对参错），报告里加一段全等口径的分布，
  并直接打出"名口径=1.0 但全等<1.0"的条数——这个数 round3 是 296，
  **以后每轮报告第一眼就能看到，不用再手工复算。**

`step2_select_curriculum.py`：
- `drop_saturated` 改用 `solver_agreement_full`。
- `assign_difficulty_from_agreement` 改用 `solver_agreement_full`，
  `judge_raw` 前缀改成 `agreement_full=`。
- 老数据（没有 full 键）自动退回名口径，并在 `judge_raw` 里加
  `fallback=names_only` 痕。

**`tau_verify` 门故意不改**，仍用名口径。理由：那个门问的是"这题碰不碰得到"，
而"名对参错"正是要训的那批；用全等去卡门会把最该训的题滤掉。
三个用途分开：门用名、剔饱和用全等、分档用全等。

### 两个测试，都是零 GPU

`test_agreement_full.py`（22 项，假 LLM）：验计数语义。
`test_select_agreement.py`（12 项，真跑 select 子进程）：验分档路径。

**测试抓到两个真 bug，都不是"跑一遍看看"能发现的：**

1. `if ok < k and local:` 漏了 `ok_full < k`。k 个样本全点名正确但有几个参数不对时，
   `ok == k`，reason 停在 `"ok"`——而"名对参错"正是这次要抓的情况，
   会被藏起来。改成 `if (ok < k or ok_full < k) and local:`。
2. `assign_difficulty_from_agreement` 末尾无条件重写 `judge_raw`，
   把 `drop_saturated` 那边加的 `fallback=names_only` 痕冲掉了
   （`drop_saturated` 在 main 里先跑，:823 vs :839）。
   留痕改成只在这一处加。

### 真数据回归（这条最要紧）

用 iter2 的 `intermediate.verified.json`（只有名口径）跑新 select：

```
[saturation] dropped 332 ...; 586 left, 0 of them carry a full-match value
            (586 carry the legacy names-only value)
[difficulty] ... 586 条有命中率，0 条缺失
[select] diff: {'easy': 21, 'medium': 49, 'hard': 30}
```

和当年 `iter2_data.json` 逐条比：**100 条题目相同、顺序相同、(题,档位) 全部相同。**
`judge_raw` 从 `agreement=0.80` 变成 `agreement_full=0.80 fallback=names_only`
——值是名口径那个 0.80，痕迹说明它是退回判的。

**所以这次改动对历史数据是严格无行为的**（除了多留一列痕），
iter2→iter3 的旧对照不会被这次修改污染；round4 起才走新口径。

### round4 要看的（事前写下来）

换成全等口径后，两件事方向已知、一件方向未知：

- ✅ **饱和集合变小、池子变大**：round3 那 296 条会回来，
  池子从 296 涨到 ≥592。这是修复的目的，预期方向。
- ✅ **hard 池该回来**：全等口径下"名对参错"不再被算成 easy/medium。
- ❓ **`tau_verify=0.2` 仍按名口径，所以门不变**；但分档阈值 0.4/0.8 是在
  名口径的分布上标定的，全等口径整体分布会下移，
  **这两个阈值可能不再合适**——是 round4 第一个要看的数，
  别急着调，先看一遍分布再说。

### 这个 bug 的影响范围：只在课程流水线，不在评测（00:25 界定）

刚发现"只比工具名把 solver 能力放大约 2 倍"时，第一反应是"那 iter1→iter2 的
+0.1598 是不是也虚"。查完 `rewards_solver.py` 之后**不是**，两边量的不是一回事：

| | 用什么比 | 在哪 |
|---|---|---|
| `solver_agreement`（verify 关，喂给剔饱和和分档） | **只比工具名** `normalize_call_names_only` | `step2_genverify.py` |
| `sol_acc` / `acc_pos`（holdout 评测，判定"这轮有没有涨"） | **软匹配，含参数**：`compute_accuracy_score` 出 `name_scores`/`key_scores`/`value_scores`/`pair_rewards` | `rewards_solver.py:465` |
| `name_hit`（holdout 的一个辅助列） | 只比工具名 | `eval_iter.py` |

**所以 iter1→iter2 的 `sol_acc` +0.1598 是含参数的真实增益，不虚。**
评测侧一次都没被这个口径问题污染过——**判定"多轮有没有上升"的那个尺子本身是好的**，
坏的是"决定下一轮学什么"的那个尺子。

这也解释了 `sol_acc_pos` 为什么已经 0.8889：软匹配下"点名对"就能拿到部分分，
所以"拿到任意部分分"的比例天然偏高。**`acc_pos` 接近饱和不代表 solver 会做对题，
只代表它大概能点名。** 这条对 round3 的判读很重要：别把 acc_pos 见顶读成"到顶了"。

（`name_hit` 仍是名口径，但它是辅助列，判据主要看 `sol_acc`。
没有去改它——改了 iter1/iter2 的 `name_hit` 历史值就不可比了，
而这次要修的是课程侧，不是评测侧。）

### 顺带量到的 generator 天花板（00:22，gen 评测跑到 112/192 时）

固定 192 条 spec 上的 generator 指标：

| tag | gen_four_tag | gen_pv_ok |
|---|---|---|
| base | 0.0833 | 0.0052 |
| iter1（lr=1e-4 50 步） | 0.9323 | 0.4740 |
| iter2 | 0.9531 | **0.6094** |
| iter3（跑到 112/192 时） | 111/112 = 0.9911 | 70/112 = 0.6250 |

`four_tag`（四标签格式契约）已经 **0.99，只剩 0.009 空间**，再训也涨不动。
`pv_ok`（解析+有效）还有空间。

**和 solver 侧同一个形状**：先涨的那个指标先见顶。
round3 之后 generator 侧的增量只能来自 pv_ok 那一支。

### 完备性检查：这个模式还在哪用（00:24）

`grep -rn normalize_call_names_only --include=*.py`（排除 .venv 和快照）：

| 位置 | 用途 | 处置 |
|---|---|---|
| `step2_genverify.py:291` | `tau_verify` 门的命中率 | **故意保留**（门问"碰不碰得到"） |
| `eval_iter.py:198` | holdout 的 `name_hit` 辅助列 | **故意保留**（改了 iter1/iter2 历史值就不可比） |
| `step2_select_curriculum.py:176` | 只是注释里提了一句 | 无需处置 |
| `round_snapshots/round3.*` | round3 的历史留证 | **不许动** |

**`rewards_solver.py` 和 `rewards_generator.py` 里一次都没出现。**
所以训练奖励从来不是名口径的——这次坏的只是"决定下一轮学什么"的尺子，
"判定有没有学会"的尺子（`compute_accuracy_score` 软匹配）一直是好的。

---

## round4 起跑方案（00:26 定稿）

### 起跑命令

```
bash launch_round.sh 4 50
```

`launch_round.sh` 现在会在 `run_round.sh` 起跑**之前**调 `snapshot_round.sh`，
拿到真正的"起跑前"快照（round3 那份是起跑后补的，靠 mtime 早于起跑时间才等效）。

### 这一轮相对 round3 的唯一代码差异

`step2_genverify.py` + `step2_select_curriculum.py`：agreement 拆成名口径和全等口径，
剔饱和和分档改用全等。已用 22 + 12 个零 GPU 测试和一次真数据回归验证过
（新代码跑 iter2 老数据，100 条题目/顺序/档位与当年逐条相同）。

**除此之外什么都不改**——lr、步数、k_verify、tau_verify、mix 全部维持。
一次只改一件事，否则 round3→round4 又说不清是哪个改动起的作用。

### 阈值不预先调（这条是纪律）

`--agr_hard_lt 0.4` / `--agr_easy_ge 0.8` 是在**名口径**的分布上标定的。
全等口径整体分布会下移，这两个阈值很可能不再合适。
但**现在没有全等口径的分布数据**（`solver_agreement_full` 是 k=10 平均，
我手里只有每个样本存的一条预测，反推不出那个平均）。
所以 round4 的做法是：**先用默认阈值跑，第一件事看 `[difficulty]` 那行和
报告里的全等口径分布，看完再决定要不要调。** 不许凭"感觉会偏低"就改。

### round4 要看的数（按优先级）

1. **全等口径的饱和条数**：`[saturation] dropped N`。round3 等价口径是 656，
   预期大幅下降。报告里还直接打了"名口径=1.0 但全等<1.0"的条数。
2. **hard 池是否回来**：iter2 是 91 条、iter3 塌到 30 条。
   如果回到 90+ 说明修复奏效；如果还是 30 左右，说明另外两个混淆源
   （solver 变强 / generator 写得更规范）才是主因，口径只是表象。
3. **最终 mix**：目标是回到接近 30/50/20。若 easy 档反而空了
   （全等口径下几乎没有样本 ≥0.8），说明阈值该调，那是下一步的事。
4. **iter4 vs iter3 的 held-out 判据**（沿用 round3 那套）：
   `sol_acc` / `sol_acc_pos` / `name_hit` 三项配对检验 + McNemar，
   净翻转 < 10 条不算证据。

### iter3 generator 侧收摊（00:25，固定 192 spec，eval 1109.6s）

| tag | gen_four_tag | gen_pv_ok |
|---|---|---|
| base | 0.0833 | 0.0052 |
| iter1 | 0.9323 | 0.4740 |
| iter2 | 0.9531 | 0.6094 |
| **iter3** | **0.9844** | **0.6250** |
| 增量 iter2→iter3 | **+0.0313** | **+0.0156** |

两项都涨。但 `four_tag` 只剩 **0.0156** 就到 1.0——**round4 起 generator 的
格式增益基本到头**，往后只能看 pv_ok（还剩 0.375）。

（这一步是和 iter1→iter2 的 +0.0208 / +0.1354 比的：four_tag 涨幅缩小到 1/6.7，
pv_ok 涨幅缩小到 1/8.7。和 solver 侧同一个形状：越往后增量越小。）

---

# round3 收摊报告（00:30）

## 1. 六项指标全程单调不降 ✅

| tag | g.four_tag | g.pv_ok | s.fmt | s.acc | s.acc_pos | s.name_hit |
|---|---|---|---|---|---|---|
| base_gen | 0.0833 | 0.0052 | | | | |
| base_solver | | | 0.1719 | 0.0285 | 0.0519 | 0.0741 |
| iter1_gen | 0.9323 | 0.4740 | | | | |
| iter1_solver | | | 0.9007 | 0.4782 | 0.6889 | 0.5778 |
| iter2_gen | 0.9531 | 0.6094 | | | | |
| iter2_solver | | | 0.9770 | 0.6380 | 0.8889 | 0.7926 |
| **iter3_gen** | **0.9844** | **0.6250** | | | | |
| **iter3_solver** | | | **0.9822** | **0.6627** | **0.9037** | **0.8444** |

`compare_rounds.py` 对六列全部判"单调不降"。**用户要的"多轮都能上升"在这一层成立。**

## 2. 但增量在急剧缩小，且全部不显著

| 指标 | iter1→iter2 | iter2→iter3 | 倍数 |
|---|---|---|---|
| s.name_hit | +0.2148 | **+0.0519** | 24% |
| s.acc | +0.1598 | **+0.0247** | 15% |
| s.acc_pos | +0.2000 | **+0.0148** | 7.4% |
| s.fmt | +0.0763 | +0.0052 | 7% |
| g.four_tag | +0.0208 | +0.0313 | — |
| g.pv_ok | +0.1354 | +0.0156 | 12% |

配对检验（n=135，净翻转 < 10 不算证据）：

| 检验 | p | 判定 |
|---|---|---|
| gen/four_tag | 0.1460 | 不显著 |
| gen/pv_ok | 0.8264 | 不显著 |
| solver/fmt | 0.8467（置换） | 不显著 |
| solver/acc | 0.3226（置换） | 不显著 |
| solver/name_hit | 0.1892 | 不显著 |
| solver/acc_pos | 0.8238 | 不显著（+2.0 题） |

**六项配对检验全部不显著。** 均值方向都是正的，但样本量撑不起"确实变了"的结论。

## 3. 增益的真实形状：7 道题，全在最简单的三档

按 holdout 自带的固定 `solver_agreement` 分档（档位是 holdout 的固定属性，
**不拿本轮状态分组**，遵守 [[self-conditioning-breaks-round-comparison]]）：

| 档位 | n | a点名 | b点名 | a条件acc | b条件acc | a acc | b acc | acc差 |
|---|---|---|---|---|---|---|---|---|
| lt0.3 | 16 | 0.812 | **0.812** | 0.868 | 0.837 | 0.727 | 0.723 | −0.003 |
| 0.3-0.5 | 33 | 0.788 | **0.788** | 0.722 | 0.702 | 0.624 | 0.620 | −0.004 |
| 0.5-0.7 | 50 | 0.820 | **0.820** | 0.722 | 0.758 | 0.662 | 0.664 | +0.002 |
| 0.7-0.85 | 19 | 0.789 | 0.895 | 0.680 | 0.718 | 0.647 | 0.695 | +0.047 |
| 0.85-1.0 | 7 | 0.429 | **1.000** | 0.688 | 0.691 | 0.395 | 0.691 | **+0.296** |
| eq1.0 | 10 | 0.900 | 1.000 | 0.639 | 0.617 | 0.575 | 0.617 | +0.042 |

**前三档合计 99/135 题（73%）点名率一个字都没动。**
整个 name_hit 的 +0.0519 = **7 道题**，全在最容易的三档，全是 iter2 本来点错名的。

**最难两档（ag<0.5，49 题 = 36%）点名率零变化、acc 还各掉了 0.003 / 0.004。**

## 4. 一条我事前判错的（必须自己说）

事前判据写的是"acc 若还在涨应主要来自**条件 acc**（0.726 → 更高）"。
实测：条件 acc 只从 0.7260 到 0.7315（+0.0055），
命中支增益 +0.0423 里它只占 **11%**，另外 **89% 来自点名率**（0.7926→0.8444）。

**这条预测错了。** iter2→iter3 的 acc 增益几乎全部是"多认出几个工具"，
不是"认出来之后参数写得更准"。而参数那一支才是任务真正难的地方。

## 5. 拆东墙补西墙是真的

固定分组（按 iter2 的 name_hit 切，同一批题贴到 iter3 上）：

| 组 | n | a acc | b acc | 差 | 进步/退步 | b点名率 |
|---|---|---|---|---|---|---|
| iter2 点名命中 | 107 | 0.7260 | 0.7045 | **−0.0215** | 9/15 | **0.935**（从 1.000 掉下来） |
| iter2 点名未中 | 28 | 0.3017 | 0.5030 | +0.2012 | 16/5 | 0.500 |

iter3 在 iter2 已经点对的 107 题上**点名率从 1.000 掉到 0.935**（≈7 题丢了），
acc 也退了 0.0215（退步 15 题 > 进步 9 题）。
它在 iter2 点错的 28 题上修回来 14 题。**净 +7，但确实是修 14 丢 7。**

## 6. 判据逐条对账

| 事前判据 | 实测 | 结论 |
|---|---|---|
| name_hit 涨幅 < iter2 的 +0.2148 | +0.0519 | ✅ 过 |
| acc 增益主要来自条件 acc | 11% 来自条件 acc，89% 来自点名率 | ❌ **判错** |
| McNemar 净翻转 ≥ 10 才算证据 | 全部 < 10（acc_pos 仅 +2 题） | ✅ 正确地判为不显著 |
| sol_acc < 0.05 且 name_hit < 0.05 才算撞墙 | 0.0247 ✅ / 0.0519 ❌ | 未同时满足（贴边） |

## 7. 所以"跑通"了吗

**在多轮单调不降这个意义上：跑通了。** base→iter1→iter2→iter3 六列全涨。

**在"每轮都有实质进步"这个意义上：没有，而且看不到路。**
- 增益从 +0.1598（acc）缩到 +0.0247；
- 全部不显著；
- 73% 的题点名率零变化，增益就是 7 道题；
- 参数那一支（条件 acc）几乎不动，而那才是任务的核心难点；
- 课程侧的 hard 池同时塌了（91→30），两者方向一致：
  **不喂难题，难题上就不进步。**

**下一步不是加轮数，是把课程侧修好**（agreement 拆口径那件事，已落地并测过），
让 hard 池回来，再看参数那一支动不动。

---

# round4 起跑前的两个前置修补（2026-10-16 00:35）

这两件都不是"改指标"，是补正确性门，且都会让一轮白跑而不是让脏数字进账本。

## 修补 1：污染门从 post_round 前移到 step3 之前

原来 `check_holdout_clean.py` 只在 `post_round.sh` 里跑，而 `post_round.sh` 第一句是
"`round<ITER>.step3.ok` 不在就退出"。也就是说：**教材撞了固定评测集，也是训完才发现的。**
到那时 solver 已经在评测集的题上训过，"held-out 比 in-sample 涨得多所以是泛化"
这个结论直接作废——而作废这件事日志里看不出来，数字照样进账本。

现在加在 `run_round.sh` 的 step2 完 / step3 起之间，带 `round<ITER>.contam.ok`
幂等标记（续跑时不会重复查）。

## 修补 2：`check_holdout_clean.py` 把"读不到"误报成"污染"

原代码里读不到文件也走 `bad += 1`，于是路径写错会被说成"教材污染了固定评测集"。
**一个错的诊断比没有诊断更糟**——它会让人去查教材，而问题在路径上。
现在拆开：撞题退 1，读不到退 2，各自说各自的话；门的 `die` 也按退出码分开报。

实测三条路径：
```
iter3_data.json（真教材）  rc=0 -> 过门
/tmp/nope.json             rc=2 -> 报"读不到，先查路径"
撞题的假教材               rc=1 -> 拒跑
```

## 附带确认

拿真实的 `iter2_data.json` / `iter3_data.json` 各 100 条逐条比过 holdout：
**都干净**。所以 iter1→iter2、iter2→iter3 那些 held-out 数字 retrospective 也立得住，
不存在"其实在评测集上训过"的隐患。

---

# round4 的量化预测（00:40，step1 还在跑，事前钉住）

## 先修正我自己的一个说法

我此前写"hard 池塌（91→30）是饱和规则误剔造成的"。**这个说法是错的。**

实测（`report_funnel.py 2 3`）：

| round | 出题 | verify 活 | 剔饱和 | 剔前 hard | 剔后 hard | 终选 hard |
|---|---|---|---|---|---|---|
| iter2 | 1118 | 918 | 332 | **91** | **91** | 30 |
| iter3 | 1149 | 952 | 656 | **30** | **30** | 22 |

iter3 被剔的 656 条，名口径 agreement **全是 1.0**，按定义全在 easy 档
（952 条里 easy 829 → 剔后 173，正好 656）。**hard 那 30 条名口径都 <1.0，
饱和规则一条都没剔过它们。**

所以 hard 池 91→30 是**产出端**的事：iter3_generator 产出的 952 题里，
iter3_solver 在 10 个样本里**一个都没点对名**的只有 30 题（iter2 是 91/918）。

最可能的机制不是 bug，是**自指**：iter3_solver 点名率 0.7926→0.8444 变强了，
于是"它完全点不出名"的题集合机械性缩小。这正是之前记过的
"agreement 自指（solver 变强消灭自己的训练信号）"。

## 饱和 bug 的真实代价（另一件事，但是真的）

656 条里：首个预测全等对得上 360 条、对不上 296 条。
对不上 ⟹ 10 个样本里至少一个全等失败 ⟹ 全等 agreement < 1.0 ⟹ **新规则下必定活**。

所以新规则：
- 剔饱和 656 → 落在 **[0, 360]**
- 教材池 296 → 落在 **[592, 952]**

## 但是：救回的是体量，不是难度

拿 iter3_solver 在固定评测集上估 P(全等 | 名对)：
点名命中 114/135 = 0.8444，其中 acc 均值 0.7315 ⟹ **c ≈ 0.7315**。

对那 296 条恢复项（10 个样本名全对），若各样本独立，
全等 agreement ~ Binomial(10, 0.7315)/10：

| 档 | 概率 | 折算条数 |
|---|---|---|
| hard (<0.4) | 0.005 | **≈ 2** |
| medium [0.4,0.8) | 0.524 | ≈ 155 |
| easy (≥0.8) | 0.427 | ≈ 126 |

**296 条里只有约 2 条会变成 hard。**

## 所以 round4 的事前判据（现在钉住，不许赛后改）

| 量 | iter3 实际 | round4 预测 | 判据 |
|---|---|---|---|
| 剔饱和数 | 656 | **< 100**（点估 ~16，硬上界 360） | > 360 说明修复没生效 |
| 教材池 | 296 | **> 590**（硬下界 592） | < 592 说明口径没切过去 |
| hard 池 | 30 | **30–60，不会回到 91** | 若 > 91 说明我对自指的判断错了 |
| medium 池 | 93 | **大幅增长（≈250）** | 恢复项主要落这里 |

**若 hard 池真的回到 91 以上，那说明难度不是自指的，我这段分析就错了——那是好消息，但要认。**
**若 hard 池还在 30 附近，那说明"多轮能不能上升"的瓶颈不在课程侧，在 generator 产出端，
下一步该动的是 generator 的题目分布，不是 select 的阈值。**

---

# 硬发现：hard 池塌陷是"solver 变强"造成的，不是"题变简单"（00:48，纯 CPU 分析）

这段不需要 GPU，用的是 iter2/iter3 已经落盘的 verify 中间文件。

## 1. generator 的产出分布确实变了，但方向解释不了塌陷

| | iter2 | iter3 |
|---|---|---|
| 工具数 mean | 4.0 | 3.3 |
| 题面长 mean | 100 | 87 |
| hard 率（名口径 <0.4） | 9.9% (91/918) | 3.2% (30/952) |
| domain 种类 | 32 | 32 |

工具数与 agreement 负相关，两轮都成立（iter2 −0.175，iter3 −0.190）。
工具越多名越难点——这是真的，但相关只有 −0.18，很弱。

## 2. 反事实：把 iter2 的难度特征套到 iter3 的题上

按工具数分四桶，iter2 各桶 hard 率：`1-2: 6.5% / 3-4: 9.8% / 5-6: 11.1% / 7+: 15.6%`
iter3 的桶分布：`1-2: 388 / 3-4: 312 / 5-6: 185 / 7+: 67`

```
期望 hard = 388×6.5% + 312×9.8% + 185×11.1% + 67×15.6% = 87 条
实际 hard =                                                     30 条
```

**iter2 实际是 91 条。也就是说光看题目分布，iter3 本该产出约 87 条 hard，
和 iter2 差不多。它只产出了 30。**

## 3. 同桶对比：题面难度可比，只有 solver 不同

| 工具数 | iter2 agreement | iter3 agreement | 差 |
|---|---|---|---|
| 1-2 | 0.839 (n=231) | 0.951 (n=388) | **+0.113** |
| 3-4 | 0.769 (n=347) | 0.893 (n=312) | **+0.125** |
| 5-6 | 0.757 (n=244) | 0.894 (n=185) | **+0.137** |
| 7+ | 0.681 (n=96) | 0.845 (n=67) | **+0.164** |

**每一个桶里 iter3 的 agreement 都高出一截。** 题没变简单（甚至略微变简单了一点，
但那只会让期望 hard 从 91 降到 87），是 solver 变强了。

## 4. 结论：自指，坐实

hard 池 91→30 的机制是：**agreement 是拿当前 solver 量的，solver 一变强，
"它完全点不出名"的题集合就机械性缩小。** 这不是 bug，是这个指标的结构性性质。

这和之前记的"agreement 自指（solver 变强消灭自己的训练信号）"是同一件事，
现在有了干净的量化证据（反事实 87 vs 实际 30，同桶 +0.11~+0.16）。

**推论：修 select 的阈值、修口径，都救不了 hard 池。** 只要难度还是由当前 solver
量的，池子就会随 solver 变强而持续缩小。round4 跑完会给出第三个数据点
（预测 hard 池 30-60，见上一节的事前判据）。

## 5. 下一步该动什么（等 round4 数据确认后）

难度分档要换一个**不随当前 solver 漂移**的尺子。候选：

| 方案 | 好处 | 代价 |
|---|---|---|
| 用**冻结的 reference solver**（比如永远用 base_solver）量 agreement 分档 | 难度尺度跨轮稳定，round 之间可比 | 要常驻一个旧模型；且"对 base 都难的题"可能对当前 solver 太难、梯度为零 |
| 用 generator 侧属性（工具数/call 数/题长/domain）分档 | 零 GPU、不自指 | 与 agreement 只相关 −0.18，区分度弱 |
| 分档用冻结尺子，**留存门仍用当前 solver** | 留存门本来问的就是"还能不能训"，该用当前尺子 | 两套尺度要各自解释 |

我倾向第三个：留存门（`tau_verify`、`drop_agreement_ge`）保持当前 solver——
它回答"这题现在还有没有梯度"，本来就该随时间变；**只有难度分档换冻结尺子**——
它回答"这题有多难"，这是个绝对量，不该随 solver 漂移。

## 6. domain 混杂也排除了（补上最后一层）

iter2/iter3 的 domain 构成不同（iter2 头部 web_search/retail，iter3 头部 legal/real_estate），
所以逐 domain 比一遍，只取两轮样本都 ≥8 的 domain：

**32 个 domain，iter3 agreement 更高的有 32 个，更低 0 个。**
平均 +0.139，中位 +0.139；最小 +0.038（vehicle_control），最大 +0.278（education）。

不是"iter3 恰好出了些容易 domain 的题"——**每一个 domain 都变强了。**

## 7. 三层证据闭合

| 层次 | 证据 | 排除了什么 |
|---|---|---|
| 题目分布 | 反事实：套 iter2 各桶 hard 率，iter3 该出 87 条，实际 30 | "题变简单了" |
| 同桶 | 4 个工具数桶 agreement 全高 +0.113~+0.164 | "只是题面属性变了" |
| 同 domain | 32/32 个 domain 全高，均值 +0.139 | "只是 domain 混合变了" |

**hard 池 91→30 的唯一解释是：solver 变强了，而难度是拿这个变强的 solver 量的。**

这不是 bug，是自指指标的结构性性质。修 select 的阈值和口径都治不了它
（round4 的口径修复仍然值得做——它救回 296 条该训的题——但救不回难度）。

---

# round5 的分叉预案（01:20 写，round4 还没跑完——所以这是预案不是结论）

上一节结论是"修 select 治不了自指"。但在动手之前，还有一个**零 GPU 成本**的选项
上面漏了，它值得和"冻结 reference solver"并列，因为成本差一个数量级。

## 方案 D：分位数分档，代替绝对阈值分档

现在是 `hard = agreement < 0.4`，这个 0.4 是在 **iter1_solver 还很弱**的时候标定的。
solver 变强 → 整条分布右移 → 落在 0.4 左边的越来越少。这就是 91→30 的机制。

**改成"本轮池子里最难的 X%"**（按 agreement 排序取底部分位），hard 池条数就不再随
solver 强度漂移——每一轮都稳定拿到一批"当前够不着"的题。

**代价（必须写清楚，别只顾好处）**：它把难度配比变成了**恒等式的构造**
（"30% hard"由定义保证，不是因为题目真的那么难）。于是"这一轮课程变容易了还是变难了"
这个信号被消灭了——而那正是我们想监控的东西。缓解办法是**照旧报告绝对分布**
（`report_funnel.py` 已经在打 `两口径均值` 和剔前/剔后各档条数），
把"配比是构造出来的"和"绝对难度是多少"分成两个数看。

**另一个代价**：分位数分档仍然拿当前 solver 排序，所以"哪些题最难"的排序本身还是自指的。
它治的是**条数塌陷**，不治**排序漂移**。冻结 reference solver 两个都治。
所以 D 是便宜的一半解，不是等价解。

## 所以 round5 的决策规则（等 round4 的数字，不预选）

| round4 的 hard 池 | 诊断 | round5 动作 |
|---|---|---|
| 回到 90+ | 口径就是主因，自指判断错 | 认错，保留 round4 口径，下一步查别的 |
| 60–90 | 口径是主因之一，自指也有份 | 保留 round4 口径，先看 held-out 判据再决定 |
| 30–60（我押这个） | 坐实自指，口径只是表象 | 上方案 D（零成本），同时把绝对分布报清楚 |
| < 30 | 自指 + 产出端同时恶化 | 先查 generator 产出，别动 select |

**以及一条独立于 hard 池的判据**：round4 的 `sol_acc` / `sol_acc_pos` / `name_hit`
三项配对检验。如果教材池从 296 涨到 900+ 而三项还是净翻转 < 10，
那说明"多轮能不能上升"的瓶颈**根本不在课程侧**——那时候再改分档是在优化一个
不影响结果的中间量，改多少次都不会让 held-out 动。这条比 hard 池更重要，
但因为它要等 step3 训完才能测，所以排在后面看。

## 纪律（延续 round4 那条，不许破）

round4 是"只改口径、阈值不预调"。round5 同样只改一件事：
- 若走 D：只换分档规则（`assign_difficulty_from_agreement` 加一个 `--agr_mode quantile`），
  `tau_verify` / `drop_agreement_ge` / lr / 步数 / mix 全不动。
- **分位数本身（X%）不许看着 round4 的数据挑**。就用 `run_step2.sh` 里**早就配好的**
  `--mix_easy 0.20 --mix_medium 0.50 --mix_hard 0.30`，即 hard 30% / medium 50% / easy 20%。
  这个数是 round4 之前就存在的配置，不是从 round4 的分布里挑的；
  round2 的终选实际是 easy 21 / medium 49 / hard 30，说明这套配比本来就被执行得很好，
  分位数分档要做的只是**把"实际拿到多少"从"池子碰巧有多少"变成构造保证**。

---

# round4 评测的事前判据（01:26 写，round4 step1 还在跑，评测未发生）

在 step2/step3 跑完之前，把"该看哪个数"钉住。依据是刚才对 iter2→iter3 跑的一次
`channel_split.py`（纯 CPU，读已落盘逐条评测，结论可复现）：

## 1. 现状的三个通道余量

| 通道 | iter3 实测 | 离顶还有 | 每轮增量（历史） |
|---|---|---|---|
| `sol_fmt` | 0.9822 | 0.018 | 已见顶，无余量 |
| `sol_name_hit` | 0.8444 | 0.156 | +0.5778 → +0.1259 → +0.0518（约每轮减半） |
| 条件 acc（点名命中时的 acc） | 0.7315 | 0.269（×点名率 0.844 = **0.227 的总 acc 空间**） | +0.0055（几乎不动） |

**点名这条通道约按每轮减半的速度在关**，最多再撑两三轮。
总 acc 空间的大头（0.227 / 0.337 ≈ 67%）在**条件 acc**里，不在点名里。

## 2. 所以 round4 的主判据不是 sol_acc，是条件 acc

round4 的教材修复（救回 296 条"名对参错"）**瞄准的正是条件 acc 这个通道**——
那些题点名已经对了、参数写不对，训的就是参数填充。点名通道它帮不上忙。

| 判据 | iter3 | round4 预期 | 不过说明什么 |
|---|---|---|---|
| **条件 acc** | 0.7315 | **> 0.7450**（即 +0.0135，要超过 round3 的 +0.0055 一倍以上） | 修复没触及真正的瓶颈 |
| `sol_acc` | 0.6627 | 0.68–0.70 | 主要由点名驱动，**不单独作为判据** |
| `sol_name_hit` | 0.8444 | 0.86–0.88 | 快关门的通道，别指望它 |
| `sol_acc_pos` | 0.9037 | > 0.91 | 已经很高，空间小 |

**为什么要单独立条件 acc 这一条**：round3 的事前判据是"acc 若还在涨应主要来自条件 acc"，
实测只有 11% 来自条件 acc，89% 来自点名。如果 round4 还只看 acc，会再次把
"点名又挪了一点"误读成"多轮有用"。**点名挪动和参数填充是两件事，必须分开记账。**

## 3. 一个反向的可能，先写下来

`channel_split.py` 的难度表显示 **b条件acc 各档接近（0.617~0.837，最低的正是最易的 eq1.0 档）**。
若 round4 的条件 acc 仍然不随难度变、也不涨，那说明它是**与题目难度无关的参数填充上限**，
教材侧换什么题都推不动它——那时候该动的是 step3 的训练信号（奖励怎么给参数部分），
不是 step2 的选材。**这条会在 round4 数据出来后第一次被检验，因为它区分
"换教材能解决"和"必须换训练信号"这两种完全不同的下一步。**

## 4. 检验效能（先说清楚，免得以弱检验当证据）

n=135，McNemar 净翻转 < 10 不算证据。条件 acc 是 114 条子集上的均值，
+0.0135 约等于子集上 +1.5 条的质量改善——**很可能落在噪声里**。
所以：条件 acc 的判据要**和分档表一起读**（看各档是否一起动），
单看一个数字、又不显著，不许说"修复有效"，也不许说"修复无效"——只能说"这一轮没测出来"。

---

# 硬发现②：难度尺子量错了轴（01:35 写，round4 step1 还在跑）

上一节的结论是"自指"。这一节更狠：**agreement 不只是自指，它和真实奖励基本无关。**

## 证据（固定评测集 135 题，纯 CPU，join holdout_solver.json 的 gold calls 与
evals/iter3_solver.json 的逐条 acc；两文件题目文本逐条 assert 过，顺序可靠）

| 量 | 与 acc 的相关 | t (n=135) | 读法 |
|---|---|---|---|
| `solver_agreement`（难度尺子） | **+0.009** | 0.10, p≈0.92 | **与零不可区分** |
| gold 参数键总数 | **−0.262** | −3.12, p≈0.002 | 唯一有信号的量 |
| gold 调用个数 | 见下（n=7 极端值） | — | 多调用近乎不会 |
| `solver_agreement` vs `name_hit` | +0.200 | — | 它只预测点名，不预测奖励 |

**为什么 r(ag, acc)≈0 而不是正**：ag 高 → 点名率高 → 命中支涨；但同时 ag 高 → 条件 acc 低
（分档表里 ag=1.0 条件 acc 0.6167、ag=0.2 是 0.8904）。两个效应互相抵消。
这不是小样本噪声——`corr(ag, 键数) = −0.109`，说明反向关系不是"ag 高的题参数更多"造成的。

## 参数键数的效应是双通道的，而且稳健

| 分组 | n | acc | 点名率 | 条件acc |
|---|---|---|---|---|
| 1 键 | 73 | 0.7082 | 0.8904 | 0.7954 |
| 2+ 键 | 62 | 0.6091 | 0.7903 | 0.7707 |
| 3+ 键 | 23 | 0.5180 | 0.7391 | 0.7009 |

**点名和条件 acc 一起掉**，所以这不是"某一个通道的坑"。
组内看点名差距恒定（1 键 0.416、2 键 0.407、3+ 键 0.458），说明点名仍是主导通道，
但键数在主导通道之上再叠一层衰减。

## 多调用题：7 道，点名率 14.3%

`ncalls>=2` 的 7 道题：acc 0.2786、点名率 **0.1429**（7 道里只 1 道点名对）。
占评测集 5.2%，但对 acc 的拖累是 (1.0−0.2786)×7/135 = **0.0374**——
比 round3 一整轮的 acc 增量（+0.0247）还大。**这是单块最大的可收复失地。**

## 我要纠正自己一个判据用反的错误

PLAN 上面「5. 下一步该动什么」里，我把"用 generator 侧属性分档"否掉，理由是
**"与 agreement 只相关 −0.18，区分度弱"**。这个检验**用错了靶子**：
评判一个难度特征好不好，该看它和**奖励**的相关，不是和**当前尺子**的相关。
agreement 自己和 acc 的相关只有 +0.009，那么"和 agreement 相关弱"根本不构成反对理由。
按正确的靶子重测，参数键数与 acc 相关 −0.262（p≈0.002）——**是被我误判掉的那个选项里最强的信号。**

## 边界（不许夸大）

1. 这是**固定评测集上、base 时代测量的那个 agreement** 与**当前 solver 的 acc** 的相关。
   课程里用的是**当前 solver 在新题上**的 agreement，两者不是同一个量。
   所以这里证明的是"点名一致率这个量本身不预测奖励"，不是"round4 课程里的 agreement 完全没用"。
2. n=135，多调用只有 7 条，那 14.3% 是 1/7——方向极强但样本极小，别当成精确估计。
3. 参数键数在 domain 上有混杂（travel 2.86 键 acc 0.3940；cybersecurity 1.17 键 acc 0.8333），
   但 government 与 web_search 键数几乎相同（1.71）acc 也接近（0.4762 / 0.4929），
   说明 domain 的差异大半能用键数解释，不是独立主因。

## 对 round5 的影响（round4 不改，它已经在跑了）

round4 修的是"名对参错被当饱和丢掉"，方向仍然正确——那 296 条正是参数通道的训练信号。
但**难度分档换不换尺子这件事，优先级下降了**：既然 agreement 不预测 acc，
拿它分档本来就分不出"对奖励来说难的题"。真正的候选是**按 gold 调用数 / 参数键数分档**，
零 GPU、不自指、且与 acc 有真实相关。这条要等 round4 的 held-out 判据出来再定，
不要现在改——一次只改一件事，round4 那件事还没验完。

---

# 硬发现③：多轮增益递减的根因是 tau_verify，不是 solver 变强（01:45 写）

前两节讲的是"难度尺子选得不准"。这一节找到**增益为什么逐轮递减**，而且原因不在 select、
不在 solver、在 **verify 的一道门**。

## 机制（iter3，纯 CPU，逐条 join 原始产出与 verify 留存）

| | 生成条数 | 被 verify 剔 | 剔除率 |
|---|---|---|---|
| 多调用题（`len(calls)>=2`） | 86 | **82** | **95.3%** |
| 单调用题 | 1063 | 115 | 10.8% |

**多调用题被剔的概率是单调用的 8.8 倍。** 留存的 4 条多调用题 agreement 是
0.4 / 0.4 / 0.6 / 0.2——全贴在 `tau_verify=0.2` 的地板上。

verify 报告的 drop 原因只有一行 `mismatch: 197`，且 197 条**全部**在 `[0.0, 0.2)` 桶里。
**没有格式解析失败的成分，这道门是唯一的删除原因。**

## 于是训练分布和评测分布系统性分叉

| 阶段 | 多调用% | 3+ 参数键% | 平均键数 |
|---|---|---|---|
| iter3 原始产出（verify 前） | 7.5 | 13.8 | 1.67 |
| iter3 verify 后（课程来源） | **0.4** | **9.8** | 1.53 |
| iter3 终选教材（100 条） | 1.0 | 13.0 | 1.59 |
| **固定评测集（135 条）** | **5.2** | **17.0** | 1.80 |

**评测集代表的是原始产出分布（7.5% 多调用），课程代表的是被筛过的分布（0.4%）。**
`tau_verify` 名义上是"这题碰不碰得到"的可达性门，实际执行成了**难度过滤器**，
而且它滤掉的正是 solver 最需要训练的那类题。

## 这解释了此前三个各自成立但连不起来的现象

1. **acc 增量逐轮减半**（+0.1598 → +0.0247）：训练分布在往"容易"漂，评测分布不变，
   两者的差距每轮扩大一点，能从训练里拿到的东西就少一点。
2. **多调用评测题点名率只有 14.3%**（7 道里 1 道）：教材里 0.4% 是多调用，
   等于从没教过——不是学不会，是没学过。
3. **0.85-1.0 / eq1.0 这两个最易档的条件 acc 反而最低**：课程被筛到只剩简单题，
   而简单题在评测集里恰好是"点名容易但参数不 trivial"的那批。

## 预期效果（round5 的候选实验，round4 不改）

把 `--tau_verify` 从 0.2 降到 0.0，1149 条原始产出全部进池，多调用占比回到 7.5%。
- 可收复的最大单块：7 道多调用评测题从 acc 0.2786 抬到单调用水平 ~0.68，
  = (0.68−0.2786)×7/135 = **+0.0208**，约等于 round3 整整一轮的增量。
- 加上 3+ 键题（评测 23 条、acc 0.5180）占比从 9.8% 回到 13.8%，还有一点。

**为什么这是高价值单点实验**：一行参数、机制已量化、有明确预期值、且能一次性检验
"多轮增益递减到底是课程侧还是别的"。比继续调 select 的阈值信息量大得多。

## 必须同时承认的风险（不许只报好消息）

`tau_verify` 降到 0 会放进 agreement=0.0 的题，也就是 k=10 个样本**没有一个点名对**的题。
GRPO 的 advantage 是组内相对量，若组内奖励完全没有方差，梯度就是 0，这些题占名额不产出。
`tau_verify=0.2` 要求"10 个里至少 2 个点名对"，是在粗暴地保证一点方差。
所以 **0.0 可能大跨步扯到零梯度题**，中间值（0.1，即 10 个里 1 个）也许更稳。
这个只能实测，不能推——**列为 round5 的 A/B，不预选 0.0**。

另一个代价：池子从 952 涨到 1149，`drop_agreement_ge` 仍是饱和门，
所以"零梯度"在顶端已经被管住了；要防的是**底端**。这与饱和门是两个方向，别混。

## 推荐的验证实验：round 4b（隔离 tau_verify，不掺 generator 重训）

跑完整轮的问题是"改了 tau 又换了 generator"，两个变动挤在一轮里说不清。
而 **iter3 的原始池 `iter3_data.json.intermediate.json`（1149 条）已经在盘上**，
所以可以把 tau_verify 单独摘出来：

```
# 1) 重跑 verify，只换 tau（GPU，~10min）
python step2_genverify.py \
    --solver_model <iter3_solver> \
    --in_intermediate_json  iter3_data.json.intermediate.json \
    --out_intermediate_json /tmp/iter3_tau0.verified.json \
    --report_txt           /tmp/iter3_tau0.verify_report.txt \
    --k_verify 10 --tau_verify 0.0 --temp_verify 0.7 \
    --max_tokens_verify 1024 --verify_batch_size 32

# 2) 选教材（CPU，0.1s）——其余参数与 run_step2.sh 逐字相同
python step2_select_curriculum.py \
    --in_json /tmp/iter3_tau0.verified.json \
    --out_json /tmp/iter3_tau0.data.json \
    --n_final 100 --mix_easy 0.20 --mix_medium 0.50 --mix_hard 0.30 \
    --seed 13 --default_diff medium --drop_agreement_ge 1.0 \
    --difficulty_source agreement --judge_model <iter3_solver>

# 3) 从 iter3_solver 接，训新 solver（GPU，~10min）→ 得到 iter3b_solver
# 4) eval_round 同套判据，对照 iter3_solver
```

**为什么这样更好**：solver 起点固定为 iter3_solver，generator 完全不重训，
唯一变量是 tau_verify。round4 与 round4b 正好互补——round4 检验"口径修复"，
round4b 检验"难度地板"。两个都跑完，"多轮增益为什么递减"就有两条独立证据。

**顺序**：round4 先跑完（它已经在跑，且是原定计划），round4b 排在它后面，
因为两者都要用 GPU，而 round4 的 step2/step3 正在排队。

**判据（事前钉住）**：
| 量 | iter3 实际 | round4b 预期 |
|---|---|---|
| verify 留存 | 952 | **1149**（tau=0 全进） |
| 池里多调用占比 | 0.4% | **7.5%** |
| 教材多调用条数 | 1 | **≥5** |
| 教材 3+ 键条数 | 13 | **≥18** |
| held-out `sol_acc` | 0.6627 | **> 0.6835**（即 +0.0208 的那一块） |
| held-out 多调用题点名率 | 0.143 | **> 0.30** |

**若教材多调用条数上去了而 held-out 多调用题点名率没动**，说明"solver 见过但学不会"，
那时候问题在 step3 的训练信号（怎么给多调用题算奖励），不在选材。
**若 held-out `sol_acc` 反而降**，说明放进来的零梯度题拖累了训练，那就取中间 tau（0.1）再测。

## 证据分级与一个必须排除的混淆（01:55 补，同一次分析内）

上面写"多调用被剔率 76.4%→95.3%"时，有一个替代解释没排除：
**会不会 iter3 生成的多调用题本身更难？** 查了，**没有，反而更简单**：

| | iter2 多调用题 | iter3 多调用题 |
|---|---|---|
| 平均调用数 | 2.56 | **2.33** |
| 平均参数键数 | 4.32 | **3.40** |
| 调用数分布 | 2:53 3:7 4:7 5:3 7:2 | 2:74 3:6 4:1 5:1 6:3 7:1 |

同期单调用留存 agreement 从 **0.782 涨到 0.916**（solver 确实在变强）。
所以：题变简单了、solver 整体变强了、而多调用通过率从 23.6% 掉到 4.7%。
**若 solver 的多调用能力只是没进步，更简单的题通过率该升不该降。**

**证据分级（不许把推断写成实测）**：
- **实测**：197 条被剔全部 agreement < 0.2（报告分箱 + 5 个留样 agreement 0.000×4 / 0.100×1）。
- **实测**：多调用通过率 23.6%→4.7%、题变简单、单调用 agreement 0.782→0.916。
- **实测不了**：被剔的 82 条多调用题各自的 agreement 值不在盘上
  （报告只留 5 个 truncated 例子），要拿它得重跑 verify。
- **推断**（由上面三条推出，前提是"tau 不变 + 题变简单 + 整体变强 ⟹ 通过率该升"）：
  solver 的多调用能力相对单调用**下降了**，即"单调用课程把 solver 训窄了"。
  这是一个**恶性循环**：单调用课程 → 多调用变差 → 多调用题过不了 verify → 课程更单一。

**为什么仍然值得按这个推断行动**：即使把"多调用能力下降"这条推断整个撤掉，
剩下的实测部分（训练分布 0.4% vs 评测 5.2% 多调用）已经足够支持 round4b 那个实验。
推断只是让预期效果更大，不是实验成立的前提。**实验设计不依赖未实测的部分。**

---

# round4 step2 实测（02:39，step3 还在跑，评测未发生）

step2 已收摊，污染门过。四个事前判据 2 过 2 不过，**不过的两条是我的判据定错了，
不是修复没生效**——这一节把这点写清楚，免得多轮之后把"判据没过"读成"改动无用"。

## 漏斗数字（round4 vs iter3）

| | iter3 | round4 |
|---|---|---|
| 生成 | 1149 | 1143 |
| verify 留存 | 952 | 952 |
| verify 剔（mismatch） | 197 | 191 |
| **剔饱和** | **656** | **292** |
| 饱和后池 | 296 | **660** |
| 池分档 hard/medium/easy | 30 / 93 / 173 | **463 / 100 / 97** |
| 终选配比 | 33/45/22 | 20/48/32 |

**修复起效的直接证据**：`名对参错 469 条`（名口径=1.0 但全等<1.0）。
旧口径会把这 469 条全当"已经教会了"丢掉。两口径均值 **名 0.9485 / 全等 0.4730，
差 0.4755**（n=952）——名口径把 solver 的能力放大了约一倍，和代码注释里那个
45.3% vs 91.0% 的留样一致，但这是在整池上量的。

## 事前判据的判决

| 判据 | 要求 | iter3 | round4 | 结论 |
|---|---|---|---|---|
| 剔饱和数 | < 360 | 656 | **292** | PASS |
| 教材池 | > 590 | 296 | **660** | PASS |
| hard 池 | [30, 60] | 30 | **463** | FAIL |
| medium 池 | > 150 | 93 | **100** | FAIL |

前两条是修复的**直接检验**（360 = 656 − 296 那个硬下界），都过。
后两条量的是**难度分档条数**，我定阈值时犯了一个错：

**我怎么定错的**：我按 iter3 在**名口径**下的分档分布去定 hard/medium 的条数区间
（hard 30-60、medium >150），却没想到换成全等口径会把尺子的**绝对刻度**整个挪走。
名口径均值 0.9485，`hard` 的阈值是 <0.4，于是 iter3 里 883/952 落在 easy；
全等口径均值 0.4730，同一条阈值下 463/952 落在 hard。**尺子还是那把尺子，
刻度翻了面。** 所以我期待的"hard 池回到 91 条（iter2 水平）"根本不可能发生——
在名口径下 iter2 的 91 条 hard，换算到全等口径也会变成几百条。

**恢复项落在哪了**：469 条名对参错里，hard 343 / medium 62 / easy 64。
我事前写"恢复项主要该落 medium"，也错了。直方图说明为什么：
全等值 **0.0 的有 263 条（56%）**，0.1 的 42 条，0.2/0.3 各 19 条，
0.4-0.7 合计 62 条，0.8/0.9 合计 64 条。
它们不是"对了一半所以落在中间"，是**参数系统性全错**——点名永远对、
参数从来没对过。这正是梯度最该密的地方，不是中间档。

**所以后两条 FAIL 的正确读法**：它们检验的是"尺子读数"，而尺子被我同时换了，
两条互相污染，不构成对修复的反驳。**修复是否真的有用，只能由 held-out 判**
（条件 acc > 0.7450 那条主判据），不能由分档条数判。

## 一个附带的实测：教材有多少条真在产梯度

`frac_reward_zero_std` 是 TRL 直接打的"本步有多少个 prompt 组内奖励零方差"，
零方差 = GRPO 优势恒 0 = 对这些题白训。

| 轮次 | 步数 | 零方差均值 | 有零方差步占比 |
|---|---|---|---|
| 2 | 50 | 0.1200 | 20.0% |
| 3 | 50 | 0.2400 | 36.0% |
| 4 | 21（未完） | 0.1667 | 33.3% |

方向是 round3 → round4 下降（-0.0733），和"修复回收了有梯度的题"一致。
**但这个量不是纯课程属性**：它是 课程 × 模型 的乘积——模型变强会让更多项饱和
（零方差升），课程变难会让更多项有梯度（零方差降），round4 两者同时发生。
所以这一行是**佐证**，不是证据。memory 里 `selfplay-reward-components-that-call-the-model`
那条说的就是这类量不能跨轮当质量指标。

## 奖励是稠密的，所以 agreement_full=0.0 不等于零梯度

查了 `rewards_solver.py`：单次调用得分
`r = 0.2*名 + 0.3*键名F1 + 0.5*键值匹配比例`。
"点名对、参数全错"拿 0.2，剩下 0.8 的方差空间还在——一条样本可能对 2/3 个键
（键名 F1 = 0.8）、另一条 0 个键，奖励仍然不同。
所以那 263 条 agreement_full=0.0 的题**不必然**空转，前提是 solver 的错法
不是四个样本完全一致。实测的零方差率（见上表）说明大部分没有完全一致。

## 一个差点犯的混为一谈（02:48，round4 评测跑着的时候查的）

channel_split 的难度分档表里有两个 n=7，我一开始以为同一批：

| | 多调用集 | 0.85-1.0 档 |
|---|---|---|
| n | 7 | 7 |
| 调用数 | 全部 2 | 全部 1 |
| solver_agreement | 0.2-0.7（低档） | 全部 0.90 |
| iter3 点名率 | **1/7 = 0.143** | 7/7 = 1.000 |
| iter3 acc 均值 | 0.2786 | 0.6914 |

**完全是两批题。** 分档表里 0.85-1.0 档那个 +0.296 的 acc 涨幅（点名率 0.429→1.000）
是单调用题，和多调用无关；多调用题落在**低 agreement 档**里，即基座本来就点不全名的题。

所以"多调用题点名率 14.3%"和"最易档点名率已经 100%"两件事同时成立，不矛盾。
可收复的那一块仍然是多调用 7 道：(0.68 − 0.2786) × 7/135 = **+0.0208**。
汇报时这两组必须分开报，否则会把单调用档的涨幅当成多调用恢复了。

## 多调用子集的跨轮轨迹：一条推断升级成了实测（02:59）

`multi_call_check.py`（新写，固定按 holdout 的 gold 调用数分组，不按任何轮次的状态分组）
给出 7 道多调用题的完整轨迹：

| | base | iter1 | iter2 | iter3 |
|---|---|---|---|---|
| 多调用点名率 | 0.000 | **0.286** | 0.143 | 0.143 |
| 多调用 acc | 0.0000 | **0.3321** | 0.2792 | 0.2786 |
| 全集 acc | 0.0285 | 0.4782 | 0.6380 | 0.6627 |

**多调用能力在 iter1 见顶（点名 2/7、acc 0.3321），iter2 回落到 1/7、0.2792，
iter3 持平 0.2786，而全集 acc 一路从 0.4782 涨到 0.6627。**

这直接把「证据分级」那一节里标为**推断**的那条变成了实测：
原来写的是"被剔的 82 条多调用题各自的 agreement 值不在盘上，测不了"，
所以只能从"题变简单 + 整体变强 ⟹ 通过率该升"反推"多调用能力相对下降了"。
现在不需要那个反推——held-out 上这 7 道题的逐条 acc 和 name_hit 一直在盘上，
直接读就是下降。

**可比性核对（必须写，因为 base/iter1 各有两个评测记录）**：
`iter_metrics.jsonl` 里 base_solver 有 16:08（acc 0.0207）和 20:36（0.0285）两条，
iter1_solver 有 16:42（0.4453）和 20:32（0.4782）两条；iter2/iter3 各只有一条。
16:xx 那两条早于一次代码修正（`eval_iter.py` 的 `padding_side = "left"`，
memory `tool-r0-sft-coldstart-baseline` 记着"必须左 padding 才准"）。
**上表取的是 20:xx 之后那两条，四条同一版代码，互相可比。**

**这条对 round4/round4b 的意义**：多调用块的理论上限是
(0.68 − 0.2786) × 7/135 = **+0.0208**。如果 round4 或 round4b 把这一块抬起来，
增益就有了明确的出处；如果全集 acc 涨而这一块不动，说明涨的全是单调用，
那"多轮有用"就只是单通道的反复打磨，多调用这条线仍然躺着。

---

## round4 收摊（03:05，post_round 4 3 跑完）

### held-out 主判据：过，而且机制换了

| | iter3 | iter4 | Δ |
|---|---|---|---|
| **条件 acc**（主判据） | 0.7315 | **0.7655** | **+0.0340** |
| sol_acc | 0.6627 | 0.6785 | +0.0158 |
| 点名率 name_hit | 0.8444 | 0.8370 | **−0.0074** |
| 未命中题 acc | 0.2893 | 0.2316 | −0.0577 |
| sol_fmt | 0.9822 | 0.9874 | +0.0052 |

事前钉的门是 **条件 acc > 0.7450**（不是 sol_acc）。**0.7655 过，余量 0.0205。**

**关键是增量构成翻了个面**：

| 轮次 | 点名率贡献 | 条件 acc 贡献 | 定性 |
|---|---|---|---|
| iter1→2 | +53% | +47% | 敢答和答对一起涨 |
| iter2→3 | **+89%** | +11% | 几乎全是"学会敢答" |
| iter3→4 | **−23%** | **+123%** | 点名率微降，全靠"学会答对" |

这正是口径修复该有的效果：修复救回的 469 条"名对参错"题在名口径下
`solver_agreement=1.0` 被当饱和丢掉（零梯度），在全等口径下 `<1.0` 留下来（有梯度），
所以 round4 学到的正是**参数**而不是**点名**。round3 的增量是名字增量，round4 的是参数增量。
条件 acc 的增量也从 round3 的 +0.0055 涨到 +0.0340（6 倍），**增益递减停了**。

### 但必须先审一件事：条件 acc 是拿本轮自己的命中题算的

`条件acc = mean(acc | name_hit)`，iter3 用的是 iter3 的 114 道命中题，
iter4 用的是 iter4 的 113 道，**不是同一批题**（memory
`self-conditioning-breaks-round-comparison` 就是这条）。查了：

| 比法 | iter3 | iter4 | Δ |
|---|---|---|---|
| 各自本轮命中题（主判据原定义） | 0.7315 | 0.7655 | +0.0340 |
| **只在两轮都命中的同一批 108 道** | 0.7432 | 0.7686 | **+0.0254** |

同题口径下仍是 +0.0254，且 0.7686 同样过 0.7450。剩下 +0.0086 来自换血方向有利：
进场的 5 道 acc 0.7000 > 退场的 6 道 0.5208。
**两种定义下判据都过，结论不是换血造出来的。**

（注意我用的分解式是 `acc = nh·cond + (1−nh)·cond_miss`，
**不是** memory 里说的错的 `acc = nh·cond`——未命中题的 acc 不是 0，
奖励是稠密的，iter4 未命中题还有 0.2316 分。cond_acc_check.py 里显式算了 cond_miss。）

### 统计显著性：三项全部"方向为正但不显著"

| 指标 | Δ | 净翻 | McNemar p | 连续配对置换 p | 判定 |
|---|---|---|---|---|---|
| fmt | +0.0052 | 1↑/1↓ | 1.0000 | 0.8772 | 持平 |
| acc | +0.0158 | 6↑/3↓ | 0.5078 | 0.3981 | 持平 |
| name_hit | −0.0074 | 5↑/6↓ | 1.0000 | — | 持平 |

n=135 的 McNemar 要净翻 ≥10 才够显著，acc 只净翻 3。
**所以诚实说法是：方向明确、机制清楚、事前判据过，但效应量低于 n=135 的分辨能力。**
这不矛盾——事前判据是连续量的阈值比较，不是显著性检验；两回事，别混着报。

### 多调用块：还没收复（这是 round4b 的靶子）

| | base | iter1 | iter2 | iter3 | iter4 |
|---|---|---|---|---|---|
| 多调用点名率 | 0.000 | **0.286** | 0.143 | 0.143 | **0.000** |
| 多调用 acc | 0.0000 | **0.3321** | 0.2792 | 0.2786 | 0.3136 |

iter4 多调用 acc 微升 +0.0350 但点名率归零，对总 acc 只贡献 +0.0018（占总增量 11%）。
**全集涨的不是多调用。** `tau_verify` 那道门 round4 没动，所以这一块没被训到是预期的——
round4b 就是专门摘它出来。

### 难度分档（涨在哪里）

| 档 | n | iter3 | iter4 | Δ |
|---|---|---|---|---|
| lt0.3 | 16 | 0.723 | 0.714 | −0.009 |
| 0.3-0.5 | 33 | 0.620 | 0.627 | +0.007 |
| 0.5-0.7 | 50 | 0.664 | 0.702 | **+0.038** |
| 0.7-0.85 | 19 | 0.695 | 0.711 | +0.016 |
| 0.85-1.0 | 7 | 0.691 | 0.724 | +0.032 |
| eq1.0 | 10 | 0.617 | 0.582 | −0.035 |

涨集中在中间两档（0.5-0.7 和 0.85-1.0），两端各掉一点。
round4 终选配比 easy 20 / medium 48 / hard 32，和"medium 涨最多"对得上。

### 顺手修的两个 bug（都是报告器，不是流水线）

1. **`compare_rounds.py` 崩了**：`drops` 存的是 `(a,b)` 两个 float，
   打印时却按 `((ta,a),(tb,b))` 解包 → `TypeError: cannot unpack non-iterable float object`。
   这条路径**从没被执行过**，因为之前六项指标一直单调不降；round4 的
   `sol_name_hit` 第一次回落（0.8444→0.8370）才踩到。改成存完整对。
   **教训：只跑成功路径的检查代码等于没检查。**
2. **`run_round4b.sh` 的 `$?` 在管道后**：`python ... | tee -a log` 之后
   `CRC=$?` 取到的是 **tee 的退出码**，恒 0——那样"事前判据未过几条"永远是 0，
   等于没核对，而且不报错。改成 `${PIPESTATUS[0]}`。全仓扫了一遍同类模式，0 处残留。

### round4b（03:08 起跑）

唯一变量 `tau_verify 0.2 → 0.0`，solver 起点固定 iter3_solver，generator 不重训。
verify 已跑到 160/1149。判据表见「推荐的验证实验：round 4b」。

### 机制的精确量化（以及它自己的限定）

单次采样已确认：`eval_iter.py` 每题只采一条，`name_hit` 和 `acc` 都从同一个
`c[0]` 算；`score_tool_call` 的 `name_score` 是二值的
（`rewards_solver.py:325`）。所以**单调用命中题上 `acc = 0.2 + 0.3·键名F1 + 0.5·键值`，
0.2 恒定，acc 的变化只能来自参数**。多调用题在 iter4 点名率 0/7，自动不在此集合内，
所以两轮都命中的 108 道全是单调用，这个论证在它上面是干净的：

| 两轮都命中的 108 道 | iter3 | iter4 | Δ |
|---|---|---|---|
| acc | 0.7432 | 0.7686 | +0.0254 |
| 参数部分（acc − 0.2） | 0.5432 | 0.5686 | +0.0254 |
| **参数 attainment**（占满分 0.8） | **67.9%** | **71.1%** | **+3.2pp** |

**但这个 +0.0254 自己的限定更要写**：108 道里 **86 道完全没动**，
+0.0254 来自 13 涨 / 9 跌，**净 4 道**。所以"参数变好了"是真的，
但它是少数题上的净位移，不是普遍性提升。配上配对检验不显著，
诚实的总结是：**方向、机制、事前判据三者都指向"round4 学到了参数"，
但效应量是 n=135 分辨不出来的级别。**

### 点名已经饱和，剩下的是长尾

iter4 未点名 22 道 = 7 道多调用 + 15 道单调用。15 道单调用**分散在 15 个不同工具上**
（send_message / get_user_info / add_order_to_order / find_route(5 个参数键) / …），
没有可集中的靶子。其中 5 道是 iter3 能命名而 iter4 丢掉的（代价侧），
10 道两轮都没突破。

**结论：点名这条线接近天花板（113/135），再加轮数的增量只能来自参数（round4 在做）
或多调用（round4b 的靶子）。**

### verify 门对多调用：round4 比 iter3 更狠（03:18，修正过一次解析 bug）

第一版算出来"round4 原始产出 0 条多调用"，是**我的解析函数用错了字段**：
intermediate 文件存的是 `calls`，终选教材才是 `answer`；只用 `answer` 会把
原始产出全解析成空列表，然后静默得出"0 条"的假结论。改成两个字段都试之后：

| | 原始产出 | verify 后（名口径 tau=0.2） | 终选教材 |
|---|---|---|---|
| round4 多调用 | 115 条（**10.1%**） | **2 条（0.2%）** | **0 条** |
| iter3 多调用 | 86 条（7.5%） | 4 条（0.4%） | 1 条 |
| round4 单调用 | 1028 条 | 950 条（剔 7.6%） | — |
| iter3 单调用 | 1063 条 | 948 条（剔 10.8%） | — |

**generator 在往好的方向走**（多调用产出 7.5% → 10.1%，3+键 13.8% → 15.6%），
但 verify 门按 **98.3%** 剔多调用（iter3 是 95.3%，单调用只剔 7.6%）。

机制是合取：`normalize_call_names_only` 要求**完整名称序列全对**，
而 solver 被单调用课程训到只吐一个 call，所以 2-call 题永远匹配不上——
**单调用课程 → solver 只吐一个 call → 多调用过不了 verify → 还是单调用课程**。
这是个自锁循环，round4 的 口径修复没打破它（口径修复管的是剔饱和和分档，
不管这道可达性门；这是设计，三条判据本来就用三把尺子）。

**但 round4b 只打开门可能不够**（这条必须连着一块说）：多调用题在全等口径下
agreement 必然低 → 落 hard 档，而 mix 是 20/50/30，hard 只有 20 个名额。
86 条多调用全进池，按比例也只选得出约 3 条，而预注册判据要 ≥5。
**如果这条 FAIL，结论不是 tau_verify 修错了，是 mix 配额才是真正的瓶颈**，
下一步该动 hard 配额或按 gold 调用数分档
（[[tool-r0-difficulty-ruler-measures-wrong-axis]]：agreement 与 acc 只相关 +0.009，
gold 参数键数 −0.262 才是对的轴）。

工具：`.venv/bin/python curriculum_composition.py 3 4 4b`
（把"原始产出 → verify 后 → 终选"三段并排，单看终选条数看不出这个）

### round4b 前两步实测（03:21，step3 在跑）

verify + select 已收摊，污染门过。**开门的效果立竿见影**：

| | iter3 | round4 | round4b（tau=0.0） |
|---|---|---|---|
| verify 留存 | 952 | 952 | **1149（全进）** |
| 池里多调用 | 4（0.4%） | 2（0.2%） | **86（7.5%，回到原始分布）** |
| 池难度分布（hard/medium/easy） | — | 463/100/389 | 658/103/388 |
| **教材多调用** | 1 | 0 | **5（5.0%）** |
| 教材 3+键 | 13 | 7 | 12 |

select 参数与 round4 逐字相同（只差 `--judge_model`，而 `--difficulty_source agreement`
下 judge 不参与），所以差异全部来自 tau_verify 和池子本身。

**我之前估"86 条全进池也只选得出 ~3 条多调用"是估错了**：我用 20 个 hard 名额算的，
实际 mix 是 `easy .20 / medium .50 / hard .30`，hard 约 30 个名额；
86 条多调用全落在 hard 档（658 条里占 13%），30 个名额期望 ~4 条，实测 5 条。
**所以"mix 配额才是瓶颈"这个担忧目前不成立**，开门确实够把多调用带进教材。

但 `教材 3+键 ≥18` 这条判据会 FAIL（只有 12）——3+键条数比 round4 的 7 条高了，
比 iter3 的 13 条略低，够不到我事前钉的 18。这条判据本身定高了，
和 round4 那两条一样是"拿旧尺子定新尺子读数区间"的错。
**开门解决的是多调用，不是 3+键**，这两件事我事前混在了一起。

### round4b 判据脚本里的同一个字段 bug（03:21 抓到）

跑 `check_round4b_criteria.py` 预检时，判据 [2] 报"池里多调用占比 实测 0.0% (0/1149)"，
而 `curriculum_composition.py` 独立测出 **86/1149 = 7.5%**。

原因和我在自己脚本里犯的是**同一个**：`calls_of` 只读 `item.get("answer")`，
但 **verified 池文件存的是 `calls`，终选教材才存 `answer`**。
只用 `answer` 会把池文件全解析成空列表 → "多调用占比 0.0%" → 报 FAIL。

**修完之后判据从"未过 2 条"变成"未过 1 条"**——也就是说这个 bug 会
**凭空造出一条假 FAIL**。它比崩溃危险：崩溃会停下来，假 FAIL 会让人
去解释一个不存在的问题（我差点就要去写"tau 不是唯一的门"的分析）。

同类扫描：`report_funnel.py` 不解析 calls/answer（它读 verify_report.txt），
所以没这个问题；它的 952 / 463/100/389 与我的独立测量一致。
**教训：判据脚本自己也要用一个独立实现交叉验证，不能只信它一个。**

修正后的 round4b 判据（eval 还没出）：
- [1] verify 留存 1149/1149 **过**
- [2] 池多调用占比 7.5%（86/1149）**过**
- [3] 教材多调用 5 条 ≥5 **过**（零余量）
- [4] 教材 3+键 12 条 < 18 **FAIL**（判据本身定高了，见上一节）
- [5][6] 等 held-out

---

## round4b 收摊 + 死题假设的判决性检验（03:40）

`round4b.log` 出现 `round4b 完成`，driver 退出，看守器按预设条件收尾。
GPU 已空（compute-apps 无进程，显存 316 MiB）。

### 1. 主判据过线，但只过 0.0020

| | iter3 | iter3b | Δ |
|---|---|---|---|
| **条件 acc**（事前主判据 > 0.7450） | 0.7315 | **0.7470** | **+0.0155** |
| sol_acc | 0.6627 | 0.6539 | **−0.0088** |
| 点名率 | 0.8444 | 0.8222 | −3/135 |

三项配对检验全部"与 0 分不开"：acc 净翻 5↑/5↓，McNemar p=1.0，连续置换 p=0.6414。
固定分组看这笔交易：iter3 命中的 114 道 **−0.0261**（10↑/20↓）；
iter3 未命中的 21 道 **+0.0850**（7↑/1↓，6 道新点名，点名率 28.6%）。
**是换血换来的，不是同批题变强。**

### 2. 判据分支有个静默缺陷：两个分支的条件同时成立

`check_round4b_criteria.py` 报的是"教材多调用上去了(5 条)但点名率仍 0.000
=> 「solver 见过但学不会」"。但它是 `if/elif` 顺序决定的，
**另一条分支的条件（acc 0.6539 < 0.6627）同时也成立**：

```python
if multi_up and rate <= MULTI_NAMEHIT_FLOOR:   # 真 => 报这条
elif m < 0.6627:                                # 也真，但被 elif 挡住
```

两条分支不是互斥的，脚本却只报一条。这跟
[[silent-failure-in-verification-code]] 是同一类病：**看起来给了判决，其实给了排序**。
下面第 3、4 节把两条假设分别测掉，不靠脚本的分支。

### 3. 假设 A「零梯度题拖累训练」——机制上不成立，证伪

这条是 PLAN.md:2385 预注册的分支。测法：数 round4b 教材里有多少条
`solver_agreement == 0`（tau=0 放进来的候选死题）。

| 轮次 | 教材 100 条里名口径=0 | 全等=0 | `frac_reward_zero_std` |
|---|---|---|---|
| round3 | 0 | 0 | 0.2400 |
| round4 | 0 | 22（名对参错，有梯度） | 0.2100 |
| **round4b** | **9** | 29 | **0.3500** |

**死题是真的：9 条，占教材 9%。** 但"拖累别人"这个机制不可能：

- `scale_rewards` 全仓没有任何地方覆盖，用 TRL 默认 **`"group"`**
  （grpo_config.py:784；`run_step3.sh` 传的参数里没有它）。
- `group` 模式下 `advantages = rewards - mean_grouped_rewards`，
  再除以**该组自己**的 std（grpo_trainer.py:2811-2813）。
- 所以一个死题组（4 个 reward 全 0）的优势恒 0，**数学上漏不进任何别的 prompt**。
  它只浪费自己那 9% 的名额。

**结论：`取中间 tau=0.1 再测` 这条预注册动作的依据被证伪，不执行。**

### 4. 那 0.35 的零方差是什么？——它是个混合指标，不能读成"教材信息量"

先修 `step3_signal.py`：原来 `rounds = [int(a) for a in sys.argv[1:]]`，
`int("4b")` 直接 ValueError，**round4b 这个数根本取不出来**——而它正是判据要用的那个。
改成按字符串标签拼路径。

拿到数之后发现**这个指标的读法从一开始就错了**：

- `is_std_zero` 是 `nanstd(组内 4 个 reward) == 0`（grpo_trainer.py:2814）。
  **"4 个全失败"和"4 个拿到完全一样的稠密奖励"都算零方差。**
- `per_device_train_batch_size=2`、`gradient_accumulation_steps=4`，
  所以每个 **optimizer step 实际过 8 个 prompt**，
  零方差取值只有 0 / 0.50 / 1.00 三档（实测分布确认，说明它是按 2 个 prompt 计的）。
  **我在这里错过一次**：日志里 `epoch` 从 0.02 走到 1.00，我据此写成
  "50 步 × 2 = 100 槽，每条教材只训一次"——**错的**。
  `epoch` 是没把梯度累积算进去的估计值。权威记录是
  `completions/completions_*.parquet`：50 个文件、每文件 8 行、
  100 个不同 prompt 每个正好出现 4 次 => **4 个 epoch，每条教材训 4 次**。
  所以 `mean(frac_reward_zero_std)` 读成"100 条里几条在空转"要乘 4 才等于总槽数
  （round4b 是 35 个槽 / 400 槽，不是 35/100）。
- 零方差 prompt 的 acc **高于**零方差为 0 的步，corr(零方差, acc) = +0.12~+0.19
  （四轮全部为正）：

| 轮次 | 零方差占比 | 这些 prompt 所在步的 acc | 全轮 acc |
|---|---|---|---|
| round2 | 0.1200（≈48/400 槽） | 0.5210 | 0.4881 |
| round3 | 0.2400（≈96/400 槽） | 0.5708 | 0.5324 |
| round4 | 0.2100（≈84/400 槽） | 0.7013 | 0.6837 |
| round4b | 0.3500（≈140/400 槽） | 0.6694 | 0.6436 |

（占比与粒度无关，可以直接跨轮比；槽数是占比 × 400 的估计，
因为每步只记到一个 2-prompt 微批的指标值，不是 4 个微批全记。）

**所以零方差大头是"已经会了/答得一致"，不是死题。**
round4b 的 35 条里 9 条是死题，剩下 26 条是饱和——
这也说明"零方差越低教材信息量越高"这个我写在脚本 docstring 里的读法是**单向错的**：
它同时数饱和。这个指标只能当佐证，不能当教材质量的度量。

### 5. 假设 B「solver 见过但学不会」——多调用确实有梯度，问题在探索

先排除"奖励看不见多调用"。`test_multicall_reward.py`（零 GPU，纯函数，
7 项全过）实测 `compute_accuracy_score`：

| 输入（gold 2 条） | reward |
|---|---|
| 只吐 1 个 call，完全答对其中一条 | **0.5000** |
| 吐 2 个 call，两条都对 | **1.0000** |
| 吐 3 个 call（多一个） | 0.8000 |

**性质 1（上界）**：只吐 1 个 call 时 reward 恒 ≤ 0.5。
所以**评测里 acc > 0.5 就是"这一条吐了 ≥2 个 call"的充分必要条件**——
这给了我们一个不依赖原始预测文件的指纹。
**性质 3（有梯度）**：第二个 call 值 0.5 的奖励空间，真实存在。

写测试时我两个假设都被自己推翻，都记在这里免得再犯：
1. 我猜 greedy 配对有顺序偏差（pred 被强行配给 gt[0]）——**错**。
   `if r > best_r` 严格大于且 `best_r` 初始 0.0，**零奖励的 pred 永不被占用**，
   单个 call 会被正确配到它对应的那条 gold，与 gold 书写顺序无关。
2. 我猜"2 条里答对 1 条"是 0.75——**错**，配不上的那条贡献 0 但仍占分母，
   所以是 1.0/2 = 0.5。

### 6. acc>0.5 指纹的全轮轨迹

| 轮次 | 多调用点名 | acc>0.5 条数 | 最大 acc | 多调用 acc | 判定 |
|---|---|---|---|---|---|
| base | 0/7 | 0 | 0.000 | 0.0000 | 从没吐过 |
| iter1 | 2/7 | 1 | **1.000** | 0.3321 | 吐过 |
| iter2 | 1/7 | 1 | 0.646 | 0.2792 | 吐过 |
| iter3 | 1/7 | 1 | 0.625 | 0.2786 | 吐过 |
| iter4 | 0/7 | 0 | 0.450 | 0.3136 | 从没吐过 |
| iter3b | 0/7 | 0 | 0.400 | 0.2230 | 从没吐过 |

**这条轨迹本身说明不了 p=0**：单次采样下若真实 p=0.05，7 次全没中的概率是 0.70。
所以下一步是多样本直接量（见下节）。

### 7. 判决：多调用不是"学不会"，是**被另外 95 条教材主动压制**

两条预注册假设都测完了，都不成立。真正的机制在训练日志里直接量到了。

**先量探索**（`probe_multicall.py`，7 道多调用 holdout 题 × 24 采样，temp 0.7 同评测条件）：

| checkpoint | 吐出 ≥2 个 call 的采样 |
|---|---|
| iter3_solver | 11/168 = **0.0655** |
| iter3b_solver | 7/168 = **0.0417** |

**两个都不是 0。** 所以"GRPO 根本采不到第二个 call"这个解释证伪——
行为是存在的，只是概率低（4~7%）。这也说明单次评测的 0/7 确实证明不了什么。

**再看训练期间它到底遭遇了什么**（`completions/completions_*.parquet`，
TRL `--log_completions 落的盘，每行一条 completion 带 reward 和 advantage）：

round4b 全量 400 个槽里，被记到的 completion 有 **8 个吐了 2 个 call**：

| step | calls | acc | advantage | 落在 |
|---|---|---|---|---|
| 2 | 2 | 0.4000 | **−0.8645** | 单调用题 |
| 5 | 2 | 0.8000 | **−1.4985** | 单调用题 |
| 10 | 2 | 0.4000 | **−1.4970** | 单调用题 |
| 11 | 2 | 0.4000 | **−1.4970** | 单调用题 |
| 14 | 2 | 0.6000 | +0.7605 | 单调用题 |
| 18 | 2 | 0.1600 | **−0.2712** | 单调用题 |
| 26 | 2 | 0.8000 | **−1.4985** | 单调用题 |
| 27 | 2 | 0.0000 | **−1.2346** | 单调用题 |

**8 个全部落在单调用题上，0 个落在多调用题上；7 个负优势、只有 1 个正。**
而在 5 条多调用题的 20 个训练槽里（4 epoch × 5 条），
被记到的 completion 是 19 个 1-call + 1 个 0-call，**2-call 一次都没出现**。

**完整的因果链（每一环都是实测，没有推断）**：

1. 2-call 行为的基线概率约 4~7%（多样本探针实测）
2. round4b 训练中它出现 8/400 次
3. **8 次全在单调用题上**——那里 gold 只有 1 条，多吐一个 call 触发
   `extra=1 → penalty=0.8`（rewards_solver.py:420），奖励被打折
4. 于是 **7/8 拿到负优势，被主动压制**；唯一 1 次正优势也在单调用题上，
   对多调用能力没有帮助
5. 该被奖励的 5 条多调用题上，20 个采样里这个行为**一次都没出现**
6. => 整个 round4b，2-call 行为收到的净梯度是**负的**
7. => held-out 多调用点名率 0.143 → **0.000**

**所以"教材里见过但学不会"的准确说法是：见过了，但见的方式是反的。**
5 条多调用教材不是"太少所以没用"，是**被 95 条单调用教材盖过了**——
后者每一次都在惩罚这个行为。这解释了为什么 round4b 和 iter4 的多调用
比 iter3 更差（点名率 0.143 → 0.000，iter4 也是 0.000）。

### 8. 这一条的代价上限，和它值不值得修

多调用在固定评测集里占 **7/135 = 5.2%**。修好它的上限就是预注册的 **+0.0208**
（7 道从 0.28 抬到 ~0.68）。代价是要把教材里多调用占比推到能盖过单调用压制的程度
（粗估 f 过半），而那会挤掉单调用题——**用一个 5.2% 的子集换 94.8% 的子集**。

参照 [[tool-r0-spec-count-no-effect]] 的同一结论：0.5B 基座没被教过这个输出契约，
RL 发现不了，该做的是 **SFT 冷启动**（教"什么时候该吐两个 call"的格式），
不是继续调 step2 的选材比例。选材这条路（tau_verify、mix、难度分档）已经走到头了。

**结论：tau_verify 不是多轮增益递减的根因。** 打开它确实让多调用教材从 1 条回到 5 条
（池里回到 7.5%），但对 held-out 多调用是净负的。iter3→iter4 的 +0.0340 条件 acc
来自口径修复（469 条"名对参错"回到教材），跟 tau_verify 无关——那条证据是独立的。

---

# 总结论（04:05，round4b 收摊后）

## 1. 三个问题的回答

**"多轮是有用的吗"——是，四轮连续上升，而且第 4 轮的上涨换了机制。**

固定评测集 135 题、同一份代码、同一个 seed、单次采样：

| 轮次 | sol_acc | 点名率 | 条件 acc | 未中支 | 多调用 acc |
|---|---|---|---|---|---|
| base | 0.0285 | 0.0741 | 0.3167 | 0.0055 | 0.0000 |
| iter1 | 0.4782 | 0.5778 | 0.5837 | 0.3339 | 0.3321 |
| iter2 | 0.6380 | 0.7926 | 0.7260 | 0.3017 | 0.2792 |
| iter3 | 0.6627 | 0.8444 | 0.7315 | 0.2893 | 0.2786 |
| iter4 | 0.6785 | 0.8370 | 0.7655 | 0.2316 | 0.3136 |

| 轮次 | sol_acc | 条件 acc | 点名 | 定性 |
|---|---|---|---|---|
| base→iter1 | +0.4497 | +0.2671 | +68/135 | 从零到敢答 |
| iter1→iter2 | +0.1598 | +0.1423 | +29/135 | 敢答和答对一起涨 |
| iter2→iter3 | +0.0247 | +0.0055 | +7/135 | **几乎全是"学会敢答"** |
| iter3→iter4 | +0.0158 | **+0.0340** | −1/135 | **几乎全是"学会答对"** |
| iter3→iter3b | −0.0088 | +0.0155 | −3/135 | 隔离实验，不是一轮 |

**增益递减在 iter3→iter4 停了**：条件 acc 增量从 +0.0055 跳到 +0.0340（6 倍），
而点名率已经饱和（113/135）。前 3 轮的增量主要来自点名率，
第 4 轮反过来——这是 [[tool-r0-agreement-full-match-fix]] 那个口径修复的直接证据：
469 条"名对参错"的题在名口径下 `agreement=1.0` 被当饱和丢掉（零梯度），
全等口径下 `<1.0` 留下（有梯度），所以第 4 轮学到的正是参数。
参数到位率从 67.9% 抬到 71.1%（108 道共同单调用题上量）。

**iter3b 不是一轮**，它和 iter4 一样从 iter3_solver 起跑，只换了 `tau_verify`，
所以不能接在 iter4 后面算"第 5 轮下降"。它的正确对照是 iter3（见第 3 节）。

**"能复现这一套吗"——能，而且复现过程本身抓出了 6 个会污染结论的 bug。**

**"是正确的吗"——四项独立证据都过，但每项都有必须连着的限定**（见第 4 节）。

## 2. 判据怎么定的（事前，不许赛后改）

- 主判据是**条件 acc > 0.7450**（连续量阈值），不是 sol_acc、不是显著性检验。
  理由：点名率饱和后，总量被"敢不敢答"主导，条件 acc 才反映"答得对不对"。
- round4b 的六条判据连阈值一起写在 `check_round4b_criteria.py` 里
  （`EXP_KEPT=1149` / `EXP_MULTI_FRAC=0.075` / `EXP_MULTI_N=5` /
  `EXP_KEYS3_N=18` / `ACC_FLOOR=0.6835` / `MULTI_NAMEHIT_FLOOR=0.30`），
  看到数字之前钉死。

## 3. round4b 的结论：tau_verify 不是根因，而且打开它有害

门确实打开了（池 1149 全进、多调用回到 7.5%、教材 5 条），
但 held-out 多调用点名率 0.143 → **0.000**。

两条预注册假设都测了，都不成立：
- **"零梯度题拖累训练"**——机制上不可能。`scale_rewards` 用 TRL 默认 `"group"`，
  优势按本组 std 归一化，死题组的零优势漏不进别的 prompt。证伪，不取 tau=0.1。
- **"GRPO 采不到第二个 call"**——多样本实测 P(吐2个call) = 6.55% / 4.17%，非零。证伪。

真机制（训练日志实测，PLAN.md 第 7 节）：**2-call 行为被 95 条单调用教材主动压制**。
round4b 的 400 个训练槽里 8 个 completion 吐了 2 个 call，
**8 个全在单调用题上、0 个在多调用题上、7/8 负优势**（多吐一个 call 触发
`extra=1 → penalty=0.8`）。净梯度为负，所以点名率归零。

**代价上限**：多调用占评测集 5.2%，修好上限 +0.0208，但要盖过压制得把教材占比推到过半
——用 5.2% 的子集换 94.8%。该做的是 SFT 冷启动教输出契约（同
[[tool-r0-spec-count-no-effect]] 的结论），不是继续调 step2 选材。

## 4. 必须连着的限定（否则上面的数字会被读错）

1. **三项配对检验全部"方向为正但不显著"**：round4 的 acc 净翻 6↑/3↓
   （n=135 要净翻 ≥10），McNemar p=0.5078，连续置换 p=0.3981。
   round4b 更平：5↑/5↓，p=1.0 / 0.6414。
   **事前判据是连续量阈值比较，不是显著性检验，两回事。**
2. **条件 acc 拿本轮自己的命中题算，两轮不是同一批题**
   （[[self-conditioning-breaks-round-comparison]]）。同题口径下 round4 仍 +0.0254、
   0.7686 同样过线，所以不是换血造出来的。
3. **round4 的同题增益是净 4 条的位移**：108 道共同题里 13 升 / 9 降 / **86 条平**。
   参数到位率 +0.0254 也是同一批上的净位移，不是普遍提升。
4. **`frac_reward_zero_std` 度量不了教材信息量**（见
   [[frac-reward-zero-std-counts-saturation-too]]）：它同时数"全失败"和"全答得一样"，
   实测零方差步的 acc 反而更高。只能当佐证。
5. **多调用样本量 n=7**，任何"多调用恢复了/没恢复"的结论都只有 7 个数据点。

## 5. 这一轮自查抓到的 6 个 bug（都会静默改变结论）

| # | 坑 | 后果 | 怎么发现的 |
|---|---|---|---|
| 1 | `run_round4b.sh` 在 `\| tee` 后用 `$?` 取退出码 | 判据永远显示"未过 0 条" | 写完后通读，改 `${PIPESTATUS[0]}`，全仓扫一遍 |
| 2 | `check_round4b_criteria.py` 的 `calls_of` 只读 `answer` | 池里多调用算出 **0.0% (0/1149)**，真实 7.5%，**凭空造了一条假 FAIL** | 与独立实现 `curriculum_composition.py` 输出对不上 |
| 3 | `compare_rounds.py` 的回落分支解包错误 | 前四轮六项指标一直单调，该分支从没执行过，一执行就崩 | round4 首次回落才踩到 |
| 4 | `step3_signal.py` 用 `int()` 解析轮次 | `int("4b")` ValueError，**round4b 的零方差取不出来**——正是判据要用的数 | 真要跑的时候 |
| 5 | 判据分支 `if/elif` 顺序 | 两个分支条件同时成立，只报第一个，另一个被静默吞掉 | 复核时逐条重算条件 |
| 6 | 我用日志 `epoch` 推"每条教材只训一次" | 实际 **4 个 epoch**（`gradient_accumulation_steps=4`，epoch 字段没算累积） | 数 `completions/*.parquet` 才发现 |

第 2 条和第 5 条是同类病：**失败的样子和成功的样子长得一样**
（[[silent-failure-in-verification-code]]）。第 2 条输出一个看起来合理的 FAIL，
第 5 条输出一个看起来是判决的排序。修法都是**用独立实现交叉验证**。

## 6. 现在盘上有什么（可复跑）

| 文件 | 作用 |
|---|---|
| `cond_acc_check.py` | 主判据（条件 acc 阈值）+ acc 分解式 |
| `check_round4b_criteria.py` | round4b 六条事前判据，退出码 = 未过条数 |
| `curriculum_composition.py` | 教材构成的独立实现（判据脚本的交叉验证） |
| `step3_signal.py` | 跨轮 step3 训练信号（轮次参数是字符串） |
| `mc_training_signal.py` | **读训练日志数 2-call 行为的强化/压制** |
| `probe_multicall.py` | 多样本量 P(吐≥2个call) |
| `test_multicall_reward.py` | 多调用计分三性质回归测试（零 GPU，7 项） |
| `report_round4.sh` | round4 五段汇报 |
| `watch_round4b.sh` | 心跳看守（含坏行过滤、按完成标志退出） |

## 7. 还没做的（按优先级）

1. **多调用要 SFT 冷启动**：0.5B 基座没被教过"什么时候吐两个 call"，RL 发现不了。
   这是唯一被证据指向的下一步，不是继续调 step2。
2. `--beta 0.01` 已补，但 `beta` 的影响没单独量过（iter1 是在 beta=0 下跑的）。
3. 未决：V1 与 V0 引擎是否改变数值；`GEN_MAX_NEW=4096` 对 iter1_generator 的影响。
4. 点名率已饱和（113/135），剩下 22 条未命中的是长尾：7 条多调用 + 15 条单调用
   分散在 15 个不同工具上。单调用那 15 条没有单一抓手。
