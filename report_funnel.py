#!/usr/bin/env python3
""" 一轮的漏斗报告：step2 出了多少题 -> verify 活多少 -> 剔多少饱和 -> 难度池长什么样 -> 终选配比。

为什么单独写一个：round3 栽在"饱和"这个数上——656 条被当饱和丢掉，其中 296 条其实是
"名对参错"，该留着训。当时是我手工 grep 日志才发现的，而那种手工活在 round4 还要再来
一遍，且这次要盯的是"hard 池有没有回来"。写成脚本，iter2/iter3 先跑一遍确认它复现得出
已知数（复现不出就是脚本错了，不是数据错了），round4 一完直接出表。

用法：
  python report_funnel.py 2 3        # 报两轮对照
  python report_funnel.py 4          # 只报一轮
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "qwen2.5-0.5b-instruct-tool-r0")


def band(a):
    """和 step2_select_curriculum.assign_difficulty_from_agreement 同一套边界。"""
    if a is None:
        return "medium(兜底)"
    if a < 0.4:
        return "hard"
    if a < 0.8:
        return "medium"
    return "easy"


def report(it, data_override=None):
    log = os.path.join(HERE, "round%d.step2.log" % it)
    inter = data_override or os.path.join(
        BASE, "iter%d_data.json.intermediate.verified.json" % it)
    final = os.path.join(BASE, "iter%d_data.json" % it)

    print("=" * 78)
    print("round %d" % it)
    print("=" * 78)

    if not os.path.exists(log):
        # 不早退：日志只是佐证，数据本身的分析照样能跑。
        # 早退会让人误以为"没有日志就什么都看不了"，而 round 跑到一半时正是这样。
        print("  （没有 round%d.step2.log，只有数据侧的数字）" % it)
        txt = ""
    else:
        txt = open(log, errors="replace").read()

    out = {"round": it}

    # 1) verify 产出
    m = re.search(r"wrote kept=(\d+) dropped=(\d+)", txt)
    if m:
        kept, dropped = int(m.group(1)), int(m.group(2))
        tot = kept + dropped
        print("  verify      出题 %d -> 活 %d（格式/解析不过关掉 %d，%.1f%%）"
              % (tot, kept, dropped, 100.0 * dropped / max(tot, 1)))
        out.update(gen_total=tot, verify_kept=kept, verify_dropped=dropped)
    else:
        print("  verify      （日志里没找到 kept= 行）")

    # 2) 剔饱和
    m = re.search(r"\[saturation\] dropped (\d+) already-solved", txt)
    if m:
        print("  剔饱和      %s 条（agreement >= 1.0）" % m.group(1))
        out["n_saturated"] = int(m.group(1))
    else:
        print("  剔饱和      （没找到）")

    # 3) 难度池。round3 真正出事的地方。
    #    注意 intermediate.verified.json 是**剔饱和之前**的（952 条），
    #    而日志里的 "296 left" 是剔之后的。这两个数回答两个不同的问题：
    #      剔前 hard 数 = generator 到底产出了多少难题（产出端）
    #      剔后 hard 数 = 还有多少难题能拿来训（可用端，round4 的预测指这个）
    #    round3 的毛病正是"产出端不算少，可用端塌了"——所以两个都要看。
    if not os.path.exists(inter):
        print("  难度池      （没有 %s）" % os.path.basename(inter))
        rows = None
    else:
        rows = json.load(open(inter))
        # 新代码写 solver_agreement_full，老数据只有 solver_agreement。
        # 用 full 当 full，没有才退名口径，并在标签里注明，免得两轮口径混着比。
        vals, used_fallback = [], 0
        for r in rows:
            meta = r.get("meta", {}) or {}
            a = meta.get("solver_agreement_full")
            if a is None:
                a = meta.get("solver_agreement")
                if a is not None:
                    used_fallback += 1
            vals.append(a)

        def dist(vs):
            d = {}
            for v in vs:
                d[band(v)] = d.get(band(v), 0) + 1
            return d

        order = ["hard", "medium", "easy", "medium(兜底)"]

        def fmt(d):
            return "  ".join("%s=%d" % (k, d.get(k, 0)) for k in order if d.get(k))

        pre = dist(vals)
        print("  难度池(剔前) %d 条 -> %s" % (len(rows), fmt(pre)))

        # 剔饱和：和日志同一个规则（agreement >= 1.0），但用当前文件里的口径。
        # 老数据只有名口径，所以 iter3 这里复现出来的就是"656"那个数；
        # round4 起 intermediate 会带 solver_agreement_full，剔的数会小很多。
        post_vals = [v for v in vals if v is None or v < 1.0]
        post = dist(post_vals)
        print("  难度池(剔后) %d 条 -> %s" % (len(post_vals), fmt(post)))
        if used_fallback:
            print("              其中 %d 条只有名口径（老数据），已按名口径分档" % used_fallback)
        out.update(pool_pre=pre, pool_post=post, n_pool_pre=len(rows),
                   n_pool_post=len(post_vals), n_names_only=used_fallback)

    # 4) 终选配比
    m = re.search(r"\[select\] diff: \{([^}]*)\}", txt)
    if m:
        print("  终选配比    %s" % m.group(1).strip())
        out["select_mix"] = m.group(1).strip()

    # 5) 两个口径的对比。round4 的修复到底救回了多少，全在这一块。
    #
    # 为什么不 grep 日志：step2_genverify.py 把这两行写进 verify_report.txt
    # （`with open(args.report_txt) as rf: rf.write(...)`），不是 print，
    # 所以 round{N}.step2.log 里**根本没有它们**。第一版就是 grep 日志的，
    # 于是永远不匹配、也不报错，只是安静地什么都不打——
    # 一个不匹配的正则和一个"没有名对参错"的世界长得一模一样。
    # 而"没有名对参错"恰好是 round4 修复**失败**时的样子，正好是最不能看错的那一个。
    # 改成从 intermediate 的每行 meta 直接算：同一批数，但是原始值，
    # 而且 round2/3 这种只有名口径的老数据也能算（full 记 None）。
    if rows is not None:
        pairs = []
        for r in rows:
            m = r.get("meta", {}) or {}
            a, f = m.get("solver_agreement"), m.get("solver_agreement_full")
            if a is not None or f is not None:
                pairs.append((a, f))
        have_full = [(a, f) for a, f in pairs if f is not None]
        if have_full:
            names = [a for a, f in have_full if a is not None]
            fulls = [f for a, f in have_full]
            both = sum(1 for a, f in have_full
                       if a is not None and a == 1.0 and f < 1.0)
            print("  名对参错    %d 条（名口径=1.0 但全等<1.0，旧口径把这些当饱和丢掉）" % both)
            if names:
                mn, mf = sum(names) / len(names), sum(fulls) / len(fulls)
                print("  两口径均值  名 %.4f / 全等 %.4f（差 %.4f，样本 %d）"
                      % (mn, mf, mn - mf, len(have_full)))
                out.update(both_perfect=both, mean_names=mn, mean_full=mf,
                           n_full=len(have_full))
        else:
            print("  名对参错    （这批数据没有 solver_agreement_full，"
                  "是 2026-10-16 改口径之前跑的，旧口径无从对比）")
    print()
    # 返回数字而不只返回 hard 数：事前判据是几个数一起看的，
    # 只返回一个 hard 会逼着调用方再把这段解析一遍。
    out["hard"] = out.get("pool_post", {}).get("hard")
    return out


if __name__ == "__main__":
    argv = sys.argv[1:]
    data = None
    if "--data" in argv:
        i = argv.index("--data")
        data = argv[i + 1]
        del argv[i:i + 2]
    if not argv:
        sys.exit(__doc__)
    rounds = []
    for a in argv:
        try:
            rounds.append(int(a))
        except ValueError:
            sys.exit("轮次要是整数，收到 %r" % a)
    res = {r: report(r, data if len(rounds) == 1 else None) for r in rounds}

    if len(rounds) >= 2:
        print("=" * 78)
        print("hard 池逐轮（round3 从这里开始塌，round4 看它回不回来）")
        print("=" * 78)
        prev = None
        for r in rounds:
            h = res[r].get("hard")
            if h is None:
                continue
            delta = "" if prev is None else "  (%+d)" % (h - prev)
            print("  round %-3d hard 池 = %3d 条%s" % (r, h, delta))
            prev = h
