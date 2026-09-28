#!/usr/bin/env python3
"""ベクトル化の実行記録（logs/embed_runlog.csv）を条件ごとに集計する

usage: python tools/RQ1/summarize_embed_runlog.py [--log logs/embed_runlog.csv]
"""
import argparse, csv, json, os
from collections import Counter

LEXICAL = ["tfidf", "lsa", "lsa-full"]
MIN_FREE_MB = 1024
MAX_BUSY_PCT = 10
# 8コアを使い切れていれば8付近．8の-25%を境にする
ENCODE_RATIO_MIN = 6.0


def num(r, k):
    # 列が無い古いログでも落ちないようにする
    return float(r[k]) if r.get(k) else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="logs/embed_runlog.csv")
    a = ap.parse_args()

    with open(a.log, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    conds = list(dict.fromkeys(r["model"] for r in rows))
    by = {c: [r for r in rows if r["model"] == c] for c in conds}

    def total(c, k):
        return sum(num(r, k) for r in by[c])

    def values(c, k):
        return "/".join(sorted({r[k] for r in by[c]})) or "-"

    print(f"行数 {len(rows)} / 条件・アプリ・バリアントの組 {len({(r['model'], r['app'], r['variant']) for r in rows})}"
          f" / 条件 {len(conds)}")

    print("\n1. 所要時間と資源の使用量")
    print("   経過・CPU・構築 = total_s・cpu_total_s・t_load_s の合計（分），"
          "ピーク = peak_rss_mb の最大，空き = mem_free_min_mb の最小")
    print(f"{'条件':<28}{'行':>3}{'dtype':>9}{'batch':>6}{'params(B)':>10}"
          f"{'経過':>8}{'CPU':>8}{'構築':>7}{'ピーク':>7}{'空き':>6}")
    for c in conds:
        params = "---" if c in LEXICAL else f"{num(by[c][0], 'params') / 1e9:.2f}"
        print(f"{c:<28}{len(by[c]):>3}{values(c, 'dtype'):>9}{values(c, 'batch'):>6}{params:>10}"
              f"{total(c, 'total_s') / 60:>8.1f}{total(c, 'cpu_total_s') / 60:>8.1f}{total(c, 't_load_s') / 60:>7.1f}"
              f"{max(num(r, 'peak_rss_mb') for r in by[c]):>7.0f}{min(num(r, 'mem_free_min_mb') for r in by[c]):>6.0f}")

    print("\n2. CPU時間と経過時間の比（語彙的手法を除く．全体の降順）")
    print("   全体 = Σcpu_total_s / Σtotal_s，本番 = Σc_encode_s / Σt_encode_s，構築の割合 = Σt_load_s / Σtotal_s")
    print(f"   本番が {ENCODE_RATIO_MIN} を下回る条件に [x] を付ける（8コアを使い切れていない）")
    ratio = {c: (total(c, "cpu_total_s") / total(c, "total_s"), total(c, "c_encode_s") / total(c, "t_encode_s"),
                 total(c, "t_load_s") / total(c, "total_s")) for c in conds if c not in LEXICAL}
    for c in sorted(ratio, key=lambda c: -ratio[c][0]):
        mark = "[x]" if ratio[c][1] < ENCODE_RATIO_MIN else "   "
        print(f"{mark} {c:<28} 全体 {ratio[c][0]:.2f}  本番 {ratio[c][1]:.2f}  構築の割合 {ratio[c][2]:.0%}")
    slow = sorted((c for c in ratio if ratio[c][1] < ENCODE_RATIO_MIN), key=lambda c: ratio[c][1])
    print(f"   時間を使わない条件（本番 < {ENCODE_RATIO_MIN}）: "
          + (" ".join(f"{c}({ratio[c][1]:.2f})" for c in slow) if slow else "なし"))
    if slow:
        print("   → 所要時間は報告に使わず，メモリのピークとパラメータ数で報告する（完成基準 3）")

    print("\n3. モデルファイルの先読み（1GiB = 2^30 バイト）")
    for r in rows:
        if r["t_warm_s"]:
            gib = num(r, "warm_bytes") / 2**30
            print(f"{r['model']:<28} {gib:5.1f} GiB  {num(r, 't_warm_s'):6.2f} 秒  {gib / num(r, 't_warm_s'):.2f} GiB/秒")

    print("\n4. status の件数")
    for k, n in Counter(r["status"] for r in rows).most_common():
        print(f"  {k:<11}{n:>4}")

    print(f"\n5. 空きメモリ（mem_free_min_mb）が {MIN_FREE_MB}MB を下回った実行")
    low = [r for r in rows if num(r, "mem_free_min_mb") < MIN_FREE_MB]
    print(f"  合計 {len(low)} 件 / status の内訳 {dict(Counter(r['status'] for r in low))}")
    for c in conds:
        lc = [r for r in low if r["model"] == c]
        if lc:
            print(f"  {c:<28}{len(lc):>3}/{len(by[c])}  最小 {min(num(r, 'mem_free_min_mb') for r in lc):.0f}MB"
                  f"  status の内訳 {dict(Counter(r['status'] for r in lc))}")

    print(f"\n6. 他プロセスの負荷（{MAX_BUSY_PCT}% 超）")
    before = [r for r in rows if num(r, "cpu_busy_before_pct") > MAX_BUSY_PCT]
    during = [r for r in rows if num(r, "other_cpu_during_pct") > MAX_BUSY_PCT]
    print(f"  実行前（cpu_busy_before_pct） {len(before)} 件 / 実行中（other_cpu_during_pct） {len(during)} 件")
    for c in conds:
        dc = [r for r in during if r["model"] == c]
        if dc:
            print(f"  {c:<28}{len(dc):>3}/{len(by[c])}  最大 {max(num(r, 'other_cpu_during_pct') for r in dc):.1f}%")

    print("\n7. 計測環境")
    for name in sorted({r["env_file"] for r in rows}):
        with open(os.path.join(os.path.dirname(a.log), name), encoding="utf-8-sig") as f:
            env = json.load(f)
        print(f"  {name}: ram_total_mb={env['ram_total_mb']} logical_procs={env['logical_procs']} "
              f"threads={env['threads']} affinity_mask={env['affinity_mask']}")

    print("\n8. 入力の切り捨て（truncated）")
    cut = [r for r in rows if num(r, "truncated") > 0]
    if not cut:
        print("  切り捨ては無い")
    else:
        print(f"  切り捨てが起きた実行 {len(cut)} 件 / 全 {len(rows)} 件")
        for c in conds:
            cc = [r for r in cut if r["model"] == c]
            if cc:
                detail = " ".join(f"{r['app']}/{r['variant']}:{int(num(r, 'truncated'))}" for r in cc)
                print(f"  {c:<28}{len(cc):>3}/{len(by[c])}  {detail}")


if __name__ == "__main__":
    main()
