#!/usr/bin/env python3
"""ベクトル化の実行記録（logs/embed_runlog.csv）を条件ごとに集計する

usage: python tools/RQ1/summarize_embed_runlog.py [--log logs/embed_runlog.csv]
"""
import argparse, csv, json, os
from collections import Counter

LEXICAL = ["tfidf", "lsa", "lsa-full"]


def num(r, k):
    return float(r[k]) if r.get(k) else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="logs/embed_runlog.csv")
    a = ap.parse_args()

    with open(a.log, encoding="utf-8-sig") as f:
        log = list(csv.DictReader(f))
    latest = {}
    for r in log:
        latest[(r["model"], r["app"], r["variant"])] = r
    failed = [r for r in latest.values() if r["status"] == "failed"]
    rows = [r for r in log if latest[(r["model"], r["app"], r["variant"])] is r and r["status"] != "failed"]
    conds = list(dict.fromkeys(r["model"] for r in rows))
    by = {c: [r for r in rows if r["model"] == c] for c in conds}
    envs = {}
    for name in sorted({r["env_file"] for r in rows}):
        with open(os.path.join(os.path.dirname(a.log), name), encoding="utf-8-sig") as f:
            envs[name] = json.load(f)

    def total(c, k):
        return sum(num(r, k) for r in by[c])

    def values(c, k):
        return "/".join(sorted({r[k] for r in by[c]})) or "-"

    print(f"行数 {len(log)} / 条件・アプリ・バリアントの組 {len(latest)} / 集計に使う行 {len(rows)}"
          f"（組ごとに最後の行．再実行で置き換わった {len(log) - len(latest)} 行と，最後が failed の {len(failed)} 組を除く）"
          f" / 条件 {len(conds)}")
    for r in failed:
        print(f"  最後が failed: {r['model']} {r['app']}/{r['variant']}"
              f"（経過 {num(r, 'total_s'):.0f} 秒，外から見た確保量のピーク {r['peak_commit_ext_mb'] or '-'} MB）")

    print("\n1. 所要時間と資源の使用量")
    print("   経過 = total_s の合計（分），CPU = c_load_s + c_encode_s の合計（分），構築 = t_load_s の合計（分），"
          "本番 = t_encode_s の合計（秒），ピーク = peak_commit_mb の最大（MB）")
    print(f"{'条件':<28}{'行':>3}{'dtype':>9}{'batch':>6}{'params(B)':>10}"
          f"{'経過':>8}{'CPU':>8}{'構築':>7}{'本番':>11}{'ピーク':>8}")
    for c in conds:
        params = "---" if c in LEXICAL else f"{num(by[c][0], 'params') / 1e9:.2f}"
        print(f"{c:<28}{len(by[c]):>3}{values(c, 'dtype'):>9}{values(c, 'batch'):>6}{params:>10}"
              f"{total(c, 'total_s') / 60:>8.1f}{(total(c, 'c_load_s') + total(c, 'c_encode_s')) / 60:>8.1f}"
              f"{total(c, 't_load_s') / 60:>7.1f}{total(c, 't_encode_s'):>11.3f}{max(num(r, 'peak_commit_mb') for r in by[c]):>8.0f}")

    print("\n2. 時間の採否")
    print("   時間を使うのは，ok_unbound の行が無い条件だけ．本番の比と5の負荷は記述の値で，採否には使わない")
    print("   本番の比 = Σc_encode_s / Σt_encode_s（語彙的手法は -），構築の割合 = Σt_load_s / Σtotal_s")
    print("   [u] ok_unbound の行がある")
    marks = {}
    for c in conds:
        unbound = sum(r["status"] == "ok_unbound" for r in by[c])
        marks[c] = "[u]" if unbound else ""
        ratio = "-" if c in LEXICAL else f"{total(c, 'c_encode_s') / total(c, 't_encode_s'):.2f}"
        print(f"{marks[c]:<4} {c:<28} 本番の比 {ratio:>4}  unbound {unbound}/{len(by[c])}"
              f"  構築の割合 {total(c, 't_load_s') / total(c, 'total_s'):.0%}")
    unused = [c for c in conds if marks[c]]
    print("   時間を使わない条件: " + (" ".join(f"{c}{marks[c]}" for c in unused) if unused else "なし"))
    if unused:
        print("   → その行の出力ファイルを消し，ほかのアプリを閉じて再実行する．再実行しても残る条件は時間を使わない")

    print("\n3. モデルファイルの先読み（1GiB = 2^30 バイト．置き換わった行も含め，先読みした回ごと）")
    for r in log:
        if r["t_warm_s"]:
            gib, t = num(r, "warm_bytes") / 2**30, num(r, "t_warm_s")
            # 記録は小数第2位までなので，0.01秒未満は 0.00 になる
            speed = f"{gib / t:.2f}" if t > 0 else "-"
            print(f"{r['model']:<28} {gib:5.1f} GiB  {t:6.2f} 秒  {speed} GiB/秒")

    print("\n4. status の件数（組ごとの最後の行）")
    for k, n in Counter(r["status"] for r in latest.values()).most_common():
        print(f"  {k:<11}{n:>4}")

    print("\n5. 計測中の負荷（記述の値．マシン全体の論理プロセッサに対する割合．条件ごとの最大）")
    print("   他プロセス = other_cpu_during_pct，OS のメモリ管理 = os_mem_cpu_pct")
    for c in conds:
        print(f"  {c:<28}他プロセス {max(num(r, 'other_cpu_during_pct') for r in by[c]):5.1f}%"
              f"  OS のメモリ管理 {max(num(r, 'os_mem_cpu_pct') for r in by[c]):5.1f}%")

    print("\n6. 計測環境")
    for name, env in envs.items():
        print(f"  {name}: ram_total_mb={env['ram_total_mb']} logical_procs={env['logical_procs']} "
              f"threads={env['threads']} affinity_mask={env['affinity_mask']} "
              f"KMP_BLOCKTIME={env['python']['env']['KMP_BLOCKTIME']}")

    print("\n7. 入力の切り捨て（truncated）")
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
