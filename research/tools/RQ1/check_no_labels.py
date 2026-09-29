#!/usr/bin/env python3
"""BEWT コーパスに「正解ラベル（機能分類）」が存在しないことを，成果物から確認する（P1-4）

usage: python tools/RQ1/check_no_labels.py [--file corpus/bewt/json/ALL_full.json]
"""
import argparse, collections, json, sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", "-f", default="corpus/bewt/json/ALL_full.json")
    a = ap.parse_args()

    with open(a.file, encoding="utf-8") as f:
        d = json.load(f)

    if not isinstance(d, list) or not d:
        sys.exit(f"[FAIL] {a.file} がレコードのリストになっていない，または0件．"
                 "P1-2 からやり直す")
    bad = [i for i, r in enumerate(d)
           if not isinstance(r, dict) or not r.get("app") or not r.get("spec")]
    if bad:
        sys.exit(f"[FAIL] app / spec が空のレコードが {len(bad)}件ある（例 [{bad[0]}]）．"
                 "spec を持たないコーパスでは1対1かどうかを判定できない")

    apps = sorted(set(r["app"] for r in d))
    spec = collections.Counter(r["app"] + "/" + r["spec"] for r in d)
    singleton = sum(1 for v in spec.values() if v == 1)
    grouped = sorted(((k, v) for k, v in spec.items() if v > 1),
                     key=lambda kv: -kv[1])

    print(f"テスト数      : {len(d)}")
    print(f"アプリ数      : {len(apps)}  ({', '.join(apps)})")
    print(f"spec 数       : {len(spec)}")
    print(f"1テストのみの spec: {singleton}")
    print("=" * 60)

    if not grouped:
        print("[PASS] spec はテストと1対1．複数テストを束ねる単位は存在しない")
        print("       -> 正解ラベル（機能分類）は無い．ラベル不使用の設計で進む")
        return

    print(f"[WARN] 複数テストを持つ spec が {len(grouped)} 件ある")
    for k, v in grouped[:10]:
        print(f"        {k}: {v}件")
    print("       外部（データセット提供元）由来の構造なら正解に使える")
    print("       ファイル名などから自作したものは使わない（-> P1-4）")
    sys.exit(1)


if __name__ == "__main__":
    main()
