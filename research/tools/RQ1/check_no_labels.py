#!/usr/bin/env python3
"""BEWT コーパスに全アプリにそろった正解ラベル（機能分類）が無いことを，成果物と原文の Feature: 行から確認する

usage: python tools/RQ1/check_no_labels.py [--file corpus/bewt/json/ALL_full.json] [--bewt BEWT]
"""
import argparse, collections, glob, json, os, re, sys

HEAD_RE = re.compile(r"^(Feature|Scenario)\s*:\s*(.*)$")


def feature_groups(bewt_dir):
    groups = collections.defaultdict(collections.Counter)
    n_files = collections.Counter()
    for p in glob.glob(os.path.join(bewt_dir, "*", "gherkin", "**", "*.feature"), recursive=True):
        app = os.path.relpath(p, bewt_dir).replace(os.sep, "/").split("/")[0]
        n_files[app] += 1
        with open(p, encoding="utf-8") as f:
            heads = [m.groups() for m in (HEAD_RE.match(ln.strip()) for ln in f) if m]
        feature = next((t.strip() for k, t in heads if k == "Feature"), None)
        if feature is not None and any(k == "Scenario" for k, _ in heads):
            groups[app][feature] += 1
    return groups, n_files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", "-f", default="corpus/bewt/json/ALL_full.json")
    ap.add_argument("--bewt", "-b", default="BEWT")
    a = ap.parse_args()

    with open(a.file, encoding="utf-8") as f:
        d = json.load(f)

    if not isinstance(d, list) or not d:
        sys.exit(f"[FAIL] {a.file} がレコードのリストになっていない，または0件．"
                 "plan2json.py で作り直すこと")
    bad = [i for i, r in enumerate(d)
           if not isinstance(r, dict) or not r.get("app") or not r.get("spec")]
    if bad:
        sys.exit(f"[FAIL] app / spec が空のレコードが {len(bad)}件ある（例 [{bad[0]}]）．"
                 "spec を持たないコーパスでは1対1かどうかを判定できない")
    groups, n_files = feature_groups(a.bewt)
    if not n_files:
        sys.exit(f"[FAIL] {a.bewt} の下に .feature が無い")

    apps = sorted(set(r["app"] for r in d))
    spec = collections.Counter(r["app"] + "/" + r["spec"] for r in d)
    singleton = sum(1 for v in spec.values() if v == 1)
    grouped = sorted(((k, v) for k, v in spec.items() if v > 1),
                     key=lambda kv: -kv[1])
    with_feature = [app for app in apps if groups[app]]

    print(f"テスト数      : {len(d)}")
    print(f"アプリ数      : {len(apps)}  ({', '.join(apps)})")
    print(f"spec 数       : {len(spec)}")
    print(f"1テストのみの spec: {singleton}")
    print("Feature: と Scenario: を両方持つ .feature（Feature: がテストをまとめる見出し）:")
    for app in apps:
        g = groups[app]
        detail = f"  {len(g)} 種（" + ", ".join(f"{k} {v}" for k, v in g.most_common()) + "）" if g else ""
        print(f"  {app}: {sum(g.values())} / {n_files[app]} ファイル{detail}")
    print("=" * 60)

    if grouped:
        print(f"[WARN] 複数テストを持つ spec が {len(grouped)} 件ある")
        for k, v in grouped[:10]:
            print(f"        {k}: {v}件")
        print("       外部（データセット提供元）由来の構造なら正解に使える")
        print("       ファイル名などから自作したものは使わない")
        sys.exit(1)
    if len(with_feature) == len(apps):
        print("[WARN] 全アプリで Feature: がテストをまとめている．全アプリにそろった機能分類がある")
        sys.exit(1)

    print("[PASS] spec はテストと1対1．提供元の Feature: によるまとまりは "
          f"{len(apps)} アプリ中 {len(with_feature)} アプリ（{', '.join(with_feature) or '-'}）だけ")
    print("       -> 全アプリにそろった正解ラベル（機能分類）は無い．ラベル不使用の設計で進む")


if __name__ == "__main__":
    main()
