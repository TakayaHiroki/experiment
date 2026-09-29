#!/usr/bin/env python3
"""埋め込みスクリプト（embedding_models/*.py）が，生成したJSONをそのまま読めるかを検証する

usage: python tools/RQ1/check_schema.py --dir corpus/bewt/json
"""
import argparse, glob, json, os, sys

VARIANTS = ["full", "title", "steps", "expect"]

PARTS = {"title": lambda r: [r["title"]],
         "steps": lambda r: r["steps"],
         "expect": lambda r: r["expects"],
         "full": lambda r: [r["title"]] + r["steps"] + r["expects"]}


def check_file(path):
    errs = []
    try:
        with open(path, encoding="utf-8") as f:
            tests = json.load(f)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        return [f"JSONとして読み込めない: {e}"], 0

    if not isinstance(tests, list):
        return [f"最上位がリストでない（{type(tests).__name__}）"], 0
    if not tests:
        return ["要素が0件"], 0

    try:
        titles = [t["title"] for t in tests]
        texts = [t["text_for_embedding"] for t in tests]
    except (KeyError, TypeError) as e:
        return [f"既存スクリプトと同じ読み方で失敗: {e}"], len(tests)

    for i, (ti, tx) in enumerate(zip(titles, texts)):
        if not isinstance(ti, str) or not ti.strip():
            errs.append(f"[{i}] title が空または文字列でない")
        if not isinstance(tx, str) or not tx.strip():
            errs.append(f"[{i}] text_for_embedding が空または文字列でない")

    app, sep, fname_v = os.path.basename(path)[:-len(".json")].rpartition("_")
    if not sep or not app or fname_v not in VARIANTS:
        errs.append(f"ファイル名が {{app}}_{{{'|'.join(VARIANTS)}}}.json の形でない")
        return errs, len(tests)
    bad = [i for i, t in enumerate(tests) if t.get("variant") != fname_v]
    if bad:
        errs.append(f"variant 欄がファイル名と一致しない {len(bad)}件"
                    f"（例 [{bad[0]}]: {tests[bad[0]].get('variant')!r} "
                    f"≠ {fname_v!r}）")
    ng = []
    for i, t in enumerate(tests):
        try:
            parts = PARTS[fname_v](t)
            ok = isinstance(parts, list) and all(isinstance(x, str) for x in parts)
        except (KeyError, TypeError):
            ok = False
        if not ok:
            ng.append(i)
            continue
        want = " ".join(x.strip() for x in parts if x.strip())
        if want != t["text_for_embedding"]:
            ng.append(i)
    if ng:
        errs.append(f"**text_for_embedding を成分から復元できない {len(ng)}件**"
                    f"（例 [{ng[0]}]）．監査できない")
    return errs, len(tests)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", "-d", required=True)
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.dir, "*.json")))
    if not files:
        sys.exit(f"{a.dir} に .json が無い")

    ng = 0
    total_records = 0
    for p in files:
        errs, n = check_file(p)
        total_records += n
        if errs:
            ng += 1
            print(f"[FAIL] {os.path.basename(p)} ({n}件)")
            for e in errs[:5]:
                print(f"        {e}")
        else:
            print(f"[ OK ] {os.path.basename(p)} ({n}件)")

    print("=" * 60)
    print(f"ファイル {len(files)} 件 / レコード合計 {total_records} 件 / 不合格 {ng} 件")
    if ng:
        print("[FAIL] 埋め込みスクリプトがそのまま読めないファイルがある")
        sys.exit(1)
    print("[PASS] 埋め込みスクリプトがそのまま読めるスキーマ")


if __name__ == "__main__":
    main()
