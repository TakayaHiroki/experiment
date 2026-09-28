#!/usr/bin/env python3
"""ベクトル化に渡すテキストが BEWT の原文の行そのものだけでできていることを検証する

usage:
  python tools/RQ1/check_no_injection.py
  python tools/RQ1/check_no_injection.py --dir corpus/bewt/json --bewt BEWT [--source corpus/bewt/source]
  --source を付けると，テストの過不足と，title・steps・expects が出典 JSON の並びと順序・個数まで一致するかも見る
"""
import argparse, glob, json, os, re, sys

KW_RE = re.compile(r"^(?:Given|When|And|Then)\s+(.*)$")
HEAD_RE = re.compile(r"^(?:Feature|Scenario)\s*:\s*(.*)$")


def load_features(bewt_dir):
    feats = {}
    for p in glob.glob(os.path.join(bewt_dir, "*", "gherkin", "**", "*.feature"), recursive=True):
        app = os.path.relpath(p, bewt_dir).replace(os.sep, "/").split("/")[0]
        heads, body = set(), set()
        with open(p, encoding="utf-8") as f:
            for ln in f:
                s = ln.strip()
                m = HEAD_RE.match(s)
                if m:
                    heads.add(m.group(1).strip())
                    continue
                m = KW_RE.match(s)
                if m:
                    body.add(m.group(1).strip())
        feats[(app, os.path.basename(p)[:-len(".feature")])] = (heads, body)
    return feats


def load_source(source_dir):
    src = {}
    for p in glob.glob(os.path.join(source_dir, "*.json")):
        with open(p, encoding="utf-8") as f:
            meta = json.load(f)
        for t in meta["tests"]:
            app, spec = t["id"].split("/", 1)
            src[(app, spec)] = (t["title"], [r["text"] for r in t["lines"] if r["role"] == "perform"],
                                [r["text"] for r in t["lines"] if r["role"] == "expect"])
    return src


def check_file(path, feats, src_map):
    try:
        with open(path, encoding="utf-8") as f:
            tests = json.load(f)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        return [f"JSONとして読み込めない: {e}"], 0
    if not isinstance(tests, list) or not tests:
        return ["最上位が空でないリストでない"], 0

    errs, ng, got = [], [], []
    for i, t in enumerate(tests):
        try:
            title, app, spec = t["title"], t["app"], t["spec"]
            steps, expects = t["steps"], t["expects"]
        except (KeyError, TypeError) as e:
            errs.append(f"[{i}] 項目が足りない: {e}")
            continue
        got.append((app, spec))
        src = feats.get((app, spec))
        if src is None:
            ng.append(f"[{i}] 元の .feature が見つからない: {app}/{spec}")
            continue
        heads, body = src
        if title not in heads:
            ng.append(f"[{i}] title が原文の見出しと一致しない: 「{title[:60]}」（{app}/{spec}）")
        for s in list(steps) + list(expects):
            if s not in body:
                ng.append(f"[{i}] 原文の行と一致しない: 「{s[:60]}」（{app}/{spec}）")
        if src_map is not None:
            want = src_map.get((app, spec))
            if want is None:
                ng.append(f"[{i}] 出典 JSON に無い: {app}/{spec}")
            else:
                for name, a, b in (("title", title, want[0]), ("steps", list(steps), want[1]), ("expects", list(expects), want[2])):
                    if a != b:
                        n = f"（{len(a)} 件 / 出典 {len(b)} 件）" if isinstance(a, list) else ""
                        ng.append(f"[{i}] {name} が出典 JSON の並びと一致しない{n}（{app}/{spec}）")
    if src_map is not None:
        want_ids = set(src_map) if os.path.basename(path).startswith("ALL_") else {k for k in src_map if k[0] in {a for a, _ in got}}
        if len(got) != len(set(got)) or set(got) != want_ids:
            ng.append(f"テストの過不足か重複がある（{len(got)} 件 / 出典 {len(want_ids)} 件）")
    if ng:
        errs.append(f"**原文または出典 JSON と一致しないものが {len(ng)}件**")
        errs += ng[:3]
    return errs, len(tests)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", "-d", default="corpus/bewt/json")
    ap.add_argument("--bewt", "-b", default="BEWT")
    ap.add_argument("--source", "-s")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.dir, "*.json")))
    if not files:
        sys.exit(f"{a.dir} に .json が無い")
    feats = load_features(a.bewt)
    if not feats:
        sys.exit(f"{a.bewt} の下に .feature が無い")
    src_map = None
    if a.source:
        src_map = load_source(a.source)
        if not src_map:
            sys.exit(f"{a.source} に出典 JSON が無い")

    ng = total = 0
    for p in files:
        errs, n = check_file(p, feats, src_map)
        total += n
        if errs:
            ng += 1
            print(f"[FAIL] {os.path.basename(p)} ({n}件)")
            for e in errs[:5]:
                print(f"        {e}")
        else:
            print(f"[ OK ] {os.path.basename(p)} ({n}件)")

    if src_map is not None:
        want_apps, seen = {a for a, _ in src_map}, {}
        for p in files:
            app, _, v = os.path.basename(p)[:-len(".json")].rpartition("_")
            if app != "ALL":
                seen.setdefault(v, set()).add(app)
        for v, apps in sorted(seen.items()):
            if apps != want_apps:
                ng += 1
                print(f"[FAIL] *_{v}.json のアプリの過不足（無い: {', '.join(sorted(want_apps - apps)) or '-'} / 余分: {', '.join(sorted(apps - want_apps)) or '-'}）")

    print("=" * 60)
    print(f"原文 {len(feats)} ファイル / 検査したファイル {len(files)} 件 / レコード合計 {total} 件 / 出典 JSON との照合 {'あり' if src_map is not None else 'なし'} / 不合格 {ng} 件")
    if ng:
        print("[FAIL] ベクトル化に渡すテキストが原文または出典 JSON と一致しない")
        sys.exit(1)
    print("[PASS] ベクトル化に渡すテキストは原文の行そのものだけ" + ("で，出典 JSON の並びと一致する" if src_map is not None else ""))


if __name__ == "__main__":
    main()
