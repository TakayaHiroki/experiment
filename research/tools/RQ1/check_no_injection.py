#!/usr/bin/env python3
"""ベクトル化に渡すテキストの成分（title・steps・expects）が BEWT の原文の行そのものだけでできていて，出典 JSON の並びと一致することを検証する

usage: python tools/RQ1/check_no_injection.py [--dir corpus/bewt/json] [--bewt BEWT] [--source corpus/bewt/source]
"""
import argparse, glob, json, os, re, sys

VARIANTS = ["full", "title", "steps", "expect"]
KW_RE = re.compile(r"^(Given|When|And|Then)\s+(.*)$")
HEAD_RE = re.compile(r"^(Feature|Scenario)\s*:\s*(.*)$")
MARK_RE = re.compile(r"^the previous assertions? passed$")


def read_feature(path):
    title, lines = None, []
    with open(path, encoding="utf-8") as f:
        for n, ln in enumerate(f, 1):
            s = ln.strip()
            if not s:
                continue
            if s.startswith("#"):
                lines.append((n, s, "comment", s))
                continue
            m = HEAD_RE.match(s)
            if m:
                if m.group(1) == "Scenario" or title is None:
                    title = m.group(2).strip()
                lines.append((n, s, "heading", m.group(2).strip()))
                continue
            m = KW_RE.match(s)
            if m:
                text = m.group(2).strip()
                kind = "marker" if m.group(1) == "Given" and MARK_RE.match(text) else "step"
                lines.append((n, s, kind, text))
            else:
                lines.append((n, s, None, s))
    return title, lines


def load_features(bewt_dir):
    feats = {}
    for p in glob.glob(os.path.join(bewt_dir, "*", "gherkin", "**", "*.feature"), recursive=True):
        app = os.path.relpath(p, bewt_dir).replace(os.sep, "/").split("/")[0]
        title, lines = read_feature(p)
        body = {text for _, _, kind, text in lines if kind in ("step", "marker")}
        feats[(app, os.path.basename(p)[:-len(".feature")])] = (title, body, lines)
    return feats


def load_source(source_dir):
    src, dup = {}, []
    for p in glob.glob(os.path.join(source_dir, "*.json")):
        with open(p, encoding="utf-8") as f:
            meta = json.load(f)
        for t in meta["tests"]:
            key = tuple(t["id"].split("/", 1))
            if key in src:
                dup.append(t["id"])
            src[key] = t
    return src, dup


def check_source(src, feats):
    ng = [f"出典 JSON に無い .feature: {a}/{s}" for a, s in sorted(set(feats) - set(src))]
    ng += [f".feature が無い出典 JSON のテスト: {a}/{s}" for a, s in sorted(set(src) - set(feats))]
    for key in sorted(set(src) & set(feats)):
        t, (title, _, lines) = src[key], feats[key]
        name = "/".join(key)
        if t["title"] != title:
            ng.append(f"title が原文の見出しと一致しない: {name}")
        if [(r["line"], r["raw"]) for r in t["lines"]] != [(n, s) for n, s, _, _ in lines]:
            ng.append(f"lines が .feature の空でない行と一致しない（{len(t['lines'])} 行 / 原文 {len(lines)} 行）: {name}")
            continue
        for r, (n, _, kind, text) in zip(t["lines"], lines):
            role_ok = r["role"] in ("perform", "expect") if kind == "step" else r["role"] == kind
            if not role_ok or r["text"] != text:
                ng.append(f"{n} 行目の役割か本文が原文と合わない（role={r['role']}）: {name}")
    return ng


def check_file(path, feats, src):
    try:
        with open(path, encoding="utf-8") as f:
            tests = json.load(f)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        return [f"JSONとして読み込めない: {e}"], 0
    if not isinstance(tests, list) or not tests:
        return ["最上位が空でないリストでない"], 0

    file_app = os.path.basename(path)[:-len(".json")].rpartition("_")[0]
    want_ids = set(src) if file_app == "ALL" else {k for k in src if k[0] == file_app}
    errs, ng, got = [], [], []
    for i, t in enumerate(tests):
        try:
            title, app, spec = t["title"], t["app"], t["spec"]
            steps, expects = t["steps"], t["expects"]
        except (KeyError, TypeError) as e:
            errs.append(f"[{i}] 項目が足りない: {e}")
            continue
        if not (isinstance(title, str) and isinstance(app, str) and isinstance(spec, str)
                and isinstance(steps, list) and isinstance(expects, list)
                and all(isinstance(s, str) for s in steps + expects)):
            errs.append(f"[{i}] title・app・spec・steps・expects の型が違う")
            continue
        got.append((app, spec))
        if (app, spec) not in feats or (app, spec) not in src:
            ng.append(f"[{i}] .feature か出典 JSON に無い: {app}/{spec}")
            continue
        f_title, body, _ = feats[(app, spec)]
        lines = src[(app, spec)]["lines"]
        if title != f_title:
            ng.append(f"[{i}] title が原文の見出しと一致しない: 「{title[:60]}」（{app}/{spec}）")
        for s in steps + expects:
            if s not in body:
                ng.append(f"[{i}] 原文の行と一致しない: 「{s[:60]}」（{app}/{spec}）")
        for name, a, role in (("steps", steps, "perform"), ("expects", expects, "expect")):
            b = [r["text"] for r in lines if r["role"] == role]
            if a != b:
                ng.append(f"[{i}] {name} が出典 JSON の並びと一致しない（{len(a)} 件 / 出典 {len(b)} 件）（{app}/{spec}）")
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
    ap.add_argument("--source", "-s", default="corpus/bewt/source")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.dir, "*.json")))
    if not files:
        sys.exit(f"{a.dir} に .json が無い")
    feats = load_features(a.bewt)
    if not feats:
        sys.exit(f"{a.bewt} の下に .feature が無い")
    src, dup = load_source(a.source)
    if not src:
        sys.exit(f"{a.source} に出典 JSON が無い")

    ng = total = 0
    src_ng = check_source(src, feats) + [f"出典 JSON で id が重複: {d}" for d in dup]
    if src_ng:
        ng += 1
        print(f"[FAIL] 出典 JSON が原文と一致しないものが {len(src_ng)}件")
        for e in src_ng[:5]:
            print(f"        {e}")
    else:
        print(f"[ OK ] 出典 JSON ({len(src)}件)")

    want_files = {f"{app}_{v}.json" for app in {k[0] for k in src} | {"ALL"} for v in VARIANTS}
    have_files = {os.path.basename(p) for p in files}
    if have_files != want_files:
        ng += 1
        print(f"[FAIL] ファイルの過不足（無い: {', '.join(sorted(want_files - have_files)) or '-'} / "
              f"余分: {', '.join(sorted(have_files - want_files)) or '-'}）")

    for p in files:
        errs, n = check_file(p, feats, src)
        total += n
        if errs:
            ng += 1
            print(f"[FAIL] {os.path.basename(p)} ({n}件)")
            for e in errs[:5]:
                print(f"        {e}")
        else:
            print(f"[ OK ] {os.path.basename(p)} ({n}件)")

    print("=" * 60)
    print(f"原文 {len(feats)} ファイル / 出典 JSON {len(src)} テスト / 検査したファイル {len(files)} 件 / "
          f"レコード合計 {total} 件 / 不合格 {ng} 件")
    if ng:
        print("[FAIL] ベクトル化に渡すテキストの成分が原文または出典 JSON と一致しない")
        sys.exit(1)
    print("[PASS] ベクトル化に渡すテキストの成分は原文の行そのものだけで，出典 JSON の並びと一致する")


if __name__ == "__main__":
    main()
