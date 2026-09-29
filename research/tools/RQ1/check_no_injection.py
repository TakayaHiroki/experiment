#!/usr/bin/env python3
"""検証用 JSON の役割が変換規則どおりで，md とベクトル化に渡すテキストの成分（title・steps・expects）が BEWT の原文の行そのものだけでできていることを検証する

usage: python tools/RQ1/check_no_injection.py [--dir corpus/bewt/json] [--bewt BEWT] [--source corpus/bewt/source] [--plans corpus/bewt/plans]
"""
import argparse, glob, json, os, re, sys

VARIANTS = ["full", "title", "steps", "expect"]
KW_RE = re.compile(r"^(Given|When|And|Then)\s+(.*)$")
HEAD_RE = re.compile(r"^(Feature|Scenario)\s*:\s*(.*)$")
MARK_RE = re.compile(r"^the previous assertions? passed$")


def expected_roles(kws):
    roles = []
    for i, (k, mark) in enumerate(kws):
        if mark:
            roles.append("marker")
            continue
        g = next((j for j in range(i - 1, -1, -1) if kws[j][0] == "Given"), None)
        if k in ("Then", "And") and g is not None and kws[g][1]:
            end = next((j for j in range(g + 1, len(kws)) if kws[j][0] == "Given"), len(kws))
            if all(kws[j][0] != "When" for j in range(g + 1, end)):
                roles.append("perform")
                continue
        base = next((kws[j][0] for j in range(i, -1, -1) if kws[j][0] != "And"), None)
        roles.append(None if base is None else "expect" if base == "Then" else "perform")
    return roles


def read_feature(path):
    title, lines, kws = None, [], []
    with open(path, encoding="utf-8", newline="") as f:
        raw = re.split(r"\r\n|\n", f.read())
    for n, ln in enumerate(raw, 1):
        s = ln.strip()
        if not s:
            continue
        if s.startswith("#"):
            lines.append((n, s, None, "comment", s))
            continue
        m = HEAD_RE.match(s)
        if m:
            if m.group(1) == "Scenario" or title is None:
                title = m.group(2).strip()
            lines.append((n, s, m.group(1), "heading", m.group(2).strip()))
            continue
        m = KW_RE.match(s)
        if m:
            text = m.group(2).strip()
            kws.append((m.group(1), m.group(1) == "Given" and bool(MARK_RE.match(text))))
            lines.append((n, s, m.group(1), len(kws) - 1, text))
        else:
            lines.append((n, s, None, None, s))
    roles = expected_roles(kws)
    lines = [(n, s, k, roles[r] if isinstance(r, int) else r, t) for n, s, k, r, t in lines]
    return title, lines


def load_features(bewt_dir):
    feats = {}
    for p in glob.glob(os.path.join(bewt_dir, "*", "gherkin", "**", "*.feature"), recursive=True):
        app = os.path.relpath(p, bewt_dir).replace(os.sep, "/").split("/")[0]
        title, lines = read_feature(p)
        body = {text for _, _, _, role, text in lines if role in ("perform", "expect", "marker")}
        feats[(app, os.path.basename(p)[:-len(".feature")])] = (title, body, lines)
    return feats


def load_source(source_dir):
    src, dup, apps = {}, [], {}
    for p in glob.glob(os.path.join(source_dir, "*.json")):
        with open(p, encoding="utf-8") as f:
            meta = json.load(f)
        apps[os.path.basename(p)[:-len(".json")]] = meta
        for t in meta["tests"]:
            key = tuple(t["id"].split("/", 1))
            if key in src:
                dup.append(t["id"])
            src[key] = t
    return src, dup, apps


def order_key(spec):
    m = re.match(r"^(\d+)_", spec)
    return (int(m.group(1)) if m else 10 ** 6, spec + ".feature")


def check_source(src, feats, apps):
    ng = [f"検証用 JSON に無い .feature: {a}/{s}" for a, s in sorted(set(feats) - set(src))]
    ng += [f".feature が無い検証用 JSON のテスト: {a}/{s}" for a, s in sorted(set(src) - set(feats))]
    for name, meta in sorted(apps.items()):
        specs = [t["id"].split("/", 1)[1] for t in meta["tests"]]
        if meta["app"] != name or any(not t["id"].startswith(name + "/") for t in meta["tests"]):
            ng.append(f"ファイル名と app・id のアプリが一致しない: {name}.json")
        if specs != sorted(specs, key=order_key):
            ng.append(f"テストの並びがファイル名の番号順でない: {name}")
    for key in sorted(set(src) & set(feats)):
        t, (title, _, lines) = src[key], feats[key]
        name = "/".join(key)
        if t["title"] != title:
            ng.append(f"title が原文の見出しと一致しない: {name}")
        if [(r["line"], r["raw"]) for r in t["lines"]] != [(n, s) for n, s, _, _, _ in lines]:
            ng.append(f"lines が .feature の空でない行と一致しない（{len(t['lines'])} 行 / 原文 {len(lines)} 行）: {name}")
            continue
        for r, (n, _, kw, role, text) in zip(t["lines"], lines):
            if role is None or (r["role"], r["keyword"], r["text"]) != (role, kw, text):
                ng.append(f"{n} 行目の役割・キーワード・本文が規則と合わない（role={r['role']} / 規則 {role}）: {name}")
    return ng


def render_md(app, tests):
    out = [f"# BEWT {app}", "", "## Application Overview", "", "", "", "## Test Scenarios",
           "", f"### 1. {app}", "", "**Seed:** ``"]
    ng = []
    for j, t in enumerate(tests, 1):
        out += ["", f"#### 1.{j}. {t['title']}", "", f"**File:** `tests/{t['id']}.spec.ts`", "", "**Steps:**"]
        k, open_step = 0, False
        for r in t["lines"]:
            if r["role"] == "perform":
                k, open_step = k + 1, True
                out.append(f"  {k}. {r['text']}")
            elif r["role"] == "marker":
                open_step = False
            elif r["role"] == "expect":
                if not open_step:
                    ng.append(f"手順の無い expect: {t['id']}:{r['line']}")
                out.append(f"    - expect: {r['text']}")
    return "\n".join(out + [""]), ng


def check_plans(plans_dir, apps):
    want = {f"{a}-test-plan.md" for a in apps}
    have = {os.path.basename(p) for p in glob.glob(os.path.join(plans_dir, "*.md"))}
    ng = []
    if have != want:
        ng.append(f"md の過不足（無い: {', '.join(sorted(want - have)) or '-'} / 余分: {', '.join(sorted(have - want)) or '-'}）")
    for app in sorted(apps):
        if f"{app}-test-plan.md" not in have:
            continue
        md, errs = render_md(app, apps[app]["tests"])
        ng += errs
        with open(os.path.join(plans_dir, f"{app}-test-plan.md"), "rb") as f:
            got = f.read()
        if got != md.encode("utf-8"):
            a, b = got.decode("utf-8", "replace").split("\n"), md.split("\n")
            i = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
            ng.append(f"md が検証用 JSON から描いたものと一致しない: {app}-test-plan.md {i + 1} 行目")
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
            ng.append(f"[{i}] .feature か検証用 JSON に無い: {app}/{spec}")
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
                ng.append(f"[{i}] {name} が検証用 JSON の並びと一致しない（{len(a)} 件 / 検証用 JSON {len(b)} 件）（{app}/{spec}）")
    if len(got) != len(set(got)) or set(got) != want_ids:
        ng.append(f"テストの過不足か重複がある（{len(got)} 件 / 検証用 JSON {len(want_ids)} 件）")
    if ng:
        errs.append(f"**原文または検証用 JSON と一致しないものが {len(ng)}件**")
        errs += ng[:3]
    return errs, len(tests)


def report(label, errs):
    if errs:
        print(f"[FAIL] {label}: 不一致 {len(errs)}件")
        for e in errs[:5]:
            print(f"        {e}")
        return 1
    print(f"[ OK ] {label}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", "-d", default="corpus/bewt/json")
    ap.add_argument("--bewt", "-b", default="BEWT")
    ap.add_argument("--source", "-s", default="corpus/bewt/source")
    ap.add_argument("--plans", "-p", default="corpus/bewt/plans")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.dir, "*.json")))
    if not files:
        sys.exit(f"{a.dir} に .json が無い")
    feats = load_features(a.bewt)
    if not feats:
        sys.exit(f"{a.bewt} の下に .feature が無い")
    src, dup, apps = load_source(a.source)
    if not src:
        sys.exit(f"{a.source} に検証用 JSON が無い")

    ng = total = 0
    ng += report(f"検証用 JSON ({len(src)}件)", check_source(src, feats, apps) + [f"検証用 JSON で id が重複: {d}" for d in dup])
    ng += report(f"md ({len(apps)}件)", check_plans(a.plans, apps))

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
    print(f"原文 {len(feats)} ファイル / 検証用 JSON {len(src)} テスト / md {len(apps)} 件 / 検査したファイル {len(files)} 件 / "
          f"レコード合計 {total} 件 / 不合格 {ng} 件")
    if ng:
        print("[FAIL] 検証用 JSON，md，ベクトル化に渡すテキストの成分のどれかが原文と変換規則に一致しない")
        sys.exit(1)
    print("[PASS] 検証用 JSON の役割は変換規則どおりで，md とベクトル化に渡すテキストの成分は原文の行そのものだけ")


if __name__ == "__main__":
    main()
