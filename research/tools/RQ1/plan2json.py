#!/usr/bin/env python3
"""テストプラン(Markdown) → 埋め込み用JSON への変換

usage:
  python plan2json.py --input <plan.md> --outdir <出力先> [--app <アプリ名>]
  python plan2json.py --input-dir <plans/> --outdir <出力先>
"""
import argparse, json, os, re, sys, glob


def orig_index(c):
    m = re.match(r"^(\d+)_", c.get("spec", "") or "")
    return int(m.group(1)) if m else 10 ** 6


def parse_plan(md_text):
    cases = []
    blocks = re.split(r"\n#### ", "\n" + md_text)[1:]
    for b in blocks:
        lines = b.split("\n")
        title = lines[0].strip()
        title = re.sub(r"^\d+(?:\.\d+)*\.?\s+", "", title, count=1)

        spec = ""
        m = re.search(r"\*\*File:\*\*\s*`([^`]+)`", b)
        if m:
            spec = os.path.basename(m.group(1)).replace(".spec.ts", "")

        steps, expects = [], []
        for ln in lines[1:]:
            s = ln.strip()
            if not s:
                continue
            if s.startswith("**") or s.startswith("###"):
                continue
            m = re.match(r"^\d+\.\s+(.*)$", s)
            if m:
                if m.group(1).strip() != "-":
                    steps.append(m.group(1).strip())
                continue
            m = re.match(r"^-\s*expect:\s*(.*)$", s)
            if m:
                expects.append(m.group(1).strip())
                continue
        cases.append(dict(title=title, spec=spec, steps=steps, expects=expects))
    return cases


VARIANTS = ["title", "steps", "expect", "full"]

SEP = " "


def build_text(c, variant):
    if variant == "title":
        parts = [c["title"]]
    elif variant == "steps":
        parts = c["steps"]
    elif variant == "expect":
        parts = c["expects"]
    elif variant == "full":
        parts = [c["title"]] + c["steps"] + c["expects"]
    else:
        raise ValueError(f"未知のバリアント: {variant}")
    if not [p for p in parts if p and p.strip()]:
        raise ValueError(
            "バリアント {} の材料が空: {}".format(variant, c.get("title", "?")[:40])
            + " **黙って別の成分で代替しない．**プランを確認すること")
    return SEP.join(p.strip() for p in parts if p and p.strip())


def build(md_path, app=None):
    app = app or os.path.basename(md_path).split("-test-plan")[0]
    with open(md_path, encoding="utf-8") as f:
        text = f.read()
    cases = parse_plan(text)
    if not cases:
        print(f"  [警告] {md_path} からテストケースを抽出できない")
        return None
    by_variant = {}
    for v in VARIANTS:
        recs = []
        for i, c in enumerate(cases):
            recs.append({
                "index": i,
                "title": c["title"],
                "text_for_embedding": build_text(c, v),
                "app": app,
                "spec": c["spec"],
                "orig_index": orig_index(c),
                "n_steps": len(c["steps"]),
                "n_expects": len(c["expects"]),
                "steps": c["steps"],
                "expects": c["expects"],
                "variant": v,
            })
        by_variant[v] = recs
    return app, len(cases), by_variant


def write(outdir, built):
    total = 0
    os.makedirs(outdir, exist_ok=True)
    for app, n, by_variant in built:
        for v, recs in by_variant.items():
            with open(os.path.join(outdir, f"{app}_{v}.json"), "w", encoding="utf-8") as f:
                json.dump(recs, f, ensure_ascii=False, indent=2)
        print(f"  {app}: {n}件 → {len(VARIANTS)}バリアント出力")
        total += n
    return total


def combine(outdir):
    n = 0
    for v in VARIANTS:
        recs = []
        for p in sorted(glob.glob(os.path.join(outdir, f"*_{v}.json"))):
            if os.path.basename(p).startswith("ALL_"):
                continue
            with open(p, encoding="utf-8") as f:
                for r in json.load(f):
                    r = dict(r)
                    r["index"] = len(recs)
                    recs.append(r)
        with open(os.path.join(outdir, f"ALL_{v}.json"), "w", encoding="utf-8") as f:
            json.dump(recs, f, ensure_ascii=False, indent=2)
        n = len(recs)
    print(f"  ALL_*.json を作成（全アプリ結合，{n}件）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", "-i")
    ap.add_argument("--input-dir")
    ap.add_argument("--outdir", "-o", required=True)
    ap.add_argument("--app")
    ap.add_argument("--no-combine", action="store_true",
                    help="全アプリ結合ファイル ALL_*.json を作らない")
    a = ap.parse_args()

    if a.input_dir:
        files = sorted(glob.glob(os.path.join(a.input_dir, "*.md")))
        if not files:
            sys.exit(f"{a.input_dir} に .md が無い")
        built = [b for b in (build(f) for f in files) if b]
        total = write(a.outdir, built)
        if not a.no_combine:
            combine(a.outdir)
    elif a.input:
        b = build(a.input, a.app)
        total = write(a.outdir, [b] if b else [])
    else:
        sys.exit("--input か --input-dir のどちらかを指定すること")
    print(f"合計 {total} 件 → {a.outdir}")


if __name__ == "__main__":
    main()
