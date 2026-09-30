#!/usr/bin/env python3
"""作成したベクトル（embeddings/bewt/{条件}/{app}_{variant}.json）を全ファイル検査する

usage: python tools/RQ1/check_embeddings_all.py [--emb-dir embeddings/bewt] [--corpus-dir corpus/bewt/json]
"""
import argparse, itertools, json, os, sys
import numpy as np

APPS = ["bludit", "claroline", "expresscart", "joomla", "kanboard", "mantisbt", "mediawiki", "prestashop"]
VARIANTS = ["full", "title", "steps", "expect"]
CONDITIONS = [
    "tfidf", "lsa", "lsa-full",
    "sbert-all-mpnet-base-v2", "bge-base-en-v1.5", "e5-base-v2",
    "qwen3-0.6b", "qwen3-0.6b+sts", "qwen3-0.6b@fp32", "qwen3-0.6b+sts@fp32",
    "gte-qwen2-1.5b-instruct", "gte-qwen2-1.5b-instruct+sts",
    "stella-en-1.5b-v5", "stella-en-1.5b-v5+sts",
    "qwen3-4b", "qwen3-4b+sts",
    "gte-qwen2-7b-instruct", "gte-qwen2-7b-instruct+sts",
    "qwen3-8b", "qwen3-8b+sts",
]
NORM_TOL = 0.01


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb-dir", default="embeddings/bewt")
    ap.add_argument("--corpus-dir", default="corpus/bewt/json")
    a = ap.parse_args()

    corpus_path = {(app, v): os.path.join(a.corpus_dir, f"{app}_{v}.json") for app in APPS for v in VARIANTS}
    corpus = {k: load(p) for k, p in corpus_path.items()}
    n_files = n_fail = 0
    examples = []

    print(f"{'条件':<28}{'ファイル':>6}{'不合格':>6}  {'次元':<8}{'|長さ-1|最大':>12}"
          f"{'異テキスト一致':>14}{'同テキスト差':>12}{'1-cos':>10}")
    for name in sorted(os.listdir(a.emb_dir)):
        if name not in CONDITIONS and os.path.isdir(os.path.join(a.emb_dir, name)):
            print(f"[FAIL] {name}: 想定外の条件のフォルダ（指標の計算に混ざる）")
            n_fail += 1
    for cond in CONDITIONS:
        if not os.path.isdir(os.path.join(a.emb_dir, cond)):
            print(f"[FAIL] {cond}: 条件のフォルダが無い（{len(APPS) * len(VARIANTS)} ファイル）")
            n_fail += len(APPS) * len(VARIANTS)
            continue
        files = fails = same_vec = 0
        dims, max_dev, max_diff, max_1cos = set(), 0.0, 0.0, 0.0
        for app in APPS:
            for v in VARIANTS:
                name = f"{cond}/{app}_{v}.json"
                path = os.path.join(a.emb_dir, cond, f"{app}_{v}.json")
                if not os.path.exists(path):
                    print(f"[FAIL] {name}: ファイルが無い")
                    fails += 1
                    continue
                files += 1
                E, C = load(path), corpus[(app, v)]
                X = np.array([r["embedding"] for r in E], dtype=np.float64)
                texts = [t["text_for_embedding"] for t in C]
                norms = np.linalg.norm(X, axis=1)
                errors = []
                if [r["title"] for r in E] != [t["title"] for t in C]:
                    errors.append("行の数または title が入力テキストと一致しない")
                if os.path.getmtime(path) < os.path.getmtime(corpus_path[(app, v)]):
                    errors.append("入力テキストより古い（コーパスを作り直した後のベクトルではない）")
                if not np.isfinite(X).all():
                    errors.append("NaN または無限大を含む")
                if np.abs(norms - 1).max() > NORM_TOL:
                    errors.append(f"|長さ-1| が {NORM_TOL} を超える")
                if errors:
                    print(f"[FAIL] {name}: {' / '.join(errors)}")
                    fails += 1
                    continue
                dims.add(X.shape[1])
                max_dev = max(max_dev, np.abs(norms - 1).max())
                Z = X / norms[:, None]
                for i, j in itertools.combinations(range(len(X)), 2):
                    if texts[i] == texts[j]:
                        max_diff = max(max_diff, np.abs(X[i] - X[j]).max())
                        max_1cos = max(max_1cos, 1 - Z[i] @ Z[j])
                    elif np.array_equal(X[i], X[j]):
                        same_vec += 1
                        examples.append(f"  {cond} {app}/{v} [{i}]「{texts[i]}」 [{j}]「{texts[j]}」")
        if cond not in ("tfidf", "lsa-full") and len(dims) > 1:
            print(f"[FAIL] {cond}: ファイルによって次元が違う {sorted(dims)}")
            fails += 1
        n_files += files
        n_fail += fails
        dim = "-" if not dims else f"{min(dims)}-{max(dims)}" if len(dims) > 1 else str(min(dims))
        print(f"{cond:<28}{files:>6}{fails:>6}  {dim:<8}{max_dev:>12.2e}"
              f"{same_vec:>14}{max_diff:>12.2e}{max_1cos:>10.2e}")

    print("テキストが異なるのにベクトルが完全に一致する組:")
    print("\n".join(examples) if examples else "  なし")

    print("入力テキストの中で，テキストが完全に一致するもの（アプリの中で数える）:")
    for v in VARIANTS:
        groups = []
        for app in APPS:
            texts = [t["text_for_embedding"] for t in corpus[(app, v)]]
            groups += [(app, t, texts.count(t)) for t in dict.fromkeys(texts) if texts.count(t) > 1]
        print(f"  {v:<7} 組 {len(groups):>2} / 組に属する行 {sum(n for _, _, n in groups):>2} / "
              f"2件目以降 {sum(n - 1 for _, _, n in groups):>2}")
        for app, t, n in groups:
            if n >= 5:
                print(f"          {app}: {n}件「{t}」")

    print(f"ファイル {n_files} 件 / 不合格 {n_fail} 件")
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
