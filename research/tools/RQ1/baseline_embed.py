#!/usr/bin/env python3
"""GPU不要の語彙的手法でベクトル化する

usage:
  python baseline_embed.py --input corpus/bewt/json/kanboard_full.json \
      --output embeddings/bewt/tfidf/kanboard_full.json --method tfidf
"""
import argparse, json, os, sys, time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _measure


def load(path):
    if not os.path.isfile(path):
        raise SystemExit(f"入力が無い: {path}")
    try:
        with open(path, encoding="utf-8") as f:
            tests = json.load(f)
        titles = [t["title"] for t in tests]
        texts = [t["text_for_embedding"] for t in tests]
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError, KeyError) as e:
        raise SystemExit(f"入力の形式が違う: {path}（{e!r}）")
    if not texts:
        raise SystemExit(f"テストが0件: {path}")
    return titles, texts


def emb_tfidf(texts, min_df):
    v = TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=min_df)
    return v.fit_transform(texts).toarray()


def emb_lsa(texts, dim):
    X = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit_transform(texts)
    full = dim is None
    # 取れる次元の上限．語彙数-1 とテスト数-1 の小さい方
    cap = min(X.shape[1] - 1, len(texts) - 1)
    req = cap if full else dim
    dim = min(req, cap)
    if dim < req:
        print(f"[WARN] lsa: 指定 {req} 次元は不可能．{dim} 次元に切り下げた "
              f"(テスト数 {len(texts)} / 語彙数 {X.shape[1]})")
    if not full and dim >= len(texts) - 1:
        print(f"[WARN] lsa: 次元 {dim} = テスト数-1 のため圧縮になっていない．"
              f"tfidf のほぼ回転であり独立した表現ではない")
    if dim < 2:
        return X.toarray()
    # 乱数を固定して再現させる
    Z = TruncatedSVD(dim, random_state=0).fit_transform(X)
    return Normalizer().fit_transform(Z)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", "-i", required=True)
    ap.add_argument("--output", "-o", required=True)
    ap.add_argument("--method", "-m", required=True, choices=["tfidf", "lsa", "lsa-full"])
    ap.add_argument("--min-df", type=int, default=1)
    ap.add_argument("--dim", type=int, default=20)
    a = ap.parse_args()
    if a.min_df < 1 or a.dim < 1:
        raise SystemExit("--min-df と --dim は1以上にすること")

    titles, texts = load(a.input)
    print(f"Loaded {len(texts)} test cases. Encoding with {a.method}...")

    c0, t0 = time.process_time(), time.perf_counter()
    t_load, c_load = time.perf_counter() - t0, time.process_time() - c0

    c1, t1 = time.process_time(), time.perf_counter()
    if a.method == "tfidf":
        E = emb_tfidf(texts, a.min_df)
    else:
        E = emb_lsa(texts, a.dim if a.method == "lsa" else None)
    t_encode, c_encode = time.perf_counter() - t1, time.process_time() - c1

    results = [{"index": i, "title": titles[i], "embedding": E[i].tolist()}
               for i in range(len(titles))]
    os.makedirs(os.path.dirname(a.output) or ".", exist_ok=True)
    tmp = a.output + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)
    # 途中で止まっても壊れたファイルを残さない
    os.replace(tmp, a.output)

    print(_measure.timing(
        t_load=f"{t_load:.2f}", t_encode=f"{t_encode:.2f}",
        c_load=f"{c_load:.2f}", c_encode=f"{c_encode:.2f}",
        n=len(texts), device="cpu", params=0,
        peak_rss_mb=_measure.peak_rss_mb(), peak_gpu_mb=0))
    print(f"\nEmbeddings saved to {a.output} ({len(results)} entries, dim={E.shape[1]})")


if __name__ == "__main__":
    main()
