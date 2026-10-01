#!/usr/bin/env python3
"""RQ1 の集計値（BEWT）を，報告書に載せる形でまとめて算出する

usage:
  python tools/RQ1/rq1_summary.py -o results/rq1_summary.md
"""
import argparse, csv, itertools, os, sys
import numpy as np
from scipy import stats as _st

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "metrics"))
import _harness

LEX = {"tfidf", "lsa"}
BEWT8 = ["bludit", "claroline", "expresscart", "joomla",
         "kanboard", "mantisbt", "mediawiki", "prestashop"]
METRIC_DIR = "results/metrics"
METRICS = ["nn_agree", "fpf_top", "dist_rho"]
FPF_KS = (5, 10)
# 同じモデルの条件違いを系統の平均に混ぜない
CONDITION_MARKS = ("+sts", "@fp32")
ORDER = {"語彙": 0, "埋込": 1}
# 最近傍の同点とみなす類似度の差．同じテキストでも処理の違いで 1e-7 程度ずれることがあるため
NEAR = 1e-6
SIZE = {"sbert-all-mpnet-base-v2": 0.11, "bge-base-en-v1.5": 0.11, "e5-base-v2": 0.11,
        "qwen3-0.6b": 0.6, "gte-qwen2-1.5b-instruct": 1.5, "stella-en-1.5b-v5": 1.5,
        "qwen3-4b": 4.0, "gte-qwen2-7b-instruct": 7.0, "qwen3-8b": 8.0}
# 土台にした事前学習モデル（stella は gte-qwen2-1.5b を元に学習している）
FAMILY = {"sbert-all-mpnet-base-v2": "mpnet", "bge-base-en-v1.5": "bert",
          "e5-base-v2": "bert", "qwen3-0.6b": "qwen3", "qwen3-4b": "qwen3",
          "qwen3-8b": "qwen3", "gte-qwen2-1.5b-instruct": "qwen2",
          "gte-qwen2-7b-instruct": "qwen2", "stella-en-1.5b-v5": "qwen2"}


def is_base(m):
    return not any(c in m for c in CONDITION_MARKS)


def grp(m):
    return "語彙" if m in LEX else "埋込"


_CACHE = {}


def load(emb_dir, corpus_dir, m, app, variant):
    key = (m, app, variant)
    if key not in _CACHE:
        # 指標のスクリプトと同じ読み込み（テストの並びもコーパスと照合する）
        X = _harness.load(emb_dir, corpus_dir, m, app, variant)
        if X is None:
            raise SystemExit(
                os.path.join(emb_dir, m, f"{app}_{variant}.json") + " が無い．" + chr(10)
                + "  指標の CSV を作ったときと埋め込みが違う．埋め込みをそろえてから指標を測り直すこと．")
        X.flags.writeable = False
        _CACHE[key] = X
    return _CACHE[key]


def nn_of(X):
    S = _harness.similarity(X)
    np.fill_diagonal(S, -np.inf)
    return S.argmax(1), S


def near_max(S):
    return S >= S.max(1, keepdims=True) - NEAR


def metric_path(metric, corpus, variant):
    return os.path.join(METRIC_DIR, metric, f"{corpus}_{variant}.csv")


def read_metric(metric, corpus, variant):
    path = metric_path(metric, corpus, variant)
    if not os.path.exists(path):
        raise SystemExit(
            path + " が無い．" + chr(10)
            + f"  python tools/RQ1/metrics/{metric}/run.py -e embeddings/{corpus} -c corpus/{corpus}/json "
            + f"--corpus-name {corpus} --apps " + " ".join(BEWT8) + chr(10)
            + "  を先に実行すること．")
    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    need = ("corpus", "variant", "k", "n", "n_apps", "apps", "excluded", "value", "value_lo", "value_hi",
            "chance_mean", "perms")
    miss = [c for c in need if rows and c not in rows[0]]
    if miss or not rows or "scaled" in rows[0]:
        raise SystemExit(
            path + " は古い版で作った CSV（" + ("必要な列が無い: " + " ".join(miss) if miss else "列の形が違う") + "）．" + chr(10)
            + "  指標のスクリプトを実行し直すこと．" + chr(10)
            + f"  python tools/RQ1/metrics/{metric}/run.py -e embeddings/{corpus} -c corpus/{corpus}/json "
            + f"--corpus-name {corpus} --apps " + " ".join(BEWT8))
    return {(r["model_a"], r["model_b"], r["k"]): r for r in rows}


def check_scope(M, corpus, variant):
    """3本の CSV が同じ表現の組・同じアプリで計算されたかを確かめ，アプリ・外した条件・テスト数を返す"""
    sig = {}
    for m in METRICS:
        scope = {(r["corpus"], r["variant"], r["apps"], r["excluded"], r["n"], r["n_apps"]) for r in M[m].values()}
        if len(scope) != 1:
            raise SystemExit(f"{m} の CSV の中で，行によって集計範囲が違う: {sorted(scope)}")
        sig[m] = (scope.pop(), frozenset((r["model_a"], r["model_b"]) for r in M[m].values()))
    ref = METRICS[0]
    for m in METRICS[1:]:
        if sig[m] == sig[ref]:
            continue
        lines = []
        if sig[m][0] != sig[ref][0]:
            lines.append(f"  集計範囲（コーパス, バリアント, アプリ, 外した条件, n, n_apps）: {ref} {sig[ref][0]} / {m} {sig[m][0]}")
        for a_, b_ in ((ref, m), (m, ref)):
            only = sorted(sig[a_][1] - sig[b_][1])
            if only:
                lines.append(f"  {a_} にだけある組 {len(only)}件: " + ", ".join(f"{x} ↔ {y}" for x, y in only[:4])
                             + (" …" if len(only) > 4 else ""))
        raise SystemExit(f"{ref} と {m} の CSV が違う範囲で計算されている．" + chr(10) + chr(10).join(lines) + chr(10)
                         + "  --exclude と --apps を3本でそろえて，指標を測り直すこと．")
    c, v, apps, excluded, n, _ = sig[ref][0]
    if (c, v) != (corpus, variant):
        raise SystemExit(f"CSV のコーパス・バリアント（{c}, {v}）が指定（{corpus}, {variant}）と違う")
    return apps.split(), excluded.split(), int(n)


def check_fresh(M, emb_dir, corpus, variant, apps):
    """CSV が，計算に使った埋め込みより後に作られていることを確かめる"""
    models = sorted({m for r in M["nn_agree"].values() for m in (r["model_a"], r["model_b"])})
    paths = [os.path.join(emb_dir, m, f"{app}_{variant}.json") for m in models for app in apps]
    lack = [p for p in paths if not os.path.exists(p)]
    if lack:
        raise SystemExit("指標の CSV にある表現の埋め込みが無い: " + " ".join(lack[:4]) + (" …" if len(lack) > 4 else ""))
    newest = max(os.path.getmtime(p) for p in paths)
    for m in METRICS:
        path = metric_path(m, corpus, variant)
        if os.path.getmtime(path) < newest:
            raise SystemExit(f"{path} が埋め込みより古い．埋め込みを作り直したあとなので，指標を測り直すこと．")


def cell(r, col="value"):
    v = r[col]
    return float(v) if v not in ("", None) else float("nan")


def ci(r):
    return f"{cell(r):.3f} [{cell(r, 'value_lo'):.3f}, {cell(r, 'value_hi'):.3f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", "-o", default="results/rq1_summary.md")
    ap.add_argument("--corpus-name", default="bewt")
    ap.add_argument("--variant", default="full")
    ap.add_argument("--embeddings", "-e", default=None, help="既定 embeddings/{corpus_name}")
    ap.add_argument("--corpus", "-c", default=None, help="既定 corpus/{corpus_name}/json")
    a = ap.parse_args()
    emb = a.embeddings or os.path.join("embeddings", a.corpus_name)
    cdir = a.corpus or os.path.join("corpus", a.corpus_name, "json")
    L = []
    W = L.append

    M = {m: read_metric(m, a.corpus_name, a.variant) for m in METRICS}
    apps, excluded, n_tests = check_scope(M, a.corpus_name, a.variant)
    check_fresh(M, emb, a.corpus_name, a.variant, apps)

    pairs_all = [(x, y) for (x, y, _) in M["nn_agree"]]
    reps = sorted({m for p in pairs_all for m in p})
    base = [m for m in reps if is_base(m)]
    cols = [("nn_agree", "最近傍", "")] + [("fpf_top", f"FPF@{k}", str(k)) for k in FPF_KS] + [("dist_rho", "RSA相関", "")]

    W("# RQ1 集計値（自動生成・BEWT）\n")
    W("**この文書は `python tools/RQ1/rq1_summary.py` の出力である．手で編集しない．**\n")
    W("報告書に載せる集計値はすべてここから引き写す．"
      "手計算を挟まないことで，集計範囲の違う数値が混ざる事故を防ぐ．\n")
    W(f"入力は `{METRIC_DIR}/{{指標}}/{a.corpus_name}_{a.variant}.csv`（`nn_agree`・`fpf_top`・`dist_rho`）．"
      "3本の CSV が同じ表現の組・同じアプリで計算されたことを確かめてから集計している．\n")
    W(f"アプリ {len(apps)}件（{' '.join(apps)}）・テスト {n_tests}件．"
      f"外した条件: {' '.join(excluded) if excluded else 'なし'}．\n")
    W(f"95%区間は，アプリを単位にした t 区間（自由度 {len(apps) - 1}）．"
      "偶然一致の水準は，片方の表現の結果についてテスト番号だけを並べ替えて測った値の平均．\n")

    # 1. 系統ごとの一致
    buckets = {}
    for x, y in itertools.combinations(base, 2):
        key = " × ".join(sorted([grp(x), grp(y)], key=lambda g: ORDER[g]))
        buckets.setdefault(key, []).append((x, y))
    order = sorted(buckets, key=lambda b: -np.mean([cell(M["nn_agree"][(*p, "")]) for p in buckets[b]]))
    W(f"\n## 1. 系統ごとの一致（前置きなし・配布された精度の{len(base)}表現）\n")
    W("| 組み合わせ | ペア数 | " + " | ".join(c[1] for c in cols) + " |")
    W("|---|---:|" + "---:|" * len(cols))
    base_pairs = list(itertools.combinations(base, 2))
    for key in order:
        ps = buckets[key]
        W(f"| {key} | {len(ps)} | " + " | ".join(f"{np.mean([cell(M[m][(*p, k)]) for p in ps]):.3f}" for m, _, k in cols) + " |")
    W("| （偶然一致の水準） | | " + " | ".join(
        f"{np.mean([cell(M[m][(*p, k)], 'chance_mean') for p in base_pairs]):.3f}" for m, _, k in cols) + " |")
    ll_pairs = buckets.get("語彙 × 語彙", [])
    if ll_pairs:
        W(f"\n`語彙 × 語彙` は {' / '.join(f'{x} ↔ {y}' for x, y in ll_pairs)} の{len(ll_pairs)}組だけで，"
          "lsa は同じ TF-IDF 行列を20次元に切り詰めたもの．系統の平均としては読まない．\n")

    # 2. 系統の差（主指標）
    E = {m: {app: load(emb, cdir, m, app, a.variant) for app in apps} for m in base}
    nns = {(m, app): nn_of(E[m][app])[0] for m in base for app in apps}
    hits = {(x, y): np.array([int((nns[(x, app)] == nns[(y, app)]).sum()) for app in apps])
            for x, y in itertools.combinations(base, 2)}
    den = np.array([len(E[base[0]][app]) for app in apps])
    if int(den.sum()) != n_tests:
        raise SystemExit(f"埋め込みのテスト数（{int(den.sum())}）が CSV の n（{n_tests}）と違う")
    # 測り直した値が CSV と一致しなければ，CSV と埋め込みのどちらかが古い
    for (x, y), h in hits.items():
        v, w = cell(M["nn_agree"][(x, y, "")]), h.sum() / den.sum()
        if abs(v - w) > 1e-6:
            raise SystemExit(f"{x} ↔ {y} の最近傍の一致が，CSV（{v:.6f}）と埋め込みから測り直した値（{w:.6f}）で違う．" + chr(10)
                             + "  CSV を作ったあとに埋め込みか指標のスクリプトが変わった．指標を測り直すこと．")
    group = {key: [p for p in hits if " × ".join(sorted([grp(p[0]), grp(p[1])], key=lambda g: ORDER[g])) == key]
             for key in buckets}
    W("\n## 2. 系統の差（主指標 `nn_agree`）\n")
    W("各系統の一致率（テスト単位）の差．アプリごとの差に t 区間を付けた．区間が0をまたがなければ差があると述べる．\n")
    W("| 対比 | 差 | 95%区間 | 0を含むか |")
    W("|---|---:|---|---|")
    for sa, sb in [("語彙 × 語彙", "埋込 × 埋込"), ("語彙 × 埋込", "埋込 × 埋込"), ("語彙 × 語彙", "語彙 × 埋込")]:
        if not group.get(sa) or not group.get(sb):
            continue
        # アプリごとの (分子, 分母)：系統ごとの一致数の平均の差と，テスト数
        num = np.mean([hits[p] for p in group[sa]], axis=0) - np.mean([hits[p] for p in group[sb]], axis=0)
        d, lo, hi = _harness.t_interval(num, den)
        W(f"| {sa} − {sb} | {d:+.3f} | [{lo:+.3f}, {hi:+.3f}] | "
          f"{'含む（差があるとは言えない）' if lo < 0 < hi else '**含まない**'} |")
    W("\n区間はアプリだけを取り直したもので，比べた表現の顔ぶれは固定として扱っている．\n")

    # 3. FPF の曲線
    ks = sorted({int(k) for (_, _, k) in M["fpf_top"]})
    W(f"\n## 3. FPF で選ばれる上位k件の重なり（k=1〜{ks[-1]}）\n")
    W("選ばれるテストの重なり（アプリごとの割合の平均）．k を大きくすると表現と関係なく重なるので，偶然一致の水準と並べて読む．\n")
    W("| 組み合わせ | " + " | ".join(f"k={k}" for k in ks) + " |")
    W("|---|" + "---:|" * len(ks))
    for key in order:
        W(f"| {key} | " + " | ".join(f"{np.mean([cell(M['fpf_top'][(*p, str(k))]) for p in buckets[key]]):.3f}" for k in ks) + " |")
    W("| （偶然一致の水準） | " + " | ".join(
        f"{np.mean([cell(M['fpf_top'][(*p, str(k))], 'chance_mean') for p in base_pairs]):.3f}" for k in ks) + " |")

    # 4. 条件の違い
    W("\n## 4. 条件の違い\n")
    prec = [(x, y) for x, y in pairs_all if y == x + "@fp32"]
    if prec:
        W("### 計算精度だけが違う組（bfloat16 ↔ float32）\n")
        W("同じモデル・同じ前置きで，計算精度だけを変えた組．表現の違いではなく，計算の誤差だけで生じる食い違いの目安になる．"
          "値の後ろの [ ] は95%区間．\n")
        W("| 組 | " + " | ".join(c[1] for c in cols) + " |")
        W("|---|" + "---|" * len(cols))
        for x, y in prec:
            W(f"| {x} ↔ {y} | " + " | ".join(ci(M[m][(x, y, k)]) for m, _, k in cols) + " |")
        W("")
    emb_m = sorted(m for m in base if m in SIZE)
    prs = list(itertools.combinations(emb_m, 2))
    if len(prs) >= 2:
        nn = {p: cell(M["nn_agree"][(*p, "")]) for p in prs}
        W("### 規模と系統（埋込どうし・最近傍の一致）\n")
        W("| 組 | ペア数 | 平均 |")
        W("|---|---:|---:|")
        for lab, sel in [("同じ規模どうし", lambda p: SIZE[p[0]] == SIZE[p[1]]),
                         ("規模が違う組", lambda p: SIZE[p[0]] != SIZE[p[1]]),
                         ("同じ系統どうし（土台のモデルが同じ）", lambda p: FAMILY[p[0]] == FAMILY[p[1]]),
                         ("別の系統どうし", lambda p: FAMILY[p[0]] != FAMILY[p[1]])]:
            v = [nn[p] for p in prs if sel(p)]
            if v:
                W(f"| {lab} | {len(v)} | {np.mean(v):.3f} |")
        xs = [abs(np.log10(SIZE[x]) - np.log10(SIZE[y])) for x, y in prs]
        if len(set(xs)) > 1:
            rho, pv = _st.spearmanr(xs, [nn[p] for p in prs])
            W(f"\n規模差（対数）と一致の順位相関: ρ = {rho:+.3f}，p = {pv:.3f}．"
              "組どうしは独立でないので参考値．\n")
    W("前置きの有無の比較は，比べ方（「位置」の定義）が決まってから足す．\n")

    # 5. 入力テキストの違い
    V = ["full", "title", "steps", "expect"]
    VPAIRS = list(itertools.combinations(V, 2))
    vmods = [m for m in base
             if all(os.path.exists(os.path.join(emb, m, f"{app}_{v}.json")) for app in apps for v in V)]
    if vmods:
        W("\n## 5. 入力テキストの切り出し方を変えると（最近傍の一致）\n")
        W("同じ表現のまま，埋め込みに渡すテキストだけを変えて，最近傍が一致するテストの割合を測った．アプリごとに計算してから合計している．"
          f"同じテキストのテストがあると最近傍が1つに決まらない（類似度の差が {NEAR:g} 以下を同点とみなす）．"
          "同点からくじ引きで1つ選んだときの一致の期待値を出し，（ ）に同点の決め方しだいで動く最小〜最大を添えた．\n")
        W("| 表現 | " + " | ".join(f"{x}↔{y}" for x, y in VPAIRS) + " |")
        W("|---" * (len(VPAIRS) + 1) + "|")
        worst = 1.0
        ties = {v: 0 for v in V}
        for m in vmods:
            T = {(app, v): near_max(nn_of(load(emb, cdir, m, app, v))[1]) for app in apps for v in V}
            for v in V:
                ties[v] = max(ties[v], sum(int((T[(app, v)].sum(1) > 1).sum()) for app in apps))
            cells = []
            for x, y in VPAIRS:
                exp = lo = hi = tot = 0.0
                for app in apps:
                    tx, ty = T[(app, x)], T[(app, y)]
                    both = (tx & ty).sum(1)
                    exp += float((both / (tx.sum(1) * ty.sum(1))).sum())
                    hi += int((both > 0).sum())
                    lo += int(((tx.sum(1) == 1) & (ty.sum(1) == 1) & (both == 1)).sum())
                    tot += len(tx)
                worst = min(worst, exp / tot)
                cells.append(f"{exp / tot:.3f}（{lo / tot:.3f}〜{hi / tot:.3f}）")
            W(f"| {m} | " + " | ".join(cells) + " |")
        W(f"\n最も低いところで **{worst:.3f}**．"
          "同点のあるテストの数（表現の中で最大）: " + " / ".join(f"{v} {ties[v]}" for v in V) + "．\n")
        if len(vmods) < len(base):
            W(f"（4バリアントがそろっていない表現は外した: {' '.join(m for m in base if m not in vmods)}）\n")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as _f:
        _f.write("\n".join(L) + "\n")
    print(f"→ {a.out}")


if __name__ == "__main__":
    main()
