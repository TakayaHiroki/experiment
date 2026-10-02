#!/usr/bin/env python3
"""RQ1 の集計値（BEWT）を，報告書に載せる形でまとめて算出する

節の構成は P5-1 の結果を見る前に決めた（2026-10-02）．
1 一致の大きさ（偶然一致の水準・計算精度だけが違う組と並べる．順位相関と Mantel 検定の一文）
2 11表現の一致の表  3 系統の対比（1つ）  4 FPF の曲線  5 前置きの効果  6 入力テキストの成分
付録 全ての組の値

usage:
  python tools/RQ1/rq1_summary.py -o results/rq1_summary.md
"""
import argparse, csv, importlib.util, itertools, json, os, sys
import numpy as np
from scipy import stats as _st

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "metrics"))
import _harness
# 測り直しには指標のスクリプトと同じ FPF の選び方を使う
_spec = importlib.util.spec_from_file_location(
    "fpf_top_run", os.path.join(os.path.dirname(os.path.abspath(__file__)), "metrics", "fpf_top", "run.py"))
_FPF = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_FPF)

LEX = {"tfidf", "lsa"}
BEWT8 = ["bludit", "claroline", "expresscart", "joomla",
         "kanboard", "mantisbt", "mediawiki", "prestashop"]
METRIC_DIR = "results/metrics"
METRICS = ["nn_agree", "fpf_top", "dist_rho"]
FPF_KS = (5, 10)
# 同じモデルの条件違いを系統の平均に混ぜない
CONDITION_MARKS = ("+sts", "@fp32")
# 入力テキストの比較で，最近傍の同点とみなす類似度の差の下限．
# 実際の幅は表現ごとに「同じテキストの2件のベクトルの距離の最大」（まとめて処理すると 1e-7 程度ずれる）とこの下限の大きい方
NEAR = 1e-6
# 入力テキストの比較は full と各成分の3組
VARIANT_PAIRS = [("full", "title"), ("full", "steps"), ("full", "expect")]
SIZE = {"sbert-all-mpnet-base-v2": 0.11, "bge-base-en-v1.5": 0.11, "e5-base-v2": 0.11,
        "qwen3-0.6b": 0.6, "gte-qwen2-1.5b-instruct": 1.5, "stella-en-1.5b-v5": 1.5,
        "qwen3-4b": 4.0, "gte-qwen2-7b-instruct": 7.0, "qwen3-8b": 8.0}
# 土台にした事前学習モデル（stella は gte-qwen2-1.5b を元に学習している）
FAMILY = {"sbert-all-mpnet-base-v2": "mpnet", "bge-base-en-v1.5": "bert",
          "e5-base-v2": "bert", "qwen3-0.6b": "qwen3", "qwen3-4b": "qwen3",
          "qwen3-8b": "qwen3", "gte-qwen2-1.5b-instruct": "qwen2",
          "gte-qwen2-7b-instruct": "qwen2", "stella-en-1.5b-v5": "qwen2"}
# 表の並び（語彙的手法のあと，土台のモデルごとに規模の順）
FAMILY_ORDER = {"mpnet": 0, "bert": 1, "qwen3": 2, "qwen2": 3}


def is_base(m):
    return not any(c in m for c in CONDITION_MARKS)


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


def near_max(S, width):
    return S >= S.max(1, keepdims=True) - width


def same_text_gap(X, texts):
    """同じテキストの2件のベクトル（長さ1）の距離の最大．同じテキストが無ければ 0．
    2件と第三のテストとの類似度の差は，この距離を超えない（|a·c − b·c| ≦ ‖a − b‖）"""
    gap = 0.0
    for t in set(texts):
        idx = [i for i, u in enumerate(texts) if u == t]
        for i, j in itertools.combinations(idx, 2):
            gap = max(gap, float(np.linalg.norm(X[i] - X[j])))
    return gap


def corpus_texts(corpus_dir, app, variant):
    with open(os.path.join(corpus_dir, f"{app}_{variant}.json"), encoding="utf-8") as f:
        return [r["text_for_embedding"] for r in json.load(f)]


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


def row_of(M, metric, x, y, k=""):
    """CSV の行を組の順によらず引く"""
    r = M[metric].get((x, y, k)) or M[metric].get((y, x, k))
    if r is None:
        raise SystemExit(f"{metric} の CSV に {x} ↔ {y}（k={k or '-'}）の行が無い")
    return r


def rep_order(m):
    """表の並び：語彙的手法（tfidf，lsa），土台のモデルの系統，規模，名前，前置き・精度の条件の順"""
    b = m.replace("+sts", "").replace("@fp32", "")
    cond = ("+sts" in m, "@fp32" in m)
    if b in LEX:
        return (0, 0 if b == "tfidf" else 1, 0.0, b, cond)
    return (1, FAMILY_ORDER.get(FAMILY.get(b, ""), 9), SIZE.get(b, 99.0), b, cond)


def ordered(p):
    """組を表の並びの順にそろえる"""
    return tuple(sorted(p, key=rep_order))


def sci(x):
    """1e-06 ではなく 1e-6 と書く"""
    m, e = f"{x:.0e}".split("e")
    return f"{m}e{int(e)}"


def trow(cells):
    """Markdown の表の1行"""
    return "| " + " | ".join(cells) + " |"


def f3(x):
    """小数第3位．丸めて 0 になる負の値は 0.000 と書く"""
    s = f"{x:.3f}"
    return "0.000" if s == "-0.000" else s


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

    pairs_all = sorted((ordered((x, y)) for (x, y, _) in M["nn_agree"]), key=lambda p: (rep_order(p[0]), rep_order(p[1])))
    reps = sorted({m for p in pairs_all for m in p}, key=rep_order)
    base = [m for m in reps if is_base(m)]
    base_pairs = list(itertools.combinations(base, 2))  # base は表の並びなので，組も表の並び
    if not base_pairs:
        raise SystemExit(f"前置きなし・配布された精度の表現が2つ未満（{' '.join(base) or 'なし'}）なので集計できない")
    lex_emb = [p for p in base_pairs if (p[0] in LEX) != (p[1] in LEX)]
    emb_emb = [p for p in base_pairs if p[0] not in LEX and p[1] not in LEX]
    lex_lex = [p for p in base_pairs if p[0] in LEX and p[1] in LEX]
    # 計算精度だけが違う組（bfloat16 ↔ float32）．表示は (配布の精度, float32) の順
    prec = sorted({(y, x) if x.endswith("@fp32") else (x, y) for x, y in pairs_all
                   if x == y + "@fp32" or y == x + "@fp32"}, key=lambda p: rep_order(p[0]))
    # 前置きだけが違う組（前置きなし ↔ +sts）．精度は同じもの同士
    pfx = sorted({(m.replace("+sts", ""), m) for m in reps if "+sts" in m and m.replace("+sts", "") in reps},
                 key=lambda p: (p[0].endswith("@fp32"), rep_order(p[0].replace("@fp32", ""))))
    cols = [("nn_agree", "最近傍", "")] + [("fpf_top", f"FPF@{k}", str(k)) for k in FPF_KS] + [("dist_rho", "順位相関", "")]

    def val(m, p, k=""):
        return cell(row_of(M, m, *p, k))

    # 全ての組について，埋め込みから測り直した最近傍の一致と FPF@5 が CSV と合うかを確かめる．
    # 合わなければ，CSV を作ったあとに埋め込みか指標のスクリプトが変わった
    E = {m: {app: load(emb, cdir, m, app, a.variant) for app in apps} for m in reps}
    den = np.array([len(E[reps[0]][app]) for app in apps])
    if int(den.sum()) != n_tests:
        raise SystemExit(f"埋め込みのテスト数（{int(den.sum())}）が CSV の n（{n_tests}）と違う")
    nns = {(m, app): nn_of(E[m][app])[0] for m in reps for app in apps}
    tops = {(m, app): set(_FPF.prepare(E[m][app])[:5]) for m in reps for app in apps}
    hits = {}
    for x, y in pairs_all:
        hits[(x, y)] = np.array([int((nns[(x, app)] == nns[(y, app)]).sum()) for app in apps])
        checks = [("最近傍の一致", "nn_agree", "", hits[(x, y)].sum() / den.sum()),
                  ("FPF@5", "fpf_top", "5", float(np.mean([len(tops[(x, app)] & tops[(y, app)]) / 5 for app in apps])))]
        for lab, m, k, w in checks:
            v = val(m, (x, y), k)
            if abs(v - w) > 1e-6:
                raise SystemExit(f"{x} ↔ {y} の{lab}が，CSV（{v:.6f}）と埋め込みから測り直した値（{w:.6f}）で違う．" + chr(10)
                                 + "  CSV を作ったあとに埋め込みか指標のスクリプトが変わった．指標を測り直すこと．")

    W("# RQ1 集計値（自動生成・BEWT）\n")
    W("**この文書は `python tools/RQ1/rq1_summary.py` の出力である．手で編集しない．**\n")
    W("報告書に載せる集計値はすべてここから引き写す．"
      "手計算を挟まないことで，集計範囲の違う数値が混ざる事故を防ぐ．"
      "節の構成は，一致の測定（P5-1）の結果を見る前に決めた（2026-10-02）．\n")
    W(f"入力は `{METRIC_DIR}/{{指標}}/{a.corpus_name}_{a.variant}.csv`（`nn_agree`・`fpf_top`・`dist_rho`）．"
      "3本の CSV が同じ表現の組・同じアプリで計算されたことを確かめてから集計している．\n")
    W(f"アプリ {len(apps)}件（{' '.join(apps)}）・テスト {n_tests}件．表現 {len(reps)}件・組 {len(pairs_all)}．"
      f"外した条件: {' '.join(excluded) if excluded else 'なし'}．\n")
    W(f"95%区間は，アプリを単位にした t 区間（自由度 {len(apps) - 1}）．区間はアプリの取り方の揺れだけを表し，比べた表現の顔ぶれは固定として扱う．"
      "偶然一致の水準は，片方の表現の結果についてテスト番号だけを並べ替えて測った値の平均．"
      "最近傍はテスト単位，FPF と順位相関はアプリ単位の平均（指標のスクリプトと同じ）．\n")

    # 1. 一致の大きさ
    W(f"\n## 1. 一致の大きさ（前置きなし・配布された精度の{len(base)}表現，{len(base_pairs)}組）\n")
    W("表現を変えたときの一致を，2つの目盛りと並べる．下の目盛りは偶然一致の水準，上の目盛りは同じモデル・同じ入力で計算精度だけを変えた組"
      "（表現の違いではなく，数値の誤差だけで生じる食い違いの目安）．"
      + ("" if prec else "計算精度だけが違う組は，その条件を外したので無い．") + "\n")
    W(trow(["指標", "偶然一致の水準", "最小", "中央値", "最大"] + [f"{x} ↔ @fp32" for x, _ in prec]))
    W(trow(["---"] + ["---:"] * (4 + len(prec))))
    for m, lab, k in cols:
        v = [val(m, p, k) for p in base_pairs]
        chance = np.mean([cell(row_of(M, m, *p, k), "chance_mean") for p in base_pairs])
        W(trow([lab, f3(chance), f3(min(v)), f3(np.median(v)), f3(max(v))] + [f3(val(m, p, k)) for p in prec]))
    if prec:
        W("\n計算精度だけが違う組の値と95%区間：\n")
        W("| 組 | " + " | ".join(c[1] for c in cols) + " |")
        W("|---|" + "---|" * len(cols))
        for x, y in prec:
            W(f"| {x} ↔ {y} | " + " | ".join(ci(row_of(M, m, x, y, k)) for m, _, k in cols) + " |")
    rho_rows = list(M["dist_rho"].values())
    perms = {int(r["perms"]) for r in rho_rows}
    n_apps = len(apps)
    if len(perms) == 1:
        perm = perms.pop()
        floor = 1.0 / (perm + 1)
        # 全アプリの p が下限のときの統合 p（CSV と同じ有効数字に丸めて比べる）
        p_min = float(f"{float(_st.chi2.sf(-2.0 * n_apps * np.log(floor), 2 * n_apps)):.3e}")
        at_floor = sum(float(r["p_fisher"]) <= p_min for r in rho_rows)
        W(f"\n**Mantel 検定：** 全 {len(rho_rows)}組のうち {at_floor}組で，{n_apps}アプリのどれでも，"
          f"{perm}回の並べ替えで観測値以上の順位相関が出なかった（アプリごとの p < {floor:.4f}）．"
          + ("" if at_floor == len(rho_rows) else
             f"残る組の統合 p の最大は {max(float(r['p_fisher']) for r in rho_rows):.3g}．")
          + "統合 p の桁数は並べ替えの回数で決まるので，根拠にしない．\n")
    else:
        W(f"\n**Mantel 検定：** 並べ替えの回数が組によって違う（{sorted(perms)}）．指標を測り直すこと．\n")

    # 2. 11表現の一致の表
    W(f"\n## 2. {len(base)}表現の一致の表（最近傍）\n")
    W("各組の最近傍の一致．表の並び（語彙的手法，土台のモデルの系統，規模）は順位ではない．"
      "区間の重なる組どうしの順位は述べない（区間は付録）．\n")
    W("| | " + " | ".join(str(i + 1) for i in range(len(base))) + " |")
    W("|---|" + "---:|" * len(base))
    for i, x in enumerate(base):
        W(f"| {i + 1} {x} | " + " | ".join("—" if x == y else f"{val('nn_agree', (x, y)):.3f}" for y in base) + " |")

    # 3. 系統の対比（主指標・前もって決めた1つ）
    W("\n## 3. 系統の対比（主指標 `nn_agree`）\n")
    if emb_emb and lex_emb:
        ee = np.mean([hits[p] for p in emb_emb], axis=0)
        le = np.mean([hits[p] for p in lex_emb], axis=0)
        d, lo, hi = _harness.t_interval(ee - le, den)
        per_app = (ee - le) / den
        W("前もって決めた対比は1つだけ：埋め込みモデルどうしの組と，語彙的手法と埋め込みモデルの組で，最近傍の一致の平均（テスト単位）が違うか．"
          "アプリごとの差に t 区間を付けた．\n")
        W("| 組み合わせ | 組の数 | 最近傍の一致 |")
        W("|---|---:|---:|")
        W(f"| 埋込 × 埋込 | {len(emb_emb)} | {ee.sum() / den.sum():.3f} |")
        W(f"| 語彙 × 埋込 | {len(lex_emb)} | {le.sum() / den.sum():.3f} |")
        W(f"\n差（埋込 × 埋込 − 語彙 × 埋込）: **{d:+.3f}**，95%区間 [{lo:+.3f}, {hi:+.3f}]"
          f"（{'0 を含む．差があるとは言えない' if lo <= 0 <= hi else '0 を含まない'}）．"
          f"差が正のアプリ {int((per_app > 0).sum())} / 負のアプリ {int((per_app < 0).sum())} / 0 のアプリ {int((per_app == 0).sum())}"
          f"（{len(apps)}アプリ中）．\n")
    else:
        W("埋め込みモデルどうしの組か，語彙的手法と埋め込みモデルの組が無い（条件を外したため）ので，この対比は出さない．\n")
    for x, y in lex_lex:
        why = "lsa は同じ TF-IDF の行列を20次元に縮めたものなので，" if {x, y} == {"tfidf", "lsa"} else ""
        W(f"語彙的手法どうし（{x} ↔ {y}）は1組だけで，{why}系統としてはまとめず個別に示す：{ci(row_of(M, 'nn_agree', x, y))}．\n")
    W("この結論は，ここで比べた表現について言えることで，埋め込みモデル一般には広げない．\n")

    # 4. FPF の曲線
    ks = sorted({int(k) for (_, _, k) in M["fpf_top"]})
    W(f"\n## 4. FPF で選ばれる上位k件の重なり（k=1〜{ks[-1]}）\n")
    W("選ばれるテストの重なり（アプリごとの割合の平均）．k を大きくすると表現と関係なく重なるので，偶然一致の水準と並べて読む．"
      "FPF は小さな違いでも選ぶ順が変わるので，計算精度だけが違う組の曲線も並べる．k によって入れ替わるので，組み合わせの順位は述べない．\n")
    W("| 組み合わせ | 組の数 | " + " | ".join(f"k={k}" for k in ks) + " |")
    W("|---|---:|" + "---:|" * len(ks))
    for lab, ps in [("埋込 × 埋込", emb_emb), ("語彙 × 埋込", lex_emb)] + [(f"{x} ↔ {y}", [(x, y)]) for x, y in lex_lex + prec]:
        if ps:
            W(f"| {lab} | {len(ps)} | " + " | ".join(f"{np.mean([val('fpf_top', p, str(k)) for p in ps]):.3f}" for k in ks) + " |")
    W("| （偶然一致の水準） | | " + " | ".join(
        f"{np.mean([cell(row_of(M, 'fpf_top', *p, str(k)), 'chance_mean') for p in base_pairs]):.3f}" for k in ks) + " |")

    # 5. 前置きの効果
    if pfx:
        W("\n## 5. 前置きの効果（同じモデルの前置きあり・なし）\n")
        W("同じモデル・同じ精度で，公式の類似度用の前置きを付けたかどうかだけが違う組．値の後ろの [ ] は95%区間．\n")
        W("| 組 | " + " | ".join(c[1] for c in cols) + " |")
        W("|---|" + "---|" * len(cols))
        for x, y in pfx:
            W(f"| {x} ↔ {y} | " + " | ".join(ci(row_of(M, m, x, y, k)) for m, _, k in cols) + " |")
        ref = [val("nn_agree", p) for p in emb_emb]
        W("\n読むときの目安（最近傍）：計算精度だけが違う組 "
          + (" / ".join(f"{val('nn_agree', p):.3f}" for p in prec) if prec else "なし")
          + (f"，別の埋め込みモデルどうし {len(ref)}組 {min(ref):.3f}〜{max(ref):.3f}（中央値 {np.median(ref):.3f}）" if ref else "")
          + "．前置きの組がどちらに近いかで，前置きの影響が計算の誤差の程度か，モデルを替えるのに近いかを読む．\n")

    # 6. 入力テキストの成分
    V = list(dict.fromkeys(v for p in VARIANT_PAIRS for v in p))
    vmods = [m for m in base
             if all(os.path.exists(os.path.join(emb, m, f"{app}_{v}.json")) for app in apps for v in V)]
    if vmods:
        W("\n## 6. 入力テキストの成分（full と各成分の最近傍の一致）\n")
        W("同じ表現のまま，埋め込みに渡すテキストを full からテスト名だけ（title）・手順だけ（steps）・確認だけ（expect）に変えて，"
          "最近傍が一致するテストの割合を測った．full の並びをどの成分が決めているかを見る．アプリごとに数えてから合計している．"
          "同じテキストのテストがあると最近傍が1つに決まらないので，最大との類似度の差が「同点の幅」以内の相手を同点とみなす．"
          "同点の幅は表現ごとに，同じテキストの2件のベクトルの距離の最大とした（2件と第三のテストとの類似度の差はこの距離を超えない）．"
          f"ただし {sci(NEAR)} を下限にする．"
          "同点からくじ引きで1つ選んだときの一致の期待値を出し，（ ）に同点の決め方しだいで動く最小〜最大を添えた．\n")
        W(trow(["表現"] + [f"{x}↔{y}" for x, y in VARIANT_PAIRS] + ["同点の幅"]))
        W(trow(["---"] * (len(VARIANT_PAIRS) + 1) + ["---:"]))
        ties = {v: 0 for v in V}
        for m in vmods:
            X = {(app, v): load(emb, cdir, m, app, v) for app in apps for v in V}
            gap = max(same_text_gap(X[(app, v)], corpus_texts(cdir, app, v)) for app in apps for v in V)
            # 類似度は小数第12位で丸めてあるので，その分（1e-12）を足す
            width = max(NEAR, gap + 1e-12)
            T = {key: near_max(nn_of(x)[1], width) for key, x in X.items()}
            for v in V:
                ties[v] = max(ties[v], sum(int((T[(app, v)].sum(1) > 1).sum()) for app in apps))
            cells = []
            for x, y in VARIANT_PAIRS:
                exp = lo = hi = tot = 0.0
                for app in apps:
                    tx, ty = T[(app, x)], T[(app, y)]
                    both = (tx & ty).sum(1)
                    exp += float((both / (tx.sum(1) * ty.sum(1))).sum())
                    hi += int((both > 0).sum())
                    lo += int(((tx.sum(1) == 1) & (ty.sum(1) == 1) & (both == 1)).sum())
                    tot += len(tx)
                cells.append(f"{exp / tot:.3f}（{lo / tot:.3f}〜{hi / tot:.3f}）")
            W(trow([m] + cells + [f"{width:.1e}".replace("e-0", "e-")]))
        W("\n同点のあるテストの数（表現の中で最大）: " + " / ".join(f"{v} {ties[v]}" for v in V) + "．"
          "prestashop の expect は33件中12種しかなく，同点の処理が値に効く（実験計画書 4.4）．\n")
        if len(vmods) < len(base):
            W(f"（4バリアントがそろっていない表現は外した: {' '.join(m for m in base if m not in vmods)}）\n")

    # 付録. 全ての組
    W(f"\n## 付録. 全 {len(pairs_all)}組の値\n")
    W("値と95%区間（[ ]）．順位相関の列の後ろは Mantel 検定の統合 p．CSV にはすべての k と偶然一致の水準も入っている．\n")
    W("| 組 | " + " | ".join(c[1] for c in cols) + " | 統合p |")
    W("|---|" + "---|" * len(cols) + "---:|")
    for x, y in pairs_all:
        W(f"| {x} ↔ {y} | " + " | ".join(ci(row_of(M, m, x, y, k)) for m, _, k in cols)
          + f" | {row_of(M, 'dist_rho', x, y)['p_fisher']} |")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as _f:
        _f.write("\n".join(L) + "\n")
    print(f"→ {a.out}")


if __name__ == "__main__":
    main()
