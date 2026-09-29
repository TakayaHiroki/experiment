#!/usr/bin/env python3
"""RQ1 の集計値（BEWT）を，報告書に載せる形でまとめて算出する

usage:
  python tools/RQ1/rq1_summary.py -o results/rq1_summary.md
"""
import argparse, csv, itertools, json, os
import numpy as np
from scipy import stats as _st

LEX = {"tfidf", "lsa"}
BEWT8 = ["bludit", "claroline", "expresscart", "joomla",
         "kanboard", "mantisbt", "mediawiki", "prestashop"]
METRIC_DIR = "results/metrics"
METRICS = ["triplet_agree", "dist_rho", "knn_jaccard", "nn_agree", "procrustes_disp", "fpf_top"]
FPF_KS = (5, 10, 20)
KNN_K = 5
KNN_SWEEP = (1, 3, 5, 10, 20)
# 同じモデルの条件違いを系統の平均に混ぜない
CONDITION_MARKS = ("+sts", "@fp32")
ORDER = {"語彙": 0, "埋込": 1}
BOOT = 10000
SEED = 0


def is_base(m):
    return not any(c in m for c in CONDITION_MARKS)


def grp(m):
    return "語彙" if m in LEX else "埋込"


_CACHE = {}


def load(path):
    if path in _CACHE:
        return _CACHE[path]
    if not os.path.exists(path):
        raise SystemExit(
            path + " が無い．" + chr(10)
            + "  その表現を run_embeddings.ps1 で作り直すか，"
            + "使わないなら models の一覧から外すこと．" + chr(10)
            + "  **一部のアプリだけ作ると，そろわないアプリが集計から落ちる．**")
    with open(path, encoding="utf-8") as _f:
        X = np.array([r["embedding"] for r in json.load(_f)], float)
    Z = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    Z.flags.writeable = False
    _CACHE[path] = Z
    return Z


def nn_of(X):
    S = X @ X.T
    np.fill_diagonal(S, -np.inf)
    return S.argmax(1), S


def fpf(X, k, start=None):
    D = 1.0 - X @ X.T
    n = len(X)
    s = int(np.argmax(D.sum(1))) if start is None else start
    sel = [s]
    mind = D[s].copy()
    while len(sel) < min(k, n):
        mind[sel] = -np.inf
        nxt = int(np.argmax(mind))
        sel.append(nxt)
        mind = np.minimum(mind, D[nxt])
    return sel


def read_metric(metric, corpus, variant):
    path = os.path.join(METRIC_DIR, metric, f"{corpus}_{variant}.csv")
    if not os.path.exists(path):
        raise SystemExit(
            path + " が無い．" + chr(10)
            + f"  python tools/RQ1/metrics/{metric}/run.py -e embeddings/{corpus} -c corpus/{corpus}/json "
            + f"--corpus-name {corpus} --apps " + " ".join(BEWT8) + chr(10)
            + "  を先に実行すること．")
    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    need = ("kind", "k", "value", "value_lo", "value_hi", "chance_mean", "positive", "scaled",
            "scaled_lo", "scaled_hi", "perms", "boot")
    miss = [c for c in need if rows and c not in rows[0]]
    if miss:
        raise SystemExit(
            path + " に必要な列が無い: " + " ".join(miss) + chr(10)
            + "  区間を出す前の版で作った CSV なので，指標のスクリプトを実行し直すこと．" + chr(10)
            + f"  python tools/RQ1/metrics/{metric}/run.py -e embeddings/{corpus} -c corpus/{corpus}/json "
            + f"--corpus-name {corpus} --apps " + " ".join(BEWT8))
    out = {}
    for r in rows:
        out[(r["model_a"], r["model_b"], r["k"])] = r
    return out


def cell(r, col="value"):
    v = r[col]
    return float(v) if v not in ("", None) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", "-o", default="results/rq1_summary.md")
    ap.add_argument("--corpus-name", default="bewt")
    ap.add_argument("--variant", default="full")
    a = ap.parse_args()
    rng = np.random.default_rng(SEED)
    L = []
    W = L.append

    M = {m: read_metric(m, a.corpus_name, a.variant) for m in METRICS}

    pairs_all = [(x, y) for (x, y, k) in M["nn_agree"] if M["nn_agree"][(x, y, k)]["kind"] == "actual"]
    reps = sorted({m for p in pairs_all for m in p})
    base = [m for m in reps if is_base(m)]
    n_rep = len(base)

    W("# RQ1 集計値（自動生成・BEWT）\n")
    W("**この文書は `python tools/RQ1/rq1_summary.py` の出力である．手で編集しない．**\n")
    W("報告書に載せる集計値はすべてここから引き写す．"
      "手計算を挟まないことで，集計範囲の違う数値が混ざる事故を防ぐ．\n")
    W("**BEWT コーパスだけの集計である．**"
      f"入力は `{METRIC_DIR}/{{指標}}/{a.corpus_name}_{a.variant}.csv`．\n")

    n_apps = M["nn_agree"][(*pairs_all[0], "")]["n_apps"]
    n_tests = M["nn_agree"][(*pairs_all[0], "")]["n"]
    W(f"\n## 系統ごとの一致 — BEWT（{n_apps}アプリ{n_tests}件・{n_rep}表現）\n")
    W("前置きなし・配布された精度の表現だけで作る（`+sts`・`@fp32` の条件は混ぜない）．\n")
    cols = [("triplet_agree", "三つ組", ""), ("nn_agree", "最近傍", ""),
            ("knn_jaccard", f"k近傍@{KNN_K}", str(KNN_K)),
            ("dist_rho", "RSA相関", ""), ("procrustes_disp", "Procr↓", "")] + \
           [("fpf_top", f"FPF@{k}", str(k)) for k in FPF_KS]
    buckets = {}
    for x, y in itertools.combinations(base, 2):
        if (x, y, "") not in M["nn_agree"]:
            continue
        key = " × ".join(sorted([grp(x), grp(y)], key=lambda g: ORDER[g]))
        buckets.setdefault(key, []).append((x, y))
    W("| 組み合わせ | ペア数 | " + " | ".join(c[1] for c in cols) + " |")
    W("|---|---:|" + "---:|" * len(cols))
    for key, ps in sorted(buckets.items(), key=lambda kv: -np.mean([cell(M["nn_agree"][(*p, "")]) for p in kv[1]])):
        vals = [np.mean([cell(M[m][(*p, k)]) for p in ps]) for m, _, k in cols]
        W(f"| {key} | {len(ps)} | " + " | ".join(f"{v:.3f}" for v in vals) + " |")
    W("\n同じ表を「位置」（偶然一致 0・実質同一 1）で見たもの:\n")
    W("| 組み合わせ | ペア数 | " + " | ".join(c[1] for c in cols) + " |")
    W("|---|---:|" + "---:|" * len(cols))
    for key, ps in sorted(buckets.items(), key=lambda kv: -np.mean([cell(M["nn_agree"][(*p, "")], "scaled") for p in kv[1]])):
        vals = [np.mean([cell(M[m][(*p, k)], "scaled") for p in ps]) for m, _, k in cols]
        W(f"| {key} | {len(ps)} | " + " | ".join(f"{v:.3f}" for v in vals) + " |")
    W("\n注: `Procr↓` は 0 が一致で向きが逆（位置は他と同じ向きにそろえてある）．"
      "偶然一致の水準は指標ごと・ペアごとに測っており，`三つ組` は 0.5 付近，`最近傍` は 1/(テスト数−1) 付近になる．\n")

    W(f"\n語彙的手法{len(LEX)}種の内訳:\n")
    W("| ペア | 最近傍 | 95%区間 | 位置 | RSA相関 | FPF@5 |")
    W("|---|---:|---|---:|---:|---:|")
    for x, y in itertools.combinations(base, 2):
        if grp(x) == "語彙" == grp(y) and (x, y, "") in M["nn_agree"]:
            r = M["nn_agree"][(x, y, "")]
            W(f"| {x} ↔ {y} | {cell(r):.3f} | [{cell(r, 'value_lo'):.3f}, {cell(r, 'value_hi'):.3f}] "
              f"| {cell(r, 'scaled'):.3f} | {cell(M['dist_rho'][(x, y, '')]):.3f} "
              f"| {cell(M['fpf_top'][(x, y, '5')]):.3f} |")
    W("")

    W("\n### k を振ったときの動き（生の値）\n")
    W("`knn_jaccard` と `fpf_top` の k は外から与える値なので，**1つ選んで語らない**．"
      "下は系統ごとの生の値を k ごとに並べたもの．上の表に載せた k は "
      f"`k近傍@{KNN_K}` と `FPF@{'/'.join(str(k) for k in FPF_KS)}` である．\n")
    for metric, label in (("knn_jaccard", "上位k近傍の重なり"), ("fpf_top", "FPF 上位k件の重なり")):
        allk = sorted({int(k) for (_, _, k) in M[metric] if k})
        if not allk:
            continue
        show = [k for k in KNN_SWEEP if k in allk]
        W(f"\n{label}（`{metric}`）:\n")
        W("| 組み合わせ | " + " | ".join(f"k={k}" for k in show) + " |")
        W("|---|" + "---:|" * len(show))
        for key, ps in sorted(buckets.items()):
            vals = [np.mean([cell(M[metric][(*pp, str(k))]) for pp in ps]) for k in show]
            W(f"| {key} | " + " | ".join(f"{v:.3f}" for v in vals) + " |")

        seq = []
        for k in allk:
            v = {key: np.mean([cell(M[metric][(*pp, str(k))]) for pp in ps])
                 for key, ps in buckets.items()}
            seq.append((k, tuple(sorted(v, key=lambda q: -v[q])), v))
        orders = {o for _, o, _ in seq}
        if len(orders) == 1:
            W(f"\n**k=1〜{allk[-1]} のどの k でも系統の順位は変わらない**"
              f"（{' > '.join(seq[0][1])}）．\n")
        else:
            flips = [k for i, (k, o, _) in enumerate(seq) if i and o != seq[i - 1][1]]
            W(f"\n【注意】 **k によって系統の順位が入れ替わる**（k=1〜{allk[-1]} で"
              f"{len(orders)}通りの順位，入れ替わりは k={', '.join(map(str, flips))}）．\n")
            for o in sorted(orders):
                ks = [str(k) for k, oo, _ in seq if oo == o]
                W(f"- {' > '.join(o)} … k={', '.join(ks)}")
            swapped = [(a_, b_) for a_, b_ in itertools.combinations(sorted(buckets), 2)
                       if len({v[a_] > v[b_] for _, _, v in seq}) > 1]
            for a_, b_ in swapped:
                W(f"\n**入れ替わるのは「{a_}」と「{b_}」で，その差は最大 "
                  f"{max(abs(v[a_] - v[b_]) for _, _, v in seq):.3f}**．"
                  "この2つに差があるとは述べない．\n")
            rest = [q for q in sorted(buckets) if not any(q in sw for sw in swapped)]
            if rest:
                W(f"「{'」「'.join(rest)}」との大小は，どの k でも変わらない．\n")

    W("\n## 2つの参照点（陽性対照・陰性対照）\n")
    pos = {m: [r for r in M[m].values() if r["kind"] == "positive"] for m in METRICS}
    W("| 指標 | 実質同一の水準（陽性対照） | 偶然一致の水準（陰性対照の平均） |")
    W("|---|---:|---:|")
    for m, label, k in cols:
        r = [x for x in pos[m] if x["k"] == k][0]
        ch = np.mean([cell(M[m][(*p, k)], "chance_mean") for p in pairs_all])
        W(f"| {label}（{m}{'@' + k if k else ''}） | {cell(r):.3f} | {ch:.3f} |")
    p0 = pos["nn_agree"][0]
    W(f"\n陽性対照は `{p0['model_a']} ↔ {p0['model_b']}`．"
      "`lsa-full` は次元を n-1 にした版で，tfidf のほぼ回転にすぎない．"
      "**独立した表現ではなく，「実質同一とはどの水準か」を示す目盛りとしてのみ使う．**\n")
    W("陰性対照は，片方の表現の結果についてテスト番号だけを付け替えて測った値の平均"
      f"（{p0['perms']}回）．ペアごとの値は指標ごとの CSV の `chance_mean` にある．\n")

    W("\n## 組ごとの位置と95%区間（主指標 `nn_agree`）\n")
    W(f"区間はアプリを単位にした復元抽出（{p0['boot']}回，乱数の種 {SEED}）．"
      "アプリが8つしかないので区間は広い．**近い組どうしの順位を述べるときは必ず区間を見る．**\n")
    W("| 表現A | 表現B | 最近傍 | 位置 | 位置の95%区間 |")
    W("|---|---|---:|---:|---|")
    for x, y in sorted(pairs_all, key=lambda p: -cell(M["nn_agree"][(*p, "")], "scaled")):
        r = M["nn_agree"][(x, y, "")]
        W(f"| {x} | {y} | {cell(r):.3f} | {cell(r, 'scaled'):.3f} "
          f"| [{cell(r, 'scaled_lo'):.3f}, {cell(r, 'scaled_hi'):.3f}] |")
    W("")

    E = {m: {app: load(f"embeddings/{a.corpus_name}/{m}/{app}_{a.variant}.json") for app in BEWT8} for m in base}
    hits = {(x, y): np.array([int((nn_of(E[x][app])[0] == nn_of(E[y][app])[0]).sum()) for app in BEWT8])
            for x, y in itertools.combinations(base, 2)}
    den = np.array([len(E[base[0]][app]) for app in BEWT8])
    ee = [k for k in hits if grp(k[0]) == "埋込" == grp(k[1])]
    el = [k for k in hits if {grp(k[0]), grp(k[1])} == {"埋込", "語彙"}]
    ll = [k for k in hits if grp(k[0]) == "語彙" == grp(k[1])]

    W("\n## 主要な対比の95%区間（アプリ単位ブートストラップ・BEWT8アプリ）\n")
    W(f"リサンプル {BOOT}回 / 乱数シード {SEED}．"
      "アプリを単位に再抽出するため，同じアプリ内のテストが似ていることを見落とさない．\n")
    W("集計は上の表と同じテスト単位のマイクロ平均で，**同じ量を測っている**"
      "（上の表の値を引き算しても，丸めのぶん 0.001 ずれるだけで一致する）．\n")
    W("| 対比 | 差 | 95%区間 | 0を含むか |")
    W("|---|---:|---|---|")
    idx = rng.integers(0, len(BEWT8), (BOOT, len(BEWT8)))

    def micro(ps, ix):
        return np.sum([hits[k] for k in ps], axis=0)[ix].sum(-1) / (len(ps) * den[ix].sum(-1))

    allapps = np.arange(len(BEWT8))
    for lab, sa, sb in [("語彙×語彙 − 埋込×埋込", ll, ee),
                        ("埋込×語彙 − 埋込×埋込", el, ee),
                        ("語彙×語彙 − 埋込×語彙", ll, el)]:
        diff = float(micro(sa, allapps) - micro(sb, allapps))
        d = micro(sa, idx) - micro(sb, idx)
        lo, hi = np.percentile(d, [2.5, 97.5])
        W(f"| {lab} | {diff:+.3f} | [{lo:+.3f}, {hi:+.3f}] | "
          f"{'含む（差があるとは言えない）' if lo < 0 < hi else '**含まない**'} |")

    V = ["full", "title", "steps", "expect"]
    VPAIRS = list(itertools.combinations(V, 2))
    vp = f"embeddings/{a.corpus_name}/{{m}}/{{app}}_{{v}}.json"
    vmods = [m for m in base
             if all(os.path.exists(vp.format(m=m, app=app, v=v)) for app in BEWT8 for v in V)]
    if vmods:
        W("\n## 入力テキストの切り出し方を変えると（BEWT・最近傍の一致）\n")
        W("同じ表現のまま，埋め込みに渡すテキストだけを変えて測った．**正解ラベルは使わない．**")
        W("**アプリごとに計算してから集計している**（アプリをまたぐと，"
          "表現の違いではなくアプリの違いを測ってしまうため）．\n")
        W("| 表現 | " + " | ".join(f"{x}↔{y}" for x, y in VPAIRS) + " |")
        W("|---" * (len(VPAIRS) + 1) + "|")
        worst = 1.0
        for m in vmods:
            vals = []
            for x, y in VPAIRS:
                hit = tot = 0
                for app in BEWT8:
                    na = nn_of(load(vp.format(m=m, app=app, v=x)))[0]
                    nb = nn_of(load(vp.format(m=m, app=app, v=y)))[0]
                    hit += int((na == nb).sum())
                    tot += len(na)
                vals.append(hit / tot)
            worst = min(worst, min(vals))
            W(f"| {m} | " + " | ".join(f"{x:.3f}" for x in vals) + " |")
        W(f"\n最も低いところで **{worst:.3f}**．バリアントによって本文が変わるので，"
          "同じ表現でも渡すテキストが違えば別の条件として扱う．\n")
        if len(vmods) < len(base):
            W(f"（4バリアントがそろっていない表現は外した: {' '.join(m for m in base if m not in vmods)}）\n")

    SIZE = {"sbert-all-mpnet-base-v2": 0.11, "bge-base-en-v1.5": 0.11, "e5-base-v2": 0.11,
            "qwen3-0.6b": 0.6, "gte-qwen2-1.5b-instruct": 1.5, "stella-en-1.5b-v5": 1.5,
            "qwen3-4b": 4.0, "gte-qwen2-7b-instruct": 7.0, "qwen3-8b": 8.0}
    FAMILY = {"sbert-all-mpnet-base-v2": "mpnet", "bge-base-en-v1.5": "bert",
              "e5-base-v2": "bert", "qwen3-0.6b": "qwen3", "qwen3-4b": "qwen3",
              "qwen3-8b": "qwen3", "gte-qwen2-1.5b-instruct": "qwen2",
              "gte-qwen2-7b-instruct": "qwen2", "stella-en-1.5b-v5": "qwen2"}
    emb = [m for m in base if grp(m) == "埋込" and m in SIZE]
    if len(emb) >= 2:
        W("\n## 規模の影響（BEWT・最近傍の一致）\n")
        W("| 比較 | 規模 | 系統 | 一致 | 位置 |")
        W("|---|---|---|---:|---:|")
        for x, y in itertools.combinations(sorted(emb), 2):
            if (x, y, "") not in M["nn_agree"]:
                continue
            r = M["nn_agree"][(x, y, "")]
            same_f = "同系統" if FAMILY[x] == FAMILY[y] else "別系統"
            same_s = "同規模" if SIZE[x] == SIZE[y] else f"{max(SIZE[x], SIZE[y]) / min(SIZE[x], SIZE[y]):.1f}倍"
            W(f"| {x} ↔ {y} | {same_s} | {same_f} | {cell(r):.3f} | {cell(r, 'scaled'):.3f} |")
        prs = [(x, y) for x, y in itertools.combinations(sorted(emb), 2) if (x, y, "") in M["nn_agree"]]
        nn = {p: cell(M["nn_agree"][(*p, "")]) for p in prs}
        same_size = [nn[p] for p in prs if SIZE[p[0]] == SIZE[p[1]]]
        diff_size = [nn[p] for p in prs if SIZE[p[0]] != SIZE[p[1]]]
        xs = [abs(np.log10(SIZE[x]) - np.log10(SIZE[y])) for x, y in prs]
        W(f"\n| 組 | ペア数 | 平均 |")
        W("|---|---:|---:|")
        if same_size:
            W(f"| 同じ規模どうし | {len(same_size)} | {np.mean(same_size):.3f} |")
        if diff_size:
            W(f"| 規模が違う組 | {len(diff_size)} | {np.mean(diff_size):.3f} |")
        if len(set(xs)) > 1:
            rho, pv = _st.spearmanr(xs, [nn[p] for p in prs])
            W(f"\n規模差（対数）と一致の順位相関: **ρ = {rho:+.3f}，p = {pv:.3f}**"
              f"（{'有意ではない' if pv >= 0.05 else '有意'}）．ペア数が少ないので参考値．\n")
        else:
            W("\n規模の違う組が無いので，規模差との順位相関は出していない．\n")

        same_fam = [nn[p] for p in prs if FAMILY[p[0]] == FAMILY[p[1]]]
        diff_fam = [nn[p] for p in prs if FAMILY[p[0]] != FAMILY[p[1]]]
        W("\n### 系統内 vs 系統外の一致（規模ではなく，土台モデルで分ける）\n")
        W("| 組 | ペア数 | 平均 |")
        W("|---|---:|---:|")
        if same_fam:
            W(f"| 同系統どうし | {len(same_fam)} | {np.mean(same_fam):.3f} |")
        if diff_fam:
            W(f"| 別系統どうし | {len(diff_fam)} | {np.mean(diff_fam):.3f} |")

        fam_members = {}
        for m in emb:
            fam_members.setdefault(FAMILY[m], []).append(m)
        multi_fam = {f: sorted(ms) for f, ms in fam_members.items() if len(ms) >= 2}
        if multi_fam:
            W("\n複数の規模がある系統ごとに，系統内（軽量↔重量）の一致と，"
              "その系統のモデルが系統外と組んだときの一致を分けて見る．\n")
            W("| 系統 | 系統内の表現 | 系統内平均（組数） | 系統外平均（組数） |")
            W("|---|---|---:|---:|")
            for f in sorted(multi_fam):
                ms = multi_fam[f]
                inn = [nn[p] for p in prs if p[0] in ms and p[1] in ms]
                out = [nn[p] for p in prs if (p[0] in ms) != (p[1] in ms)]
                W(f"| {f} | {'・'.join(ms)} | {np.mean(inn):.3f}（{len(inn)}） | {np.mean(out):.3f}（{len(out)}） |")

    emb_all = [m for m in base if grp(m) == "埋込"]
    if len(emb_all) >= 2:
        W("\n## 感度1: FPF@5 は開始点規則にどれだけ依存するか（埋込どうし・BEWT8アプリ）\n")
        W("| 開始点の決め方 | FPF@5 |")
        W("|---|---:|")
        for lab, fn in [("最外れ点から開始（本研究の規則）", lambda X: None),
                        ("中心点から開始", lambda X: int(np.argmax((X @ X.T).sum(1)))),
                        ("両表現で同じ点から開始", lambda X: 0)]:
            sel = {(m, app): set(fpf(E[m][app], 5, fn(E[m][app])))
                   for m in emb_all for app in BEWT8}
            v = [np.mean([len(sel[(x, app)] & sel[(y, app)]) / 5 for app in BEWT8])
                 for x, y in itertools.combinations(emb_all, 2)]
            W(f"| {lab} | {np.mean(v):.3f} |")
        start = {(m, app): int(np.argmax((1.0 - E[m][app] @ E[m][app].T).sum(1)))
                 for m in emb_all for app in BEWT8}
        agree = [np.mean([start[(x, app)] == start[(y, app)] for app in BEWT8])
                 for x, y in itertools.combinations(emb_all, 2)]
        W(f"\n埋込どうしが同じ開始点を選ぶ割合 = **{np.mean(agree):.3f}**"
          f"（8アプリ中 {np.mean(agree) * 8:.1f}アプリ）．\n")

    W("\n## 感度2: 最近傍の同点（BEWT8アプリ）\n")
    W("| 表現 | 同点で最近傍が定まらないテスト |")
    W("|---|---:|")
    for m in base:
        tie = tot = 0
        for app in BEWT8:
            _, S = nn_of(E[m][app])
            mx = S.max(1, keepdims=True)
            tie += int(((S >= mx - 1e-9).sum(1) > 1).sum())
            tot += len(S)
        W(f"| {m} | {tie}/{tot} ({tie / tot:.1%}) |")
    W("\n同点のときは行番号の小さい方を最近傍とする（どの表現でも同じ規則）．\n")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as _f:
        _f.write("\n".join(L) + "\n")
    print(f"→ {a.out}")


if __name__ == "__main__":
    main()
