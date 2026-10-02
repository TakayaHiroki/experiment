#!/usr/bin/env python3
"""RQ1 の集計値（BEWT）を表ごとの CSV に書き，画面にも出す

usage:
  python tools/RQ1/rq1_summary.py --out-dir results/rq1_summary
"""
import argparse, csv, importlib.util, itertools, json, os, sys
import numpy as np
from scipy import stats

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "metrics"))
import _harness
_spec = importlib.util.spec_from_file_location("farthest_top_run", os.path.join(_HERE, "metrics", "farthest_top", "run.py"))
_FARTHEST = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_FARTHEST)

with open(os.path.join(_HERE, "conditions.json"), encoding="utf-8") as _f:
    _CONDITIONS = json.load(_f)
ORDER = {c["name"]: i for i, c in enumerate(_CONDITIONS)}
LEX = {c["name"] for c in _CONDITIONS if c["method"] != "model"}
BEWT8 = ["bludit", "claroline", "expresscart", "joomla",
         "kanboard", "mantisbt", "mediawiki", "prestashop"]
METRIC_DIR = "results/metrics"
METRICS = ["nn_agree", "farthest_top", "dist_rho"]
COLS = [("nn_agree", ""), ("farthest_top", "5"), ("farthest_top", "10"), ("dist_rho", "")]
CONDITION_MARKS = ("+sts", "@fp32")
NEAR = 1e-6
VARIANT_PAIRS = [("full", "title"), ("full", "steps"), ("full", "expect")]
FILES = ["range", "pairs", "mantel", "nn_matrix", "contrast", "farthest_curve", "variants"]


def is_base(m):
    return not any(c in m for c in CONDITION_MARKS)


def col_name(m, k):
    return f"{m}@{k}" if k else m


_CACHE = {}


def load(emb_dir, corpus_dir, m, app, variant):
    key = (m, app, variant)
    if key not in _CACHE:
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


def row_of(M, metric, x, y, k=""):
    r = M[metric].get((x, y, k)) or M[metric].get((y, x, k))
    if r is None:
        raise SystemExit(f"{metric} の CSV に {x} ↔ {y}（k={k or '-'}）の行が無い")
    return r


def ordered(p):
    return tuple(sorted(p, key=ORDER.__getitem__))


def to_csv(v):
    if isinstance(v, str):
        return v
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return "" if v != v else f"{v:.6f}".replace("-0.000000", "0.000000")


def to_screen(v):
    if isinstance(v, str):
        return v
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return "-" if v != v else f"{v:.3f}".replace("-0.000", "0.000")


def show(title, head, rows):
    cells = [head] + [[to_screen(v) for v in r] for r in rows]
    width = [max(len(r[i]) for r in cells) for i in range(len(head))]
    print(f"\n[{title}]")
    for r in cells:
        print("  ".join(s.ljust(w) if i == 0 else s.rjust(w) for i, (s, w) in enumerate(zip(r, width))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", "-o", default=os.path.join("results", "rq1_summary"))
    ap.add_argument("--corpus-name", default="bewt")
    ap.add_argument("--variant", default="full")
    ap.add_argument("--embeddings", "-e", default=None, help="既定 embeddings/{corpus_name}")
    ap.add_argument("--corpus", "-c", default=None, help="既定 corpus/{corpus_name}/json")
    a = ap.parse_args()
    emb = a.embeddings or os.path.join("embeddings", a.corpus_name)
    cdir = a.corpus or os.path.join("corpus", a.corpus_name, "json")

    M = {m: read_metric(m, a.corpus_name, a.variant) for m in METRICS}
    apps, excluded, n_tests = check_scope(M, a.corpus_name, a.variant)
    check_fresh(M, emb, a.corpus_name, a.variant, apps)

    pairs_all = sorted((ordered((x, y)) for (x, y, _) in M["nn_agree"]), key=lambda p: (ORDER[p[0]], ORDER[p[1]]))
    reps = sorted({m for p in pairs_all for m in p}, key=ORDER.__getitem__)
    base = [m for m in reps if is_base(m)]
    base_pairs = list(itertools.combinations(base, 2))
    if not base_pairs:
        raise SystemExit(f"前置きなし・配布された精度の表現が2つ未満（{' '.join(base) or 'なし'}）")
    lex_emb = [p for p in base_pairs if (p[0] in LEX) != (p[1] in LEX)]
    emb_emb = [p for p in base_pairs if p[0] not in LEX and p[1] not in LEX]
    lex_lex = [p for p in base_pairs if p[0] in LEX and p[1] in LEX]
    prec = sorted({(y, x) if x.endswith("@fp32") else (x, y) for x, y in pairs_all
                   if x == y + "@fp32" or y == x + "@fp32"}, key=lambda p: ORDER[p[0]])
    pfx = sorted({(m.replace("+sts", ""), m) for m in reps if "+sts" in m and m.replace("+sts", "") in reps},
                 key=lambda p: ORDER[p[0]])

    def val(m, p, k=""):
        return cell(row_of(M, m, *p, k))

    E = {m: {app: load(emb, cdir, m, app, a.variant) for app in apps} for m in reps}
    den = np.array([len(E[reps[0]][app]) for app in apps])
    if int(den.sum()) != n_tests:
        raise SystemExit(f"埋め込みのテスト数（{int(den.sum())}）が CSV の n（{n_tests}）と違う")
    nns = {(m, app): nn_of(E[m][app])[0] for m in reps for app in apps}
    tops = {(m, app): set(_FARTHEST.prepare(E[m][app])[:5]) for m in reps for app in apps}
    hits = {}
    for x, y in pairs_all:
        hits[(x, y)] = np.array([int((nns[(x, app)] == nns[(y, app)]).sum()) for app in apps])
        checks = [("最近傍の一致", "nn_agree", "", hits[(x, y)].sum() / den.sum()),
                  ("最遠点@5", "farthest_top", "5", float(np.mean([len(tops[(x, app)] & tops[(y, app)]) / 5 for app in apps])))]
        for lab, m, k, w in checks:
            v = val(m, (x, y), k)
            if abs(v - w) > 1e-6:
                raise SystemExit(f"{x} ↔ {y} の{lab}が，CSV（{v:.6f}）と埋め込みから測り直した値（{w:.6f}）で違う．" + chr(10)
                                 + "  CSV を作ったあとに埋め込みか指標のスクリプトが変わった．指標を測り直すこと．")

    print(f"アプリ {len(apps)}件: {' '.join(apps)} / テスト {n_tests}件 / 表現 {len(reps)}件 / 組 {len(pairs_all)}"
          f" / 外した条件: {' '.join(excluded) if excluded else 'なし'}")
    print(f"照合: {len(pairs_all)}組の最近傍の一致と最遠点@5 が埋め込みからの測り直しと一致")

    T = {}

    rows = []
    for m, k in COLS:
        v = [val(m, p, k) for p in base_pairs]
        chance = float(np.mean([cell(row_of(M, m, *p, k), "chance_mean") for p in base_pairs]))
        rows.append([col_name(m, k), len(base_pairs), chance, min(v), float(np.median(v)), max(v)])
    T["range"] = (["metric", "pairs", "chance_mean", "min", "median", "max"], rows)
    show(f"range  前置きなし・配布された精度の{len(base)}表現", *T["range"])

    head = ["kind", "model_a", "model_b"]
    for m, k in COLS:
        c = col_name(m, k)
        head += [c, c + "_lo", c + "_hi"]
    rows, screen = [], []
    for kind, ps in (("precision", prec), ("prefix", pfx)):
        for x, y in ps:
            rs = [row_of(M, m, x, y, k) for m, k in COLS]
            rows.append([kind, x, y] + [cell(r, c) for r in rs for c in ("value", "value_lo", "value_hi")])
            screen.append([kind, f"{x} ↔ {y}"] + [f"{cell(r):.3f} [{cell(r, 'value_lo'):.3f}, {cell(r, 'value_hi'):.3f}]" for r in rs])
    T["pairs"] = (head, rows)
    show("pairs  値 [95%区間]", ["kind", "pair"] + [col_name(m, k) for m, k in COLS], screen)

    rho_rows = list(M["dist_rho"].values())
    perms = sorted({int(r["perms"]) for r in rho_rows})
    if len(perms) != 1:
        raise SystemExit(f"dist_rho の並べ替えの回数が組によって違う: {perms}．指標を測り直すこと．")
    floor = 1.0 / (perms[0] + 1)
    p_min = float(f"{float(stats.chi2.sf(-2.0 * len(apps) * np.log(floor), 2 * len(apps))):.3e}")
    p_max = max(rho_rows, key=lambda r: float(r["p_fisher"]))["p_fisher"]
    T["mantel"] = (["pairs", "apps", "perms", "p_floor", "pairs_all_apps_at_floor", "p_fisher_max"],
                   [[len(rho_rows), len(apps), perms[0], f"{floor:.3e}", sum(float(r["p_fisher"]) <= p_min for r in rho_rows), p_max]])
    show("mantel", *T["mantel"])

    T["nn_matrix"] = (["model"] + base,
                      [[x] + ["" if x == y else val("nn_agree", (x, y)) for y in base] for x in base])
    show("nn_matrix  最近傍の一致", [""] + [str(i + 1) for i in range(len(base))],
         [[f"{i + 1} {x}"] + ["—" if x == y else val("nn_agree", (x, y)) for y in base] for i, x in enumerate(base)])

    head = ["app", "n", "emb_emb", "lex_emb", "diff", "diff_lo", "diff_hi"]
    rows = []
    if emb_emb and lex_emb:
        ee = np.mean([hits[p] for p in emb_emb], axis=0)
        le = np.mean([hits[p] for p in lex_emb], axis=0)
        d, lo, hi = _harness.t_interval(ee - le, den)
        rows = [[app, int(n), e / n, l / n, (e - l) / n, "", ""] for app, n, e, l in zip(apps, den, ee, le)]
        rows.append(["all", int(den.sum()), float(ee.sum() / den.sum()), float(le.sum() / den.sum()), d, lo, hi])
    T["contrast"] = (head, rows)
    show(f"contrast  最近傍の一致  埋込×埋込 {len(emb_emb)}組 / 語彙×埋込 {len(lex_emb)}組", *T["contrast"])

    ks = sorted({int(k) for (_, _, k) in M["farthest_top"]})
    rows = []
    for g, ps in (("埋込×埋込", emb_emb), ("語彙×埋込", lex_emb)):
        if ps:
            rows.append([g, "", "", len(ps)] + [float(np.mean([val("farthest_top", p, str(k)) for p in ps])) for k in ks])
    for g, ps in (("語彙×語彙", lex_lex), ("精度だけ違う組", prec)):
        for x, y in ps:
            rows.append([g, x, y, 1] + [val("farthest_top", (x, y), str(k)) for k in ks])
    rows.append(["偶然一致", "", "", len(base_pairs)]
                + [float(np.mean([cell(row_of(M, "farthest_top", *p, str(k)), "chance_mean") for p in base_pairs])) for k in ks])
    T["farthest_curve"] = (["group", "model_a", "model_b", "pairs"] + [f"k{k}" for k in ks], rows)
    show("farthest_curve  最遠点@k の重なり", *T["farthest_curve"])

    V = list(dict.fromkeys(v for p in VARIANT_PAIRS for v in p))
    vmods = [m for m in base
             if all(os.path.exists(os.path.join(emb, m, f"{app}_{v}.json")) for app in apps for v in V)]
    rows = []
    for m in vmods:
        X = {(app, v): load(emb, cdir, m, app, v) for app in apps for v in V}
        gap = max(same_text_gap(X[(app, v)], corpus_texts(cdir, app, v)) for app in apps for v in V)
        width = max(NEAR, gap + 1e-12)
        Tn = {key: near_max(nn_of(x)[1], width) for key, x in X.items()}
        ties = {v: sum(int((Tn[(app, v)].sum(1) > 1).sum()) for app in apps) for v in V}
        for x, y in VARIANT_PAIRS:
            exp = lo = hi = tot = 0.0
            for app in apps:
                tx, ty = Tn[(app, x)], Tn[(app, y)]
                both = (tx & ty).sum(1)
                exp += float((both / (tx.sum(1) * ty.sum(1))).sum())
                hi += int((both > 0).sum())
                lo += int(((tx.sum(1) == 1) & (ty.sum(1) == 1) & (both == 1)).sum())
                tot += len(tx)
            rows.append([m, x, y, exp / tot, lo / tot, hi / tot, f"{width:.3e}", ties[x], ties[y]])
    T["variants"] = (["model", "variant_a", "variant_b", "expected", "min", "max", "tie_width", "ties_a", "ties_b"], rows)
    show("variants  最近傍の一致（同点はくじ引きの期待値と最小・最大）", *T["variants"])
    if len(vmods) < len(base):
        print(f"4バリアントがそろわない表現（variants に無い）: {' '.join(m for m in base if m not in vmods)}")

    os.makedirs(a.out_dir, exist_ok=True)
    for name in FILES:
        head, rows = T[name]
        path = os.path.join(a.out_dir, f"{name}.csv")
        tmp = path + ".tmp"
        with open(tmp, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(head)
            w.writerows([[to_csv(v) for v in r] for r in rows])
        os.replace(tmp, path)
    print(f"\n→ {a.out_dir}（{' '.join(f'{n}.csv' for n in FILES)}）")


if __name__ == "__main__":
    main()
