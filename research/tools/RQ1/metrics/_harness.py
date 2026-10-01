#!/usr/bin/env python3
"""指標ごとの run.py が共有する土台．直接実行しない"""
import argparse, csv, itertools, json, os, zlib
import numpy as np
from scipy import stats

SEED = 0
# 類似度はこの桁で丸めてから比べる．数学的に等しい値（同じテキストの行など）が 1e-16 の誤差で別の順位にならないように
DECIMALS = 12
with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "conditions.json"), encoding="utf-8") as _f:
    _CONDITIONS = json.load(_f)
KNOWN = [c["name"] for c in _CONDITIONS]
DEFAULT = [c["name"] for c in _CONDITIONS if c["default"]]


def similarity(X):
    return np.round(X @ X.T, DECIMALS)


def load(emb_dir, corpus_dir, model, app, variant):
    p = os.path.join(emb_dir, model, f"{app}_{variant}.json")
    if not os.path.isfile(p):
        return None
    c = os.path.join(corpus_dir, f"{app}_{variant}.json")
    try:
        with open(c, encoding="utf-8") as f:
            titles = [r["title"] for r in json.load(f)]
    except (OSError, ValueError, TypeError, KeyError) as e:
        raise SystemExit(f"コーパス {c} を読めない（{e!r}）")
    try:
        with open(p, encoding="utf-8") as f:
            E = json.load(f)
        got = [r["title"] for r in E]
        X = np.array([r["embedding"] for r in E], dtype=float)
    except (OSError, ValueError, TypeError, KeyError) as e:
        raise SystemExit(f"{p} を読めない（{e!r}）")
    # 古い埋め込みが混ざると別のテストを突き合わせたまま計算してしまう
    if got != titles:
        raise SystemExit(f"{p} のテストの並びがコーパスと一致しない（作り直すこと）")
    if X.ndim != 2 or len(X) < 2 or not np.isfinite(X).all():
        raise SystemExit(f"{p} の埋め込みが2件以上の有限な行列になっていない")
    # 行ごとに長さ1にそろえる
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


# 表現の組とアプリごとに別の乱数列．消費順序を変えると値が変わる
def rng_for(*names):
    return np.random.default_rng([SEED, zlib.crc32("|".join(names).encode("utf-8"))])


def contrib_micro(raw, n, k):
    return raw, n


def contrib_macro(v, n, k):
    return (0.0, 0.0) if v != v else (v, 1.0)


def t_interval(num, den, bounds=(-np.inf, np.inf), level=0.95):
    """アプリごとの (分子, 分母) から，分子の和 ÷ 分母の和と，その区間を返す

    アプリを単位にした残差 e = 分子 − 値 × 分母 から標準誤差を出し，t 分布（自由度 = アプリ数 − 1）を使う．
    分母が1ならアプリごとの値の平均 ± t × 標準偏差 / √アプリ数 と同じになる．区間は値の取りうる範囲 bounds で切る
    """
    num, den = np.asarray(num, dtype=float), np.asarray(den, dtype=float)
    use = den > 0
    g, total = int(use.sum()), den[use].sum()
    if g == 0:
        return float("nan"), float("nan"), float("nan")
    value = float(num[use].sum() / total)
    if g < 2:
        return value, float("nan"), float("nan")
    e = num[use] - value * den[use]
    half = stats.t.ppf(0.5 + level / 2, g - 1) * np.sqrt(g / (g - 1) * (e ** 2).sum()) / total
    return value, float(max(value - half, bounds[0])), float(min(value + half, bounds[1]))


def run(metric, prepare, compare=None, permute=None, contrib=contrib_micro,
        key_list=None, show_keys=None, perms_default=1000,
        pair_fn=None, extra_cols=None, show_cols=(), footer=None, bounds=(0.0, 1.0)):
    if (compare is None) == (pair_fn is None):
        raise SystemExit("compare か pair_fn のどちらか一方を渡すこと")
    if compare is not None and permute is None:
        raise SystemExit("compare を渡すときは permute も渡すこと")
    if extra_cols is not None and pair_fn is None:
        raise SystemExit("extra_cols は pair_fn と併せて渡すこと（アプリごとの生の値が要る）")
    key_list = key_list or (lambda ns: ("",))
    show_keys = show_keys or (lambda keys: keys)

    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", "-e", required=True, help="embeddings/{コーパス} のパス")
    ap.add_argument("--corpus", "-c", required=True, help="corpus/{コーパス}/json のパス（テストの並びの照合に使う）")
    ap.add_argument("--corpus-name", required=True)
    ap.add_argument("--apps", nargs="+", required=True)
    ap.add_argument("--variant", default="full", choices=["title", "steps", "expect", "full"])
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--perms", type=int, default=perms_default,
                    help=f"偶然一致の水準を測る並べ替えの回数（既定 {perms_default}）")
    ap.add_argument("--out", "-o", default=None,
                    help=f"出力 CSV（既定 results/metrics/{metric}/{{corpus_name}}_{{variant}}.csv）")
    a = ap.parse_args()
    if a.perms < 1:
        raise SystemExit("--perms は1以上にすること")
    if len(set(a.apps)) != len(a.apps):
        raise SystemExit("--apps に同じアプリが2回ある")
    out = a.out or os.path.join("results", "metrics", metric, f"{a.corpus_name}_{a.variant}.csv")

    if not os.path.isdir(a.embeddings):
        raise SystemExit(f"{a.embeddings} が無い")
    found = sorted(d for d in os.listdir(a.embeddings) if os.path.isdir(os.path.join(a.embeddings, d)))
    for m in list(a.exclude) + list(a.only or []):
        if m not in KNOWN:
            raise SystemExit(f"{m} は conditions.json の条件に無い")
    for m in a.only or []:
        if m in a.exclude or m not in found:
            raise SystemExit(f"{m} が {a.embeddings} に無いか，--exclude で外されている")
    unknown = [d for d in found if d not in KNOWN]
    if unknown:
        raise SystemExit(f"conditions.json に無い条件のフォルダがある（指標の計算に混ざる）: {' '.join(unknown)}\n"
                         f"  {a.embeddings} の外へ移すこと")
    if a.only is None:
        lack = [m for m in DEFAULT if m not in found and m not in a.exclude]
        if lack:
            raise SystemExit(f"既定の条件のフォルダが無い: {' '.join(lack)}\n"
                             "  作るか，作れなかった条件なら --exclude で外すこと")
    models = [m for m in found if m not in a.exclude and (a.only is None or m in a.only)]
    if len(models) < 2:
        raise SystemExit("比べる表現が2件未満")
    print(f"表現 {len(models)}件: {' '.join(models)}")
    if a.exclude:
        print(f"外した条件 {len(a.exclude)}件: {' '.join(a.exclude)}")

    per = {m: {} for m in models}
    for m in models:
        for app in a.apps:
            X = load(a.embeddings, a.corpus, m, app, a.variant)
            if X is not None:
                per[m][app] = X
    # アプリだけを外すと全ての組の集計範囲が変わるので，黙って外さない
    missing = [f"{m}/{app}_{a.variant}.json" for m in models for app in a.apps if app not in per[m]]
    if missing:
        raise SystemExit(f"ベクトルが無い（{len(missing)}件）: {' '.join(missing)}\n"
                         "  作り直すか，作れなかった条件なら --exclude で条件ごと外すこと")
    common = list(a.apps)
    print(f"アプリ {len(common)}件: {' '.join(common)}")

    ns = {app: len(per[models[0]][app]) for app in common}
    prep = {(m, app): prepare(per[m][app]) for m in models for app in common}
    keys = key_list(ns)

    def to_arrays(raw):
        out = {}
        for k in keys:
            # 集計は分子の和 ÷ 分母の和．マイクロかマクロかは contrib の返し方で決まる
            nd = np.array([contrib(raw[app], ns[app], k) for app in common], dtype=float)
            out[k] = (nd[:, 0], nd[:, 1])
        return out

    def agg(num, den):
        s = den.sum()
        return float(num.sum() / s) if s > 0 else float("nan")

    def evaluate(x, y):
        if pair_fn is not None:
            obs, nulls = {}, {}
            for app in common:
                obs[app], nulls[app] = pair_fn(prep[(x, app)], prep[(y, app)],
                                               ns[app], a.perms, rng_for(x, y, app))
            null = [{app: nulls[app][r] for app in common} for r in range(a.perms)]
            return to_arrays(obs), [to_arrays(d) for d in null], obs, nulls
        vals = {app: compare(prep[(x, app)], prep[(y, app)]) for app in common}
        null = []
        rngs = {app: rng_for(x, y, app) for app in common}
        for _ in range(a.perms):
            null.append({app: compare(prep[(x, app)], permute(prep[(y, app)], rngs[app].permutation(ns[app])))
                         for app in common})
        return to_arrays(vals), [to_arrays(d) for d in null], None, None

    rows = []
    for x, y in itertools.combinations(models, 2):
        arr, null, obs, nulls = evaluate(x, y)
        for k in keys:
            value, lo, hi = t_interval(*arr[k], bounds=bounds)
            series = np.array([agg(*d[k]) for d in null])
            row = dict(
                corpus=a.corpus_name, variant=a.variant, metric=metric, k=k,
                model_a=x, model_b=y, n=sum(ns[app] for app in common), n_apps=len(common),
                apps=" ".join(common), excluded=" ".join(sorted(set(a.exclude))),
                value=round(value, 6), value_lo=round(lo, 6), value_hi=round(hi, 6),
                chance_mean=round(float(np.nanmean(series)), 6), chance_sd=round(float(np.nanstd(series)), 6),
                perms=a.perms)
            if extra_cols is not None:
                row.update(extra_cols(obs, nulls, common, a.perms))
            rows.append(row)

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    # 書き終えてから置き換える
    tmp = out + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, out)

    show = [r for r in rows if r["k"] in show_keys(keys)]
    with_k = len(keys) > 1 or keys[0] != ""
    head = f"\n{'表現A':<28}{'表現B':<28}"
    if with_k:
        head += f"{'k':>4}"
    head += f"{'値':>9}{'95%区間':>20}{'偶然':>9}"
    print(head + "".join(f"{label:>{w}}" for label, w, _ in show_cols))
    for r in show:
        line = f"{r['model_a']:<28}{r['model_b']:<28}"
        if with_k:
            line += f"{str(r['k']):>4}"
        ci = f"[{r['value_lo']:.3f}, {r['value_hi']:.3f}]"
        line += f"{r['value']:>9.4f}{ci:>20}{r['chance_mean']:>9.4f}"
        print(line + "".join(f"{col(r):>{w}}" for _, w, col in show_cols))
    if len(show) < len(rows):
        print(f"（表示は k={' '.join(str(k) for k in show_keys(keys))} のみ．CSV には k={keys[0]}〜{keys[-1]} の全てが入っている）")
    if footer is not None:
        print(f"\n{footer(a)}")
    print(f"\n→ {out}")
