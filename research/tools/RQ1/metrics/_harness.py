#!/usr/bin/env python3
"""指標ごとの run.py が共有する土台．直接実行しない"""
import argparse, csv, itertools, json, os, zlib
import numpy as np


BOOT = 10000
SEED = 0
# 陽性対照にだけ使い，表現どうしの比較には入れない
REFERENCE_ONLY = ("lsa-full",)


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


def run(metric, prepare, compare=None, permute=None, contrib=contrib_micro,
        key_list=None, show_keys=None, perms_default=1000,
        pair_fn=None, extra_cols=None, show_cols=(), footer=None):
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
    ap.add_argument("--positive", nargs=2, default=["tfidf", "lsa-full"], metavar=("A", "B"),
                    help="陽性対照にする2表現（既定 tfidf lsa-full）")
    ap.add_argument("--perms", type=int, default=perms_default,
                    help=f"陰性対照の並べ替え回数（既定 {perms_default}）")
    ap.add_argument("--boot", type=int, default=BOOT, help=f"区間のための復元抽出の回数（既定 {BOOT}）")
    ap.add_argument("--out", "-o", default=None,
                    help=f"出力 CSV（既定 results/metrics/{metric}/{{corpus_name}}_{{variant}}.csv）")
    a = ap.parse_args()
    if a.perms < 1:
        raise SystemExit("--perms は1以上にすること")
    if a.boot < 100:
        raise SystemExit("--boot は100以上にすること（区間が粗くなりすぎる）")
    out = a.out or os.path.join("results", "metrics", metric, f"{a.corpus_name}_{a.variant}.csv")

    if not os.path.isdir(a.embeddings):
        raise SystemExit(f"{a.embeddings} が無い")
    found = sorted(d for d in os.listdir(a.embeddings) if os.path.isdir(os.path.join(a.embeddings, d)))
    for m in list(a.exclude) + list(a.only or []) + list(a.positive):
        if m not in found:
            raise SystemExit(f"{m} が {a.embeddings} に無い")
    models = [m for m in found if m not in REFERENCE_ONLY]
    if a.only:
        models = [m for m in models if m in a.only]
    models = [m for m in models if m not in a.exclude]
    if len(models) < 2:
        raise SystemExit("比べる表現が2件未満")
    print(f"表現 {len(models)}件: {' '.join(models)}")

    need = sorted(set(models) | set(a.positive))
    per = {m: {} for m in need}
    for m in need:
        for app in a.apps:
            X = load(a.embeddings, a.corpus, m, app, a.variant)
            if X is not None:
                per[m][app] = X
    common = [app for app in a.apps if all(app in per[m] for m in need)]
    for app in a.apps:
        if app not in common:
            print(f"  [除外] {app}: {' '.join(m for m in need if app not in per[m])} が無い")
    if not common:
        raise SystemExit("全表現が揃うアプリが無い")
    print(f"全表現が揃うアプリ {len(common)}件: {' '.join(common)}")

    ns = {app: len(per[need[0]][app]) for app in common}
    prep = {(m, app): prepare(per[m][app]) for m in need for app in common}
    keys = key_list(ns)
    boot = np.random.default_rng(SEED).integers(0, len(common), size=(a.boot, len(common)))

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

    def agg_boot(num, den):
        d = den[boot].sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(d > 0, num[boot].sum(1) / np.where(d > 0, d, 1), np.nan)

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

    # 陽性対照は1回だけ測り，全ての行で使い回す
    pos = evaluate(*a.positive)
    pos_arr, pos_null = pos[0], pos[1]
    positive = {k: agg(*pos_arr[k]) for k in keys}
    pos_boot = {k: agg_boot(*pos_arr[k]) for k in keys}

    rows = []
    pairs = [("positive", *a.positive)] + [("actual", x, y) for x, y in itertools.combinations(models, 2)]
    for kind, x, y in pairs:
        arr, null, obs, nulls = pos if kind == "positive" else evaluate(x, y)
        for k in keys:
            num, den = arr[k]
            value = agg(num, den)
            series = np.array([agg(*d[k]) for d in null])
            cm, csd = float(np.nanmean(series)), float(np.nanstd(series))
            cnum = np.nanmean([d[k][0] for d in null], axis=0)
            cden = np.nanmean([d[k][1] for d in null], axis=0)
            vb, cb = agg_boot(num, den), agg_boot(cnum, cden)
            with np.errstate(invalid="ignore", divide="ignore"):
                sb = (vb - cb) / (pos_boot[k] - cb)
            lo, hi = np.nanpercentile(vb, [2.5, 97.5])
            slo, shi = np.nanpercentile(sb, [2.5, 97.5])
            wide = positive[k] - cm
            row = dict(
                corpus=a.corpus_name, variant=a.variant, metric=metric, k=k, kind=kind,
                model_a=x, model_b=y, n=sum(ns[app] for app in common), n_apps=len(common),
                value=round(value, 6), value_lo=round(float(lo), 6), value_hi=round(float(hi), 6),
                chance_mean=round(cm, 6), chance_sd=round(csd, 6), positive=round(positive[k], 6),
                scaled=round((value - cm) / wide, 6) if wide != 0 else "",
                scaled_lo=round(float(slo), 6) if wide != 0 else "",
                scaled_hi=round(float(shi), 6) if wide != 0 else "",
                perms=a.perms, boot=a.boot)
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
    head = f"\n{'種類':<10}{'表現A':<28}{'表現B':<28}"
    if with_k:
        head += f"{'k':>4}"
    head += f"{'値':>9}{'偶然':>9}{'位置':>8}{'位置の95%区間':>20}"
    print(head + "".join(f"{label:>{w}}" for label, w, _ in show_cols))
    for r in show:
        sc = f"{r['scaled']:.3f}" if r["scaled"] != "" else "-"
        ci = f"[{r['scaled_lo']:.3f}, {r['scaled_hi']:.3f}]" if r["scaled_lo"] != "" else "-"
        line = f"{r['kind']:<10}{r['model_a']:<28}{r['model_b']:<28}"
        if with_k:
            line += f"{str(r['k']):>4}"
        line += f"{r['value']:>9.4f}{r['chance_mean']:>9.4f}{sc:>8}{ci:>20}"
        print(line + "".join(f"{col(r):>{w}}" for _, w, col in show_cols))
    if len(show) < len(rows):
        print(f"（表示は k={' '.join(str(k) for k in show_keys(keys))} のみ．CSV には k={keys[0]}〜{keys[-1]} の全てが入っている）")
    if footer is not None:
        print(f"\n{footer(a)}")
    print(f"\n→ {out}")
