#!/usr/bin/env python3
"""FPF 上位k件の重なり（fpf_top）．k は 1 から順に全部出す

usage:
  python tools/RQ1/metrics/fpf_top/run.py -e embeddings/bewt -c corpus/bewt/json \
      --corpus-name bewt --apps bludit claroline expresscart joomla kanboard mantisbt mediawiki prestashop
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _harness import run

SHOW_KS = (5, 10, 20)


def prepare(X):
    D = 1.0 - X @ X.T
    n = len(X)
    # 開始点は他との距離の総和が最大の1件
    sel = [int(np.argmax(D.sum(axis=1)))]
    mind = D[sel[0]].copy()
    while len(sel) < n:
        mind[sel] = -np.inf
        nxt = int(np.argmax(mind))
        sel.append(nxt)
        # 選んだ集合との最小距離を更新し，最も遠い1件を次に選ぶ
        mind = np.minimum(mind, D[nxt])
    return sel


def permute(order, p):
    inv = np.argsort(p)
    return [int(inv[i]) for i in order]


def compare(ox, oy):
    sx, sy, hit, out = set(), set(), 0, []
    for u, v in zip(ox, oy):
        sx.add(u)
        if u in sy:
            hit += 1
        sy.add(v)
        if v in sx:
            hit += 1
        out.append(hit)
    return out


def contrib(hits, n, k):
    # アプリ単位のマクロ平均なので分母は1
    return hits[k - 1] / k, 1.0


def key_list(ns):
    return tuple(range(1, min(ns.values()) + 1))


def show_keys(keys):
    return tuple(k for k in SHOW_KS if k in keys)


if __name__ == "__main__":
    run("fpf_top", prepare, compare=compare, permute=permute,
        contrib=contrib, key_list=key_list, show_keys=show_keys)
