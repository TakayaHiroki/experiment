#!/usr/bin/env python3
"""上位k近傍の重なり（knn_jaccard）．k は 1 から順に全部出す

usage:
  python tools/RQ1/metrics/knn_jaccard/run.py -e embeddings/bewt -c corpus/bewt/json \\
      --corpus-name bewt --apps bludit claroline expresscart joomla kanboard mantisbt mediawiki prestashop
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _harness import contrib_micro, run

SHOW_KS = (1, 5, 10, 20)


def prepare(X):
    S = X @ X.T
    # 自分自身を近傍にしない
    np.fill_diagonal(S, -np.inf)
    # 同点は行番号の小さい方を先にする
    order = np.argsort(-S, axis=1, kind="stable")
    R = np.empty_like(order)
    np.put_along_axis(R, order, np.arange(S.shape[1]), axis=1)
    return R


def permute(R, p):
    return R[np.ix_(p, p)]


def compare(Rx, Ry):
    n = Rx.shape[0]
    M = np.maximum(Rx, Ry)
    ks = np.arange(1, n)
    h = (M[:, :, None] < ks).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        # Jaccard．どちらも上位k件なので和集合は 2k-共通数
        J = h / (2 * ks - h)
    return J.sum(axis=0).tolist()


def contrib(totals, n, k):
    return totals[k - 1], n


def key_list(ns):
    return tuple(range(1, min(ns.values())))


def show_keys(keys):
    return tuple(k for k in SHOW_KS if k in keys)


if __name__ == "__main__":
    run("knn_jaccard", prepare, compare=compare, permute=permute,
        contrib=contrib, key_list=key_list, show_keys=show_keys)
