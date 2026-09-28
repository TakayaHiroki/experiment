#!/usr/bin/env python3
"""Procrustes の残差（procrustes_disp）

usage:
  python tools/RQ1/metrics/procrustes_disp/run.py -e embeddings/bewt -c corpus/bewt/json \
      --corpus-name bewt --apps bludit claroline expresscart joomla kanboard mantisbt mediawiki prestashop
"""
import os, sys
import numpy as np
from scipy.spatial import procrustes

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _harness import contrib_macro, run


def prepare(X):
    n = X.shape[0]
    r = min(n - 1, X.shape[1])
    Xc = X - X.mean(0)
    # 中心化して主成分の座標にそろえる
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    return U[:, :r] * S[:r]


def permute(A, p):
    return A[p]


def compare(A, B):
    r = max(A.shape[1], B.shape[1])
    if r < 2:
        return float("nan")
    # 次元の少ない側に0列を足して形をそろえる
    A = np.pad(A, ((0, 0), (0, r - A.shape[1])))
    B = np.pad(B, ((0, 0), (0, r - B.shape[1])))
    return float(procrustes(A, B)[2])


if __name__ == "__main__":
    run("procrustes_disp", prepare, compare=compare, permute=permute, contrib=contrib_macro)
