#!/usr/bin/env python3
"""最近傍の一致（nn_agree）．RQ1 の主指標

usage:
  python tools/RQ1/metrics/nn_agree/run.py -e embeddings/bewt -c corpus/bewt/json \
      --corpus-name bewt --apps bludit claroline expresscart joomla kanboard mantisbt mediawiki prestashop
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _harness import contrib_micro, run


def prepare(X):
    S = X @ X.T
    # 自分自身を最近傍にしない
    np.fill_diagonal(S, -np.inf)
    return S.argmax(axis=1)


def permute(nn, p):
    # テスト i を元の p[i] と読み替える（陰性対照）
    return np.argsort(p)[nn[p]]


def compare(nx, ny):
    return int((nx == ny).sum())


if __name__ == "__main__":
    run("nn_agree", prepare, compare=compare, permute=permute, contrib=contrib_micro)
