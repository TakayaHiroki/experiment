#!/usr/bin/env python3
"""三つ組の一致（triplet_agree）

usage:
  python tools/RQ1/metrics/triplet_agree/run.py -e embeddings/bewt -c corpus/bewt/json \
      --corpus-name bewt --apps bludit claroline expresscart joomla kanboard mantisbt mediawiki prestashop
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _harness import contrib_macro, run, similarity


def prepare(X):
    return similarity(X)


def permute(S, p):
    return S[np.ix_(p, p)]


def compare(Sx, Sy):
    n = Sx.shape[0]
    idx = np.arange(n)
    # 基準 a を除いた2件の組み合わせ
    iu = np.triu_indices(n - 1, 1)
    conc = tot = 0
    for a in range(n):
        keep = idx != a
        u, v = Sx[a][keep], Sy[a][keep]
        du = np.sign(u[:, None] - u[None, :])[iu]
        dv = np.sign(v[:, None] - v[None, :])[iu]
        # どちらかが同点の三つ組は数えない
        ok = (du != 0) & (dv != 0)
        conc += int((du[ok] == dv[ok]).sum())
        tot += int(ok.sum())
    return conc / tot if tot else float("nan")


if __name__ == "__main__":
    run("triplet_agree", prepare, compare=compare, permute=permute, contrib=contrib_macro)
