#!/usr/bin/env python3
"""全ペア類似度の順位相関（dist_rho，RSA 相関）と Mantel 検定

usage:
  python tools/RQ1/metrics/dist_rho/run.py -e embeddings/bewt -c corpus/bewt/json \
      --corpus-name bewt --apps bludit claroline expresscart joomla kanboard mantisbt mediawiki prestashop
"""
import os, sys
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _harness import contrib_macro, run, similarity

# p の下限を 1/(perms+1) にするための回数
PERMS = 1999


def prepare(X):
    S = similarity(X)
    n = S.shape[0]
    iu = np.triu_indices(n, 1)
    R = np.zeros_like(S)
    # 上三角の類似度を順位に直す
    r = stats.rankdata(S[iu])
    R[iu] = r
    R.T[iu] = r
    return R


def rho_and_null(Rx, Ry, n, perms, rng):
    iu = np.triu_indices(n, 1)
    a = Rx[iu] - Rx[iu].mean()
    b = Ry[iu]
    denom = np.linalg.norm(a) * np.linalg.norm(b - b.mean())
    obs = float(a @ b / denom)
    # 並べ替えをまとめて展開する件数．メモリとの折り合い
    step = max(1, 2_000_000 // len(b))
    null = []
    for s in range(0, perms, step):
        pi = np.array([rng.permutation(n) for _ in range(min(step, perms - s))])
        null.append((Ry[pi[:, iu[0]], pi[:, iu[1]]] @ a) / denom)
    return obs, np.concatenate(null)


def mantel(obs, nulls, common, perms):
    ps = [(1 + int((nulls[app] >= obs[app]).sum())) / (perms + 1) for app in common]
    # アプリごとの p を Fisher の方法で統合する
    p_fisher = float(stats.chi2.sf(-2.0 * float(np.sum(np.log(ps))), 2 * len(ps)))
    return {"p_fisher": f"{p_fisher:.3e}"}


if __name__ == "__main__":
    run("dist_rho", prepare, contrib=contrib_macro, perms_default=PERMS,
        pair_fn=rho_and_null, extra_cols=mantel,
        show_cols=(("p(統合)", 11, lambda r: r["p_fisher"]),),
        footer=lambda a: f"p の下限は 1/(perms+1) = {1.0 / (a.perms + 1):.5f}", bounds=(-1.0, 1.0))
