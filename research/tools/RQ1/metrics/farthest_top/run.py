#!/usr/bin/env python3
"""最遠点優先走査で選んだ上位k件の重なり（farthest_top）．k は 1 から K_MAX まで出す

最遠点優先走査（farthest-first traversal; Gonzalez 1985．k-center 問題の貪欲法）でアプリの全テストを並べ，
2つの表現で上位k件がどれだけ重なるかを測る．標準の手順とは次の3点が違う．
- 開始点: 標準の手順では任意の1件だが，ここでは他との距離の総和が最大の1件にする．
  乱数を使わず同じ入力から同じ順を出すため．開始点の引き方の揺れが表現どうしの違いに混ざらない
- クラスタリングはしない: k-center として使うときは選んだ k 件を中心に各テストを割り当てるが，ここでは選ぶ順だけを使う．
  測りたいのは「表現を変えると先に選ばれるテストが変わるか」で，割り当ては使わないため
- 2近似の保証は前提にしない: 距離は 1 − コサイン類似度で，これは三角不等式を満たさない
  （例: x=(1,0), y=(1,1)/√2, z=(0,1) で d(x,y)+d(y,z)≈0.59 < d(x,z)=1）．
  なので Gonzalez の2近似（k 件の半径が最適の2倍以内）は成り立つとは限らない．この指標は選ぶ順の一致だけを見るので保証は要らない

usage:
  python tools/RQ1/metrics/farthest_top/run.py -e embeddings/bewt -c corpus/bewt/json \
      --corpus-name bewt --apps bludit claroline expresscart joomla kanboard mantisbt mediawiki prestashop
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _harness import run, similarity

# k がテスト数に近いと表現と関係なく重なるので，最小のアプリ（23件）の半分より手前で止める
K_MAX = 10
SHOW_KS = (5, 10)


def prepare(X):
    D = 1.0 - similarity(X)
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
    return tuple(range(1, min(K_MAX, *ns.values()) + 1))


def show_keys(keys):
    return tuple(k for k in SHOW_KS if k in keys)


if __name__ == "__main__":
    run("farthest_top", prepare, compare=compare, permute=permute,
        contrib=contrib, key_list=key_list, show_keys=show_keys)
