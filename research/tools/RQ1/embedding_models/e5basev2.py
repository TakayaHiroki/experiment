#!/usr/bin/env python3
"""intfloat/e5-base-v2 でベクトル化する

usage:
  python tools/RQ1/embedding_models/e5basev2.py -i corpus/bewt/json/kanboard_full.json -o embeddings/bewt/e5-base-v2/kanboard_full.json
"""
import argparse, json, os, sys, time
import torch
from sentence_transformers import SentenceTransformer

MODEL = "intfloat/e5-base-v2"
PREFIX = "query: "
TRUST_REMOTE_CODE = False
DTYPE = torch.float32
BATCH = os.environ.get("EMB_BATCH", "32")
if not BATCH.isdigit() or int(BATCH) < 1:
    raise SystemExit(f"EMB_BATCH は1以上の整数にすること（{BATCH!r}）")
BATCH = int(BATCH)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _measure

# 演算内の並列は OMP_NUM_THREADS に従う．演算間の並列は使わない
torch.set_num_interop_threads(1)


ap = argparse.ArgumentParser()
ap.add_argument("--input", "-i", required=True)
ap.add_argument("--output", "-o", required=True)
ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                help="既定 cpu．cuda は GPU のある環境だけ（CPU の結果と混ぜない）")
a = ap.parse_args()

DEVICE = _measure.resolve_device(a.device)

if not os.path.isfile(a.input):
    raise SystemExit(f"{a.input} が無い")
try:
    with open(a.input, encoding="utf-8") as f:
        tests = json.load(f)
    titles = [t["title"] for t in tests]
    # 前置きは本文の直前に付ける．切り捨ての判定にも含まれる
    texts = [PREFIX + t["text_for_embedding"] for t in tests]
except (json.JSONDecodeError, UnicodeDecodeError, TypeError, KeyError) as e:
    raise SystemExit(f"{a.input} を読めない（title と text_for_embedding を持つ JSON の配列が要る）: {e!r}")
if not tests:
    raise SystemExit(f"{a.input} にテストが1件も無い")
os.makedirs(os.path.dirname(a.output) or ".", exist_ok=True)

print(f"Loaded {len(texts)} test cases. Encoding with {MODEL}...")

c0, t0 = time.process_time(), time.perf_counter()
model = SentenceTransformer(MODEL, trust_remote_code=TRUST_REMOTE_CODE, model_kwargs={"dtype": DTYPE}, device=DEVICE)
t_load, c_load = time.perf_counter() - t0, time.process_time() - c0
dtype = next(model.parameters()).dtype
# 指定した精度で読み込めたか確かめる
if dtype != DTYPE:
    raise SystemExit(f"{MODEL} が {DTYPE} ではなく {dtype} で読み込まれた")
n_params = _measure.count_params(model)

lengths = [len(ids) for ids in model.tokenizer(texts)["input_ids"]]
# 上限を超えた入力は後ろが切り捨てられる．件数を記録する
truncated = [i for i, n in enumerate(lengths) if n > model.max_seq_length]
if truncated:
    print(f"[WARN] {len(truncated)} 件が上限 {model.max_seq_length} トークンを超え，後ろを切り捨てた: index {truncated}")

_measure.sync(DEVICE)
cf, tf = time.process_time(), time.perf_counter()
# 初回呼び出しの準備時間を t_encode に含めない
model.encode(texts[:1], batch_size=1, show_progress_bar=False, normalize_embeddings=True)
_measure.sync(DEVICE)
t_first, c_first = time.perf_counter() - tf, time.process_time() - cf

c1, t1 = time.process_time(), time.perf_counter()
emb = model.encode(texts, batch_size=BATCH, show_progress_bar=True, normalize_embeddings=True)
_measure.sync(DEVICE)
t_encode, c_encode = time.perf_counter() - t1, time.process_time() - c1

out = [{"index": i, "title": titles[i], "embedding": emb[i].tolist()} for i in range(len(titles))]
tmp = a.output + ".tmp"
with open(tmp, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False)
# 途中で止まっても壊れたファイルを残さない
os.replace(tmp, a.output)

print(_measure.timing(
    t_load=f"{t_load:.2f}", t_first=f"{t_first:.2f}", t_encode=f"{t_encode:.2f}",
    c_load=f"{c_load:.2f}", c_first=f"{c_first:.2f}", c_encode=f"{c_encode:.2f}",
    n=len(texts), batch=BATCH, truncated=len(truncated),
    threads=torch.get_num_threads(), interop=torch.get_num_interop_threads(),
    dtype=str(dtype).replace("torch.", ""),
    device=DEVICE, params=n_params,
    peak_rss_mb=_measure.peak_rss_mb(), peak_gpu_mb=_measure.peak_gpu_mb(DEVICE)))
print(f"Embeddings saved to {a.output} ({len(out)} entries, dim={emb.shape[1]})")
