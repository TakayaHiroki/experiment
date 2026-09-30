#!/usr/bin/env python3
"""dunzhang/stella_en_1.5B_v5 でベクトル化する

usage:
  python tools/RQ1/embedding_models/stellaen1.5bv5.py -i corpus/bewt/json/kanboard_full.json -o embeddings/bewt/stella-en-1.5b-v5/kanboard_full.json --prompt none
  python tools/RQ1/embedding_models/stellaen1.5bv5.py -i corpus/bewt/json/kanboard_full.json -o embeddings/bewt/stella-en-1.5b-v5+sts/kanboard_full.json --prompt sts
"""
import argparse, json, os, sys, time
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from tokenizers import processors
from sentence_transformers import SentenceTransformer

MODEL = "dunzhang/stella_en_1.5B_v5"
PREFIXES = {"none": "", "sts": "Instruct: Retrieve semantically similar text.\nQuery: "}
TRUST_REMOTE_CODE = True
DTYPE = torch.float32
BATCH = os.environ.get("EMB_BATCH", "32")
if not BATCH.isdigit() or int(BATCH) < 1:
    raise SystemExit(f"EMB_BATCH は1以上の整数にすること（{BATCH!r}）")
BATCH = int(BATCH)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _measure

torch.set_num_interop_threads(1)


@torch.no_grad()
def repair_remote_model(model):
    auto = model[0].auto_model
    params = auto.state_dict()
    snap = os.path.dirname(hf_hub_download(MODEL, "config.json"))
    done = set()
    for fname in sorted(os.listdir(snap)):
        if not fname.endswith(".safetensors"):
            continue
        with safe_open(os.path.join(snap, fname), "pt") as f:
            for key in f.keys():
                name = key[len("model."):] if key.startswith("model.") else key
                if name in params:
                    params[name].copy_(f.get_tensor(key).to(params[name].dtype))
                    done.add(name)
    missing = sorted(set(params) - done)
    if missing:
        raise SystemExit(f"{MODEL} の保存ファイルに無い重みがある: {missing[:3]}")

    cfg = auto.config
    base = (getattr(cfg, "rope_parameters", None) or {}).get("rope_theta") or cfg.rope_theta
    seq_len = min(model.max_seq_length, cfg.max_position_embeddings)
    for layer in auto.layers:
        rot = layer.self_attn.rotary_emb
        layer.self_attn.rope_theta = base
        rot.base = base
        rot.inv_freq = 1.0 / (base ** (torch.arange(0, rot.dim, 2, dtype=torch.int64).float() / rot.dim))
        rot._set_cos_sin_cache(seq_len=seq_len, device=rot.inv_freq.device, dtype=torch.get_default_dtype())

    tok = model.tokenizer
    eos = tok.eos_token
    if tok("a")["input_ids"][-1] != tok.eos_token_id:
        tok.backend_tokenizer.post_processor = processors.Sequence([
            tok.backend_tokenizer.post_processor,
            processors.TemplateProcessing(single=f"$A {eos}", pair=f"$A {eos} $B:1 {eos}:1",
                                          special_tokens=[(eos, tok.eos_token_id)]),
        ])
    if tok("a")["input_ids"][-1] != tok.eos_token_id:
        raise SystemExit(f"{MODEL} の入力の末尾に {eos} を付けられない")


ap = argparse.ArgumentParser()
ap.add_argument("--input", "-i", required=True)
ap.add_argument("--output", "-o", required=True)
ap.add_argument("--prompt", required=True, choices=sorted(PREFIXES))
ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                help="既定 cpu．cuda は GPU のある環境だけ（CPU の結果と混ぜない）")
a = ap.parse_args()
PREFIX = PREFIXES[a.prompt]

DEVICE = _measure.resolve_device(a.device)

if not os.path.isfile(a.input):
    raise SystemExit(f"{a.input} が無い")
try:
    with open(a.input, encoding="utf-8") as f:
        tests = json.load(f)
    titles = [t["title"] for t in tests]
    texts = [PREFIX + t["text_for_embedding"] for t in tests]
except (json.JSONDecodeError, UnicodeDecodeError, TypeError, KeyError) as e:
    raise SystemExit(f"{a.input} を読めない（title と text_for_embedding を持つ JSON の配列が要る）: {e!r}")
if not tests:
    raise SystemExit(f"{a.input} にテストが1件も無い")
os.makedirs(os.path.dirname(a.output) or ".", exist_ok=True)

print(f"Loaded {len(texts)} test cases. Encoding with {MODEL}...")

c0, t0 = time.process_time(), time.perf_counter()
model = SentenceTransformer(MODEL, trust_remote_code=TRUST_REMOTE_CODE, model_kwargs={"dtype": DTYPE}, device=DEVICE)
repair_remote_model(model)
t_load, c_load = time.perf_counter() - t0, time.process_time() - c0
dtype = next(model.parameters()).dtype
if dtype != DTYPE:
    raise SystemExit(f"{MODEL} が {DTYPE} ではなく {dtype} で読み込まれた")
n_params = _measure.count_params(model)

lengths = [len(ids) for ids in model.tokenizer(texts)["input_ids"]]
truncated = [i for i, n in enumerate(lengths) if n > model.max_seq_length]
if truncated:
    print(f"[WARN] {len(truncated)} 件が上限 {model.max_seq_length} トークンを超え，後ろを切り捨てた: index {truncated}")

_measure.sync(DEVICE)
cf, tf = time.process_time(), time.perf_counter()
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
os.replace(tmp, a.output)

print(_measure.timing(
    t_load=f"{t_load:.2f}", t_first=f"{t_first:.2f}", t_encode=f"{t_encode:.2f}",
    c_load=f"{c_load:.2f}", c_first=f"{c_first:.2f}", c_encode=f"{c_encode:.2f}",
    n=len(texts), batch=BATCH, truncated=len(truncated),
    threads=torch.get_num_threads(), interop=torch.get_num_interop_threads(),
    dtype=str(dtype).replace("torch.", ""),
    prompt=a.prompt,
    device=DEVICE, params=n_params,
    peak_commit_mb=_measure.peak_commit_mb(), peak_gpu_mb=_measure.peak_gpu_mb(DEVICE)))
print(f"Embeddings saved to {a.output} ({len(out)} entries, dim={emb.shape[1]})")
