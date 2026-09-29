#!/usr/bin/env python3
"""ベクトル化の計測環境（ライブラリの版・スレッド設定・モデルの版）を JSON で出す

usage:
  python tools/RQ1/embed_env.py --models Qwen/Qwen3-Embedding-0.6B BAAI/bge-base-en-v1.5
  python tools/RQ1/embed_env.py --models Qwen/Qwen3-Embedding-0.6B --warm
"""
import argparse, json, os, platform, sys, time
import numpy, sklearn, torch, transformers, sentence_transformers, huggingface_hub, safetensors
from huggingface_hub import snapshot_download

ENV_KEYS = ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "KMP_AFFINITY", "KMP_BLOCKTIME",
            "EMB_BATCH", "HF_HUB_OFFLINE", "CUDA_VISIBLE_DEVICES"]

ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="*", default=[])
ap.add_argument("--warm", action="store_true")
a = ap.parse_args()

models = {}
warm_bytes = 0
_buf = memoryview(bytearray(64 << 20))
t0, c0 = time.perf_counter(), time.process_time()
for m in a.models:
    path = snapshot_download(m, local_files_only=True)
    files = sorted(os.listdir(path))
    size = 0
    for fn in files:
        p = os.path.join(path, fn)
        if os.path.isfile(p):
            size += os.path.getsize(p)
            if a.warm:
                with open(p, "rb") as f:
                    while f.readinto(_buf):
                        pass
    models[m] = {"revision": os.path.basename(path), "bytes": size}
    warm_bytes += size

if a.warm:
    print(f"warmed {len(models)} model(s)", file=sys.stderr)
    print(f"[WARM] t_warm={time.perf_counter() - t0:.2f} c_warm={time.process_time() - c0:.2f} "
          f"warm_bytes={warm_bytes}", file=sys.stderr)
    sys.exit(0)

gpus = []
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        gpus.append({"name": p.name, "total_memory_mb": round(p.total_memory / (1 << 20)),
                     "capability": f"{p.major}.{p.minor}"})

info = {
    "python": sys.version.split()[0],
    "platform": platform.platform(),
    "versions": {mod.__name__: mod.__version__ for mod in
                 (torch, transformers, sentence_transformers, huggingface_hub, safetensors, numpy, sklearn)},
    "torch_threads": torch.get_num_threads(),
    "torch_parallel_info": torch.__config__.parallel_info(),
    "cuda_available": torch.cuda.is_available(),
    "cuda_version": torch.version.cuda,
    "gpus": gpus,
    "env": {k: os.environ.get(k) for k in ENV_KEYS},
    "models": models,
}
json.dump(info, sys.stdout, ensure_ascii=False, indent=2)
