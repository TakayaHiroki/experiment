#!/usr/bin/env python3
"""ベクトル化の計測（時間・CPU時間・メモリのピーク・規模・デバイス）"""
import ctypes
import sys


def resolve_device(name):
    import torch
    if name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda を指定したが CUDA が使えない（torch が CPU 版か，GPU が無い）")
    return name


def sync(device):
    if device == "cuda":
        import torch
        torch.cuda.synchronize()


def count_params(model):
    return sum(p.numel() for p in model.parameters())


class _MemCounters(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_uint32), ("PageFaultCount", ctypes.c_uint32),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]


def peak_commit_mb():
    if sys.platform != "win32":
        return -1
    cur = ctypes.windll.kernel32.GetCurrentProcess
    cur.restype = ctypes.c_void_p
    cur.argtypes = []
    c = _MemCounters()
    c.cb = ctypes.sizeof(_MemCounters)
    for dll, name in ((ctypes.windll.psapi, "GetProcessMemoryInfo"),
                      (ctypes.windll.kernel32, "K32GetProcessMemoryInfo")):
        try:
            fn = getattr(dll, name)
        except AttributeError:
            continue
        fn.restype = ctypes.c_int
        fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(_MemCounters), ctypes.c_uint32]
        if fn(cur(), ctypes.byref(c), c.cb):
            return round(c.PeakPagefileUsage / (1 << 20))
    return -1


def peak_gpu_mb(device):
    if device != "cuda":
        return 0
    import torch
    return round(torch.cuda.max_memory_allocated() / (1 << 20))


def timing(**fields):
    return "[TIMING] " + " ".join(f"{k}={v}" for k, v in fields.items())
