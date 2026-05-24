"""Collect hardware + GPU-acceleration data for the TFG.

Writes ~/esnfed_gpu_out/gpu_report.json with:
* GPU / CPU / memory / disk info,
* an achievable bf16 matmul-TFLOPS measurement, and
* a reservoir-harvest throughput benchmark **on the GPU** across large reservoir
  sizes (extends the CPU acceleration study, exp9, to sizes a GPU enables).

    python gpu_report.py
"""
from __future__ import annotations

import json
import platform
import subprocess
import time
from pathlib import Path

import torch

OUT = Path.home() / "esnfed_gpu_out"
OUT.mkdir(exist_ok=True)


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=30).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"(err: {e})"


def hardware():
    g = {}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        g = dict(name=p.name, vram_gb=round(p.total_memory / 1e9, 1),
                 sm_count=p.multi_processor_count,
                 capability=f"{p.major}.{p.minor}",
                 torch=torch.__version__, cuda=torch.version.cuda)
    return dict(gpu=g, platform=platform.platform(),
                nvidia_smi=sh("nvidia-smi"),
                lscpu=sh("lscpu | grep -E 'Model name|Socket|Core|Thread|MHz' | head"),
                mem=sh("free -h | head -2"), disk=sh("df -h / | tail -1"))


def matmul_tflops(n=8192, dtype=torch.bfloat16, iters=30):
    a = torch.randn(n, n, device="cuda", dtype=dtype)
    b = torch.randn(n, n, device="cuda", dtype=dtype)
    for _ in range(3):
        _ = a @ b
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(iters):
        c = a @ b
    torch.cuda.synchronize()
    dt = (time.time() - t) / iters
    return round(2 * n ** 3 / dt / 1e12, 1)


def reservoir_harvest_gpu(sizes=(500, 1000, 2000, 4000, 8000, 16000), T=2000,
                          density=0.1, dtype=torch.float32):
    """Leaky-ESN harvest on the GPU; report steps/second vs reservoir size."""
    res = []
    for N in sizes:
        try:
            W = torch.randn(N, N, device="cuda", dtype=dtype)
            W *= (torch.rand(N, N, device="cuda") < density).to(dtype)
            Win = torch.randn(N, 2, device="cuda", dtype=dtype)
            u = torch.randn(T, device="cuda", dtype=dtype)
            x = torch.zeros(N, device="cuda", dtype=dtype)
            a = 0.5
            for _ in range(5):  # warm-up
                x = (1 - a) * x + a * torch.tanh(W @ x)
            torch.cuda.synchronize()
            t0 = time.time()
            for t in range(T):
                ub = torch.stack([torch.ones((), device="cuda", dtype=dtype), u[t]])
                pre = Win @ ub + W @ x
                x = (1 - a) * x + a * torch.tanh(pre)
            torch.cuda.synchronize()
            dt = time.time() - t0
            row = dict(N=N, steps_per_sec=round(T / dt, 1),
                       vram_gb=round(torch.cuda.memory_allocated() / 1e9, 2))
            print(f"  N={N:6d}: {T / dt:8.0f} steps/s  ({row['vram_gb']} GB)")
            res.append(row)
            del W, Win
            torch.cuda.empty_cache()
        except RuntimeError as e:  # OOM at very large N
            print(f"  N={N}: skipped ({e})")
            res.append(dict(N=N, error=str(e)[:80]))
    return res


def main():
    print("[gpu_report] collecting hardware + GPU benchmarks")
    rep = dict(hardware=hardware())
    print("  GPU:", rep["hardware"]["gpu"].get("name"))
    rep["matmul_bf16_tflops_n8192"] = matmul_tflops()
    print("  bf16 matmul TFLOPS:", rep["matmul_bf16_tflops_n8192"])
    rep["reservoir_harvest_gpu"] = reservoir_harvest_gpu()
    (OUT / "gpu_report.json").write_text(json.dumps(rep, indent=2))
    (OUT / "gpu_info.txt").write_text(rep["hardware"]["nvidia_smi"])
    print(f"[gpu_report] saved -> {OUT / 'gpu_report.json'}")


if __name__ == "__main__":
    main()
