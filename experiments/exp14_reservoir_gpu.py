"""Experiment 14 - reservoir computing at scale on the GPU (no LLM).

Uses the GPU to push reservoir sizes far beyond what the CPU study (exp1, exp9,
exp11) reached, measuring two core reservoir-computing quantities versus size:

* **Short-term memory capacity** MC(N) — how much of a white-noise input the
  reservoir can linearly reconstruct at increasing delays (Jaeger, 2002); and
* **Mackey-Glass one-step NRMSE** vs N — accuracy on a chaotic task.

Self-contained (torch + numpy). Writes ~/esnfed_gpu_out/reservoir_gpu.json.

    python exp14_reservoir_gpu.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

OUT = Path.home() / "esnfed_gpu_out"
OUT.mkdir(exist_ok=True)
DEV = "cuda"
DT = torch.float32


def spectral_radius(W, iters=80):
    v = torch.randn(W.shape[0], device=W.device, dtype=W.dtype)
    v /= v.norm() + 1e-12
    s = torch.tensor(0.0, device=W.device)
    for _ in range(iters):
        w = W @ v
        s = w.norm()
        v = w / (s + 1e-12)
    return float(s)


def make_esn(N, sr, leak, density=0.1, d_in=1, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    W = torch.randn(N, N, generator=g, device=DEV, dtype=DT)
    W *= (torch.rand(N, N, generator=g, device=DEV) < density).to(DT)
    s = spectral_radius(W)
    if s > 0:
        W *= sr / s
    Win = torch.randn(N, 1 + d_in, generator=g, device=DEV, dtype=DT)
    return W, Win, leak


def harvest(W, Win, leak, u):
    N, T, din = W.shape[0], u.shape[0], u.shape[1]
    x = torch.zeros(N, device=DEV, dtype=DT)
    Z = torch.empty(T, 1 + din + N, device=DEV, dtype=DT)
    one = torch.ones(1, device=DEV, dtype=DT)
    for t in range(T):
        ub = torch.cat([one, u[t]])
        x = (1 - leak) * x + leak * torch.tanh(Win @ ub + W @ x)
        Z[t, 0] = 1.0
        Z[t, 1:1 + din] = u[t]
        Z[t, 1 + din:] = x
    return Z


def ridge(Z, Y, beta=1e-6):
    D = Z.shape[1]
    A = Z.T @ Z + beta * torch.eye(D, device=DEV, dtype=DT)
    return torch.linalg.solve(A, Z.T @ Y)


def memory_capacity(N, T=4000, kmax=120, washout=200):
    g = torch.Generator(device=DEV).manual_seed(1)
    u = torch.rand(T, 1, generator=g, device=DEV, dtype=DT) * 1.6 - 0.8
    W, Win, leak = make_esn(N, sr=0.95, leak=1.0, seed=0)
    t0 = time.time()
    Z = harvest(W, Win, leak, u)
    split = int(T * 0.7)
    MC = 0.0
    for k in range(1, kmax + 1):
        y = torch.zeros(T, 1, device=DEV, dtype=DT)
        y[k:, 0] = u[:-k, 0]
        Wout = ridge(Z[washout:split], y[washout:split])
        pred = (Z[split:] @ Wout)[:, 0]
        yt = y[split:, 0]
        vx, vy = pred - pred.mean(), yt - yt.mean()
        corr = (vx * vy).sum() / (vx.norm() * vy.norm() + 1e-12)
        MC += float(corr ** 2)
    return MC, time.time() - t0


def mackey_glass(n, tau=17, beta=0.2, gamma=0.1, power=10, discard=250, seed=0):
    rng = np.random.default_rng(seed)
    h = tau + 1
    total = n + discard + 1
    x = np.empty(total + h)
    x[:h] = 1.2 + 0.2 * (rng.random(h) - 0.5)
    for t in range(h, total + h):
        x[t] = x[t - 1] + beta * x[t - tau] / (1 + x[t - tau] ** power) - gamma * x[t - 1]
    return x[h + discard:]


def mg_nrmse(N, washout=100):
    s = mackey_glass(3000)
    u = torch.tensor(s[:-1].reshape(-1, 1), device=DEV, dtype=DT)
    y = torch.tensor(s[1:].reshape(-1, 1), device=DEV, dtype=DT)
    W, Win, leak = make_esn(N, sr=0.95, leak=0.3, seed=0)
    Z = harvest(W, Win, leak, u)
    split = int(len(u) * 0.7)
    Wout = ridge(Z[washout:split], y[washout:split])
    pred = Z[split:] @ Wout
    yt = y[split:]
    return float(torch.sqrt(((pred - yt) ** 2).mean() / (yt.var() + 1e-12)))


def main():
    print("[exp14] reservoir computing at scale on GPU")
    print("  GPU:", torch.cuda.get_device_name(0))
    rep = {"gpu": torch.cuda.get_device_name(0), "memory_capacity": [], "mackey_glass": []}
    for N in [500, 1000, 2000, 4000]:
        mc, dt = memory_capacity(N)
        rep["memory_capacity"].append(dict(N=N, MC=round(mc, 2), harvest_s=round(dt, 2)))
        print(f"  MC: N={N:5d}  MC={mc:7.2f}  (harvest {dt:.1f}s)")
        torch.cuda.empty_cache()
    for N in [500, 1000, 2000, 4000]:
        e = mg_nrmse(N)
        rep["mackey_glass"].append(dict(N=N, nrmse=round(e, 5)))
        print(f"  MG: N={N:5d}  NRMSE={e:.5f}")
        torch.cuda.empty_cache()
    (OUT / "reservoir_gpu.json").write_text(json.dumps(rep, indent=2))
    print(f"[exp14] saved -> {OUT / 'reservoir_gpu.json'}")


if __name__ == "__main__":
    main()
