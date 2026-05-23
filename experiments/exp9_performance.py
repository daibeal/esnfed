"""Experiment 9 - Acceleration benchmark for the harvest hot path.

Measures the wall-clock time of the reservoir state-harvest across reservoir
sizes for the four code paths: pure-NumPy float64 (the default), Numba-JIT
float64, NumPy float32, and a SciPy sparse reservoir. Produces a log-log figure
and a LaTeX table for the thesis. Numbers are machine-dependent; they are meant
to show the *regimes* in which each accelerator helps.

Needs the optional accelerators: ``pip install "esnfed[fast]"``.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

import common
from esnfed import EchoStateNetwork, topologies

SIZES = [50, 100, 200, 400, 800, 1500]
T = 3000        # timesteps per harvest
REPEATS = 5

try:
    import numba  # noqa: F401
    _HAVE_NUMBA = True
except ImportError:
    _HAVE_NUMBA = False
try:
    import scipy.sparse  # noqa: F401
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False


def _harvest_ms(N, **kw):
    W = topologies.random_reservoir(N, density=0.1, rng=common.MASTER_SEED)
    u = np.random.default_rng(0).uniform(0.0, 0.5, size=(T, 1))
    esn = EchoStateNetwork(1, 1, W, spectral_radius=0.9, washout=100, seed=0, **kw)
    esn.harvest(u)  # warm-up (also triggers Numba JIT compilation)
    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        esn.harvest(u)
        times.append(time.perf_counter() - t0)
    return min(times) * 1e3  # best-of-N: robust to background noise


def run():
    rows = []
    for N in SIZES:
        r = {"N": N, "numpy_f64": _harvest_ms(N, use_numba=False)}
        r["numba_f64"] = _harvest_ms(N, use_numba=True) if _HAVE_NUMBA else np.nan
        r["numpy_f32"] = _harvest_ms(N, use_numba=False, dtype=np.float32)
        r["sparse"] = _harvest_ms(N, sparse=True) if _HAVE_SCIPY else np.nan
        rows.append(r)
        print(f"  N={N:5d}  numpy64={r['numpy_f64']:8.1f}  numba={r['numba_f64']:8.1f}"
              f"  float32={r['numpy_f32']:8.1f}  sparse={r['sparse']:8.1f}  (ms)")
    return pd.DataFrame(rows)


def plot(df):
    common.set_style()
    fig, ax = common.plt.subplots(figsize=(6.4, 4.2))
    series = [
        ("numpy_f64", "NumPy float64 (default)", "#000000", "o"),
        ("numba_f64", "Numba float64", "#4C72B0", "s"),
        ("numpy_f32", "NumPy float32", "#55A868", "^"),
        ("sparse", "Sparse (CSR)", "#C44E52", "D"),
    ]
    for col, label, color, marker in series:
        if df[col].notna().any():
            ax.plot(df["N"], df[col], marker=marker, color=color, label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Reservoir size $N$")
    ax.set_ylabel(f"Harvest time for {T} steps (ms)")
    ax.set_title("Reservoir harvest: acceleration vs. size")
    ax.legend()
    common.save_figure(fig, "exp9_performance")


def main():
    print("[exp9] acceleration benchmark "
          f"(numba={_HAVE_NUMBA}, scipy={_HAVE_SCIPY})")
    df = run()
    common.save_table(df, "exp9_performance")

    # Compact LaTeX table: times (ms) + best speedup vs the NumPy default.
    tex = df.copy()
    best = tex[["numba_f64", "numpy_f32", "sparse"]].min(axis=1)
    tex["speedup"] = tex["numpy_f64"] / best
    out = pd.DataFrame({
        "$N$": tex["N"].astype(int),
        "NumPy f64": tex["numpy_f64"],
        "Numba": tex["numba_f64"],
        "float32": tex["numpy_f32"],
        "Sparse": tex["sparse"],
        "Best speedup": tex["speedup"].map(lambda x: f"{x:.1f}x"),
    })
    common.save_latex_table(
        out, "exp9_performance",
        caption="Reservoir harvest time (ms, " + str(T) + " steps, density 0.1) by "
                "code path and reservoir size, and the best speedup over the "
                "NumPy default. Machine-dependent; shown to illustrate the regimes.",
        label="tab:exp9-performance", float_format="%.0f")
    plot(df)
    print("[exp9] done")


if __name__ == "__main__":
    main()
