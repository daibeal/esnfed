"""Experiment 1 - Single-ESN hyperparameter sweep.

Characterises a centralised ESN on NARMA-10 as a function of the two most
influential hyperparameters: the spectral radius and the reservoir size. This
establishes a sensible operating point reused by the federated experiments.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import common
from esnfed import EchoStateNetwork, datasets, metrics, topologies


def run(seeds=(0, 1, 2, 3, 4)):
    spectral_radii = np.round(np.arange(0.1, 1.31, 0.1), 2)
    sizes = [50, 100, 200, 400]
    rows = []
    for size in sizes:
        for sr in spectral_radii:
            errs = []
            for s in seeds:
                rng = np.random.default_rng(common.MASTER_SEED + s)
                u, y = datasets.narma10(3000, rng=rng)
                utr, ytr, ute, yte = datasets.split(u, y, 0.7)
                W = topologies.random_reservoir(size, density=0.1, rng=rng)
                esn = EchoStateNetwork(
                    1, 1, W, spectral_radius=sr, leaking_rate=1.0,
                    input_scaling=0.5, ridge=1e-7, washout=100, seed=s,
                )
                esn.fit(utr, ytr)
                pred = esn.predict(ute)
                errs.append(metrics.nrmse(yte[100:], pred[100:]))
            rows.append(
                {"size": size, "spectral_radius": sr,
                 "nrmse_mean": np.mean(errs), "nrmse_std": np.std(errs)}
            )
            print(f"  N={size:4d}  rho={sr:.2f}  NRMSE={np.mean(errs):.4f}")
    return pd.DataFrame(rows)


def plot(df):
    common.set_style()
    fig, ax = common.plt.subplots(figsize=(6.2, 4.0))
    for size in sorted(df["size"].unique()):
        sub = df[df["size"] == size]
        ax.errorbar(
            sub["spectral_radius"], sub["nrmse_mean"], yerr=sub["nrmse_std"],
            marker="o", markersize=3, capsize=2, label=f"N = {size}",
        )
    ax.axvline(1.0, color="grey", ls="--", lw=1, alpha=0.7)
    ax.text(1.01, ax.get_ylim()[1] * 0.95, r"$\rho = 1$", color="grey", fontsize=8)
    ax.set_xlabel(r"Spectral radius $\rho$")
    ax.set_ylabel("Test NRMSE (NARMA-10)")
    ax.set_title("ESN performance vs. spectral radius and reservoir size")
    ax.legend(title="Reservoir size")
    common.save_figure(fig, "exp1_hyperparams")


def main():
    print("[exp1] ESN hyperparameter sweep on NARMA-10")
    df = run()
    common.save_table(df, "exp1_esn_sweep")
    # Best configuration per size, as a compact LaTeX table.
    best = (
        df.loc[df.groupby("size")["nrmse_mean"].idxmin()]
        .sort_values("size")[["size", "spectral_radius", "nrmse_mean", "nrmse_std"]]
        .rename(columns={
            "size": "Reservoir size $N$",
            "spectral_radius": r"Best $\rho$",
            "nrmse_mean": "NRMSE (mean)",
            "nrmse_std": "NRMSE (std)",
        })
    )
    common.save_latex_table(
        best, "exp1_best",
        caption="Best NARMA-10 test NRMSE per reservoir size (mean over 5 seeds), "
                "with the spectral radius achieving it.",
        label="tab:exp1-best",
    )
    plot(df)
    print("[exp1] done")


if __name__ == "__main__":
    main()
