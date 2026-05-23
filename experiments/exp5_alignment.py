"""Experiment 5 - Structural alignment of heterogeneous reservoirs.

Interpolates each client's reservoir toward a shared target structure, sweeping
the alignment level ``alpha`` from 0 (fully heterogeneous) to 1 (identical
structures). Two readouts are tracked on the global test set:

* ensemble of locally trained readouts, and
* parameter aggregation (exact federated ridge over the shared input weights),
  which only becomes valid as the reservoirs coincide (alpha -> 1).

This exposes the trade-off between the two ways of handling structural
heterogeneity proposed in the project.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import common
from esnfed import datasets, federated, topologies

N = 200
ESN_KW = dict(spectral_radius=0.9, leaking_rate=1.0, input_scaling=0.5,
              ridge=1e-7, washout=100)


def run(n_clients=8, seeds=(0, 1, 2, 3), n_alpha=11):
    alphas = np.linspace(0.0, 1.0, n_alpha)
    all_records = []
    for s in seeds:
        rng = np.random.default_rng(common.MASTER_SEED + 13 * s)
        u, y = datasets.narma10(8000, rng=rng)
        utr, ytr, ute, yte = datasets.split(u, y, 0.7)
        parts = datasets.partition_iid(utr, ytr, n_clients, rng=rng)

        # Distinct local reservoirs + one shared target reservoir.
        local_res = [topologies.random_reservoir(N, density=0.1, rng=rng)
                     for _ in range(n_clients)]
        target = topologies.random_reservoir(N, density=0.1, rng=rng)

        recs = federated.structural_alignment(
            local_res, target, parts, ute, yte,
            alphas=alphas, esn_kwargs=ESN_KW, shared_input_seed=0,
        )
        for r in recs:
            r["seed"] = s
        all_records.extend(recs)
        print(f"  seed={s} done ({len(recs)} alpha values)")
    return pd.DataFrame(all_records)


def plot(df):
    common.set_style()
    g = df.groupby("alpha")
    fig, ax1 = common.plt.subplots(figsize=(6.4, 4.2))

    ens = g["ensemble_nrmse"].mean()
    ens_sd = g["ensemble_nrmse"].std()
    fed = g["fedavg_nrmse"].mean()
    fed_sd = g["fedavg_nrmse"].std()

    ax1.errorbar(ens.index, ens.values, yerr=ens_sd.values, marker="^", capsize=2,
                 color=common.PALETTE["ensemble"], label="Ensemble")
    ax1.errorbar(fed.index, fed.values, yerr=fed_sd.values, marker="o", capsize=2,
                 color=common.PALETTE["federated_ridge"],
                 label="Parameter aggregation (federated ridge)")
    ax1.set_xlabel(r"Alignment level $\alpha$  (0 = heterogeneous, 1 = identical)")
    ax1.set_ylabel("Test NRMSE (NARMA-10)")

    ax2 = ax1.twinx()
    het = g["heterogeneity"].mean()
    ax2.plot(het.index, het.values, color="grey", ls=":", lw=1.5,
             label="Reservoir heterogeneity")
    ax2.set_ylabel("Mean pairwise reservoir distance", color="grey")
    ax2.tick_params(axis="y", colors="grey")
    ax2.grid(False)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper center")
    ax1.set_title("Structural alignment: ensemble vs. parameter aggregation")
    common.save_figure(fig, "exp5_alignment")


def main():
    print("[exp5] structural alignment of heterogeneous reservoirs")
    df = run()
    common.save_table(df, "exp5_alignment")
    summary = df.groupby("alpha").agg(
        ensemble_nrmse=("ensemble_nrmse", "mean"),
        fedavg_nrmse=("fedavg_nrmse", "mean"),
        heterogeneity=("heterogeneity", "mean"),
    ).reset_index()
    # Compact table at a few alpha values.
    pick = summary.iloc[[0, len(summary) // 2, -1]].copy()
    pick = pick.rename(columns={
        "alpha": r"$\alpha$", "ensemble_nrmse": "Ensemble NRMSE",
        "fedavg_nrmse": "Aggregation NRMSE", "heterogeneity": "Heterogeneity",
    })
    common.save_latex_table(
        pick, "exp5_alignment",
        caption="Structural alignment on NARMA-10 at three alignment levels: "
                "ensemble vs. parameter aggregation test NRMSE and the remaining "
                "reservoir heterogeneity (mean over 4 seeds).",
        label="tab:exp5-alignment",
    )
    plot(df)
    print("[exp5] done")


if __name__ == "__main__":
    main()
