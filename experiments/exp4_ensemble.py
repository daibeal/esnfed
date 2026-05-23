"""Experiment 4 - Heterogeneous clients: ensemble vs. local and centralized.

When each client owns a *different* reservoir (different topology and input
weights), readout parameters cannot be averaged. The server can still combine
the clients' predictions into an ensemble. This experiment compares, as the
federation grows:

* centralized (shared reservoir, pooled data) -- upper bound;
* local-only -- mean of independently trained client models;
* ensemble -- average of heterogeneous client predictions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import common
from esnfed import datasets, federated, metrics, topologies

N = 200
ESN_KW = dict(spectral_radius=0.9, leaking_rate=1.0, input_scaling=0.5,
              ridge=1e-7, washout=100)
KINDS = ["random", "small_world", "scale_free", "ring"]
KIND_KW = {"random": dict(density=0.1), "small_world": dict(k=10, p=0.1),
           "scale_free": dict(m=5), "ring": dict()}


def run(client_counts=(2, 4, 8, 16, 32), seeds=(0, 1, 2, 3)):
    rows = []
    for nc in client_counts:
        for s in seeds:
            rng = np.random.default_rng(common.MASTER_SEED + 7 * s + nc)
            u, y = datasets.narma10(8000, rng=rng)
            utr, ytr, ute, yte = datasets.split(u, y, 0.7)
            parts = datasets.partition_iid(utr, ytr, nc, rng=rng)
            washout = ESN_KW["washout"]

            # Centralized upper bound (single shared reservoir on pooled data).
            W_shared = topologies.random_reservoir(N, density=0.1, rng=rng)
            sh_clients, ref = federated.make_shared_clients(
                W_shared, parts, input_seed=0, esn_kwargs=ESN_KW)
            W_out = federated.federated_ridge(sh_clients, ref)
            Z_test = ref.harvest(ute)[washout:]
            central_err = metrics.nrmse(yte[washout:], Z_test @ W_out)

            # Heterogeneous clients: each gets a different topology in round-robin.
            reservoirs = [
                topologies.make_reservoir(KINDS[i % len(KINDS)], N, rng=rng,
                                          **KIND_KW[KINDS[i % len(KINDS)]])
                for i in range(nc)
            ]
            het = federated.make_heterogeneous_clients(reservoirs, parts, esn_kwargs=ESN_KW)
            federated.train_local(het)
            local_errs = [
                metrics.nrmse(yte[washout:], c.esn.predict(ute)[washout:]) for c in het
            ]
            ens_pred = federated.ensemble_predict(het, ute)
            ens_err = metrics.nrmse(yte[washout:], ens_pred[washout:])

            rows.append({
                "n_clients": nc, "seed": s,
                "centralized": central_err,
                "local_mean": float(np.mean(local_errs)),
                "local_best": float(np.min(local_errs)),
                "ensemble": ens_err,
            })
            print(f"  clients={nc:3d} seed={s}  central={central_err:.4f} "
                  f"local={np.mean(local_errs):.4f} ensemble={ens_err:.4f}")
    return pd.DataFrame(rows)


def plot(df):
    common.set_style()
    fig, ax = common.plt.subplots(figsize=(6.2, 4.0))
    g = df.groupby("n_clients")
    for key, color, label, marker in [
        ("centralized", common.PALETTE["centralized"], common.LABELS["centralized"], "o"),
        ("ensemble", common.PALETTE["ensemble"], common.LABELS["ensemble"], "^"),
        ("local_mean", common.PALETTE["local"], common.LABELS["local"], "s"),
    ]:
        m = g[key].mean()
        sd = g[key].std()
        ax.errorbar(m.index, m.values, yerr=sd.values, marker=marker, capsize=2,
                    color=color, label=label)
    counts = sorted(df["n_clients"].unique())
    ax.set_xscale("log", base=2)
    ax.set_xticks(counts)
    ax.set_xticklabels([str(c) for c in counts])
    ax.set_xlabel("Number of clients")
    ax.set_ylabel("Test NRMSE (NARMA-10)")
    ax.set_title("Heterogeneous federation: ensemble beats local-only")
    ax.legend()
    common.save_figure(fig, "exp4_ensemble")


def main():
    print("[exp4] heterogeneous clients - ensemble vs local vs centralized")
    df = run()
    common.save_table(df, "exp4_ensemble")
    summary = df.groupby("n_clients").agg(
        centralized=("centralized", "mean"),
        ensemble=("ensemble", "mean"),
        local_mean=("local_mean", "mean"),
        local_best=("local_best", "mean"),
    ).reset_index().rename(columns={
        "n_clients": "Clients", "centralized": "Centralized",
        "ensemble": "Ensemble", "local_mean": "Local (mean)",
        "local_best": "Local (best)",
    })
    common.save_latex_table(
        summary, "exp4_ensemble",
        caption="Heterogeneous federation on NARMA-10: test NRMSE of the "
                "centralized bound, the prediction ensemble, and local-only "
                "models (mean over 4 seeds).",
        label="tab:exp4-ensemble",
    )
    plot(df)
    print("[exp4] done")


if __name__ == "__main__":
    main()
