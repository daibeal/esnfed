"""Experiment 3 - Federated learning with a shared reservoir.

With a reservoir shared across clients, two federated readout schemes are
compared against the centralized upper bound:

* exact federated ridge (one round of summed sufficient statistics), and
* iterative FedAvg (gradient averaging over communication rounds).

Shows (a) FedAvg's convergence curve vs. the exact optima, and (b) how the gap
behaves as the number of clients grows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import common
from esnfed import datasets, federated, metrics, topologies

N = 200
ESN_KW = dict(spectral_radius=0.9, leaking_rate=1.0, input_scaling=0.5,
              ridge=1e-7, washout=100)


def _prepare(n_clients, seed):
    rng = np.random.default_rng(common.MASTER_SEED + seed)
    # Large enough that even 50 clients keep > washout samples each.
    u, y = datasets.narma10(15000, rng=rng)
    utr, ytr, ute, yte = datasets.split(u, y, 0.7)
    W = topologies.random_reservoir(N, density=0.1, rng=rng)
    parts = datasets.partition_iid(utr, ytr, n_clients, rng=rng)
    clients, ref = federated.make_shared_clients(W, parts, input_seed=0, esn_kwargs=ESN_KW)
    return clients, ref, ute, yte


def convergence(n_clients=10, seed=0, rounds=200):
    clients, ref, ute, yte = _prepare(n_clients, seed)
    washout = ref.washout

    W_central = federated.federated_ridge(clients, ref)
    Z_test = ref.harvest(ute)[washout:]
    exact_err = metrics.nrmse(yte[washout:], Z_test @ W_central)

    _, hist = federated.fedavg(clients, ref, ute, yte,
                               rounds=rounds, local_epochs=5, lr=0.9)
    return exact_err, hist


def scaling(client_counts=(1, 2, 5, 10, 20, 50), seeds=(0, 1, 2)):
    rows = []
    for nc in client_counts:
        for s in seeds:
            clients, ref, ute, yte = _prepare(nc, s)
            washout = ref.washout
            W_fed = federated.federated_ridge(clients, ref)
            Z_test = ref.harvest(ute)[washout:]
            fed_err = metrics.nrmse(yte[washout:], Z_test @ W_fed)

            federated.train_local(clients)
            local_errs = [
                metrics.nrmse(yte[washout:], c.esn.predict(ute)[washout:])
                for c in clients
            ]
            rows.append({
                "n_clients": nc, "seed": s,
                "federated_ridge": fed_err,
                "local_mean": float(np.mean(local_errs)),
            })
            print(f"  clients={nc:3d} seed={s}  fed={fed_err:.4f} "
                  f"local={np.mean(local_errs):.4f}")
    return pd.DataFrame(rows)


def plot_convergence(exact_err, hist):
    common.set_style()
    fig, ax = common.plt.subplots(figsize=(6.0, 4.0))
    ax.plot(range(1, len(hist) + 1), hist, color=common.PALETTE["fedavg"],
            label="FedAvg (iterative)")
    ax.axhline(exact_err, color=common.PALETTE["federated_ridge"], ls="--",
               label="Federated ridge (exact)")
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Global test NRMSE")
    ax.set_title("FedAvg convergence vs. exact federated ridge (10 clients)")
    ax.legend()
    common.save_figure(fig, "exp3_fedavg_convergence")


def plot_scaling(df):
    common.set_style()
    fig, ax = common.plt.subplots(figsize=(6.0, 4.0))
    g = df.groupby("n_clients")
    fed = g["federated_ridge"].mean()
    fed_sd = g["federated_ridge"].std()
    loc = g["local_mean"].mean()
    loc_sd = g["local_mean"].std()
    ax.errorbar(fed.index, fed.values, yerr=fed_sd.values, marker="o", capsize=2,
                color=common.PALETTE["federated_ridge"],
                label="Federated ridge (exact)")
    ax.errorbar(loc.index, loc.values, yerr=loc_sd.values, marker="s", capsize=2,
                color=common.PALETTE["local"], label="Local-only (mean)")
    ax.set_xscale("log")
    ax.set_xlabel("Number of clients")
    ax.set_ylabel("Test NRMSE")
    ax.set_title("Federated vs. local as the federation grows")
    ax.legend()
    common.save_figure(fig, "exp3_scaling")


def main():
    print("[exp3] federated learning with shared reservoir")
    exact_err, hist = convergence()
    pd.DataFrame({"round": range(1, len(hist) + 1), "fedavg_nrmse": hist,
                  "exact_nrmse": exact_err}).pipe(
        common.save_table, "exp3_convergence")
    plot_convergence(exact_err, hist)

    df = scaling()
    common.save_table(df, "exp3_scaling")
    summary = df.groupby("n_clients").agg(
        federated_ridge=("federated_ridge", "mean"),
        local_mean=("local_mean", "mean"),
    ).reset_index().rename(columns={
        "n_clients": "Clients", "federated_ridge": "Federated ridge",
        "local_mean": "Local-only (mean)",
    })
    common.save_latex_table(
        summary, "exp3_scaling",
        caption="NARMA-10 test NRMSE: exact federated ridge vs. the mean local "
                "model, as the number of clients grows (mean over 3 seeds).",
        label="tab:exp3-scaling",
    )
    plot_scaling(df)
    print("[exp3] done")


if __name__ == "__main__":
    main()
