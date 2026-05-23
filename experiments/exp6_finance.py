"""Experiment 6 - Federated counterparty-risk forecasting (real data).

A real-world application of the federated strategies: several simulated
financial institutions jointly forecast the TED spread -- the gap between
interbank and Treasury rates, a classic gauge of counterparty/credit risk in the
banking system -- without sharing their data.

Left panel: a centralised ESN tracks the TED spread on the held-out test period,
including the 2007--2009 crisis spike. Right panel: as the federation grows,
exact federated ridge holds the centralised accuracy while purely local training
degrades -- the same pattern as on the synthetic benchmarks, now on real data.

Source of the data: Federal Reserve Bank of St. Louis (FRED), series TEDRATE.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import common
import esnfed
from esnfed import EchoStateNetwork, datasets, federated, metrics, topologies

N = 200
# A sensible ridge keeps the (ill-conditioned) readout well-behaved on the short,
# non-stationary financial slices; a shorter washout suits the daily series.
ESN_KW = dict(spectral_radius=0.9, leaking_rate=0.5, input_scaling=1.0,
              ridge=1e-5, washout=50)
TED_CSV = Path(esnfed.__file__).parent / "data" / "ted_spread.csv"


def _load_with_dates():
    df = pd.read_csv(TED_CSV, parse_dates=["date"])
    return df["date"].to_numpy(), df["ted_spread"].to_numpy(dtype=float)


def forecast_panel(ax):
    dates, raw = _load_with_dates()
    mu, sd = raw.mean(), raw.std()
    s = (raw - mu) / sd
    u, y = s[:-1].reshape(-1, 1), s[1:].reshape(-1, 1)
    cut = int(0.7 * len(u))
    W = topologies.random_reservoir(N, density=0.1, rng=common.MASTER_SEED)
    esn = EchoStateNetwork(1, 1, W, seed=0, **ESN_KW)
    esn.fit(u[:cut], y[:cut])
    pred = esn.predict(u[cut:])
    wo = esn.washout
    # De-normalise back to percentage points for an interpretable axis.
    test_dates = dates[1:][cut:][wo:]
    actual = y[cut:][wo:] * sd + mu
    predicted = pred[wo:] * sd + mu
    err = metrics.nrmse(y[cut:][wo:], pred[wo:])

    ax.plot(test_dates, actual, color="#333333", lw=1.0, label="Actual")
    ax.plot(test_dates, predicted, color=common.PALETTE["federated_ridge"],
            lw=1.0, alpha=0.8, label="ESN forecast")
    ax.set_title(f"TED spread one-step forecast (test NRMSE = {err:.3f})")
    ax.set_ylabel("TED spread (pp)")
    ax.set_xlabel("Year")
    ax.legend(loc="upper right")
    return err


def scaling(client_counts=(1, 2, 5, 10, 20), seeds=(0, 1, 2, 3, 4)):
    u, y = datasets.load_ted_spread()
    u_tr, y_tr, u_te, y_te = datasets.split(u, y, 0.7)
    rows = []
    for nc in client_counts:
        for s in seeds:
            rng = np.random.default_rng(common.MASTER_SEED + s)
            W = topologies.random_reservoir(N, density=0.1, rng=rng)
            parts = datasets.partition_iid(u_tr, y_tr, nc, rng=rng)
            clients, ref = federated.make_shared_clients(
                W, parts, input_seed=0, esn_kwargs=ESN_KW)
            wo = ref.washout
            W_out = federated.federated_ridge(clients, ref)
            Z_test = ref.harvest(u_te)[wo:]
            fed = metrics.nrmse(y_te[wo:], Z_test @ W_out)
            federated.train_local(clients)
            loc = float(np.mean([
                metrics.nrmse(y_te[wo:], c.esn.predict(u_te)[wo:]) for c in clients
            ]))
            rows.append({"n_clients": nc, "seed": s, "federated_ridge": fed,
                         "local_mean": loc})
            print(f"  clients={nc:3d} seed={s}  fed={fed:.4f} local={loc:.4f}")
    return pd.DataFrame(rows)


def scaling_panel(ax, df):
    # Median is robust to the occasional blow-up of an under-regularised local
    # model on a short, non-stationary slice.
    g = df.groupby("n_clients")
    fed = g["federated_ridge"].median()
    loc = g["local_mean"].median()
    ax.plot(fed.index, fed.values, marker="o",
            color=common.PALETTE["federated_ridge"], label="Federated ridge (exact)")
    ax.plot(loc.index, loc.values, marker="s",
            color=common.PALETTE["local"], label="Local-only (median)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(sorted(df["n_clients"].unique()))
    ax.set_xticklabels([str(c) for c in sorted(df["n_clients"].unique())])
    ax.set_xlabel("Number of institutions")
    ax.set_ylabel("Test NRMSE (log scale)")
    ax.set_title("Federated vs. local on real TED-spread data")
    ax.legend()


def main():
    print("[exp6] federated counterparty-risk (TED spread) forecasting")
    common.set_style()
    fig, axes = common.plt.subplots(1, 2, figsize=(10.0, 3.9))
    forecast_panel(axes[0])
    df = scaling()
    scaling_panel(axes[1], df)
    common.save_figure(fig, "exp6_finance")
    common.save_table(df, "exp6_finance")
    summary = df.groupby("n_clients").agg(
        federated_ridge=("federated_ridge", "median"),
        local_mean=("local_mean", "median"),
    ).reset_index().rename(columns={
        "n_clients": "Institutions", "federated_ridge": "Federated ridge",
        "local_mean": "Local-only (median)"})
    common.save_latex_table(
        summary, "exp6_finance",
        caption="Federated forecasting of the TED spread (real counterparty-risk "
                "data): median test NRMSE of exact federated ridge vs. the local "
                "model as the number of institutions grows (5 seeds).",
        label="tab:exp6-finance")
    print("[exp6] done")


if __name__ == "__main__":
    main()
