"""Regenerate the LaTeX tables from the saved CSV results (no re-running).

Useful after changing table formatting in ``common.py``.
"""
from __future__ import annotations

import pandas as pd

import common


def exp1():
    df = pd.read_csv(common.OUTPUTS / "exp1_esn_sweep.csv")
    best = (
        df.loc[df.groupby("size")["nrmse_mean"].idxmin()]
        .sort_values("size")[["size", "spectral_radius", "nrmse_mean", "nrmse_std"]]
        .rename(columns={
            "size": "Reservoir size $N$", "spectral_radius": r"Best $\rho$",
            "nrmse_mean": "NRMSE (mean)", "nrmse_std": "NRMSE (std)",
        })
    )
    common.save_latex_table(
        best, "exp1_best",
        caption="Best NARMA-10 test NRMSE per reservoir size (mean over 5 seeds), "
                "with the spectral radius achieving it.",
        label="tab:exp1-best")


def exp2():
    s = pd.read_csv(common.OUTPUTS / "exp2_topology_summary.csv")
    narma = s[s["task"] == "narma10"][
        ["topology", "nrmse_mean", "nrmse_std", "clustering", "mean_degree"]].copy()
    narma["topology"] = narma["topology"].map(common.LABELS)
    narma = narma.rename(columns={
        "topology": "Topology", "nrmse_mean": "NRMSE (mean)",
        "nrmse_std": "NRMSE (std)", "clustering": "Clustering",
        "mean_degree": "Mean degree"})
    common.save_latex_table(
        narma, "exp2_narma",
        caption="NARMA-10 test NRMSE and graph descriptors by reservoir topology "
                "(N=200, mean over 8 seeds).",
        label="tab:exp2-narma")


def exp3():
    df = pd.read_csv(common.OUTPUTS / "exp3_scaling.csv")
    s = df.groupby("n_clients").agg(
        federated_ridge=("federated_ridge", "mean"),
        local_mean=("local_mean", "mean")).reset_index().rename(columns={
            "n_clients": "Clients", "federated_ridge": "Federated ridge",
            "local_mean": "Local-only (mean)"})
    common.save_latex_table(
        s, "exp3_scaling",
        caption="NARMA-10 test NRMSE: exact federated ridge vs. the mean local "
                "model, as the number of clients grows (mean over 3 seeds).",
        label="tab:exp3-scaling")


def exp4():
    df = pd.read_csv(common.OUTPUTS / "exp4_ensemble.csv")
    s = df.groupby("n_clients").agg(
        centralized=("centralized", "mean"), ensemble=("ensemble", "mean"),
        local_mean=("local_mean", "mean"), local_best=("local_best", "mean"),
    ).reset_index().rename(columns={
        "n_clients": "Clients", "centralized": "Centralized",
        "ensemble": "Ensemble", "local_mean": "Local (mean)",
        "local_best": "Local (best)"})
    common.save_latex_table(
        s, "exp4_ensemble",
        caption="Heterogeneous federation on NARMA-10: test NRMSE of the "
                "centralized bound, the prediction ensemble, and local-only "
                "models (mean over 4 seeds).",
        label="tab:exp4-ensemble")


def exp5():
    df = pd.read_csv(common.OUTPUTS / "exp5_alignment.csv")
    s = df.groupby("alpha").agg(
        ensemble_nrmse=("ensemble_nrmse", "mean"),
        fedavg_nrmse=("fedavg_nrmse", "mean"),
        heterogeneity=("heterogeneity", "mean")).reset_index()
    pick = s.iloc[[0, len(s) // 2, -1]].copy().rename(columns={
        "alpha": r"$\alpha$", "ensemble_nrmse": "Ensemble NRMSE",
        "fedavg_nrmse": "Aggregation NRMSE", "heterogeneity": "Heterogeneity"})
    common.save_latex_table(
        pick, "exp5_alignment",
        caption="Structural alignment on NARMA-10 at three alignment levels: "
                "ensemble vs. parameter aggregation test NRMSE and the remaining "
                "reservoir heterogeneity (mean over 4 seeds).",
        label="tab:exp5-alignment")


if __name__ == "__main__":
    for fn in (exp1, exp2, exp3, exp4, exp5):
        fn()
    print("tables regenerated")
