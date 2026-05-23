"""Experiment 2 - Reservoir topology comparison.

Compares the four reservoir topologies (Erdos-Renyi, small-world, scale-free,
ring) on NARMA-10 and Mackey-Glass at a fixed reservoir size, and records their
graph descriptors. This isolates the effect of connectivity structure -- the
quantity that becomes heterogeneous across clients in the federated setting.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import common
from esnfed import EchoStateNetwork, datasets, metrics, topologies


N = 200
TOPOS = ["random", "small_world", "scale_free", "ring"]
TOPO_KWARGS = {
    "random": dict(density=0.1),
    "small_world": dict(k=10, p=0.1),
    "scale_free": dict(m=5),
    "ring": dict(),
}


def _evaluate(task, sr, leaking, input_scaling, seeds):
    rows = []
    for kind in TOPOS:
        for s in seeds:
            rng = np.random.default_rng(common.MASTER_SEED + 100 * s)
            if task == "narma10":
                u, y = datasets.narma10(3000, rng=rng)
            else:
                u, y = datasets.mackey_glass(3000, seed=int(common.MASTER_SEED + s))
            utr, ytr, ute, yte = datasets.split(u, y, 0.7)
            W = topologies.make_reservoir(kind, N, rng=rng, **TOPO_KWARGS[kind])
            gm = topologies.graph_metrics(W)
            esn = EchoStateNetwork(
                1, 1, W, spectral_radius=sr, leaking_rate=leaking,
                input_scaling=input_scaling, ridge=1e-7, washout=100, seed=s,
            )
            esn.fit(utr, ytr)
            pred = esn.predict(ute)
            rows.append({
                "task": task, "topology": kind, "seed": s,
                "nrmse": metrics.nrmse(yte[100:], pred[100:]),
                "mean_degree": gm["mean_degree"], "clustering": gm["clustering"],
                "avg_path_length": gm["avg_path_length"],
            })
    return rows


def run(seeds=(0, 1, 2, 3, 4, 5, 6, 7)):
    rows = []
    print("  task=narma10")
    rows += _evaluate("narma10", sr=0.9, leaking=1.0, input_scaling=0.5, seeds=seeds)
    print("  task=mackey_glass")
    rows += _evaluate("mackey_glass", sr=0.95, leaking=0.3, input_scaling=0.2, seeds=seeds)
    return pd.DataFrame(rows)


def plot(df):
    common.set_style()
    tasks = ["narma10", "mackey_glass"]
    titles = {"narma10": "NARMA-10", "mackey_glass": "Mackey-Glass"}
    fig, axes = common.plt.subplots(1, 2, figsize=(8.4, 3.8))
    for ax, task in zip(axes, tasks):
        sub = df[df["task"] == task]
        means = sub.groupby("topology")["nrmse"].mean().reindex(TOPOS)
        stds = sub.groupby("topology")["nrmse"].std().reindex(TOPOS)
        colors = [common.PALETTE[t] for t in TOPOS]
        ax.bar(range(len(TOPOS)), means.values, yerr=stds.values,
               color=colors, capsize=3, alpha=0.9)
        ax.set_xticks(range(len(TOPOS)))
        ax.set_xticklabels([common.LABELS[t] for t in TOPOS], rotation=20, ha="right")
        ax.set_ylabel("Test NRMSE")
        ax.set_title(titles[task])
    fig.suptitle(f"Reservoir topology comparison (N = {N}, 8 seeds)", y=1.02)
    common.save_figure(fig, "exp2_topology")


def main():
    print("[exp2] reservoir topology comparison")
    df = run()
    common.save_table(df, "exp2_topology")
    summary = (
        df.groupby(["task", "topology"])
        .agg(nrmse_mean=("nrmse", "mean"), nrmse_std=("nrmse", "std"),
             clustering=("clustering", "mean"), mean_degree=("mean_degree", "mean"))
        .reset_index()
    )
    common.save_table(summary, "exp2_topology_summary")

    # LaTeX summary table for NARMA-10.
    narma = summary[summary["task"] == "narma10"].copy()
    narma = narma[["topology", "nrmse_mean", "nrmse_std", "clustering", "mean_degree"]]
    narma["topology"] = narma["topology"].map(common.LABELS)
    narma = narma.rename(columns={
        "topology": "Topology", "nrmse_mean": "NRMSE (mean)",
        "nrmse_std": "NRMSE (std)", "clustering": "Clustering",
        "mean_degree": "Mean degree",
    })
    common.save_latex_table(
        narma, "exp2_narma",
        caption="NARMA-10 test NRMSE and graph descriptors by reservoir topology "
                f"(N={N}, mean over 8 seeds).",
        label="tab:exp2-narma",
    )
    plot(df)
    print("[exp2] done")


if __name__ == "__main__":
    main()
