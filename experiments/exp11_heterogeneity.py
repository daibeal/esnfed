"""Experiment 11 - Reservoir heterogeneity extensions.

Compares, on two chaotic / multi-scale tasks (Mackey-Glass and Lorenz), a plain
homogeneous reservoir against three heterogeneity extensions, at a matched total
reservoir size (N = 150):

* baseline          -- homogeneous leaking rate, tanh, single layer;
* hetero_leaking    -- per-node layered leaking rates (multi-scale time constants);
* multi_activation  -- mixed node nonlinearities (tanh / sigmoid / sin);
* deep              -- 3 stacked reservoirs (50 each) with decreasing time-scales;
* deep_hetero       -- deep + mixed activations per layer.

Reports NRMSE (mean +/- std over seeds); writes a figure, a LaTeX table and a CSV.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import common
from esnfed import (DeepEchoStateNetwork, EchoStateNetwork, datasets, metrics,
                    topologies)

SR, LEAK, WASH, N = 0.95, 0.3, 100, 150
SEEDS = [0, 1, 2]
CONFIGS = ["baseline", "hetero_leaking", "multi_activation", "deep", "deep_hetero"]
TASKS = {"mackey_glass": datasets.mackey_glass, "lorenz": datasets.lorenz}


def build(config, seed):
    if config == "baseline":
        W = topologies.random_reservoir(N, density=0.1, rng=seed)
        return EchoStateNetwork(1, 1, W, spectral_radius=SR, leaking_rate=LEAK,
                                washout=WASH, seed=seed)
    if config == "hetero_leaking":
        W = topologies.random_reservoir(N, density=0.1, rng=seed)
        a = topologies.leaking_rates(N, "layered", low=0.1, high=0.9, n_layers=4, rng=seed)
        return EchoStateNetwork(1, 1, W, spectral_radius=SR, leaking_rate=a,
                                washout=WASH, seed=seed)
    if config == "multi_activation":
        W = topologies.random_reservoir(N, density=0.1, rng=seed)
        acts = topologies.mixed_activations(N, ("tanh", "sigmoid", "sin"), rng=seed)
        return EchoStateNetwork(1, 1, W, spectral_radius=SR, leaking_rate=LEAK,
                                activation=acts, washout=WASH, seed=seed)
    if config == "deep":
        Ws = [topologies.random_reservoir(50, density=0.1, rng=seed * 10 + i) for i in range(3)]
        return DeepEchoStateNetwork(1, 1, Ws, spectral_radius=SR,
                                    leaking_rate=[0.9, 0.5, 0.2], washout=WASH, seed=seed)
    if config == "deep_hetero":
        Ws = [topologies.random_reservoir(50, density=0.1, rng=seed * 10 + i) for i in range(3)]
        acts = [topologies.mixed_activations(50, ("tanh", "sin"), rng=seed * 10 + i) for i in range(3)]
        return DeepEchoStateNetwork(1, 1, Ws, spectral_radius=SR,
                                    leaking_rate=[0.9, 0.5, 0.2], activation=acts,
                                    washout=WASH, seed=seed)
    raise ValueError(config)


def main():
    print("[exp11] reservoir-heterogeneity extensions")
    rows = []
    for task, gen in TASKS.items():
        u, y = gen(3000, seed=0)
        u_tr, y_tr, u_te, y_te = datasets.split(u, y, 0.7)
        for cfg in CONFIGS:
            for s in SEEDS:
                esn = build(cfg, s).fit(u_tr, y_tr)
                e = metrics.nrmse(y_te[WASH:], esn.predict(u_te)[WASH:])
                rows.append(dict(task=task, config=cfg, seed=s, nrmse=float(e)))
        print(f"  {task}: done")
    df = pd.DataFrame(rows)
    common.save_table(df, "exp11_heterogeneity")

    summ = df.groupby(["task", "config"])["nrmse"].agg(["mean", "std"]).reset_index()
    print(summ.to_string(index=False))

    # figure: grouped bar (config x task), log scale
    common.set_style()
    fig, ax = common.plt.subplots(figsize=(9.5, 4))
    labels = {"baseline": "baseline", "hetero_leaking": "hetero leaking",
              "multi_activation": "multi-activation", "deep": "deep (3 layers)",
              "deep_hetero": "deep + mixed"}
    cmap = [common.PALETTE["local"], common.PALETTE["federated_ridge"],
            common.PALETTE["ensemble"], common.PALETTE["fedavg"],
            common.PALETTE["alignment"]]
    x = np.arange(len(TASKS)); width = 0.16
    for i, cfg in enumerate(CONFIGS):
        s = summ[summ["config"] == cfg].set_index("task").reindex(list(TASKS))
        ax.bar(x + (i - 2) * width, s["mean"], width, yerr=s["std"],
               label=labels[cfg], color=cmap[i], capsize=3)
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(list(TASKS))
    ax.set_ylabel("test NRMSE (log)"); ax.set_title("Reservoir heterogeneity extensions")
    ax.legend(ncol=3, fontsize=8)
    common.save_figure(fig, "exp11_heterogeneity")

    # latex table: mean +/- std per task/config
    piv = summ.assign(cell=summ.apply(
        lambda r: f"{r['mean']:.4f} $\\pm$ {r['std']:.4f}", axis=1)
    ).pivot(index="config", columns="task", values="cell").reindex(CONFIGS)
    piv.index = [labels[c] for c in piv.index]
    piv = piv.rename(columns={"mackey_glass": "Mackey-Glass", "lorenz": "Lorenz"})
    piv = piv.reset_index().rename(columns={"index": "configuration"})
    common.save_latex_table(
        piv, "exp11_heterogeneity",
        caption="Test NRMSE (mean $\\pm$ std over 3 seeds) of the reservoir "
                "heterogeneity extensions on two chaotic tasks, at a matched total "
                "reservoir size ($N=150$). Lower is better.",
        label="tab:exp11-heterogeneity")
    print("[exp11] done")


if __name__ == "__main__":
    main()
