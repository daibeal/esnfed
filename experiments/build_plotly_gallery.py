"""Generate interactive Plotly charts of the thesis results for the docs site.

Reads the consolidated results database (``results/esnfed_results.db``) and writes
small, self-contained interactive HTML charts (Plotly.js from CDN) into
``docs/plotly/result_*.html`` --- embedded on the documentation website.

    pip install plotly pandas
    python experiments/build_plotly_gallery.py
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go


def _root() -> Path:
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "results").is_dir() and (p / "docs").is_dir():
            return p
    for p in here.parents:
        if (p / "pyproject.toml").exists():
            return p
    return here.parents[1]


ROOT = _root()
OUT = ROOT / "docs" / "plotly"
OUT.mkdir(parents=True, exist_ok=True)
CON = sqlite3.connect(ROOT / "results" / "esnfed_results.db")

PAL = {"federated_ridge": "#4C72B0", "federated": "#4C72B0", "exact_nrmse": "#4C72B0",
       "fedavg": "#DD8452", "fedavg_nrmse": "#DD8452", "ensemble": "#55A868",
       "ensemble_nrmse": "#55A868", "local": "#C44E52", "local_mean": "#C44E52",
       "local_median": "#C44E52", "centralized": "#333333", "sparse": "#8172B3",
       "numpy_f64": "#4C72B0", "numba_f64": "#55A868", "numpy_f32": "#DD8452"}


def _style(fig, title, xt, yt, logy=False):
    fig.update_layout(template="plotly_white", title=title, xaxis_title=xt,
                      yaxis_title=yt, height=420, margin=dict(l=60, r=30, t=60, b=55),
                      legend=dict(orientation="h", y=1.08, x=0))
    if logy:
        fig.update_yaxes(type="log")
    return fig


def save(fig, name):
    fig.write_html(str(OUT / f"result_{name}.html"), include_plotlyjs="cdn",
                   full_html=True)
    print(f"  docs/plotly/result_{name}.html")


def line(df, x, ycols, title, xt, yt, names=None, logy=False):
    fig = go.Figure()
    for i, c in enumerate(ycols):
        fig.add_scatter(x=df[x], y=df[c], mode="lines+markers",
                        name=(names[i] if names else c),
                        line=dict(color=PAL.get(c, None), width=2.5))
    return _style(fig, title, xt, yt, logy)


def main():
    print("[gallery] building interactive result charts")

    # exp1 — hyper-parameter heatmap
    d = pd.read_sql("SELECT * FROM exp1_esn_sweep", CON)
    piv = d.pivot(index="spectral_radius", columns="size", values="nrmse_mean")
    fig = go.Figure(go.Heatmap(z=piv.values, x=piv.columns, y=piv.index,
                               colorscale="Viridis_r", colorbar=dict(title="NRMSE")))
    save(_style(fig, "ESN sweep — NRMSE over size and spectral radius (NARMA-10)",
                "reservoir size N", "spectral radius ρ"), "exp1_sweep")

    # exp2 — topology comparison (grouped bar, log y)
    d = pd.read_sql("SELECT * FROM exp2_topology_summary", CON)
    fig = go.Figure()
    for task in d["task"].unique():
        s = d[d["task"] == task]
        fig.add_bar(x=s["topology"], y=s["nrmse_mean"],
                    error_y=dict(type="data", array=s["nrmse_std"]), name=task)
    fig.update_layout(barmode="group")
    save(_style(fig, "Reservoir topology comparison", "topology", "NRMSE", logy=True),
         "exp2_topology")

    # exp3 — scaling + convergence
    d = pd.read_sql("SELECT n_clients, AVG(federated_ridge) federated, "
                    "AVG(local_mean) local FROM exp3_scaling GROUP BY n_clients", CON)
    save(line(d, "n_clients", ["federated", "local"],
              "Federated vs local as the federation grows", "clients", "NRMSE",
              names=["federated ridge", "local-only"]), "exp3_scaling")
    d = pd.read_sql("SELECT * FROM exp3_convergence", CON)
    save(line(d, "round", ["fedavg_nrmse", "exact_nrmse"],
              "FedAvg convergence vs exact federated ridge", "round", "NRMSE",
              names=["FedAvg", "exact ridge"], logy=True), "exp3_convergence")

    # exp4 — ensemble
    d = pd.read_sql("SELECT n_clients, AVG(centralized) centralized, "
                    "AVG(local_mean) local_mean, AVG(ensemble) ensemble "
                    "FROM exp4_ensemble GROUP BY n_clients", CON)
    save(line(d, "n_clients", ["centralized", "local_mean", "ensemble"],
              "Prediction ensemble (heterogeneous reservoirs)", "clients", "NRMSE",
              names=["centralized", "local-only", "ensemble"]), "exp4_ensemble")

    # exp5 — structural alignment
    d = pd.read_sql("SELECT alpha, AVG(ensemble_nrmse) ensemble_nrmse, "
                    "AVG(fedavg_nrmse) fedavg_nrmse FROM exp5_alignment GROUP BY alpha", CON)
    save(line(d, "alpha", ["ensemble_nrmse", "fedavg_nrmse"],
              "Structural alignment (heterogeneous → shared)", "interpolation α",
              "NRMSE", names=["ensemble", "parameter aggregation"]), "exp5_alignment")

    # exp6 — finance (log y)
    d = pd.read_sql("SELECT n_clients, AVG(federated_ridge) federated, "
                    "AVG(local_mean) local FROM exp6_finance GROUP BY n_clients", CON)
    save(line(d, "n_clients", ["federated", "local"],
              "Federated counterparty-risk forecasting (TED spread)", "institutions",
              "NRMSE", names=["federated ridge", "local-only"], logy=True),
         "exp6_finance")

    # exp7 — FedResPrompt savings (log y)
    d = pd.read_sql("SELECT * FROM exp7_fedresprompt ORDER BY params", CON)
    fig = go.Figure()
    fig.add_bar(x=d["model"], y=d["bytes_ratio"], name="communication savings ×",
                marker_color="#4C72B0")
    fig.add_bar(x=d["model"], y=d["flops_ratio"], name="edge FLOP savings ×",
                marker_color="#55A868")
    fig.update_layout(barmode="group")
    save(_style(fig, "FedResPrompt vs federated LoRA (savings factor)", "model",
                "× savings (log)", logy=True), "exp7_fedresprompt")

    # exp9 — acceleration throughput
    d = pd.read_sql("SELECT * FROM exp9_performance", CON)
    fig = go.Figure()
    for c in ["numpy_f64", "numpy_f32", "numba_f64", "sparse"]:
        if c in d.columns:
            fig.add_bar(x=d["N"].astype(str), y=d[c], name=c,
                        marker_color=PAL.get(c))
    fig.update_layout(barmode="group")
    save(_style(fig, "Harvest throughput by backend", "reservoir size N",
                "steps / second (log)", logy=True), "exp9_performance")

    # exp10 — benchmark accuracy
    d = pd.read_sql("SELECT dataset, strategy, value FROM exp10_benchmarks "
                    "WHERE metric='accuracy'", CON)
    fig = go.Figure()
    for strat in d["strategy"].unique():
        s = d[d["strategy"] == strat]
        fig.add_bar(x=s["dataset"], y=s["value"], name=strat,
                    marker_color=PAL.get(strat))
    fig.update_layout(barmode="group")
    save(_style(fig, "Benchmark classification accuracy (federated vs ensemble vs local)",
                "dataset", "test accuracy"), "exp10_accuracy")

    print("[gallery] done")


if __name__ == "__main__":
    main()
