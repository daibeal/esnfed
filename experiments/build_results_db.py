"""Consolidate every experimental metric of the thesis into one database.

Reads the per-experiment outputs in ``salidas/`` and writes a single, tidy
results database in two standard, framework-neutral formats used across the
machine-learning / data-science ecosystem:

* **SQLite**  ``results/esnfed_results.db`` --- one wide table per experiment
  (full fidelity), a unified long-format ``metrics`` table, and an
  ``experiments`` metadata table. Queryable with plain SQL from any language.
* **Parquet** ``results/parquet/*.parquet`` --- the Apache Arrow / Parquet
  columnar format (read natively by pandas, polars, DuckDB, Spark, ...), plus
  the consolidated ``metrics_long.parquet``.

The long table follows the *tidy data* convention (one row per
``experiment × scenario × metric``), so all experiments can be queried together.

Run with a Python that has numpy/pandas/pyarrow:
    python experiments/build_results_db.py
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SALIDAS = ROOT / "salidas"
OUT = ROOT / "results"
PARQUET = OUT / "parquet"

# Per-experiment metadata + how to read the wide CSV into the long table.
# id    : columns that identify a scenario (everything else is a metric/value)
# the dataset/method/n_clients/seed mappings populate the convenience columns
# of the long table; the full scenario is always kept as JSON.
EXPERIMENTS = {
    "exp1_esn_sweep": dict(
        title="ESN hyper-parameter sweep",
        desc="Single-ESN NRMSE over reservoir size and spectral radius (NARMA-10).",
        section="sec:res-hyper", file="exp1_esn_sweep.csv",
        id=["size", "spectral_radius"], dataset="narma10"),
    "exp2_topology": dict(
        title="Reservoir topology comparison (per seed)",
        desc="NRMSE and graph descriptors per topology and task.",
        section="sec:res-topology", file="exp2_topology.csv",
        id=["task", "topology", "seed"], dataset_col="task", method_col="topology"),
    "exp2_topology_summary": dict(
        title="Reservoir topology comparison (summary)",
        desc="Mean/std NRMSE per topology and task.",
        section="sec:res-topology", file="exp2_topology_summary.csv",
        id=["task", "topology"], dataset_col="task", method_col="topology"),
    "exp3_convergence": dict(
        title="FedAvg vs exact federated ridge (convergence)",
        desc="Test NRMSE per communication round.",
        section="sec:res-fedridge", file="exp3_convergence.csv",
        id=["round"], dataset="narma10"),
    "exp3_scaling": dict(
        title="Federated vs local scaling",
        desc="Federated-ridge vs local-only NRMSE as the federation grows.",
        section="sec:res-fedridge", file="exp3_scaling.csv",
        id=["n_clients", "seed"], dataset="narma10"),
    "exp4_ensemble": dict(
        title="Prediction ensemble (heterogeneous reservoirs)",
        desc="Centralized, local, and ensemble NRMSE vs number of clients.",
        section="sec:res-ensemble", file="exp4_ensemble.csv",
        id=["n_clients", "seed"], dataset="narma10"),
    "exp5_alignment": dict(
        title="Structural alignment",
        desc="Ensemble vs parameter-aggregation NRMSE while interpolating "
             "reservoirs toward a shared target.",
        section="sec:res-alignment", file="exp5_alignment.csv",
        id=["alpha", "seed"], dataset="narma10"),
    "exp6_finance": dict(
        title="Federated counterparty-risk forecasting",
        desc="Federated-ridge vs local-only NRMSE on the TED spread.",
        section="sec:res-finance", file="exp6_finance.csv",
        id=["n_clients", "seed"], dataset="ted_spread"),
    "exp7_fedresprompt": dict(
        title="FedResPrompt vs federated LoRA",
        desc="Per-round communication bytes and edge FLOPs across model scales.",
        section="sec:res-fedresprompt", file="exp7_fedresprompt.csv",
        id=["model", "d_model", "n_layers", "params"], method_col="model"),
    "exp9_performance": dict(
        title="Acceleration benchmark",
        desc="Harvest throughput (steps/s) for NumPy f64/f32, Numba, sparse.",
        section="sec:res-performance", file="exp9_performance.csv",
        id=["N"]),
    "exp11_heterogeneity": dict(
        title="Reservoir heterogeneity extensions",
        desc="NRMSE of heterogeneous leaking rates, multi-type node activations "
             "and deep reservoirs on chaotic tasks (Mackey-Glass, Lorenz).",
        section="sec:res-heterogeneity", file="exp11_heterogeneity.csv",
        id=["task", "config", "seed"], dataset_col="task", method_col="config"),
}


def _to_long(name, spec, df):
    """Melt a wide experiment table into tidy long rows."""
    idc = [c for c in spec["id"] if c in df.columns]
    metrics = [c for c in df.columns if c not in idc]
    rows = []
    for _, r in df.iterrows():
        scenario = {c: r[c] for c in idc}
        dataset = spec.get("dataset") or (
            r[spec["dataset_col"]] if spec.get("dataset_col") else None)
        method = r[spec["method_col"]] if spec.get("method_col") else None
        nclients = r["n_clients"] if "n_clients" in df.columns else None
        seed = r["seed"] if "seed" in df.columns else None
        for m in metrics:
            rows.append(dict(experiment=name, dataset=dataset, method=method,
                             n_clients=nclients, seed=seed, metric=m,
                             value=float(r[m]) if pd.notna(r[m]) else None,
                             scenario=json.dumps(scenario, default=str)))
    return rows


def _exp8_long():
    """Parse the key=value Qwen-validation summary."""
    txt = (SALIDAS / "exp8_qwen_validation.txt").read_text(encoding="utf-8")
    kv = dict(l.split("=", 1) for l in txt.splitlines() if "=" in l)
    scen = json.dumps({"model": kv.get("model"), "classes": kv.get("classes")})
    rows = []
    for m in ("acc_before", "acc_after", "loss_first", "loss_last"):
        if m in kv:
            rows.append(dict(experiment="exp8_qwen_validation",
                             dataset="qwen2.5-0.5b", method="fedresprompt",
                             n_clients=2, seed=None, metric=m,
                             value=float(kv[m]), scenario=scen))
    return rows, kv


def _exp10_long():
    """exp10 is already saved in tidy long form."""
    df = pd.read_csv(SALIDAS / "exp10_benchmarks.csv")
    rows = []
    for _, r in df.iterrows():
        rows.append(dict(experiment="exp10_benchmarks", dataset=r["dataset"],
                         method=r["strategy"],
                         n_clients=r["n_clients"] if pd.notna(r["n_clients"]) else None,
                         seed=None, metric=r["metric"], value=float(r["value"]),
                         scenario=json.dumps({"task": r["task"]})))
    return rows, df


def main():
    OUT.mkdir(exist_ok=True)
    PARQUET.mkdir(exist_ok=True)
    con = sqlite3.connect(OUT / "esnfed_results.db")

    long_rows, meta = [], []
    # standard wide-table experiments
    for name, spec in EXPERIMENTS.items():
        path = SALIDAS / spec["file"]
        if not path.exists():
            print(f"  [skip] {name}: {spec['file']} not found")
            continue
        df = pd.read_csv(path)
        df.to_sql(name, con, if_exists="replace", index=False)
        df.to_parquet(PARQUET / f"{name}.parquet", index=False)
        long_rows += _to_long(name, spec, df)
        meta.append(dict(experiment=name, title=spec["title"],
                         description=spec["desc"], thesis_section=spec["section"],
                         n_rows=len(df), n_metrics=len([c for c in df.columns
                                                        if c not in spec["id"]])))
        print(f"  [ok] {name}: {len(df)} rows")

    # special-format experiments
    r8, kv8 = _exp8_long(); long_rows += r8
    pd.DataFrame([kv8]).to_sql("exp8_qwen_validation", con, if_exists="replace", index=False)
    pd.DataFrame([kv8]).to_parquet(PARQUET / "exp8_qwen_validation.parquet", index=False)
    meta.append(dict(experiment="exp8_qwen_validation",
                     title="FedResPrompt validation on a real Qwen LLM",
                     description="Accuracy before/after and loss for the "
                                 "split-federated prompt-tuning of Qwen2.5-0.5B.",
                     thesis_section="sec:res-fedresprompt", n_rows=1, n_metrics=4))
    print("  [ok] exp8_qwen_validation: 4 metrics")

    r10, df10 = _exp10_long(); long_rows += r10
    df10.to_sql("exp10_benchmarks", con, if_exists="replace", index=False)
    df10.to_parquet(PARQUET / "exp10_benchmarks.parquet", index=False)
    meta.append(dict(experiment="exp10_benchmarks",
                     title="Validation on external benchmark datasets",
                     description="Federated/ensemble/local accuracy and NRMSE on "
                                 "Japanese Vowels, HAR and a multivariate FRED panel.",
                     thesis_section="sec:res-benchmarks",
                     n_rows=len(df10), n_metrics=df10["metric"].nunique()))
    print(f"  [ok] exp10_benchmarks: {len(df10)} rows")

    # consolidated tidy long table + metadata
    long = pd.DataFrame(long_rows)
    long.to_sql("metrics", con, if_exists="replace", index=False)
    long.to_parquet(OUT / "esnfed_metrics_long.parquet", index=False)
    long.to_csv(OUT / "esnfed_metrics_long.csv", index=False)
    mdf = pd.DataFrame(meta)
    mdf.to_sql("experiments", con, if_exists="replace", index=False)
    mdf.to_parquet(PARQUET / "experiments.parquet", index=False)
    con.commit(); con.close()

    print(f"\n[db] {long['experiment'].nunique()} experiments, "
          f"{len(long)} metric rows, {long['metric'].nunique()} distinct metrics")
    print(f"[db] SQLite  -> {(OUT/'esnfed_results.db').relative_to(ROOT)}")
    print(f"[db] Parquet -> {PARQUET.relative_to(ROOT)}/ (+ esnfed_metrics_long.parquet)")


if __name__ == "__main__":
    main()
