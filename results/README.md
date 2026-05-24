# esnfed results database

A single, consolidated database with **every experimental metric** reported in
the thesis *Ensemble of Recurrent Networks for Federated Learning*. It is
generated from the raw experiment outputs in `../salidas/` by
[`experiments/build_results_db.py`](../src/python/experiments/build_results_db.py)
and is provided in two standard, framework-neutral formats.

## Files

| File | Format | Contents |
|------|--------|----------|
| `esnfed_results.db` | **SQLite** | one wide table per experiment + a tidy `metrics` table + an `experiments` metadata table |
| `parquet/*.parquet` | **Apache Arrow / Parquet** | the same per-experiment tables, one file each |
| `esnfed_metrics_long.parquet` / `.csv` | Parquet / CSV | the consolidated tidy (long) table |

SQLite and Parquet are read natively by the whole data-science / ML stack
(pandas, polars, DuckDB, Spark, R, Julia, …). The long table follows the
**tidy-data** convention: one row per `experiment × scenario × metric`.

## Schema

**`experiments`** — one row per experiment:
`experiment, title, description, thesis_section, n_rows, n_metrics`.

**`metrics`** (tidy long, 1162 rows, 30 distinct metrics) — every measurement:

| column | meaning |
|--------|---------|
| `experiment` | experiment id (e.g. `exp4_ensemble`) |
| `dataset` | task/dataset (`narma10`, `ted_spread`, `har`, `japanese_vowels`, …) |
| `method` | strategy / topology / model where applicable |
| `n_clients`, `seed` | federation size / random seed (nullable) |
| `metric` | metric name (`nrmse`, `accuracy`, `fr_bytes`, …) |
| `value` | the numeric value |
| `scenario` | JSON of the remaining identifying columns |

Each experiment also keeps its **full wide table** (e.g. `exp1_esn_sweep`) for
detailed queries.

## Quick start

```python
import sqlite3, pandas as pd
con = sqlite3.connect("results/esnfed_results.db")

pd.read_sql("SELECT * FROM experiments", con)                       # what's inside
pd.read_sql("SELECT * FROM exp4_ensemble", con)                     # one experiment
pd.read_sql("SELECT dataset, method, metric, value FROM metrics "   # cross-experiment
            "WHERE metric='accuracy'", con)
```

```python
import pandas as pd
df = pd.read_parquet("results/esnfed_metrics_long.parquet")         # everything, tidy
```

## Reproduce

```bash
python experiments/run_all.py            # regenerate salidas/*.csv (needs deps)
python experiments/build_results_db.py   # rebuild this database
```
