# Results database

Every metric reported in the thesis is consolidated into a single database, built
from the raw experiment outputs by
[`experiments/build_results_db.py`](https://github.com/daibeal/esnfed/blob/main/experiments/build_results_db.py)
and shipped in two standard, framework-neutral formats:

- **SQLite** — `results/esnfed_results.db`: one wide table per experiment, a tidy
  `metrics` table, and an `experiments` catalogue. Queryable with plain SQL.
- **Apache Arrow / Parquet** — `results/parquet/*.parquet` and the consolidated
  `results/esnfed_metrics_long.parquet` (read natively by pandas, polars, DuckDB,
  R, …).

The long table follows the **tidy-data** convention — one row per
`experiment × scenario × metric` (**1,162 rows, 30 distinct metrics** across 12
experiments) — so results from every experiment can be queried together.

## Schema

`experiments` — catalogue: `experiment, title, description, thesis_section, n_rows, n_metrics`.

`metrics` — tidy long table:

| column | meaning |
|--------|---------|
| `experiment` | id (e.g. `exp4_ensemble`) |
| `dataset` | `narma10`, `ted_spread`, `har`, `japanese_vowels`, `fred_panel`, … |
| `method` | strategy / topology / model where applicable |
| `n_clients`, `seed` | federation size / seed (nullable) |
| `metric` | `nrmse`, `accuracy`, `fr_bytes`, `fr_flops`, … |
| `value` | numeric value |
| `scenario` | JSON of the remaining identifying columns |

Each experiment also keeps its **full wide table** (e.g. `exp1_esn_sweep`).

## Query it

=== "SQLite + pandas"

    ```python
    import sqlite3, pandas as pd
    con = sqlite3.connect("results/esnfed_results.db")

    pd.read_sql("SELECT * FROM experiments", con)            # what's inside
    pd.read_sql("SELECT * FROM exp4_ensemble", con)          # one experiment

    # federated vs local across every dataset, in one query
    pd.read_sql("SELECT dataset, method, metric, value FROM metrics "
                "WHERE method LIKE 'federated%' OR method LIKE 'local%'", con)
    ```

=== "Parquet"

    ```python
    import pandas as pd
    df = pd.read_parquet("results/esnfed_metrics_long.parquet")   # everything, tidy
    df.query("metric == 'accuracy'")
    ```

=== "DuckDB"

    ```sql
    SELECT dataset, avg(value) FROM 'results/esnfed_metrics_long.parquet'
    WHERE metric = 'nrmse' GROUP BY dataset;
    ```

## Rebuild

```bash
pip install -e ".[experiments]"          # numpy, pandas, pyarrow, ...
python experiments/run_all.py            # regenerate the raw salidas/*.csv
python experiments/build_results_db.py   # rebuild the database
```
