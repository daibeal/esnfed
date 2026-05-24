# ---
# jupyter:
#   jupytext:
#     text_representation: {extension: .py, format_name: percent}
#   kernelspec: {display_name: Python (Pyodide), language: python, name: python}
# ---

# %% [markdown]
# # Explore the results database — live
#
# Query **every metric from the thesis** with real SQL, right here in your
# browser (JupyterLite + Pyodide — no install, no server). The SQLite database is
# bundled with this notebook. Run each cell with **Shift+Enter** and edit the
# queries freely.

# %%
%pip install -q pandas
import sqlite3, pandas as pd
con = sqlite3.connect("esnfed_results.db")   # bundled alongside this notebook
print("connected ·", pd.read_sql("SELECT count(*) AS n FROM metrics", con).n[0], "metric rows")

# %% [markdown]
# ## What's inside

# %%
pd.read_sql("SELECT experiment, title, n_rows, n_metrics FROM experiments", con)

# %% [markdown]
# ## Federated vs local, across every dataset
#
# One query spans all experiments thanks to the tidy `metrics` table.

# %%
pd.read_sql("""
    SELECT dataset, method, metric, ROUND(value, 3) AS value
    FROM metrics
    WHERE method LIKE 'federated%' OR method LIKE 'local%'
    ORDER BY dataset, method
""", con)

# %% [markdown]
# ## A whole experiment, wide form

# %%
pd.read_sql("SELECT * FROM exp4_ensemble", con)

# %% [markdown]
# ## Plot a query result
#
# FRED panel: federated stays low while local-only collapses as the federation grows.

# %%
%pip install -q matplotlib
import matplotlib.pyplot as plt
df = pd.read_sql("""
    SELECT n_clients, method, value FROM metrics
    WHERE experiment='exp10_benchmarks' AND metric='nrmse'
""", con)
piv = df.pivot(index="n_clients", columns="method", values="value")
ax = piv.plot(marker="o", logy=True, figsize=(6, 3.6))
ax.set_ylabel("test NRMSE (log)"); ax.set_xlabel("institutions")
ax.set_title("FRED panel — federated vs local"); plt.tight_layout(); plt.show()

# %% [markdown]
# ## Your turn
#
# Edit the SQL below — e.g. list the distinct metrics, or filter by `dataset`.

# %%
pd.read_sql("SELECT DISTINCT metric FROM metrics ORDER BY metric", con)
