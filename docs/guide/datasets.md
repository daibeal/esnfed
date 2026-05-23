# Datasets

`esnfed.datasets` provides synthetic reservoir-computing benchmarks, a bundled
real-world series, and loaders to bring your own data.

## Synthetic benchmarks

```python
from esnfed import datasets

u, y = datasets.narma10(3000, rng=0)         # NARMA-10 (memory + nonlinearity)
u, y = datasets.mackey_glass(3000, seed=0)   # Mackey-Glass (mildly chaotic)
u, y = datasets.lorenz(3000, seed=0)         # Lorenz x-coordinate (normalised)
```

## Real data: counterparty risk

A real series ships with the package — the **TED spread**, the gap between the
3-month interbank rate and the 3-month Treasury bill, a classic gauge of
interbank/counterparty credit risk (daily, 1986–2022, source: FRED `TEDRATE`).

```python
u, y = datasets.load_ted_spread()        # normalised one-step-ahead task
raw  = datasets.load_ted_spread(raw=True) # the raw spread, in percentage points
```

## Bring your own data

```python
# any 1-D series -> one-step-ahead (or next-change) forecasting task
u, y = datasets.from_array(my_series, predict="next", normalize=True)

# a column of a CSV file
u, y = datasets.load_csv("prices.csv", column="close")

# any FRED series, downloaded on demand and cached
u, y = datasets.load_fred("BAMLH0A0HYM2")   # US high-yield credit spread
```

## Splitting and partitioning

```python
u_tr, y_tr, u_te, y_te = datasets.split(u, y, train_frac=0.7)   # chronological
parts = datasets.partition_iid(u_tr, y_tr, n_clients=10)        # contiguous blocks
```

Contiguous blocks keep each client's slice a valid time series for state
harvesting. See the [API reference](../api.md#esnfeddatasets).
