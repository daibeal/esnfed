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

## High-dimensional multivariate (FRED panel)

`load_fred_matrix` aligns several FRED series on their common dates into a
multivariate task — forecast the next value of a `target` series from the whole
panel. This exercises high-dimensional input scaling (a larger reservoir helps as
`d` grows).

```python
# forecast counterparty risk (TED spread) from a 4-series financial panel
u, y = datasets.load_fred_matrix(["TEDRATE", "VIXCLS", "DGS10", "DFF"])
print(u.shape)   # (T-1, 4)  ->  d_in = 4
```

## Benchmark classification datasets

Two standard sequence-classification benchmarks, each with a **natural client
split**, are downloaded on demand and cached. They return a `SequenceDataset`
(`X_*` lists of `(T_i, n_features)` sequences, integer labels, and `groups_*`
giving the federation unit). See [Sequence classification](classification.md).

```python
jv = datasets.load_japanese_vowels()   # UCI 128: 9 speakers, 12-d cepstra
har = datasets.load_har()              # UCI 240: 30 subjects, 6 activities, 9 channels

# one client per natural group (speaker / subject)
clients = datasets.group_clients(jv.X_train, jv.y_train, jv.groups_train)
```

| Loader | Task | Natural clients | Heterogeneity |
|--------|------|-----------------|---------------|
| `load_japanese_vowels` | speaker ID (9 classes) | 9 speakers (= labels) | extreme **label skew** |
| `load_har` | activity (6 classes) | 30 subjects | **feature** non-i.i.d. |

## Splitting and partitioning

```python
u_tr, y_tr, u_te, y_te = datasets.split(u, y, train_frac=0.7)   # chronological
parts = datasets.partition_iid(u_tr, y_tr, n_clients=10)        # contiguous blocks
```

Contiguous blocks keep each client's slice a valid time series for state
harvesting. See the [API reference](../api.md#esnfeddatasets).
