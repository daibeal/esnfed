# esnfed — a federated reservoir computing toolkit

`esnfed` is a small, dependency-light Python library for training **Echo State
Networks** (reservoir computing) in a **federated** setting — where several
parties jointly train a model without sharing their raw data. It accompanies the
Final Degree Project *Ensemble of Recurrent Networks for Federated Learning*
(ETSINF, Universitat Politècnica de València).

The core library depends only on **NumPy** and **NetworkX**.

> **Why this and not ReservoirPy?** [ReservoirPy](https://reservoirpy.readthedocs.io)
> is the mature, full-featured library for *building and tuning* reservoir
> computing models, and `esnfed` does **not** try to replace it. `esnfed`
> focuses on the part ReservoirPy does not cover: **federating** reservoir
> models across parties. You can design a reservoir in ReservoirPy and federate
> it here in one line (see *Interoperability* below).

## Gallery

All four plots below are produced by `esnfed.viz` (see *Visualising your ESN*).

| Reservoir topology | Eigenvalue spectrum |
|:---:|:---:|
| ![Reservoir topology graph](https://raw.githubusercontent.com/daibeal/esnfed/main/docs/images/reservoir.png) | ![Eigenvalue spectrum](https://raw.githubusercontent.com/daibeal/esnfed/main/docs/images/spectrum.png) |
| **Reservoir activations** | **Forecast vs. actual** |
| ![Reservoir state activations](https://raw.githubusercontent.com/daibeal/esnfed/main/docs/images/states.png) | ![Forecast vs actual](https://raw.githubusercontent.com/daibeal/esnfed/main/docs/images/forecast.png) |

## Features

- **Echo State Network** with leaky-integrator neurons and a closed-form ridge
  readout (`EchoStateNetwork`); a **deep / hierarchical** stack
  (`DeepEchoStateNetwork`); heterogeneous leaking rates and mixed activations.
- **Federated strategies**:
  - `federated_ridge` — *exact* federated training for a shared reservoir
    (clients exchange only ridge sufficient statistics; provably equal to pooled
    training, in one communication round);
  - `fedavg` — iterative FedAvg on the readout;
  - `ensemble_predict` — prediction ensemble for *heterogeneous* reservoirs;
  - `structural_alignment` — interpolate reservoirs toward a shared structure.
- **Privacy & continual learning** — all on the same sufficient statistics:
  - `federated_ridge_dp` — `(ε, δ)`-**differential privacy** (analytic Gaussian
    mechanism);
  - `federated_ridge_secure` — **secure aggregation** (additive masking; the
    server only sees the sum);
  - `StreamingRidge` / `RLSReadout` — **incremental / online** ridge for
    streaming and continual federated learning.
- **Sequence classification** (`classification`) with the same *exact* federated
  aggregation (e.g. speaker ID, activity recognition).
- **Reservoir topologies**: Erdős–Rényi, small-world, scale-free, ring.
- **Data**: synthetic benchmarks (NARMA-10, Mackey-Glass, Lorenz); real series — a
  bundled counterparty-risk series (the TED spread) and FRED loaders (`load_fred`,
  multivariate `load_fred_matrix`); and UCI sequence-classification benchmarks
  (`load_japanese_vowels`, `load_har`).
- **Interoperability**: adapters for [ReservoirPy](https://reservoirpy.readthedocs.io)
  reservoirs and an example integration with the
  [Flower](https://flower.ai) federated-learning framework.

## Installation

```bash
pip install esnfed                  # core (numpy + networkx)
pip install "esnfed[viz]"           # + plotly/matplotlib/seaborn plots
pip install "esnfed[experiments]"   # + matplotlib/pandas/scipy/scikit-learn
pip install "esnfed[reservoirpy]"   # + ReservoirPy interop
pip install "esnfed[flower]"        # + Flower integration
```

## Quick start

### Federated counterparty-risk forecasting (real data)

Several institutions jointly forecast the **TED spread** (a classic gauge of
interbank/counterparty credit risk) without sharing their data. Exact federated
ridge equals pooled training in a single round:

```python
from esnfed import datasets, federated, topologies, metrics

u, y = datasets.load_ted_spread()                 # bundled real series (FRED)
u_tr, y_tr, u_te, y_te = datasets.split(u, y, 0.7)
parts = datasets.partition_iid(u_tr, y_tr, n_clients=8)   # 8 "institutions"

W = topologies.random_reservoir(200, density=0.1, rng=0)
esn_kw = dict(spectral_radius=0.9, leaking_rate=0.5, washout=100, ridge=1e-6)
clients, ref = federated.make_shared_clients(W, parts, input_seed=0, esn_kwargs=esn_kw)

W_out = federated.federated_ridge(clients, ref)   # one round, exact, private
Z_test = ref.harvest(u_te)[ref.washout:]
print("federated NRMSE:", metrics.nrmse(y_te[ref.washout:], Z_test @ W_out))
```

Need another series? `datasets.load_fred("BAMLH0A0HYM2")` pulls a high-yield
credit spread from FRED; `datasets.from_array(my_series)` or
`datasets.load_csv("my.csv")` turn any series into a forecasting task.

### Train a single ESN

```python
import numpy as np
from esnfed import EchoStateNetwork, datasets, topologies, metrics

u, y = datasets.narma10(3000, rng=0)
u_tr, y_tr, u_te, y_te = datasets.split(u, y)
W = topologies.random_reservoir(200, density=0.1, rng=0)
esn = EchoStateNetwork(1, 1, W, spectral_radius=0.9, washout=100).fit(u_tr, y_tr)
print("NRMSE:", metrics.nrmse(y_te[100:], esn.predict(u_te)[100:]))
```

### Differential privacy & secure aggregation

Harden the federated exchange — the same `(A, B)` statistics, now with a formal
privacy guarantee, or hidden from the server:

```python
from esnfed import federated
from esnfed.privacy import PrivacyConfig

# (ε, δ)-differentially private readout (clip + analytic Gaussian mechanism)
cfg = PrivacyConfig(epsilon=1.0, delta=1e-5, clip_state=5.0, clip_target=1.0)
W_dp = federated.federated_ridge_dp(clients, ref, cfg)

# secure aggregation: the server only ever sees the masked sum of clients
W_secure = federated.federated_ridge_secure(clients, ref, seed=0)
```

## Interoperability

### ReservoirPy — design there, federate here

```python
from reservoirpy.nodes import Reservoir
from esnfed import interop, datasets, federated

res = Reservoir(200, sr=0.9, lr=0.5, input_dim=1)   # design/tune in ReservoirPy
esn = interop.to_esn(res, n_inputs=1)               # -> esnfed EchoStateNetwork
W = interop.reservoir_matrix(res)                   # ... then federate it
```

### Flower — a real FL framework

`examples/flower_federated_ridge.py` shows the exact federated-ridge scheme as a
Flower `NumPyClient` + custom `Strategy`: each client's `fit` returns its ridge
sufficient statistics and the server sums them and solves once. The Flower-routed
result is identical to `federated_ridge` to numerical precision.

## Visualising your ESN

The optional `esnfed.viz` module (`pip install "esnfed[viz]"`) provides four
plots. The default backend is **Plotly** (interactive); pass
`backend="matplotlib"` or `backend="seaborn"` for static figures. Each function
returns the native figure object.

```python
from esnfed import EchoStateNetwork, topologies, viz

W = topologies.small_world_reservoir(100, k=6, p=0.1, rng=0)
esn = EchoStateNetwork(1, 1, W, spectral_radius=0.9).fit(u_tr, y_tr)

viz.plot_reservoir(esn).show()           # connectivity graph (degree-coloured)
viz.plot_spectrum(esn).show()            # eigenvalues + unit circle + ρ
viz.plot_states(esn, u_te).show()        # reservoir activations over time
viz.plot_forecast(y_te, esn.predict(u_te), washout=100).show()
viz.save(viz.plot_spectrum(esn), "spectrum.html")   # or .png (needs kaleido)
```

| Function | Shows |
|----------|-------|
| `plot_reservoir` | reservoir connectivity as a network graph |
| `plot_spectrum` | eigenvalues in the complex plane (echo state property) |
| `plot_states` | a sample of reservoir activations over time |
| `plot_forecast` | predicted vs. actual with NRMSE |

## Modules

| Module | Contents |
|--------|----------|
| `esnfed.esn` | `EchoStateNetwork`, ridge helpers |
| `esnfed.deep` | `DeepEchoStateNetwork` (stacked reservoirs) |
| `esnfed.topologies` | reservoir generators + `graph_metrics` |
| `esnfed.datasets` | NARMA-10, Mackey-Glass, Lorenz; `from_array`, `load_csv`, `load_ted_spread`, `load_fred`, `load_fred_matrix`, `load_japanese_vowels`, `load_har` |
| `esnfed.metrics` | `nrmse`, `rmse`, `mse`, `mae`, `r2_score` |
| `esnfed.federated` | `Client`, `federated_ridge`, `fedavg`, `ensemble_predict`, `structural_alignment`, `federated_ridge_dp`, `federated_ridge_secure` |
| `esnfed.privacy` | differential privacy (`dp_statistics`, `gaussian_sigma`) + secure aggregation (`secure_sum`) |
| `esnfed.streaming` | `StreamingRidge`, `RLSReadout` (incremental / online ridge) |
| `esnfed.classification` | sequence classification + exact federated / ensemble variants |
| `esnfed.interop` | ReservoirPy adapters (`to_esn`, `reservoir_matrix`) |
| `esnfed.viz` | `plot_reservoir`, `plot_spectrum`, `plot_states`, `plot_forecast` |
| `esnfed.llm_orchestration` | experimental FedResPrompt (reservoir soft-prompt control of a frozen LLM) |

## Reproducing the research experiments

```bash
pip install -e ".[experiments,reservoirpy,flower,viz]"
python -m pytest                    # 499 tests
python experiments/run_all.py       # synthetic-benchmark figures and tables
python experiments/exp6_finance.py  # federated counterparty-risk (TED spread)
```

## Correctness

The library is written for producing research results, so it is built to fail
loudly rather than return a plausible-looking wrong number:

- **Shapes are checked, not coerced.** A `(T,)` target compared against a `(T, 1)`
  prediction, or a transposed `(n_inputs, T)` input, raises instead of silently
  broadcasting or reinterleaving the data.
- **Degenerate configurations are rejected** at construction — a washout longer
  than the sequence, a non-positive leaking rate, a negative spectral radius, a
  zero-sum weight vector.
- **Numerical claims are tested as invariants**, not as golden numbers: exactness
  of federated aggregation, streaming/batch equivalence, the echo state property,
  permutation invariance of reservoir node labelling, and the `(ε, δ)` guarantee
  of the analytic Gaussian mechanism.

See [`CHANGELOG.md`](https://github.com/daibeal/esnfed/blob/main/CHANGELOG.md)
for the defects this uncovered, and the note on
what *exact* federated ridge means precisely (it is exact per partition; the
readout itself is limited by the conditioning of the Gram matrix).

## Releasing

Releases are published to PyPI by pushing a version tag; the workflow verifies
that the tag matches `esnfed.__version__`, runs the test suite, lints, builds and
uploads via PyPI Trusted Publishing (no API token is stored in the repository).

```bash
# 1. bump __version__ in esnfed/__init__.py and update CHANGELOG.md
# 2. tag and push
git tag v1.7.0 && git push origin v1.7.0
```

A one-time setup is needed on PyPI before the first automated release — add a
trusted publisher at <https://pypi.org/manage/project/esnfed/settings/publishing/>
with workflow `release.yml` and environment `pypi`. Until then, publish manually:

```bash
python -m build && python -m twine upload dist/*
```

## Data attribution

The bundled TED spread is sourced from the Federal Reserve Bank of St. Louis
(FRED, series `TEDRATE`) and is used for demonstration under FRED's terms.

## Citation

```bibtex
@thesis{benites2026esnfed,
  author = {Benites Aldaz, Dairon Andres},
  title  = {Ensemble of Recurrent Networks for Federated Learning},
  school = {Universitat Politecnica de Valencia (ETSINF)},
  year   = {2026},
  type   = {Bachelor's thesis},
}
```

## License

MIT — see [LICENSE](https://github.com/daibeal/esnfed/blob/main/LICENSE).
