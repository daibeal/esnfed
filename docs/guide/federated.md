# Federated strategies

In an ESN only the linear readout is trained, so federated learning reduces to
learning a shared (or combinable) readout across clients whose reservoirs may or
may not share the same structure.

## Shared reservoir

### Exact federated ridge

Each client computes the ridge sufficient statistics of its local data and sends
only those; the server sums and solves once. **Mathematically identical to
pooled training**, in a single communication round.

```python
from esnfed import federated, topologies, datasets, metrics

parts = datasets.partition_iid(u_tr, y_tr, n_clients=10)
W = topologies.random_reservoir(200, density=0.1, rng=0)
clients, ref = federated.make_shared_clients(W, parts, esn_kwargs=dict(washout=100))

W_out = federated.federated_ridge(clients, ref)
```

!!! success "Verified"
    A unit test asserts `federated_ridge` equals centralized training to within
    `1e-9`. The privacy gain (raw data stays local) costs nothing in accuracy.

Federated ridge holds its accuracy as the federation grows, while local-only
training degrades (interactive — see the [gallery](../gallery.md) for more):

<iframe src="../../plotly/result_exp3_scaling.html" width="100%" height="430" frameborder="0"></iframe>

### Iterative FedAvg

For comparison, the readout can also be trained with iterative
[FedAvg](https://arxiv.org/abs/1602.05629):

```python
W_out, history = federated.fedavg(clients, ref, u_te, y_te, rounds=100)
```

On reservoir readouts this converges slowly (the state covariance is
ill-conditioned) — which is exactly why the closed-form `federated_ridge` is
preferred.

## Heterogeneous reservoirs

When clients hold different reservoirs, parameters are incommensurable. Two
options:

```python
# 1) Prediction ensemble — combine outputs, no parameter averaging
het = federated.make_heterogeneous_clients(reservoirs, parts, esn_kwargs=...)
federated.train_local(het)
pred = federated.ensemble_predict(het, u_te)

# 2) Structural alignment — interpolate reservoirs toward a shared target,
#    then parameter aggregation becomes valid again
records = federated.structural_alignment(local_reservoirs, target, parts, u_te, y_te)
```

!!! info "Finding from the research"
    Under structural heterogeneity the **ensemble is robust** and beats
    local-only training; **parameter aggregation** only becomes competitive once
    the reservoirs are nearly identical. On real, non-stationary financial data,
    local-only training *collapses* while federated ridge stays accurate.

See the [API reference](../api.md#esnfedfederated).
