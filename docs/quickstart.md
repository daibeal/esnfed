# Quickstart

## Train a single Echo State Network

An ESN has a large, fixed, random *reservoir* and a single trained linear
*readout*. Training is a one-shot ridge regression — no backprop.

```python
import numpy as np
from esnfed import EchoStateNetwork, datasets, topologies, metrics

# A benchmark task
u, y = datasets.narma10(3000, rng=0)
u_tr, y_tr, u_te, y_te = datasets.split(u, y, train_frac=0.7)

# Build a reservoir and an ESN, then fit the readout
W = topologies.random_reservoir(200, density=0.1, rng=0)
esn = EchoStateNetwork(1, 1, W, spectral_radius=0.9, leaking_rate=1.0, washout=100)
esn.fit(u_tr, y_tr)

pred = esn.predict(u_te)
print("NRMSE:", metrics.nrmse(y_te[100:], pred[100:]))
```

## Federate it — exactly

With a shared reservoir, each client sends only the **sufficient statistics** of
its local ridge problem (`A = ZᵀZ`, `B = ZᵀY`). The server sums them and solves
once. The result is *identical* to pooling all the data centrally — but no raw
data ever leaves a client.

```python
from esnfed import datasets, federated, topologies, metrics

u, y = datasets.load_ted_spread()                  # bundled real risk series
u_tr, y_tr, u_te, y_te = datasets.split(u, y, 0.7)
parts = datasets.partition_iid(u_tr, y_tr, n_clients=10)   # 10 "institutions"

W = topologies.random_reservoir(200, density=0.1, rng=0)
esn_kw = dict(spectral_radius=0.9, leaking_rate=0.5, washout=100, ridge=1e-6)
clients, ref = federated.make_shared_clients(W, parts, input_seed=0, esn_kwargs=esn_kw)

W_out = federated.federated_ridge(clients, ref)    # one round, exact, private
Z_test = ref.harvest(u_te)[ref.washout:]
print("Federated NRMSE:", metrics.nrmse(y_te[ref.washout:], Z_test @ W_out))
```

## Heterogeneous clients — ensemble

When clients hold *different* reservoirs, parameters can't be averaged — but
predictions can.

```python
kinds = ["random", "small_world", "scale_free", "ring"]
reservoirs = [topologies.make_reservoir(kinds[i % 4], 200, rng=i) for i in range(len(parts))]
het = federated.make_heterogeneous_clients(reservoirs, parts, esn_kwargs=esn_kw)
federated.train_local(het)
pred = federated.ensemble_predict(het, u_te)
print("Ensemble NRMSE:", metrics.nrmse(y_te[100:], pred[100:]))
```

Continue to the [User guide](guide/esn.md) for the details of each piece, or
jump to the [Examples](examples.md).
