# Sequence classification

Besides one-step forecasting, `esnfed` does **sequence classification**: each
input is a (possibly variable-length) sequence that maps to one class. The
reservoir turns each sequence into a fixed feature vector (the mean or last
extended state); a ridge readout on one-hot targets then classifies by argmax.

Because the readout is again a ridge regression, the **exact federated scheme
carries over unchanged** — clients exchange the summed statistics `A = FᵀF`,
`B = FᵀY` and the server solves once, recovering the pooled classifier exactly.

```python
import numpy as np
from esnfed import EchoStateNetwork, topologies, datasets
from esnfed import classification as clf

jv = datasets.load_japanese_vowels()          # 9 speakers, 12-d cepstra
W = topologies.random_reservoir(120, density=0.1, rng=0)
esn = EchoStateNetwork(jv.n_features, jv.n_classes, W,
                       spectral_radius=0.9, leaking_rate=0.3, washout=0, ridge=1e-3)

# centralised
W_out = clf.train_classifier(esn, jv.X_train, jv.y_train, jv.n_classes)
acc = clf.accuracy(jv.y_test, clf.predict_labels(esn, jv.X_test, W_out))

# exact federated — one client per speaker, no raw audio shared
clients = datasets.group_clients(jv.X_train, jv.y_train, jv.groups_train)
W_fed = clf.federated_classifier(esn, clients, jv.n_classes)        # == centralised
```

## API

| Function | Purpose |
|----------|---------|
| `reservoir_features(esn, sequences, pool)` | sequence → feature (`pool="mean"`/`"last"`) |
| `train_classifier(esn, X, y, n_classes)` | centralised ridge classifier |
| `federated_classifier(esn, client_data, n_classes)` | exact federated (sum `A_k,B_k`) |
| `ensemble_classify(members, X)` | average softmax over heterogeneous members |
| `predict_labels` / `predict_proba` | argmax / class probabilities |
| `accuracy(y_true, y_pred)` | classification accuracy |

## Heterogeneous clients

When each client has its own reservoir, parameters can't be averaged — combine
predictions instead:

```python
members = [(esn_k, clf.train_classifier(esn_k, X_k, y_k, n_classes)) for ...]
y_pred = clf.ensemble_classify(members, X_test)
```

## Validated on real benchmarks

See [Benchmark datasets](datasets.md#benchmark-classification-datasets). Measured
results (one client per natural group, no raw data shared):

| Dataset | Task | Clients | Federated | Local-only |
|---------|------|--------:|----------:|-----------:|
| Japanese Vowels | speaker ID (9-class) | 9 speakers | **0.98** | ~0.11 (chance) |
| HAR Smartphones | activity (6-class) | 21 subjects | **0.92** | 0.66 |

In both cases the federated classifier is **numerically identical** to centralised
training on the pooled data; on Japanese Vowels the per-speaker split is extreme
label skew, so local-only training cannot work at all — federation is essential.
On HAR, a prediction ensemble over *heterogeneous* per-subject reservoirs reaches
≈0.84, between local-only and the shared-reservoir federated classifier.
