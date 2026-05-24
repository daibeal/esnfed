"""Federated sequence classification with Echo State Networks (self-contained).

A synthetic toy: each "speaker" (client) emits variable-length 2-D sequences whose
oscillation frequency encodes its class. The federated classifier sums each
client's sufficient statistics and equals centralized training -- no raw sequence
is shared. This is the same exact-aggregation as ``federated_ridge``, applied to
classification (cf. the Japanese Vowels / HAR benchmarks).

Run:  python examples/classification_demo.py
"""
import numpy as np

from esnfed import EchoStateNetwork, topologies
from esnfed.classification import accuracy, federated_classifier, predict_labels


def make_client(label: int, n_seq: int, rng) -> list:
    """``n_seq`` variable-length 2-D sequences whose frequency encodes the class."""
    freq = 0.10 + 0.12 * label
    seqs = []
    for _ in range(n_seq):
        T = int(rng.integers(15, 30))
        t = np.arange(T)
        phase = rng.uniform(0, 2 * np.pi)
        x = np.stack([np.sin(freq * t + phase), np.cos(freq * t + phase)], axis=1)
        seqs.append(x + 0.05 * rng.standard_normal(x.shape))
    return seqs


def main() -> None:
    rng = np.random.default_rng(0)
    n_classes = 4
    W = topologies.random_reservoir(80, density=0.1, rng=rng)
    esn = EchoStateNetwork(2, n_classes, W, spectral_radius=0.9, washout=0, ridge=1e-3)

    # one client per class (extreme label skew: client identity == label)
    client_data, X_test, y_test = [], [], []
    for c in range(n_classes):
        seqs = make_client(c, 40, rng)
        client_data.append((seqs[:30], np.full(30, c)))   # local training set
        X_test += seqs[30:]
        y_test += [c] * 10                                 # pooled, held-out test
    y_test = np.asarray(y_test)

    W_out = federated_classifier(esn, client_data, n_classes)
    acc = accuracy(y_test, predict_labels(esn, X_test, W_out))
    print(f"federated classifier accuracy = {acc:.3f}   "
          f"({n_classes} clients / classes, exact = centralized)")


if __name__ == "__main__":
    main()
