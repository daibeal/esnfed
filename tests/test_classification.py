"""Tests for sequence classification and its federated variants.

Uses synthetic class-structured sequences (no network), so it is fast and
deterministic. The key invariant is that the federated classifier equals the
centralised one exactly, mirroring the regression case.
"""
import numpy as np
import pytest

from esnfed import EchoStateNetwork, topologies
from esnfed import classification as clf


def make_data(n_per=20, n_classes=3, seed=0):
    """Variable-length 1-D sequences whose class sets the sinusoid frequency."""
    rng = np.random.default_rng(seed)
    X, y = [], []
    for c in range(n_classes):
        freq = 0.1 + 0.3 * c
        for _ in range(n_per):
            T = int(rng.integers(15, 30))
            t = np.arange(T)
            sig = np.sin(2 * np.pi * freq * t) + 0.1 * rng.standard_normal(T)
            X.append(sig.reshape(-1, 1))
            y.append(c)
    return X, np.asarray(y)


@pytest.fixture
def esn():
    W = topologies.random_reservoir(80, density=0.1, rng=0)
    return EchoStateNetwork(1, 3, W, spectral_radius=0.9, leaking_rate=0.3,
                            washout=0, ridge=1e-3, seed=0)


def test_one_hot():
    Y = clf.one_hot([0, 2, 1], 3)
    assert Y.shape == (3, 3)
    assert Y[0, 0] == 1 and Y[1, 2] == 1 and Y[2, 1] == 1
    assert Y.sum() == 3


def test_reservoir_features_shape(esn):
    X, _ = make_data()
    assert clf.reservoir_features(esn, X).shape == (len(X), esn.readout_dim)


def test_classifier_learns(esn):
    Xtr, ytr = make_data(seed=0)
    Xte, yte = make_data(seed=1)
    W = clf.train_classifier(esn, Xtr, ytr, 3)
    assert clf.accuracy(yte, clf.predict_labels(esn, Xte, W)) > 0.6  # >> chance 0.33


def test_federated_equals_centralized(esn):
    """Federated classification recovers the pooled classifier exactly."""
    X, y = make_data(seed=0)
    Wc = clf.train_classifier(esn, X, y, 3)
    # one client per class -> extreme label skew; exact aggregation still holds
    clients = [([X[i] for i in range(len(X)) if y[i] == c], y[y == c])
               for c in range(3)]
    Wf = clf.federated_classifier(esn, clients, 3)
    assert np.allclose(Wc, Wf, atol=1e-9)


def test_predict_proba_normalized(esn):
    X, y = make_data()
    W = clf.train_classifier(esn, X, y, 3)
    P = clf.predict_proba(esn, X, W)
    assert P.shape == (len(X), 3)
    assert np.allclose(P.sum(axis=1), 1.0)


def test_ensemble_beats_chance(esn):
    Xtr, ytr = make_data(seed=0)
    Xte, yte = make_data(seed=1)
    members = []
    for s in range(3):
        Wi = topologies.make_reservoir("random", 80, rng=s)
        e = EchoStateNetwork(1, 3, Wi, spectral_radius=0.9, leaking_rate=0.3,
                             washout=0, ridge=1e-3, seed=s)
        members.append((e, clf.train_classifier(e, Xtr, ytr, 3)))
    pred = clf.ensemble_classify(members, Xte)
    assert pred.shape == (len(Xte),)
    assert clf.accuracy(yte, pred) > 0.5


def test_last_pool_works(esn):
    X, y = make_data()
    F = clf.reservoir_features(esn, X, pool="last")
    assert F.shape == (len(X), esn.readout_dim)
    with pytest.raises(ValueError):
        clf.reservoir_features(esn, X, pool="bogus")
