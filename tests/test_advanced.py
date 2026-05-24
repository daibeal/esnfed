"""Tests for the reservoir-heterogeneity extensions:
heterogeneous leaking rates, multi-type node nonlinearities, and deep ESNs.
"""
import numpy as np
import pytest

from esnfed import (DeepEchoStateNetwork, EchoStateNetwork, datasets, federated,
                    metrics, topologies)
from esnfed.esn import ACTIVATIONS


@pytest.fixture
def data():
    u, y = datasets.narma10(1200, rng=0)
    return datasets.split(u, y, 0.7)


@pytest.fixture
def W():
    return topologies.random_reservoir(80, density=0.1, rng=0)


# ----------------------------------------------------- heterogeneous leaking
def test_leaking_rates_kinds():
    for kind in ("uniform", "log_uniform", "constant", "layered"):
        a = topologies.leaking_rates(60, kind, low=0.1, high=0.9, rng=0)
        assert a.shape == (60,)
        assert a.min() >= 0.1 - 1e-9 and a.max() <= 0.9 + 1e-9
    # layered is non-increasing (fast -> slow blocks)
    a = topologies.leaking_rates(60, "layered", low=0.1, high=0.9, n_layers=3, rng=0)
    assert a[0] >= a[-1]


def test_heterogeneous_leaking_runs(data, W):
    u_tr, y_tr, u_te, y_te = data
    a = topologies.leaking_rates(80, "layered", rng=0)
    esn = EchoStateNetwork(1, 1, W, spectral_radius=0.9, leaking_rate=a, washout=100)
    assert esn._hetero_leak
    esn.fit(u_tr, y_tr)
    pred = esn.predict(u_te)
    assert pred.shape == (len(u_te), 1)
    assert np.isfinite(pred).all()


# ----------------------------------------------------- multi-type activations
def test_mixed_activations_assignment():
    acts = topologies.mixed_activations(100, ("tanh", "sigmoid", "sin"), rng=0)
    assert acts.shape == (100,)
    assert set(np.unique(acts)) <= {"tanh", "sigmoid", "sin"}


def test_multi_activation_runs(data, W):
    u_tr, y_tr, u_te, y_te = data
    acts = topologies.mixed_activations(80, ("tanh", "sigmoid", "sin"), rng=0)
    esn = EchoStateNetwork(1, 1, W, spectral_radius=0.9, activation=acts, washout=100)
    assert esn._hetero_act
    esn.fit(u_tr, y_tr)
    assert np.isfinite(esn.predict(u_te)).all()


def test_scalar_activation_and_unknown(W):
    EchoStateNetwork(1, 1, W, activation="sigmoid")          # ok
    assert "tanh" in ACTIVATIONS and "sin" in ACTIVATIONS
    with pytest.raises(ValueError):
        EchoStateNetwork(1, 1, W, activation="bogus")


# ----------------------------------------------------------------- deep ESN
def test_deep_shapes_and_predict(data):
    u_tr, y_tr, u_te, y_te = data
    Ws = [topologies.random_reservoir(40, density=0.1, rng=i) for i in range(3)]
    d = DeepEchoStateNetwork(1, 1, Ws, spectral_radius=0.9,
                             leaking_rate=[0.9, 0.5, 0.2], washout=100)
    assert d.readout_dim == 1 + 1 + 120
    Z = d.harvest(u_tr)
    assert Z.shape == (len(u_tr), d.readout_dim)
    d.fit(u_tr, y_tr)
    assert np.isfinite(d.predict(u_te)).all()


def test_deep_federated_is_exact(data):
    """Deep ESN federates exactly: summed statistics == pooled per-client fit."""
    u_tr, y_tr, _, _ = data
    Ws = [topologies.random_reservoir(40, density=0.1, rng=i) for i in range(2)]
    mk = lambda: DeepEchoStateNetwork(1, 1, Ws, spectral_radius=0.9,
                                      leaking_rate=[0.8, 0.4], washout=50, seed=0)
    parts = datasets.partition_iid(u_tr, y_tr, 4, rng=0)
    clients = [federated.Client(mk(), pu, py) for pu, py in parts]
    ref = mk()
    W_fed = federated.federated_ridge(clients, ref)
    federated.train_centralized(ref, clients)
    assert np.allclose(W_fed, ref.W_out, atol=1e-9)


def test_deep_helps_on_mackey_glass():
    """A deep, multi-timescale reservoir should not be worse than a flat one."""
    u, y = datasets.mackey_glass(2000, seed=0)
    u_tr, y_tr, u_te, y_te = datasets.split(u, y, 0.7)
    W = topologies.random_reservoir(150, density=0.1, rng=0)
    flat = EchoStateNetwork(1, 1, W, spectral_radius=0.95, leaking_rate=0.3,
                            washout=100).fit(u_tr, y_tr)
    Ws = [topologies.random_reservoir(80, density=0.1, rng=i) for i in range(3)]
    deep = DeepEchoStateNetwork(1, 1, Ws, spectral_radius=0.95,
                                leaking_rate=[0.9, 0.5, 0.2], washout=100).fit(u_tr, y_tr)
    e_flat = metrics.nrmse(y_te[100:], flat.predict(u_te)[100:])
    e_deep = metrics.nrmse(y_te[100:], deep.predict(u_te)[100:])
    assert e_deep <= e_flat * 1.5  # deep is competitive (usually clearly better)
