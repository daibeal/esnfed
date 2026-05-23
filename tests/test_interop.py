"""Tests for ReservoirPy interoperability (skipped if reservoirpy is absent)."""
import numpy as np
import pytest

from esnfed import datasets, federated, interop, metrics

rpy = pytest.importorskip("reservoirpy")
from reservoirpy.nodes import Reservoir  # noqa: E402


def test_reservoir_matrix_shape():
    res = Reservoir(60, sr=0.9, input_dim=1)
    W = interop.reservoir_matrix(res, n_inputs=1)
    assert W.shape == (60, 60)
    assert np.isfinite(W).all()


def test_input_matrix_has_bias_column():
    res = Reservoir(40, sr=0.9, input_dim=1)
    Win = interop.input_matrix(res, n_inputs=1)
    # esnfed convention: (units, 1 + n_inputs), column 0 = bias
    assert Win.shape == (40, 2)


def test_to_esn_preserves_dynamics_and_trains():
    res = Reservoir(150, sr=0.95, lr=0.4, input_dim=1)
    esn = interop.to_esn(res, n_inputs=1, n_outputs=1, washout=80, ridge=1e-7)
    assert esn.n_reservoir == 150
    assert esn.leaking_rate == pytest.approx(0.4)
    # spectral radius preserved from the ReservoirPy reservoir (~0.95)
    from esnfed.esn import _spectral_radius
    assert _spectral_radius(esn.W) == pytest.approx(0.95, abs=0.05)

    u, y = datasets.narma10(2000, rng=0)
    utr, ytr, ute, yte = datasets.split(u, y)
    esn.fit(utr, ytr)
    err = metrics.nrmse(yte[80:], esn.predict(ute)[80:])
    assert err < 0.8  # a ReservoirPy-built ESN trains fine through esnfed


def test_federate_a_reservoirpy_reservoir():
    res = Reservoir(150, sr=0.9, lr=1.0, input_dim=1)
    W = interop.reservoir_matrix(res)
    u, y = datasets.narma10(3000, rng=1)
    utr, ytr, ute, yte = datasets.split(u, y)
    parts = datasets.partition_iid(utr, ytr, 5, rng=1)
    kw = dict(spectral_radius=0.9, leaking_rate=1.0, washout=80, ridge=1e-7)
    clients, ref = federated.make_shared_clients(W, parts, input_seed=0, esn_kwargs=kw)
    W_out = federated.federated_ridge(clients, ref)
    Z_test = ref.harvest(ute)[80:]
    assert metrics.nrmse(yte[80:], Z_test @ W_out) < 0.8
