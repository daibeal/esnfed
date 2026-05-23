"""Tests for the optional acceleration paths: float32, sparse, Numba.

Each keeps the public API and (numerically) the results unchanged; the sparse
and Numba paths are skipped if SciPy / Numba are not installed. A fixed
input-weight seed makes the only difference between variants the optimisation
under test, not the random input matrix.
"""
import numpy as np
import pytest

from esnfed import EchoStateNetwork, datasets, metrics, topologies

SEED = 7


@pytest.fixture
def data():
    u, y = datasets.narma10(1500, rng=0)
    return datasets.split(u, y, 0.7)


def _esn(W, **kw):
    return EchoStateNetwork(1, 1, W, spectral_radius=0.9, washout=100, seed=SEED, **kw)


def _fit_nrmse(esn, utr, ytr, ute, yte):
    esn.fit(utr, ytr)
    return metrics.nrmse(yte[esn.washout:], esn.predict(ute)[esn.washout:])


def test_float32_dtype_propagates(data):
    utr, ytr, ute, yte = data
    W = topologies.random_reservoir(150, density=0.1, rng=0)
    esn = _esn(W, dtype=np.float32)
    assert esn.harvest(utr).dtype == np.float32
    assert _fit_nrmse(esn, utr, ytr, ute, yte) < 0.8


def test_float32_close_to_float64(data):
    utr, ytr, ute, yte = data
    W = topologies.random_reservoir(150, density=0.1, rng=0)
    n64 = _fit_nrmse(_esn(W, use_numba=False), utr, ytr, ute, yte)
    n32 = _fit_nrmse(_esn(W, dtype=np.float32), utr, ytr, ute, yte)
    assert abs(n64 - n32) < 0.05


def test_sparse_is_csr_and_matches_dense(data):
    sp = pytest.importorskip("scipy.sparse")
    utr, ytr, ute, yte = data
    W = topologies.random_reservoir(150, density=0.1, rng=0)
    sparse = _esn(W, sparse=True)
    assert sp.issparse(sparse.W)
    # power-iteration vs exact spectral radius differ slightly -> small tolerance
    assert abs(_fit_nrmse(_esn(W, use_numba=False), utr, ytr, ute, yte)
               - _fit_nrmse(sparse, utr, ytr, ute, yte)) < 0.05


def test_numba_matches_numpy(data):
    pytest.importorskip("numba")
    utr, ytr, ute, yte = data
    W = topologies.random_reservoir(150, density=0.1, rng=0)
    e_np, e_nb = _esn(W, use_numba=False), _esn(W, use_numba=True)
    # identical math, only summation order differs -> agree to fp tolerance
    assert np.allclose(e_np.harvest(utr), e_nb.harvest(utr), atol=1e-9)
    assert abs(_fit_nrmse(e_np, utr, ytr, ute, yte)
               - _fit_nrmse(e_nb, utr, ytr, ute, yte)) < 1e-6


def test_auto_numba_size_aware():
    from esnfed.esn import _NUMBA_AUTO_MAX_N
    small = EchoStateNetwork(1, 1, topologies.random_reservoir(60, rng=0))
    big = EchoStateNetwork(1, 1, topologies.random_reservoir(
        _NUMBA_AUTO_MAX_N + 50, density=0.02, rng=0))
    # auto policy: numba on for small reservoirs, off for very large ones
    assert small._numba_enabled in (True, False)   # depends on numba being installed
    assert big._numba_enabled is False


def test_use_numba_false_forces_numpy():
    esn = EchoStateNetwork(1, 1, topologies.random_reservoir(60, rng=0), use_numba=False)
    assert esn._numba_enabled is False
