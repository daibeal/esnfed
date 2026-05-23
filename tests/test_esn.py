"""Tests for the Echo State Network core."""
import numpy as np
import pytest

from esnfed import EchoStateNetwork, datasets, metrics, topologies
from esnfed.esn import _spectral_radius


@pytest.fixture
def rng():
    return np.random.default_rng(12345)


def test_spectral_radius_is_rescaled(rng):
    W = topologies.random_reservoir(80, density=0.1, rng=rng)
    esn = EchoStateNetwork(1, 1, W, spectral_radius=0.8)
    assert _spectral_radius(esn.W) == pytest.approx(0.8, rel=1e-6)


def test_zero_reservoir_does_not_crash(rng):
    W = np.zeros((10, 10))
    esn = EchoStateNetwork(1, 1, W, spectral_radius=0.9)
    # spectral radius is 0, so it stays the zero matrix
    assert np.allclose(esn.W, 0.0)


def test_harvest_shape(rng):
    W = topologies.random_reservoir(50, rng=rng)
    esn = EchoStateNetwork(1, 1, W)
    u = rng.standard_normal((200, 1))
    Z = esn.harvest(u)
    assert Z.shape == (200, 1 + 1 + 50)
    # First column is the bias constant.
    assert np.allclose(Z[:, 0], esn.bias)


def test_echo_state_property_state_forgetting(rng):
    """With spectral radius < 1 the state must forget its initial condition."""
    W = topologies.random_reservoir(100, density=0.1, rng=rng)
    esn = EchoStateNetwork(1, 1, W, spectral_radius=0.9, leaking_rate=1.0)
    u = rng.uniform(0, 0.5, size=(300, 1))
    Z1 = esn.harvest(u, x0=np.zeros(100))
    Z2 = esn.harvest(u, x0=rng.standard_normal(100))
    # States from different initial conditions converge after a transient.
    early = np.linalg.norm(Z1[5] - Z2[5])
    late = np.linalg.norm(Z1[-1] - Z2[-1])
    assert late < early
    assert late < 1e-3


def test_fit_predict_reduces_error(rng):
    u, y = datasets.narma10(1500, rng=rng)
    utr, ytr, ute, yte = datasets.split(u, y)
    W = topologies.random_reservoir(200, density=0.05, rng=rng)
    esn = EchoStateNetwork(1, 1, W, spectral_radius=0.9, leaking_rate=1.0, washout=100)
    esn.fit(utr, ytr)
    pred = esn.predict(ute)
    err = metrics.nrmse(yte[100:], pred[100:])
    # A trained ESN must clearly beat the trivial mean predictor (NRMSE = 1).
    assert err < 0.8


def test_predict_before_fit_raises(rng):
    W = topologies.random_reservoir(20, rng=rng)
    esn = EchoStateNetwork(1, 1, W)
    with pytest.raises(RuntimeError):
        esn.predict(rng.standard_normal((10, 1)))


def test_non_square_reservoir_raises():
    with pytest.raises(ValueError):
        EchoStateNetwork(1, 1, np.zeros((4, 5)))
