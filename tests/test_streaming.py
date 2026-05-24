"""Tests for incremental / streaming ridge and the RLS readout."""
import numpy as np

from esnfed.esn import ridge_statistics, solve_readout
from esnfed.streaming import RLSReadout, StreamingRidge


def test_streaming_equals_batch():
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(300, 12))
    Y = rng.normal(size=(300, 2))
    ridge = 1e-3

    sr = StreamingRidge(readout_dim=12, n_outputs=2, ridge=ridge)
    for chunk in np.array_split(np.arange(300), 5):
        sr.update(Z[chunk], Y[chunk])

    A_b, B_b = ridge_statistics(Z, Y)
    assert np.allclose(sr.A, A_b)
    assert np.allclose(sr.B, B_b)
    assert sr.n_seen == 300
    assert np.allclose(sr.readout(), solve_readout(A_b, B_b, ridge))


def test_streaming_merge_is_federated_sum():
    rng = np.random.default_rng(1)
    Z = rng.normal(size=(200, 6))
    Y = rng.normal(size=(200, 1))
    ridge = 1e-4

    a = StreamingRidge(6, 1, ridge).update(Z[:120], Y[:120])
    b = StreamingRidge(6, 1, ridge).update(Z[120:], Y[120:])
    a.merge(b)

    A_b, B_b = ridge_statistics(Z, Y)
    assert a.n_seen == 200
    assert np.allclose(a.A, A_b)
    assert np.allclose(a.readout(), solve_readout(A_b, B_b, ridge))


def test_rls_equals_batch_ridge():
    rng = np.random.default_rng(2)
    Z = rng.normal(size=(400, 8))
    Y = rng.normal(size=(400, 1))
    ridge = 1e-2

    rls = RLSReadout(8, 1, ridge=ridge).update_batch(Z, Y)

    A, B = ridge_statistics(Z, Y)
    W_batch = solve_readout(A, B, ridge)
    assert rls.n_seen == 400
    assert np.allclose(rls.readout(), W_batch, rtol=1e-4, atol=1e-6)
