"""Tests for differential privacy and secure aggregation of the federated stats."""
import numpy as np
import pytest

from esnfed import datasets, federated, topologies
from esnfed.esn import ridge_statistics
from esnfed.privacy import (
    PrivacyConfig,
    clip_rows,
    dp_statistics,
    gaussian_sigma,
    secure_sum,
    zero_sum_masks,
)


@pytest.fixture
def setup():
    rng = np.random.default_rng(7)
    u, y = datasets.narma10(1500, rng=rng)
    utr, ytr, _, _ = datasets.split(u, y, 0.7)
    W = topologies.random_reservoir(80, density=0.1, rng=rng)
    parts = datasets.partition_iid(utr, ytr, 4, rng=rng)
    kw = dict(spectral_radius=0.9, ridge=1e-6, washout=50)
    clients, ref = federated.make_shared_clients(W, parts, input_seed=0, esn_kwargs=kw)
    return clients, ref


# ---------------------------------------------------------------- DP primitives
def test_gaussian_sigma_scaling():
    s = gaussian_sigma(0.5, 1e-5, 1.0)
    assert s > 0
    assert gaussian_sigma(0.25, 1e-5, 1.0) == pytest.approx(2 * s)  # ~1/epsilon
    assert gaussian_sigma(0.5, 1e-5, 2.0) == pytest.approx(2 * s)   # ~sensitivity
    with pytest.raises(ValueError):
        gaussian_sigma(0.5, 1.0, 1.0)    # delta must be in (0,1)
    with pytest.raises(ValueError):
        gaussian_sigma(0.0, 1e-5, 1.0)   # epsilon must be > 0


def test_clip_rows_bounds_norm():
    M = np.array([[3.0, 4.0], [0.1, 0.0]])  # row norms 5.0 and 0.1
    C = clip_rows(M, 1.0)
    assert np.linalg.norm(C[0]) == pytest.approx(1.0)  # clipped down
    assert np.linalg.norm(C[1]) == pytest.approx(0.1)  # left untouched


def test_sensitivity_formula():
    cfg = PrivacyConfig(epsilon=1.0, clip_state=2.0, clip_target=0.5)
    assert cfg.sensitivity() == pytest.approx(2.0 * np.sqrt(2.0**2 + 0.5**2))


def test_dp_statistics_unbiased_symmetric_and_noisy(setup):
    clients, _ = setup
    Z, Y = clients[0].states(), clients[0].targets()
    # clip at the data's max row norm -> no actual clipping, realistic sensitivity
    cz = float(np.linalg.norm(Z, axis=1).max())
    cy = float(np.linalg.norm(np.atleast_2d(Y), axis=1).max())
    cfg = PrivacyConfig(epsilon=0.5, delta=1e-5, clip_state=cz, clip_target=cy)
    A_ref, B_ref = ridge_statistics(clip_rows(Z, cz), clip_rows(np.atleast_2d(Y), cy))
    sigma = gaussian_sigma(cfg.epsilon, cfg.delta, cfg.sensitivity())

    N = 200
    draws = [dp_statistics(Z, Y, cfg, rng=np.random.default_rng(i)) for i in range(N)]
    A_mean = np.mean([a for a, _ in draws], axis=0)
    B_mean = np.mean([b for _, b in draws], axis=0)

    # zero-mean noise: the sample mean is within a (very loose) 6-sigma band
    assert np.max(np.abs(A_mean - A_ref)) < 6 * sigma / np.sqrt(N)
    assert np.max(np.abs(B_mean - B_ref)) < 6 * sigma / np.sqrt(N)
    # a single private release is genuinely noisy, and A stays symmetric
    A1, _ = draws[0]
    assert not np.allclose(A1, A_ref)
    assert np.allclose(A1, A1.T)


def test_federated_ridge_dp_runs_and_differs_from_exact(setup):
    clients, ref = setup
    W_exact = federated.federated_ridge(clients, ref)
    cfg = PrivacyConfig(epsilon=0.5, delta=1e-5, clip_state=5.0, clip_target=1.0, seed=0)
    W_dp = federated.federated_ridge_dp(clients, ref, cfg)
    assert W_dp.shape == W_exact.shape
    assert np.all(np.isfinite(W_dp))
    assert not np.allclose(W_dp, W_exact)  # privacy noise perturbs the readout


# ----------------------------------------------------------- secure aggregation
def test_zero_sum_masks_cancel_and_hide():
    masks = zero_sum_masks(5, (4, 4), rng=np.random.default_rng(2), scale=3.0)
    assert np.allclose(np.sum(masks, axis=0), 0.0, atol=1e-9)  # masks cancel
    assert all(np.linalg.norm(m) > 0 for m in masks)           # each hides its value


def test_secure_sum_equals_plain_sum():
    rng = np.random.default_rng(0)
    mats = [rng.normal(size=(6, 6)) for _ in range(5)]
    plain = sum(mats)
    secure = secure_sum(mats, rng=np.random.default_rng(1), scale=10.0)
    assert np.allclose(secure, plain, atol=1e-9)


def test_federated_ridge_secure_equals_exact(setup):
    clients, ref = setup
    W_exact = federated.federated_ridge(clients, ref)
    W_secure = federated.federated_ridge_secure(clients, ref, seed=3, mask_scale=1.0)
    assert np.allclose(W_exact, W_secure, rtol=1e-3, atol=1e-6)
