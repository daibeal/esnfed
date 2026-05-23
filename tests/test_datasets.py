"""Tests for benchmark dataset generators."""
import numpy as np
import pytest

from esnfed import datasets


def test_narma10_shapes_and_range():
    u, y = datasets.narma10(1000, rng=0)
    assert u.shape == (1000, 1) and y.shape == (1000, 1)
    assert u.min() >= 0.0 and u.max() <= 0.5  # input is U(0, 0.5)
    assert np.all(np.isfinite(y))


def test_narma10_is_reproducible():
    a = datasets.narma10(500, rng=42)
    b = datasets.narma10(500, rng=42)
    assert np.allclose(a[0], b[0]) and np.allclose(a[1], b[1])


def test_mackey_glass_prediction_pairs():
    u, y = datasets.mackey_glass(800, seed=1)
    assert u.shape == (800, 1) and y.shape == (800, 1)
    # y is the one-step-ahead shift of u.
    assert np.allclose(u[1:], y[:-1])


def test_lorenz_is_normalised():
    u, y = datasets.lorenz(1000, seed=2)
    assert abs(u.mean()) < 0.1 and abs(u.std() - 1.0) < 0.1


def test_split_is_chronological():
    u, y = datasets.narma10(1000, rng=0)
    utr, ytr, ute, yte = datasets.split(u, y, 0.7)
    assert len(utr) == 700 and len(ute) == 300
    assert np.allclose(utr, u[:700])


def test_partition_iid_covers_all_data():
    u, y = datasets.narma10(1000, rng=0)
    parts = datasets.partition_iid(u, y, 4)
    assert len(parts) == 4
    total = sum(len(pu) for pu, _ in parts)
    assert total == len(u)


def test_make_dataset_dispatch():
    u, y = datasets.make_dataset("narma10", 200, rng=0)
    assert u.shape == (200, 1)
    with pytest.raises(KeyError):
        datasets.make_dataset("nope", 10)


# ---- real-world data ---------------------------------------------------------
def test_from_array_next_and_change():
    u, y = datasets.from_array([1, 2, 3, 4, 5], predict="next", normalize=False)
    assert np.allclose(u.ravel(), [1, 2, 3, 4])
    assert np.allclose(y.ravel(), [2, 3, 4, 5])
    du, dy = datasets.from_array([1, 2, 4, 7, 11], predict="change", normalize=False)
    # first differences are [1,2,3,4]; task pairs them one-step-ahead
    assert np.allclose(du.ravel(), [1, 2, 3])
    assert np.allclose(dy.ravel(), [2, 3, 4])


def test_from_array_normalizes():
    u, y = datasets.from_array(np.arange(100.0), normalize=True)
    full = np.concatenate([u.ravel(), y.ravel()[-1:]])
    assert abs(full.mean()) < 1e-6 and abs(full.std() - 1.0) < 1e-6


def test_load_csv(tmp_path):
    p = tmp_path / "series.csv"
    p.write_text("date,value\n2020-01-01,10\n2020-01-02,.\n2020-01-03,12\n",
                 encoding="utf-8")
    u, y = datasets.load_csv(p, column="value", normalize=False)
    # the missing '.' row is dropped, leaving [10, 12]
    assert np.allclose(u.ravel(), [10.0]) and np.allclose(y.ravel(), [12.0])


def test_load_ted_spread_bundled_and_learnable():
    raw = datasets.load_ted_spread(raw=True)
    assert raw.ndim == 1 and len(raw) > 5000
    assert raw.min() >= 0  # a spread is non-negative
    u, y = datasets.load_ted_spread()
    cut = int(0.7 * len(u))
    from esnfed import EchoStateNetwork, topologies, metrics
    W = topologies.random_reservoir(100, density=0.1, rng=0)
    esn = EchoStateNetwork(1, 1, W, spectral_radius=0.9, leaking_rate=0.5, washout=100)
    esn.fit(u[:cut], y[:cut])
    pred = esn.predict(u[cut:])
    # one-step-ahead level forecasting must beat the trivial predictor
    assert metrics.nrmse(y[cut:][100:], pred[100:]) < 0.9


def test_load_fred_uses_cache_offline(tmp_path):
    # Pre-seed the cache so no network call is made.
    (tmp_path / "fred_TESTSERIES.csv").write_text(
        "observation_date,TESTSERIES\n2020-01-01,1.0\n2020-01-02,2.0\n2020-01-03,3.0\n",
        encoding="utf-8",
    )
    u, y = datasets.load_fred("TESTSERIES", cache_dir=tmp_path, normalize=False)
    assert np.allclose(u.ravel(), [1.0, 2.0]) and np.allclose(y.ravel(), [2.0, 3.0])
