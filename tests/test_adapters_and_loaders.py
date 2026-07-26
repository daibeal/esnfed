"""Adapters and data loaders, tested without their optional dependencies.

``esnfed.interop`` duck-types a ReservoirPy ``Reservoir`` and the loaders in
``esnfed.datasets`` duck-type a URL opener, so both can be exercised with small
fakes. That covers the adapter logic itself -- bias conventions, spectral-radius
and leaking-rate preservation, cache handling, archive parsing -- on every
machine, instead of only where ReservoirPy is installed and the network is up.
"""
import io
import zipfile

import numpy as np
import pytest

from esnfed import datasets, interop, topologies
from esnfed.esn import _spectral_radius


# ─────────────────────────────────────────────────────────── ReservoirPy fake
class FakeReservoir:
    """The subset of the ReservoirPy ``Reservoir`` surface that interop reads."""

    def __init__(self, units=40, n_inputs=1, sr=0.85, lr=0.4, bias="vector",
                 sparse=False, seed=0):
        rng = np.random.default_rng(seed)
        W = topologies.random_reservoir(units, density=0.2, rng=seed)
        radius = _spectral_radius(W)
        if radius > 0:
            W = W * (sr / radius)
        self.W = _as_sparse(W) if sparse else W
        self.Win = rng.uniform(-1, 1, size=(units, n_inputs))
        self.lr = lr
        self.is_initialized = True
        self._init_calls = 0
        if bias == "vector":
            self.bias = rng.uniform(-0.5, 0.5, size=(units, 1))
        elif bias == "scalar":
            self.bias = np.array([0.25])
        elif bias == "empty":
            self.bias = np.zeros((0,))
        elif bias == "none":
            self.bias = None
        # "missing": leave the attribute unset entirely, as some node versions do

    def initialize(self, x):
        self._init_calls += 1
        self.is_initialized = True


def _as_sparse(W):
    sp = pytest.importorskip("scipy.sparse")
    return sp.csr_matrix(W)


class TestReservoirPyAdapter:

    def test_reservoir_matrix_is_dense_and_square(self):
        W = interop.reservoir_matrix(FakeReservoir(units=30), n_inputs=1)
        assert W.shape == (30, 30) and np.isfinite(W).all()

    def test_sparse_reservoir_is_densified(self):
        W = interop.reservoir_matrix(FakeReservoir(units=25, sparse=True))
        assert isinstance(W, np.ndarray) and W.shape == (25, 25)

    def test_uninitialised_reservoir_is_initialised_once(self):
        res = FakeReservoir(units=20)
        res.is_initialized = False
        interop.reservoir_matrix(res, n_inputs=1)
        assert res._init_calls == 1 and res.is_initialized

    def test_input_matrix_packs_bias_into_column_zero(self):
        res = FakeReservoir(units=20, n_inputs=3, bias="vector")
        Win = interop.input_matrix(res, n_inputs=3)
        assert Win.shape == (20, 4)
        assert np.allclose(Win[:, :1], res.bias.reshape(20, 1))
        assert np.allclose(Win[:, 1:], res.Win)

    def test_scalar_bias_is_broadcast(self):
        res = FakeReservoir(units=15, bias="scalar")
        assert np.allclose(interop.input_matrix(res)[:, 0], 0.25)

    @pytest.mark.parametrize("bias", ["empty", "none", "missing"])
    def test_absent_bias_becomes_a_zero_column(self, bias):
        res = FakeReservoir(units=15, bias=bias)
        assert np.allclose(interop.input_matrix(res)[:, 0], 0.0)

    def test_to_esn_preserves_spectral_radius_and_leaking_rate(self):
        res = FakeReservoir(units=50, sr=0.85, lr=0.35)
        esn = interop.to_esn(res, n_inputs=1, n_outputs=1, washout=10)
        assert esn.n_reservoir == 50
        assert esn.leaking_rate == pytest.approx(0.35)
        assert _spectral_radius(np.asarray(esn.W)) == pytest.approx(0.85, rel=1e-6)

    def test_to_esn_reuses_the_input_weights(self):
        res = FakeReservoir(units=30, n_inputs=2, bias="vector")
        esn = interop.to_esn(res, n_inputs=2, n_outputs=1, washout=5)
        assert esn.W_in.shape == (30, 3)
        assert np.allclose(esn.W_in[:, 1:], res.Win)

    def test_explicit_overrides_win(self):
        res = FakeReservoir(units=30, sr=0.85, lr=0.35)
        esn = interop.to_esn(res, n_inputs=1, spectral_radius=0.5,
                             leaking_rate=0.9, washout=5)
        assert esn.leaking_rate == pytest.approx(0.9)
        assert _spectral_radius(np.asarray(esn.W)) == pytest.approx(0.5, rel=1e-6)

    def test_fresh_input_weights_when_requested(self):
        res = FakeReservoir(units=30, n_inputs=1)
        esn = interop.to_esn(res, n_inputs=1, use_input_weights=False,
                             seed=0, washout=5)
        assert esn.W_in.shape == (30, 2)
        assert not np.allclose(esn.W_in[:, 1:], res.Win)

    def test_adapted_reservoir_trains(self):
        from esnfed import metrics
        res = FakeReservoir(units=120, sr=0.9, lr=0.6, seed=1)
        esn = interop.to_esn(res, n_inputs=1, n_outputs=1, washout=50,
                             ridge=1e-6)
        u, y = datasets.narma10(1200, rng=0)
        utr, ytr, ute, yte = datasets.split(u, y, 0.7)
        esn.fit(utr, ytr)
        assert metrics.nrmse(yte[50:], esn.predict(ute)[50:]) < 0.9


# ────────────────────────────────────────────────────────────── data loaders
FRED_CSV = ("observation_date,SERIESX\n"
            "2020-01-01,1.0\n2020-01-02,.\n2020-01-03,2.0\n"
            "2020-01-06,3.0\n2020-01-07,4.0\n")


class FakeResponse(io.BytesIO):
    """Context-manager wrapper so a BytesIO can stand in for a URL response."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


@pytest.fixture
def fake_network(monkeypatch):
    """Serve canned bytes for any URL, and record what was requested."""
    calls = []

    def opener(url, timeout):
        calls.append(url)
        if "fredgraph" in url:
            sid = url.rsplit("id=", 1)[-1]
            return FakeResponse(
                FRED_CSV.replace("SERIESX", sid).encode("utf-8"))
        if "japanese" in url:
            return FakeResponse(_japanese_vowels_zip())
        raise AssertionError(f"unexpected URL {url}")

    monkeypatch.setattr(datasets, "_urlopen", opener)
    return calls


def _japanese_vowels_zip() -> bytes:
    """A miniature stand-in for the UCI Japanese Vowels archive."""
    def block(rows, feats=12, offset=0.0):
        return "\n".join(" ".join(f"{offset + i + j / 10:.3f}"
                                  for j in range(feats))
                         for i in range(rows))

    train = "\n\n".join(block(4, offset=s) for s in range(6))
    test = "\n\n".join(block(3, offset=s) for s in range(4))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("ae.train", train + "\n")
        z.writestr("ae.test", test + "\n")
        z.writestr("size_ae.train", "3 3")       # 6 utterances over 2 speakers
        z.writestr("size_ae.test", "2 2")
    return buf.getvalue()


class TestFredLoaders:

    def test_downloads_parses_and_caches(self, tmp_path, fake_network):
        u, y = datasets.load_fred("SERIESX", cache_dir=tmp_path, normalize=False)
        # the '.' row is dropped, leaving [1, 2, 3, 4] -> pairs (1,2) (2,3) (3,4)
        assert np.allclose(u.ravel(), [1.0, 2.0, 3.0])
        assert np.allclose(y.ravel(), [2.0, 3.0, 4.0])
        assert (tmp_path / "fred_SERIESX.csv").exists()
        assert len(fake_network) == 1

    def test_second_call_is_served_from_cache(self, tmp_path, fake_network):
        datasets.load_fred("SERIESX", cache_dir=tmp_path, normalize=False)
        datasets.load_fred("SERIESX", cache_dir=tmp_path, normalize=False)
        assert len(fake_network) == 1        # no second download

    def test_raw_returns_the_level_series(self, tmp_path, fake_network):
        raw = datasets.load_fred("SERIESX", cache_dir=tmp_path, raw=True)
        assert np.allclose(raw, [1.0, 2.0, 3.0, 4.0])

    def test_matrix_aligns_series_on_common_dates(self, tmp_path, fake_network):
        M = datasets.load_fred_matrix(["AAA", "BBB"], cache_dir=tmp_path,
                                      normalize=False, raw=True)
        assert M.shape == (4, 2)
        assert np.allclose(M[:, 0], M[:, 1])   # the fake serves identical data

    def test_matrix_builds_a_forecasting_task(self, tmp_path, fake_network):
        u, y = datasets.load_fred_matrix(["AAA", "BBB"], cache_dir=tmp_path,
                                         normalize=False)
        assert u.shape == (3, 2) and y.shape == (3, 1)
        assert np.allclose(y.ravel(), [2.0, 3.0, 4.0])

    def test_matrix_target_selects_a_column(self, tmp_path, fake_network):
        u, y = datasets.load_fred_matrix(["AAA", "BBB"], target="BBB",
                                         cache_dir=tmp_path, normalize=False)
        assert np.allclose(y.ravel(), [2.0, 3.0, 4.0])

    def test_matrix_change_mode(self, tmp_path, fake_network):
        u, y = datasets.load_fred_matrix(["AAA"], cache_dir=tmp_path,
                                         normalize=False, predict="change")
        assert np.allclose(u.ravel(), [1.0, 1.0])   # first differences

    def test_matrix_accepts_a_single_id(self, tmp_path, fake_network):
        M = datasets.load_fred_matrix("AAA", cache_dir=tmp_path, raw=True,
                                      normalize=False)
        assert M.shape == (4, 1)

    def test_matrix_normalisation_is_applied(self, tmp_path, fake_network):
        M = datasets.load_fred_matrix(["AAA"], cache_dir=tmp_path, raw=True,
                                      normalize=True)
        assert abs(M.mean()) < 1e-12 and abs(M.std() - 1.0) < 1e-12

    def test_matrix_unknown_mode_raises(self, tmp_path, fake_network):
        with pytest.raises(ValueError, match="'next' or 'change'"):
            datasets.load_fred_matrix(["AAA"], cache_dir=tmp_path,
                                      predict="bogus")

    def test_disjoint_series_raise(self, tmp_path, monkeypatch):
        def opener(url, timeout):
            body = ("observation_date,X\n2020-01-01,1.0\n2020-01-02,2.0\n"
                    if url.endswith("AAA") else
                    "observation_date,X\n2021-05-01,5.0\n2021-05-02,6.0\n")
            return FakeResponse(body.encode())
        monkeypatch.setattr(datasets, "_urlopen", opener)
        with pytest.raises(ValueError, match="no overlapping dates"):
            datasets.load_fred_matrix(["AAA", "BBB"], cache_dir=tmp_path)


class TestSequenceLoaders:

    def test_japanese_vowels_parses_and_caches(self, tmp_path, fake_network):
        ds = datasets.load_japanese_vowels(cache_dir=tmp_path)
        assert ds.n_features == 12
        assert len(ds.X_train) == 6 and len(ds.X_test) == 4
        assert all(x.shape[1] == 12 for x in ds.X_train)
        assert ds.y_train.tolist() == [0, 0, 0, 1, 1, 1]
        assert ds.y_test.tolist() == [0, 0, 1, 1]
        assert np.array_equal(ds.groups_train, ds.y_train)
        assert (tmp_path / "japanese_vowels.zip").exists()

    def test_archive_is_downloaded_once(self, tmp_path, fake_network):
        datasets.load_japanese_vowels(cache_dir=tmp_path)
        datasets.load_japanese_vowels(cache_dir=tmp_path)
        assert len(fake_network) == 1

    def test_group_clients_splits_by_group(self):
        X = [np.zeros((5, 2)) for _ in range(6)]
        y = np.array([0, 1, 0, 1, 0, 1])
        groups = np.array(["a", "a", "b", "b", "c", "c"])
        clients = datasets.group_clients(X, y, groups)
        assert len(clients) == 3
        assert all(len(seqs) == 2 and len(labels) == 2
                   for seqs, labels in clients)
        assert clients[0][1].tolist() == [0, 1]

    def test_group_clients_covers_every_sequence(self):
        X = [np.zeros((4, 1)) for _ in range(10)]
        y = np.arange(10) % 3
        groups = np.arange(10) % 4
        clients = datasets.group_clients(X, y, groups)
        assert sum(len(s) for s, _ in clients) == 10

    def test_a_loaded_dataset_can_be_classified(self, tmp_path, fake_network):
        """End-to-end: loader -> reservoir features -> federated classifier."""
        from esnfed import EchoStateNetwork, classification as clf
        ds = datasets.load_japanese_vowels(cache_dir=tmp_path)
        W = topologies.random_reservoir(40, density=0.15, rng=0)
        esn = EchoStateNetwork(ds.n_features, ds.n_classes, W, washout=0,
                               ridge=1e-2, seed=0)
        clients = datasets.group_clients(ds.X_train, ds.y_train, ds.groups_train)
        W_out = clf.federated_classifier(esn, clients, ds.n_classes)
        preds = clf.predict_labels(esn, ds.X_test, W_out)
        assert preds.shape == (len(ds.X_test),)
        assert set(np.unique(preds)) <= set(range(ds.n_classes))


class TestBundledData:

    def test_ted_spread_is_bundled_and_sane(self):
        raw = datasets.load_ted_spread(raw=True)
        assert raw.ndim == 1 and len(raw) > 5000
        assert raw.min() >= 0 and np.isfinite(raw).all()

    def test_ted_spread_builds_a_task(self):
        u, y = datasets.load_ted_spread()
        assert u.shape == y.shape and u.shape[1] == 1
        assert abs(u.mean()) < 0.1        # normalised by default

    def test_ted_spread_change_mode(self):
        u, y = datasets.load_ted_spread(predict="change")
        assert len(u) == len(y) and np.isfinite(u).all()
