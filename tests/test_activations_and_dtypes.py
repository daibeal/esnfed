"""Node nonlinearities, numeric dtypes, and state continuation.

These exercise the parts of the harvest loop that the accelerated paths
(Numba / sparse / float32) must agree with, plus the multi-type reservoir
machinery that indexes activations per node.
"""
import numpy as np
import pytest

from esnfed import EchoStateNetwork, datasets, metrics, topologies
from esnfed.esn import ACTIVATIONS, _harvest_numpy


@pytest.fixture(scope="module")
def W():
    return topologies.random_reservoir(60, density=0.15, rng=3)


@pytest.fixture(scope="module")
def u():
    return np.random.default_rng(4).uniform(0.0, 0.5, size=(200, 1))


class TestActivations:

    @pytest.mark.parametrize("name", sorted(ACTIVATIONS))
    def test_every_activation_runs_and_stays_finite(self, W, u, name):
        esn = EchoStateNetwork(1, 1, W, spectral_radius=0.5, activation=name,
                               leaking_rate=0.5, seed=0)
        Z = esn.harvest(u)
        assert Z.shape == (len(u), esn.readout_dim)
        assert np.all(np.isfinite(Z))

    @pytest.mark.parametrize("name,fn", sorted(ACTIVATIONS.items()))
    def test_activation_table_matches_its_definition(self, name, fn):
        z = np.linspace(-2.0, 2.0, 9)
        expected = {
            "tanh": np.tanh(z),
            "sigmoid": 1.0 / (1.0 + np.exp(-z)),
            "relu": np.maximum(0.0, z),
            "sin": np.sin(z),
            "identity": z,
        }[name]
        assert np.allclose(fn(z), expected)

    def test_bounded_activations_stay_in_range(self, W, u):
        for name, lo, hi in [("tanh", -1, 1), ("sigmoid", 0, 1), ("sin", -1, 1)]:
            esn = EchoStateNetwork(1, 1, W, spectral_radius=0.9, activation=name,
                                   seed=0)
            x = esn.harvest(u)[:, 2:]
            assert x.min() >= lo - 1e-12 and x.max() <= hi + 1e-12

    def test_relu_is_non_negative(self, W, u):
        esn = EchoStateNetwork(1, 1, W, spectral_radius=0.5, activation="relu",
                               leaking_rate=1.0, seed=0)
        assert np.all(esn.harvest(u)[:, 2:] >= 0.0)

    def test_unknown_activation_lists_the_choices(self, W):
        with pytest.raises(ValueError, match="unknown activation"):
            EchoStateNetwork(1, 1, W, activation="bogus")

    def test_callable_activation_is_used(self, W, u):
        esn_named = EchoStateNetwork(1, 1, W, spectral_radius=0.9, seed=1,
                                     activation="tanh", use_numba=False)
        esn_callable = EchoStateNetwork(1, 1, W, spectral_radius=0.9, seed=1,
                                       activation=np.tanh)
        assert np.allclose(esn_named.harvest(u), esn_callable.harvest(u))

    def test_callable_activation_disables_numba(self, W):
        esn = EchoStateNetwork(1, 1, W, activation=np.tanh)
        assert esn._hetero_act and esn._numba_enabled is False

    def test_mixed_activation_applies_each_function_per_node(self, W, u):
        """A mixed reservoir must equal a manual per-node dispatch, exactly."""
        n = W.shape[0]
        # Build via a list so numpy sizes the string dtype for the longest name;
        # np.array(["tanh"] * n) is <U4 and would truncate "sigmoid" to "sigm".
        names = np.array([("sigmoid", "sin", "tanh")[i % 3] for i in range(n)])
        esn = EchoStateNetwork(1, 1, W, spectral_radius=0.9, leaking_rate=0.7,
                               seed=2, activation=names)
        assert esn._hetero_act

        def manual(pre):
            out = np.empty_like(pre)
            for i, nm in enumerate(names):
                out[i] = ACTIVATIONS[nm](pre[i])
            return out

        expected = _harvest_numpy(np.asarray(esn.W), esn.W_in, u, esn._a,
                                  esn.bias, 1, n, None, act=manual)
        assert np.allclose(esn.harvest(u), expected)

    def test_mixed_activation_wrong_length_raises(self, W):
        with pytest.raises(ValueError, match="length n_reservoir"):
            EchoStateNetwork(1, 1, W, activation=np.array(["tanh"] * 5))

    def test_mixed_activation_unknown_name_raises(self, W):
        names = np.array(["nope"] + ["tanh"] * (W.shape[0] - 1))
        with pytest.raises(ValueError, match="unknown activation"):
            EchoStateNetwork(1, 1, W, activation=names)

    def test_truncated_activation_name_is_rejected(self, W):
        """A <U4 array silently truncates longer names; that must not pass."""
        names = np.array(["tanh"] * W.shape[0])       # dtype <U4
        names[0] = "sigmoid"                          # becomes "sigm"
        with pytest.raises(ValueError, match="unknown activation"):
            EchoStateNetwork(1, 1, W, activation=names)

    def test_uniform_mixed_array_equals_the_scalar_name(self, W, u):
        n = W.shape[0]
        a = EchoStateNetwork(1, 1, W, spectral_radius=0.9, seed=5,
                             activation="sigmoid")
        b = EchoStateNetwork(1, 1, W, spectral_radius=0.9, seed=5,
                             activation=np.array(["sigmoid"] * n))
        assert np.allclose(a.harvest(u), b.harvest(u))

    def test_mixed_reservoir_can_learn(self):
        u_tr, y_tr, u_te, y_te = datasets.split(*datasets.narma10(1200, rng=0), 0.7)
        Wm = topologies.random_reservoir(120, density=0.1, rng=0)
        acts = topologies.mixed_activations(120, ("tanh", "sigmoid", "sin"), rng=0)
        esn = EchoStateNetwork(1, 1, Wm, spectral_radius=0.9, activation=acts,
                               leaking_rate=0.7, washout=50, seed=0).fit(u_tr, y_tr)
        assert metrics.nrmse(y_te[50:], esn.predict(u_te)[50:]) < 0.8


class TestHeterogeneousLeaking:

    @pytest.mark.parametrize("kind", ["uniform", "log_uniform", "constant", "layered"])
    def test_generated_rates_are_in_range(self, kind):
        a = topologies.leaking_rates(80, kind, low=0.1, high=0.9, rng=0)
        assert a.shape == (80,)
        assert a.min() >= 0.1 - 1e-12 and a.max() <= 0.9 + 1e-12

    def test_layered_is_non_increasing(self):
        a = topologies.leaking_rates(90, "layered", low=0.1, high=0.9,
                                     n_layers=3, rng=0)
        assert np.all(np.diff(a) <= 1e-12)
        assert len(np.unique(a)) == 3

    def test_constant_matches_a_scalar_esn(self, W, u):
        n = W.shape[0]
        a = topologies.leaking_rates(n, "constant", high=0.4, rng=0)
        scalar = EchoStateNetwork(1, 1, W, spectral_radius=0.9, leaking_rate=0.4,
                                 seed=7, use_numba=False)
        vector = EchoStateNetwork(1, 1, W, spectral_radius=0.9, leaking_rate=a,
                                  seed=7)
        assert np.allclose(scalar.harvest(u), vector.harvest(u))

    def test_heterogeneous_reservoir_can_learn(self):
        u_tr, y_tr, u_te, y_te = datasets.split(*datasets.narma10(1200, rng=0), 0.7)
        Wh = topologies.random_reservoir(120, density=0.1, rng=0)
        a = topologies.leaking_rates(120, "layered", low=0.2, high=0.9, rng=0)
        esn = EchoStateNetwork(1, 1, Wh, spectral_radius=0.9, leaking_rate=a,
                               washout=50, seed=0).fit(u_tr, y_tr)
        assert metrics.nrmse(y_te[50:], esn.predict(u_te)[50:]) < 0.8

    def test_scalar_leak_disables_the_heterogeneous_path(self, W):
        assert EchoStateNetwork(1, 1, W, leaking_rate=0.5)._hetero_leak is False
        assert EchoStateNetwork(1, 1, W, leaking_rate=np.full(60, 0.5))._hetero_leak


class TestStateContinuation:
    """Harvesting in chunks with an explicit ``x0`` must equal one long harvest."""

    def test_two_chunk_harvest_matches_single_run(self, W, u):
        esn = EchoStateNetwork(1, 1, W, spectral_radius=0.9, leaking_rate=0.6,
                               seed=8, use_numba=False)
        full = esn.harvest(u)
        cut = 80
        first = esn.harvest(u[:cut])
        state = first[-1, 2:]
        second = esn.harvest(u[cut:], x0=state)
        assert np.allclose(first, full[:cut], atol=1e-12)
        assert np.allclose(second, full[cut:], atol=1e-12)

    def test_many_chunk_harvest_matches_single_run(self, W, u):
        esn = EchoStateNetwork(1, 1, W, spectral_radius=0.9, leaking_rate=0.6,
                               seed=8, use_numba=False)
        full = esn.harvest(u)
        pieces, state = [], None
        for idx in np.array_split(np.arange(len(u)), 7):
            Z = esn.harvest(u[idx], x0=state)
            pieces.append(Z)
            state = Z[-1, 2:]
        assert np.allclose(np.vstack(pieces), full, atol=1e-12)

    def test_zero_x0_is_the_default(self, W, u):
        esn = EchoStateNetwork(1, 1, W, spectral_radius=0.9, seed=9,
                               use_numba=False)
        assert np.allclose(esn.harvest(u), esn.harvest(u, x0=np.zeros(60)))

    def test_predict_accepts_a_warm_start(self, W):
        u_tr, y_tr, u_te, _ = datasets.split(*datasets.narma10(800, rng=0), 0.7)
        esn = EchoStateNetwork(1, 1, W, spectral_radius=0.9, washout=50,
                               seed=10, use_numba=False).fit(u_tr, y_tr)
        cold = esn.predict(u_te)
        warm = esn.predict(u_te, x0=np.full(60, 0.3))
        assert cold.shape == warm.shape
        assert not np.allclose(cold[0], warm[0])       # the warm start matters
        assert np.allclose(cold[-1], warm[-1], atol=1e-6)   # ...then is forgotten


class TestNumericPaths:

    @pytest.fixture(scope="class")
    def task(self):
        return datasets.split(*datasets.narma10(1500, rng=0), 0.7)

    def _fit_err(self, esn, task):
        u_tr, y_tr, u_te, y_te = task
        esn.fit(u_tr, y_tr)
        return metrics.nrmse(y_te[esn.washout:], esn.predict(u_te)[esn.washout:])

    def test_float32_states_are_float32(self, task):
        Wf = topologies.random_reservoir(150, density=0.1, rng=0)
        esn = EchoStateNetwork(1, 1, Wf, spectral_radius=0.9, washout=100,
                               seed=1, dtype=np.float32)
        assert esn.harvest(task[0]).dtype == np.float32
        assert np.asarray(esn.W).dtype == np.float32

    def test_float32_readout_stays_float64_for_stability(self, task):
        """The Gram is accumulated in float64 even for a float32 harvest."""
        Wf = topologies.random_reservoir(150, density=0.1, rng=0)
        esn = EchoStateNetwork(1, 1, Wf, spectral_radius=0.9, washout=100,
                               seed=1, dtype=np.float32)
        esn.fit(task[0], task[1])
        assert esn.W_out.dtype == np.float64

    def test_float32_tracks_float64_accuracy(self, task):
        Wf = topologies.random_reservoir(150, density=0.1, rng=0)
        kw = dict(spectral_radius=0.9, washout=100, seed=1)
        e64 = self._fit_err(EchoStateNetwork(1, 1, Wf, use_numba=False, **kw), task)
        e32 = self._fit_err(
            EchoStateNetwork(1, 1, Wf, dtype=np.float32, **kw), task)
        assert abs(e64 - e32) < 0.05

    def test_sparse_matches_dense_exactly(self, task):
        pytest.importorskip("scipy.sparse")
        Wf = topologies.random_reservoir(150, density=0.1, rng=0)
        kw = dict(spectral_radius=0.9, washout=100, seed=1)
        dense = EchoStateNetwork(1, 1, Wf, use_numba=False, **kw)
        sparse = EchoStateNetwork(1, 1, Wf, sparse=True, **kw)
        # Both now scale by the same (exact) spectral radius, so the dynamics
        # agree to floating-point tolerance rather than merely "closely".
        assert np.allclose(dense.harvest(task[0]), sparse.harvest(task[0]),
                           atol=1e-9)
        assert self._fit_err(dense, task) == pytest.approx(
            self._fit_err(sparse, task), abs=1e-6)

    def test_sparse_stores_a_csr_matrix(self):
        sp = pytest.importorskip("scipy.sparse")
        Wf = topologies.random_reservoir(80, density=0.1, rng=0)
        assert sp.issparse(EchoStateNetwork(1, 1, Wf, sparse=True, seed=0).W)

    def test_scipy_sparse_input_is_accepted(self):
        sp = pytest.importorskip("scipy.sparse")
        Wf = topologies.random_reservoir(80, density=0.1, rng=0)
        esn = EchoStateNetwork(1, 1, sp.csr_matrix(Wf), spectral_radius=0.9, seed=0)
        assert np.asarray(esn.W).shape == (80, 80)

    def test_numba_matches_numpy(self, task):
        pytest.importorskip("numba")
        Wf = topologies.random_reservoir(150, density=0.1, rng=0)
        kw = dict(spectral_radius=0.9, washout=100, seed=1)
        e_np = EchoStateNetwork(1, 1, Wf, use_numba=False, **kw)
        e_nb = EchoStateNetwork(1, 1, Wf, use_numba=True, **kw)
        assert e_nb._numba_enabled is True
        assert np.allclose(e_np.harvest(task[0]), e_nb.harvest(task[0]), atol=1e-9)
        assert self._fit_err(e_np, task) == pytest.approx(
            self._fit_err(e_nb, task), abs=1e-6)

    def test_numba_is_off_for_paths_it_cannot_handle(self):
        Wf = topologies.random_reservoir(80, density=0.1, rng=0)
        assert EchoStateNetwork(1, 1, Wf, use_numba=True,
                                dtype=np.float32)._numba_enabled is False
        assert EchoStateNetwork(1, 1, Wf, use_numba=True,
                                activation="sin")._numba_enabled is False
        assert EchoStateNetwork(1, 1, Wf, use_numba=True,
                                leaking_rate=np.full(80, 0.5))._numba_enabled is False

    def test_auto_numba_respects_the_size_threshold(self):
        from esnfed.esn import _NUMBA_AUTO_MAX_N
        big = EchoStateNetwork(1, 1, topologies.random_reservoir(
            _NUMBA_AUTO_MAX_N + 20, density=0.01, rng=0))
        assert big._numba_enabled is False

    def test_use_numba_false_forces_the_numpy_path(self):
        Wf = topologies.random_reservoir(60, density=0.1, rng=0)
        assert EchoStateNetwork(1, 1, Wf, use_numba=False)._numba_enabled is False
