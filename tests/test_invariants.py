"""Mathematical invariants the library must satisfy.

These tests encode the *claims* the library makes -- exactness of federated
aggregation, equivalence of streaming and batch training, the echo state
property, the formal DP guarantee -- rather than particular numbers. They are the
tests that would catch a subtly wrong reimplementation of any component.
"""
import math

import numpy as np
import pytest

from conftest import class_sequences, multivariate_task
from esnfed import (
    DeepEchoStateNetwork,
    EchoStateNetwork,
    datasets,
    federated,
    metrics,
    topologies,
)
from esnfed import classification as clf
from esnfed.esn import ridge_statistics, solve_readout
from esnfed.privacy import (
    PrivacyConfig,
    _log_std_normal_sf,
    _std_normal_cdf,
    dp_statistics,
    gaussian_sigma,
    secure_sum,
)
from esnfed.streaming import RLSReadout, StreamingRidge


def assert_relatively_close(a, b, rtol):
    """Assert ``a`` and ``b`` agree to ``rtol`` *relative to their own scale*.

    ``np.allclose``'s per-element rtol is unhelpful for quantities whose entries
    span many orders of magnitude (a Gram matrix, a reservoir readout): a tiny
    entry fails a relative test that the array as a whole passes comfortably.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    scale = max(np.max(np.abs(a)), np.max(np.abs(b)), 1e-300)
    err = np.max(np.abs(a - b)) / scale
    assert err <= rtol, f"relative deviation {err:.3g} exceeds {rtol:g}"


# ─────────────────────────────────────────────────────────────────────────────
#  Exactness of federated aggregation
# ─────────────────────────────────────────────────────────────────────────────
class TestFederatedExactness:
    """Summed sufficient statistics must reproduce pooled training *exactly*.

    This is the central claim of the project: it is what makes the federated
    readout lossless rather than an approximation.
    """

    def test_matches_centralized(self, federation):
        clients, ref, *_ = federation
        W_fed = federated.federated_ridge(clients, ref)
        W_central = federated.train_centralized(ref, clients).W_out
        assert np.allclose(W_fed, W_central, atol=1e-10)

    @pytest.mark.parametrize("n_clients", [1, 2, 3, 7, 11])
    def test_summed_statistics_equal_the_pooled_gram(self, narma, reservoir,
                                                     n_clients):
        """``sum_k Z_k^T Z_k == (stack Z_k)^T (stack Z_k)`` to machine precision.

        This is the exactness claim at the level where it actually holds --- the
        *statistics*. It is independent of how many clients the (fixed) data is
        split across.
        """
        u_tr, y_tr, _, _ = narma
        kw = dict(spectral_radius=0.9, leaking_rate=0.6, ridge=1e-6, washout=20)
        parts = datasets.partition_iid(u_tr, y_tr, n_clients)
        clients, _ = federated.make_shared_clients(
            reservoir, parts, input_seed=1, esn_kwargs=kw)
        A = sum(ridge_statistics(c.states(), c.targets())[0] for c in clients)
        B = sum(ridge_statistics(c.states(), c.targets())[1] for c in clients)
        A_pool, B_pool = ridge_statistics(
            np.vstack([c.states() for c in clients]),
            np.vstack([c.targets() for c in clients]))
        assert_relatively_close(A, A_pool, 1e-12)
        assert_relatively_close(B, B_pool, 1e-12)

    def test_matches_single_pooled_fit(self, narma, reservoir):
        """Against a genuine pooled fit, not just the same summation path.

        The *readout* is compared through its predictions: with ridge=1e-6 the
        Gram matrix has a condition number around 1e10, so W_out itself is only
        pinned down to about cond * eps even though the statistics agree to 1e-16
        (see :func:`test_readout_is_conditioning_limited`).
        """
        u_tr, y_tr, u_te, y_te = narma
        kw = dict(spectral_radius=0.9, leaking_rate=0.6, ridge=1e-6, washout=20)
        parts = datasets.partition_iid(u_tr, y_tr, 4)
        clients, ref = federated.make_shared_clients(
            reservoir, parts, input_seed=1, esn_kwargs=kw)
        W_fed = federated.federated_ridge(clients, ref)
        A, B = ridge_statistics(np.vstack([c.states() for c in clients]),
                                np.vstack([c.targets() for c in clients]))
        W_pool = solve_readout(A, B, ref.ridge)
        Z = ref.harvest(u_te)[ref.washout:]
        assert_relatively_close(Z @ W_fed, Z @ W_pool, 1e-5)

    def test_client_order_does_not_matter(self, federation):
        """Summation is commutative, so only floating-point order can differ."""
        clients, ref, u_te, _, _ = federation
        a = federated.federated_ridge(clients, ref)
        b = federated.federated_ridge(list(reversed(clients)), ref)
        Z = ref.harvest(u_te)[ref.washout:]
        assert_relatively_close(Z @ a, Z @ b, 1e-5)

    def test_readout_is_conditioning_limited(self, narma, reservoir):
        """Document *why* W_out cannot be compared at 1e-12.

        The statistics agree to machine precision, but the ridge solve amplifies
        that by the condition number of the regularised Gram matrix. A larger
        ridge conditions the problem better and tightens the agreement --- which
        is the behaviour to expect, and a guard against someone "fixing" a future
        tolerance failure by loosening the wrong assertion.
        """
        u_tr, y_tr, _, _ = narma
        deviations = {}
        for ridge in (1e-6, 1e-1):
            kw = dict(spectral_radius=0.9, leaking_rate=0.6, ridge=ridge,
                      washout=20)
            parts = datasets.partition_iid(u_tr, y_tr, 5)
            clients, ref = federated.make_shared_clients(
                reservoir, parts, input_seed=1, esn_kwargs=kw)
            A = sum(ridge_statistics(c.states(), c.targets())[0] for c in clients)
            B = sum(ridge_statistics(c.states(), c.targets())[1] for c in clients)
            A_pool, B_pool = ridge_statistics(
                np.vstack([c.states() for c in clients]),
                np.vstack([c.targets() for c in clients]))
            # statistics: machine precision
            assert_relatively_close(A, A_pool, 1e-12)
            w1 = solve_readout(A, B, ridge)
            w2 = solve_readout(A_pool, B_pool, ridge)
            deviations[ridge] = (np.max(np.abs(w1 - w2))
                                 / max(np.max(np.abs(w1)), 1e-300))
        assert deviations[1e-1] < deviations[1e-6]
        assert deviations[1e-1] < 1e-10

    def test_partitioning_shifts_the_states_because_clients_start_at_rest(
            self, narma, reservoir):
        """A documented limitation, not a defect.

        Every client harvests from a zero initial state, so splitting one series
        across more clients introduces more transients and the *aggregate*
        statistics genuinely change. Exactness is therefore a statement about a
        fixed partition. A washout suppresses most of the effect, which is the
        reason it exists.
        """
        u_tr, y_tr, _, _ = narma

        def per_sample_gram(n_clients, washout):
            kw = dict(spectral_radius=0.9, leaking_rate=0.6, ridge=1e-6,
                      washout=washout)
            parts = datasets.partition_iid(u_tr, y_tr, n_clients)
            clients, _ = federated.make_shared_clients(
                reservoir, parts, input_seed=1, esn_kwargs=kw)
            total = sum(c.n_samples for c in clients)
            A = sum(ridge_statistics(c.states(), c.targets())[0] for c in clients)
            return A / total

        def drift(washout):
            a, b = per_sample_gram(1, washout), per_sample_gram(5, washout)
            return np.max(np.abs(a - b)) / np.max(np.abs(a))

        assert drift(0) > 1e-3               # the transient really does matter
        assert drift(50) < drift(0) / 5      # and a washout largely removes it

    def test_multivariate_multioutput_is_exact(self, reservoir):
        u, y = multivariate_task(n_steps=800, n_inputs=3, n_outputs=2)
        kw = dict(spectral_radius=0.9, leaking_rate=0.5, ridge=1e-5, washout=30)
        parts = datasets.partition_iid(u, y, 4)
        clients, ref = federated.make_shared_clients(
            reservoir, parts, input_seed=2, esn_kwargs=kw)
        W_fed = federated.federated_ridge(clients, ref)
        assert W_fed.shape == (ref.readout_dim, 2)
        W_central = federated.train_centralized(ref, clients).W_out
        assert np.allclose(W_fed, W_central, atol=1e-10)

    def test_deep_esn_is_exact(self, narma):
        u_tr, y_tr, _, _ = narma
        Ws = [topologies.random_reservoir(40, density=0.1, rng=i) for i in range(3)]

        def make():
            return DeepEchoStateNetwork(1, 1, Ws, spectral_radius=0.9,
                                        leaking_rate=[0.9, 0.5, 0.2],
                                        washout=40, seed=0)

        parts = datasets.partition_iid(u_tr, y_tr, 4)
        clients = [federated.Client(make(), pu, py) for pu, py in parts]
        ref = make()
        W_fed = federated.federated_ridge(clients, ref)
        federated.train_centralized(ref, clients)
        assert np.allclose(W_fed, ref.W_out, atol=1e-10)

    def test_classification_is_exact_under_label_skew(self, reservoir):
        """One client per class -- the hardest possible split -- is still exact."""
        X, y = class_sequences()
        esn = EchoStateNetwork(1, 3, reservoir, spectral_radius=0.9,
                               leaking_rate=0.3, washout=0, ridge=1e-3, seed=0)
        W_central = clf.train_classifier(esn, X, y, 3)
        per_class = [([X[i] for i in np.where(y == c)[0]], y[y == c])
                     for c in range(3)]
        W_fed = clf.federated_classifier(esn, per_class, 3)
        assert np.allclose(W_central, W_fed, atol=1e-9)

    def test_secure_aggregation_recovers_the_exact_readout(self, federation):
        clients, ref, *_ = federation
        exact = federated.federated_ridge(clients, ref)
        secure = federated.federated_ridge_secure(clients, ref, seed=5,
                                                  mask_scale=1.0)
        assert np.allclose(exact, secure, rtol=1e-3, atol=1e-6)

    def test_secure_sum_is_the_plain_sum(self):
        rng = np.random.default_rng(0)
        mats = [rng.normal(size=(7, 4)) for _ in range(6)]
        assert np.allclose(secure_sum(mats, rng=np.random.default_rng(1), scale=50.0),
                           sum(mats), atol=1e-8)

    def test_fit_equals_statistics_then_solve(self, narma, reservoir):
        u_tr, y_tr, _, _ = narma
        kw = dict(spectral_radius=0.9, leaking_rate=0.6, ridge=1e-6,
                  washout=50, seed=3)
        a = EchoStateNetwork(1, 1, reservoir, **kw).fit(u_tr, y_tr)
        b = EchoStateNetwork(1, 1, reservoir, **kw)
        A, B = b.local_statistics(u_tr, y_tr)
        assert np.allclose(a.W_out, solve_readout(A, B, b.ridge), atol=1e-12)


# ─────────────────────────────────────────────────────────────────────────────
#  Streaming / incremental equivalence
# ─────────────────────────────────────────────────────────────────────────────
class TestStreamingEquivalence:

    @pytest.fixture
    def data(self):
        rng = np.random.default_rng(4)
        return rng.normal(size=(240, 9)), rng.normal(size=(240, 2))

    @pytest.mark.parametrize("n_chunks", [1, 2, 5, 240])
    def test_chunking_does_not_matter(self, data, n_chunks):
        Z, Y = data
        sr = StreamingRidge(9, 2, ridge=1e-3)
        for idx in np.array_split(np.arange(len(Z)), n_chunks):
            sr.update(Z[idx], Y[idx])
        A, B = ridge_statistics(Z, Y)
        assert sr.n_seen == len(Z)
        assert np.allclose(sr.readout(), solve_readout(A, B, 1e-3), atol=1e-10)

    def test_merge_equals_pooled(self, data):
        Z, Y = data
        parts = np.array_split(np.arange(len(Z)), 4)
        accs = [StreamingRidge(9, 2, ridge=1e-3).update(Z[p], Y[p]) for p in parts]
        merged = accs[0]
        for other in accs[1:]:
            merged.merge(other)
        A, B = ridge_statistics(Z, Y)
        assert merged.n_seen == len(Z)
        assert np.allclose(merged.readout(), solve_readout(A, B, 1e-3), atol=1e-10)

    def test_merge_is_commutative(self, data):
        Z, Y = data
        p1, p2 = np.arange(100), np.arange(100, 240)
        a = StreamingRidge(9, 2, 1e-3).update(Z[p1], Y[p1])
        b = StreamingRidge(9, 2, 1e-3).update(Z[p2], Y[p2])
        c = StreamingRidge(9, 2, 1e-3).update(Z[p1], Y[p1])
        d = StreamingRidge(9, 2, 1e-3).update(Z[p2], Y[p2])
        assert np.allclose(a.merge(b).readout(), d.merge(c).readout(), atol=1e-10)

    def test_rls_converges_to_batch_ridge(self, data):
        Z, Y = data
        rls = RLSReadout(9, 2, ridge=1e-2).update_batch(Z, Y)
        A, B = ridge_statistics(Z, Y)
        assert np.allclose(rls.readout(), solve_readout(A, B, 1e-2),
                           rtol=1e-4, atol=1e-6)

    def test_rls_matches_streaming_ridge(self, data):
        Z, Y = data
        rls = RLSReadout(9, 2, ridge=1e-2).update_batch(Z, Y)
        sr = StreamingRidge(9, 2, ridge=1e-2).update(Z, Y)
        assert np.allclose(rls.readout(), sr.readout(), rtol=1e-4, atol=1e-6)

    def test_forgetting_tracks_a_regime_change(self):
        """With forgetting < 1 the readout must follow a switch in the target."""
        rng = np.random.default_rng(5)
        Z = rng.normal(size=(600, 4))
        w_old, w_new = np.array([1.0, 0, 0, 0]), np.array([0, 0, 0, 1.0])
        Y = np.concatenate([Z[:300] @ w_old, Z[300:] @ w_new]).reshape(-1, 1)
        adaptive = RLSReadout(4, 1, ridge=1e-3, forgetting=0.97).update_batch(Z, Y)
        stationary = RLSReadout(4, 1, ridge=1e-3, forgetting=1.0).update_batch(Z, Y)
        err_a = np.linalg.norm(adaptive.readout().ravel() - w_new)
        err_s = np.linalg.norm(stationary.readout().ravel() - w_new)
        assert err_a < err_s

    def test_streaming_reproduces_an_esn_fit(self, narma, reservoir):
        """The streaming path is a drop-in for ESN.fit on the same states.

        Compared through the statistics (exact) and the predictions
        (well-conditioned); see ``test_readout_is_conditioning_limited`` for why
        W_out itself is not compared at machine precision.
        """
        u_tr, y_tr, u_te, y_te = narma
        esn = EchoStateNetwork(1, 1, reservoir, spectral_radius=0.9,
                               leaking_rate=0.6, ridge=1e-6, washout=50, seed=6)
        esn.fit(u_tr, y_tr)
        Z = esn.harvest(u_tr)[esn.washout:]
        Y = y_tr[esn.washout:]
        sr = StreamingRidge(esn.readout_dim, 1, ridge=esn.ridge)
        for idx in np.array_split(np.arange(len(Z)), 7):
            sr.update(Z[idx], Y[idx])
        A_batch, B_batch = ridge_statistics(Z, Y)
        assert_relatively_close(sr.A, A_batch, 1e-12)
        assert_relatively_close(sr.B, B_batch, 1e-12)
        Z_te = esn.harvest(u_te)[esn.washout:]
        assert_relatively_close(Z_te @ sr.readout(), esn.predict(u_te)[esn.washout:],
                                1e-5)


# ─────────────────────────────────────────────────────────────────────────────
#  Reservoir dynamics
# ─────────────────────────────────────────────────────────────────────────────
class TestEchoStateProperty:

    @pytest.mark.parametrize("rho", [0.3, 0.6, 0.9])
    def test_state_forgets_its_initial_condition(self, rho):
        W = topologies.random_reservoir(100, density=0.1, rng=0)
        esn = EchoStateNetwork(1, 1, W, spectral_radius=rho, leaking_rate=1.0,
                               seed=0, use_numba=False)
        u = np.random.default_rng(1).uniform(0, 0.5, size=(400, 1))
        Z0 = esn.harvest(u, x0=np.zeros(100))
        Z1 = esn.harvest(u, x0=np.random.default_rng(2).standard_normal(100))
        assert np.linalg.norm(Z0[-1] - Z1[-1]) < 1e-6
        assert np.linalg.norm(Z0[-1] - Z1[-1]) < np.linalg.norm(Z0[2] - Z1[2])

    def test_smaller_radius_forgets_faster(self):
        W = topologies.random_reservoir(100, density=0.1, rng=0)
        u = np.random.default_rng(1).uniform(0, 0.5, size=(200, 1))
        x0 = np.random.default_rng(2).standard_normal(100)

        def divergence_at(rho, t):
            esn = EchoStateNetwork(1, 1, W, spectral_radius=rho,
                                   leaking_rate=1.0, seed=0, use_numba=False)
            a = esn.harvest(u, x0=np.zeros(100))
            b = esn.harvest(u, x0=x0)
            return np.linalg.norm(a[t] - b[t])

        assert divergence_at(0.4, 12) < divergence_at(0.95, 12)

    def test_states_stay_bounded_in_tanh_reservoirs(self):
        W = topologies.random_reservoir(80, density=0.15, rng=0)
        esn = EchoStateNetwork(1, 1, W, spectral_radius=0.99, seed=0)
        u = np.random.default_rng(3).uniform(-2, 2, size=(300, 1))
        states = esn.harvest(u)[:, 2:]
        assert np.all(np.abs(states) <= 1.0 + 1e-12)

    def test_leaking_rate_controls_smoothness(self):
        """A small leaking rate must produce slower-varying states."""
        W = topologies.random_reservoir(80, density=0.1, rng=0)
        u = np.random.default_rng(4).uniform(0, 0.5, size=(400, 1))

        def roughness(a):
            esn = EchoStateNetwork(1, 1, W, spectral_radius=0.9, leaking_rate=a,
                                   seed=0, use_numba=False)
            x = esn.harvest(u)[:, 2:]
            return float(np.mean(np.abs(np.diff(x, axis=0))))

        assert roughness(0.1) < roughness(1.0)

    def test_permuting_reservoir_nodes_leaves_predictions_unchanged(self, narma):
        """Node labels are arbitrary: conjugating W by a permutation is a no-op.

        Catches indexing errors in the harvest loop that a plain shape check
        would miss.
        """
        u_tr, y_tr, u_te, y_te = narma
        W = topologies.random_reservoir(60, density=0.15, rng=0)
        rng = np.random.default_rng(7)
        W_in = rng.uniform(-1, 1, size=(60, 2))
        perm = rng.permutation(60)

        base = EchoStateNetwork(1, 1, W, spectral_radius=0.9, leaking_rate=0.7,
                                washout=50, input_weights=W_in, use_numba=False)
        permuted = EchoStateNetwork(
            1, 1, W[np.ix_(perm, perm)], spectral_radius=0.9, leaking_rate=0.7,
            washout=50, input_weights=W_in[perm], use_numba=False)
        base.fit(u_tr, y_tr)
        permuted.fit(u_tr, y_tr)
        assert np.allclose(base.predict(u_te), permuted.predict(u_te), atol=1e-8)

    def test_heterogeneous_leak_permutes_consistently(self):
        """Per-node leaking rates must be permuted along with the nodes."""
        W = topologies.random_reservoir(50, density=0.15, rng=0)
        rng = np.random.default_rng(8)
        W_in = rng.uniform(-1, 1, size=(50, 2))
        a = topologies.leaking_rates(50, "uniform", low=0.2, high=0.9, rng=1)
        perm = rng.permutation(50)
        u = rng.uniform(0, 0.5, size=(150, 1))

        base = EchoStateNetwork(1, 1, W, spectral_radius=0.9, leaking_rate=a,
                                input_weights=W_in)
        perm_esn = EchoStateNetwork(1, 1, W[np.ix_(perm, perm)],
                                    spectral_radius=0.9, leaking_rate=a[perm],
                                    input_weights=W_in[perm])
        x_base = base.harvest(u)[:, 2:]
        x_perm = perm_esn.harvest(u)[:, 2:]
        assert np.allclose(x_base[:, perm], x_perm, atol=1e-9)

    def test_bias_column_is_the_configured_constant(self, reservoir):
        esn = EchoStateNetwork(1, 1, reservoir, bias=0.37, seed=0)
        Z = esn.harvest(np.zeros((25, 1)))
        assert np.allclose(Z[:, 0], 0.37)

    def test_input_column_echoes_the_input(self, reservoir):
        esn = EchoStateNetwork(2, 1, reservoir, seed=0)
        u = np.random.default_rng(0).normal(size=(30, 2))
        Z = esn.harvest(u)
        assert np.allclose(Z[:, 1:3], u)

    def test_zero_reservoir_is_a_static_readout(self, narma):
        """With W = 0 the reservoir is memoryless, so it cannot solve NARMA-10."""
        u_tr, y_tr, u_te, y_te = narma
        esn = EchoStateNetwork(1, 1, np.zeros((30, 30)), washout=50, seed=0)
        esn.fit(u_tr, y_tr)
        err = metrics.nrmse(y_te[50:], esn.predict(u_te)[50:])
        assert err > 0.5      # no memory -> cannot beat the trivial predictor much


class TestLearningBehaviour:

    def test_trained_esn_beats_the_trivial_predictor(self, narma, reservoir):
        u_tr, y_tr, u_te, y_te = narma
        esn = EchoStateNetwork(1, 1, reservoir, spectral_radius=0.9,
                               leaking_rate=1.0, washout=50, seed=0)
        esn.fit(u_tr, y_tr)
        assert metrics.nrmse(y_te[50:], esn.predict(u_te)[50:]) < 0.5

    def test_bigger_reservoir_is_not_worse(self, narma):
        u_tr, y_tr, u_te, y_te = narma

        def err(n):
            W = topologies.random_reservoir(n, density=0.1, rng=0)
            esn = EchoStateNetwork(1, 1, W, spectral_radius=0.9, leaking_rate=1.0,
                                   washout=50, ridge=1e-6, seed=0).fit(u_tr, y_tr)
            return metrics.nrmse(y_te[50:], esn.predict(u_te)[50:])

        assert err(200) < err(20)

    def test_stronger_ridge_shrinks_the_readout(self, narma, reservoir):
        u_tr, y_tr, _, _ = narma
        norms = []
        for ridge in (1e-8, 1e-4, 1e-1, 1e2):
            esn = EchoStateNetwork(1, 1, reservoir, spectral_radius=0.9,
                                   washout=50, ridge=ridge, seed=0).fit(u_tr, y_tr)
            norms.append(np.linalg.norm(esn.W_out))
        assert norms == sorted(norms, reverse=True)

    def test_ridge_solution_satisfies_its_normal_equations(self, narma, reservoir):
        u_tr, y_tr, _, _ = narma
        esn = EchoStateNetwork(1, 1, reservoir, spectral_radius=0.9, washout=50,
                               ridge=1e-4, seed=0)
        A, B = esn.local_statistics(u_tr, y_tr)
        W = solve_readout(A, B, 1e-4)
        assert np.allclose(A @ W + 1e-4 * W, B, atol=1e-6)

    def test_federated_is_no_worse_than_the_average_local_model(self, federation):
        clients, ref, u_te, y_te, _ = federation
        wo = ref.washout
        W_fed = federated.federated_ridge(clients, ref)
        fed_err = metrics.nrmse(y_te[wo:], ref.harvest(u_te)[wo:] @ W_fed)
        federated.train_local(clients)
        local = [metrics.nrmse(y_te[wo:], c.esn.predict(u_te)[wo:]) for c in clients]
        assert fed_err <= np.mean(local) + 1e-9

    def test_ensemble_beats_the_average_member(self, narma, federation):
        _, _, u_te, y_te, kw = federation
        clients, *_ = federation
        parts = [(c.u, c.y) for c in clients]
        kinds = ["random", "small_world", "scale_free", "ring", "random"]
        reservoirs = [topologies.make_reservoir(k, 120, rng=i)
                      for i, k in enumerate(kinds)]
        het = federated.make_heterogeneous_clients(reservoirs, parts, esn_kwargs=kw)
        federated.train_local(het)
        wo = het[0].esn.washout
        members = [metrics.nrmse(y_te[wo:], c.esn.predict(u_te)[wo:]) for c in het]
        ens = metrics.nrmse(
            y_te[wo:], federated.ensemble_predict(het, u_te)[wo:])
        assert ens <= np.mean(members) + 1e-9

    def test_fedavg_with_equal_clients_is_the_plain_mean(self, narma, reservoir):
        """Weighted aggregation must reduce to a simple average for equal shares."""
        u_tr, y_tr, u_te, y_te = narma
        n = (len(u_tr) // 4) * 4
        kw = dict(spectral_radius=0.9, leaking_rate=0.6, ridge=1e-6, washout=20)
        parts = datasets.partition_iid(u_tr[:n], y_tr[:n], 4)
        assert len({len(p) for p, _ in parts}) == 1        # equal shares
        clients, ref = federated.make_shared_clients(
            reservoir, parts, input_seed=0, esn_kwargs=kw)
        W, history = federated.fedavg(clients, ref, u_te, y_te, rounds=3,
                                      local_epochs=2, lr=1.0)
        manual = np.zeros_like(W)
        for _ in range(3):
            updates = [federated._local_gradient_step(
                manual, c.states(), c.targets(), 1.0, ref.ridge, 2)
                for c in clients]
            manual = sum(updates) / len(updates)
        assert np.allclose(W, manual, atol=1e-10)
        assert len(history) == 3 and np.all(np.isfinite(history))

    def test_fedavg_error_decreases(self, federation):
        clients, ref, u_te, y_te, _ = federation
        _, history = federated.fedavg(clients, ref, u_te, y_te, rounds=25,
                                      local_epochs=5, lr=0.9)
        assert history[-1] < history[0]

    def test_classifier_beats_chance(self, reservoir):
        Xtr, ytr = class_sequences(seed=1)
        Xte, yte = class_sequences(seed=2)
        esn = EchoStateNetwork(1, 3, reservoir, spectral_radius=0.9,
                               leaking_rate=0.3, washout=0, ridge=1e-3, seed=0)
        W = clf.train_classifier(esn, Xtr, ytr, 3)
        assert clf.accuracy(yte, clf.predict_labels(esn, Xte, W)) > 0.7

    def test_probabilities_are_a_distribution(self, reservoir):
        X, y = class_sequences()
        esn = EchoStateNetwork(1, 3, reservoir, washout=0, ridge=1e-3, seed=0)
        P = clf.predict_proba(esn, X, clf.train_classifier(esn, X, y, 3))
        assert np.allclose(P.sum(axis=1), 1.0)
        assert np.all(P >= 0) and np.all(P <= 1)

    def test_argmax_of_proba_equals_predicted_label(self, reservoir):
        X, y = class_sequences()
        esn = EchoStateNetwork(1, 3, reservoir, washout=0, ridge=1e-3, seed=0)
        W = clf.train_classifier(esn, X, y, 3)
        assert np.array_equal(clf.predict_labels(esn, X, W),
                              np.argmax(clf.predict_proba(esn, X, W), axis=1))


# ─────────────────────────────────────────────────────────────────────────────
#  Differential privacy
# ─────────────────────────────────────────────────────────────────────────────
class TestDifferentialPrivacy:

    @pytest.mark.parametrize("eps", [0.05, 0.5, 1.0, 2.0, 8.0, 50.0, 500.0])
    @pytest.mark.parametrize("delta", [1e-8, 1e-5, 1e-2])
    def test_analytic_sigma_satisfies_the_dp_condition(self, eps, delta):
        """B(sigma/Delta) <= delta is the exact (eps, delta)-DP criterion."""
        sens = 2.5
        s = gaussian_sigma(eps, delta, sens) / sens
        t = 1.0 / (2.0 * s) + eps * s
        B = _std_normal_cdf(1.0 / (2.0 * s) - eps * s) - \
            math.exp(eps + _log_std_normal_sf(t))
        assert delta * (1.0 + 1e-6) + 1e-12 >= B

    @pytest.mark.parametrize("eps", [0.05, 0.5, 1.0])
    def test_analytic_never_looser_than_classic(self, eps):
        a = gaussian_sigma(eps, 1e-5, 1.0, method="analytic")
        c = gaussian_sigma(eps, 1e-5, 1.0, method="classic")
        assert a <= c + 1e-12

    def test_sigma_is_linear_in_sensitivity(self):
        base = gaussian_sigma(0.5, 1e-5, 1.0)
        for k in (2.0, 7.5):
            assert gaussian_sigma(0.5, 1e-5, k) == pytest.approx(k * base)

    def test_sigma_decreases_with_epsilon(self):
        sigmas = [gaussian_sigma(e, 1e-5, 1.0)
                  for e in (0.01, 0.1, 1.0, 10.0, 100.0, 1e4)]
        assert sigmas == sorted(sigmas, reverse=True)

    def test_sigma_decreases_with_delta(self):
        sigmas = [gaussian_sigma(0.5, d, 1.0) for d in (1e-9, 1e-6, 1e-3, 1e-1)]
        assert sigmas == sorted(sigmas, reverse=True)

    def test_large_epsilon_does_not_overflow(self):
        """exp(epsilon) used to overflow before multiplying an underflowed tail."""
        assert gaussian_sigma(1e5, 1e-5, 1.0) > 0

    def test_zero_sensitivity_needs_no_noise(self):
        assert gaussian_sigma(1.0, 1e-5, 0.0) == 0.0

    def test_classic_bound_warns_above_epsilon_one(self):
        with pytest.warns(UserWarning, match="epsilon <= 1"):
            gaussian_sigma(2.0, 1e-5, 1.0, method="classic")

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match=r"analytic.*classic"):
            gaussian_sigma(1.0, 1e-5, 1.0, method="bogus")

    def test_noise_is_zero_mean(self, federation):
        clients, _, *_ = federation
        Z, Y = clients[0].states(), clients[0].targets()
        cz = float(np.linalg.norm(Z, axis=1).max())
        cy = float(np.linalg.norm(Y, axis=1).max())
        cfg = PrivacyConfig(epsilon=0.5, delta=1e-5, clip_state=cz, clip_target=cy)
        A_ref, B_ref = ridge_statistics(Z, Y)
        sigma = gaussian_sigma(cfg.epsilon, cfg.delta, cfg.sensitivity())
        draws = [dp_statistics(Z, Y, cfg, rng=np.random.default_rng(i))
                 for i in range(150)]
        A_mean = np.mean([a for a, _ in draws], axis=0)
        assert np.max(np.abs(A_mean - A_ref)) < 6 * sigma / math.sqrt(150)

    def test_released_gram_stays_symmetric(self, federation):
        clients, _, *_ = federation
        cfg = PrivacyConfig(epsilon=0.5, clip_state=3.0, clip_target=1.0)
        A, _ = dp_statistics(clients[0].states(), clients[0].targets(), cfg,
                             rng=np.random.default_rng(0))
        assert np.allclose(A, A.T)

    def test_tighter_budget_means_more_noise(self, federation):
        clients, _, *_ = federation
        Z, Y = clients[0].states(), clients[0].targets()
        A_ref, _ = ridge_statistics(Z, Y)

        def deviation(eps):
            cfg = PrivacyConfig(epsilon=eps, clip_state=3.0, clip_target=1.0)
            A, _ = dp_statistics(Z, Y, cfg, rng=np.random.default_rng(0))
            return np.linalg.norm(A - A_ref)

        assert deviation(0.1) > deviation(10.0)

    def test_clipping_bounds_every_row(self):
        from esnfed.privacy import clip_rows
        M = np.random.default_rng(0).normal(size=(50, 6)) * 10
        C = clip_rows(M, 2.0)
        assert np.all(np.linalg.norm(C, axis=1) <= 2.0 + 1e-9)

    def test_clipping_leaves_small_rows_untouched(self):
        from esnfed.privacy import clip_rows
        M = np.array([[0.1, 0.0], [3.0, 4.0]])
        C = clip_rows(M, 1.0)
        assert np.allclose(C[0], M[0])
        assert np.linalg.norm(C[1]) == pytest.approx(1.0)

    def test_dp_readout_is_finite_and_usable(self, federation):
        clients, ref, u_te, y_te, _ = federation
        cfg = PrivacyConfig(epsilon=0.5, delta=1e-5, clip_state=5.0,
                            clip_target=1.0, seed=0)
        W = federated.federated_ridge_dp(clients, ref, cfg)
        assert np.all(np.isfinite(W))
        wo = ref.washout
        assert metrics.nrmse(y_te[wo:], ref.harvest(u_te)[wo:] @ W) < 3.0

    def test_dp_differs_from_exact(self, federation):
        clients, ref, *_ = federation
        exact = federated.federated_ridge(clients, ref)
        cfg = PrivacyConfig(epsilon=0.5, clip_state=5.0, clip_target=1.0, seed=1)
        assert not np.allclose(federated.federated_ridge_dp(clients, ref, cfg),
                               exact)

    def test_per_client_noise_is_independent(self, federation):
        """Two clients must not receive the same noise draw."""
        clients, ref, *_ = federation
        cfg = PrivacyConfig(epsilon=0.5, clip_state=5.0, clip_target=1.0, seed=0)
        a = dp_statistics(clients[0].states(), clients[0].targets(), cfg,
                          rng=np.random.default_rng(1))[0]
        b = dp_statistics(clients[0].states(), clients[0].targets(), cfg,
                          rng=np.random.default_rng(2))[0]
        assert not np.allclose(a, b)


# ─────────────────────────────────────────────────────────────────────────────
#  Structural alignment
# ─────────────────────────────────────────────────────────────────────────────
class TestStructuralAlignment:

    def test_interpolation_endpoints(self):
        A = np.ones((4, 4))
        B = np.zeros((4, 4))
        assert np.allclose(federated.interpolate_reservoir(A, B, 0.0), A)
        assert np.allclose(federated.interpolate_reservoir(A, B, 1.0), B)
        assert np.allclose(federated.interpolate_reservoir(A, B, 0.25), 0.75 * A)

    def test_heterogeneity_falls_to_zero(self, narma):
        u_tr, y_tr, u_te, y_te = narma
        parts = datasets.partition_iid(u_tr, y_tr, 3)
        Ws = [topologies.random_reservoir(60, density=0.1, rng=i) for i in range(3)]
        target = topologies.random_reservoir(60, density=0.1, rng=99)
        records = federated.structural_alignment(
            Ws, target, parts, u_te, y_te, alphas=np.linspace(0, 1, 5),
            esn_kwargs=dict(spectral_radius=0.9, washout=50))
        het = [r["heterogeneity"] for r in records]
        assert het == sorted(het, reverse=True)
        assert het[-1] == pytest.approx(0.0, abs=1e-9)

    def test_records_are_well_formed(self, narma):
        u_tr, y_tr, u_te, y_te = narma
        parts = datasets.partition_iid(u_tr, y_tr, 2)
        Ws = [topologies.random_reservoir(40, density=0.1, rng=i) for i in range(2)]
        records = federated.structural_alignment(
            Ws, Ws[0], parts, u_te, y_te, alphas=[0.0, 0.5, 1.0],
            esn_kwargs=dict(washout=50))
        assert len(records) == 3
        for r in records:
            assert set(r) == {"alpha", "heterogeneity", "ensemble_nrmse",
                              "fedavg_nrmse"}
            assert np.isfinite(r["ensemble_nrmse"])

    def test_mismatched_counts_raise(self, narma):
        u_tr, y_tr, u_te, y_te = narma
        parts = datasets.partition_iid(u_tr, y_tr, 3)
        Ws = [topologies.random_reservoir(40, density=0.1, rng=0)]
        with pytest.raises(ValueError, match="reservoirs for 3 partitions"):
            federated.structural_alignment(Ws, Ws[0], parts, u_te, y_te,
                                            alphas=[0.0])


# ─────────────────────────────────────────────────────────────────────────────
#  Client bookkeeping
# ─────────────────────────────────────────────────────────────────────────────
class TestClient:

    def test_states_are_harvested_once_and_cached(self, federation):
        clients, *_ = federation
        c = clients[0]
        assert c._Z is None
        first = c.states()
        assert c._Z is not None
        # The returned slice is a view onto the one cached harvest.
        assert np.shares_memory(first, c.states())

    def test_invalidate_forces_a_reharvest(self, federation):
        clients, *_ = federation
        c = clients[0]
        first = c.states().copy()
        c.invalidate()
        assert c._Z is None
        assert not np.shares_memory(first, c.states())
        assert np.allclose(c.states(), first)

    def test_n_samples_excludes_the_washout(self, federation):
        clients, *_ = federation
        c = clients[0]
        assert c.n_samples == len(c.u) - c.esn.washout
        assert c.states().shape[0] == c.n_samples
        assert c.targets().shape[0] == c.n_samples

    def test_length_mismatch_rejected(self, esn):
        with pytest.raises(ValueError, match="disagree on length"):
            federated.Client(esn, np.zeros((200, 1)), np.zeros((150, 1)))

    def test_flat_targets_accepted(self, esn):
        c = federated.Client(esn, np.zeros((200, 1)), np.zeros(200))
        assert c.targets().shape == (200 - esn.washout, 1)
