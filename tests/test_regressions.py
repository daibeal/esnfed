"""Regression tests: one per defect found in the library review.

Each test pins down a bug that used to produce a *silently wrong* result rather
than an error. They are grouped by the failure mode instead of by module, because
that is what makes the class of bug recognisable.
"""
import warnings

import numpy as np
import pytest

from esnfed import (
    DeepEchoStateNetwork,
    EchoStateNetwork,
    datasets,
    federated,
    metrics,
    topologies,
    viz,
)
from esnfed import classification as clf
from esnfed.esn import (
    _spectral_radius,
    _spectral_radius_gelfand,
    _spectral_radius_sparse,
    solve_readout,
)
from esnfed.llm_orchestration import BottleneckProjection, EdgeClient, Server, SurrogateLM
from esnfed.privacy import PrivacyConfig, dp_statistics, secure_sum, zero_sum_masks
from esnfed.streaming import RLSReadout, StreamingRidge


# ─────────────────────────────────────────────────────────────────────────────
#  Silently wrong numbers
# ─────────────────────────────────────────────────────────────────────────────
class TestMetricBroadcasting:
    """Metrics used to broadcast (T,) against (T, 1) into a (T, T) matrix.

    A near-perfect model then reported NRMSE 1.41 instead of 0.03, which is the
    difference between "useless" and "excellent".
    """

    def test_flat_vs_column_agree(self):
        yt = np.linspace(0.0, 1.0, 100)
        yp = yt.reshape(-1, 1) + 0.01
        assert metrics.nrmse(yt, yp) == pytest.approx(metrics.nrmse(yt, yp.ravel()))
        assert metrics.mse(yt, yp) == pytest.approx(1e-4, rel=1e-9)
        assert metrics.nrmse(yt, yp) < 0.05      # was 1.41 before the fix

    @pytest.mark.parametrize("metric", [metrics.mse, metrics.rmse, metrics.nrmse,
                                        metrics.r2_score, metrics.mae])
    def test_row_and_column_vectors_interchangeable(self, metric):
        yt = np.linspace(0.0, 1.0, 40)
        yp = yt + 0.05
        expected = metric(yt, yp)
        assert metric(yt.reshape(-1, 1), yp) == pytest.approx(expected)
        assert metric(yt, yp.reshape(1, -1)) == pytest.approx(expected)
        assert metric(yt.reshape(1, -1), yp.reshape(-1, 1)) == pytest.approx(expected)

    @pytest.mark.parametrize("metric", [metrics.mse, metrics.rmse, metrics.nrmse,
                                        metrics.r2_score, metrics.mae])
    def test_genuine_mismatch_raises(self, metric):
        with pytest.raises(ValueError, match="incompatible shapes"):
            metric(np.zeros(50), np.zeros(40))
        with pytest.raises(ValueError, match="incompatible shapes"):
            metric(np.zeros((30, 1)), np.zeros((30, 3)))

    def test_multioutput_is_compared_elementwise(self):
        yt = np.zeros((20, 3))
        yp = np.ones((20, 3))
        assert metrics.mse(yt, yp) == pytest.approx(1.0)
        assert metrics.mae(yt, yp) == pytest.approx(1.0)

    def test_constant_target_is_nan_not_a_crash(self):
        assert np.isnan(metrics.nrmse(np.ones(10), np.ones(10)))
        assert np.isnan(metrics.r2_score(np.ones(10), np.zeros(10)))

    def test_empty_input_is_nan(self):
        assert np.isnan(metrics.mse(np.zeros(0), np.zeros(0)))


class TestSparseSpectralRadius:
    """``sparse=True`` used to rescale with a power-iteration estimate.

    A real reservoir's dominant eigenvalue is normally a complex conjugate pair,
    so the iterates rotate and the growth ratio never converges: the reservoir
    silently ended up with rho = 0.9335 when 0.9 was requested.
    """

    @pytest.mark.parametrize("seed", range(4))
    def test_estimator_matches_exact_eigenvalue(self, seed):
        W = topologies.random_reservoir(150, density=0.06, rng=seed)
        assert _spectral_radius_sparse(W) == pytest.approx(
            _spectral_radius(W), rel=1e-6)

    @pytest.mark.parametrize("seed", range(4))
    def test_gelfand_fallback_is_accurate(self, seed):
        """The no-SciPy fallback must also be within ~1% (power iteration was 4%)."""
        W = topologies.random_reservoir(150, density=0.06, rng=seed)
        exact = _spectral_radius(W)
        assert _spectral_radius_gelfand(W) == pytest.approx(exact, rel=0.02)

    def test_sparse_esn_hits_the_requested_radius(self):
        pytest.importorskip("scipy.sparse")
        W = topologies.random_reservoir(150, density=0.06, rng=0)
        esn = EchoStateNetwork(1, 1, W, spectral_radius=0.9, sparse=True, seed=0)
        assert _spectral_radius(esn.W.toarray()) == pytest.approx(0.9, rel=1e-6)

    def test_sparse_and_dense_dynamics_now_agree(self):
        pytest.importorskip("scipy.sparse")
        W = topologies.random_reservoir(150, density=0.06, rng=0)
        u = np.random.default_rng(0).uniform(0, 0.5, size=(200, 1))
        dense = EchoStateNetwork(1, 1, W, spectral_radius=0.9, seed=3,
                                 use_numba=False)
        sparse = EchoStateNetwork(1, 1, W, spectral_radius=0.9, seed=3, sparse=True)
        assert np.allclose(dense.harvest(u), sparse.harvest(u), atol=1e-8)

    def test_zero_and_empty_matrices(self):
        assert _spectral_radius_sparse(np.zeros((5, 5))) == 0.0
        assert _spectral_radius_sparse(np.zeros((0, 0))) == 0.0


class TestNarma10Divergence:
    """NARMA-10 is only conditionally stable; ~1 seed in 100 overflowed to inf."""

    def test_the_known_bad_seed_is_now_stable(self):
        with pytest.warns(UserWarning, match="diverging NARMA-10"):
            u, y = datasets.narma10(1000, rng=83)
        assert np.all(np.isfinite(y))
        assert np.abs(y).max() < 5.0

    def test_no_seed_in_a_wide_sweep_diverges(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for seed in range(60):
                _, y = datasets.narma10(500, rng=seed)
                assert np.all(np.isfinite(y)), f"seed {seed} diverged"

    def test_stable_seeds_are_bit_identical_to_before(self):
        """The retry loop must not perturb the (vast majority of) good draws."""
        rng = np.random.default_rng(7)
        total = 500 + 50
        u_expected = rng.uniform(0.0, 0.5, size=total)[50:]
        u, _ = datasets.narma10(500, rng=7)
        assert np.allclose(u.ravel(), u_expected)

    def test_reproducible(self):
        a = datasets.narma10(400, rng=11)
        b = datasets.narma10(400, rng=11)
        assert np.allclose(a[0], b[0]) and np.allclose(a[1], b[1])


class TestInputLayoutCorruption:
    """A 2-D input whose second axis was not ``n_inputs`` used to be reshaped.

    For a transposed ``(n_inputs, T)`` array that reinterleaves the samples, so
    the reservoir ran on scrambled data and returned plausible-looking states.
    """

    def test_transposed_input_raises_with_a_hint(self, reservoir):
        esn = EchoStateNetwork(3, 1, reservoir, seed=0)
        with pytest.raises(ValueError, match="looks transposed"):
            esn.harvest(np.zeros((3, 40)))

    def test_wrong_width_raises(self, reservoir):
        esn = EchoStateNetwork(1, 1, reservoir, seed=0)
        with pytest.raises(ValueError, match=r"shape \(T, n_inputs\)"):
            esn.harvest(np.zeros((50, 4)))

    def test_indivisible_flat_input_raises(self, reservoir):
        esn = EchoStateNetwork(3, 1, reservoir, seed=0)
        with pytest.raises(ValueError, match="cannot interpret"):
            esn.harvest(np.zeros(50))       # 50 is not a multiple of 3

    def test_3d_input_raises(self, reservoir):
        esn = EchoStateNetwork(1, 1, reservoir, seed=0)
        with pytest.raises(ValueError, match="1-D or 2-D"):
            esn.harvest(np.zeros((4, 5, 6)))

    @pytest.mark.parametrize("shape", [(40,), (40, 1), (1, 40)])
    def test_unambiguous_univariate_layouts_still_work(self, reservoir, shape):
        esn = EchoStateNetwork(1, 1, reservoir, seed=0)
        assert esn.harvest(np.zeros(shape)).shape == (40, esn.readout_dim)

    def test_multivariate_layout_works(self, reservoir):
        esn = EchoStateNetwork(3, 1, reservoir, seed=0)
        assert esn.harvest(np.zeros((40, 3))).shape == (40, esn.readout_dim)

    def test_transposed_input_would_have_scrambled_the_data(self, reservoir):
        """Document *why* it matters: reshape != transpose."""
        u = np.arange(12.0).reshape(4, 3)          # 4 steps of 3 inputs
        assert not np.allclose(u.T.reshape(-1, 3), u)


class TestSilentZeroReadout:
    """A washout at least as long as the sequence produced an all-zero readout."""

    def test_fit_raises_instead(self, reservoir):
        esn = EchoStateNetwork(1, 1, reservoir, washout=100, seed=0)
        u, y = datasets.narma10(60, rng=0)
        with pytest.raises(ValueError, match="too short for washout"):
            esn.fit(u, y)

    def test_local_statistics_raises(self, reservoir):
        esn = EchoStateNetwork(1, 1, reservoir, washout=100, seed=0)
        u, y = datasets.narma10(60, rng=0)
        with pytest.raises(ValueError, match="too short for washout"):
            esn.local_statistics(u, y)

    def test_client_construction_raises(self, reservoir):
        esn = EchoStateNetwork(1, 1, reservoir, washout=100, seed=0)
        u, y = datasets.narma10(60, rng=0)
        with pytest.raises(ValueError, match="not more than the ESN washout"):
            federated.Client(esn, u, y)

    def test_deep_fit_raises(self):
        Ws = [topologies.random_reservoir(20, density=0.2, rng=i) for i in range(2)]
        deep = DeepEchoStateNetwork(1, 1, Ws, washout=100, seed=0)
        u, y = datasets.narma10(60, rng=0)
        with pytest.raises(ValueError, match="too short for washout"):
            deep.fit(u, y)


class TestNegativeAndDegenerateWeights:
    """Weight vectors that sum to zero returned all-``nan`` predictions."""

    def test_zero_weights_raise(self, federation):
        clients, *_ = federation
        federated.train_local(clients)
        with pytest.raises(ValueError, match="sum to a positive value"):
            federated.ensemble_predict(clients, clients[0].u,
                                       weights=np.zeros(len(clients)))

    def test_negative_weights_raise(self, federation):
        clients, *_ = federation
        federated.train_local(clients)
        w = np.ones(len(clients))
        w[0] = -1.0
        with pytest.raises(ValueError, match="non-negative"):
            federated.ensemble_predict(clients, clients[0].u, weights=w)

    def test_wrong_weight_count_raises(self, federation):
        clients, *_ = federation
        federated.train_local(clients)
        with pytest.raises(ValueError, match="expected 5 weights"):
            federated.ensemble_predict(clients, clients[0].u, weights=[1.0, 1.0])

    def test_valid_weights_still_work(self, federation):
        clients, _, u_te, _, _ = federation
        federated.train_local(clients)
        w = np.arange(1.0, len(clients) + 1.0)
        pred = federated.ensemble_predict(clients, u_te, weights=w)
        assert pred.shape == (len(u_te), 1)
        assert np.all(np.isfinite(pred))


class TestEmptyCohorts:
    """Aggregating zero clients produced a zero readout or an opaque matmul error."""

    @pytest.mark.parametrize("fn", ["federated_ridge", "train_local"])
    def test_aggregators_reject_empty(self, esn, fn):
        with pytest.raises(ValueError, match="no clients"):
            if fn == "train_local":
                federated.train_local([])
            else:
                federated.federated_ridge([], esn)

    def test_train_centralized_rejects_empty(self, esn):
        with pytest.raises(ValueError, match="no clients"):
            federated.train_centralized(esn, [])

    def test_fedavg_rejects_empty(self, esn, narma):
        _, _, u_te, y_te = narma
        with pytest.raises(ValueError, match="no clients"):
            federated.fedavg([], esn, u_te, y_te, rounds=2)

    def test_federated_classifier_rejects_empty(self, esn):
        with pytest.raises(ValueError, match="no clients"):
            clf.federated_classifier(esn, [], 3)

    def test_ensemble_classify_rejects_empty(self):
        with pytest.raises(ValueError, match="no ensemble members"):
            clf.ensemble_classify([], [np.zeros((10, 1))])


# ─────────────────────────────────────────────────────────────────────────────
#  Crashes on legitimate input
# ─────────────────────────────────────────────────────────────────────────────
class TestSparseVisualisation:
    """A ``sparse=True`` ESN broke every plot with an opaque numpy error."""

    @pytest.fixture
    def sparse_esn(self, reservoir):
        pytest.importorskip("scipy.sparse")
        return EchoStateNetwork(1, 1, reservoir, sparse=True, seed=0)

    def test_spectrum(self, sparse_esn):
        pytest.importorskip("matplotlib")
        assert viz.plot_spectrum(sparse_esn, backend="matplotlib") is not None

    def test_topology(self, sparse_esn):
        pytest.importorskip("matplotlib")
        assert viz.plot_reservoir(sparse_esn, backend="matplotlib") is not None

    def test_states(self, sparse_esn):
        pytest.importorskip("matplotlib")
        u = np.random.default_rng(0).uniform(0, 0.5, size=(120, 1))
        assert viz.plot_states(sparse_esn, u, backend="matplotlib") is not None

    def test_structural_alignment_handles_sparse(self, narma):
        pytest.importorskip("scipy.sparse")
        u_tr, y_tr, u_te, y_te = narma
        parts = datasets.partition_iid(u_tr, y_tr, 2)
        Ws = [topologies.random_reservoir(60, density=0.1, rng=i) for i in range(2)]
        target = topologies.random_reservoir(60, density=0.1, rng=9)
        records = federated.structural_alignment(
            Ws, target, parts, u_te, y_te, alphas=[0.0, 1.0],
            esn_kwargs=dict(washout=50, sparse=True),
        )
        assert records[-1]["heterogeneity"] == pytest.approx(0.0, abs=1e-9)

    def test_non_square_matrix_rejected(self):
        with pytest.raises(ValueError, match="square reservoir"):
            viz.plot_spectrum(np.zeros((3, 4)), backend="matplotlib")


class TestDpStatisticsTargets:
    """A 1-D target vector became a single row and crashed the Gram product."""

    def test_flat_targets_accepted(self):
        rng = np.random.default_rng(0)
        Z, Y = rng.normal(size=(60, 8)), rng.normal(size=60)
        A, B = dp_statistics(Z, Y, PrivacyConfig(epsilon=1.0),
                             rng=np.random.default_rng(1))
        assert A.shape == (8, 8) and B.shape == (8, 1)

    def test_flat_and_column_targets_agree_without_noise(self):
        """With a huge epsilon the noise vanishes, so both layouts must match."""
        rng = np.random.default_rng(0)
        Z, Y = rng.normal(size=(60, 5)), rng.normal(size=60)
        cfg = PrivacyConfig(epsilon=5000.0, clip_state=100.0, clip_target=100.0)
        A1, B1 = dp_statistics(Z, Y, cfg, rng=np.random.default_rng(2))
        A2, B2 = dp_statistics(Z, Y.reshape(-1, 1), cfg, rng=np.random.default_rng(2))
        assert np.allclose(A1, A2) and np.allclose(B1, B2)

    def test_sample_count_mismatch_raises(self):
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="disagree on sample count"):
            dp_statistics(rng.normal(size=(60, 5)), rng.normal(size=40),
                          PrivacyConfig(epsilon=1.0))

    def test_non_2d_states_raise(self):
        with pytest.raises(ValueError, match="2-D"):
            dp_statistics(np.zeros(10), np.zeros(10), PrivacyConfig(epsilon=1.0))


class TestGraphMetricsDegenerate:
    """``graph_metrics`` divided by zero on an empty reservoir."""

    def test_empty(self):
        m = topologies.graph_metrics(np.zeros((0, 0)))
        assert m["n_nodes"] == 0 and m["n_edges"] == 0
        assert np.isnan(m["clustering"])

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_tiny(self, n):
        m = topologies.graph_metrics(np.zeros((n, n)))
        assert m["n_nodes"] == n and m["density"] == 0.0

    def test_non_square_raises(self):
        with pytest.raises(ValueError, match="square"):
            topologies.graph_metrics(np.zeros((3, 4)))


class TestSingularRidgeSolve:
    """A rank-deficient Gram with ridge=0 raised a bare LinAlgError."""

    def test_falls_back_to_least_squares(self):
        with pytest.warns(UserWarning, match="singular"):
            W = solve_readout(np.ones((4, 4)), np.ones((4, 1)), 0.0)
        assert np.all(np.isfinite(W))
        # minimum-norm solution of a rank-1 system
        assert np.allclose(W.ravel(), 0.25)

    def test_regularised_solve_is_exact(self):
        rng = np.random.default_rng(0)
        Z = rng.normal(size=(80, 6))
        Y = rng.normal(size=(80, 2))
        A, B = Z.T @ Z, Z.T @ Y
        W = solve_readout(A, B, 1e-4)
        assert np.allclose((A + 1e-4 * np.eye(6)) @ W, B, atol=1e-8)

    def test_shape_and_sign_validation(self):
        with pytest.raises(ValueError, match="square"):
            solve_readout(np.zeros((3, 4)), np.zeros((3, 1)), 1e-6)
        with pytest.raises(ValueError, match="disagree"):
            solve_readout(np.eye(3), np.zeros((4, 1)), 1e-6)
        with pytest.raises(ValueError, match="ridge must be"):
            solve_readout(np.eye(3), np.zeros((3, 1)), -1.0)


class TestRlsIllPosedInitialisation:
    """``RLSReadout(ridge=0)`` seeded the inverse Gram with inf and never recovered."""

    def test_zero_ridge_raises(self):
        with pytest.raises(ValueError, match="ridge must be finite and > 0"):
            RLSReadout(4, 1, ridge=0.0)

    def test_negative_ridge_raises(self):
        with pytest.raises(ValueError, match="ridge must be finite and > 0"):
            RLSReadout(4, 1, ridge=-1e-3)

    @pytest.mark.parametrize("bad", [0.0, -0.5, 1.5])
    def test_invalid_forgetting_raises(self, bad):
        with pytest.raises(ValueError, match="forgetting"):
            RLSReadout(4, 1, ridge=1e-3, forgetting=bad)

    def test_valid_configuration_stays_finite(self):
        rng = np.random.default_rng(0)
        rls = RLSReadout(5, 1, ridge=1e-2, forgetting=0.99)
        rls.update_batch(rng.normal(size=(50, 5)), rng.normal(size=50))
        assert np.all(np.isfinite(rls.readout()))
        assert np.all(np.isfinite(rls.P))


class TestPromptControllerPreconditions:
    """Calling the backward pass first raised AttributeError on a private field."""

    def test_edge_client_requires_a_prompt_first(self, reservoir):
        esn = EchoStateNetwork(1, 1, reservoir, washout=5, seed=0)
        client = EdgeClient(esn, bottleneck_dim=4, embed_dim=8, seed=0)
        with pytest.raises(RuntimeError, match="call make_prompt"):
            client.apply_server_gradient(np.ones((1, 8)))

    def test_projection_requires_forward_first(self):
        proj = BottleneckProjection(k=3, d=8, seed=0)
        with pytest.raises(RuntimeError, match="forward"):
            proj.backward(np.ones((1, 8)))
        with pytest.raises(RuntimeError, match="backward"):
            proj.step(0.1)

    def test_projection_validates_shapes(self):
        proj = BottleneckProjection(k=3, d=8, seed=0)
        with pytest.raises(ValueError, match="3 entries"):
            proj.forward(np.ones(5))
        proj.forward(np.ones(3))
        with pytest.raises(ValueError, match="n_tokens\\*d"):
            proj.backward(np.ones(3))

    def test_surrogate_lm_rejects_out_of_range_target(self):
        lm = SurrogateLM(vocab_size=4, d=8, seed=0)
        with pytest.raises(ValueError, match="token id"):
            lm.loss_and_grad(np.zeros((1, 8)), 9)

    def test_surrogate_lm_rejects_wrong_embedding_width(self):
        lm = SurrogateLM(vocab_size=4, d=8, seed=0)
        with pytest.raises(ValueError, match="embedding dimensions"):
            lm.loss_and_grad(np.zeros((1, 5)), 1)

    def test_normal_round_still_works(self, reservoir):
        esn = EchoStateNetwork(1, 1, reservoir, washout=5, seed=0)
        client = EdgeClient(esn, bottleneck_dim=4, embed_dim=8, seed=0, lr=0.1)
        server = Server(SurrogateLM(vocab_size=3, d=8, seed=0))
        ctx = np.random.default_rng(0).uniform(0, 0.5, size=(30, 1))
        prompt = client.make_prompt(ctx)
        loss, grad = server.evaluate(prompt, 1)
        client.apply_server_gradient(grad)
        assert np.isfinite(loss)


class TestDeepEsnWarmStart:
    """``predict(u, x0=...)`` silently ignored x0 and returned the cold-start run."""

    def test_x0_raises_rather_than_being_ignored(self):
        Ws = [topologies.random_reservoir(20, density=0.2, rng=i) for i in range(2)]
        deep = DeepEchoStateNetwork(1, 1, Ws, washout=10, seed=0)
        u, y = datasets.narma10(200, rng=0)
        deep.fit(u, y)
        with pytest.raises(NotImplementedError, match="initial state"):
            deep.predict(u, x0=np.zeros(20))

    def test_cold_start_prediction_unaffected(self):
        Ws = [topologies.random_reservoir(20, density=0.2, rng=i) for i in range(2)]
        deep = DeepEchoStateNetwork(1, 1, Ws, washout=10, seed=0)
        u, y = datasets.narma10(200, rng=0)
        deep.fit(u, y)
        assert np.all(np.isfinite(deep.predict(u)))


class TestSecureAggregationDisclosure:
    """A single participant got a zero mask, so the server saw its data in clear."""

    def test_single_client_warns(self):
        with pytest.warns(UserWarning, match="single client cannot hide"):
            masks = zero_sum_masks(1, (3, 3), rng=0)
        assert np.allclose(masks[0], 0.0)

    def test_two_clients_are_masked(self):
        masks = zero_sum_masks(2, (3, 3), rng=0, scale=2.0)
        assert np.allclose(sum(masks), 0.0, atol=1e-12)
        assert all(np.linalg.norm(m) > 0 for m in masks)

    def test_mismatched_shapes_raise(self):
        with pytest.raises(ValueError, match="share one shape"):
            secure_sum([np.zeros((2, 2)), np.zeros((3, 3))], rng=0)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="no arrays"):
            secure_sum([], rng=0)


# ─────────────────────────────────────────────────────────────────────────────
#  Parameter validation that used to pass silently
# ─────────────────────────────────────────────────────────────────────────────
class TestHyperparameterValidation:

    def test_negative_spectral_radius_raises(self, reservoir):
        with pytest.raises(ValueError, match="non-negative"):
            EchoStateNetwork(1, 1, reservoir, spectral_radius=-0.9)

    def test_zero_spectral_radius_is_allowed(self, reservoir):
        esn = EchoStateNetwork(1, 1, reservoir, spectral_radius=0.0)
        assert np.allclose(np.asarray(esn.W), 0.0)

    @pytest.mark.parametrize("kwargs", [
        dict(n_inputs=0), dict(n_outputs=0), dict(washout=-1),
        dict(ridge=-1e-6), dict(spectral_radius=np.inf), dict(bias=np.nan),
    ])
    def test_invalid_construction_raises(self, reservoir, kwargs):
        base = dict(n_inputs=1, n_outputs=1, reservoir=reservoir)
        base.update(kwargs)
        with pytest.raises(ValueError):
            EchoStateNetwork(**base)

    @pytest.mark.parametrize("bad", [0.0, -0.3])
    def test_non_positive_leaking_rate_raises(self, reservoir, bad):
        with pytest.raises(ValueError, match="leaking_rate must be > 0"):
            EchoStateNetwork(1, 1, reservoir, leaking_rate=bad)

    def test_wrong_length_leaking_rate_array_raises(self, reservoir):
        with pytest.raises(ValueError, match="one per reservoir node"):
            EchoStateNetwork(1, 1, reservoir, leaking_rate=np.ones(7))

    def test_per_node_leaking_rate_of_right_length_works(self, reservoir):
        a = topologies.leaking_rates(reservoir.shape[0], "uniform", rng=0)
        esn = EchoStateNetwork(1, 1, reservoir, leaking_rate=a)
        assert esn._hetero_leak

    def test_target_length_mismatch_raises(self, reservoir):
        esn = EchoStateNetwork(1, 1, reservoir, washout=10, seed=0)
        with pytest.raises(ValueError, match="disagree on length"):
            esn.fit(np.zeros((100, 1)), np.zeros((60, 1)))

    def test_bad_x0_raises(self, reservoir):
        esn = EchoStateNetwork(1, 1, reservoir, seed=0)
        with pytest.raises(ValueError, match="x0 must have shape"):
            esn.harvest(np.zeros((20, 1)), x0=np.zeros(3))

    def test_bad_readout_shape_raises(self, reservoir):
        esn = EchoStateNetwork(1, 1, reservoir, seed=0)
        with pytest.raises(ValueError, match="W_out must have shape"):
            esn.set_readout(np.zeros((4, 1)))

    def test_bad_input_weights_shape_raises(self, reservoir):
        with pytest.raises(ValueError, match="input_weights must have shape"):
            EchoStateNetwork(1, 1, reservoir, input_weights=np.zeros((4, 4)))


class TestTopologyValidation:

    @pytest.mark.parametrize("kwargs,match", [
        (dict(n=0), "n must be >= 1"),
        (dict(n=10, density=1.5), "density"),
        (dict(n=10, density=-0.1), "density"),
    ])
    def test_random_reservoir(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            topologies.random_reservoir(**kwargs)

    @pytest.mark.parametrize("k", [0, 10, 20])
    def test_small_world_neighbour_bounds(self, k):
        with pytest.raises(ValueError, match="1 <= k < n"):
            topologies.small_world_reservoir(10, k=k, rng=0)

    @pytest.mark.parametrize("m", [0, 10, 15])
    def test_scale_free_attachment_bounds(self, m):
        with pytest.raises(ValueError, match="1 <= m < n"):
            topologies.scale_free_reservoir(10, m=m, rng=0)

    @pytest.mark.parametrize("p", [-0.1, 1.1])
    def test_small_world_rewiring_probability(self, p):
        with pytest.raises(ValueError, match="p must be in"):
            topologies.small_world_reservoir(20, k=4, p=p, rng=0)

    @pytest.mark.parametrize("kwargs,match", [
        (dict(low=0.0), "low must be finite and > 0"),
        (dict(low=-1.0), "low must be finite and > 0"),
        (dict(high=0.0), "high must be finite and > 0"),
        (dict(low=0.9, high=0.2), "low <= high"),
        (dict(n_layers=0), "n_layers"),
    ])
    def test_leaking_rates(self, kwargs, match):
        base = dict(n=20, kind="uniform", low=0.1, high=0.9, rng=0)
        base.update(kwargs)
        with pytest.raises(ValueError, match=match):
            topologies.leaking_rates(**base)

    def test_leaking_rates_unknown_kind_lists_choices(self):
        with pytest.raises(ValueError, match="log_uniform"):
            topologies.leaking_rates(10, "nope", rng=0)

    def test_log_uniform_stays_in_range(self):
        a = topologies.leaking_rates(200, "log_uniform", low=0.05, high=0.9, rng=0)
        assert a.min() >= 0.05 - 1e-12 and a.max() <= 0.9 + 1e-12

    @pytest.mark.parametrize("kwargs,match", [
        (dict(types=()), "at least one activation"),
        (dict(types=("tanh", "bogus")), "unknown activation"),
        (dict(weights=[1.0]), "one entry per type"),
        (dict(weights=[0.0, 0.0, 0.0]), "positive"),
        (dict(weights=[-1.0, 1.0, 1.0]), "non-negative"),
    ])
    def test_mixed_activations(self, kwargs, match):
        base = dict(n=20, types=("tanh", "sigmoid", "sin"), rng=0)
        base.update(kwargs)
        with pytest.raises(ValueError, match=match):
            topologies.mixed_activations(**base)

    def test_mixed_activations_respects_weights(self):
        acts = topologies.mixed_activations(2000, ("tanh", "sin"),
                                            weights=[0.9, 0.1], rng=0)
        assert 0.85 < np.mean(acts == "tanh") < 0.95


class TestDatasetValidation:

    @pytest.mark.parametrize("frac", [0.0, 1.0, -0.5, 1.5])
    def test_split_fraction_bounds(self, frac):
        u, y = datasets.narma10(100, rng=0)
        with pytest.raises(ValueError, match="train_frac"):
            datasets.split(u, y, frac)

    def test_split_length_mismatch(self):
        with pytest.raises(ValueError, match="disagree on length"):
            datasets.split(np.zeros((100, 1)), np.zeros((50, 1)), 0.7)

    def test_partition_more_clients_than_samples(self):
        u, y = datasets.narma10(10, rng=0)
        with pytest.raises(ValueError, match="empty partitions"):
            datasets.partition_iid(u, y, 20)

    def test_partition_rejects_zero_clients(self):
        u, y = datasets.narma10(50, rng=0)
        with pytest.raises(ValueError, match="n_clients"):
            datasets.partition_iid(u, y, 0)

    def test_partitions_are_contiguous_and_complete(self):
        u, y = datasets.narma10(200, rng=0)
        parts = datasets.partition_iid(u, y, 4)
        assert np.allclose(np.concatenate([p for p, _ in parts]), u)

    def test_partition_ignores_rng_deterministically(self):
        u, y = datasets.narma10(200, rng=0)
        a = datasets.partition_iid(u, y, 4, rng=1)
        b = datasets.partition_iid(u, y, 4, rng=999)
        assert all(np.allclose(x[0], z[0]) for x, z in zip(a, b))

    def test_narma_rejects_bad_length(self):
        with pytest.raises(ValueError, match="n must be >= 1"):
            datasets.narma10(0, rng=0)

    def test_from_array_warns_on_dropped_values(self):
        with pytest.warns(UserWarning, match="non-finite"):
            datasets.from_array([1.0, np.nan, 3.0, 4.0], normalize=False)

    def test_from_array_too_short(self):
        with pytest.raises(ValueError, match="at least 2 finite"):
            datasets.from_array([1.0], normalize=False)
        with pytest.raises(ValueError, match="at least 3 finite"):
            datasets.from_array([1.0, 2.0], predict="change", normalize=False)

    def test_from_array_unknown_mode(self):
        with pytest.raises(ValueError, match="'next' or 'change'"):
            datasets.from_array([1.0, 2.0, 3.0], predict="bogus")

    def test_load_csv_missing_column_names_options(self, tmp_path):
        p = tmp_path / "s.csv"
        p.write_text("date,value\n2020-01-01,1\n2020-01-02,2\n2020-01-03,3\n",
                     encoding="utf-8")
        with pytest.raises(KeyError, match="available columns"):
            datasets.load_csv(p, column="nope")

    def test_load_csv_out_of_range_index(self, tmp_path):
        p = tmp_path / "s.csv"
        p.write_text("a,b\n1,2\n3,4\n5,6\n", encoding="utf-8")
        with pytest.raises(IndexError, match="out of range"):
            datasets.load_csv(p, column=9)


class TestStreamingValidation:

    def test_readout_dim_mismatch_raises(self):
        sr = StreamingRidge(6, 1, ridge=1e-4)
        with pytest.raises(ValueError, match=r"shape \(n_samples, 6\)"):
            sr.update(np.zeros((10, 4)), np.zeros((10, 1)))

    def test_sample_count_mismatch_raises(self):
        sr = StreamingRidge(6, 1, ridge=1e-4)
        with pytest.raises(ValueError, match="disagree on sample count"):
            sr.update(np.zeros((10, 6)), np.zeros((7, 1)))

    def test_output_width_mismatch_raises(self):
        sr = StreamingRidge(6, 2, ridge=1e-4)
        with pytest.raises(ValueError, match=r"shape \(n_samples, 2\)"):
            sr.update(np.zeros((10, 6)), np.zeros((10, 3)))

    def test_merge_shape_mismatch_raises(self):
        a, b = StreamingRidge(6, 1), StreamingRidge(5, 1)
        with pytest.raises(ValueError, match="different shape"):
            a.merge(b)

    def test_flat_targets_accepted(self):
        sr = StreamingRidge(4, 1, ridge=1e-4)
        sr.update(np.ones((5, 4)), np.ones(5))
        assert sr.n_seen == 5

    def test_rls_rejects_wrong_record_width(self):
        rls = RLSReadout(4, 1, ridge=1e-3)
        with pytest.raises(ValueError, match="4 entries"):
            rls.update(np.ones(3), np.ones(1))


class TestPrivacyConfigValidation:

    @pytest.mark.parametrize("kwargs,match", [
        (dict(epsilon=0.0), "epsilon"),
        (dict(epsilon=-1.0), "epsilon"),
        (dict(epsilon=np.inf), "epsilon"),
        (dict(epsilon=1.0, delta=0.0), "delta"),
        (dict(epsilon=1.0, delta=1.0), "delta"),
        (dict(epsilon=1.0, clip_state=0.0), "clip_state"),
        (dict(epsilon=1.0, clip_target=-2.0), "clip_target"),
    ])
    def test_invalid_budget_raises_at_construction(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            PrivacyConfig(**kwargs)

    def test_valid_budget_constructs(self):
        cfg = PrivacyConfig(epsilon=0.5, delta=1e-6, clip_state=2.0, clip_target=1.0)
        assert cfg.sensitivity() > 0


class TestClassificationValidation:

    def test_labels_out_of_range_raise(self):
        with pytest.raises(ValueError, match=r"\[0, 2\]"):
            clf.one_hot([0, 1, 5], 3)

    def test_negative_labels_raise(self):
        with pytest.raises(ValueError, match=r"\[0, 2\]"):
            clf.one_hot([-1, 0], 3)

    def test_non_integer_labels_raise(self):
        with pytest.raises(ValueError, match="integer class indices"):
            clf.one_hot([0.5, 1.5], 3)

    def test_integral_float_labels_accepted(self):
        assert clf.one_hot(np.array([0.0, 2.0]), 3).sum() == 2

    def test_label_count_mismatch_raises(self, esn):
        X = [np.zeros((30, 1)) for _ in range(4)]
        with pytest.raises(ValueError, match="4 sequences but 3 labels"):
            clf.class_statistics(esn, X, [0, 1, 2], 3)

    def test_empty_sequence_list_raises(self, esn):
        with pytest.raises(ValueError, match="no sequences"):
            clf.reservoir_features(esn, [])

    def test_washout_longer_than_sequence_raises(self, esn):
        X = [np.zeros((10, 1))]
        with pytest.raises(ValueError, match="too short for washout"):
            clf.reservoir_features(esn, X, washout=20)

    def test_explicit_washout_changes_features(self, esn):
        X = [np.sin(np.arange(60.0)).reshape(-1, 1)]
        f0 = clf.reservoir_features(esn, X, washout=0)
        f10 = clf.reservoir_features(esn, X, washout=10)
        assert not np.allclose(f0, f10)

    def test_esn_washout_is_not_silently_inherited(self, reservoir):
        """Short classification sequences must not be wiped out by a long washout."""
        esn = EchoStateNetwork(1, 3, reservoir, washout=500, seed=0)
        F = clf.reservoir_features(esn, [np.zeros((20, 1))])
        assert F.shape == (1, esn.readout_dim)

    def test_bad_readout_shape_raises(self, esn):
        X = [np.zeros((30, 1))]
        with pytest.raises(ValueError, match="W_out must have shape"):
            clf.predict_labels(esn, X, np.zeros((3, 3)))
