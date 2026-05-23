"""Tests for federated strategies.

The key correctness property is that exact federated ridge over a shared
reservoir is *numerically identical* to centralized (pooled) training -- the
foundation of the privacy-preserving baseline.
"""
import numpy as np
import pytest

from esnfed import EchoStateNetwork, datasets, federated, metrics, topologies


@pytest.fixture
def setup():
    rng = np.random.default_rng(2024)
    u, y = datasets.narma10(2400, rng=rng)
    utr, ytr, ute, yte = datasets.split(u, y, 0.7)
    W = topologies.random_reservoir(150, density=0.05, rng=rng)
    parts = datasets.partition_iid(utr, ytr, 5, rng=rng)
    kw = dict(spectral_radius=0.9, leaking_rate=1.0, ridge=1e-6, washout=80)
    clients, ref = federated.make_shared_clients(W, parts, input_seed=0, esn_kwargs=kw)
    return clients, ref, ute, yte, kw


def test_federated_ridge_equals_centralized(setup):
    clients, ref, ute, yte, kw = setup
    # Centralized over the *same* clients/partitions.
    W_central = federated.train_centralized(ref, clients).W_out
    W_fed = federated.federated_ridge(clients, ref)
    assert np.allclose(W_central, W_fed, atol=1e-9)


def test_federated_beats_local_average(setup):
    clients, ref, ute, yte, kw = setup
    washout = ref.washout

    # Federated (collaborative) readout.
    W_fed = federated.federated_ridge(clients, ref)
    Z_test = ref.harvest(ute)[washout:]
    fed_err = metrics.nrmse(yte[washout:], Z_test @ W_fed)

    # Average error of purely local models (each trained on its own slice).
    federated.train_local(clients)
    local_errs = []
    for c in clients:
        pred = c.esn.predict(ute)
        local_errs.append(metrics.nrmse(yte[washout:], pred[washout:]))
    # Collaboration should not be worse than the mean local model.
    assert fed_err <= np.mean(local_errs) + 1e-6


def test_fedavg_converges(setup):
    clients, ref, ute, yte, kw = setup
    _, history = federated.fedavg(
        clients, ref, ute, yte, rounds=40, local_epochs=5, lr=0.8
    )
    # Error should decrease overall across rounds.
    assert history[-1] < history[0]
    assert np.all(np.isfinite(history))


def test_fedavg_improves_but_lags_exact_ridge(setup):
    """Iterative FedAvg makes clear progress but the ill-conditioned reservoir
    readout means plain gradient descent stays well above the closed-form optimum
    -- the motivation for the exact federated-ridge approach."""
    clients, ref, ute, yte, kw = setup
    washout = ref.washout
    W_fed = federated.federated_ridge(clients, ref)
    Z_test = ref.harvest(ute)[washout:]
    exact_err = metrics.nrmse(yte[washout:], Z_test @ W_fed)
    _, history = federated.fedavg(
        clients, ref, ute, yte, rounds=150, local_epochs=10, lr=0.9
    )
    # FedAvg clearly beats the trivial mean predictor (NRMSE = 1) ...
    assert history[-1] < 0.9
    # ... improves monotonically in the long run ...
    assert history[-1] < history[10]
    # ... yet does not reach the exact optimum (slow on an ill-conditioned readout).
    assert history[-1] > exact_err


def test_ensemble_not_worse_than_mean_member(setup):
    clients, ref, ute, yte, kw = setup
    rng = np.random.default_rng(99)
    # Build genuinely heterogeneous clients (different topologies).
    parts = [(c.u, c.y) for c in clients]
    kinds = ["random", "small_world", "scale_free", "ring", "random"]
    reservoirs = [
        topologies.make_reservoir(k, 150, rng=rng) for k in kinds
    ]
    het = federated.make_heterogeneous_clients(reservoirs, parts, esn_kwargs=kw)
    federated.train_local(het)
    washout = het[0].esn.washout

    member_errs = [
        metrics.nrmse(yte[washout:], c.esn.predict(ute)[washout:]) for c in het
    ]
    ens_pred = federated.ensemble_predict(het, ute)
    ens_err = metrics.nrmse(yte[washout:], ens_pred[washout:])
    # The ensemble should beat the average member (ensembles reduce variance).
    assert ens_err <= np.mean(member_errs) + 1e-9


def test_structural_alignment_monotone_heterogeneity(setup):
    clients, ref, ute, yte, kw = setup
    rng = np.random.default_rng(5)
    parts = [(c.u, c.y) for c in clients]
    reservoirs = [topologies.random_reservoir(150, density=0.05, rng=rng) for _ in parts]
    target = topologies.random_reservoir(150, density=0.05, rng=rng)
    records = federated.structural_alignment(
        reservoirs, target, parts, ute, yte,
        alphas=np.linspace(0, 1, 6), esn_kwargs=kw,
    )
    hetero = [r["heterogeneity"] for r in records]
    # Heterogeneity must fall monotonically to ~0 as alpha -> 1.
    assert hetero[0] > hetero[-1]
    assert hetero[-1] == pytest.approx(0.0, abs=1e-9)
