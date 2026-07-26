"""Federated learning strategies for Echo State Networks.

In an ESN only the linear readout is trained, so the federated problem reduces
to learning a shared (or combinable) readout across clients whose reservoirs may
or may not share the same structure.

Strategies implemented
-----------------------
train_centralized
    Pool all client data and fit one readout (the privacy-violating upper bound).
train_local
    Each client fits its own readout on its own data only (no collaboration).
federated_ridge
    *Exact* federated training for a shared reservoir: clients exchange the ridge
    sufficient statistics ``A = Z^T Z`` and ``B = Z^T Y``; the server sums them
    and solves once. Mathematically identical to pooled training, but no raw data
    leaves a client.
fedavg
    Iterative FedAvg (McMahan et al., 2017) on the readout for a shared
    reservoir: clients run local gradient steps and the server averages weights
    each round. Produces an accuracy-vs-rounds curve.
ensemble_predict
    For *heterogeneous* reservoirs: each client keeps its own ESN and the server
    averages their predictions (an ensemble), so no parameter averaging is needed.
structural_alignment
    Interpolates heterogeneous reservoirs toward a shared target structure; at
    full alignment the readouts become averageable and exact federated ridge
    applies. Sweeps the alignment level to expose the transition.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .esn import EchoStateNetwork, ridge_statistics, solve_readout
from .metrics import nrmse
from .privacy import PrivacyConfig, dp_statistics, gaussian_sigma, secure_sum


@dataclass
class Client:
    """A federated client: an ESN plus a local time-series partition."""

    esn: EchoStateNetwork
    u: np.ndarray
    y: np.ndarray
    _Z: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        n_u, n_y = len(self.u), len(np.atleast_1d(self.y))
        if n_u != n_y:
            raise ValueError(
                f"client inputs and targets disagree on length: {n_u} vs {n_y}"
            )
        # A partition shorter than the washout used to yield a *negative*
        # n_samples, which then became a negative FedAvg aggregation weight and
        # quietly corrupted the global model.
        if n_u <= self.esn.washout:
            raise ValueError(
                f"client holds {n_u} steps, which is not more than the ESN washout "
                f"({self.esn.washout}); it would contribute no training samples"
            )

    @property
    def n_samples(self) -> int:
        return len(self.u) - self.esn.washout

    def states(self) -> np.ndarray:
        """Harvest (and cache) post-washout extended states for local data."""
        if self._Z is None:
            self._Z = self.esn.harvest(self.u)
        return self._Z[self.esn.washout :]

    def targets(self) -> np.ndarray:
        y = np.asarray(self.y, dtype=np.float64)
        n_out = self.esn.n_outputs
        if y.ndim == 1 or (y.ndim == 2 and 1 in y.shape and n_out == 1):
            y = y.reshape(-1, n_out)
        elif y.ndim != 2 or y.shape[1] != n_out:
            raise ValueError(
                f"client targets must have shape (T, {n_out}), got {y.shape}"
            )
        return y[self.esn.washout :]

    def invalidate(self) -> None:
        """Drop the cached states (call after changing the reservoir)."""
        self._Z = None


def _require_clients(clients, what: str = "clients") -> None:
    """Reject an empty cohort, which otherwise yields an all-zero readout."""
    if not clients:
        raise ValueError(f"no {what} to aggregate")


# --------------------------------------------------------------- baselines
def train_centralized(esn: EchoStateNetwork, clients: list[Client]) -> EchoStateNetwork:
    """Fit one readout on the concatenation of all client states (upper bound)."""
    _require_clients(clients)
    A = np.zeros((esn.readout_dim, esn.readout_dim))
    B = np.zeros((esn.readout_dim, esn.n_outputs))
    for c in clients:
        Ak, Bk = ridge_statistics(c.states(), c.targets())
        A += Ak
        B += Bk
    esn.set_readout(solve_readout(A, B, esn.ridge))
    return esn


def train_local(clients: list[Client]) -> list[EchoStateNetwork]:
    """Each client fits its own readout on its own data only."""
    _require_clients(clients)
    for c in clients:
        Ak, Bk = ridge_statistics(c.states(), c.targets())
        c.esn.set_readout(solve_readout(Ak, Bk, c.esn.ridge))
    return [c.esn for c in clients]


# ------------------------------------------------------- exact federated ridge
def federated_ridge(clients: list[Client], esn: EchoStateNetwork) -> np.ndarray:
    """Exact federated readout via summed sufficient statistics (shared reservoir).

    Returns the shared ``W_out``. Clients transmit only ``A_k`` and ``B_k`` (which
    are independent of dataset size and reveal no individual samples), never raw
    data. The result equals :func:`train_centralized`.
    """
    _require_clients(clients)
    A = np.zeros((esn.readout_dim, esn.readout_dim))
    B = np.zeros((esn.readout_dim, esn.n_outputs))
    for c in clients:
        Ak, Bk = ridge_statistics(c.states(), c.targets())
        A += Ak
        B += Bk
    return solve_readout(A, B, esn.ridge)


def federated_ridge_dp(
    clients: list[Client], esn: EchoStateNetwork, cfg: PrivacyConfig
) -> np.ndarray:
    """Differentially private federated ridge (shared reservoir).

    Each client privatises its own statistics with the Gaussian mechanism (clip +
    noise, see :func:`esnfed.privacy.dp_statistics`) before they are summed and
    solved once. The readout is then :math:`(\\varepsilon, \\delta)`-DP w.r.t. every
    client's records. Independent noise is drawn per client from ``cfg.seed``.
    Unlike :func:`federated_ridge` this is *not* exact -- the clipping and noise
    are the price of the formal privacy guarantee. The noisy, ill-conditioned Gram
    is solved with a ridge augmented by the spectral scale of the injected noise,
    so the readout stays well-posed at any budget.
    """
    _require_clients(clients)
    A = np.zeros((esn.readout_dim, esn.readout_dim))
    B = np.zeros((esn.readout_dim, esn.n_outputs))
    base = np.random.default_rng(cfg.seed)
    for c in clients:
        child = np.random.default_rng(int(base.integers(0, 2**63 - 1)))
        Ak, Bk = dp_statistics(c.states(), c.targets(), cfg, rng=child)
        A += Ak
        B += Bk
    # Summed Gaussian noise has spectral scale ~ 2*sigma*sqrt(K*D) (semicircle law);
    # regularise by that much, else the readout explodes at small epsilon.
    sigma = gaussian_sigma(cfg.epsilon, cfg.delta, cfg.sensitivity())
    dp_ridge = esn.ridge + 2.0 * sigma * np.sqrt(len(clients) * esn.readout_dim)
    return solve_readout(A, B, dp_ridge)


def federated_ridge_secure(
    clients: list[Client], esn: EchoStateNetwork, *, seed: int | None = None,
    mask_scale: float = 1.0,
) -> np.ndarray:
    """Exact federated ridge via secure aggregation (additive masking).

    Clients mask their ``(A_k, B_k)`` with pairwise-cancelling noise (see
    :func:`esnfed.privacy.secure_sum`), so the server obtains only the masked sum
    and never an individual client's statistics. The masks cancel, so the solved
    readout equals :func:`federated_ridge` up to floating-point round-off
    (it is exact in the fixed-point/modular arithmetic of a real protocol).
    """
    _require_clients(clients)
    As, Bs = [], []
    for c in clients:
        Ak, Bk = ridge_statistics(c.states(), c.targets())
        As.append(Ak)
        Bs.append(Bk)
    rng = np.random.default_rng(seed)
    A = secure_sum(As, rng=rng, scale=mask_scale)
    B = secure_sum(Bs, rng=rng, scale=mask_scale)
    return solve_readout(A, B, esn.ridge)


# --------------------------------------------------------------------- FedAvg
def _local_gradient_step(
    W: np.ndarray, Z: np.ndarray, Y: np.ndarray, lr: float, ridge: float, epochs: int
) -> np.ndarray:
    """Full-batch gradient descent on ridge-regularised MSE for one client.

    ``lr`` is a *normalised* step in (0, 2): the actual step is ``lr / L`` where
    ``L`` is the local smoothness constant (largest eigenvalue of ``Z^T Z / n``
    plus the ridge term). This makes the step scale-free, so it stays stable
    regardless of reservoir size or state magnitude.
    """
    n = Z.shape[0]
    W = W.copy()
    ZtZ = Z.T @ Z
    ZtY = Z.T @ Y
    L = float(np.linalg.eigvalsh(ZtZ / n)[-1]) + ridge
    step = lr / L if L > 0 else lr
    for _ in range(epochs):
        grad = (ZtZ @ W - ZtY) / n + ridge * W
        W -= step * grad
    return W


def fedavg(
    clients: list[Client],
    esn: EchoStateNetwork,
    u_test: np.ndarray,
    y_test: np.ndarray,
    *,
    rounds: int = 30,
    local_epochs: int = 5,
    lr: float = 1.0,
) -> tuple[np.ndarray, list[float]]:
    """Iterative FedAvg on the readout for a shared reservoir.

    ``lr`` is a normalised step in (0, 2); see :func:`_local_gradient_step`.
    Returns the final ``W_out`` and the list of global test NRMSE values, one per
    communication round.
    """
    _require_clients(clients)
    if rounds < 1:
        raise ValueError(f"rounds must be >= 1, got {rounds}")
    if local_epochs < 1:
        raise ValueError(f"local_epochs must be >= 1, got {local_epochs}")
    if not 0.0 < lr < 2.0:
        raise ValueError(
            f"lr is a normalised step and must lie in (0, 2), got {lr}"
        )
    W = np.zeros((esn.readout_dim, esn.n_outputs))
    total = sum(c.n_samples for c in clients)
    if total <= 0:
        raise ValueError("clients contribute no post-washout samples")
    u_test = esn._as_inputs(u_test)
    y_test = esn._as_targets(y_test, u_test.shape[0])
    esn._check_trainable(u_test.shape[0])
    Z_test = esn.harvest(u_test)[esn.washout :]
    y_eval = y_test[esn.washout :]

    history: list[float] = []
    for _ in range(rounds):
        updates = []
        for c in clients:
            Wk = _local_gradient_step(
                W, c.states(), c.targets(), lr, esn.ridge, local_epochs
            )
            updates.append((c.n_samples, Wk))
        # Weighted average (FedAvg aggregation). Accumulated explicitly rather
        # than with sum(), whose empty-sequence result is the integer 0.
        W = np.zeros_like(W)
        for nk, Wk in updates:
            W += (nk / total) * Wk
        history.append(nrmse(y_eval, Z_test @ W))
    return W, history


# ------------------------------------------------------------------ ensemble
def ensemble_predict(
    clients: list[Client], u_test: np.ndarray, weights: np.ndarray | None = None
) -> np.ndarray:
    """Average the predictions of locally-trained, heterogeneous client ESNs.

    Each client must already have a trained readout (see :func:`train_local`).
    No parameter averaging is performed, so the clients may have completely
    different reservoir structures and input weights.
    """
    _require_clients(clients)
    preds = [c.esn.predict(u_test) for c in clients]
    P = np.stack(preds, axis=0)  # (n_clients, T, n_outputs)
    if weights is None:
        return P.mean(axis=0)
    w = _normalised_weights(weights, len(clients))
    return np.tensordot(w, P, axes=(0, 0))


def _normalised_weights(weights, n: int) -> np.ndarray:
    """Validate mixing weights and normalise them to sum to one.

    Weights that sum to zero used to divide through by zero and return an
    all-``nan`` prediction with no error raised.
    """
    w = np.asarray(weights, dtype=float).ravel()
    if w.size != n:
        raise ValueError(f"expected {n} weights, got {w.size}")
    if not np.all(np.isfinite(w)):
        raise ValueError("weights must all be finite")
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")
    total = w.sum()
    if not total > 0:
        raise ValueError("weights must sum to a positive value")
    return w / total


# -------------------------------------------------------- structural alignment
def interpolate_reservoir(
    W_local: np.ndarray, W_target: np.ndarray, alpha: float
) -> np.ndarray:
    """Blend a client reservoir toward a shared target: ``(1-a)W_local + a W*``."""
    return (1.0 - alpha) * W_local + alpha * W_target


def structural_alignment(
    local_reservoirs: list[np.ndarray],
    target_reservoir: np.ndarray,
    partitions: list[tuple[np.ndarray, np.ndarray]],
    u_test: np.ndarray,
    y_test: np.ndarray,
    *,
    alphas=np.linspace(0.0, 1.0, 11),
    esn_kwargs: dict | None = None,
    shared_input_seed: int = 0,
) -> list[dict]:
    """Sweep alignment level and report ensemble vs. parameter-averaging error.

    At each ``alpha`` every client's reservoir is blended toward the shared
    target. Two readouts are evaluated on the global test set:

    * **ensemble** -- clients keep individual readouts, predictions averaged;
    * **fedavg/ridge** -- exact federated ridge over the (now more similar)
      reservoirs, valid in the limit ``alpha = 1`` when structures coincide.

    Returns one record per ``alpha`` with both test NRMSEs and the mean pairwise
    reservoir distance (a measure of remaining heterogeneity).
    """
    esn_kwargs = dict(esn_kwargs or {})
    if not partitions:
        raise ValueError("no partitions given")
    if len(local_reservoirs) != len(partitions):
        raise ValueError(
            f"got {len(local_reservoirs)} reservoirs for {len(partitions)} "
            "partitions"
        )
    n_in = partitions[0][0].shape[1]
    n_out = partitions[0][1].shape[1]
    y_test = np.atleast_2d(y_test).reshape(-1, n_out)

    records = []
    for alpha in alphas:
        clients = []
        for W_local, (u, y) in zip(local_reservoirs, partitions):
            W = interpolate_reservoir(W_local, target_reservoir, alpha)
            esn = EchoStateNetwork(
                n_in, n_out, W, seed=shared_input_seed, **esn_kwargs
            )
            clients.append(Client(esn, u, y))

        # Heterogeneity measure: mean pairwise Frobenius distance of reservoirs.
        # Densify first: a sparse=True reservoir is a CSR matrix, which np.linalg
        # cannot take a norm of.
        mats = [np.asarray(m.toarray() if hasattr(m, "toarray") else m)
                for m in (c.esn.W for c in clients)]
        dists = [
            np.linalg.norm(mats[i] - mats[j])
            for i in range(len(mats))
            for j in range(i + 1, len(mats))
        ]
        heterogeneity = float(np.mean(dists)) if dists else 0.0

        # Ensemble of locally trained readouts.
        train_local(clients)
        ens_pred = ensemble_predict(clients, u_test)
        washout = clients[0].esn.washout
        ens_err = nrmse(y_test[washout:], ens_pred[washout:])

        # Parameter aggregation (exact federated ridge over shared input weights).
        ref = clients[0].esn
        W_out = federated_ridge(clients, ref)
        Z_test = ref.harvest(u_test)[washout:]
        fed_err = nrmse(y_test[washout:], Z_test @ W_out)

        records.append(
            {
                "alpha": float(alpha),
                "heterogeneity": heterogeneity,
                "ensemble_nrmse": ens_err,
                "fedavg_nrmse": fed_err,
            }
        )
    return records


# ----------------------------------------------------------------- utilities
def make_shared_clients(
    reservoir: np.ndarray,
    partitions: list[tuple[np.ndarray, np.ndarray]],
    *,
    input_seed: int = 0,
    esn_kwargs: dict | None = None,
) -> tuple[list[Client], EchoStateNetwork]:
    """Build clients that all share one reservoir and input weights (homogeneous)."""
    esn_kwargs = dict(esn_kwargs or {})
    if not partitions:
        raise ValueError("no partitions given")
    n_in = partitions[0][0].shape[1]
    n_out = partitions[0][1].shape[1]
    clients = [
        Client(EchoStateNetwork(n_in, n_out, reservoir, seed=input_seed,
                                **esn_kwargs), u, y)
        for (u, y) in partitions
    ]
    reference = EchoStateNetwork(n_in, n_out, reservoir, seed=input_seed, **esn_kwargs)
    return clients, reference


def make_heterogeneous_clients(
    reservoirs: list[np.ndarray],
    partitions: list[tuple[np.ndarray, np.ndarray]],
    *,
    esn_kwargs: dict | None = None,
) -> list[Client]:
    """Build clients each with its own reservoir and input weights (heterogeneous)."""
    esn_kwargs = dict(esn_kwargs or {})
    if not partitions:
        raise ValueError("no partitions given")
    if len(reservoirs) != len(partitions):
        raise ValueError(
            f"got {len(reservoirs)} reservoirs for {len(partitions)} partitions"
        )
    n_in = partitions[0][0].shape[1]
    n_out = partitions[0][1].shape[1]
    clients = []
    for i, (W, (u, y)) in enumerate(zip(reservoirs, partitions)):
        esn = EchoStateNetwork(n_in, n_out, W, seed=1000 + i, **esn_kwargs)
        clients.append(Client(esn, u, y))
    return clients


# ----------------------------------------------- FedResPrompt aggregation
def federated_prompt_average(prompt_clients, weights=None):
    """FedAvg of FedResPrompt controllers (the prompt-specific counterpart of
    :func:`federated_ridge`'s statistics sharing).

    Averages each client's readout ``W_out`` and bottleneck projection ``P`` and
    broadcasts the means back, so the edge devices converge on a shared prompt
    controller while keeping their data local. Clients are duck-typed: each must
    expose ``W_out`` and ``projection.P`` (see
    :class:`esnfed.llm_orchestration.EdgeClient`).
    """
    n = len(prompt_clients)
    if n == 0:
        raise ValueError("no clients to aggregate")
    w = np.ones(n) / n if weights is None else _normalised_weights(weights, n)
    shapes = {c.W_out.shape for c in prompt_clients}
    if len(shapes) > 1:
        raise ValueError(f"clients disagree on W_out shape: {sorted(shapes)}")
    shapes = {c.projection.P.shape for c in prompt_clients}
    if len(shapes) > 1:
        raise ValueError(f"clients disagree on projection shape: {sorted(shapes)}")
    W_mean = sum(wi * c.W_out for wi, c in zip(w, prompt_clients))
    P_mean = sum(wi * c.projection.P for wi, c in zip(w, prompt_clients))
    for c in prompt_clients:
        c.W_out = W_mean.copy()
        c.projection.P = P_mean.copy()
    return W_mean, P_mean
