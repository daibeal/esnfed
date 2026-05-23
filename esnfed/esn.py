"""Echo State Network (ESN).

An ESN is a recurrent neural network from the Reservoir Computing paradigm
(Jaeger, 2001). A large, fixed, randomly connected recurrent layer -- the
*reservoir* -- projects the input into a high-dimensional dynamical feature
space; only a linear *readout* is trained, here by ridge regression. Training is
therefore convex and fast, with no backpropagation through time.

State update (leaky-integrator neurons, Jaeger et al. 2007):

    x(t) = (1 - a) x(t-1) + a * tanh( W_in [b; u(t)] + W x(t-1) )

Readout:

    y(t) = W_out [b; u(t); x(t)]

where ``a`` is the leaking rate, ``b`` a bias constant, ``W`` the (spectral-radius
scaled) reservoir matrix and ``W_in`` the input matrix. ``W_out`` is the only
trained quantity.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class EchoStateNetwork:
    """A leaky-integrator Echo State Network with a ridge-regression readout.

    Parameters
    ----------
    n_inputs, n_outputs
        Input and output dimensionality.
    reservoir
        Square reservoir weight matrix ``W`` (n_reservoir x n_reservoir). Usually
        produced by :mod:`esnfed.topologies`. It is rescaled internally to
        ``spectral_radius``.
    spectral_radius
        Target largest absolute eigenvalue of ``W``. Must be < 1 for the echo
        state property to hold for typical inputs (Jaeger, 2001).
    input_scaling
        Scaling applied to the random input weights.
    leaking_rate
        Leaky-integrator rate ``a`` in (0, 1]; 1.0 recovers a standard ESN.
    ridge
        Tikhonov (ridge) regularisation strength for the readout.
    washout
        Number of initial timesteps discarded when harvesting states, to remove
        the dependence on the (zero) initial state.
    bias
        Constant bias fed to the reservoir and readout.
    seed
        Seed for the input-weight RNG (the reservoir itself is supplied).
    """

    n_inputs: int
    n_outputs: int
    reservoir: np.ndarray
    spectral_radius: float = 0.9
    input_scaling: float = 1.0
    leaking_rate: float = 1.0
    ridge: float = 1e-6
    washout: int = 100
    bias: float = 1.0
    seed: int | None = None
    input_weights: np.ndarray | None = None

    W: np.ndarray = field(init=False, repr=False)
    W_in: np.ndarray = field(init=False, repr=False)
    W_out: np.ndarray | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        W = np.asarray(self.reservoir, dtype=float).copy()
        if W.ndim != 2 or W.shape[0] != W.shape[1]:
            raise ValueError("reservoir must be a square 2-D matrix")
        self.n_reservoir = W.shape[0]

        # Rescale the reservoir to the requested spectral radius.
        sr = _spectral_radius(W)
        if sr > 0:
            W *= self.spectral_radius / sr
        self.W = W

        if self.input_weights is not None:
            # Caller-supplied input matrix (e.g. from a ReservoirPy reservoir).
            W_in = np.asarray(self.input_weights, dtype=float)
            expected = (self.n_reservoir, self.n_inputs + 1)
            if W_in.shape != expected:
                raise ValueError(
                    f"input_weights must have shape {expected}, got {W_in.shape}"
                )
            self.W_in = W_in
        else:
            rng = np.random.default_rng(self.seed)
            # Input weights in [-input_scaling, +input_scaling]; col 0 is bias.
            self.W_in = self.input_scaling * (
                rng.uniform(-1.0, 1.0, size=(self.n_reservoir, self.n_inputs + 1))
            )

    # ------------------------------------------------------------------ states
    def harvest(self, u: np.ndarray, x0: np.ndarray | None = None) -> np.ndarray:
        """Run the reservoir over inputs ``u`` and return extended states.

        Parameters
        ----------
        u
            Input array of shape (T, n_inputs).
        x0
            Optional initial reservoir state (defaults to zeros).

        Returns
        -------
        Z
            Extended-state matrix of shape (T, 1 + n_inputs + n_reservoir),
            each row being ``[bias, u(t), x(t)]``.
        """
        u = np.atleast_2d(u)
        if u.shape[1] != self.n_inputs:
            u = u.reshape(-1, self.n_inputs)
        T = u.shape[0]
        a = self.leaking_rate

        x = np.zeros(self.n_reservoir) if x0 is None else np.asarray(x0, float).copy()
        Z = np.empty((T, 1 + self.n_inputs + self.n_reservoir))
        for t in range(T):
            u_b = np.concatenate(([self.bias], u[t]))
            pre = self.W_in @ u_b + self.W @ x
            x = (1.0 - a) * x + a * np.tanh(pre)
            Z[t, 0] = self.bias
            Z[t, 1 : 1 + self.n_inputs] = u[t]
            Z[t, 1 + self.n_inputs :] = x
        self._last_state = x
        return Z

    # ----------------------------------------------------------------- training
    def fit(self, u: np.ndarray, y: np.ndarray) -> "EchoStateNetwork":
        """Train the readout by ridge regression on a single sequence."""
        y = np.atleast_2d(y)
        if y.shape[0] != (np.atleast_2d(u).reshape(-1, self.n_inputs)).shape[0]:
            y = y.reshape(-1, self.n_outputs)
        Z = self.harvest(u)
        Zw, Yw = Z[self.washout :], y[self.washout :]
        A, B = ridge_statistics(Zw, Yw)
        self.W_out = solve_readout(A, B, self.ridge)
        return self

    def predict(self, u: np.ndarray, x0: np.ndarray | None = None) -> np.ndarray:
        """Predict outputs for inputs ``u`` using the trained readout."""
        if self.W_out is None:
            raise RuntimeError("ESN readout is not trained; call fit() first")
        Z = self.harvest(u, x0=x0)
        return Z @ self.W_out

    # ----------------------------------------------------- federated primitives
    def local_statistics(self, u: np.ndarray, y: np.ndarray):
        """Return the ridge sufficient statistics (A, B) for local data.

        ``A = Z^T Z`` and ``B = Z^T Y`` over the post-washout extended states.
        Summing these across clients and solving once yields *exactly* the
        readout that pooled training would produce -- the basis of exact
        federated ridge regression.
        """
        y = np.atleast_2d(y).reshape(-1, self.n_outputs)
        Z = self.harvest(u)
        return ridge_statistics(Z[self.washout :], y[self.washout :])

    def set_readout(self, W_out: np.ndarray) -> "EchoStateNetwork":
        self.W_out = np.asarray(W_out, dtype=float)
        return self

    @property
    def readout_dim(self) -> int:
        return 1 + self.n_inputs + self.n_reservoir


# --------------------------------------------------------------------- helpers
def _spectral_radius(W: np.ndarray) -> float:
    """Largest absolute eigenvalue of a square matrix."""
    if W.shape[0] == 0:
        return 0.0
    eigs = np.linalg.eigvals(W)
    return float(np.max(np.abs(eigs)))


def ridge_statistics(Z: np.ndarray, Y: np.ndarray):
    """Return sufficient statistics ``A = Z^T Z`` and ``B = Z^T Y``."""
    return Z.T @ Z, Z.T @ Y


def solve_readout(A: np.ndarray, B: np.ndarray, ridge: float) -> np.ndarray:
    """Solve ``(A + ridge * I) W_out = B`` for the readout weights."""
    d = A.shape[0]
    return np.linalg.solve(A + ridge * np.eye(d), B)
