"""Incremental / streaming ridge for continual (federated) learning.

The ridge readout depends on the data only through the sums ``A = Z^T Z`` and
``B = Z^T Y`` (see :mod:`esnfed.federated`), so training can be made *incremental*
simply by accumulating those sums as new data arrives -- the result is identical
to batch training on all data seen so far. Two tools are provided:

* :class:`StreamingRidge` -- accumulate ``(A, B)`` over chunks and solve on
  demand. :meth:`StreamingRidge.merge` adds another accumulator's statistics,
  which is exactly the federated sum, so clients can stream locally and the server
  periodically merges and re-solves (continual federated learning).
* :class:`RLSReadout` -- recursive least squares: rank-1 updates of the readout
  and of the inverse Gram matrix (Sherman-Morrison) at ``O(D^2)`` per sample, for
  true per-sample online learning without re-solving. With unit forgetting it
  converges to the same readout as batch ridge.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .esn import ridge_statistics, solve_readout


@dataclass
class StreamingRidge:
    """Accumulate ridge sufficient statistics incrementally; solve on demand.

    Exact: after any sequence of :meth:`update` calls the readout equals batch
    ridge over all data seen so far.
    """

    readout_dim: int
    n_outputs: int
    ridge: float = 1e-6
    A: np.ndarray = field(init=False, repr=False)
    B: np.ndarray = field(init=False, repr=False)
    n_seen: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.A = np.zeros((self.readout_dim, self.readout_dim))
        self.B = np.zeros((self.readout_dim, self.n_outputs))

    def update(self, Z, Y) -> "StreamingRidge":
        """Accumulate a new batch of extended states ``Z`` and targets ``Y``."""
        Z = np.asarray(Z, dtype=np.float64)
        Y = np.atleast_2d(np.asarray(Y, dtype=np.float64))
        if Y.shape[0] != Z.shape[0]:
            Y = Y.reshape(Z.shape[0], -1)
        Ak, Bk = ridge_statistics(Z, Y)
        self.A += Ak
        self.B += Bk
        self.n_seen += Z.shape[0]
        return self

    def merge(self, other: "StreamingRidge") -> "StreamingRidge":
        """Add another accumulator's statistics (exactly the federated sum)."""
        self.A += other.A
        self.B += other.B
        self.n_seen += other.n_seen
        return self

    def readout(self) -> np.ndarray:
        """Solve ``(A + ridge I) W = B`` for the current readout."""
        return solve_readout(self.A, self.B, self.ridge)


@dataclass
class RLSReadout:
    """Recursive least squares readout (online ridge via Sherman-Morrison).

    Maintains the readout ``W`` and the inverse Gram ``P = (sum z z^T + ridge I)^{-1}``
    and updates both with each sample at ``O(D^2)`` cost. Initialised with
    ``P = I / ridge`` so the ``ridge`` term acts as the usual Tikhonov prior; with
    ``forgetting = 1.0`` the readout after processing all samples equals batch
    ridge. ``forgetting < 1`` down-weights old samples (useful for non-stationary
    streams).
    """

    readout_dim: int
    n_outputs: int
    ridge: float = 1e-6
    forgetting: float = 1.0
    P: np.ndarray = field(init=False, repr=False)
    W: np.ndarray = field(init=False, repr=False)
    n_seen: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.P = np.eye(self.readout_dim) / self.ridge
        self.W = np.zeros((self.readout_dim, self.n_outputs))

    def update(self, z, y) -> "RLSReadout":
        """One rank-1 update from a single record ``(z, y)``."""
        z = np.asarray(z, dtype=np.float64).reshape(-1)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        lam = self.forgetting
        Pz = self.P @ z
        denom = lam + float(z @ Pz)
        k = Pz / denom
        err = y - self.W.T @ z
        self.W = self.W + np.outer(k, err)
        self.P = (self.P - np.outer(k, Pz)) / lam
        self.n_seen += 1
        return self

    def update_batch(self, Z, Y) -> "RLSReadout":
        """Apply :meth:`update` to each row of ``(Z, Y)`` in order."""
        Z = np.asarray(Z, dtype=np.float64)
        Y = np.atleast_2d(np.asarray(Y, dtype=np.float64))
        if Y.shape[0] != Z.shape[0]:
            Y = Y.reshape(Z.shape[0], -1)
        for z, y in zip(Z, Y):
            self.update(z, y)
        return self

    def readout(self) -> np.ndarray:
        return self.W
