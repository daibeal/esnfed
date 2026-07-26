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


def _check_dims(readout_dim: int, n_outputs: int) -> None:
    if readout_dim < 1:
        raise ValueError(f"readout_dim must be >= 1, got {readout_dim}")
    if n_outputs < 1:
        raise ValueError(f"n_outputs must be >= 1, got {n_outputs}")


def _as_batch(Z, Y, readout_dim: int, n_outputs: int):
    """Validate a ``(Z, Y)`` batch against the accumulator's dimensions.

    Reshaping ``Y`` blindly to fit ``Z`` used to hide genuine mismatches (a
    wrong-length target block would be silently folded into the wrong number of
    outputs), so the layout is checked instead.
    """
    Z = np.asarray(Z, dtype=np.float64)
    if Z.ndim == 1:
        Z = Z.reshape(-1, readout_dim) if Z.size != readout_dim else Z.reshape(1, -1)
    if Z.ndim != 2 or Z.shape[1] != readout_dim:
        raise ValueError(
            f"Z must have shape (n_samples, {readout_dim}), got {Z.shape}"
        )
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1 or (Y.ndim == 2 and 1 in Y.shape and n_outputs == 1):
        Y = Y.reshape(-1, n_outputs)
    if Y.ndim != 2 or Y.shape[1] != n_outputs:
        raise ValueError(
            f"Y must have shape (n_samples, {n_outputs}), got {Y.shape}"
        )
    if Y.shape[0] != Z.shape[0]:
        raise ValueError(
            f"Z and Y disagree on sample count: {Z.shape[0]} vs {Y.shape[0]}"
        )
    return Z, Y


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
        _check_dims(self.readout_dim, self.n_outputs)
        if self.ridge < 0:
            raise ValueError(f"ridge must be >= 0, got {self.ridge}")
        self.A = np.zeros((self.readout_dim, self.readout_dim))
        self.B = np.zeros((self.readout_dim, self.n_outputs))

    def update(self, Z, Y) -> StreamingRidge:
        """Accumulate a new batch of extended states ``Z`` and targets ``Y``."""
        Z, Y = _as_batch(Z, Y, self.readout_dim, self.n_outputs)
        Ak, Bk = ridge_statistics(Z, Y)
        self.A += Ak
        self.B += Bk
        self.n_seen += Z.shape[0]
        return self

    def merge(self, other: StreamingRidge) -> StreamingRidge:
        """Add another accumulator's statistics (exactly the federated sum)."""
        if (self.readout_dim, self.n_outputs) != (other.readout_dim, other.n_outputs):
            raise ValueError(
                f"cannot merge accumulators of different shape: "
                f"({self.readout_dim}, {self.n_outputs}) vs "
                f"({other.readout_dim}, {other.n_outputs})"
            )
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
        _check_dims(self.readout_dim, self.n_outputs)
        # P is initialised as I / ridge, so a zero ridge yields an infinite
        # (then NaN) inverse Gram and every subsequent update is garbage.
        if not np.isfinite(self.ridge) or self.ridge <= 0:
            raise ValueError(
                f"ridge must be finite and > 0 for RLS (it seeds the inverse Gram "
                f"as I/ridge), got {self.ridge}"
            )
        if not 0.0 < self.forgetting <= 1.0:
            raise ValueError(
                f"forgetting must be in (0, 1], got {self.forgetting}"
            )
        self.P = np.eye(self.readout_dim) / self.ridge
        self.W = np.zeros((self.readout_dim, self.n_outputs))

    def update(self, z, y) -> RLSReadout:
        """One rank-1 update from a single record ``(z, y)``."""
        z = np.asarray(z, dtype=np.float64).reshape(-1)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        if z.size != self.readout_dim:
            raise ValueError(
                f"z must have {self.readout_dim} entries, got {z.size}"
            )
        if y.size != self.n_outputs:
            raise ValueError(
                f"y must have {self.n_outputs} entries, got {y.size}"
            )
        lam = self.forgetting
        Pz = self.P @ z
        denom = lam + float(z @ Pz)
        k = Pz / denom
        err = y - self.W.T @ z
        self.W = self.W + np.outer(k, err)
        self.P = (self.P - np.outer(k, Pz)) / lam
        self.n_seen += 1
        return self

    def update_batch(self, Z, Y) -> RLSReadout:
        """Apply :meth:`update` to each row of ``(Z, Y)`` in order."""
        Z, Y = _as_batch(Z, Y, self.readout_dim, self.n_outputs)
        for z, y in zip(Z, Y):
            self.update(z, y)
        return self

    def readout(self) -> np.ndarray:
        return self.W
