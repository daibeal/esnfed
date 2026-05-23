"""Interoperability with ReservoirPy.

Use `ReservoirPy <https://reservoirpy.readthedocs.io>`_ to *design* reservoirs
(its strength: rich node API, hyper-parameter search) and ``esnfed`` to
*federate* them. These adapters lift the reservoir and input weights out of a
ReservoirPy ``Reservoir`` node and wrap them in an :class:`esnfed.EchoStateNetwork`,
so a reservoir tuned in ReservoirPy can be dropped straight into the federated
strategies of :mod:`esnfed.federated`.

ReservoirPy is an *optional* dependency::

    pip install "esnfed[reservoirpy]"
"""
from __future__ import annotations

import numpy as np

from .esn import EchoStateNetwork, _spectral_radius


def _to_dense(matrix):
    """Return a dense float array from a (possibly sparse) ReservoirPy weight."""
    if matrix is None:
        return None
    if hasattr(matrix, "toarray"):
        return np.asarray(matrix.toarray(), dtype=float)
    return np.asarray(matrix, dtype=float)


def _ensure_initialized(reservoir, n_inputs: int) -> None:
    """ReservoirPy creates weights lazily; force initialisation if needed."""
    if not getattr(reservoir, "is_initialized", False):
        reservoir.initialize(np.zeros((1, n_inputs)))


def reservoir_matrix(reservoir, n_inputs: int = 1) -> np.ndarray:
    """Extract the dense recurrent weight matrix ``W`` from a ReservoirPy reservoir."""
    _ensure_initialized(reservoir, n_inputs)
    return _to_dense(reservoir.W)


def input_matrix(reservoir, n_inputs: int = 1) -> np.ndarray:
    """Build an esnfed input matrix ``[bias | Win]`` from a ReservoirPy reservoir.

    esnfed packs the bias into column 0 of the input matrix, whereas ReservoirPy
    keeps a separate bias vector; this helper reconciles the two conventions.
    """
    _ensure_initialized(reservoir, n_inputs)
    win = _to_dense(reservoir.Win)  # (units, input_dim)
    n_units = win.shape[0]
    bias = _to_dense(getattr(reservoir, "bias", None))
    if bias is None or bias.size == 0:
        bias_col = np.zeros((n_units, 1))
    elif bias.size == n_units:
        bias_col = bias.reshape(n_units, 1)
    elif bias.size == 1:
        # Scalar bias shared across units.
        bias_col = np.full((n_units, 1), float(bias.ravel()[0]))
    else:
        bias_col = np.zeros((n_units, 1))
    return np.hstack([bias_col, win])


def to_esn(
    reservoir,
    *,
    n_inputs: int = 1,
    n_outputs: int = 1,
    use_input_weights: bool = True,
    spectral_radius: float | None = None,
    **esn_kwargs,
) -> EchoStateNetwork:
    """Wrap a ReservoirPy ``Reservoir`` as an :class:`esnfed.EchoStateNetwork`.

    By default the reservoir's own spectral radius and leaking rate are preserved
    (so the dynamics are unchanged) and its input weights and bias are reused.
    Pass ``spectral_radius`` or other ESN keyword arguments to override.

    Parameters
    ----------
    reservoir
        A ``reservoirpy.nodes.Reservoir`` instance.
    n_inputs, n_outputs
        Task dimensions (used to initialise the reservoir if needed).
    use_input_weights
        If true, reuse ReservoirPy's input weights and bias; otherwise let
        esnfed draw fresh random input weights.
    """
    _ensure_initialized(reservoir, n_inputs)
    W = reservoir_matrix(reservoir, n_inputs)

    # Preserve the reservoir's actual spectral radius unless told otherwise.
    if spectral_radius is None:
        spectral_radius = _spectral_radius(W)
    # Preserve the leaking rate if the caller did not set one.
    if "leaking_rate" not in esn_kwargs and hasattr(reservoir, "lr"):
        esn_kwargs["leaking_rate"] = float(np.asarray(reservoir.lr).ravel()[0])

    if use_input_weights:
        win = input_matrix(reservoir, n_inputs)
        # Column 0 already holds ReservoirPy's bias vector, so use bias=1.0.
        esn_kwargs.setdefault("bias", 1.0)
        return EchoStateNetwork(
            n_inputs, n_outputs, W,
            spectral_radius=spectral_radius, input_weights=win, **esn_kwargs,
        )
    return EchoStateNetwork(
        n_inputs, n_outputs, W, spectral_radius=spectral_radius, **esn_kwargs,
    )
