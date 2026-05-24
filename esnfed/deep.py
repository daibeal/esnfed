"""Hierarchical (deep) Echo State Networks.

A deep ESN stacks several reservoir layers: the *states* of layer ``l`` are the
input of layer ``l+1`` (Gallicchio, Micheli & Pedrelli, 2017). Giving each layer
its own properties --- size, spectral radius, leaking rate / time-scale,
topology or node nonlinearity --- builds a hierarchy of progressively slower,
more abstract dynamics, which markedly increases memory and nonlinear capacity
over a single-layer reservoir of the same total size.

Only a single linear readout is trained, on the concatenation of *all* layers'
states, so training stays a convex ridge regression and the exact federated
scheme of :mod:`esnfed.federated` applies unchanged (a :class:`DeepEchoStateNetwork`
is a drop-in for :class:`~esnfed.esn.EchoStateNetwork` in a federated
:class:`~esnfed.federated.Client`).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .esn import EchoStateNetwork, ridge_statistics, solve_readout


def _per_layer(value, n_layers, i):
    """Resolve a possibly-per-layer hyper-parameter for layer ``i``."""
    if isinstance(value, (list, tuple)) and not isinstance(value, str):
        return value[i % len(value)]
    return value


@dataclass
class DeepEchoStateNetwork:
    """A stack of reservoir layers with a single ridge readout over all states.

    Parameters
    ----------
    n_inputs, n_outputs
        Input and output dimensionality.
    reservoirs
        List of square reservoir matrices, one per layer (e.g. from
        :mod:`esnfed.topologies`); ``len(reservoirs)`` is the depth.
    spectral_radius, leaking_rate, activation, input_scaling
        Per-layer hyper-parameters. Each may be a single value (shared by all
        layers) or a list with one entry per layer. Notably ``leaking_rate`` can
        decrease with depth to create progressively slower time-scales, and may
        itself be a per-node array (heterogeneous within a layer).
    ridge, washout, bias, seed
        As in :class:`~esnfed.esn.EchoStateNetwork`.
    """

    n_inputs: int
    n_outputs: int
    reservoirs: list
    spectral_radius: object = 0.9
    leaking_rate: object = 0.7
    activation: object = "tanh"
    input_scaling: object = 1.0
    ridge: float = 1e-6
    washout: int = 100
    bias: float = 1.0
    seed: int | None = None

    layers: list = field(init=False, repr=False)
    W_out: np.ndarray | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.reservoirs:
            raise ValueError("need at least one reservoir layer")
        n_layers = len(self.reservoirs)
        self.layers = []
        in_dim = self.n_inputs
        for i, W in enumerate(self.reservoirs):
            layer = EchoStateNetwork(
                n_inputs=in_dim, n_outputs=self.n_outputs, reservoir=W,
                spectral_radius=_per_layer(self.spectral_radius, n_layers, i),
                leaking_rate=_per_layer(self.leaking_rate, n_layers, i),
                activation=_per_layer(self.activation, n_layers, i),
                input_scaling=_per_layer(self.input_scaling, n_layers, i),
                ridge=self.ridge, washout=0, bias=self.bias,
                seed=None if self.seed is None else self.seed + i,
            )
            self.layers.append(layer)
            in_dim = layer.n_reservoir
        self.n_reservoir = sum(l.n_reservoir for l in self.layers)

    # ------------------------------------------------------------------ states
    def harvest(self, u: np.ndarray, x0=None) -> np.ndarray:
        """Run the stack and return ``[bias, u(t), x^(1)(t), ..., x^(L)(t)]``."""
        u = np.atleast_2d(u).reshape(-1, self.n_inputs)
        T = u.shape[0]
        signal = u
        states = []
        for layer in self.layers:
            Z = layer.harvest(signal)
            X = Z[:, 1 + layer.n_inputs:]   # reservoir states of this layer
            states.append(X)
            signal = X                      # feed states to the next layer
        Z = np.empty((T, self.readout_dim), dtype=states[0].dtype)
        Z[:, 0] = self.bias
        Z[:, 1:1 + self.n_inputs] = u
        col = 1 + self.n_inputs
        for X in states:
            Z[:, col:col + X.shape[1]] = X
            col += X.shape[1]
        return Z

    # ----------------------------------------------------------------- training
    def fit(self, u: np.ndarray, y: np.ndarray) -> "DeepEchoStateNetwork":
        y = np.atleast_2d(y).reshape(-1, self.n_outputs)
        Z = self.harvest(u)
        A, B = ridge_statistics(Z[self.washout:], y[self.washout:])
        self.W_out = solve_readout(A, B, self.ridge)
        return self

    def predict(self, u: np.ndarray, x0=None) -> np.ndarray:
        if self.W_out is None:
            raise RuntimeError("readout not trained; call fit() first")
        return self.harvest(u) @ self.W_out

    # ----------------------------------------------------- federated primitives
    def local_statistics(self, u: np.ndarray, y: np.ndarray):
        y = np.atleast_2d(y).reshape(-1, self.n_outputs)
        Z = self.harvest(u)
        return ridge_statistics(Z[self.washout:], y[self.washout:])

    def set_readout(self, W_out: np.ndarray) -> "DeepEchoStateNetwork":
        self.W_out = np.asarray(W_out, dtype=float)
        return self

    @property
    def readout_dim(self) -> int:
        return 1 + self.n_inputs + self.n_reservoir
