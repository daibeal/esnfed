"""Reservoir topology generators.

The recurrent connectivity of an ESN reservoir can be modelled as a directed
graph. The choice of graph model shapes the reservoir dynamics and is the
central experimental variable of this project. Each generator returns a dense
``n x n`` weight matrix whose non-zero entries are drawn from a symmetric
uniform distribution; the Echo State Network rescales the matrix to the desired
spectral radius afterwards.

Topologies
----------
random        Erdos-Renyi G(n, p): each directed edge present with prob. p.
small_world   Watts-Strogatz ring lattice with random rewiring.
scale_free    Barabasi-Albert preferential attachment (degree power law).
ring          Simple deterministic uni-directional ring (delay line).

All functions accept a ``numpy.random.Generator`` (or an int seed) so that
experiments are fully reproducible.
"""
from __future__ import annotations

from typing import Callable

import networkx as nx
import numpy as np


def _as_rng(rng) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def _weight_edges(adj: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Assign uniform[-1, 1] weights to the non-zero structure of ``adj``."""
    mask = adj != 0
    W = np.zeros_like(adj, dtype=float)
    W[mask] = rng.uniform(-1.0, 1.0, size=int(mask.sum()))
    return W


def _check_n(n: int) -> int:
    n = int(n)
    if n < 1:
        raise ValueError(f"reservoir size n must be >= 1, got {n}")
    return n


def random_reservoir(n: int, density: float = 0.1, rng=None) -> np.ndarray:
    """Erdos-Renyi reservoir: each directed edge present with prob. ``density``."""
    n = _check_n(n)
    if not 0.0 <= density <= 1.0:
        raise ValueError(f"density must be in [0, 1], got {density}")
    rng = _as_rng(rng)
    mask = rng.uniform(size=(n, n)) < density
    np.fill_diagonal(mask, False)
    W = np.zeros((n, n))
    W[mask] = rng.uniform(-1.0, 1.0, size=int(mask.sum()))
    return W


def small_world_reservoir(
    n: int, k: int = 6, p: float = 0.1, rng=None
) -> np.ndarray:
    """Watts-Strogatz small-world reservoir.

    A ring lattice where each node connects to its ``k`` nearest neighbours, with
    each edge rewired with probability ``p``. Small-world reservoirs combine high
    clustering with short path lengths.
    """
    n = _check_n(n)
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"rewiring probability p must be in [0, 1], got {p}")
    if not 1 <= k < n:
        raise ValueError(
            f"the ring lattice needs 1 <= k < n neighbours, got k={k} for n={n}"
        )
    rng = _as_rng(rng)
    seed = int(rng.integers(0, 2**31 - 1))
    g = nx.watts_strogatz_graph(n, k, p, seed=seed)
    adj = nx.to_numpy_array(g)
    return _weight_edges(adj, rng)


def scale_free_reservoir(n: int, m: int = 3, rng=None) -> np.ndarray:
    """Barabasi-Albert scale-free reservoir.

    Growth with preferential attachment; each new node attaches to ``m`` existing
    nodes. Produces a power-law degree distribution with a few high-degree hubs.
    """
    n = _check_n(n)
    if not 1 <= m < n:
        raise ValueError(
            f"preferential attachment needs 1 <= m < n, got m={m} for n={n}"
        )
    rng = _as_rng(rng)
    seed = int(rng.integers(0, 2**31 - 1))
    g = nx.barabasi_albert_graph(n, m, seed=seed)
    adj = nx.to_numpy_array(g)
    return _weight_edges(adj, rng)


def ring_reservoir(n: int, weight: float = 1.0, rng=None) -> np.ndarray:
    """Deterministic uni-directional ring (a.k.a. simple cycle reservoir).

    Each node feeds the next; node ``n-1`` feeds node ``0``. Despite its
    simplicity this minimal-complexity reservoir is competitive on many tasks
    (Rodan & Tino, 2011).
    """
    n = _check_n(n)
    rng = _as_rng(rng)
    W = np.zeros((n, n))
    signs = rng.choice([-1.0, 1.0], size=n)
    for i in range(n):
        W[(i + 1) % n, i] = weight * signs[i]
    return W


# Registry used by experiment scripts and tests.
GENERATORS: dict[str, Callable[..., np.ndarray]] = {
    "random": random_reservoir,
    "small_world": small_world_reservoir,
    "scale_free": scale_free_reservoir,
    "ring": ring_reservoir,
}


def make_reservoir(kind: str, n: int, rng=None, **kwargs) -> np.ndarray:
    """Dispatch to a named generator from :data:`GENERATORS`."""
    if kind not in GENERATORS:
        raise KeyError(f"unknown topology {kind!r}; choices: {list(GENERATORS)}")
    return GENERATORS[kind](n, rng=rng, **kwargs)


def graph_metrics(W: np.ndarray) -> dict:
    """Return basic graph descriptors of a reservoir weight matrix.

    Useful to characterise structural heterogeneity across federated nodes.
    ``n_edges`` and ``density`` count *directed* edges; ``mean_degree`` and
    ``clustering`` are computed on the undirected projection, so a reciprocal
    pair contributes to the degree of both endpoints once.

    An empty matrix yields all-zero / ``nan`` descriptors rather than raising.
    """
    W = np.asarray(W)
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError(f"W must be a square 2-D matrix, got shape {W.shape}")
    adj = (W != 0).astype(int)
    n = adj.shape[0]
    if n == 0:
        # nx.average_clustering divides by the node count, so guard the empty case.
        return {"n_nodes": 0, "n_edges": 0, "density": 0.0, "mean_degree": 0.0,
                "clustering": float("nan"), "avg_path_length": float("nan")}
    g = nx.from_numpy_array(adj, create_using=nx.DiGraph)
    ug = g.to_undirected()
    n_edges = int(adj.sum())
    degrees = np.array([d for _, d in ug.degree()])
    try:
        avg_path = (nx.average_shortest_path_length(ug)
                    if nx.is_connected(ug) else float("nan"))
    except (nx.NetworkXError, nx.NetworkXPointlessConcept):
        avg_path = float("nan")
    return {
        "n_nodes": n,
        "n_edges": n_edges,
        "density": n_edges / (n * (n - 1)) if n > 1 else 0.0,
        "mean_degree": float(degrees.mean()) if degrees.size else 0.0,
        "clustering": float(nx.average_clustering(ug)),
        "avg_path_length": float(avg_path),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Heterogeneity generators (per-node leaking rates and node nonlinearities)
# ─────────────────────────────────────────────────────────────────────────────
def leaking_rates(n, kind="uniform", low=0.1, high=1.0, n_layers=3, rng=None):
    """Per-node leaking rates ``a`` for a *heterogeneous / multi-scale* reservoir.

    Different neurons integrate at different speeds, which helps tasks with mixed
    time-scales (finance, chaotic systems such as Mackey-Glass).

    Parameters
    ----------
    n
        Number of reservoir nodes.
    kind
        ``"uniform"`` (i.i.d. in ``[low, high]``), ``"log_uniform"`` (log-spaced,
        emphasising fast nodes), ``"layered"`` (``n_layers`` contiguous blocks
        with decreasing rates, high → low, i.e. fast → slow groups), or
        ``"constant"`` (all ``high``).
    low, high
        Range of leaking rates (each in ``(0, 1]``).
    n_layers
        Number of decreasing blocks for ``kind="layered"``.
    rng
        Seed or ``numpy`` generator.

    Returns
    -------
    a : ndarray of shape (n,)
        Per-node leaking rates, ready to pass as ``EchoStateNetwork(leaking_rate=a)``.
    """
    n = _check_n(n)
    # A leaking rate of 0 freezes a neuron for ever, and low > high silently
    # produced an empty/NaN range, so both are rejected up front.
    if not np.isfinite(low) or low <= 0.0:
        raise ValueError(f"low must be finite and > 0, got {low}")
    if not np.isfinite(high) or high <= 0.0:
        raise ValueError(f"high must be finite and > 0, got {high}")
    if low > high:
        raise ValueError(f"need low <= high, got low={low}, high={high}")
    if n_layers < 1:
        raise ValueError(f"n_layers must be >= 1, got {n_layers}")
    rng = _as_rng(rng)
    if kind == "uniform":
        return rng.uniform(low, high, size=n)
    if kind == "log_uniform":
        return np.exp(rng.uniform(np.log(low), np.log(high), size=n))
    if kind == "constant":
        return np.full(n, float(high))
    if kind == "layered":
        levels = np.linspace(high, low, max(1, n_layers))
        return np.concatenate([
            np.full(len(idx), levels[k])
            for k, idx in enumerate(np.array_split(np.arange(n), max(1, n_layers)))
        ])
    raise ValueError(
        f"unknown kind {kind!r}; choices: "
        "'uniform', 'log_uniform', 'layered', 'constant'"
    )


def mixed_activations(n, types=("tanh", "sigmoid", "sin"), weights=None, rng=None):
    """Assign a node nonlinearity to each reservoir node (*multi-type* reservoir).

    Mixing activation functions within one reservoir mimics biological neuronal
    diversity and broadens the basis of dynamics. Returns an array of activation
    names ready to pass as ``EchoStateNetwork(activation=...)``.

    Parameters
    ----------
    n
        Number of reservoir nodes.
    types
        Activation names to mix (see :data:`esnfed.esn.ACTIVATIONS`).
    weights
        Optional mixing proportions (defaults to uniform).
    rng
        Seed or ``numpy`` generator.
    """
    from .esn import ACTIVATIONS

    n = _check_n(n)
    rng = _as_rng(rng)
    types = list(types)
    if not types:
        raise ValueError("types must name at least one activation")
    unknown = [t for t in types if t not in ACTIVATIONS]
    if unknown:
        raise ValueError(
            f"unknown activation(s) {unknown}; choices: {sorted(ACTIVATIONS)}"
        )
    p = None
    if weights is not None:
        p = np.asarray(weights, dtype=float)
        if p.shape != (len(types),):
            raise ValueError(
                f"weights must have one entry per type ({len(types)}), got {p.shape}"
            )
        if np.any(p < 0):
            raise ValueError("weights must be non-negative")
        total = p.sum()
        if not total > 0:
            raise ValueError("weights must sum to a positive value")
        p = p / total
    return rng.choice(types, size=n, p=p)
