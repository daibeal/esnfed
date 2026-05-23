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


def random_reservoir(n: int, density: float = 0.1, rng=None) -> np.ndarray:
    """Erdos-Renyi reservoir: each directed edge present with prob. ``density``."""
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
    rng = _as_rng(rng)
    W = np.zeros((n, n))
    signs = rng.choice([-1.0, 1.0], size=n)
    for i in range(n):
        W[(i + 1) % n, i] = weight * signs[i]
    return W


# Registry used by experiment scripts and tests.
GENERATORS = {
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
    """
    adj = (W != 0).astype(int)
    g = nx.from_numpy_array(adj, create_using=nx.DiGraph)
    ug = g.to_undirected()
    n = adj.shape[0]
    n_edges = int(adj.sum())
    degrees = np.array([d for _, d in ug.degree()])
    try:
        avg_path = nx.average_shortest_path_length(ug) if nx.is_connected(ug) else float("nan")
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
