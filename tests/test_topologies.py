"""Tests for reservoir topology generators."""
import numpy as np
import pytest

from esnfed import topologies


@pytest.fixture
def rng():
    return np.random.default_rng(7)


def test_random_density_in_range(rng):
    n = 300
    W = topologies.random_reservoir(n, density=0.1, rng=rng)
    actual = (W != 0).sum() / (n * (n - 1))
    assert 0.08 < actual < 0.12  # close to requested density
    assert np.all(np.diag(W) == 0)  # no self-loops


def test_ring_has_n_edges(rng):
    n = 50
    W = topologies.ring_reservoir(n, rng=rng)
    assert (W != 0).sum() == n  # exactly one outgoing edge per node


def test_scale_free_is_connected_and_hubbed(rng):
    W = topologies.scale_free_reservoir(200, m=3, rng=rng)
    degrees = (W != 0).sum(axis=0) + (W != 0).sum(axis=1)
    # Scale-free networks have a heavy-tailed degree distribution: the max degree
    # is far above the mean.
    assert degrees.max() > 3 * degrees.mean()


def test_small_world_high_clustering(rng):
    W_sw = topologies.small_world_reservoir(200, k=6, p=0.1, rng=rng)
    W_er = topologies.random_reservoir(200, density=6 / 200, rng=rng)
    m_sw = topologies.graph_metrics(W_sw)
    m_er = topologies.graph_metrics(W_er)
    # Small-world lattices cluster much more than Erdos-Renyi at equal density.
    assert m_sw["clustering"] > m_er["clustering"]


def test_registry_dispatch(rng):
    for kind in topologies.GENERATORS:
        W = topologies.make_reservoir(kind, 60, rng=rng)
        assert W.shape == (60, 60)


def test_unknown_topology_raises(rng):
    with pytest.raises(KeyError):
        topologies.make_reservoir("does_not_exist", 10, rng=rng)


def test_graph_metrics_keys(rng):
    W = topologies.small_world_reservoir(60, rng=rng)
    m = topologies.graph_metrics(W)
    assert {"n_nodes", "n_edges", "density", "clustering"} <= set(m)
