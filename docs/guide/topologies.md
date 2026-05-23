# Reservoir topologies

The reservoir's connectivity is a directed graph, and its structure shapes the
dynamics. `esnfed.topologies` provides four models, and graph descriptors to
characterise them — the central source of *structural heterogeneity* in the
federated experiments.

![Reservoir topology graph](../images/reservoir.png){ width="420" }

```python
from esnfed import topologies

W1 = topologies.random_reservoir(200, density=0.1, rng=0)      # Erdős–Rényi
W2 = topologies.small_world_reservoir(200, k=6, p=0.1, rng=0)  # Watts–Strogatz
W3 = topologies.scale_free_reservoir(200, m=3, rng=0)          # Barabási–Albert
W4 = topologies.ring_reservoir(200, rng=0)                     # simple cycle

# or by name
W = topologies.make_reservoir("small_world", 200, rng=0, k=6, p=0.1)
```

| Topology | Character |
|----------|-----------|
| `random` (Erdős–Rényi) | each directed edge present with probability `density` |
| `small_world` (Watts–Strogatz) | ring lattice + random rewiring; high clustering, short paths |
| `scale_free` (Barabási–Albert) | preferential attachment; power-law degree, hubs |
| `ring` | deterministic uni-directional cycle; minimal complexity, competitive |

## Graph descriptors

```python
m = topologies.graph_metrics(W2)
# {'n_nodes', 'n_edges', 'density', 'mean_degree', 'clustering', 'avg_path_length'}
```

!!! info "Finding from the research"
    On NARMA-10 the four topologies are statistically close — even the minimal
    *ring* is competitive — while on chaotic Mackey-Glass the densely connected
    Erdős–Rényi reservoir is clearly best. Topology matters in a *task-dependent*
    way.

Visualise any reservoir with [`viz.plot_reservoir`](viz.md).
