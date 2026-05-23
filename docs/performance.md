# Performance

The hot path is the reservoir **harvest** — the per-timestep state update. For a
dense reservoir this is dominated by the matrix–vector product `W @ x`, which
NumPy already runs on optimized BLAS, so the default is fast. Three *optional*
accelerators help in different regimes, and **all keep the public API and the
results unchanged** (each has a transparent NumPy fallback):

| Accelerator | How to enable | Best for |
|-------------|---------------|----------|
| **Numba** JIT | `pip install "esnfed[fast]"` (auto) | small/medium reservoirs |
| **Sparse** reservoir | `EchoStateNetwork(..., sparse=True)` | large, low-density reservoirs |
| **float32** | `EchoStateNetwork(..., dtype=np.float32)` | large reservoirs (memory-bound) |

## What actually helps where

Honest measurements (one CPU, 3000 timesteps, density 0.1, harvest time in ms —
lower is better), reproducible with `python experiments/exp9_performance.py`
(needs `esnfed[fast]`). Use them as a guide to the *regimes*, not exact figures:

| Reservoir N | NumPy f64 | Numba f64 | NumPy f32 | Sparse |
|------------:|----------:|----------:|----------:|-------:|
| 50   | 50  | **4**  (12×) | 31 | 42 |
| 200  | 81  | **21** (3.8×) | 41 | 62 |
| 800  | 411 | 340 | 322 | **261** (1.6×) |
| 2000 | 3422 | 3982 (slower) | **1776** (1.9×) | 2808 |

The takeaways:

- **Numba is a big win for *small* reservoirs** (12× at N=50, 3.8× at N=200),
  where Python per-step overhead dominates. For very large reservoirs the BLAS
  matvec dominates and Numba does *not* help (and can be slower), so the library
  **auto-enables Numba only up to N = 1000**; override with `use_numba=True/False`.
- **Sparse reservoirs win for large, low-density N** — the matvec becomes
  O(edges) instead of O(N²).
- **float32 wins for large reservoirs** (memory-bound), ~1.9× at N = 2000.

!!! note "Numerical stability with float32"
    Only the reservoir and states are stored in float32 (the large O(T·N²) part);
    the small, ill-conditioned ridge solve is always done in float64, so float32
    stays accurate.

## Recommendations

```python
import numpy as np
from esnfed import EchoStateNetwork, topologies

# Small/medium reservoir: just install esnfed[fast] — Numba kicks in automatically.
esn = EchoStateNetwork(1, 1, topologies.random_reservoir(200, rng=0))

# Large reservoir: go sparse and/or float32.
W = topologies.random_reservoir(2000, density=0.05, rng=0)
esn = EchoStateNetwork(1, 1, W, sparse=True, dtype=np.float32)
```

A few more notes:

- The **pure-NumPy default is unchanged** — none of these are required, and the
  library still installs with only NumPy + NetworkX.
- In the browser ([live demo](https://daibeal.github.io/esnfed/live/lab/index.html?path=quickstart.ipynb)),
  Numba isn't available under Pyodide; the NumPy fallback is used automatically.
- Numba pays a one-off JIT compilation cost (~1 s) on the first harvest.
