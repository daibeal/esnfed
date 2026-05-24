# Privacy & streaming

Exact federated ridge ([Federated strategies](federated.md)) already keeps raw
samples on the device — clients exchange only the ridge **sufficient statistics**
`A = Zᵀ Z` and `B = Zᵀ Y`. This page adds three things that build directly on that
additive design:

- **Differential privacy** — a formal `(ε, δ)` guarantee on what each client releases.
- **Secure aggregation** — the server learns only the *sum*, never an individual client.
- **Streaming / continual learning** — accumulate `A, B` incrementally instead of re-harvesting.

All three are NumPy-only.

## Differential privacy

Summed second moments can still leak information about individual records. The
[`privacy`](../api.md#esnfedprivacy) module adds the **Gaussian mechanism**: each
client clips its per-record contribution (bounding the sensitivity) and adds
calibrated Gaussian noise to `(A, B)` before they are aggregated.

```python
from esnfed import datasets, federated, topologies
from esnfed.privacy import PrivacyConfig

W = topologies.random_reservoir(200, density=0.1, rng=0)
parts = datasets.partition_iid(*datasets.narma10(4000, rng=0)[:2], n_clients=10)
clients, ref = federated.make_shared_clients(W, parts,
                                             esn_kwargs=dict(ridge=1e-6, washout=100))

# Exact (no privacy noise) — the upper bound.
W_exact = federated.federated_ridge(clients, ref)

# (ε, δ)-differentially private readout.
cfg = PrivacyConfig(epsilon=1.0, delta=1e-5, clip_state=5.0, clip_target=1.0, seed=0)
W_dp = federated.federated_ridge_dp(clients, ref, cfg)
```

`clip_state` (`C_z`) and `clip_target` (`C_y`) bound the per-record L2 norms, which
fixes the mechanism's sensitivity `C_z·√(C_z² + C_y²)`; the noise std follows the
classic bound `σ = sensitivity·√(2 ln(1.25/δ)) / ε`.

!!! warning "Privacy costs accuracy"
    Unlike `federated_ridge`, the DP variant is **not exact** — clipping and noise
    are the price of the guarantee. Smaller `ε` (stronger privacy) and tighter
    clips mean more error. Sweep `ε` to see the privacy–utility trade-off, and set
    the clips near the typical record norm so most records are not clipped.

!!! note
    `gaussian_sigma` uses the **analytic Gaussian mechanism** (Balle & Wang, 2018)
    by default — valid for any `ε > 0` and never looser than the classic bound;
    pass `method="classic"` for the textbook `ε ≤ 1` formula. The noise is
    zero-mean, so averaging many private releases converges to the (clipped) exact
    statistics.

## Secure aggregation

The server only needs `Σₖ (Aₖ, Bₖ)`. With **additive masking** (Bonawitz et al.,
2017) each client adds a pairwise-cancelling random mask, so the masks cancel in
the sum but no individual client's statistics are ever exposed.

```python
W_secure = federated.federated_ridge_secure(clients, ref, seed=0)
# equals federated_ridge up to floating-point round-off (exact under the
# fixed-point/modular arithmetic of a real protocol)
```

DP and secure aggregation are complementary — *output* privacy vs *aggregation*
privacy — and can be combined (privatise, then mask).

## Streaming / continual learning

Because the readout depends on the data only through the sums `A, B`, training can
be **incremental**: accumulate the statistics as new data arrives. The result is
identical to batch training on everything seen so far.

```python
from esnfed.streaming import StreamingRidge

acc = StreamingRidge(ref.readout_dim, ref.n_outputs, ridge=1e-6)
for u_chunk, y_chunk in stream_of_chunks:          # new data over time
    Z = ref.harvest(u_chunk)[ref.washout:]
    acc.update(Z, y_chunk[ref.washout:])
W_out = acc.readout()                               # exact ridge over all chunks
```

`StreamingRidge.merge(other)` adds another accumulator's statistics — which is
*exactly* the federated sum — so each client can stream locally and the server
periodically merges and re-solves (continual federated learning).

For true per-sample online updates, [`RLSReadout`](../api.md#esnfedstreaming) does
recursive least squares (rank-1 Sherman–Morrison updates, `O(D²)` per sample);
with unit forgetting it converges to the same readout as batch ridge, and
`forgetting < 1` down-weights old samples for non-stationary streams.

```python
from esnfed.streaming import RLSReadout

rls = RLSReadout(ref.readout_dim, ref.n_outputs, ridge=1e-6)
for z, y in per_sample_stream:
    rls.update(z, y)
W_out = rls.readout()
```
