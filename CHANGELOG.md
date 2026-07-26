# Changelog

## 1.7.1 — packaging metadata

* **Fixed the links in the PyPI long description.** The README linked to
  `CHANGELOG.md` and `LICENSE` relatively. That is correct on GitHub, but PyPI
  renders the README at `pypi.org`, where a relative target resolves against
  `pypi.org` and 404s. Both are now absolute. Because the long description is
  baked into the uploaded artifact and a PyPI version can never be re-uploaded,
  fixing this required a new release.
* Declared `Changelog` and `Playground` in `[project.urls]`, so the changelog is
  reachable from the PyPI sidebar rather than only from the rendered README.
* Added tests that fail on any relative link or image in the README, on a
  non-absolute project URL, and when no `Changelog` URL is declared.

No library code changed; 1.7.0 and 1.7.1 are functionally identical.

## 1.7.0 — correctness review

A full review of the library. The theme is **silent wrongness**: several
code paths returned plausible-looking numbers instead of failing, which is the
worst failure mode for a research library because the results look publishable.

### Fixed — results were silently wrong

* **`metrics`: shape broadcasting.** Comparing a `(T,)` target against a `(T, 1)`
  prediction broadcast into a `(T, T)` difference matrix. A model with a true
  NRMSE of 0.034 was reported as **1.41** — i.e. "excellent" read as "useless".
  Shapes are now aligned when unambiguous (`(T,)` / `(T, 1)` / `(1, T)`) and
  rejected otherwise. Added `metrics.mae`.
* **`esn`: sparse reservoirs got the wrong spectral radius.** `sparse=True`
  rescaled `W` using single-vector power iteration, which does not converge when
  the dominant eigenvalue is a complex conjugate pair — the normal case for a
  random reservoir. Asking for `spectral_radius=0.9` produced 0.9335. The
  estimator is now exact for reservoirs up to 2000 nodes, uses ARPACK with `k=2`
  above that, and falls back to a Gelfand-formula estimate (~0.5% error, versus
  ~4% for the old one) when SciPy is absent.
* **`esn`: transposed / mis-shaped inputs were reshaped, not rejected.** A
  `(n_inputs, T)` array was silently `reshape`-d, which *reinterleaves* the
  samples — the reservoir then ran on scrambled data. A `(50, 4)` array fed to a
  1-input ESN silently became 200 timesteps. Only unambiguous layouts are now
  accepted; a transposed array gets an error that says so.
* **`esn`: `washout >= len(u)` produced an all-zero readout.** Training on too
  short a sequence left no post-washout samples, and the ridge solve returned
  zeros — a model that predicts 0 for ever, with no error anywhere. Now raises.
* **`datasets.narma10`: diverged to `inf` for ~1 seed in 100.** The NARMA-10
  recursion is only conditionally stable; a diverging draw poisons training and
  turns every metric into `nan`. Divergence is detected and the draw retried
  (with a warning). Stable seeds are bit-identical to before.
* **`federated.Client`: negative sample counts.** A partition shorter than the
  washout gave `n_samples < 0`, which became a *negative* FedAvg aggregation
  weight. Now rejected at construction.
* **`federated.ensemble_predict`: zero weights returned all-`nan`.** Weight
  vectors are validated (length, finite, non-negative, positive sum).
* **`privacy.gaussian_sigma`: `OverflowError` for large epsilon.** `exp(epsilon)`
  overflowed before multiplying an underflowed Gaussian tail. The term is now
  formed in log space; sigma is stable and monotone from `eps=0.01` to `1e5`.
* **`streaming.RLSReadout(ridge=0)`: silently produced `inf`/`nan`.** `P` is
  seeded as `I / ridge`, so a zero ridge poisoned every subsequent update.
  `ridge > 0` and `forgetting in (0, 1]` are now enforced.

### Fixed — crashes on legitimate input

* `viz.plot_*` raised `ValueError: setting an array element with a sequence` for
  any `sparse=True` ESN; sparse reservoirs are now densified for plotting.
* `privacy.dp_statistics` crashed on a 1-D target vector (`np.atleast_2d` made it
  a single row), inconsistent with every other entry point.
* `topologies.graph_metrics` raised `ZeroDivisionError` on an empty reservoir.
* `topologies.leaking_rates(low=0, kind="log_uniform")` produced zero leaking
  rates (frozen neurons) via `log(0)`; `low > high` raised an opaque NumPy error.
* `llm_orchestration`: calling `apply_server_gradient` before `make_prompt` raised
  `AttributeError` on a private attribute; now a `RuntimeError` that explains.
* `esn.solve_readout` propagated a bare `LinAlgError` for a singular Gram matrix;
  it now uses a Cholesky solve and falls back to the minimum-norm least-squares
  readout with a warning.

### Changed

* `DeepEchoStateNetwork.predict(u, x0=...)` **silently ignored** `x0` and returned
  the cold-start run. It now raises `NotImplementedError` rather than lie.
* `classification.*` take an explicit `washout=0` argument. `esn.washout` was
  never applied to sequence pooling and is *deliberately* not inherited: a
  regression washout (100) exceeds a whole Japanese Vowels utterance (7–29
  frames). Existing results are unchanged.
* Hyper-parameters are validated at construction: a negative `spectral_radius`
  used to be silently replaced by its absolute value; `leaking_rate <= 0` froze
  neurons; `n_inputs`, `n_outputs`, `washout`, `ridge` and `bias` are checked.
* `datasets.partition_iid` documents that its `rng` argument is ignored (the
  contiguous split is deterministic).
* `datasets.from_array` warns when non-finite observations are dropped, since
  that closes gaps in time rather than interpolating them.
* `privacy.zero_sum_masks` warns that a single client cannot be masked — its
  pairwise mask is necessarily zero, so the server sees its statistics in clear.
* Exported `esn`, `ridge_statistics`, `solve_readout`, `mae` and `r2_score` from
  the top-level package.

### Testing

87 → 499 tests (491 passed, 8 skipped for absent optional dependencies),
coverage 83% → 90%.

* `tests/test_regressions.py` — one test per defect above.
* `tests/test_invariants.py` — the mathematical claims: exactness of federated
  aggregation (regression, classification, deep, multivariate), streaming/RLS
  equivalence, the echo state property, permutation invariance of reservoir node
  labelling, and the analytic Gaussian mechanism's `(ε, δ)` condition verified
  across a grid.
* `tests/test_activations_and_dtypes.py` — every activation, mixed reservoirs,
  chunked harvesting with `x0`, and float32 / sparse / Numba parity.
* `tests/test_adapters_and_loaders.py` — ReservoirPy adapters and the FRED /
  Japanese Vowels loaders, driven by fakes so they run offline and without the
  optional dependency (`interop` coverage 14% → 97%).
* `tests/test_api_surface.py` — `__all__` correctness, docstring coverage, and a
  guard that `import esnfed` never pulls in SciPy, Numba, matplotlib or pandas.

**Documented, not fixed:** exact federated ridge is exact *for a fixed
partition*. Because every client harvests from a zero initial state, splitting
one series across more clients adds transients and shifts the aggregate
statistics (1.5% with no washout, 0.1% with `washout=50`). Separately, the
summed statistics agree to machine precision but `W_out` only to ~1e-6, because
the ridge solve amplifies by the Gram condition number (~4e10 at `ridge=1e-6`).
Both are pinned by tests so a future change cannot quietly alter them.

### Tooling

* `.github/workflows/tests.yml`: a core-dependency job (numpy + networkx only,
  Python 3.9–3.13, Linux/macOS/Windows) that would catch a stray heavy import; an
  extras job with coverage gating; ruff + mypy; and a build job that installs the
  wheel into a clean environment and smoke-tests it.
* ruff and mypy configured and clean; pytest runs with `--strict-markers`,
  `--strict-config` and esnfed's own warnings promoted to errors.

## 1.6.2 and earlier

See the git history.
