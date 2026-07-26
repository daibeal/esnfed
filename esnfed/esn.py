"""Echo State Network (ESN).

An ESN is a recurrent neural network from the Reservoir Computing paradigm
(Jaeger, 2001). A large, fixed, randomly connected recurrent layer -- the
*reservoir* -- projects the input into a high-dimensional dynamical feature
space; only a linear *readout* is trained, here by ridge regression. Training is
therefore convex and fast, with no backpropagation through time.

State update (leaky-integrator neurons, Jaeger et al. 2007):

    x(t) = (1 - a) x(t-1) + a * tanh( W_in [b; u(t)] + W x(t-1) )

Readout:

    y(t) = W_out [b; u(t); x(t)]

where ``a`` is the leaking rate, ``b`` a bias constant, ``W`` the (spectral-radius
scaled) reservoir matrix and ``W_in`` the input matrix. ``W_out`` is the only
trained quantity.

Performance
-----------
The state-harvesting loop is the hot path. Three optional accelerations are
available and all keep the public API and results unchanged:

* **Numba** -- if installed (``pip install "esnfed[fast]"``), the dense
  ``float64`` harvest loop is JIT-compiled to native speed automatically;
  otherwise a pure-NumPy fallback is used.
* **Sparse reservoirs** (``sparse=True``) -- store ``W`` as a SciPy CSR matrix,
  turning the per-step matrix-vector product from O(N^2) into O(edges); a large
  win for big, sparse reservoirs.
* **float32** (``dtype=np.float32``) -- halves memory traffic for a modest speed
  gain on memory-bound workloads.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Union

import numpy as np
import numpy.typing as npt

# Reservoir size up to which Numba is auto-enabled (above this NumPy/BLAS wins).
_NUMBA_AUTO_MAX_N = 1000

#: A leaking rate: one value shared by every node, or one value per node.
LeakingRate = Union[float, np.ndarray]

#: A node nonlinearity: the name of a built-in (see :data:`ACTIVATIONS`), an
#: array of such names for a mixed reservoir, or any element-wise callable.
Activation = Union[str, np.ndarray, Callable[[np.ndarray], np.ndarray]]


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# Available node nonlinearities, for homogeneous or per-node (mixed) reservoirs.
ACTIVATIONS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "tanh": np.tanh,
    "sigmoid": _sigmoid,
    "relu": lambda z: np.maximum(0.0, z),
    "sin": np.sin,            # oscillator-like
    "identity": lambda z: z,
}


@dataclass
class EchoStateNetwork:
    """A leaky-integrator Echo State Network with a ridge-regression readout.

    Parameters
    ----------
    n_inputs, n_outputs
        Input and output dimensionality.
    reservoir
        Square reservoir weight matrix ``W`` (n_reservoir x n_reservoir). Usually
        produced by :mod:`esnfed.topologies`. It is rescaled internally to
        ``spectral_radius``.
    spectral_radius
        Target largest absolute eigenvalue of ``W``. Must be < 1 for the echo
        state property to hold for typical inputs (Jaeger, 2001).
    input_scaling
        Scaling applied to the random input weights.
    leaking_rate
        Leaky-integrator rate ``a`` in (0, 1]; 1.0 recovers a standard ESN. May
        be a **scalar** (homogeneous) or a per-node **array** of shape
        ``(n_reservoir,)`` for *heterogeneous leaking rates / time constants* ---
        different neurons then integrate at different speeds, giving a multi-scale
        reservoir (see :func:`esnfed.topologies.leaking_rates`).
    activation
        Node nonlinearity. ``"tanh"`` (default), ``"sigmoid"``, ``"relu"``,
        ``"sin"`` or ``"identity"``; a per-node **array** of those names for a
        *multi-type* (mixed) reservoir (see
        :func:`esnfed.topologies.mixed_activations`); or any callable applied
        element-wise.
    ridge
        Tikhonov (ridge) regularisation strength for the readout.
    washout
        Number of initial timesteps discarded when harvesting states, to remove
        the dependence on the (zero) initial state.
    bias
        Constant bias fed to the reservoir and readout.
    seed
        Seed for the input-weight RNG (the reservoir itself is supplied).
    input_weights
        Optional caller-supplied input matrix of shape ``(N, 1 + n_inputs)``
        (e.g. lifted from a ReservoirPy reservoir); if ``None``, drawn randomly.
    dtype
        Floating-point type for the reservoir and states (``np.float64`` by
        default; ``np.float32`` trades precision for memory/speed).
    sparse
        If true, store the reservoir as a SciPy CSR matrix (needs SciPy); the
        per-step matvec then costs O(edges) instead of O(N^2). Best for large,
        low-density reservoirs.
    use_numba
        ``None`` (default) uses Numba for the dense ``float64`` harvest if it is
        installed; ``True``/``False`` force it on/off.
    """

    n_inputs: int
    n_outputs: int
    reservoir: np.ndarray
    spectral_radius: float = 0.9
    input_scaling: float = 1.0
    leaking_rate: LeakingRate = 1.0
    activation: Activation = "tanh"
    ridge: float = 1e-6
    washout: int = 100
    bias: float = 1.0
    seed: int | None = None
    input_weights: np.ndarray | None = None
    dtype: npt.DTypeLike = np.float64
    sparse: bool = False
    use_numba: bool | None = None

    W: Any = field(init=False, repr=False)          # ndarray, or a SciPy CSR matrix
    W_in: np.ndarray = field(init=False, repr=False)
    W_out: np.ndarray | None = field(init=False, default=None, repr=False)

    # Resolved in __post_init__: a scalar leaking rate or one value per node, and
    # the element-wise nonlinearity (possibly a per-node dispatcher).
    _a: float | np.ndarray = field(init=False, repr=False)
    _activation_fn: Callable[[np.ndarray], np.ndarray] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        dt = np.dtype(self.dtype)
        res: Any = self.reservoir
        if _is_scipy_sparse(res):
            res = res.toarray()
        W = np.asarray(res, dtype=dt).copy()
        if W.ndim != 2 or W.shape[0] != W.shape[1]:
            raise ValueError("reservoir must be a square 2-D matrix")
        self.n_reservoir = W.shape[0]
        self._validate_hyperparameters()

        # Rescale the reservoir to the requested spectral radius. The sparse path
        # uses a Krylov/ARPACK estimate, which -- unlike plain power iteration --
        # is correct when the dominant eigenvalue is a complex conjugate pair (the
        # common case for random reservoirs).
        sr = _spectral_radius_sparse(W) if self.sparse else _spectral_radius(W)
        if sr > 0:
            W = (W * (self.spectral_radius / sr)).astype(dt)

        self._is_sparse = bool(self.sparse)
        if self._is_sparse:
            sp = _scipy_sparse()
            self.W = sp.csr_matrix(W)
        else:
            self.W = np.ascontiguousarray(W, dtype=dt)

        if self.input_weights is not None:
            W_in = np.asarray(self.input_weights, dtype=dt)
            expected = (self.n_reservoir, self.n_inputs + 1)
            if W_in.shape != expected:
                raise ValueError(
                    f"input_weights must have shape {expected}, got {W_in.shape}"
                )
            self.W_in = np.ascontiguousarray(W_in, dtype=dt)
        else:
            rng = np.random.default_rng(self.seed)
            W_in = self.input_scaling * rng.uniform(
                -1.0, 1.0, size=(self.n_reservoir, self.n_inputs + 1)
            )
            self.W_in = np.ascontiguousarray(W_in, dtype=dt)

        # Heterogeneous leaking rates / time constants: a scalar a, or a per-node
        # vector so different neurons integrate at different speeds (multi-scale).
        a = np.asarray(self.leaking_rate, dtype=dt)
        if a.ndim == 0:
            self._a = float(a)
            self._hetero_leak = False
        else:
            if a.size not in (1, self.n_reservoir):
                raise ValueError(
                    f"leaking_rate array has {a.size} entries; expected a scalar "
                    f"or one per reservoir node ({self.n_reservoir})"
                )
            a = np.broadcast_to(a.ravel(), (self.n_reservoir,)).astype(dt)
            self._a = np.ascontiguousarray(a)
            self._hetero_leak = True
        # A non-positive leaking rate freezes a neuron at its initial state, which
        # is never intended and silently kills part of the reservoir.
        if np.any(np.asarray(self._a) <= 0.0):
            raise ValueError("leaking_rate must be > 0 (a = 1 recovers a standard ESN)")

        # Multi-type node nonlinearities: one activation, a per-node array of
        # activation names, or a callable.
        self._activation_fn, self._hetero_act = self._build_activation()

        # Numba accelerates the dense float64 path only, and only for the
        # homogeneous scalar-leak / tanh case (heterogeneous reservoirs use the
        # NumPy path). It is a large win for small reservoirs but on par with
        # NumPy for large ones, so it is auto-enabled up to a size threshold.
        base = ((not self._is_sparse) and dt == np.dtype(np.float64)
                and not self._hetero_leak and not self._hetero_act)
        if self.use_numba is None:
            self._numba_enabled = base and self.n_reservoir <= _NUMBA_AUTO_MAX_N
        else:
            self._numba_enabled = base and bool(self.use_numba)

    def _validate_hyperparameters(self) -> None:
        """Reject hyper-parameters that would silently produce nonsense.

        In particular a *negative* ``spectral_radius`` used to be accepted and
        silently rescaled to its absolute value, and a negative ``washout`` would
        slice from the end of the state matrix.
        """
        if self.n_inputs < 1:
            raise ValueError(f"n_inputs must be >= 1, got {self.n_inputs}")
        if self.n_outputs < 1:
            raise ValueError(f"n_outputs must be >= 1, got {self.n_outputs}")
        if not np.isfinite(self.spectral_radius) or self.spectral_radius < 0:
            raise ValueError(
                "spectral_radius must be a finite, non-negative number "
                f"(got {self.spectral_radius}); the sign of W is not a free "
                "parameter, rescaling only changes the magnitude"
            )
        if self.washout < 0:
            raise ValueError(f"washout must be >= 0, got {self.washout}")
        if not np.isfinite(self.ridge) or self.ridge < 0:
            raise ValueError(f"ridge must be finite and >= 0, got {self.ridge}")
        if not np.isfinite(self.bias):
            raise ValueError(f"bias must be finite, got {self.bias}")

    def _as_inputs(self, u) -> np.ndarray:
        """Coerce ``u`` to a contiguous ``(T, n_inputs)`` matrix, or fail loudly.

        A 2-D array whose second axis is *not* ``n_inputs`` used to be silently
        ``reshape``-d, which for a transposed ``(n_inputs, T)`` input reinterleaves
        the samples and produces plausible-looking but wrong states. Only genuinely
        unambiguous layouts are accepted now.
        """
        u = np.asarray(u)
        n = self.n_inputs
        if u.ndim == 1 or (u.ndim == 2 and 1 in u.shape and n == 1):
            # A flat stream (or a row/column vector for a single-input ESN).
            if u.size % n:
                raise ValueError(
                    f"cannot interpret {u.size} values as a sequence of "
                    f"{n}-dimensional inputs"
                )
            u = u.reshape(-1, n)
        elif u.ndim != 2:
            raise ValueError(f"inputs must be 1-D or 2-D, got {u.ndim}-D")
        elif u.shape[1] != n:
            hint = (f"; the array looks transposed -- pass u.T to read it as "
                    f"{u.shape[1]} steps of {u.shape[0]} inputs"
                    if u.shape[0] == n else "")
            raise ValueError(
                f"inputs must have shape (T, n_inputs) with n_inputs={n}, "
                f"got {u.shape}{hint}"
            )
        return np.ascontiguousarray(u, dtype=self.W_in.dtype)

    def _as_targets(self, y, n_steps: int) -> np.ndarray:
        """Coerce ``y`` to ``(n_steps, n_outputs)``, or fail loudly."""
        y = np.asarray(y, dtype=np.float64)
        if y.ndim == 1 or (y.ndim == 2 and 1 in y.shape and self.n_outputs == 1):
            y = y.reshape(-1, self.n_outputs)
        elif y.ndim != 2:
            raise ValueError(f"targets must be 1-D or 2-D, got {y.ndim}-D")
        elif y.shape[1] != self.n_outputs:
            raise ValueError(
                f"targets must have shape (T, n_outputs) with "
                f"n_outputs={self.n_outputs}, got {y.shape}"
            )
        if y.shape[0] != n_steps:
            raise ValueError(
                f"inputs and targets disagree on length: {n_steps} input steps "
                f"vs {y.shape[0]} target steps"
            )
        return y

    def _check_trainable(self, n_steps: int) -> None:
        """Ensure there is post-washout data to fit on.

        Without this a ``washout`` at least as long as the sequence yields empty
        statistics, and the ridge solve returns an all-zero readout that predicts
        zeros for ever, with no error anywhere.
        """
        if n_steps <= self.washout:
            raise ValueError(
                f"sequence of {n_steps} steps is too short for washout="
                f"{self.washout}: no samples would remain to train on"
            )

    def _build_activation(self):
        """Return ``(fn, is_heterogeneous)`` where ``fn(pre) -> activations``."""
        act = self.activation
        if callable(act):
            return act, True
        if isinstance(act, str):
            if act not in ACTIVATIONS:
                raise ValueError(f"unknown activation {act!r}; "
                                 f"choices: {sorted(ACTIVATIONS)}")
            return ACTIVATIONS[act], act != "tanh"
        names = np.asarray(act)
        if names.shape != (self.n_reservoir,):
            raise ValueError("activation array must have length n_reservoir")
        groups: list[tuple[Callable[[np.ndarray], np.ndarray], np.ndarray]] = []
        for name in np.unique(names):
            key = str(name)
            if key not in ACTIVATIONS:
                raise ValueError(f"unknown activation {key!r}")
            groups.append((ACTIVATIONS[key], np.where(names == name)[0]))

        def mixed(pre):
            out = np.empty_like(pre)
            for fn, idx in groups:
                out[idx] = fn(pre[idx])
            return out

        return mixed, True

    # ------------------------------------------------------------------ states
    def harvest(self, u: np.ndarray, x0: np.ndarray | None = None) -> np.ndarray:
        """Run the reservoir over inputs ``u`` and return extended states.

        Parameters
        ----------
        u
            Input array of shape (T, n_inputs).
        x0
            Optional initial reservoir state (defaults to zeros).

        Returns
        -------
        Z
            Extended-state matrix of shape (T, 1 + n_inputs + n_reservoir),
            each row being ``[bias, u(t), x(t)]``.
        """
        u = self._as_inputs(u)
        if x0 is not None:
            x0 = np.asarray(x0, dtype=self.W_in.dtype).ravel()
            if x0.shape != (self.n_reservoir,):
                raise ValueError(
                    f"x0 must have shape ({self.n_reservoir},), got {x0.shape}"
                )

        if self._numba_enabled and x0 is None:
            fn = _get_numba_harvest()
            if fn is not None:
                Z = fn(self.W, self.W_in, u, float(self._a),
                       float(self.bias), int(self.n_inputs))
                self._last_state = Z[-1, 1 + self.n_inputs:] if len(Z) else \
                    np.zeros(self.n_reservoir)
                return Z

        Z = _harvest_numpy(self.W, self.W_in, u, self._a, self.bias,
                           self.n_inputs, self.n_reservoir, x0,
                           act=self._activation_fn)
        self._last_state = Z[-1, 1 + self.n_inputs:] if len(Z) else \
            np.zeros(self.n_reservoir)
        return Z

    # ----------------------------------------------------------------- training
    def fit(self, u: np.ndarray, y: np.ndarray) -> EchoStateNetwork:
        """Train the readout by ridge regression on a single sequence."""
        u = self._as_inputs(u)
        y = self._as_targets(y, u.shape[0])
        self._check_trainable(u.shape[0])
        Z = self.harvest(u)
        A, B = ridge_statistics(Z[self.washout :], y[self.washout :])
        self.W_out = solve_readout(A, B, self.ridge)
        return self

    def predict(self, u: np.ndarray, x0: np.ndarray | None = None) -> np.ndarray:
        """Predict outputs for inputs ``u`` using the trained readout."""
        if self.W_out is None:
            raise RuntimeError("ESN readout is not trained; call fit() first")
        Z = self.harvest(u, x0=x0)
        return Z @ self.W_out

    # ----------------------------------------------------- federated primitives
    def local_statistics(self, u: np.ndarray, y: np.ndarray):
        """Return the ridge sufficient statistics (A, B) for local data.

        ``A = Z^T Z`` and ``B = Z^T Y`` over the post-washout extended states.
        Summing these across clients and solving once yields *exactly* the
        readout that pooled training would produce -- the basis of exact
        federated ridge regression.
        """
        u = self._as_inputs(u)
        y = self._as_targets(y, u.shape[0])
        self._check_trainable(u.shape[0])
        Z = self.harvest(u)
        return ridge_statistics(Z[self.washout :], y[self.washout :])

    def set_readout(self, W_out: np.ndarray) -> EchoStateNetwork:
        """Install a readout (e.g. one aggregated by the federated server)."""
        W_out = np.asarray(W_out, dtype=float)
        expected = (self.readout_dim, self.n_outputs)
        if W_out.shape != expected:
            raise ValueError(
                f"W_out must have shape {expected}, got {W_out.shape}"
            )
        self.W_out = W_out
        return self

    @property
    def readout_dim(self) -> int:
        return 1 + self.n_inputs + self.n_reservoir


# --------------------------------------------------------------------- harvest
def _harvest_numpy(W, W_in, u, a, bias, n_inputs, n_reservoir, x0, act=np.tanh):
    """Pure-NumPy harvest; works for a dense ndarray or a SciPy CSR reservoir.

    ``a`` may be a scalar or a per-node array (heterogeneous leaking rates) and
    ``act`` any element-wise callable (supports multi-type node nonlinearities);
    both broadcast over the state vector.
    """
    dt = W_in.dtype
    T = u.shape[0]
    x = (np.zeros(n_reservoir, dtype=dt) if x0 is None
         else np.asarray(x0, dtype=dt).copy())
    Z = np.empty((T, 1 + n_inputs + n_reservoir), dtype=dt)
    for t in range(T):
        u_b = np.empty(1 + n_inputs, dtype=dt)
        u_b[0] = bias
        u_b[1:] = u[t]
        pre = W_in @ u_b + W @ x
        x = ((1.0 - a) * x + a * act(pre)).astype(dt, copy=False)
        Z[t, 0] = bias
        Z[t, 1 : 1 + n_inputs] = u[t]
        Z[t, 1 + n_inputs :] = x
    return Z


def _harvest_kernel(W, W_in, u, a, bias, n_inputs):
    """Vectorised dense harvest (float64), compiled by Numba when available.

    Keeps the BLAS matrix-vector products (``W @ x``) while running the timestep
    loop natively, removing Python per-step overhead. A clear win for small,
    overhead-bound reservoirs; on par with NumPy for large, BLAS-bound ones.
    """
    T = u.shape[0]
    n = W.shape[0]
    Z = np.empty((T, 1 + n_inputs + n))
    x = np.zeros(n)
    ub = np.zeros(1 + n_inputs)
    ub[0] = bias
    for t in range(T):
        for k in range(n_inputs):
            ub[1 + k] = u[t, k]
        pre = W_in @ ub + W @ x
        x = (1.0 - a) * x + a * np.tanh(pre)
        Z[t, 0] = bias
        for k in range(n_inputs):
            Z[t, 1 + k] = u[t, k]
        for i in range(n):
            Z[t, 1 + n_inputs + i] = x[i]
    return Z


_NUMBA_FN = None


def _get_numba_harvest():
    """Lazily compile (and cache) the Numba harvest; return None if unavailable."""
    global _NUMBA_FN
    if _NUMBA_FN is None:
        try:
            from numba import njit

            fn = njit(cache=False)(_harvest_kernel)
            # Eager warm-up so a compilation failure falls back to NumPy cleanly.
            fn(np.zeros((1, 1)), np.zeros((1, 2)), np.zeros((2, 1)), 1.0, 1.0, 1)
            _NUMBA_FN = fn
        except Exception:
            _NUMBA_FN = False
    return _NUMBA_FN or None


# --------------------------------------------------------------------- helpers
def _is_scipy_sparse(obj) -> bool:
    return type(obj).__module__.startswith("scipy.sparse")


def _scipy_sparse():
    try:
        import scipy.sparse as sp
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "sparse=True requires SciPy; install esnfed[fast] (or scipy)."
        ) from e
    return sp


def _spectral_radius(W: np.ndarray) -> float:
    """Largest absolute eigenvalue of a (dense) square matrix."""
    if W.shape[0] == 0:
        return 0.0
    return float(np.max(np.abs(np.linalg.eigvals(W))))


# Below this size a dense eigendecomposition is fast enough to always prefer,
# because it is exact; above it we fall back to iterative estimates.
_DENSE_EIG_MAX_N = 2000


def _spectral_radius_sparse(W) -> float:
    """Spectral radius for the ``sparse=True`` path.

    Plain power iteration is *not* usable here: a real reservoir matrix typically
    has a complex-conjugate pair as its dominant eigenvalue, and the iterates then
    rotate instead of converging, so the growth ratio oscillates around
    ``|lambda_max|`` and lands several percent off. That silently gave sparse
    reservoirs a different spectral radius from the one requested (0.9335 for a
    target of 0.9).

    Strategy, in order of preference:

    1. a dense eigendecomposition while the reservoir is small enough for it to be
       cheap -- this is *exact*, and covers the great majority of real uses;
    2. ARPACK (``scipy.sparse.linalg.eigs``) with ``k=2`` so both members of a
       complex conjugate pair fit in the requested invariant subspace;
    3. a Gelfand-formula estimate ``||W^m||^(1/m)``, which is insensitive to the
       rotation and converges to the spectral radius from below.
    """
    n = W.shape[0]
    if n == 0 or not _any_nonzero(W):
        return 0.0
    if n <= _DENSE_EIG_MAX_N:
        dense = W.toarray() if _is_scipy_sparse(W) else np.asarray(W)
        return _spectral_radius(dense.astype(np.float64))
    try:
        from scipy.sparse.linalg import eigs

        # k=2 (not 1) so a dominant complex pair is resolved rather than
        # approximated by a single real Ritz value; tol=0 asks for machine
        # precision. ARPACK requires k < n - 1.
        vals = eigs(W.astype(np.float64), k=2, which="LM",
                    ncv=min(n - 1, 40), tol=0, return_eigenvectors=False)
        return float(np.max(np.abs(vals)))
    except Exception:
        return _spectral_radius_gelfand(W)


def _any_nonzero(W) -> bool:
    """True if the (dense or sparse) matrix has a non-zero entry."""
    if _is_scipy_sparse(W):
        return W.nnz > 0 and bool(np.any(W.data))
    return bool(np.any(W))


def _spectral_radius_gelfand(W, iters: int = 200, seed: int = 0) -> float:
    """Gelfand-formula spectral radius estimate ``lim ||W^m||^(1/m)``.

    Iterating a *block* of vectors and rescaling by the geometric mean of the
    per-step growth makes the estimate robust to the rotation induced by a complex
    dominant pair, which is what breaks single-vector power iteration.
    """
    n = W.shape[0]
    rng = np.random.default_rng(seed)
    V = rng.standard_normal((n, min(4, n)))
    V /= np.linalg.norm(V, axis=0, keepdims=True)
    log_growth = 0.0
    steps = 0
    for _ in range(iters):
        V = W @ V
        nrm = float(np.linalg.norm(V))
        if nrm == 0.0:
            return 0.0
        log_growth += np.log(nrm)
        V = V / nrm
        steps += 1
        # Renormalising by the Frobenius norm of the block leaves the average
        # log-growth per step as the estimate of log(spectral radius).
    return float(np.exp(log_growth / steps)) if steps else 0.0


# Retained for backwards compatibility (the old, less accurate estimator).
def _spectral_radius_iter(W, iters: int = 1000, seed: int = 0) -> float:
    """Deprecated alias of :func:`_spectral_radius_sparse`.

    The original single-vector power iteration is inaccurate for reservoirs with a
    complex dominant eigenvalue pair; it now delegates to the accurate estimator.
    """
    return _spectral_radius_sparse(W)


def ridge_statistics(Z: np.ndarray, Y: np.ndarray):
    """Return sufficient statistics ``A = Z^T Z`` and ``B = Z^T Y``.

    The Gram matrix is accumulated in float64 even when the states are float32:
    the harvest keeps the float32 speed/memory benefit, while the (small,
    ill-conditioned) ridge solve stays numerically stable.
    """
    Z = np.asarray(Z, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    return Z.T @ Z, Z.T @ Y


def solve_readout(A: np.ndarray, B: np.ndarray, ridge: float) -> np.ndarray:
    """Solve ``(A + ridge * I) W_out = B`` for the readout weights.

    ``A`` is a Gram matrix, hence symmetric positive semi-definite, so the system
    is solved via a Cholesky factorisation of the regularised matrix. With
    ``ridge = 0`` (or a rank-deficient ``A``) that system can be singular; rather
    than propagating a bare ``LinAlgError`` we fall back to the minimum-norm
    least-squares solution, which is the natural limit of ridge regression as the
    regularisation vanishes.
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be a square matrix, got shape {A.shape}")
    if B.shape[0] != A.shape[0]:
        raise ValueError(
            f"A and B disagree: A is {A.shape}, B is {B.shape}"
        )
    if ridge < 0:
        raise ValueError(f"ridge must be >= 0, got {ridge}")
    G = A + ridge * np.eye(A.shape[0])
    try:
        # Symmetric-positive-definite solve: ~2x faster than a general LU and it
        # fails cleanly (rather than silently amplifying error) when G is singular.
        chol = np.linalg.cholesky(G)
        return np.linalg.solve(chol.T, np.linalg.solve(chol, B))
    except np.linalg.LinAlgError:
        pass
    try:
        return np.linalg.solve(G, B)
    except np.linalg.LinAlgError:
        warnings.warn(
            "the regularised Gram matrix is singular; falling back to the "
            "minimum-norm least-squares readout. Increase `ridge` for a "
            "well-posed solve.",
            stacklevel=2,
        )
        return np.linalg.lstsq(G, B, rcond=None)[0]
