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

from dataclasses import dataclass, field

import numpy as np

# Reservoir size up to which Numba is auto-enabled (above this NumPy/BLAS wins).
_NUMBA_AUTO_MAX_N = 1000


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# Available node nonlinearities, for homogeneous or per-node (mixed) reservoirs.
ACTIVATIONS = {
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
    leaking_rate: object = 1.0
    activation: object = "tanh"
    ridge: float = 1e-6
    washout: int = 100
    bias: float = 1.0
    seed: int | None = None
    input_weights: np.ndarray | None = None
    dtype: object = np.float64
    sparse: bool = False
    use_numba: bool | None = None

    W: object = field(init=False, repr=False)
    W_in: np.ndarray = field(init=False, repr=False)
    W_out: np.ndarray | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        dt = np.dtype(self.dtype)
        res = self.reservoir
        if _is_scipy_sparse(res):
            res = res.toarray()
        W = np.asarray(res, dtype=dt).copy()
        if W.ndim != 2 or W.shape[0] != W.shape[1]:
            raise ValueError("reservoir must be a square 2-D matrix")
        self.n_reservoir = W.shape[0]

        # Rescale the reservoir to the requested spectral radius. Power iteration
        # (matvec-based) is used for the sparse path to avoid a dense O(N^3) eig.
        sr = _spectral_radius_iter(W) if self.sparse else _spectral_radius(W)
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
            a = np.broadcast_to(a.ravel(), (self.n_reservoir,)).astype(dt)
            self._a = np.ascontiguousarray(a)
            self._hetero_leak = True

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
        groups = []
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
        u = np.atleast_2d(u)
        if u.shape[1] != self.n_inputs:
            u = u.reshape(-1, self.n_inputs)
        u = np.ascontiguousarray(u, dtype=self.W_in.dtype)

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
    def fit(self, u: np.ndarray, y: np.ndarray) -> "EchoStateNetwork":
        """Train the readout by ridge regression on a single sequence."""
        y = np.atleast_2d(y)
        if y.shape[0] != (np.atleast_2d(u).reshape(-1, self.n_inputs)).shape[0]:
            y = y.reshape(-1, self.n_outputs)
        Z = self.harvest(u)
        Zw, Yw = Z[self.washout :], y[self.washout :]
        A, B = ridge_statistics(Zw, Yw)
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
        y = np.atleast_2d(y).reshape(-1, self.n_outputs)
        Z = self.harvest(u)
        return ridge_statistics(Z[self.washout :], y[self.washout :])

    def set_readout(self, W_out: np.ndarray) -> "EchoStateNetwork":
        self.W_out = np.asarray(W_out, dtype=float)
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


def _spectral_radius_iter(W, iters: int = 1000, seed: int = 0) -> float:
    """Spectral radius by power iteration (Gelfand growth ratio); matvec-based,
    so it works on dense or sparse matrices and avoids a dense O(N^3) eig."""
    n = W.shape[0]
    if n == 0:
        return 0.0
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(n)
    nrm = np.linalg.norm(v)
    if nrm == 0:
        return 0.0
    v /= nrm
    est = 0.0
    for _ in range(iters):
        w = W @ v
        nrm = float(np.linalg.norm(w))
        if nrm == 0:
            return 0.0
        v = w / nrm
        est = nrm
    return est


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
    """Solve ``(A + ridge * I) W_out = B`` for the readout weights."""
    d = A.shape[0]
    return np.linalg.solve(A + ridge * np.eye(d), B)
