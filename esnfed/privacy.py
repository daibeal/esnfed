"""Privacy-preserving aggregation for exact federated ridge.

Exact federated ridge (:mod:`esnfed.federated`) already keeps raw samples on the
device: clients exchange only the sufficient statistics ``A = Z^T Z`` and
``B = Z^T Y``. Those summed second moments still encode information about
individual records, so this module hardens the exchange in two complementary
ways, both NumPy-only:

* **Differential privacy** (:func:`dp_statistics`) -- *output* privacy. Each
  client clips its per-record contribution and adds calibrated Gaussian noise to
  ``(A, B)``, giving a formal :math:`(\\varepsilon, \\delta)` guarantee on the
  statistics it releases (Dwork & Roth, 2014). Even the released statistics
  cannot be used to single out a record.
* **Secure aggregation** (:func:`secure_sum`) -- *aggregation* privacy. Clients
  add pairwise-cancelling random masks, so the server learns only the sum
  ``sum_k (A_k, B_k)`` and never any individual client's statistics
  (Bonawitz et al., 2017).

The two compose: a client can privatise its statistics and then mask them. See
:func:`esnfed.federated.federated_ridge_dp` and
:func:`esnfed.federated.federated_ridge_secure` for the client-level wrappers.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np

from .esn import ridge_statistics


def _as_rng(rng) -> np.random.Generator:
    return rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)


_SQRT2 = math.sqrt(2.0)
_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)


def _std_normal_cdf(t: float) -> float:
    return 0.5 * (1.0 + math.erf(t / _SQRT2))


def _log_std_normal_sf(t: float) -> float:
    """``log Phi(-t)`` for ``t >= 0``, without underflowing for large ``t``.

    Needed because the analytic Gaussian mechanism evaluates
    ``exp(epsilon) * Phi(-t)``: for a large budget the factor overflows while the
    tail underflows, so the product must be formed in log space (the direct
    version raised ``OverflowError: math range error``).
    """
    tail = 0.5 * math.erfc(t / _SQRT2)
    if tail > 0.0:
        return math.log(tail)
    # Asymptotic expansion of the Gaussian tail: Phi(-t) ~ phi(t) / t.
    return -0.5 * t * t - math.log(t) - _LOG_SQRT_2PI


def gaussian_sigma(epsilon: float, delta: float, sensitivity: float,
                   *, method: str = "analytic") -> float:
    """Noise std dev for the Gaussian mechanism under :math:`(\\varepsilon,\\delta)`-DP.

    ``method="analytic"`` (default) uses the **analytic Gaussian mechanism**
    (Balle & Wang, 2018): the smallest ``sigma`` for which the mechanism is
    :math:`(\\varepsilon, \\delta)`-DP, found by a short bisection. It is valid for
    *any* ``epsilon > 0`` and is never looser than the classic bound.

    ``method="classic"`` uses the textbook bound (Dwork & Roth, 2014, App. A),
    :math:`\\sigma = \\Delta_2 \\sqrt{2\\ln(1.25/\\delta)} / \\varepsilon`, which is only
    valid for ``epsilon <= 1`` (a warning is issued above that).
    """
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must be in (0, 1)")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be > 0")
    if sensitivity < 0.0:
        raise ValueError("sensitivity must be >= 0")
    if sensitivity == 0.0:
        return 0.0

    if method == "classic":
        if epsilon > 1.0:
            warnings.warn(
                "the classic Gaussian-mechanism bound assumes epsilon <= 1; use "
                "method='analytic' for a guarantee at larger epsilon",
                stacklevel=2,
            )
        return sensitivity * math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon

    if method != "analytic":
        raise ValueError("method must be 'analytic' or 'classic'")

    # Analytic Gaussian mechanism. With the scale-free ratio s = sigma/Delta, the
    # mechanism is (epsilon, delta)-DP iff B(s) <= delta, where B is decreasing:
    #   B(s) = Phi(1/(2s) - eps*s) - e^eps * Phi(-1/(2s) - eps*s).
    # The second term is evaluated as exp(eps + log Phi(-t)) so that a large
    # epsilon cannot overflow before multiplying an underflowed tail.
    def B(s: float) -> float:
        t = 1.0 / (2.0 * s) + epsilon * s          # > 0 for s > 0
        log_term = epsilon + _log_std_normal_sf(t)
        second = math.exp(log_term) if log_term < 700.0 else math.inf
        return _std_normal_cdf(1.0 / (2.0 * s) - epsilon * s) - second

    lo, hi = 1e-9, 1.0
    for _ in range(200):                            # bracket: B is decreasing in s
        if B(hi) <= delta:
            break
        hi *= 2.0
    else:  # pragma: no cover - unreachable for finite epsilon/delta
        raise RuntimeError(
            f"could not bracket sigma for epsilon={epsilon}, delta={delta}"
        )
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if B(mid) > delta:
            lo = mid
        else:
            hi = mid
    return sensitivity * hi


def clip_rows(M: np.ndarray, max_norm: float) -> np.ndarray:
    """Scale each row of ``M`` so its L2 norm is at most ``max_norm``."""
    if max_norm <= 0.0:
        raise ValueError("max_norm must be > 0")
    M = np.asarray(M, dtype=np.float64)
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    return M * np.minimum(1.0, max_norm / np.maximum(norms, 1e-12))


@dataclass
class PrivacyConfig:
    """:math:`(\\varepsilon, \\delta)`-DP configuration for releasing ``(A, B)``.

    ``clip_state`` (:math:`C_z`) and ``clip_target`` (:math:`C_y`) bound the
    per-record L2 norms of the extended state and the target; this fixes the
    mechanism's sensitivity. Smaller clips mean less noise but more bias.
    """

    epsilon: float
    delta: float = 1e-5
    clip_state: float = 1.0
    clip_target: float = 1.0
    seed: int | None = None

    def __post_init__(self) -> None:
        # Validate here rather than deep inside gaussian_sigma, so a bad budget is
        # reported where it was configured.
        if not np.isfinite(self.epsilon) or self.epsilon <= 0.0:
            raise ValueError(f"epsilon must be finite and > 0, got {self.epsilon}")
        if not 0.0 < self.delta < 1.0:
            raise ValueError(f"delta must be in (0, 1), got {self.delta}")
        if not np.isfinite(self.clip_state) or self.clip_state <= 0.0:
            raise ValueError(
                f"clip_state must be finite and > 0, got {self.clip_state}"
            )
        if not np.isfinite(self.clip_target) or self.clip_target <= 0.0:
            raise ValueError(
                f"clip_target must be finite and > 0, got {self.clip_target}"
            )

    def sensitivity(self) -> float:
        """Joint L2 (Frobenius) sensitivity of ``(A, B)`` to one record.

        Adding/removing one record changes ``A`` by ``z z^T`` and ``B`` by
        ``z y^T``; with ``||z|| <= C_z`` and ``||y|| <= C_y`` their Frobenius norms
        are ``C_z^2`` and ``C_z C_y``, so the joint sensitivity of the released
        pair is ``C_z * sqrt(C_z^2 + C_y^2)``.
        """
        cz, cy = self.clip_state, self.clip_target
        return cz * math.sqrt(cz * cz + cy * cy)


def dp_statistics(Z, Y, cfg: PrivacyConfig, rng=None):
    """Differentially private sufficient statistics ``(A, B)`` for one client.

    Clips each post-washout record to the norms in ``cfg``, forms ``A, B`` from
    the clipped data, and adds Gaussian noise calibrated by :func:`gaussian_sigma`
    to every entry (the Gaussian mechanism). ``A`` is symmetrised afterwards
    (post-processing, which preserves DP). The result is
    :math:`(\\varepsilon, \\delta)`-DP w.r.t. the client's records; summing the
    private statistics across clients and solving yields a private federated
    readout. Because of clipping and noise this is *not* exact (unlike
    :func:`esnfed.federated.federated_ridge`) -- it trades accuracy for privacy.
    """
    rng = _as_rng(cfg.seed if rng is None else rng)
    Z = np.asarray(Z, dtype=np.float64)
    if Z.ndim != 2:
        raise ValueError(f"Z must be a 2-D (n_samples, d) matrix, got {Z.shape}")
    Y = np.asarray(Y, dtype=np.float64)
    # ``np.atleast_2d`` turns a 1-D target vector into a single *row*, which then
    # clips the whole client's targets as one record and makes Z^T Y fail; treat a
    # flat vector as one target per sample instead.
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    if Y.ndim != 2:
        raise ValueError(f"Y must be 1-D or 2-D, got {Y.ndim}-D")
    if Y.shape[0] != Z.shape[0]:
        raise ValueError(
            f"Z and Y disagree on sample count: {Z.shape[0]} vs {Y.shape[0]}"
        )
    Zc = clip_rows(Z, cfg.clip_state)
    Yc = clip_rows(Y, cfg.clip_target)
    A, B = ridge_statistics(Zc, Yc)
    sigma = gaussian_sigma(cfg.epsilon, cfg.delta, cfg.sensitivity())
    A = A + rng.normal(0.0, sigma, size=A.shape)
    A = 0.5 * (A + A.T)  # restore symmetry (post-processing of the DP output)
    B = B + rng.normal(0.0, sigma, size=B.shape)
    return A, B


def zero_sum_masks(n: int, shape, rng=None, scale: float = 1.0) -> list[np.ndarray]:
    """Return ``n`` mask arrays of ``shape`` that sum to (numerically) zero.

    Pairwise masking (Bonawitz et al., 2017): for each pair ``(i, j)`` a shared
    random mask is added by client ``i`` and subtracted by client ``j``, so every
    client's contribution is hidden yet all masks cancel on summation.

    At least two clients are required: pairwise masks are built *between* clients,
    so a single participant necessarily gets a zero mask and its statistics reach
    the server in the clear.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if n == 1:
        warnings.warn(
            "secure aggregation with a single client cannot hide anything: the "
            "pairwise mask is empty, so the server sees that client's statistics "
            "verbatim. Aggregate over >= 2 clients.",
            stacklevel=2,
        )
    if scale < 0:
        raise ValueError(f"scale must be >= 0, got {scale}")
    rng = _as_rng(rng)
    masks = [np.zeros(shape, dtype=np.float64) for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            r = rng.normal(0.0, scale, size=shape)
            masks[i] += r
            masks[j] -= r
    return masks


def secure_sum(arrays, rng=None, scale: float = 1.0) -> np.ndarray:
    """Sum per-client arrays without revealing any individual one.

    Each client adds a pairwise-cancelling mask before sending; the masks cancel
    in the sum, so the server recovers ``sum_k arrays[k]`` (exactly in
    fixed-point/modular arithmetic; here up to floating-point round-off) while
    never seeing an unmasked client array.
    """
    arrays = [np.asarray(a, dtype=np.float64) for a in arrays]
    if not arrays:
        raise ValueError("no arrays to aggregate")
    shapes = {a.shape for a in arrays}
    if len(shapes) > 1:
        raise ValueError(
            f"all client arrays must share one shape; got {sorted(shapes)}"
        )
    masks = zero_sum_masks(len(arrays), arrays[0].shape, rng=rng, scale=scale)
    total = np.zeros_like(arrays[0])
    for a, m in zip(arrays, masks):
        total += a + m  # the server only ever adds masked contributions
    return total
