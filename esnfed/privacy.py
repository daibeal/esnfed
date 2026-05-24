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


def _std_normal_cdf(t: float) -> float:
    return 0.5 * (1.0 + math.erf(t / math.sqrt(2.0)))


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
    def B(s: float) -> float:
        return (_std_normal_cdf(1.0 / (2.0 * s) - epsilon * s)
                - math.exp(epsilon) * _std_normal_cdf(-1.0 / (2.0 * s) - epsilon * s))

    lo, hi = 1e-9, 1.0
    while B(hi) > delta:
        hi *= 2.0
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
    Zc = clip_rows(Z, cfg.clip_state)
    Yc = clip_rows(np.atleast_2d(np.asarray(Y, dtype=np.float64)), cfg.clip_target)
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
    """
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
    masks = zero_sum_masks(len(arrays), arrays[0].shape, rng=rng, scale=scale)
    total = np.zeros_like(arrays[0])
    for a, m in zip(arrays, masks):
        total += a + m  # the server only ever adds masked contributions
    return total
