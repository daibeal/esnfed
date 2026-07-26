"""Error metrics for reservoir computing tasks.

All metrics compare ``y_true`` and ``y_pred`` *elementwise*, so they insist that
the two arrays describe the same samples. A shape such as ``(T,)`` against
``(T, 1)`` is accepted (both denote ``T`` samples of a single output), but genuine
disagreements are rejected rather than being resolved by NumPy broadcasting: a
``(T,)`` target against a ``(T, 1)`` prediction broadcasts to a ``(T, T)``
difference matrix and yields a silently, badly wrong error -- for a good model,
an NRMSE of 1.41 rather than 0.03.
"""
from __future__ import annotations

import numpy as np


def _align(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(y_true, y_pred)`` as arrays of identical shape, or raise.

    Trailing/leading singleton axes are squeezed away first, so ``(T,)``,
    ``(T, 1)`` and ``(1, T)`` are interchangeable; anything else must match
    exactly.
    """
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    if yt.shape != yp.shape:
        st, sp = np.squeeze(yt), np.squeeze(yp)
        if st.shape != sp.shape:
            raise ValueError(
                f"y_true and y_pred have incompatible shapes {yt.shape} and "
                f"{yp.shape}; elementwise comparison would broadcast them into a "
                "larger array and report a meaningless error"
            )
        yt, yp = st, sp
    return yt, yp


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared error."""
    yt, yp = _align(y_true, y_pred)
    if yt.size == 0:
        return float("nan")
    return float(np.mean((yt - yp) ** 2))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(mse(y_true, y_pred)))


def nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Normalised RMSE, scaled by the standard deviation of the target.

    NRMSE = 1 corresponds to a trivial predictor outputting the target mean.
    This is the standard figure of merit for NARMA and Mackey-Glass tasks.
    Returns ``nan`` for a constant target, whose standard deviation is zero.
    """
    yt, yp = _align(y_true, y_pred)
    if yt.size == 0:
        return float("nan")
    sigma = float(yt.std())
    if sigma == 0:
        return float("nan")
    return float(np.sqrt(np.mean((yt - yp) ** 2))) / sigma


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination R^2."""
    yt, yp = _align(y_true, y_pred)
    if yt.size == 0:
        return float("nan")
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    if ss_tot == 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error (less sensitive to outliers than :func:`mse`)."""
    yt, yp = _align(y_true, y_pred)
    if yt.size == 0:
        return float("nan")
    return float(np.mean(np.abs(yt - yp)))
