"""Benchmark sequential tasks for Echo State Networks.

These are the standard tasks used to evaluate reservoir computing systems.

NARMA-10
    A non-linear auto-regressive moving-average system with a 10-step memory
    (Atiya & Parlos, 2000). The task is to reproduce ``y`` from the random input
    ``u``; it stresses both memory and non-linearity.

Mackey-Glass
    A delay differential equation (Mackey & Glass, 1977) that is mildly chaotic
    for ``tau = 17``. The task is one-step-ahead prediction.

Lorenz
    The classic 3-variable chaotic attractor (Lorenz, 1963); we predict the
    next x-coordinate.

Real-world finance
    ``from_array`` / ``load_csv`` turn any 1-D series into a forecasting task;
    ``load_ted_spread`` loads a bundled real counterparty-risk series (the TED
    spread); ``load_fred`` downloads any series from FRED on demand. These power
    the federated counterparty-risk use case: institutions that cannot share raw
    data jointly forecast a credit/counterparty-risk signal.

All generators return ``(u, y)`` as float arrays of shape (T, 1).
"""
from __future__ import annotations

import csv
import io
import urllib.request
from pathlib import Path

import numpy as np

_DATA_DIR = Path(__file__).parent / "data"


def narma10(n: int, rng=None, warmup: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Generate a NARMA-10 input/target sequence of length ``n``."""
    rng = np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng
    total = n + warmup
    u = rng.uniform(0.0, 0.5, size=total)
    y = np.zeros(total)
    for t in range(10, total):
        y[t] = (
            0.3 * y[t - 1]
            + 0.05 * y[t - 1] * np.sum(y[t - 10 : t])
            + 1.5 * u[t - 10] * u[t - 1]
            + 0.1
        )
    u, y = u[warmup:], y[warmup:]
    return u.reshape(-1, 1), y.reshape(-1, 1)


def mackey_glass(
    n: int,
    tau: int = 17,
    beta: float = 0.2,
    gamma: float = 0.1,
    power: int = 10,
    dt: float = 1.0,
    seed: int | None = None,
    discard: int = 250,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a Mackey-Glass series; return ``(x_t, x_{t+1})`` for prediction."""
    rng = np.random.default_rng(seed)
    history_len = tau + 1
    total = n + discard + 1
    x = np.empty(total + history_len)
    x[:history_len] = 1.2 + 0.2 * (rng.random(history_len) - 0.5)
    for t in range(history_len, total + history_len):
        x_tau = x[t - tau]
        x[t] = x[t - 1] + dt * (
            beta * x_tau / (1.0 + x_tau**power) - gamma * x[t - 1]
        )
    series = x[history_len + discard :]
    u = series[:-1].reshape(-1, 1)
    y = series[1:].reshape(-1, 1)
    return u, y


def lorenz(
    n: int,
    dt: float = 0.02,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
    seed: int | None = None,
    discard: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate the Lorenz attractor; return ``(x_t, x_{t+1})`` for prediction."""
    rng = np.random.default_rng(seed)
    state = np.array([1.0, 1.0, 1.0]) + 0.01 * rng.standard_normal(3)
    total = n + discard + 1
    xs = np.empty(total)
    s = state
    for t in range(total):
        xs[t] = s[0]
        dx = sigma * (s[1] - s[0])
        dy = s[0] * (rho - s[2]) - s[1]
        dz = s[0] * s[1] - beta * s[2]
        s = s + dt * np.array([dx, dy, dz])
    series = xs[discard:]
    # Normalise to a sensible range for tanh reservoirs.
    series = (series - series.mean()) / series.std()
    return series[:-1].reshape(-1, 1), series[1:].reshape(-1, 1)


def split(
    u: np.ndarray, y: np.ndarray, train_frac: float = 0.7
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Chronological train/test split (no shuffling, to preserve dynamics)."""
    cut = int(len(u) * train_frac)
    return u[:cut], y[:cut], u[cut:], y[cut:]


def partition_iid(
    u: np.ndarray, y: np.ndarray, n_clients: int, rng=None
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split a sequence into ``n_clients`` contiguous blocks (federated clients).

    Contiguous blocks (rather than shuffled samples) keep each client's slice a
    valid time series, which is required for reservoir state harvesting.
    """
    bounds = np.linspace(0, len(u), n_clients + 1, dtype=int)
    return [(u[a:b], y[a:b]) for a, b in zip(bounds[:-1], bounds[1:])]


# ─────────────────────────────────────────────────────────────────────────────
#  Real-world data: turn any 1-D series into a forecasting task
# ─────────────────────────────────────────────────────────────────────────────
def from_array(
    series, *, predict: str = "next", normalize: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Turn a 1-D real series into a one-step-ahead forecasting task.

    Parameters
    ----------
    series
        1-D sequence of observations (list or array).
    predict
        ``"next"`` to forecast the next value (level), or ``"change"`` to
        forecast the next first difference.
    normalize
        If true, z-score the series (recommended for tanh reservoirs).

    Returns
    -------
    (u, y)
        Input/target arrays of shape (T-1, 1).
    """
    s = np.asarray(series, dtype=float).ravel()
    s = s[np.isfinite(s)]
    if normalize and s.std() > 0:
        s = (s - s.mean()) / s.std()
    if predict == "next":
        return s[:-1].reshape(-1, 1), s[1:].reshape(-1, 1)
    if predict == "change":
        d = np.diff(s)
        return d[:-1].reshape(-1, 1), d[1:].reshape(-1, 1)
    raise ValueError("predict must be 'next' or 'change'")


def _column_from_csv(text: str, column) -> np.ndarray:
    reader = csv.reader(io.StringIO(text))
    rows = [r for r in reader if r]
    header = rows[0]
    data = rows[1:]
    if column is None:
        idx = len(header) - 1  # last column by default
    elif isinstance(column, int):
        idx = column
    else:
        idx = header.index(column)
    vals = []
    for r in data:
        if idx >= len(r):
            continue
        v = r[idx].strip()
        if v in ("", ".", "NA", "NaN", "null"):
            continue
        try:
            vals.append(float(v))
        except ValueError:
            continue
    return np.asarray(vals, dtype=float)


def load_csv(
    path, column=None, *, predict: str = "next", normalize: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Load a column from a CSV file and build a forecasting task.

    ``column`` may be a header name, an integer index, or ``None`` (last column).
    Missing markers ('', '.', 'NA', 'NaN', 'null') are dropped.
    """
    text = Path(path).read_text(encoding="utf-8")
    series = _column_from_csv(text, column)
    return from_array(series, predict=predict, normalize=normalize)


def load_ted_spread(
    *, predict: str = "next", normalize: bool = True, raw: bool = False
):
    """Load the bundled TED spread series (interbank/counterparty-risk measure).

    The TED spread is the difference between the 3-month interbank rate and the
    3-month Treasury bill; it is a classic gauge of perceived counterparty/credit
    risk in the banking system. Daily, 1986--2022.

    Source: Federal Reserve Bank of St. Louis, FRED, series ``TEDRATE``.

    With ``raw=True`` returns the unprocessed 1-D level array instead of (u, y).
    """
    series = _column_from_csv(
        (_DATA_DIR / "ted_spread.csv").read_text(encoding="utf-8"), "ted_spread"
    )
    if raw:
        return series
    return from_array(series, predict=predict, normalize=normalize)


def load_fred(
    series_id: str,
    *,
    predict: str = "next",
    normalize: bool = True,
    cache_dir=None,
    timeout: float = 30.0,
    raw: bool = False,
):
    """Download a series from FRED (no API key) and build a forecasting task.

    Useful counterparty/credit-risk series include ``TEDRATE`` (TED spread),
    ``BAMLH0A0HYM2`` (US high-yield credit spread), ``BAA10Y`` (Baa credit
    spread) and ``STLFSI4`` / ``NFCI`` (financial stress/conditions). The CSV is
    cached on disk so repeated calls are offline.

    Data are retrieved from FRED (Federal Reserve Bank of St. Louis) and are
    subject to FRED's terms of use.
    """
    cache = Path(cache_dir) if cache_dir else (Path.home() / ".cache" / "esnfed")
    cache.mkdir(parents=True, exist_ok=True)
    cached = cache / f"fred_{series_id}.csv"
    if cached.exists():
        text = cached.read_text(encoding="utf-8")
    else:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        req = urllib.request.Request(url, headers={"User-Agent": "esnfed"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            text = resp.read().decode("utf-8")
        cached.write_text(text, encoding="utf-8")
    series = _column_from_csv(text, series_id)
    if raw:
        return series
    return from_array(series, predict=predict, normalize=normalize)


REGISTRY = {
    "narma10": narma10,
    "mackey_glass": mackey_glass,
    "lorenz": lorenz,
}


def make_dataset(name: str, n: int, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    if name not in REGISTRY:
        raise KeyError(f"unknown dataset {name!r}; choices: {list(REGISTRY)}")
    return REGISTRY[name](n, **kwargs)
