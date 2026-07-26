"""Visualisation utilities for Echo State Networks (optional).

Four ways to *see* an ESN:

* :func:`plot_reservoir` -- the reservoir connectivity as a network graph
  (nodes sized/coloured by degree); makes the topology differences visible.
* :func:`plot_spectrum` -- the reservoir eigenvalues in the complex plane with
  the unit circle and spectral radius; makes the echo state property visible.
* :func:`plot_states` -- a sample of reservoir activations over time (the
  "echoes"); shows whether the reservoir is rich or saturated.
* :func:`plot_forecast` -- predicted vs. actual with the NRMSE in the title.

Three backends are supported and selected with ``backend=``:
``"plotly"`` (the default, interactive), ``"matplotlib"`` and ``"seaborn"``
(``seaborn`` is matplotlib with the seaborn theme applied). The plotting
libraries are imported lazily, so importing this module never pulls them in;
install them with::

    pip install "esnfed[viz]"

Each function returns the native figure object (a Plotly ``Figure`` or a
Matplotlib ``Figure``); use :func:`save` or the object's own methods to render.
"""
from __future__ import annotations

import numpy as np

DEFAULT_BACKEND = "plotly"
_BACKENDS = ("plotly", "matplotlib", "seaborn")


# ─────────────────────────────────────────────────────────────── helpers
def _check_backend(backend: str) -> None:
    if backend not in _BACKENDS:
        raise ValueError(f"backend must be one of {_BACKENDS}, got {backend!r}")


def _reservoir_matrix(obj) -> np.ndarray:
    """Accept an EchoStateNetwork or a raw weight matrix and return a dense ``W``.

    An ESN built with ``sparse=True`` stores ``W`` as a SciPy CSR matrix, which
    ``np.asarray`` wraps in a 0-d object array instead of converting -- every plot
    then failed with "setting an array element with a sequence". Sparse matrices
    are densified explicitly.
    """
    W = getattr(obj, "W", None)
    if W is None:
        W = obj
    if hasattr(W, "toarray"):        # SciPy sparse matrix / array
        W = W.toarray()
    W = np.asarray(W, dtype=float)
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError(f"expected a square reservoir matrix, got shape {W.shape}")
    return W


def _plotly():
    try:
        import plotly.graph_objects as go
    except ImportError as e:  # pragma: no cover - exercised only without plotly
        raise ImportError(
            "plotly is required for backend='plotly'; install esnfed[viz]"
        ) from e
    return go


def _matplotlib(backend: str):
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "matplotlib is required for this backend; install esnfed[viz]"
        ) from e
    if backend == "seaborn":
        try:
            import seaborn as sns
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "seaborn is required for backend='seaborn'; install esnfed[viz]"
            ) from e
        sns.set_theme(style="whitegrid")
    return plt


def _degrees(W: np.ndarray) -> np.ndarray:
    A = (W != 0).astype(int)
    return A.sum(axis=0) + A.sum(axis=1)  # in + out degree


# ─────────────────────────────────────────────────────────── 1. topology
def plot_reservoir(obj, *, backend: str = DEFAULT_BACKEND, layout: str = "spring",
                   seed: int = 0, title: str | None = None):
    """Draw the reservoir connectivity as a network graph."""
    import networkx as nx

    _check_backend(backend)
    W = _reservoir_matrix(obj)
    n = W.shape[0]
    A = (W != 0).astype(int)
    G = nx.from_numpy_array(A, create_using=nx.DiGraph)
    layouts = {
        "spring": lambda g: nx.spring_layout(g, seed=seed),
        "circular": nx.circular_layout,
        "kamada_kawai": nx.kamada_kawai_layout,
    }
    if layout not in layouts:
        raise ValueError(f"layout must be one of {list(layouts)}")
    pos = layouts[layout](G)
    deg = _degrees(W)
    dmax = max(int(deg.max()), 1)
    title = title or f"Reservoir topology (N={n}, {int(A.sum())} edges)"

    if backend == "plotly":
        go = _plotly()
        ex, ey = [], []
        for u, v in G.edges():
            ex += [pos[u][0], pos[v][0], None]
            ey += [pos[u][1], pos[v][1], None]
        edge_trace = go.Scatter(x=ex, y=ey, mode="lines",
                                line=dict(width=0.4, color="rgba(120,120,120,0.4)"),
                                hoverinfo="none")
        node_trace = go.Scatter(
            x=[pos[i][0] for i in G.nodes()], y=[pos[i][1] for i in G.nodes()],
            mode="markers", hoverinfo="text",
            text=[f"node {i} · degree {int(deg[i])}" for i in G.nodes()],
            marker=dict(size=[6 + 22 * deg[i] / dmax for i in G.nodes()],
                        color=[int(deg[i]) for i in G.nodes()],
                        colorscale="Viridis", showscale=True,
                        colorbar=dict(title="degree"), line=dict(width=0.5,
                        color="white")),
        )
        fig = go.Figure([edge_trace, node_trace])
        fig.update_layout(title=title, showlegend=False, template="plotly_white",
                          xaxis=dict(visible=False), yaxis=dict(visible=False))
        return fig

    plt = _matplotlib(backend)
    fig, ax = plt.subplots(figsize=(6, 5))
    for u, v in G.edges():
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                color="grey", lw=0.3, alpha=0.4, zorder=1)
    xs = [pos[i][0] for i in G.nodes()]
    ys = [pos[i][1] for i in G.nodes()]
    sc = ax.scatter(xs, ys, c=deg, s=[20 + 120 * d / dmax for d in deg],
                    cmap="viridis", edgecolors="white", linewidths=0.5, zorder=2)
    fig.colorbar(sc, ax=ax, label="degree")
    ax.set_title(title)
    ax.set_axis_off()
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────── 2. spectrum
def plot_spectrum(obj, *, backend: str = DEFAULT_BACKEND, title: str | None = None):
    """Plot the reservoir eigenvalues with the unit circle and spectral radius."""
    _check_backend(backend)
    W = _reservoir_matrix(obj)
    eig = np.linalg.eigvals(W)
    sr = float(np.max(np.abs(eig))) if eig.size else 0.0
    title = title or f"Reservoir spectrum (spectral radius = {sr:.3f})"
    theta = np.linspace(0, 2 * np.pi, 256)

    if backend == "plotly":
        go = _plotly()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.cos(theta), y=np.sin(theta), mode="lines",
                                 line=dict(color="grey", dash="dash"),
                                 name="unit circle"))
        if sr > 0:
            fig.add_trace(go.Scatter(x=sr * np.cos(theta), y=sr * np.sin(theta),
                                     mode="lines", line=dict(color="#C44E52"),
                                     name=f"ρ = {sr:.3f}"))
        fig.add_trace(go.Scatter(x=eig.real, y=eig.imag, mode="markers",
                                 marker=dict(size=5, color="#4C72B0", opacity=0.7),
                                 name="eigenvalues"))
        fig.update_layout(title=title, template="plotly_white",
                          xaxis_title="Re", yaxis_title="Im")
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        return fig

    plt = _matplotlib(backend)
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.plot(np.cos(theta), np.sin(theta), color="grey", ls="--", label="unit circle")
    if sr > 0:
        ax.plot(sr * np.cos(theta), sr * np.sin(theta), color="#C44E52",
                label=f"$\\rho$ = {sr:.3f}")
    ax.scatter(eig.real, eig.imag, s=18, color="#4C72B0", alpha=0.7,
               label="eigenvalues")
    ax.axhline(0, color="black", lw=0.5)
    ax.axvline(0, color="black", lw=0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("Re")
    ax.set_ylabel("Im")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────── 3. states
def plot_states(esn, u, *, n_neurons: int = 8, max_steps: int = 300,
                backend: str = DEFAULT_BACKEND, seed: int = 0,
                title: str | None = None):
    """Plot a sample of reservoir activations over time."""
    _check_backend(backend)
    if n_neurons < 1:
        raise ValueError(f"n_neurons must be >= 1, got {n_neurons}")
    if max_steps < 1:
        raise ValueError(f"max_steps must be >= 1, got {max_steps}")
    # Let the ESN own the input-layout rules, so an ambiguous array is rejected
    # here exactly as it would be in fit()/predict() rather than being reshaped.
    u = esn._as_inputs(u) if hasattr(esn, "_as_inputs") else \
        np.asarray(u, dtype=float).reshape(-1, esn.n_inputs)
    Z = esn.harvest(u[:max_steps])
    states = Z[:, 1 + esn.n_inputs:]
    rng = np.random.default_rng(seed)
    k = min(n_neurons, states.shape[1])
    idx = rng.choice(states.shape[1], size=k, replace=False)
    t = np.arange(states.shape[0])
    title = title or f"Reservoir activations ({k} of {states.shape[1]} neurons)"

    if backend == "plotly":
        go = _plotly()
        fig = go.Figure()
        for j in idx:
            fig.add_trace(go.Scatter(x=t, y=states[:, j], mode="lines",
                                     line=dict(width=1), name=f"neuron {j}"))
        fig.update_layout(title=title, template="plotly_white",
                          xaxis_title="time step", yaxis_title="activation x(t)")
        return fig

    plt = _matplotlib(backend)
    fig, ax = plt.subplots(figsize=(8, 4))
    for j in idx:
        ax.plot(t, states[:, j], lw=0.8, alpha=0.85, label=f"neuron {j}")
    ax.set_xlabel("time step")
    ax.set_ylabel("activation x(t)")
    ax.set_title(title)
    if k <= 10:
        ax.legend(ncol=2, fontsize=7)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────── 4. forecast
def plot_forecast(y_true, y_pred, *, backend: str = DEFAULT_BACKEND,
                  washout: int = 0, title: str | None = None,
                  max_points: int = 2000):
    """Overlay predicted vs. actual, with the NRMSE in the title."""
    from .metrics import nrmse

    _check_backend(backend)
    yt = np.asarray(y_true, dtype=float).ravel()[washout:]
    yp = np.asarray(y_pred, dtype=float).ravel()[washout:]
    err = nrmse(yt, yp)
    if len(yt) > max_points:  # keep the figure light
        yt, yp = yt[:max_points], yp[:max_points]
    t = np.arange(len(yt))
    title = title or f"Forecast vs. actual (NRMSE = {err:.3f})"

    if backend == "plotly":
        go = _plotly()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=yt, mode="lines", name="actual",
                                 line=dict(color="#333333", width=1)))
        fig.add_trace(go.Scatter(x=t, y=yp, mode="lines", name="forecast",
                                 line=dict(color="#4C72B0", width=1)))
        fig.update_layout(title=title, template="plotly_white",
                          xaxis_title="time step", yaxis_title="value")
        return fig

    plt = _matplotlib(backend)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(t, yt, color="#333333", lw=0.9, label="actual")
    ax.plot(t, yp, color="#4C72B0", lw=0.9, alpha=0.85, label="forecast")
    ax.set_xlabel("time step")
    ax.set_ylabel("value")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────── save helper
def save(fig, path: str) -> str:
    """Save a figure produced by this module.

    Plotly figures are written to ``.html`` (interactive) or, if the extension
    is an image format and ``kaleido`` is installed, to a static image.
    Matplotlib figures are written with ``savefig``.
    """
    path = str(path)
    if type(fig).__module__.startswith("plotly"):
        if path.lower().endswith(".html"):
            fig.write_html(path)
        else:
            fig.write_image(path)  # needs kaleido
    else:
        fig.savefig(path, bbox_inches="tight", dpi=200)
    return path
