"""Generate interactive Plotly examples with esnfed.viz.

Writes small, self-contained interactive HTML files (Plotly.js loaded from CDN)
for the four plot types, using the library's default Plotly backend. Open any of
the resulting ``docs/plotly/*.html`` files in a browser to pan/zoom/hover.

    pip install "esnfed[viz,experiments]"
    python examples/plotly_gallery.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from esnfed import EchoStateNetwork, datasets, metrics, topologies, viz


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "memoria").is_dir():   # thesis monorepo
            return p
    for p in here.parents:
        if (p / "pyproject.toml").exists():  # standalone esnfed repo
            return p
    return here.parents[1]


OUT = _repo_root() / "docs" / "plotly"
OUT.mkdir(parents=True, exist_ok=True)


def _save(fig, name: str):
    path = OUT / f"{name}.html"
    # CDN keeps each file ~tens of KB instead of embedding all of Plotly.js.
    fig.write_html(str(path), include_plotlyjs="cdn", full_html=True)
    print(f"  {path.relative_to(OUT.parents[2])}")


def main():
    print("[plotly] generating interactive examples via esnfed.viz (Plotly backend)")
    rng = np.random.default_rng(20260524)
    W = topologies.small_world_reservoir(120, k=6, p=0.1, rng=rng)
    esn = EchoStateNetwork(1, 1, W, spectral_radius=0.9, leaking_rate=0.5, washout=100)

    u, y = datasets.load_ted_spread()
    u_tr, y_tr, u_te, y_te = datasets.split(u, y, 0.7)
    esn.fit(u_tr, y_tr)
    pred = esn.predict(u_te)
    nrmse = metrics.nrmse(y_te[esn.washout:], pred[esn.washout:])
    print(f"  (TED-spread forecast NRMSE = {nrmse:.3f})")

    _save(viz.plot_reservoir(esn, backend="plotly"), "reservoir")
    _save(viz.plot_spectrum(esn, backend="plotly"), "spectrum")
    _save(viz.plot_states(esn, u_te, n_neurons=6, backend="plotly"), "states")
    _save(viz.plot_forecast(y_te, pred, washout=esn.washout, backend="plotly"),
          "forecast")
    print("[plotly] done -> docs/plotly/")


if __name__ == "__main__":
    main()
