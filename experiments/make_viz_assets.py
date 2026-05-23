"""Generate visual assets *with the library itself* (esnfed.viz).

Produces:
* README gallery PNGs in ``docs/images/`` (rendered on GitHub and PyPI), and
* thesis figure PDFs in ``memoria/figuras/`` (the four reservoir topologies and
  an eigenvalue spectrum), which the thesis embeds and attributes to esnfed.viz.

Run: ``python experiments/make_viz_assets.py``
"""
from __future__ import annotations

import numpy as np

import common  # paths + headless Agg backend
from esnfed import EchoStateNetwork, datasets, topologies, viz

DOCS = common.REPO_ROOT / "docs" / "images"
DOCS.mkdir(parents=True, exist_ok=True)
FIG = common.FIGURES  # memoria/figuras

TOPOS = {
    "random": ("Erdos-Renyi", dict(density=0.12), "spring"),
    "small_world": ("Small-world", dict(k=6, p=0.1), "circular"),
    "scale_free": ("Scale-free", dict(m=3), "spring"),
    "ring": ("Ring", dict(), "circular"),
}


def thesis_topology_figures(n=64):
    """One reservoir-graph PDF per topology, for the thesis (drawn by esnfed.viz)."""
    rng = np.random.default_rng(common.MASTER_SEED)
    for kind, (label, kw, layout) in TOPOS.items():
        W = topologies.make_reservoir(kind, n, rng=rng, **kw)
        fig = viz.plot_reservoir(W, backend="matplotlib", layout=layout, title=label)
        viz.save(fig, str(FIG / f"viz_topo_{kind}.pdf"))


def thesis_spectrum_figure():
    """An eigenvalue-spectrum PDF for the thesis (drawn by esnfed.viz)."""
    rng = np.random.default_rng(common.MASTER_SEED)
    W = topologies.random_reservoir(200, density=0.1, rng=rng)
    esn = EchoStateNetwork(1, 1, W, spectral_radius=0.9)
    fig = viz.plot_spectrum(esn, backend="matplotlib")
    viz.save(fig, str(FIG / "viz_spectrum.pdf"))


def thesis_states_figure():
    """A reservoir-activation PDF for the thesis (drawn by esnfed.viz)."""
    rng = np.random.default_rng(common.MASTER_SEED)
    W = topologies.random_reservoir(200, density=0.1, rng=rng)
    esn = EchoStateNetwork(1, 1, W, spectral_radius=0.9, leaking_rate=0.3, washout=100)
    u, _ = datasets.mackey_glass(600, seed=common.MASTER_SEED)
    fig = viz.plot_states(esn, u, n_neurons=8, max_steps=300, backend="matplotlib")
    viz.save(fig, str(FIG / "viz_states.pdf"))


def readme_gallery():
    """Four PNGs for the README gallery, themed on the finance use case."""
    rng = np.random.default_rng(common.MASTER_SEED)
    W = topologies.small_world_reservoir(120, k=6, p=0.1, rng=rng)
    esn = EchoStateNetwork(1, 1, W, spectral_radius=0.9, leaking_rate=0.5, washout=100)
    u, y = datasets.load_ted_spread()
    u_tr, y_tr, u_te, y_te = datasets.split(u, y, 0.7)
    esn.fit(u_tr, y_tr)
    pred = esn.predict(u_te)

    viz.save(viz.plot_reservoir(esn, backend="matplotlib"), str(DOCS / "reservoir.png"))
    viz.save(viz.plot_spectrum(esn, backend="matplotlib"), str(DOCS / "spectrum.png"))
    viz.save(viz.plot_states(esn, u_te, n_neurons=6, backend="matplotlib"),
             str(DOCS / "states.png"))
    viz.save(viz.plot_forecast(y_te, pred, washout=100, backend="matplotlib"),
             str(DOCS / "forecast.png"))


def main():
    print("[viz-assets] generating thesis figures and README gallery via esnfed.viz")
    thesis_topology_figures()
    thesis_spectrum_figure()
    thesis_states_figure()
    readme_gallery()
    print(f"  thesis PDFs -> {FIG.relative_to(common.REPO_ROOT)}")
    print(f"  gallery PNGs -> {DOCS.relative_to(common.REPO_ROOT)}")
    print("[viz-assets] done")


if __name__ == "__main__":
    main()
