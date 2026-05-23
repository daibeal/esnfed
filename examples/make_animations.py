"""Generate animated GIFs for the documentation site.

Writes to ``docs/animations/``:
* ``spectral_sweep.gif`` -- reservoir eigenvalues as the spectral radius grows
  past the unit circle (the echo state property boundary), and
* ``reservoir_echoes.gif`` -- reservoir activations responding to an input
  impulse and fading over time (the "echoes").

    pip install "esnfed[viz]" pillow
    python examples/make_animations.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from esnfed import EchoStateNetwork, topologies
from esnfed.esn import _spectral_radius

OUT = Path(__file__).resolve().parents[1] / "docs" / "animations"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 80, "font.size": 10})


def spectral_sweep():
    rng = np.random.default_rng(0)
    W0 = topologies.random_reservoir(150, density=0.1, rng=rng)
    eig0 = np.linalg.eigvals(W0)
    sr0 = float(np.max(np.abs(eig0)))
    rhos = np.linspace(0.2, 1.4, 45)
    theta = np.linspace(0, 2 * np.pi, 200)

    fig, ax = plt.subplots(figsize=(4.6, 4.6))

    def update(i):
        ax.clear()
        rho = rhos[i]
        eig = eig0 * (rho / sr0)
        ax.plot(np.cos(theta), np.sin(theta), color="grey", ls="--", lw=1)
        color = "#55A868" if rho <= 1.0 else "#C44E52"
        ax.scatter(eig.real, eig.imag, s=14, color="#4C72B0", alpha=0.75)
        ax.set_title(f"Spectral radius $\\rho = {rho:.2f}$  "
                     f"({'echo state OK' if rho <= 1 else 'echo state lost'})",
                     color=color)
        ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.6, 1.6)
        ax.set_aspect("equal"); ax.axhline(0, color="k", lw=.4); ax.axvline(0, color="k", lw=.4)
        ax.set_xlabel("Re"); ax.set_ylabel("Im")
        fig.tight_layout()

    anim = FuncAnimation(fig, update, frames=len(rhos), interval=90)
    path = OUT / "spectral_sweep.gif"
    anim.save(str(path), writer=PillowWriter(fps=12))
    plt.close(fig)
    print(f"  {path.name}")


def reservoir_echoes():
    rng = np.random.default_rng(1)
    W = topologies.small_world_reservoir(120, k=6, p=0.1, rng=rng)
    esn = EchoStateNetwork(1, 1, W, spectral_radius=0.95, leaking_rate=0.3,
                           washout=0, input_scaling=1.0, seed=0)
    T = 120
    u = np.zeros((T, 1))
    u[5, 0] = 1.0  # impulse
    Z = esn.harvest(u)
    states = Z[:, 1 + esn.n_inputs:]
    neurons = rng.choice(states.shape[1], size=6, replace=False)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(neurons)))

    fig, ax = plt.subplots(figsize=(7.2, 3.4))

    def update(t):
        ax.clear()
        for j, c in zip(neurons, colors):
            ax.plot(range(t), states[:t, j], color=c, lw=1.2)
        ax.axvline(5, color="grey", ls=":", lw=1)
        ax.text(6, 0.85, "impulse", color="grey", fontsize=8)
        ax.set_xlim(0, T); ax.set_ylim(-1, 1)
        ax.set_xlabel("time step"); ax.set_ylabel("activation x(t)")
        ax.set_title("Reservoir echoes: response to an input impulse")
        fig.tight_layout()

    anim = FuncAnimation(fig, update, frames=range(2, T + 1, 2), interval=70)
    path = OUT / "reservoir_echoes.gif"
    anim.save(str(path), writer=PillowWriter(fps=14))
    plt.close(fig)
    print(f"  {path.name}")


def main():
    print("[animations] rendering GIFs via matplotlib -> docs/animations/")
    spectral_sweep()
    reservoir_echoes()
    print("[animations] done")


if __name__ == "__main__":
    main()
