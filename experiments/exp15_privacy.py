"""Experiment 15: privacy--utility trade-off of differentially private federation.

Sweeps the differential-privacy budget ``epsilon`` for differentially private
federated ridge on NARMA-10 (shared reservoir, ``K`` clients) and compares the
test NRMSE against the exact (non-private) federated ridge baseline. It also
verifies that secure aggregation and streaming accumulation reproduce the exact
readout -- privacy/continual extensions that are *exact*, in contrast to DP which
trades accuracy for the formal guarantee.

DP adds Gaussian noise (analytic Gaussian mechanism, valid for any epsilon) to the
sufficient statistics ``(A, B)``. Because ``A`` is a high-dimensional, ill-conditioned
Gram matrix, the perturbed system is solved with a ridge set to the spectral scale
of the injected noise, ``rho_DP = 2 sigma sqrt(K D)`` (semicircle law), which keeps
the solve well-posed at every budget. Utility then degrades monotonically as the
budget tightens; the cost is substantial and persistent -- the known difficulty of
privatising second-moment matrices.

Outputs ``exp15_privacy.{pdf,tex}`` into the thesis tree.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import FIGURES, MASTER_SEED, TABLES, save_figure, save_latex_table, set_style

from esnfed import datasets, federated, metrics, topologies
from esnfed.esn import solve_readout
from esnfed.privacy import PrivacyConfig, dp_statistics, gaussian_sigma
from esnfed.streaming import StreamingRidge

N_RESERVOIR = 80
K_CLIENTS = 4
N_SEEDS = 20
EPSILONS = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
DELTA = 1e-5


def _dp_readout(clients, ref, cfg, dp_ridge, seed):
    """Sum each client's differentially private (A, B) and solve once."""
    D = ref.readout_dim
    A = np.zeros((D, D))
    B = np.zeros((D, ref.n_outputs))
    base = np.random.default_rng(seed)
    for c in clients:
        child = np.random.default_rng(int(base.integers(0, 2**63 - 1)))
        Ak, Bk = dp_statistics(c.states(), c.targets(), cfg, rng=child)
        A += Ak
        B += Bk
    return solve_readout(A, B, dp_ridge)


def main() -> None:
    rng = np.random.default_rng(MASTER_SEED)
    u, y = datasets.narma10(12000, rng=rng)
    utr, ytr, ute, yte = datasets.split(u, y, 0.7)
    W = topologies.random_reservoir(N_RESERVOIR, density=0.1, rng=rng)
    parts = datasets.partition_iid(utr, ytr, K_CLIENTS, rng=rng)
    kw = dict(spectral_radius=0.9, leaking_rate=1.0, ridge=1e-6, washout=100)
    clients, ref = federated.make_shared_clients(W, parts, input_seed=0, esn_kwargs=kw)

    washout = ref.washout
    Z_test = ref.harvest(ute)[washout:]
    y_eval = np.atleast_2d(yte).reshape(-1, ref.n_outputs)[washout:]
    D = ref.readout_dim

    W_exact = federated.federated_ridge(clients, ref)
    nrmse_exact = metrics.nrmse(y_eval, Z_test @ W_exact)

    # Clip at the 90th percentile of the per-record norms (clip outliers only).
    z_norms = np.concatenate([np.linalg.norm(c.states(), axis=1) for c in clients])
    y_norms = np.concatenate([np.linalg.norm(c.targets(), axis=1) for c in clients])
    cz = float(np.quantile(z_norms, 0.90))
    cy = float(np.quantile(y_norms, 0.90))
    sens = cz * np.sqrt(cz * cz + cy * cy)

    rows = []
    for eps in EPSILONS:
        sigma = gaussian_sigma(eps, DELTA, sens)              # analytic mechanism
        dp_ridge = kw["ridge"] + 2.0 * sigma * np.sqrt(K_CLIENTS * D)
        cfg = PrivacyConfig(epsilon=eps, delta=DELTA, clip_state=cz, clip_target=cy)
        errs = [metrics.nrmse(y_eval, Z_test @ _dp_readout(clients, ref, cfg, dp_ridge, 1000 + s))
                for s in range(N_SEEDS)]
        errs = np.asarray(errs)
        rows.append({"epsilon": eps, "nrmse_mean": float(errs.mean()),
                     "nrmse_std": float(errs.std())})
        print(f"  eps={eps:>5}: NRMSE {errs.mean():.3f} +/- {errs.std():.3f}")
    df = pd.DataFrame(rows)

    # Exact extensions: secure aggregation and streaming must match the exact readout.
    W_secure = federated.federated_ridge_secure(clients, ref, seed=0, mask_scale=1.0)
    secure_ok = bool(np.allclose(W_secure, W_exact, rtol=1e-3, atol=1e-6))
    acc = StreamingRidge(ref.readout_dim, ref.n_outputs, ridge=kw["ridge"])
    for c in clients:
        acc.update(c.states(), c.targets())
    streaming_ok = bool(np.allclose(acc.readout(), W_exact, atol=1e-9))
    print(f"  exact baseline NRMSE = {nrmse_exact:.3f}")
    print(f"  secure==exact: {secure_ok}   streaming==exact: {streaming_ok}")

    # ---- figure -------------------------------------------------------------
    set_style()
    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    ax.errorbar(df.epsilon, df.nrmse_mean, yerr=df.nrmse_std, marker="o",
                capsize=3, color="#8172B3", label="DP federated ridge")
    ax.axhline(nrmse_exact, color="#4C72B0", ls="--",
               label=f"exact / secure / streaming ({nrmse_exact:.3f})")
    ax.axhline(1.0, color="0.6", ls=":", label="trivial predictor (NRMSE = 1)")
    ax.set_xscale("log")
    ax.set_xlabel(r"privacy budget $\varepsilon$ (log scale; smaller $=$ more private)")
    ax.set_ylabel("NARMA-10 test NRMSE")
    ax.legend(loc="upper right")
    save_figure(fig, "exp15_privacy")

    # ---- table --------------------------------------------------------------
    out = pd.DataFrame({
        r"$\varepsilon$": [f"{e:g}" for e in df.epsilon],
        "DP NRMSE (mean)": df.nrmse_mean.to_numpy(),
        "DP NRMSE (std)": df.nrmse_std.to_numpy(),
    })
    save_latex_table(
        out, "exp15_privacy",
        caption=(
            f"Privacy--utility trade-off of differentially private federated ridge "
            f"on NARMA-10 ({K_CLIENTS} clients, $N={N_RESERVOIR}$, "
            f"$\\delta=10^{{-5}}$, analytic Gaussian mechanism, mean over {N_SEEDS} "
            f"noise seeds). The exact baseline is NRMSE {nrmse_exact:.3f}; secure "
            f"aggregation and streaming accumulation reproduce it exactly, whereas "
            f"differential privacy carries a substantial, persistent cost."),
        label="tab:exp15-privacy", float_format="%.3f",
    )
    print(f"  ({FIGURES.name}/exp15_privacy.pdf, {TABLES.name}/exp15_privacy.tex)")


if __name__ == "__main__":
    main()
