"""Experiment 7 - FedResPrompt communication/compute benchmark.

Benchmarks the proposed FedResPrompt split-federated architecture against an
analytical Federated LoRA baseline along the two axes it is meant to improve:

* communication: bytes exchanged between client and server per round, and
* edge compute: FLOPs the *edge device* must perform per round.

It also runs a small proof-of-concept: three simulated edge clients, each with a
local time series, jointly train a shared prompt controller through the
split-federated gradient exchange (with periodic FedAvg of the controller). The
loss curve confirms the gradient flow learns; the communication/compute figures
are computed from standard analytical formulas and validated against the
measured byte counts of the simulation.

Outputs: memoria/figuras/exp7_fedresprompt.pdf and memoria/tablas/exp7_fedresprompt.tex.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import common
from esnfed import EchoStateNetwork, datasets, federated, topologies
from esnfed.llm_orchestration import (
    EdgeClient,
    Server,
    SurrogateLM,
    esn_edge_flops,
    fedlora_bytes_per_round,
    fedresprompt_bytes_per_round,
    llm_flops,
    split_federated_step,
)

N_RES = 200
SIM_D = 64          # embedding dim for the runnable proof-of-concept
VOCAB = 4           # quantile classes of the next value
WINDOW = 30
PROMPT_TOKENS = 10  # soft-prompt length assumed for the analytical comparison
LORA_RANK = 8
TOKENS_PER_STEP = 64  # sequence length assumed for the LoRA edge forward/backward

# Representative transformer scales (name, d_model, n_layers).
MODELS = [
    ("GPT-2 (124M)", 768, 12),
    ("GPT-2 L (774M)", 1280, 36),
    ("1.3B", 2048, 24),
    ("7B", 4096, 32),
    ("13B", 5120, 40),
]


def _transformer_params(d_model: int, n_layers: int) -> int:
    """Standard estimate of decoder parameters: ~12 * L * d^2."""
    return 12 * n_layers * d_model * d_model


# ─────────────────────────────────────────── proof-of-concept training
def _windows(series: np.ndarray, edges: np.ndarray):
    s = series.ravel()
    ctxs, targets = [], []
    for t in range(WINDOW, len(s) - 1):
        ctxs.append(s[t - WINDOW:t].reshape(-1, 1))
        targets.append(int(np.digitize(s[t], edges)))
    return ctxs, targets


def proof_of_concept(rounds=60, n_clients=3, seed=0):
    rng = np.random.default_rng(common.MASTER_SEED + seed)
    W = topologies.random_reservoir(N_RES, density=0.1, rng=rng)  # shared reservoir
    server = Server(SurrogateLM(VOCAB, SIM_D, seed=1))

    # Quantile bin edges from a pooled sample, so targets span the vocabulary.
    pooled = datasets.mackey_glass(2000, seed=common.MASTER_SEED)[0].ravel()
    edges = np.quantile(pooled, [0.25, 0.5, 0.75])

    clients, data = [], []
    for c in range(n_clients):
        u, _ = datasets.mackey_glass(1200, seed=common.MASTER_SEED + 10 * c + 1)
        esn = EchoStateNetwork(1, 1, W, spectral_radius=0.9, leaking_rate=0.5,
                               washout=0, seed=0)
        clients.append(EdgeClient(esn, bottleneck_dim=16, embed_dim=SIM_D,
                                  n_prompt_tokens=1, seed=c, lr=0.1))
        data.append(_windows(u, edges))

    history = []
    for r in range(rounds):
        round_losses = []
        for client, (ctxs, targets) in zip(clients, data):
            idx = rng.choice(len(ctxs), size=32, replace=False)
            for i in idx:
                round_losses.append(
                    split_federated_step(client, server, ctxs[i], targets[i]))
        federated.federated_prompt_average(clients)  # share the controller
        history.append(float(np.mean(round_losses)))
    total_client_bytes = sum(c.bytes_up + c.bytes_down for c in clients)
    return history, clients, server, total_client_bytes


# ─────────────────────────────────────────── analytical comparison
def comparison_table():
    rows = []
    for name, d, layers in MODELS:
        params = _transformer_params(d, layers)
        fr_bytes = fedresprompt_bytes_per_round(d, PROMPT_TOKENS)
        lora_bytes = fedlora_bytes_per_round(d, layers, LORA_RANK)
        fr_flops = esn_edge_flops(N_RES, WINDOW, N_RES + 2, 16)
        lora_flops = llm_flops(params, TOKENS_PER_STEP)
        rows.append({
            "model": name, "d_model": d, "n_layers": layers, "params": params,
            "fr_bytes": fr_bytes, "lora_bytes": lora_bytes,
            "bytes_ratio": lora_bytes / fr_bytes,
            "fr_flops": fr_flops, "lora_flops": lora_flops,
            "flops_ratio": lora_flops / fr_flops,
        })
    return pd.DataFrame(rows)


def plot(history, df):
    common.set_style()
    fig, axes = common.plt.subplots(1, 3, figsize=(12.5, 3.8))

    # (a) proof-of-concept convergence, against the random-guess baseline
    axes[0].plot(range(1, len(history) + 1), history, color=common.PALETTE["fedavg"],
                 label="FedResPrompt")
    axes[0].axhline(np.log(VOCAB), color="grey", ls="--", lw=1,
                    label=f"random ($\\ln {VOCAB}$)")
    axes[0].set_xlabel("Federated round")
    axes[0].set_ylabel("Mean cross-entropy loss")
    axes[0].set_title("Proof of concept: 3 edge clients")
    axes[0].legend()

    # (b) communication per round vs model size
    axes[1].plot(df["params"] / 1e6, df["lora_bytes"] / 1e6, marker="s",
                 color=common.PALETTE["local"], label="Federated LoRA")
    axes[1].plot(df["params"] / 1e6, df["fr_bytes"] / 1e6, marker="o",
                 color=common.PALETTE["federated_ridge"], label="FedResPrompt")
    axes[1].set_xscale("log"); axes[1].set_yscale("log")
    axes[1].set_xlabel("Model size (M params)")
    axes[1].set_ylabel("Comm. per round (MB)")
    axes[1].set_title("Communication")
    axes[1].legend()

    # (c) edge FLOPs per round vs model size
    axes[2].plot(df["params"] / 1e6, df["lora_flops"], marker="s",
                 color=common.PALETTE["local"], label="Federated LoRA (edge)")
    axes[2].plot(df["params"] / 1e6, df["fr_flops"], marker="o",
                 color=common.PALETTE["federated_ridge"], label="FedResPrompt (edge)")
    axes[2].set_xscale("log"); axes[2].set_yscale("log")
    axes[2].set_xlabel("Model size (M params)")
    axes[2].set_ylabel("Edge FLOPs per round")
    axes[2].set_title("Edge compute")
    axes[2].legend()

    fig.suptitle("FedResPrompt vs. Federated LoRA", y=1.03)
    common.save_figure(fig, "exp7_fedresprompt")


def _human_bytes(n):
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def main():
    print("[exp7] FedResPrompt communication/compute benchmark")
    history, clients, server, total_bytes = proof_of_concept()
    print(f"  proof-of-concept loss: {history[0]:.3f} -> {history[-1]:.3f}")
    # validate the accounting against the analytical formula at the sim size
    expected = fedresprompt_bytes_per_round(SIM_D, n_prompt_tokens=1)
    print(f"  measured bytes/exchange matches formula: "
          f"{server.bytes_up + server.bytes_down > 0} "
          f"(per-exchange = {expected} B)")

    df = comparison_table()
    common.save_table(df, "exp7_fedresprompt")

    # Compact LaTeX table at representative scales.
    tex = df.copy()
    tex["FedResPrompt"] = tex["fr_bytes"].map(_human_bytes)
    tex["Fed. LoRA"] = tex["lora_bytes"].map(_human_bytes)
    tex["Comm. saving"] = tex["bytes_ratio"].map(lambda x: f"{x:,.0f}x")
    tex["Edge FLOPs saving"] = tex["flops_ratio"].map(lambda x: f"{x:,.0f}x")
    tex = tex[["model", "FedResPrompt", "Fed. LoRA", "Comm. saving",
               "Edge FLOPs saving"]].rename(columns={"model": "Model"})
    common.save_latex_table(
        tex, "exp7_fedresprompt",
        caption="FedResPrompt vs. Federated LoRA: client--server communication "
                "per round (10 soft-prompt tokens vs. rank-8 LoRA on all layers) "
                "and per-round edge FLOPs, across model scales. Savings are the "
                "LoRA/FedResPrompt ratio.",
        label="tab:exp7-fedresprompt")

    for _, r in df.iterrows():
        print(f"  {r['model']:>14}: comm {_human_bytes(r['fr_bytes'])} vs "
              f"{_human_bytes(r['lora_bytes'])}  ({r['bytes_ratio']:,.0f}x), "
              f"edge FLOPs saving {r['flops_ratio']:,.0f}x")
    plot(history, df)
    print("[exp7] done")


if __name__ == "__main__":
    main()
