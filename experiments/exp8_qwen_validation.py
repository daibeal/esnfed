"""Experiment 8 - FedResPrompt validation on a real Qwen LLM.

Validates that the FedResPrompt split-federated gradient flow learns when the
server hosts a *real* frozen language model (Qwen2.5-0.5B) instead of the NumPy
surrogate. Two edge clients jointly train a shared ESN prompt controller so that
distinct local contexts are routed, through the frozen Qwen, to distinct class
tokens. We report classification accuracy before and after training (chance is
1/K) and the training loss curve.

This requires torch + transformers and downloads Qwen2.5-0.5B on first run:
    pip install "esnfed[llm]" torch
    python experiments/exp8_qwen_validation.py
"""
from __future__ import annotations

import numpy as np

import common
from esnfed import EchoStateNetwork, federated, topologies
from esnfed.llm_orchestration import EdgeClient


def restricted_step(client, lm, ctx, class_ids, target_idx):
    """One split-federated example with the restricted-class objective: the
    client builds the soft prompt, the server scores it over the candidate class
    tokens and returns dL/dprompt, the client updates locally."""
    prompt = client.make_prompt(ctx)
    loss, grad = lm.restricted_loss_and_grad(prompt, class_ids, target_idx)
    client.apply_server_gradient(grad)
    return loss

MODEL = "Qwen/Qwen2.5-0.5B"
N_RES = 200
N_PROMPT_TOKENS = 8
BOTTLENECK = 32
WINDOW = 16
N_CLIENTS = 2
EPOCHS = 15
TRAIN_PER_CLASS = 3   # per client
TEST_PER_CLASS = 4
LR = 0.5


def pick_class_tokens(tokenizer, k):
    """Find k candidate words that each encode to a single token id."""
    candidates = [" up", " down", " flat", " high", " low", " yes", " no",
                  " one", " two", " three", " A", " B", " C"]
    ids, words = [], []
    for w in candidates:
        toks = tokenizer.encode(w, add_special_tokens=False)
        if len(toks) == 1 and toks[0] not in ids:
            ids.append(toks[0]); words.append(w)
        if len(ids) == k:
            break
    return ids, words


def make_context(level: float, rng, n=WINDOW):
    """A short, near-constant input series at the given level (+ small noise)."""
    return (level + 0.02 * rng.standard_normal((n, 1))).clip(0, 1)


def build_dataset(levels, rng, per_class):
    """Return list of (context, class_index) over the K levels."""
    data = []
    for ci, lvl in enumerate(levels):
        for _ in range(per_class):
            data.append((make_context(lvl, rng), ci))
    rng.shuffle(data)
    return data


def accuracy(client, lm, data, class_ids):
    correct = 0
    for ctx, ci in data:
        prompt = client.projection.forward(client.W_out @ client.esn.harvest(
            np.atleast_2d(ctx))[-1])
        logit = lm.logits(prompt)
        pred = int(np.argmax([logit[t] for t in class_ids]))
        correct += int(pred == ci)
    return correct / len(data)


def main():
    try:
        from esnfed.llm_orchestration import TransformersLM
        lm = TransformersLM(MODEL)
    except Exception as e:  # noqa: BLE001
        print(f"[exp8] cannot load {MODEL}: {type(e).__name__}: {e}")
        print("       install torch + transformers to run this validation.")
        return

    print(f"[exp8] FedResPrompt validation on {MODEL} (d={lm.d})")
    rng = np.random.default_rng(common.MASTER_SEED)
    class_ids, words = pick_class_tokens(lm.tokenizer, 3)
    levels = [0.0, 0.25, 0.5][: len(class_ids)]
    print(f"  classes -> tokens {words} (ids {class_ids})")

    W = topologies.random_reservoir(N_RES, density=0.1, rng=rng)  # shared reservoir
    clients = [
        EdgeClient(EchoStateNetwork(1, 1, W, spectral_radius=0.9, leaking_rate=0.6,
                                    washout=0, seed=0),
                   bottleneck_dim=BOTTLENECK, embed_dim=lm.d,
                   n_prompt_tokens=N_PROMPT_TOKENS, seed=c, lr=LR)
        for c in range(N_CLIENTS)
    ]
    train = [build_dataset(levels, rng, TRAIN_PER_CLASS) for _ in clients]
    test = build_dataset(levels, rng, TEST_PER_CLASS)

    acc_before = accuracy(clients[0], lm, test, class_ids)
    print(f"  accuracy before training: {acc_before:.2f} (chance = {1/len(class_ids):.2f})")

    history = []
    for ep in range(EPOCHS):
        losses = []
        for client, data in zip(clients, train):
            for ctx, ci in data:
                losses.append(restricted_step(client, lm, ctx, class_ids, ci))
        federated.federated_prompt_average(clients)  # share the controller
        history.append(float(np.mean(losses)))
        print(f"  epoch {ep + 1:2d}/{EPOCHS}  loss={history[-1]:.3f}")

    acc_after = accuracy(clients[0], lm, test, class_ids)
    print(f"  accuracy after training:  {acc_after:.2f}")

    # Save a small figure: loss curve + accuracy before/after.
    common.set_style()
    fig, axes = common.plt.subplots(1, 2, figsize=(8.4, 3.6))
    axes[0].plot(range(1, EPOCHS + 1), history, marker="o",
                 color=common.PALETTE["fedavg"])
    axes[0].set_xlabel("Federated epoch"); axes[0].set_ylabel("Cross-entropy loss")
    axes[0].set_title(f"Prompt-tuning {MODEL.split('/')[-1]} (frozen)")
    axes[1].bar(["before", "after"], [acc_before, acc_after],
                color=[common.PALETTE["local"], common.PALETTE["federated_ridge"]])
    axes[1].axhline(1 / len(class_ids), color="grey", ls="--", lw=1, label="chance")
    axes[1].set_ylim(0, 1.05); axes[1].set_ylabel("Test accuracy")
    axes[1].set_title("Classification accuracy"); axes[1].legend()
    fig.suptitle("FedResPrompt validation on a real Qwen LLM", y=1.03)
    common.save_figure(fig, "exp8_qwen_validation")

    # Persist the headline numbers for the thesis.
    out = common.OUTPUTS / "exp8_qwen_validation.txt"
    out.write_text(
        f"model={MODEL}\nclasses={words}\nacc_before={acc_before:.3f}\n"
        f"acc_after={acc_after:.3f}\nloss_first={history[0]:.3f}\n"
        f"loss_last={history[-1]:.3f}\n", encoding="utf-8")
    print(f"[exp8] done -> {out.relative_to(common.REPO_ROOT)}")


if __name__ == "__main__":
    main()
