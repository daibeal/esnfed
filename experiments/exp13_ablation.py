"""Experiment 13b - FedResPrompt ablations (GPU).

Varies one design choice at a time around a base configuration and reports test
accuracy: reservoir size N, soft-prompt length m, bottleneck k, number of
clients, and non-i.i.d. severity (Dirichlet alpha). FedResPrompt does not modify
the LLM, so a single model load covers the whole study.

    python exp13_ablation.py --model Qwen/Qwen2.5-32B --load bf16 --task sst2
"""
from __future__ import annotations

import argparse
import json
from types import SimpleNamespace
from pathlib import Path

import torch

from exp12_fedresprompt_gpu import (Reservoir, load_task, noniid_split,
                                    run_fedresprompt)
from exp12_sweep import load_model

OUT = Path.home() / "esnfed_gpu_out"
OUT.mkdir(exist_ok=True)

BASE = dict(clients=4, rounds=10, local_epochs=2, batch=8, n_train=512,
            n_test=512, reservoir=200, prompt_tokens=4, k=16, lora_r=8,
            lr=3e-3, max_len=64, alpha=0.3, seed=0)

# axis -> values to try
AXES = {
    "reservoir": [50, 100, 200, 400],
    "prompt_tokens": [1, 2, 4, 8],
    "k": [8, 16, 32],
    "clients": [2, 4, 8],
    "alpha": [0.1, 0.3, 1.0, 10.0],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B")
    ap.add_argument("--load", default="bf16")
    ap.add_argument("--task", default="sst2")
    args = ap.parse_args()

    model, tok, _ = load_model(args.model, args.load)
    d = model.get_input_embeddings().embedding_dim

    def run(cfg):
        a = SimpleNamespace(**cfg)
        train, test, label_ids, words = load_task(args.task, tok, a.n_train,
                                                  a.n_test, a.seed)
        cd = noniid_split(train, a.clients, len(words), a.alpha, a.seed)
        res = Reservoir(a.reservoir, d, device="cuda", seed=a.seed)
        r = run_fedresprompt(model, tok, res, cd, test, label_ids, a, "cuda",
                             torch.bfloat16)
        return round(r["acc"], 4)

    results = {"model": args.model, "task": args.task, "base": dict(BASE), "axes": {}}
    print(f"[ablation] base acc (reference): ", end="", flush=True)
    results["base_acc"] = run(dict(BASE))
    print(results["base_acc"])
    for axis, values in AXES.items():
        results["axes"][axis] = []
        for v in values:
            cfg = dict(BASE); cfg[axis] = v
            acc = run(cfg)
            results["axes"][axis].append(dict(value=v, acc=acc))
            print(f"  {axis}={v}: acc={acc:.3f}")
            (OUT / "ablation.json").write_text(json.dumps(results, indent=2))
    print(f"[ablation] saved -> {OUT / 'ablation.json'}")


if __name__ == "__main__":
    main()
