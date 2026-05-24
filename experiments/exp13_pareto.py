"""Experiment 13a - communication vs accuracy Pareto (GPU).

Sweeps FedResPrompt (prompt-token count) and Federated LoRA (rank) on a frozen
LLM and plots test accuracy against per-round communication. Shows that
FedResPrompt occupies the cheap-but-accurate corner of the trade-off.

    python exp13_pareto.py --model Qwen/Qwen2.5-32B --load bf16 --task sst2
"""
from __future__ import annotations

import argparse
import gc
import json
from types import SimpleNamespace
from pathlib import Path

import torch

from exp12_fedresprompt_gpu import (Reservoir, load_task, noniid_split,
                                    run_fedlora, run_fedresprompt)
from exp12_sweep import load_model

OUT = Path.home() / "esnfed_gpu_out"
OUT.mkdir(exist_ok=True)


def base(**kw):
    a = dict(clients=4, rounds=10, local_epochs=2, batch=8, n_train=512,
             n_test=512, reservoir=200, prompt_tokens=4, k=16, lora_r=8,
             lr=3e-3, max_len=64, alpha=0.3, seed=0)
    a.update(kw)
    return SimpleNamespace(**a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B")
    ap.add_argument("--load", default="bf16")
    ap.add_argument("--task", default="sst2")
    ap.add_argument("--prompt-tokens", nargs="+", type=int, default=[1, 2, 4, 8])
    ap.add_argument("--ranks", nargs="+", type=int, default=[2, 4, 8, 16])
    args = ap.parse_args()
    pts = {"model": args.model, "task": args.task,
           "fedresprompt": [], "fedlora": []}

    # FedResPrompt does not modify the model -> one load for the whole m sweep.
    model, tok, _ = load_model(args.model, args.load)
    d = model.get_input_embeddings().embedding_dim
    a0 = base()
    train, test, label_ids, words = load_task(args.task, tok, a0.n_train,
                                              a0.n_test, a0.seed)
    cd = noniid_split(train, a0.clients, len(words), a0.alpha, a0.seed)
    for m in args.prompt_tokens:
        a = base(prompt_tokens=m)
        res = Reservoir(a.reservoir, d, device="cuda", seed=a.seed)
        r = run_fedresprompt(model, tok, res, cd, test, label_ids, a, "cuda",
                             torch.bfloat16)
        pts["fedresprompt"].append(dict(prompt_tokens=m, acc=round(r["acc"], 4),
                                        comm_floats=r["comm_floats"]))
        print(f"  [FRP] m={m}: acc={r['acc']:.3f}  comm={r['comm_floats']/1e3:.0f}k")
        (OUT / "pareto.json").write_text(json.dumps(pts, indent=2))
    del model
    gc.collect(); torch.cuda.empty_cache()

    # LoRA injects adapters into the model -> reload fresh per rank.
    for rk in args.ranks:
        model, tok, _ = load_model(args.model, args.load)
        train, test, label_ids, words = load_task(args.task, tok, a0.n_train,
                                                  a0.n_test, a0.seed)
        cd = noniid_split(train, a0.clients, len(words), a0.alpha, a0.seed)
        a = base(lora_r=rk)
        r = run_fedlora(model, tok, cd, test, label_ids, a, "cuda")
        pts["fedlora"].append(dict(rank=rk, acc=round(r["acc"], 4),
                                   comm_floats=r["comm_floats"]))
        print(f"  [LoRA] r={rk}: acc={r['acc']:.3f}  comm={r['comm_floats']/1e6:.1f}M")
        (OUT / "pareto.json").write_text(json.dumps(pts, indent=2))
        del model
        gc.collect(); torch.cuda.empty_cache()

    print(f"[pareto] saved -> {OUT / 'pareto.json'}")


if __name__ == "__main__":
    main()
