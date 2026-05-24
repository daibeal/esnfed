"""Model-size sweep for FedResPrompt vs Federated LoRA (GPU).

Runs the exp12 experiment across a list of frozen LLMs of growing size and
collects accuracy + per-round communication + GPU telemetry into one JSON, to
show how FedResPrompt scales with model size (and how the communication
advantage over Federated LoRA grows). Reuses the helpers in
``exp12_fedresprompt_gpu.py``.

    python exp12_sweep.py                      # default: 0.5B,1.5B,7B,14B on SST-2
    python exp12_sweep.py --models Qwen/Qwen2.5-0.5B Qwen/Qwen2.5-7B
    python exp12_sweep.py --tasks sst2 agnews
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from types import SimpleNamespace
from pathlib import Path

import torch

from exp12_fedresprompt_gpu import (Reservoir, load_task, noniid_split,
                                    run_fedlora, run_fedresprompt)

OUT = Path.home() / "esnfed_gpu_out"
OUT.mkdir(exist_ok=True)

# Diverse, mostly open/ungated families and sizes. Add bigger or gated models
# (e.g. meta-llama/Llama-3.1-8B, google/gemma-2-9b, Qwen/Qwen2.5-32B) on the
# command line with --models; gated ones need HF_TOKEN set in the environment.
DEFAULT_MODELS = [
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-3B",
    "Qwen/Qwen2.5-7B",
    "Qwen/Qwen2.5-14B",
    "mistralai/Mistral-7B-v0.3",
    "microsoft/Phi-3.5-mini-instruct",
]


def load_model(name, load="bf16"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    auth = dict(trust_remote_code=True, token=os.environ.get("HF_TOKEN"))
    tok = AutoTokenizer.from_pretrained(name, **auth)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = dict(auth)
    if load == "bf16":
        kw.update(torch_dtype=torch.bfloat16, device_map="cuda")
    else:
        from transformers import BitsAndBytesConfig
        kw.update(device_map="cuda", quantization_config=BitsAndBytesConfig(
            load_in_8bit=(load == "8bit"), load_in_4bit=(load == "4bit"),
            bnb_4bit_compute_dtype=torch.bfloat16))
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(name, **kw)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--tasks", nargs="+", default=["sst2"])
    ap.add_argument("--load", default="bf16", choices=["bf16", "8bit", "4bit"],
                    help="precision for ALL models (use 8bit/4bit for 32B/72B)")
    ap.add_argument("--clients", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=10)
    args_cli = ap.parse_args()

    a = SimpleNamespace(clients=args_cli.clients, rounds=args_cli.rounds,
                        local_epochs=2, batch=8, n_train=512, n_test=512,
                        reservoir=200, prompt_tokens=4, k=16, lora_r=8, lr=3e-3,
                        max_len=64, alpha=0.3, seed=0)
    results = []
    for name in args_cli.models:
        try:
            print(f"\n================= {name} ({args_cli.load}) =================")
            model, tok, lt = load_model(name, args_cli.load)
            d = model.get_input_embeddings().embedding_dim
            nparam = sum(p.numel() for p in model.parameters())
            vram = torch.cuda.memory_allocated() / 1e9
            print(f"  loaded {nparam/1e9:.2f}B in {lt:.0f}s | d={d} | {vram:.1f} GB")
            for task in args_cli.tasks:
                train, test, label_ids, words = load_task(task, tok, a.n_train,
                                                          a.n_test, a.seed)
                cd = noniid_split(train, a.clients, len(words), a.alpha, a.seed)
                res = Reservoir(a.reservoir, d, device="cuda", seed=a.seed)
                fp = run_fedresprompt(model, tok, res, cd, test, label_ids, a,
                                      "cuda", torch.bfloat16)
                lo = run_fedlora(model, tok, cd, test, label_ids, a, "cuda")
                row = dict(model=name, params_b=round(nparam / 1e9, 2), d=d,
                           load_s=round(lt, 1), vram_gb=round(vram, 1),
                           task=task, classes=words, chance=1 / len(words),
                           zero_shot=round(fp["history"][0], 4),
                           fedresprompt_acc=round(fp["acc"], 4),
                           fedlora_acc=round(lo["acc"], 4),
                           fp_comm_floats=fp["comm_floats"],
                           lora_comm_floats=lo["comm_floats"],
                           comm_ratio=round(lo["comm_floats"] / max(1, fp["comm_floats"]), 1))
                results.append(row)
                (OUT / "sweep_results.json").write_text(json.dumps(results, indent=2))
                print(f"  [{task}] FedResPrompt {fp['acc']:.3f} | LoRA {lo['acc']:.3f}"
                      f" | {row['comm_ratio']:.0f}x less comm")
            del model
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001 - keep the sweep going
            print(f"  FAILED {name}: {e}")
            results.append(dict(model=name, error=str(e)[:200]))
            (OUT / "sweep_results.json").write_text(json.dumps(results, indent=2))
    print(f"\n[sweep] saved -> {OUT / 'sweep_results.json'}")


if __name__ == "__main__":
    main()
