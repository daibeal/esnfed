"""Experiment 12 - FedResPrompt at scale on a real LLM + task (GPU).

Validates FedResPrompt against a Federated LoRA baseline on a *real* frozen LLM
(default Qwen2.5-32B, bf16) and a *real* task (SST-2 sentiment by default,
AG News optional), in a *non-i.i.d. federated* setting. Reports test accuracy,
the per-round communication footprint (measured), and a zero-shot floor.

FedResPrompt here: a fixed Echo State reservoir encodes each input's token
embeddings into a state; a tiny trainable controller (W_out + bottleneck
projection P) maps that state into a soft prompt prepended to the frozen LLM's
input embeddings; the classification loss is back-propagated only to the
controller (LLM + reservoir stay frozen). Clients average their controllers each
round (FedAvg of the controller) — the prompt counterpart of federated ridge.

Self-contained: needs only torch, transformers, datasets, peft (+ bitsandbytes
for 8/4-bit). Run on a single 80-140 GB GPU. See RUNBOOK_gpu.md.

    python exp12_fedresprompt_gpu.py --model Qwen/Qwen2.5-32B --load bf16 \
        --task sst2 --clients 4 --rounds 12
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------- task setup
TASKS = {
    # task -> (hf dataset args, text field, label field, class label words)
    "sst2": (("glue", "sst2"), "sentence", "label", [" negative", " positive"]),
    "agnews": (("ag_news",), "text", "label", [" World", " Sports", " Business", " Tech"]),
}


def load_task(task, tokenizer, n_train, n_test, seed):
    from datasets import load_dataset
    (ds_args, text_f, label_f, words) = TASKS[task]
    ds = load_dataset(*ds_args)
    split_test = "validation" if "validation" in ds else "test"
    rng = np.random.default_rng(seed)

    def take(split, n):
        d = ds[split]
        idx = rng.choice(len(d), size=min(n, len(d)), replace=False)
        return [(d[int(i)][text_f], int(d[int(i)][label_f])) for i in idx]

    train, test = take("train", n_train), take(split_test, n_test)
    # single-token label ids (first token of each class word)
    label_ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in words]
    return train, test, label_ids, words


def noniid_split(train, n_clients, n_classes, alpha, seed):
    """Dirichlet label-skew partition across clients (non-i.i.d.)."""
    rng = np.random.default_rng(seed)
    by_class = [[i for i, (_, y) in enumerate(train) if y == c] for c in range(n_classes)]
    clients = [[] for _ in range(n_clients)]
    for c in range(n_classes):
        idx = by_class[c]; rng.shuffle(idx)
        props = rng.dirichlet([alpha] * n_clients)
        cuts = (np.cumsum(props) * len(idx)).astype(int)[:-1]
        for k, chunk in enumerate(np.split(idx, cuts)):
            clients[k] += [train[i] for i in chunk]
    for k in clients:
        rng.shuffle(k)
    return clients


# ------------------------------------------------------------------ reservoir
class Reservoir:
    """Fixed leaky Echo State reservoir over a (projected) embedding sequence."""

    def __init__(self, n, d, sr=0.9, leak=0.4, d_in=64, device="cuda",
                 dtype=torch.float32, seed=0):
        g = torch.Generator(device="cpu").manual_seed(seed)
        # random sparse-ish reservoir, rescaled to spectral radius
        W = torch.randn(n, n, generator=g) * (torch.rand(n, n, generator=g) < 0.1)
        ev = torch.linalg.eigvals(W).abs().max().real
        W = W * (sr / ev) if ev > 0 else W
        self.W = W.to(device, dtype)
        self.proj = (torch.randn(d, d_in, generator=g) / d ** 0.5).to(device, dtype)  # d->d_in
        self.Win = (torch.randn(n, 1 + d_in, generator=g)).to(device, dtype)
        self.a, self.n, self.device, self.dtype = leak, n, device, dtype

    @torch.no_grad()
    def encode(self, emb, mask):
        """emb (B,T,d), mask (B,T) -> final state (B,n)."""
        B, T, _ = emb.shape
        u = emb.to(self.dtype) @ self.proj                      # (B,T,d_in)
        x = torch.zeros(B, self.n, device=self.device, dtype=self.dtype)
        bias = torch.ones(B, 1, device=self.device, dtype=self.dtype)
        for t in range(T):
            ub = torch.cat([bias, u[:, t]], dim=1)              # (B,1+d_in)
            pre = ub @ self.Win.T + x @ self.W.T
            xn = (1 - self.a) * x + self.a * torch.tanh(pre)
            m = mask[:, t:t + 1].to(self.dtype)
            x = m * xn + (1 - m) * x                            # freeze padded steps
        return x


# ----------------------------------------------------------------- controller
class Controller(nn.Module):
    """Maps a reservoir state -> soft prompt (m prompt tokens x d)."""

    def __init__(self, n, d, k=16, m=4, dtype=torch.float32):
        super().__init__()
        self.m, self.d = m, d
        self.W_out = nn.Parameter(torch.randn(k, n, dtype=dtype) * 0.1)
        self.P = nn.Parameter(torch.randn(m * d, k, dtype=dtype) * (0.5 / k ** 0.5))

    def forward(self, z):                                       # z (B,n)
        b = z @ self.W_out.T                                   # (B,k)
        p = b @ self.P.T                                       # (B,m*d)
        return p.view(-1, self.m, self.d)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


# ------------------------------------------------------------------ utilities
def embed_batch(model, tok, texts, device, max_len=64):
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
              max_length=max_len).to(device)
    with torch.no_grad():
        emb = model.get_input_embeddings()(enc["input_ids"])
    return emb, enc["attention_mask"]


def llm_logits_with_prompt(model, prompt, emb, mask, label_ids):
    """Prepend soft prompt to embeddings; return logits over the label tokens."""
    B = emb.shape[0]
    pm = torch.ones(B, prompt.shape[1], device=emb.device, dtype=mask.dtype)
    inp = torch.cat([prompt.to(emb.dtype), emb], dim=1)
    am = torch.cat([pm, mask], dim=1)
    out = model(inputs_embeds=inp, attention_mask=am)
    last = out.logits[:, -1, :]                                # (B,V)
    return last[:, label_ids]                                  # (B,n_labels)


def batches(data, bs):
    for i in range(0, len(data), bs):
        yield data[i:i + bs]


# ---------------------------------------------------------------- FedResPrompt
def run_fedresprompt(model, tok, reservoir, clients_data, test, label_ids, args,
                     device, dtype):
    d = model.get_input_embeddings().embedding_dim
    global_ctrl = Controller(args.reservoir, d, k=args.k, m=args.prompt_tokens,
                             dtype=torch.float32).to(device)

    def evaluate(ctrl):
        ctrl.eval(); correct = 0
        for batch in batches(test, args.batch):
            texts = [t for t, _ in batch]; ys = torch.tensor([y for _, y in batch], device=device)
            emb, mask = embed_batch(model, tok, texts, device, args.max_len)
            z = reservoir.encode(emb, mask).float()
            with torch.no_grad():
                logits = llm_logits_with_prompt(model, ctrl(z), emb, mask, label_ids)
            correct += (logits.argmax(1) == ys).sum().item()
        return correct / len(test)

    history = [evaluate(global_ctrl)]
    for rnd in range(args.rounds):
        states = []
        for cdata in clients_data:
            ctrl = Controller(args.reservoir, d, k=args.k, m=args.prompt_tokens).to(device)
            ctrl.load_state_dict(global_ctrl.state_dict()); ctrl.train()
            opt = torch.optim.Adam(ctrl.parameters(), lr=args.lr)
            for _ in range(args.local_epochs):
                for batch in batches(cdata, args.batch):
                    texts = [t for t, _ in batch]
                    ys = torch.tensor([y for _, y in batch], device=device)
                    emb, mask = embed_batch(model, tok, texts, device, args.max_len)
                    z = reservoir.encode(emb, mask).float()
                    logits = llm_logits_with_prompt(model, ctrl(z), emb, mask, label_ids)
                    loss = F.cross_entropy(logits, ys)
                    opt.zero_grad(); loss.backward(); opt.step()
            states.append({k: v.detach() for k, v in ctrl.state_dict().items()})
        # FedAvg of the controller
        avg = {k: torch.stack([s[k] for s in states]).mean(0) for k in states[0]}
        global_ctrl.load_state_dict(avg)
        acc = evaluate(global_ctrl)
        history.append(acc)
        print(f"  [FedResPrompt] round {rnd + 1}/{args.rounds}  acc={acc:.4f}")
    # comm per round per client = controller params
    comm_floats = global_ctrl.n_params()
    return dict(history=history, acc=history[-1], comm_floats=comm_floats)


# ------------------------------------------------------------ Federated LoRA
def run_fedlora(model, tok, clients_data, test, label_ids, args, device):
    from peft import LoraConfig, get_peft_model

    def make_peft(targets):
        return get_peft_model(model, LoraConfig(
            r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.0,
            target_modules=targets, bias="none", task_type="CAUSAL_LM"))

    try:  # standard Llama/Qwen/Mistral/Phi attention projections
        peft_model = make_peft(["q_proj", "v_proj"])
    except (ValueError, KeyError):  # other architectures -> all linear layers
        peft_model = make_peft("all-linear")
    peft_model.eval()

    def lora_state():
        return {k: v.detach().clone() for k, v in peft_model.named_parameters()
                if v.requires_grad}

    def set_lora(state):
        with torch.no_grad():
            for k, v in peft_model.named_parameters():
                if k in state:
                    v.copy_(state[k])

    def logits(texts):
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=args.max_len).to(device)
        out = peft_model(**enc)
        return out.logits[:, -1, :][:, label_ids]

    def evaluate():
        correct = 0
        for batch in batches(test, args.batch):
            texts = [t for t, _ in batch]; ys = torch.tensor([y for _, y in batch], device=device)
            with torch.no_grad():
                correct += (logits(texts).argmax(1) == ys).sum().item()
        return correct / len(test)

    init = lora_state()
    for k in init:
        init[k] = torch.zeros_like(init[k])  # start from base model (zeros = no adapter)
    set_lora(init)
    history = [evaluate()]
    global_state = lora_state()
    for rnd in range(args.rounds):
        states = []
        for cdata in clients_data:
            set_lora(global_state)
            opt = torch.optim.Adam([p for p in peft_model.parameters() if p.requires_grad],
                                   lr=args.lr)
            peft_model.train()
            for _ in range(args.local_epochs):
                for batch in batches(cdata, args.batch):
                    texts = [t for t, _ in batch]
                    ys = torch.tensor([y for _, y in batch], device=device)
                    loss = F.cross_entropy(logits(texts), ys)
                    opt.zero_grad(); loss.backward(); opt.step()
            peft_model.eval()
            states.append(lora_state())
        global_state = {k: torch.stack([s[k] for s in states]).mean(0) for k in states[0]}
        set_lora(global_state)
        acc = evaluate(); history.append(acc)
        print(f"  [FedLoRA]      round {rnd + 1}/{args.rounds}  acc={acc:.4f}")
    comm_floats = sum(v.numel() for v in global_state.values())
    return dict(history=history, acc=history[-1], comm_floats=comm_floats)


# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B")
    ap.add_argument("--load", default="bf16", choices=["bf16", "8bit", "4bit"])
    ap.add_argument("--task", default="sst2", choices=list(TASKS))
    ap.add_argument("--clients", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--local-epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--n-train", type=int, default=512)
    ap.add_argument("--n-test", type=int, default=512)
    ap.add_argument("--reservoir", type=int, default=200)
    ap.add_argument("--prompt-tokens", type=int, default=4)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results_exp12.json")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cuda"
    print(f"[exp12] loading {args.model} ({args.load}) ...")
    auth = dict(trust_remote_code=True, token=os.environ.get("HF_TOKEN"))
    tok = AutoTokenizer.from_pretrained(args.model, **auth)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = dict(auth)
    if args.load == "bf16":
        kw.update(torch_dtype=torch.bfloat16, device_map=device)
    else:
        from transformers import BitsAndBytesConfig
        kw.update(device_map=device, quantization_config=BitsAndBytesConfig(
            load_in_8bit=(args.load == "8bit"), load_in_4bit=(args.load == "4bit"),
            bnb_4bit_compute_dtype=torch.bfloat16))
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(args.model, **kw)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    d = model.get_input_embeddings().embedding_dim
    print(f"  loaded in {time.time() - t0:.0f}s | d={d} | "
          f"VRAM={torch.cuda.memory_allocated() / 1e9:.1f} GB")

    train, test, label_ids, words = load_task(args.task, tok, args.n_train,
                                              args.n_test, args.seed)
    n_classes = len(words)
    clients_data = noniid_split(train, args.clients, n_classes, args.alpha, args.seed)
    print(f"[exp12] task={args.task} classes={words} | {args.clients} non-iid clients "
          f"(sizes {[len(c) for c in clients_data]})")

    reservoir = Reservoir(args.reservoir, d, device=device, seed=args.seed)

    print("[exp12] FedResPrompt ...")
    fp = run_fedresprompt(model, tok, reservoir, clients_data, test, label_ids,
                          args, device, torch.bfloat16)
    print("[exp12] Federated LoRA ...")
    lo = run_fedlora(model, tok, clients_data, test, label_ids, args, device)

    chance = 1.0 / n_classes
    res = dict(
        model=args.model, load=args.load, task=args.task, classes=words,
        clients=args.clients, rounds=args.rounds, chance=chance,
        fedresprompt=dict(acc=fp["acc"], history=fp["history"],
                          comm_floats_per_round=fp["comm_floats"],
                          zero_shot=fp["history"][0]),
        fedlora=dict(acc=lo["acc"], history=lo["history"],
                     comm_floats_per_round=lo["comm_floats"]),
        comm_ratio=lo["comm_floats"] / max(1, fp["comm_floats"]),
    )
    Path(args.out).write_text(json.dumps(res, indent=2))
    print("\n==================== RESULTS ====================")
    print(f"chance               : {chance:.3f}")
    print(f"zero-shot (prompt~0) : {fp['history'][0]:.3f}")
    print(f"FedResPrompt acc     : {fp['acc']:.3f}  "
          f"(comm {fp['comm_floats']/1e3:.0f}k floats/round)")
    print(f"Federated LoRA acc   : {lo['acc']:.3f}  "
          f"(comm {lo['comm_floats']/1e6:.1f}M floats/round)")
    print(f"FedResPrompt uses {res['comm_ratio']:.0f}x less communication")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
