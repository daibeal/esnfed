"""FedResPrompt - Federated Reservoir Prompt Orchestration (experimental).

A *split-federated* architecture in which an Echo State Network acts as an
ultra-lightweight **prompt controller** at the edge. The reservoir maps local
context to a small \"bottleneck\" vector, which a linear projection lifts into a
language model's embedding space to form a **soft prompt**. Only the soft-prompt
embedding (uplink) and its loss gradient (downlink) cross the client-server
boundary; the heavy language model lives on the server.

    local context --ESN--> state z --W_out--> bottleneck b --P--> soft prompt p
                                                                       |  (uplink)
                                                                       v
                                              server LM: loss + dL/dp  |  (downlink)
                                                                       v
    update P and W_out locally  <--- backprop dL/dp through P and W_out

Why it matters. Classical federated LLM tuning (e.g. federated LoRA) ships
adapter weights for every layer each round; FedResPrompt ships only a single
soft-prompt vector and its gradient, which is orders of magnitude smaller, and
keeps the forward/backward pass of the frozen model on the server, off the edge
device. The communication and edge-compute savings are analysed in
``experiments/exp7_fedres_prompt.py``.

The server language model is *pluggable*. By default a lightweight NumPy
surrogate is used --- a frozen linear ``soft prompt -> vocabulary`` head with the
exact cross-entropy gradient with respect to the prompt --- so the architecture
runs anywhere with only NumPy. If ``transformers`` (with a torch backend) is
installed, a real ``AutoModelForCausalLM`` can be plugged in via
:class:`TransformersLM` (the gradient w.r.t. the input embedding is obtained from
the autograd backward pass).

This module is experimental and optional: ``pip install "esnfed[llm]"``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .esn import EchoStateNetwork
from .metrics import nrmse  # noqa: F401  (re-exported convenience)

FLOAT_BYTES = 4  # float32 on the wire


# ─────────────────────────────────────────────────────────── bottleneck
class BottleneckProjection:
    """Linear projection from the bottleneck space (R^k) to the LM embedding
    space (R^{n_tokens x d}), trained by SGD.

    The soft prompt is ``p = P b`` reshaped to ``(n_tokens, d)``.
    """

    def __init__(self, k: int, d: int, n_tokens: int = 1, *, seed: int = 0,
                 scale: float = 0.1):
        rng = np.random.default_rng(seed)
        self.k = k
        self.d = d
        self.n_tokens = n_tokens
        self.P = scale * rng.standard_normal((n_tokens * d, k))

    def forward(self, b: np.ndarray) -> np.ndarray:
        """b: (k,) -> soft prompt (n_tokens, d)."""
        self._b = b
        return (self.P @ b).reshape(self.n_tokens, self.d)

    def backward(self, grad_prompt: np.ndarray):
        """Given dL/dp (n_tokens, d), return dL/db (k,) and store dL/dP."""
        g = np.asarray(grad_prompt, float).reshape(-1)  # (n_tokens*d,)
        self.grad_P = np.outer(g, self._b)  # (n_tokens*d, k)
        return self.P.T @ g  # dL/db, shape (k,)

    def step(self, lr: float):
        self.P -= lr * self.grad_P

    @property
    def n_params(self) -> int:
        return self.P.size


# ─────────────────────────────────────────────────────────── surrogate LM
class SurrogateLM:
    """A lightweight, frozen stand-in for a server-side language model.

    A single linear head ``U`` maps a (mean-pooled) soft prompt to logits over a
    small vocabulary. It is *frozen* (as the LLM is in prompt tuning) and returns
    the cross-entropy loss together with the exact gradient with respect to the
    prompt embedding -- exactly the signal a real CausalLM would back-propagate
    to its input embeddings.
    """

    def __init__(self, vocab_size: int, d: int, *, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.vocab_size = vocab_size
        self.d = d
        self.U = rng.standard_normal((vocab_size, d)) / np.sqrt(d)  # frozen

    def loss_and_grad(self, prompt: np.ndarray, target: int):
        """prompt: (n_tokens, d) -> (loss, dL/dprompt with same shape)."""
        p = np.atleast_2d(prompt)
        h = p.mean(axis=0)  # mean-pool the prompt tokens -> (d,)
        logits = self.U @ h  # (V,)
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()
        loss = -np.log(probs[target] + 1e-12)
        grad_h = self.U.T @ (probs - _onehot(target, self.vocab_size))  # (d,)
        grad_prompt = np.tile(grad_h / p.shape[0], (p.shape[0], 1))  # (n_tokens, d)
        return float(loss), grad_prompt

    def forward_flops(self, n_tokens: int) -> int:
        # mean-pool (n_tokens*d) + matvec (vocab*d); backward ~2x forward.
        return 3 * (self.vocab_size * self.d + n_tokens * self.d)


def _onehot(i: int, n: int) -> np.ndarray:
    v = np.zeros(n)
    v[i] = 1.0
    return v


# ─────────────────────────────────────────────────────────── edge client
@dataclass
class EdgeClient:
    """An edge device: a (frozen) reservoir + a trainable readout and projection.

    The reservoir and its input weights are fixed; only ``W_out`` (state ->
    bottleneck) and the :class:`BottleneckProjection` are learned, by
    back-propagating the gradient the server returns.
    """

    esn: EchoStateNetwork
    bottleneck_dim: int
    embed_dim: int
    n_prompt_tokens: int = 1
    seed: int = 0
    lr: float = 0.05

    W_out: np.ndarray = field(init=False)
    projection: BottleneckProjection = field(init=False)
    bytes_up: int = field(default=0, init=False)
    bytes_down: int = field(default=0, init=False)
    flops: int = field(default=0, init=False)

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        self.W_out = 0.1 * rng.standard_normal((self.bottleneck_dim, self.esn.readout_dim))
        self.projection = BottleneckProjection(
            self.bottleneck_dim, self.embed_dim, self.n_prompt_tokens, seed=self.seed
        )

    # context (T, n_inputs) -> reservoir state -> bottleneck -> soft prompt
    def make_prompt(self, context: np.ndarray) -> np.ndarray:
        Z = self.esn.harvest(np.atleast_2d(context).reshape(-1, self.esn.n_inputs))
        z = Z[-1]  # last extended state summarises the context
        self._z = z
        self._b = self.W_out @ z  # bottleneck (k,)
        prompt = self.projection.forward(self._b)  # (n_tokens, d)
        # uplink: the soft prompt only
        self.bytes_up += prompt.size * FLOAT_BYTES
        # ESN forward cost: O(N^2) per step + readout
        n = self.esn.n_reservoir
        self.flops += Z.shape[0] * (n * n) + self.bottleneck_dim * self.esn.readout_dim
        return prompt

    def apply_server_gradient(self, grad_prompt: np.ndarray):
        """Back-propagate dL/dp through P and W_out and take an SGD step."""
        self.bytes_down += np.asarray(grad_prompt).size * FLOAT_BYTES  # downlink
        grad_b = self.projection.backward(grad_prompt)  # (k,)
        grad_W = np.outer(grad_b, self._z)  # (k, readout_dim)
        self.projection.step(self.lr)
        self.W_out -= self.lr * grad_W

    @property
    def trainable_params(self) -> int:
        return self.W_out.size + self.projection.n_params


# ─────────────────────────────────────────────────────────── server
@dataclass
class Server:
    """The server hosting the (frozen) language model."""

    lm: SurrogateLM
    bytes_up: int = field(default=0, init=False)
    bytes_down: int = field(default=0, init=False)
    flops: int = field(default=0, init=False)

    def evaluate(self, prompt: np.ndarray, target: int):
        """Run the LM forward/backward; return (loss, dL/dprompt)."""
        self.bytes_up += prompt.size * FLOAT_BYTES  # received from client
        loss, grad_prompt = self.lm.loss_and_grad(prompt, target)
        self.bytes_down += grad_prompt.size * FLOAT_BYTES  # sent to client
        self.flops += self.lm.forward_flops(prompt.shape[0])
        return loss, grad_prompt


# ─────────────────────────────────────────────────────── split-fed round
def split_federated_step(client: EdgeClient, server: Server,
                         context: np.ndarray, target: int) -> float:
    """One split-federated example: client builds prompt, server scores it,
    client updates from the returned gradient. Returns the loss."""
    prompt = client.make_prompt(context)
    loss, grad_prompt = server.evaluate(prompt, target)
    client.apply_server_gradient(grad_prompt)
    return loss


# ─────────────────────────────────── communication / compute accounting
def fedresprompt_bytes_per_round(d_model: int, n_prompt_tokens: int = 1,
                                 dtype_bytes: int = FLOAT_BYTES) -> int:
    """Client<->server bytes per example: soft prompt up + its gradient down."""
    return 2 * n_prompt_tokens * d_model * dtype_bytes


def fedlora_bytes_per_round(d_model: int, n_layers: int, rank: int,
                            adapters_per_layer: int = 2,
                            dtype_bytes: int = FLOAT_BYTES) -> int:
    """Bytes per round for federated LoRA: adapter weights up + aggregate down.

    Each adapted projection contributes two low-rank factors of size
    ``rank x d_model`` (A and B), so ``2 * rank * d_model`` parameters; there are
    ``adapters_per_layer`` of them per layer (e.g. query and value).
    """
    params = n_layers * adapters_per_layer * 2 * rank * d_model
    return 2 * params * dtype_bytes  # upload + download


def llm_flops(n_params: int, tokens: int) -> int:
    """Standard estimate: forward ~2*P*T, backward ~4*P*T -> 6*P*T."""
    return 6 * n_params * tokens


def esn_edge_flops(n_reservoir: int, steps: int, readout_dim: int,
                   bottleneck_dim: int) -> int:
    """Edge cost of FedResPrompt: reservoir run + readout (no LLM on device)."""
    return steps * n_reservoir * n_reservoir + bottleneck_dim * readout_dim


# ─────────────────────────────────────────── optional transformers backend
class TransformersLM:  # pragma: no cover - requires transformers + torch
    """A real, frozen server-side CausalLM (optional), e.g. a small Qwen.

    The soft prompt is fed as ``inputs_embeds``; the next-token logits after the
    prompt give the cross-entropy against a target *token id*, and the gradient
    with respect to the prompt is read from the autograd backward pass --- the
    same interface as :class:`SurrogateLM`, so it is a drop-in for :class:`Server`.

    Requires ``transformers`` and ``torch`` (``pip install "esnfed[llm]"`` plus a
    torch build). Tested with ``Qwen/Qwen2.5-0.5B``.
    """

    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B"):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "TransformersLM needs `transformers` and `torch`; "
                "install esnfed[llm] (and torch). The default SurrogateLM has no "
                "such requirement."
            ) from e
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.float()  # fp32 on CPU
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)  # frozen LLM
        self.d = self.model.get_input_embeddings().embedding_dim
        self.vocab_size = self.model.config.vocab_size

    def _logits_tensor(self, prompt):
        torch = self._torch
        # leaf tensor so that .grad is populated after backward()
        p_leaf = torch.tensor(np.atleast_2d(prompt), dtype=torch.float32,
                              requires_grad=True)  # (m, d)
        out = self.model(inputs_embeds=p_leaf.unsqueeze(0))  # (1, m, d)
        return p_leaf, out.logits[:, -1, :]  # next-token logits, shape (1, V)

    def loss_and_grad(self, prompt: np.ndarray, target: int):
        """Next-token cross-entropy against token id ``target`` and dL/dprompt."""
        torch = self._torch
        p_leaf, logits = self._logits_tensor(prompt)
        loss = torch.nn.functional.cross_entropy(
            logits, torch.tensor([int(target)]))
        loss.backward()
        grad_prompt = p_leaf.grad.detach().to(torch.float32).numpy()
        return float(loss.item()), grad_prompt

    def restricted_loss_and_grad(self, prompt: np.ndarray, class_ids, target_idx: int):
        """Cross-entropy over only the candidate class tokens (the standard way to
        do classification with an LLM) and dL/dprompt. ``target_idx`` indexes
        into ``class_ids``."""
        torch = self._torch
        p_leaf, logits = self._logits_tensor(prompt)
        sub = logits[0, list(class_ids)].unsqueeze(0)  # (1, K)
        loss = torch.nn.functional.cross_entropy(sub, torch.tensor([int(target_idx)]))
        loss.backward()
        return float(loss.item()), p_leaf.grad.detach().to(torch.float32).numpy()

    def logits(self, prompt: np.ndarray) -> np.ndarray:
        """Next-token logit vector (numpy) for the given soft prompt."""
        with self._torch.no_grad():
            _, logits = self._logits_tensor(prompt)
        return logits.squeeze(0).numpy()

    def forward_flops(self, n_tokens: int) -> int:  # interface parity with Server
        return 0  # measured separately; LLM cost dominates and is on the server
