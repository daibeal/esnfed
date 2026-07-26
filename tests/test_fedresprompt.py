"""Tests for the experimental FedResPrompt module (esnfed.llm_orchestration)."""
import numpy as np

from esnfed import EchoStateNetwork, federated, topologies
from esnfed.llm_orchestration import (
    BottleneckProjection,
    EdgeClient,
    Server,
    SurrogateLM,
    fedlora_bytes_per_round,
    fedresprompt_bytes_per_round,
    split_federated_step,
)


def test_bottleneck_projection_shapes():
    proj = BottleneckProjection(k=5, d=16, n_tokens=2, seed=0)
    b = np.ones(5)
    p = proj.forward(b)
    assert p.shape == (2, 16)
    grad_b = proj.backward(np.ones((2, 16)))
    assert grad_b.shape == (5,)
    assert proj.grad_P.shape == (2 * 16, 5)


def test_surrogate_lm_gradient_matches_numerical():
    lm = SurrogateLM(vocab_size=5, d=8, seed=1)
    rng = np.random.default_rng(0)
    prompt = rng.standard_normal((1, 8))
    target = 3
    loss0, grad = lm.loss_and_grad(prompt, target)
    # central finite differences on a few coordinates
    eps = 1e-5
    flat = prompt.reshape(-1)
    for idx in [0, 3, 7]:
        d = np.zeros_like(flat)
        d[idx] = eps
        lp, _ = lm.loss_and_grad((flat + d).reshape(1, 8), target)
        lm_, _ = lm.loss_and_grad((flat - d).reshape(1, 8), target)
        num = (lp - lm_) / (2 * eps)
        assert abs(num - grad.reshape(-1)[idx]) < 1e-4


def test_split_federated_step_reduces_loss():
    rng = np.random.default_rng(0)
    W = topologies.random_reservoir(100, density=0.1, rng=rng)
    esn = EchoStateNetwork(1, 1, W, spectral_radius=0.9, leaking_rate=0.5, washout=20)
    client = EdgeClient(esn, bottleneck_dim=8, embed_dim=32, seed=0, lr=0.2)
    server = Server(SurrogateLM(vocab_size=4, d=32, seed=1))
    ctx = rng.uniform(0, 0.5, size=(50, 1))
    losses = [split_federated_step(client, server, ctx, target=2) for _ in range(120)]
    assert losses[-1] < losses[0]
    assert losses[-1] < 0.2 * losses[0]  # clear convergence on the surrogate task


def test_communication_accounting_tracks_bytes():
    rng = np.random.default_rng(0)
    W = topologies.random_reservoir(60, density=0.1, rng=rng)
    esn = EchoStateNetwork(1, 1, W, spectral_radius=0.9, washout=10)
    client = EdgeClient(esn, bottleneck_dim=4, embed_dim=16, n_prompt_tokens=1, seed=0)
    server = Server(SurrogateLM(vocab_size=3, d=16, seed=0))
    split_federated_step(client, server, rng.uniform(0, 0.5, (20, 1)), target=1)
    # one prompt up (16 floats) + one gradient down (16 floats), float32
    assert client.bytes_up == 16 * 4
    assert client.bytes_down == 16 * 4


def test_fedresprompt_cheaper_than_lora():
    # Representative LLM: 32 layers, d_model 4096, LoRA rank 8.
    fr = fedresprompt_bytes_per_round(d_model=4096, n_prompt_tokens=10)
    lora = fedlora_bytes_per_round(d_model=4096, n_layers=32, rank=8)
    assert fr < lora
    assert lora / fr > 20  # orders-of-magnitude lighter on the wire


def test_federated_prompt_average_shares_controller():
    rng = np.random.default_rng(0)
    W = topologies.random_reservoir(50, density=0.1, rng=rng)
    clients = []
    for s in range(3):
        esn = EchoStateNetwork(1, 1, W, spectral_radius=0.9, washout=10)
        clients.append(EdgeClient(esn, bottleneck_dim=4, embed_dim=8, seed=s))
    federated.federated_prompt_average(clients)
    # after averaging every client holds the same controller
    for c in clients[1:]:
        assert np.allclose(c.W_out, clients[0].W_out)
        assert np.allclose(c.projection.P, clients[0].projection.P)
