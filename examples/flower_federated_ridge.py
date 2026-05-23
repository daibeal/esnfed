"""Federated counterparty-risk forecasting with esnfed + Flower.

This example shows how the exact federated-ridge scheme maps onto `Flower
<https://flower.ai>`_, the de-facto federated-learning framework. Several
simulated financial institutions each hold a private slice of a
counterparty-risk time series (the TED spread) and jointly train a single Echo
State Network *without sharing raw data*.

The mapping is exact and elegant:

* each client's Flower ``fit`` returns the **ridge sufficient statistics**
  ``A_k = Z_k^T Z_k`` and ``B_k = Z_k^T Y_k`` of its local data (never the data);
* a custom Flower ``Strategy`` **sums** them and solves the ridge system once,
  recovering the model that pooled training would have produced.

Flower is an *optional* dependency::

    pip install "esnfed[flower]"

Run as a script::

    python examples/flower_federated_ridge.py

The Flower Simulation Engine needs Ray, which is not always available on every
platform; this example therefore drives a single federated round through the
real Flower client/strategy classes directly, exercising the same interfaces
Flower's engine uses. The very same ``FederatedRidgeClient`` and
``FederatedRidgeStrategy`` plug into ``flwr run`` for a real deployment.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Make esnfed importable when run as a plain script from the repo.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import flwr as fl
from flwr.common import (
    Code,
    FitRes,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from esnfed import datasets, federated, metrics, topologies
from esnfed.esn import EchoStateNetwork, ridge_statistics, solve_readout


# ─────────────────────────────────────────────────────────── Flower client
class FederatedRidgeClient(fl.client.NumPyClient):
    """A bank/institution: holds a local ESN and a private slice of the series."""

    def __init__(self, esn: EchoStateNetwork, u: np.ndarray, y: np.ndarray):
        self.client = federated.Client(esn, u, y)

    def get_parameters(self, config):  # no persistent local model
        return []

    def fit(self, parameters, config):
        """Return the local ridge sufficient statistics (not the raw data)."""
        A, B = ridge_statistics(self.client.states(), self.client.targets())
        return [A, B], self.client.n_samples, {}


# ─────────────────────────────────────────────────────────── Flower strategy
class FederatedRidgeStrategy(FedAvg):
    """Aggregate by summing sufficient statistics and solving ridge once."""

    def __init__(self, readout_dim, n_outputs, ridge, evaluate_fn=None, **kwargs):
        super().__init__(**kwargs)
        self.readout_dim = readout_dim
        self.n_outputs = n_outputs
        self.ridge = ridge
        self._evaluate_fn = evaluate_fn

    def aggregate_fit(self, server_round, results, failures):
        A = np.zeros((self.readout_dim, self.readout_dim))
        B = np.zeros((self.readout_dim, self.n_outputs))
        for _proxy, fit_res in results:
            Ak, Bk = parameters_to_ndarrays(fit_res.parameters)
            A += Ak
            B += Bk
        W_out = solve_readout(A, B, self.ridge)
        return ndarrays_to_parameters([W_out]), {}

    def evaluate(self, server_round, parameters):
        if self._evaluate_fn is None:
            return None
        W_out = parameters_to_ndarrays(parameters)[0]
        return self._evaluate_fn(W_out)


# ─────────────────────────────────────────── minimal proxy for a Ray-free run
class _LocalClientProxy(ClientProxy):
    """A no-op ClientProxy: used only as an identity key by ``aggregate_fit``."""

    def get_properties(self, ins, timeout, group_id=None): ...
    def get_parameters(self, ins, timeout, group_id=None): ...
    def fit(self, ins, timeout, group_id=None): ...
    def evaluate(self, ins, timeout, group_id=None): ...
    def reconnect(self, ins, timeout, group_id=None): ...


def run_one_round(client_fns, strategy) -> tuple[np.ndarray, dict]:
    """Drive a single federated round through the real Flower classes."""
    results = []
    for cid, make_client in enumerate(client_fns):
        client = make_client()
        params, n, _ = client.fit([], {})
        fit_res = FitRes(
            status=Status(Code.OK, ""),
            parameters=ndarrays_to_parameters(params),
            num_examples=n,
            metrics={},
        )
        results.append((_LocalClientProxy(str(cid)), fit_res))
    aggregated, _ = strategy.aggregate_fit(1, results, [])
    W_out = parameters_to_ndarrays(aggregated)[0]
    loss, eval_metrics = strategy.evaluate(1, aggregated)
    return W_out, eval_metrics


def build(n_clients: int = 8, seed: int = 0):
    """Set up the federated counterparty-risk forecasting task."""
    u, y = datasets.load_ted_spread()  # real TED spread, normalised
    u_tr, y_tr, u_te, y_te = datasets.split(u, y, 0.7)
    parts = datasets.partition_iid(u_tr, y_tr, n_clients, rng=seed)

    W = topologies.random_reservoir(200, density=0.1, rng=seed)
    esn_kw = dict(spectral_radius=0.9, leaking_rate=0.5, washout=100, ridge=1e-7)
    clients, ref = federated.make_shared_clients(W, parts, input_seed=0, esn_kwargs=esn_kw)

    washout = ref.washout
    Z_test = ref.harvest(u_te)[washout:]
    y_eval = np.atleast_2d(y_te).reshape(-1, 1)[washout:]

    def evaluate_fn(W_out):
        err = metrics.nrmse(y_eval, Z_test @ W_out)
        return float(err), {"nrmse": float(err)}

    client_fns = [
        (lambda c=c: FederatedRidgeClient(c.esn, c.u, c.y)) for c in clients
    ]
    strategy = FederatedRidgeStrategy(
        readout_dim=ref.readout_dim, n_outputs=1, ridge=ref.ridge,
        evaluate_fn=evaluate_fn,
    )
    return client_fns, strategy, (clients, ref, u_te, y_te)


def main():
    print("[flower] federated counterparty-risk (TED spread) forecasting")
    client_fns, strategy, (clients, ref, u_te, y_te) = build(n_clients=8)
    W_out, eval_metrics = run_one_round(client_fns, strategy)
    print(f"  clients: {len(client_fns)} simulated institutions")
    print(f"  global test NRMSE (Flower-routed): {eval_metrics['nrmse']:.4f}")

    # Sanity: identical to the direct exact federated ridge.
    W_direct = federated.federated_ridge(clients, ref)
    print(f"  max |Flower - direct federated_ridge|: "
          f"{np.max(np.abs(W_out - W_direct)):.2e}")


if __name__ == "__main__":
    main()
