"""Test the Flower integration example (skipped if flwr is absent).

Verifies that routing the exact federated-ridge scheme through the real Flower
client/strategy classes reproduces the direct result bit-for-bit.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("flwr")

from esnfed import federated  # noqa: E402

_EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "flower_federated_ridge.py"
_spec = importlib.util.spec_from_file_location("flower_federated_ridge", _EXAMPLE)
flower_example = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(flower_example)


def test_flower_matches_exact_federated_ridge():
    client_fns, strategy, (clients, ref, u_te, y_te) = flower_example.build(
        n_clients=6, seed=1
    )
    W_flower, eval_metrics = flower_example.run_one_round(client_fns, strategy)
    W_direct = federated.federated_ridge(clients, ref)
    # Flower-routed aggregation must equal the direct exact federated ridge.
    assert np.allclose(W_flower, W_direct, atol=1e-9)
    # And it should be a sensible (sub-trivial) forecast of the TED spread.
    assert 0.0 < eval_metrics["nrmse"] < 1.0
