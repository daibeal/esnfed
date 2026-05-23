# Interoperability

`esnfed` is designed to *cooperate* with the established tools of the field, not
to replace them.

## ReservoirPy — design there, federate here

[ReservoirPy](https://reservoirpy.readthedocs.io) is the mature library for
building and tuning reservoir-computing models. `esnfed.interop` lifts a tuned
ReservoirPy reservoir into an `EchoStateNetwork`, ready for the federated
strategies.

```python
from reservoirpy.nodes import Reservoir
from esnfed import interop, federated, datasets

res = Reservoir(200, sr=0.9, lr=0.5, input_dim=1)   # design / tune in ReservoirPy

esn = interop.to_esn(res, n_inputs=1)               # -> esnfed EchoStateNetwork
W = interop.reservoir_matrix(res)                   # ... or just the matrix, to federate
```

Install with `pip install "esnfed[reservoirpy]"`.

## Flower — a real FL framework

The exact federated-ridge scheme maps cleanly onto
[Flower](https://flower.ai): each client's `fit` returns its ridge sufficient
statistics and a custom `Strategy` sums them and solves once. The example
`examples/flower_federated_ridge.py` shows it end to end, and verifies the
Flower-routed result equals `federated_ridge` to numerical precision.

```python
# examples/flower_federated_ridge.py
class FederatedRidgeClient(fl.client.NumPyClient): ...
class FederatedRidgeStrategy(FedAvg): ...   # aggregate_fit: sum A_k, B_k; solve once
```

Install with `pip install "esnfed[flower]"`.

## LLMs — FedResPrompt

The experimental [FedResPrompt](../fedresprompt.md) module uses an ESN as a
lightweight *prompt controller* for a frozen language model, with an optional
Hugging Face `transformers` backend (`pip install "esnfed[llm]"`).
