"""esnfed - Echo State Networks for Federated Learning.

A small, dependency-light research library accompanying the Final Degree Project
*Ensemble of Recurrent Networks for Federated Learning* (ETSINF, UPV).

Modules
-------
esn          Echo State Network (reservoir + ridge readout); supports
             heterogeneous leaking rates and multi-type node nonlinearities.
deep         Hierarchical / deep Echo State Networks (stacked reservoirs).
topologies   Reservoir topology generators (random, small-world, scale-free,
             ring) + per-node leaking-rate and mixed-activation generators.
datasets     Benchmark sequential tasks (NARMA-10, Mackey-Glass, Lorenz, sine).
metrics      Error metrics (NRMSE, MSE, memory capacity).
federated    Federated strategies (FedAvg, exact federated ridge, ensemble, alignment).
classification  Sequence classification + its exact federated / ensemble variants.
"""

from .esn import EchoStateNetwork
from .deep import DeepEchoStateNetwork
from .metrics import nrmse, mse, rmse
from . import (
    topologies, datasets, federated, metrics, interop, viz,
    classification, deep, llm_orchestration,
)

__all__ = [
    "EchoStateNetwork",
    "DeepEchoStateNetwork",
    "nrmse",
    "mse",
    "rmse",
    "topologies",
    "datasets",
    "federated",
    "metrics",
    "interop",
    "viz",
    "classification",
    "deep",
    "llm_orchestration",
]

__version__ = "1.5.0"
