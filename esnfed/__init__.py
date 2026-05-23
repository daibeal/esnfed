"""esnfed - Echo State Networks for Federated Learning.

A small, dependency-light research library accompanying the Final Degree Project
*Ensemble of Recurrent Networks for Federated Learning* (ETSINF, UPV).

Modules
-------
esn          Echo State Network (reservoir + ridge readout).
topologies   Reservoir topology generators (random, small-world, scale-free, ring).
datasets     Benchmark sequential tasks (NARMA-10, Mackey-Glass, Lorenz, sine).
metrics      Error metrics (NRMSE, MSE, memory capacity).
federated    Federated strategies (FedAvg, exact federated ridge, ensemble, alignment).
"""

from .esn import EchoStateNetwork
from .metrics import nrmse, mse, rmse
from . import topologies, datasets, federated, metrics, interop, viz, llm_orchestration

__all__ = [
    "EchoStateNetwork",
    "nrmse",
    "mse",
    "rmse",
    "topologies",
    "datasets",
    "federated",
    "metrics",
    "interop",
    "viz",
    "llm_orchestration",
]

__version__ = "1.2.1"
