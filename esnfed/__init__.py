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
datasets     Benchmark tasks (NARMA-10, Mackey-Glass, Lorenz) and real / external
             datasets (TED spread, multivariate FRED panels, Japanese Vowels, HAR).
metrics      Error metrics (NRMSE, RMSE, MSE, R^2).
federated    Federated strategies (FedAvg, exact federated ridge, ensemble, alignment).
classification  Sequence classification + its exact federated / ensemble variants.
privacy      Differential privacy + secure aggregation for the federated statistics.
streaming    Incremental / streaming ridge (accumulate A, B; RLS online updates).
"""

from .esn import EchoStateNetwork
from .deep import DeepEchoStateNetwork
from .metrics import nrmse, mse, rmse
from . import (
    topologies, datasets, federated, metrics, interop, viz,
    classification, deep, llm_orchestration, privacy, streaming,
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
    "privacy",
    "streaming",
]

__version__ = "1.6.0"
