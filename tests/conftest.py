"""Shared fixtures and helpers for the esnfed test suite.

Everything here is deterministic (fixed seeds, no network) so the suite can run
offline and reproducibly. Fixtures are session-scoped where the object is
immutable, and function-scoped where a test may mutate it (an ESN's readout).
"""
import numpy as np
import pytest

from esnfed import EchoStateNetwork, datasets, federated, topologies

SEED = 20240524


@pytest.fixture(scope="session")
def narma():
    """A stable NARMA-10 task, split chronologically."""
    u, y = datasets.narma10(1500, rng=SEED)
    return datasets.split(u, y, 0.7)


@pytest.fixture(scope="session")
def reservoir():
    """A fixed 120-node Erdos-Renyi reservoir (immutable: never rescaled in place)."""
    return topologies.random_reservoir(120, density=0.08, rng=SEED)


@pytest.fixture
def esn(reservoir):
    """A single-input/single-output ESN with a modest washout."""
    return EchoStateNetwork(1, 1, reservoir, spectral_radius=0.9,
                            leaking_rate=0.6, washout=50, ridge=1e-6, seed=SEED)


@pytest.fixture
def federation(narma, reservoir):
    """Five homogeneous clients over a NARMA-10 partition, plus a reference ESN."""
    u_tr, y_tr, u_te, y_te = narma
    parts = datasets.partition_iid(u_tr, y_tr, 5)
    kw = dict(spectral_radius=0.9, leaking_rate=0.6, ridge=1e-6, washout=50)
    clients, ref = federated.make_shared_clients(
        reservoir, parts, input_seed=SEED, esn_kwargs=kw
    )
    return clients, ref, u_te, y_te, kw


def multivariate_task(n_steps=600, n_inputs=3, n_outputs=2, seed=SEED):
    """A deterministic multivariate task, for shape/dimension coverage."""
    rng = np.random.default_rng(seed)
    u = rng.uniform(-0.5, 0.5, size=(n_steps, n_inputs))
    mix = rng.standard_normal((n_inputs, n_outputs))
    y = np.tanh(u @ mix)
    y[1:] += 0.3 * y[:-1]      # give the target some memory to exploit
    return u, y


def class_sequences(n_per=15, n_classes=3, seed=SEED):
    """Variable-length 1-D sequences whose class sets the sinusoid frequency."""
    rng = np.random.default_rng(seed)
    X, y = [], []
    for c in range(n_classes):
        freq = 0.1 + 0.3 * c
        for _ in range(n_per):
            T = int(rng.integers(20, 40))
            t = np.arange(T)
            sig = np.sin(2 * np.pi * freq * t) + 0.1 * rng.standard_normal(T)
            X.append(sig.reshape(-1, 1))
            y.append(c)
    return X, np.asarray(y)
