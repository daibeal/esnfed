"""Sequence classification with Echo State Networks, and its federated variants.

For classification each input is a (possibly variable-length) sequence that maps
to a single class label. The reservoir turns each sequence into a fixed feature
vector (the mean or last extended state); a ridge readout on one-hot targets then
classifies by argmax. Because the readout is again a ridge regression, the exact
federated scheme of :mod:`esnfed.federated` carries over unchanged: clients
exchange the summed statistics ``A = F^T F`` and ``B = F^T Y`` and the server
solves once, recovering the pooled classifier exactly.
"""
from __future__ import annotations

import numpy as np

from .esn import EchoStateNetwork, ridge_statistics, solve_readout


def one_hot(labels, n_classes: int) -> np.ndarray:
    labels = np.asarray(labels, dtype=int)
    Y = np.zeros((labels.size, n_classes))
    Y[np.arange(labels.size), labels] = 1.0
    return Y


def reservoir_features(esn: EchoStateNetwork, sequences, pool: str = "mean") -> np.ndarray:
    """Map each sequence to a fixed feature vector via the reservoir.

    Parameters
    ----------
    sequences
        Iterable of arrays of shape ``(T_i, n_inputs)`` (lengths may differ).
    pool
        ``"mean"`` (default) averages the extended states over time; ``"last"``
        takes the final state.
    """
    feats = []
    for seq in sequences:
        Z = esn.harvest(np.atleast_2d(seq).reshape(-1, esn.n_inputs))
        if pool == "mean":
            feats.append(Z.mean(axis=0))
        elif pool == "last":
            feats.append(Z[-1])
        else:
            raise ValueError("pool must be 'mean' or 'last'")
    return np.asarray(feats, dtype=float)


def class_statistics(esn, sequences, labels, n_classes, pool="mean"):
    """Ridge sufficient statistics ``(A, B)`` for a set of labelled sequences."""
    F = reservoir_features(esn, sequences, pool)
    return ridge_statistics(F, one_hot(labels, n_classes))


def train_classifier(esn, sequences, labels, n_classes, pool="mean") -> np.ndarray:
    """Centralised classifier readout (ridge on reservoir features)."""
    A, B = class_statistics(esn, sequences, labels, n_classes, pool)
    return solve_readout(A, B, esn.ridge)


def federated_classifier(esn, client_data, n_classes, pool="mean") -> np.ndarray:
    """Exact federated classifier: sum each client's ``(A_k, B_k)`` and solve once.

    ``client_data`` is a list of ``(sequences, labels)`` pairs. The result equals
    centralised training on the pooled data, but no client shares raw sequences.
    """
    d = esn.readout_dim
    A = np.zeros((d, d))
    B = np.zeros((d, n_classes))
    for sequences, labels in client_data:
        Ak, Bk = class_statistics(esn, sequences, labels, n_classes, pool)
        A += Ak
        B += Bk
    return solve_readout(A, B, esn.ridge)


def predict_labels(esn, sequences, W_out, pool="mean") -> np.ndarray:
    """Predict class labels (argmax of the readout) for each sequence."""
    F = reservoir_features(esn, sequences, pool)
    return np.argmax(F @ W_out, axis=1)


def predict_proba(esn, sequences, W_out, pool="mean") -> np.ndarray:
    """Softmax class probabilities for each sequence."""
    F = reservoir_features(esn, sequences, pool)
    logits = F @ W_out
    logits -= logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    return p / p.sum(axis=1, keepdims=True)


def ensemble_classify(members, sequences, pool="mean") -> np.ndarray:
    """Average the class probabilities of heterogeneous member classifiers.

    ``members`` is a list of ``(esn, W_out)`` pairs (each may have its own
    reservoir); predictions are combined by averaging softmax probabilities.
    """
    probs = None
    for esn, W_out in members:
        p = predict_proba(esn, sequences, W_out, pool)
        probs = p if probs is None else probs + p
    return np.argmax(probs, axis=1)


def accuracy(y_true, y_pred) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))
