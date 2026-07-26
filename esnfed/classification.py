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
    """One-hot encode integer class labels into a ``(n, n_classes)`` matrix."""
    if n_classes < 1:
        raise ValueError(f"n_classes must be >= 1, got {n_classes}")
    labels = np.asarray(labels)
    if labels.size and not np.issubdtype(labels.dtype, np.integer):
        rounded = np.rint(labels.astype(np.float64))
        if not np.allclose(rounded, labels.astype(np.float64)):
            raise ValueError("labels must be integer class indices")
        labels = rounded
    labels = labels.astype(int).ravel()
    if labels.size and (labels.min() < 0 or labels.max() >= n_classes):
        raise ValueError(
            f"labels must lie in [0, {n_classes - 1}], got range "
            f"[{labels.min()}, {labels.max()}]"
        )
    Y = np.zeros((labels.size, n_classes))
    Y[np.arange(labels.size), labels] = 1.0
    return Y


def reservoir_features(esn: EchoStateNetwork, sequences, pool: str = "mean",
                       washout: int = 0) -> np.ndarray:
    """Map each sequence to a fixed feature vector via the reservoir.

    Parameters
    ----------
    sequences
        Iterable of arrays of shape ``(T_i, n_inputs)`` (lengths may differ).
    pool
        ``"mean"`` (default) averages the extended states over time; ``"last"``
        takes the final state.
    washout
        Number of initial timesteps to drop from each sequence before pooling.
        This is deliberately **not** taken from ``esn.washout``: a regression
        washout (typically 100) is longer than a whole classification sequence
        (Japanese Vowels utterances are 7--29 frames), so inheriting it would
        discard every frame. Pass it explicitly when the sequences are long enough
        to afford a transient.
    """
    if washout < 0:
        raise ValueError(f"washout must be >= 0, got {washout}")
    feats = []
    for i, seq in enumerate(sequences):
        arr = np.asarray(seq)
        if arr.ndim == 1:
            arr = arr.reshape(-1, esn.n_inputs)
        Z = esn.harvest(arr)
        if washout >= Z.shape[0]:
            raise ValueError(
                f"sequence {i} has {Z.shape[0]} steps, too short for "
                f"washout={washout}"
            )
        Zw = Z[washout:]
        if pool == "mean":
            feats.append(Zw.mean(axis=0))
        elif pool == "last":
            feats.append(Zw[-1])
        else:
            raise ValueError("pool must be 'mean' or 'last'")
    if not feats:
        raise ValueError("no sequences given")
    return np.asarray(feats, dtype=float)


def class_statistics(esn, sequences, labels, n_classes, pool="mean", washout=0):
    """Ridge sufficient statistics ``(A, B)`` for a set of labelled sequences."""
    F = reservoir_features(esn, sequences, pool, washout)
    Y = one_hot(labels, n_classes)
    if Y.shape[0] != F.shape[0]:
        raise ValueError(
            f"got {F.shape[0]} sequences but {Y.shape[0]} labels"
        )
    return ridge_statistics(F, Y)


def train_classifier(esn, sequences, labels, n_classes, pool="mean",
                     washout=0) -> np.ndarray:
    """Centralised classifier readout (ridge on reservoir features)."""
    A, B = class_statistics(esn, sequences, labels, n_classes, pool, washout)
    return solve_readout(A, B, esn.ridge)


def federated_classifier(esn, client_data, n_classes, pool="mean",
                         washout=0) -> np.ndarray:
    """Exact federated classifier: sum each client's ``(A_k, B_k)`` and solve once.

    ``client_data`` is a list of ``(sequences, labels)`` pairs. The result equals
    centralised training on the pooled data, but no client shares raw sequences.
    """
    if not client_data:
        raise ValueError("no clients to aggregate")
    d = esn.readout_dim
    A = np.zeros((d, d))
    B = np.zeros((d, n_classes))
    for sequences, labels in client_data:
        Ak, Bk = class_statistics(esn, sequences, labels, n_classes, pool, washout)
        A += Ak
        B += Bk
    return solve_readout(A, B, esn.ridge)


def _check_readout(esn, W_out, n_features: int) -> np.ndarray:
    W_out = np.asarray(W_out, dtype=float)
    if W_out.ndim != 2 or W_out.shape[0] != n_features:
        raise ValueError(
            f"W_out must have shape ({n_features}, n_classes), got {W_out.shape}"
        )
    return W_out


def predict_labels(esn, sequences, W_out, pool="mean", washout=0) -> np.ndarray:
    """Predict class labels (argmax of the readout) for each sequence."""
    F = reservoir_features(esn, sequences, pool, washout)
    return np.argmax(F @ _check_readout(esn, W_out, F.shape[1]), axis=1)


def predict_proba(esn, sequences, W_out, pool="mean", washout=0) -> np.ndarray:
    """Softmax class probabilities for each sequence."""
    F = reservoir_features(esn, sequences, pool, washout)
    logits = F @ _check_readout(esn, W_out, F.shape[1])
    logits = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    return p / p.sum(axis=1, keepdims=True)


def ensemble_classify(members, sequences, pool="mean", washout=0) -> np.ndarray:
    """Average the class probabilities of heterogeneous member classifiers.

    ``members`` is a list of ``(esn, W_out)`` pairs (each may have its own
    reservoir); predictions are combined by averaging softmax probabilities.
    """
    if not members:
        raise ValueError("no ensemble members given")
    first_esn, first_W = members[0]
    probs = predict_proba(first_esn, sequences, first_W, pool, washout)
    for esn, W_out in members[1:]:
        p = predict_proba(esn, sequences, W_out, pool, washout)
        if p.shape != probs.shape:
            raise ValueError(
                f"members disagree on output shape: {probs.shape} vs {p.shape}"
            )
        probs = probs + p
    return np.argmax(probs, axis=1)


def accuracy(y_true, y_pred) -> float:
    """Fraction of sequences whose predicted class equals the true class."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred have different lengths: {y_true.size} vs "
            f"{y_pred.size}"
        )
    if y_true.size == 0:
        return float("nan")
    return float(np.mean(y_true == y_pred))
