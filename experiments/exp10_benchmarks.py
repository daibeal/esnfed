"""Experiment 10 - Validation on external benchmark datasets.

Validates the federated strategies on three real, well-known datasets, each with
a *natural* client split:

* Japanese Vowels (UCI 128) -- speaker identification (9 classes); one client per
  speaker (the client id *is* the label: extreme label skew).
* HAR Smartphones (UCI 240) -- activity recognition (6 classes); one client per
  subject (feature non-i.i.d., all labels present).
* A multivariate FRED panel -- forecasting the TED spread from several financial
  series; clients are time partitions (institutions).

Produces a 3-panel figure and a summary table. Needs network access (datasets are
downloaded and cached) and the optional accelerators help: ``esnfed[fast]``.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

import common
from esnfed import (EchoStateNetwork, classification as clf, datasets,
                    federated, metrics, topologies)

warnings.filterwarnings("ignore")
RNG = common.MASTER_SEED


def japanese_vowels():
    jv = datasets.load_japanese_vowels()
    W = topologies.random_reservoir(120, density=0.1, rng=RNG)
    esn = EchoStateNetwork(jv.n_features, jv.n_classes, W, spectral_radius=0.9,
                           leaking_rate=0.3, washout=0, ridge=1e-3, seed=0)
    Wc = clf.train_classifier(esn, jv.X_train, jv.y_train, jv.n_classes)
    acc_c = clf.accuracy(jv.y_test, clf.predict_labels(esn, jv.X_test, Wc))
    clients = datasets.group_clients(jv.X_train, jv.y_train, jv.groups_train)
    Wf = clf.federated_classifier(esn, clients, jv.n_classes)
    acc_f = clf.accuracy(jv.y_test, clf.predict_labels(esn, jv.X_test, Wf))
    local = np.mean([
        clf.accuracy(jv.y_test, clf.predict_labels(
            esn, jv.X_test, clf.train_classifier(esn, s, l, jv.n_classes)))
        for s, l in clients])
    exact = bool(np.allclose(Wc, Wf, atol=1e-8))
    print(f"[JV] centralized={acc_c:.3f} federated={acc_f:.3f} local={local:.3f} "
          f"exact={exact}")
    return dict(centralized=acc_c, federated=acc_f, local=float(local), exact=exact)


def har():
    h = datasets.load_har()
    clients = datasets.group_clients(h.X_train, h.y_train, h.groups_train)
    # shared reservoir -> exact federated == centralized
    W = topologies.random_reservoir(120, density=0.1, rng=RNG)
    esn = EchoStateNetwork(9, 6, W, spectral_radius=0.9, leaking_rate=0.3,
                           washout=0, ridge=1e-2, seed=0)
    Wf = clf.federated_classifier(esn, clients, 6)
    acc_f = clf.accuracy(h.y_test, clf.predict_labels(esn, h.X_test, Wf))
    # heterogeneous reservoirs -> prediction ensemble vs local-only
    kinds = ["random", "small_world", "scale_free", "ring"]
    members, local = [], []
    for i, (s, l) in enumerate(clients):
        Wi = topologies.make_reservoir(kinds[i % 4], 120, rng=i)
        e = EchoStateNetwork(9, 6, Wi, spectral_radius=0.9, leaking_rate=0.3,
                             washout=0, ridge=1e-2, seed=i)
        Wl = clf.train_classifier(e, s, l, 6)
        members.append((e, Wl))
        local.append(clf.accuracy(h.y_test, clf.predict_labels(e, h.X_test, Wl)))
    acc_ens = clf.accuracy(h.y_test, clf.ensemble_classify(members, h.X_test))
    print(f"[HAR] federated={acc_f:.3f} ensemble={acc_ens:.3f} "
          f"local={np.mean(local):.3f}")
    return dict(federated=acc_f, ensemble=acc_ens, local=float(np.mean(local)))


def fred_panel():
    series = ["TEDRATE", "VIXCLS", "DGS10", "DFF"]
    u, y = datasets.load_fred_matrix(series)
    u_tr, y_tr, u_te, y_te = datasets.split(u, y, 0.7)
    W = topologies.random_reservoir(300, density=0.1, rng=RNG)
    esn_kw = dict(spectral_radius=0.9, leaking_rate=0.5, washout=100, ridge=1e-5)
    counts, fed, loc = [1, 5, 10, 20], [], []
    for nc in counts:
        parts = datasets.partition_iid(u_tr, y_tr, nc, rng=RNG)
        clients, ref = federated.make_shared_clients(W, parts, input_seed=0,
                                                     esn_kwargs=esn_kw)
        Wf = federated.federated_ridge(clients, ref)
        fed.append(metrics.nrmse(y_te[ref.washout:],
                                 ref.harvest(u_te)[ref.washout:] @ Wf))
        federated.train_local(clients)
        loc.append(float(np.median([
            metrics.nrmse(y_te[100:], c.esn.predict(u_te)[100:]) for c in clients])))
    print(f"[FRED] d_in={u.shape[1]} federated={fed} local={loc}")
    return dict(counts=counts, federated=fed, local=loc, d_in=u.shape[1])


def plot(jv, hr, fr):
    common.set_style()
    fig, ax = common.plt.subplots(1, 3, figsize=(11.5, 3.6))
    P = common.PALETTE
    ax[0].bar(["central", "federated", "local"],
              [jv["centralized"], jv["federated"], jv["local"]],
              color=[P["local"], P["federated_ridge"], P["fedavg"]])
    ax[0].axhline(1 / 9, color="grey", ls="--", lw=1, label="chance")
    ax[0].set_ylim(0, 1.05); ax[0].set_ylabel("test accuracy")
    ax[0].set_title("Japanese Vowels (9 speakers)"); ax[0].legend()
    ax[1].bar(["federated", "ensemble", "local"],
              [hr["federated"], hr["ensemble"], hr["local"]],
              color=[P["federated_ridge"], P["ensemble"], P["fedavg"]])
    ax[1].axhline(1 / 6, color="grey", ls="--", lw=1, label="chance")
    ax[1].set_ylim(0, 1.05); ax[1].set_ylabel("test accuracy")
    ax[1].set_title("HAR (21 subjects)"); ax[1].legend()
    ax[2].plot(fr["counts"], fr["federated"], marker="o", color=P["federated_ridge"],
               label="federated")
    ax[2].plot(fr["counts"], fr["local"], marker="s", color=P["fedavg"],
               label="local (median)")
    ax[2].set_yscale("log"); ax[2].set_xlabel("institutions")
    ax[2].set_ylabel("test NRMSE (log)")
    ax[2].set_title(f"FRED panel (d={fr['d_in']})"); ax[2].legend()
    fig.suptitle("Validation on external benchmark datasets", y=1.02)
    common.save_figure(fig, "exp10_benchmarks")


def main():
    print("[exp10] benchmark-dataset validation")
    jv, hr, fr = japanese_vowels(), har(), fred_panel()
    plot(jv, hr, fr)
    tbl = pd.DataFrame([
        ["Japanese Vowels", "speaker ID (9)", "9 speakers",
         f"{jv['federated']:.3f}", f"{jv['local']:.3f}",
         "federated $=$ centralized (exact)"],
        ["HAR Smartphones", "activity (6)", "21 subjects",
         f"{hr['federated']:.3f}", f"{hr['local']:.3f}",
         f"ensemble {hr['ensemble']:.3f}"],
        ["FRED panel ($d{=}4$)", "TED forecast", "10 inst.",
         f"{fr['federated'][2]:.3f}", f"{fr['local'][2]:.3f}",
         "NRMSE; local collapses"],
    ], columns=["Dataset", "Task", "Clients", "Federated", "Local", "Note"])
    common.save_latex_table(
        tbl, "exp10_benchmarks",
        caption="Validation of the federated strategies on external benchmark "
                "datasets with natural client splits. Accuracy (higher better) for "
                "the two classification tasks; NRMSE (lower better) for the FRED "
                "forecasting panel.",
        label="tab:exp10-benchmarks")
    print("[exp10] done")


if __name__ == "__main__":
    main()
