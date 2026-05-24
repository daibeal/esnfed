"""Incremental / streaming ridge and the recursive-least-squares readout.

Because the readout depends on the data only through ``A = Z^T Z`` and
``B = Z^T Y``, training can be made incremental (and continual / federated) by
accumulating those sums -- identical to batch ridge.

Run:  python examples/streaming_demo.py
"""
import numpy as np

from esnfed import EchoStateNetwork, datasets, metrics, topologies
from esnfed.streaming import RLSReadout, StreamingRidge


def main() -> None:
    rng = np.random.default_rng(0)
    u, y = datasets.narma10(5000, rng=rng)
    u_tr, y_tr, u_te, y_te = datasets.split(u, y, 0.7)
    W = topologies.random_reservoir(150, density=0.1, rng=rng)
    esn = EchoStateNetwork(1, 1, W, spectral_radius=0.9, washout=100, ridge=1e-6)
    Z_tr, Y_tr = esn.harvest(u_tr)[esn.washout:], y_tr[esn.washout:]
    Z_te, y_ev = esn.harvest(u_te)[esn.washout:], y_te[esn.washout:]
    nr = lambda W_out: metrics.nrmse(y_ev, Z_te @ W_out)

    # stream the training states in chunks -> identical to batch ridge
    acc = StreamingRidge(esn.readout_dim, esn.n_outputs, ridge=esn.ridge)
    for Zc, Yc in zip(np.array_split(Z_tr, 5), np.array_split(Y_tr, 5)):
        acc.update(Zc, Yc)
    print(f"streaming ridge    NRMSE = {nr(acc.readout()):.3f}   (n_seen={acc.n_seen})")

    # two clients accumulate locally, then merge: exactly the federated sum
    half = len(Z_tr) // 2
    a = StreamingRidge(esn.readout_dim, esn.n_outputs, esn.ridge).update(Z_tr[:half], Y_tr[:half])
    b = StreamingRidge(esn.readout_dim, esn.n_outputs, esn.ridge).update(Z_tr[half:], Y_tr[half:])
    a.merge(b)
    print(f"merged (federated) NRMSE = {nr(a.readout()):.3f}")

    # recursive least squares: per-sample online updates (Sherman-Morrison)
    rls = RLSReadout(esn.readout_dim, esn.n_outputs, ridge=esn.ridge).update_batch(Z_tr, Y_tr)
    print(f"RLS online         NRMSE = {nr(rls.readout()):.3f}")


if __name__ == "__main__":
    main()
