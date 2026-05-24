"""Differential privacy and secure aggregation for federated ridge.

The federated readout is built from the additive sufficient statistics
``(A_k, B_k)``; this example hardens that exchange two ways:

* secure aggregation -- the server only sees the masked sum (result unchanged);
* differential privacy -- a formal (epsilon, delta) guarantee, at an accuracy cost.

Run:  python examples/privacy_demo.py
"""
import numpy as np

from esnfed import datasets, federated, metrics, topologies
from esnfed.privacy import PrivacyConfig


def main() -> None:
    rng = np.random.default_rng(0)
    u, y = datasets.narma10(6000, rng=rng)
    u_tr, y_tr, u_te, y_te = datasets.split(u, y, 0.7)
    parts = datasets.partition_iid(u_tr, y_tr, n_clients=4)
    W = topologies.random_reservoir(60, density=0.1, rng=rng)
    clients, ref = federated.make_shared_clients(
        W, parts, input_seed=0,
        esn_kwargs=dict(spectral_radius=0.9, washout=100, ridge=1e-6),
    )
    Z_te = ref.harvest(u_te)[ref.washout:]
    y_ev = y_te[ref.washout:]
    nr = lambda W_out: metrics.nrmse(y_ev, Z_te @ W_out)

    print(f"exact federated ridge   NRMSE = {nr(federated.federated_ridge(clients, ref)):.3f}")

    # secure aggregation: identical result; the server never sees a client's stats
    W_secure = federated.federated_ridge_secure(clients, ref, seed=0)
    print(f"secure aggregation      NRMSE = {nr(W_secure):.3f}   (== exact)")

    # (epsilon, delta)-DP: accuracy degrades as the privacy budget tightens
    # (averaged over a few noise draws; DP is not exact)
    print("trivial predictor       NRMSE = 1.000")
    for eps in (10.0, 2.0, 0.5):
        errs = [nr(federated.federated_ridge_dp(
                    clients, ref,
                    PrivacyConfig(epsilon=eps, delta=1e-5, clip_state=5.0,
                                  clip_target=1.0, seed=s)))
                for s in range(5)]
        print(f"DP  epsilon={eps:>4}        NRMSE = {np.mean(errs):.3f}")


if __name__ == "__main__":
    main()
