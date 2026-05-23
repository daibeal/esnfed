# ---
# jupyter:
#   jupytext:
#     text_representation: {extension: .py, format_name: percent}
#   kernelspec: {display_name: Python (Pyodide), language: python, name: python}
# ---

# %% [markdown]
# # Federated strategies — live
#
# Runs in your browser (JupyterLite + Pyodide). The first cell installs the
# package and Matplotlib for the plot.

# %%
%pip install -q esnfed matplotlib

# %%
import numpy as np
from esnfed import EchoStateNetwork, datasets, topologies, federated, metrics

u, y = datasets.narma10(4000, rng=0)
u_tr, y_tr, u_te, y_te = datasets.split(u, y, 0.7)
W = topologies.random_reservoir(150, density=0.1, rng=0)
esn_kw = dict(spectral_radius=0.9, leaking_rate=1.0, washout=100)

# %% [markdown]
# ## Federated vs. local as the federation grows
#
# Exact federated ridge stays flat; local-only training degrades as each client
# is left with less data.

# %%
for nc in [1, 2, 5, 10, 20]:
    parts = datasets.partition_iid(u_tr, y_tr, nc, rng=0)
    clients, ref = federated.make_shared_clients(W, parts, input_seed=0, esn_kwargs=esn_kw)
    W_out = federated.federated_ridge(clients, ref)
    fed = metrics.nrmse(y_te[100:], ref.harvest(u_te)[100:] @ W_out)
    federated.train_local(clients)
    loc = np.mean([metrics.nrmse(y_te[100:], c.esn.predict(u_te)[100:]) for c in clients])
    print(f"{nc:3d} clients | federated {fed:.4f} | local-mean {loc:.4f}")

# %% [markdown]
# ## FedAvg convergence
#
# Iterative FedAvg on the readout — slower than the closed-form solution because
# the reservoir-state covariance is ill-conditioned.

# %%
import matplotlib.pyplot as plt

parts = datasets.partition_iid(u_tr, y_tr, 10, rng=0)
clients, ref = federated.make_shared_clients(W, parts, input_seed=0, esn_kwargs=esn_kw)
exact = metrics.nrmse(y_te[100:], ref.harvest(u_te)[100:] @ federated.federated_ridge(clients, ref))
_, history = federated.fedavg(clients, ref, u_te, y_te, rounds=80, local_epochs=5, lr=0.9)

plt.figure(figsize=(6, 3.5))
plt.plot(range(1, len(history) + 1), history, label="FedAvg")
plt.axhline(exact, color="C1", ls="--", label="exact federated ridge")
plt.xlabel("round"); plt.ylabel("test NRMSE"); plt.legend(); plt.title("FedAvg convergence")
plt.show()

# %% [markdown]
# ## Structural alignment (heterogeneous → shared)
#
# Interpolate heterogeneous reservoirs toward a common target; parameter
# aggregation only becomes competitive as the structures coincide.

# %%
res = [topologies.random_reservoir(150, density=0.1, rng=i) for i in range(5)]
target = topologies.random_reservoir(150, density=0.1, rng=99)
recs = federated.structural_alignment(res, target, parts, u_te, y_te,
                                      alphas=np.linspace(0, 1, 6), esn_kwargs=esn_kw)
for r in recs:
    print(f"alpha={r['alpha']:.1f} | ensemble {r['ensemble_nrmse']:.3f} "
          f"| aggregation {r['fedavg_nrmse']:.3f} | heterogeneity {r['heterogeneity']:.1f}")
