# ---
# jupyter:
#   jupytext:
#     text_representation: {extension: .py, format_name: percent}
#   kernelspec: {display_name: Python (Pyodide), language: python, name: python}
# ---

# %% [markdown]
# # esnfed — live quickstart
#
# A **real Python environment in your browser** (JupyterLite + Pyodide) — no
# install, no server. Run each cell with **Shift+Enter**, edit anything, re-run.
# The first cell installs `esnfed` (~20–40 s the first time).

# %%
%pip install -q esnfed

# %% [markdown]
# ## 1. Train a single Echo State Network on NARMA-10

# %%
import numpy as np
from esnfed import EchoStateNetwork, datasets, topologies, metrics

u, y = datasets.narma10(1500, rng=0)
u_tr, y_tr, u_te, y_te = datasets.split(u, y, 0.7)

W = topologies.random_reservoir(150, density=0.1, rng=0)
esn = EchoStateNetwork(1, 1, W, spectral_radius=0.9, leaking_rate=1.0, washout=100)
esn.fit(u_tr, y_tr)
print("NARMA-10 test NRMSE:", round(metrics.nrmse(y_te[100:], esn.predict(u_te)[100:]), 4))

# %% [markdown]
# ## 2. Compare reservoir topologies
#
# The connectivity structure shapes the dynamics — try changing the sizes/params.

# %%
for kind in ["random", "small_world", "scale_free", "ring"]:
    Wk = topologies.make_reservoir(kind, 150, rng=0)
    e = EchoStateNetwork(1, 1, Wk, spectral_radius=0.9, washout=100).fit(u_tr, y_tr)
    print(f"{kind:12s} NRMSE = {metrics.nrmse(y_te[100:], e.predict(u_te)[100:]):.4f}")

# %% [markdown]
# ## 3. Federate it *exactly* across 5 clients
#
# Each client shares only its ridge sufficient statistics; the server sums and
# solves once — identical to pooling the data, but it never leaves the client.

# %%
from esnfed import federated

parts = datasets.partition_iid(u_tr, y_tr, n_clients=5, rng=0)
esn_kw = dict(spectral_radius=0.9, leaking_rate=1.0, washout=100)
clients, ref = federated.make_shared_clients(W, parts, input_seed=0, esn_kwargs=esn_kw)

W_out = federated.federated_ridge(clients, ref)
Z_test = ref.harvest(u_te)[ref.washout:]
print("Federated NRMSE:", round(metrics.nrmse(y_te[ref.washout:], Z_test @ W_out), 4))

# %% [markdown]
# ## 4. Heterogeneous clients — prediction ensemble
#
# When each client has a *different* reservoir, parameters can't be averaged —
# but predictions can.

# %%
kinds = ["random", "small_world", "scale_free", "ring", "random"]
reservoirs = [topologies.make_reservoir(k, 150, rng=i) for i, k in enumerate(kinds)]
het = federated.make_heterogeneous_clients(reservoirs, parts, esn_kwargs=esn_kw)
federated.train_local(het)
pred = federated.ensemble_predict(het, u_te)
member = np.mean([metrics.nrmse(y_te[100:], c.esn.predict(u_te)[100:]) for c in het])
print("ensemble NRMSE :", round(metrics.nrmse(y_te[100:], pred[100:]), 4))
print("mean member    :", round(member, 4), "(ensemble should be lower)")

# %% [markdown]
# ## 5. Real data: the TED spread (counterparty risk)
#
# A real series ships with the package — the interbank/Treasury spread, a classic
# gauge of counterparty credit risk.

# %%
ru, ry = datasets.load_ted_spread()
ru_tr, ry_tr, ru_te, ry_te = datasets.split(ru, ry, 0.7)
re = EchoStateNetwork(1, 1, W, spectral_radius=0.9, leaking_rate=0.5, washout=100).fit(ru_tr, ry_tr)
print("TED-spread forecast NRMSE:", round(metrics.nrmse(ry_te[100:], re.predict(ru_te)[100:]), 4))

# %% [markdown]
# **Your turn** — change the topology, spectral radius, leaking rate or number of
# clients and re-run. Everything in the [docs](https://daibeal.github.io/esnfed/)
# works here.
