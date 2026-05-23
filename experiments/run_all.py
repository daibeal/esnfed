"""Run all experiments in order, regenerating every figure and table.

    python experiments/run_all.py
"""
from __future__ import annotations

import time

import exp1_esn_sweep
import exp2_topology
import exp3_fedavg
import exp4_ensemble
import exp5_alignment


def main():
    t0 = time.time()
    for mod in (exp1_esn_sweep, exp2_topology, exp3_fedavg, exp4_ensemble, exp5_alignment):
        t = time.time()
        mod.main()
        print(f"    ({time.time() - t:.1f}s)\n")
    print(f"All experiments finished in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
