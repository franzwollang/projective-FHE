#!/usr/bin/env python3
"""Geometric decay test for mult->proj pipeline.

Starts with k plaintext slots set to Δ/2, runs N self-multiply→project cycles and
prints the RMS magnitude of the logical ciphertext after each cycle.  We expect
a roughly geometric decay governed by sqrt((T-p)/T).
"""
import numpy as np
import os
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(ROOT)

from simulation.ff_poc.mult_proj_poc_ff import run_ff_experiment  # noqa: E402
from simulation.ff_poc.qc_matrix_ff import FHE_PRIME_Q  # noqa: E402


def main():
    k = 10
    p = 33  # 60 % of T -> under-determined
    cycles = 12
    delta = FHE_PRIME_Q // 256  # default t_param=256 in run_ff_experiment

    # run experiment with deterministic seed so x_true = Δ/2 pattern
    logger, _ = run_ff_experiment(
        k=k, p=p, t_param=256, num_cycles=cycles, seed=12345, verbose=False
    )

    # Extract after-projection RMS per cycle (in Δ units)
    rms = [np.sqrt(row[2]) for row in logger.data]

    print("Cycle | RMS noise (Δ units)")
    print("------|--------------------")
    for c, val in enumerate(rms, 1):
        print(f"{c:5d} | {val:8.4f}")

    # Expected geometric factor
    T = k * (k + 1) // 2
    geom = np.sqrt((T - p) / T)
    print(f"\nExpected geometric decay factor per cycle: sqrt((T-p)/T) = {geom:.4f}")


if __name__ == "__main__":
    main()
