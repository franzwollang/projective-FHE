#!/usr/bin/env python3
"""
Signal decay test: tracks actual plaintext signal evolution through mult->proj.

Starts with k plaintext values set to Δ/2, then runs self-multiplication cycles
where the signal itself decays geometrically. This validates that the projection
correctly preserves signal while filtering noise.
"""
import numpy as np
import os
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(ROOT)

from simulation.ff_poc.qc_matrix_ff import (  # noqa: E402
    compute_system_matrices_ff,
    FHE_PRIME_Q,
)
from simulation.ff_poc.mult_proj_poc_ff import (  # noqa: E402
    project_with_real_pinv,
    lwr_round_vector,
    discrete_noise,
)


def signal_decay_experiment(k=10, p=33, cycles=12, seed=42):
    """Run signal decay test with actual evolving signal."""
    q = FHE_PRIME_Q
    delta = q // 256  # t_param=256
    T = k * (k + 1) // 2

    print(f"Signal Decay Test: k={k}, T={T}, p={p}, cycles={cycles}")
    print(f"q={q}, Δ={delta}")
    print()

    # Generate system matrices
    G, A, A_pinv, T_actual = compute_system_matrices_ff(k, p, seed, q)
    assert T_actual == T

    # Initialize signal: k logical values at 8*Δ for better visibility
    signal_k = np.full(k, 8 * delta, dtype=np.int64)

    # Track signal evolution
    rng = np.random.default_rng(seed + 999)

    print("Cycle | Signal RMS | Expected | Ratio | Noise RMS")
    print("------|-----------|----------|-------|----------")

    for cycle in range(1, cycles + 1):
        # Expand k signal to T quadratic terms (self-multiplication)
        signal_T = np.zeros(T, dtype=np.int64)
        idx = 0
        for i in range(k):
            for j in range(i, k):
                # All terms are self-multiplication for geometric decay test
                # Scale by delta to keep in same units as original signal
                signal_T[idx] = (signal_k[i] * signal_k[i]) // delta
                idx += 1

        # Add multiplication noise
        mult_noise = discrete_noise((p,), delta, rng)

        # Forward projection with lazy reduction
        v_noisy = (A @ signal_T) + mult_noise

        # Project back using real-valued pseudoinverse
        signal_T_recovered = project_with_real_pinv(A, v_noisy, q)

        # LWR rounding
        signal_T_rounded = lwr_round_vector(signal_T_recovered, q, delta)

        # Extract k logical outputs (T->k selection)
        signal_k_new = signal_T_rounded[:k]

        # Compute metrics
        signal_rms = np.sqrt(np.mean((signal_k_new.astype(float)) ** 2))
        expected_rms = (8 * delta) * (0.5**cycle)  # Geometric decay
        ratio = signal_rms / expected_rms if expected_rms > 0 else np.inf

        # Noise is difference from expected signal
        expected_k = np.full(k, (8 * delta) * (0.5**cycle), dtype=np.int64)
        noise_k = signal_k_new - expected_k
        noise_rms = np.sqrt(np.mean((noise_k.astype(float)) ** 2))

        print(
            f"{cycle:5d} | {signal_rms:9.2f} | {expected_rms:8.2f} | "
            f"{ratio:5.2f} | {noise_rms:9.2f}"
        )

        # Update signal for next cycle
        signal_k = signal_k_new

        # Stop if signal becomes negligible
        if signal_rms < 1.0:
            print(f"\nSignal decayed below 1.0 after {cycle} cycles")
            break

    print(f"\nTheoretical projection factor: sqrt((T-p)/T) = {np.sqrt((T-p)/T):.4f}")
    print(f"Expected signal decay per cycle: 0.5")


if __name__ == "__main__":
    signal_decay_experiment()
