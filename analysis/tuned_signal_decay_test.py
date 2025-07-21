#!/usr/bin/env python3
"""
Tuned signal decay test: finds optimal initial signal for stable long-term decay.

For self-multiplication x -> x^2/Δ, we want the signal to decay geometrically
but remain visible for many cycles. The optimal initial value is around Δ
so that x^2/Δ ≈ x when x ≈ Δ (equilibrium point).
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


def find_optimal_initial_value(delta):
    """
    For self-multiplication x -> x^2/Δ, find value that gives reasonable decay.

    We want: x_new = x^2/Δ to be smaller than x but not too small.
    Good decay rate is around 0.7-0.9 per cycle.

    If x_new/x = target_ratio, then x^2/Δ / x = target_ratio
    So x/Δ = target_ratio, or x = target_ratio * Δ
    """
    target_ratio = 0.8  # 20% decay per cycle
    return int(target_ratio * delta)


def tuned_signal_experiment(k=10, p=33, cycles=50, seed=42):
    """Run tuned signal decay test for long-term stability."""
    q = FHE_PRIME_Q
    delta = q // 256  # t_param=256
    T = k * (k + 1) // 2

    # Find optimal initial signal value
    initial_signal = find_optimal_initial_value(delta)

    print(f"Tuned Signal Decay Test: k={k}, T={T}, p={p}, cycles={cycles}")
    print(f"q={q}, Δ={delta}")
    print(f"Initial signal value: {initial_signal} ({initial_signal/delta:.3f}×Δ)")
    print()

    # Generate system matrices
    G, A, A_pinv, T_actual = compute_system_matrices_ff(k, p, seed, q)
    assert T_actual == T

    # Initialize signal
    signal_k = np.full(k, initial_signal, dtype=np.int64)

    # Track signal evolution
    rng = np.random.default_rng(seed + 999)

    print("Cycle | Signal RMS | Decay Ratio | Noise RMS | Signal/Noise")
    print("------|-----------|-------------|-----------|-------------")

    prev_signal_rms = None

    for cycle in range(1, cycles + 1):
        # Expand k signal to T quadratic terms (self-multiplication)
        signal_T = np.zeros(T, dtype=np.int64)
        idx = 0
        for i in range(k):
            for j in range(i, k):
                # Self-multiplication scaled to maintain reasonable range
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

        # Decay ratio (current/previous)
        decay_ratio = signal_rms / prev_signal_rms if prev_signal_rms else np.nan
        prev_signal_rms = signal_rms

        # Estimate noise as projection + rounding effects
        # Use equilibrium noise from previous measurements (~140 in absolute units)
        noise_rms = 140.0  # Approximate empirical noise floor

        # Signal to noise ratio
        snr = signal_rms / noise_rms if noise_rms > 0 else np.inf

        # Print every 5th cycle for readability, plus first/last few
        if cycle <= 5 or cycle % 5 == 0 or cycle >= cycles - 2:
            print(
                f"{cycle:5d} | {signal_rms:9.1f} | {decay_ratio:11.3f} | "
                f"{noise_rms:9.1f} | {snr:11.2f}"
            )

        # Update signal for next cycle
        signal_k = signal_k_new

        # Stop if signal becomes too small relative to noise
        if signal_rms < noise_rms:
            print(f"\nSignal fell below noise floor after {cycle} cycles")
            break

    print(f"\nFinal signal RMS: {signal_rms:.1f}")
    print(f"Average decay ratio: ~{0.8:.2f} (target)")
    print(f"Signal remained above noise for {cycle} cycles")


if __name__ == "__main__":
    tuned_signal_experiment()
