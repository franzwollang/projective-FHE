#!/usr/bin/env python3
"""
Test signal evolution across varied initial magnitudes in packed ciphertext.
Explores behavior from 0.1Δ to 2.0Δ to understand pipeline dynamics.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from simulation.ff_poc.mult_proj_poc_ff import project_with_real_pinv, lwr_round_vector
from simulation.ff_poc.qc_matrix_ff import compute_system_matrices_ff


def centered(x):
    """Center values to range [-q/2, q/2) for display."""
    return ((x + FHE_PRIME_Q // 2) % FHE_PRIME_Q) - FHE_PRIME_Q // 2


from typing import Tuple


def discrete_uniform(shape: Tuple[int, ...], half_range: int, rng):
    """Generate discrete uniform noise in [-half_range, half_range]."""
    return rng.integers(-half_range, half_range + 1, size=shape, dtype=np.int64)


FHE_PRIME_Q = 65537


class PackedMockCipher:
    """A toy ciphertext packing n_slots coefficients per polynomial."""

    def __init__(self, plain: np.ndarray, rng: np.random.Generator):
        # plain shape: (k, n_slots)
        self.plain = plain % FHE_PRIME_Q
        noise = discrete_uniform(plain.shape, 20, rng)  # small mask
        self.ct = (self.plain + noise) % FHE_PRIME_Q
        self.mask = noise

    def decrypt(self):
        return (self.ct - self.mask) % FHE_PRIME_Q


def analyze_signal_evolution(k=10, p=33, n_slots=64, cycles=10, seed=1234):
    """Analyze how signals of different magnitudes evolve through mult->proj cycles."""
    rng = np.random.default_rng(seed)
    delta = FHE_PRIME_Q // 256
    T = k * (k + 1) // 2
    q = FHE_PRIME_Q

    # System matrices (shared across slots)
    _, A, _, _ = compute_system_matrices_ff(k, p, seed=42, mod=q)

    # Create varied initial values across different signal ranges
    init_plain = np.zeros((k, n_slots), dtype=np.int64)
    signal_ranges = []

    for slot in range(n_slots):
        # Exponential progression to cover wide range
        if slot < n_slots // 4:
            # Very low signals: 0.05Δ to 0.5Δ
            scale_factor = 0.05 + (0.45 * slot / (n_slots // 4))
        elif slot < n_slots // 2:
            # Low to medium: 0.5Δ to 1.0Δ
            scale_factor = 0.5 + (0.5 * (slot - n_slots // 4) / (n_slots // 4))
        elif slot < 3 * n_slots // 4:
            # Medium to high: 1.0Δ to 1.5Δ
            scale_factor = 1.0 + (0.5 * (slot - n_slots // 2) / (n_slots // 4))
        else:
            # High signals: 1.5Δ to 3.0Δ
            scale_factor = 1.5 + (1.5 * (slot - 3 * n_slots // 4) / (n_slots // 4))

        init_val = int(scale_factor * delta)
        init_plain[:, slot] = init_val
        signal_ranges.append(scale_factor)

    ct = PackedMockCipher(init_plain, rng)

    print(f"Analyzing signal evolution: k={k}, p={p}, slots={n_slots}, cycles={cycles}")
    print(f"Delta = {delta}")
    print(f"Signal ranges: {min(signal_ranges):.2f}Δ to {max(signal_ranges):.2f}Δ\n")

    # Track evolution of key slots
    key_slots = [0, 15, 31, 47, 63]  # Representative slots
    evolution_data = []

    print("Cycle |", end="")
    for slot in key_slots:
        print(f" slot{slot:2d}({signal_ranges[slot]:.1f}Δ) |", end="")
    print()
    print("------|", end="")
    for _ in key_slots:
        print("-------------|", end="")
    print()

    # Initial state
    dec_init = centered(ct.decrypt())[0, :]
    print(f"    0 |", end="")
    for slot in key_slots:
        print(f"    {dec_init[slot]:7.0f} |", end="")
    print()

    for c in range(1, cycles + 1):
        # Self-multiply with scaling to prevent decay
        prod_ct = (ct.ct * ct.ct) // delta
        prod_plain = (ct.plain * ct.plain) // delta

        # Build T×n_slots array with diagonal terms
        prod_T_ct = np.zeros((T, n_slots), dtype=np.int64)
        prod_T_plain = np.zeros((T, n_slots), dtype=np.int64)
        idx = 0
        for i in range(k):
            for j in range(i, k):
                if i == j:
                    prod_T_ct[idx] = prod_ct[i]
                    prod_T_plain[idx] = prod_plain[i]
                idx += 1

        # Forward projection per slot
        v = (A @ prod_T_ct) % q

        # Project each slot back using real pinv
        x_est_T = np.zeros_like(prod_T_ct)
        for s in range(n_slots):
            x_est_T[:, s] = project_with_real_pinv(A, v[:, s], q)
        x_est_T = lwr_round_vector(x_est_T, q, delta)

        # Select first k components
        new_ct_k = x_est_T[:k]
        new_plain_k = prod_T_plain[:k]

        # Build next ciphertext
        ct = PackedMockCipher.__new__(PackedMockCipher)
        ct.plain = new_plain_k % q
        ct.ct = new_ct_k % q
        ct.mask = (ct.ct - ct.plain) % q

        # Monitor key slots
        dec_current = centered(ct.decrypt())[0, :]
        evolution_data.append(dec_current.copy())

        print(f"{c:5d} |", end="")
        for slot in key_slots:
            print(f"    {dec_current[slot]:7.0f} |", end="")
        print()

    # Final analysis
    print("\n" + "=" * 80)
    print("FINAL ANALYSIS")
    print("=" * 80)

    final_values = evolution_data[-1]

    # Categorize slots by behavior
    stable_slots = []
    growing_slots = []
    decaying_slots = []

    for slot in range(n_slots):
        init_val = dec_init[slot]
        final_val = final_values[slot]

        if abs(final_val) < 10:  # Essentially zero
            decaying_slots.append((slot, signal_ranges[slot], init_val, final_val))
        elif abs(final_val - init_val) / max(abs(init_val), 1) < 0.5:  # Within 50%
            stable_slots.append((slot, signal_ranges[slot], init_val, final_val))
        else:
            growing_slots.append((slot, signal_ranges[slot], init_val, final_val))

    print(f"\nSLOT BEHAVIOR SUMMARY:")
    print(f"Stable slots ({len(stable_slots)}): maintain signal within 50%")
    print(f"Growing slots ({len(growing_slots)}): significant growth")
    print(f"Decaying slots ({len(decaying_slots)}): decay to near zero")

    if stable_slots:
        print(
            f"\nSTABLE RANGE: {min(s[1] for s in stable_slots):.2f}Δ to {max(s[1] for s in stable_slots):.2f}Δ"
        )
        for slot, scale, init_val, final_val in stable_slots[:5]:  # Show first 5
            print(
                f"  Slot {slot:2d}: {scale:.2f}Δ  {init_val:4.0f} -> {final_val:4.0f}"
            )

    if growing_slots:
        print(
            f"\nGROWING RANGE: {min(s[1] for s in growing_slots):.2f}Δ to {max(s[1] for s in growing_slots):.2f}Δ"
        )
        for slot, scale, init_val, final_val in growing_slots[:3]:  # Show first 3
            print(
                f"  Slot {slot:2d}: {scale:.2f}Δ  {init_val:4.0f} -> {final_val:4.0f}"
            )

    if decaying_slots:
        print(
            f"\nDECAYING RANGE: {min(s[1] for s in decaying_slots):.2f}Δ to {max(s[1] for s in decaying_slots):.2f}Δ"
        )
        for slot, scale, init_val, final_val in decaying_slots[:3]:  # Show first 3
            print(
                f"  Slot {slot:2d}: {scale:.2f}Δ  {init_val:4.0f} -> {final_val:4.0f}"
            )


if __name__ == "__main__":
    analyze_signal_evolution(cycles=15)
