#!/usr/bin/env python3
"""
Packed mock ciphertext pipeline test.

Simulates a realistic RLWE ciphertext that packs many coefficients (n_slots)
into one object.  Each slot undergoes the same mult→proj cycle; we broadcast
all arithmetic across the slot dimension so the test is still fast.

The goal: show that after a few cycles we can decrypt the packed ciphertext and
recover the expected values in **all slots**.
"""
import numpy as np
import os
import sys
from typing import Tuple

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(ROOT)

from simulation.ff_poc.qc_matrix_ff import (
    compute_system_matrices_ff,
    FHE_PRIME_Q,
)
from simulation.ff_poc.mult_proj_poc_ff import (
    project_with_real_pinv,
    lwr_round_vector,
)

q = FHE_PRIME_Q


def centered(x):
    return ((x + q // 2) % q) - q // 2


def discrete_uniform(shape: Tuple[int, ...], half_range: int, rng):
    return rng.integers(-half_range, half_range + 1, size=shape, dtype=np.int64)


class PackedMockCipher:
    """A toy ciphertext packing n_slots coefficients per polynomial."""

    def __init__(self, plain: np.ndarray, rng: np.random.Generator):
        # plain shape: (k, n_slots)
        self.plain = plain % q
        noise = discrete_uniform(plain.shape, 20, rng)  # small mask
        self.ct = (self.plain + noise) % q
        self.mask = noise

    def decrypt(self):
        return (self.ct - self.mask) % q


def mult_cipher(ct: PackedMockCipher, delta: int, rng):
    """Element-wise self-multiply with LWR noise, keep shape."""
    noise = discrete_uniform(ct.ct.shape, delta // 2, rng)
    new_ct = (ct.ct * ct.ct + noise) % q
    new_plain = (ct.plain * ct.plain) % q
    out = PackedMockCipher.__new__(PackedMockCipher)
    out.plain, out.ct, out.mask = new_plain, new_ct, noise % q
    return out


def run_packed_pipeline(k=10, p=33, n_slots=64, cycles=5, seed=1234):
    rng = np.random.default_rng(seed)
    delta = q // 256
    T = k * (k + 1) // 2
    # system matrices (shared across slots)
    _, A, _, _ = compute_system_matrices_ff(k, p, seed=42, mod=q)

    # Create varied initial values across different signal ranges
    # Distribute values from 10% to 200% of delta across slots
    init_plain = np.zeros((k, n_slots), dtype=np.int64)
    for slot in range(n_slots):
        # Linear progression from 0.1*delta to 2.0*delta
        scale_factor = 0.1 + (1.9 * slot / (n_slots - 1))
        init_val = int(scale_factor * delta)
        init_plain[:, slot] = init_val

    ct = PackedMockCipher(init_plain, rng)

    print(
        f"Running packed mock pipeline: k={k}, p={p}, slots={n_slots}, cycles={cycles}"
    )
    print(
        f"Initial values: slot0={init_plain[0,0]} ({init_plain[0,0]/delta:.2f}Δ), "
        f"slot31={init_plain[0,31]} ({init_plain[0,31]/delta:.2f}Δ), "
        f"slot63={init_plain[0,63]} ({init_plain[0,63]/delta:.2f}Δ)\n"
    )
    print("Cycle | slot0 val | slot0 RMS | slot31 val | slot63 val | slot63 RMS")
    print("------|-----------|-----------|------------|------------|------------")

    for c in range(1, cycles + 1):
        # Self-multiply ciphertext with scaling to prevent decay
        prod_ct = (ct.ct * ct.ct) // delta  # Scale by delta
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

        # Forward projection per slot (A @ prod_T_ct)
        v = (A @ prod_T_ct) % q
        # project each slot back using real pinv (loop slots for clarity)
        x_est_T = np.zeros_like(prod_T_ct)
        for s in range(n_slots):
            x_est_T[:, s] = project_with_real_pinv(A, v[:, s], q)
        x_est_T = lwr_round_vector(x_est_T, q, delta)
        # select first k
        new_plain_k = prod_T_plain[:k]
        new_ct_k = x_est_T[:k]
        # build next ciphertext
        ct = PackedMockCipher.__new__(PackedMockCipher)
        ct.plain = new_plain_k % q
        ct.ct = new_ct_k % q
        ct.mask = (ct.ct - ct.plain) % q

        # Monitor multiple slots to see signal evolution across ranges
        dec_all = centered(ct.decrypt())

        # Slot 0 (low signal)
        dec0 = dec_all[:, 0]
        val0 = np.mean(dec0)
        rms0 = np.sqrt(np.mean(dec0.astype(float) ** 2))

        # Slot 31 (mid signal)
        dec31 = dec_all[:, 31]
        val31 = np.mean(dec31)

        # Slot 63 (high signal)
        dec63 = dec_all[:, 63]
        val63 = np.mean(dec63)
        rms63 = np.sqrt(np.mean(dec63.astype(float) ** 2))

        print(
            f"{c:5d} | {val0:9.1f} | {rms0:9.1f} | {val31:10.1f} | {val63:10.1f} | {rms63:10.1f}"
        )

    print("\nDecrypt check: signal evolution across slots")
    dec_final = centered(ct.decrypt())[0, :]  # First lane across all slots
    print(
        f"Final slot values: 0={dec_final[0]:.0f}, 15={dec_final[15]:.0f}, "
        f"31={dec_final[31]:.0f}, 47={dec_final[47]:.0f}, 63={dec_final[63]:.0f}"
    )
    print(
        f"As fraction of Δ: 0={dec_final[0]/delta:.2f}, 15={dec_final[15]/delta:.2f}, "
        f"31={dec_final[31]/delta:.2f}, 47={dec_final[47]/delta:.2f}, 63={dec_final[63]/delta:.2f}"
    )


if __name__ == "__main__":
    run_packed_pipeline()
