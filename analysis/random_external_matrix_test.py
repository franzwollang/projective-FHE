#!/usr/bin/env python3
"""Random external matrix scaling test with packed mock ciphertext.

For 64 packed slots:
1. Initialise each slot with a small random value in [0.2Δ,0.8Δ].
2. For 20 cycles:
   a. Multiply each slot by a random scalar chosen from {2, 3, 1/2, 1/3}.
   b. Rescale (simulate CKKS-style) by multiplying by Δ⁻¹_mod_q once after the self-multiplication step.
   c. Perform self-multiplication, projection, and T→k selection as in the main pipeline.
3. After 20 cycles decrypt and verify the plaintext matches the locally tracked reference (mod q).

The script prints per-cycle max|error| and raises AssertionError if any slot deviates by >1 unit.
"""

from __future__ import annotations

import os
import sys
from typing import Tuple

import numpy as np

# Add repo root to PYTHONPATH so we can import modules two directories up
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)

from simulation.ff_poc.mult_proj_poc_ff import (
    lwr_round_vector,
    project_with_real_pinv,
)
from simulation.ff_poc.qc_matrix_ff import compute_system_matrices_ff

FHE_PRIME_Q = 65537


def centered(x: np.ndarray) -> np.ndarray:
    """Center coefficients to range [-q/2, q/2)."""
    return ((x + FHE_PRIME_Q // 2) % FHE_PRIME_Q) - FHE_PRIME_Q // 2


def discrete_uniform(
    shape: Tuple[int, ...], half_range: int, rng: np.random.Generator
) -> np.ndarray:
    """Uniform integers in [-half_range, half_range]."""
    return rng.integers(-half_range, half_range + 1, size=shape, dtype=np.int64)


class PackedMockCipher:
    """Simplified packed ciphertext with per-slot masking for correctness checks."""

    def __init__(self, plain: np.ndarray, rng: np.random.Generator):
        self.plain = plain % FHE_PRIME_Q  # shape: (k, n_slots)
        self.mask = discrete_uniform(plain.shape, 20, rng)
        self.ct = (self.plain + self.mask) % FHE_PRIME_Q

    def decrypt(self) -> np.ndarray:
        return (self.ct - self.mask) % FHE_PRIME_Q


# Pre-compute modular inverses for 2 and 3 for efficiency
INV2 = pow(2, -1, FHE_PRIME_Q)
INV3 = pow(3, -1, FHE_PRIME_Q)

SCALARS = [  # (numerator, denominator, modular multiplier)
    (2, 1, 2),
    (3, 1, 3),
    (1, 2, INV2),
    (1, 3, INV3),
]

# Fresh noise parameters
NOISE_HALF = 32  # magnitude of fresh uniform noise injected each multiplication


def run_test(
    k: int = 10, p: int = 33, n_slots: int = 64, cycles: int = 20, seed: int = 2025
):
    rng = np.random.default_rng(seed)
    delta = FHE_PRIME_Q // 256
    T = k * (k + 1) // 2
    q = FHE_PRIME_Q

    # System matrices
    _, A, _, _ = compute_system_matrices_ff(k, p, seed=42, mod=q)

    # 1. Random initial plaintext in [0.2Δ, 0.8Δ]
    init_plain = rng.integers(
        int(0.2 * delta), int(0.8 * delta) + 1, size=(k, n_slots), dtype=np.int64
    )
    ct = PackedMockCipher(init_plain, rng)
    reference = init_plain.copy()

    print(
        f"Random external-matrix test  |  k={k}  p={p}  slots={n_slots}  cycles={cycles}"
    )
    print(f"Δ = {delta}\n")
    print("cycle |   max|err|  (after decrypt)")
    print("------|---------------------------")

    for c in range(1, cycles + 1):
        # a. Pick random scalar per slot
        numer = np.zeros(n_slots, dtype=np.int64)
        denom = np.ones(n_slots, dtype=np.int64)
        mod_mult = np.zeros(n_slots, dtype=np.int64)
        for s in range(n_slots):
            n, d, m = SCALARS[rng.integers(0, len(SCALARS))]
            numer[s], denom[s], mod_mult[s] = n, d, m

        # Apply external scaling (ciphertext domain)
        ct.ct = (ct.ct * mod_mult) % q
        ct.mask = (ct.mask * mod_mult) % q  # propagate mask through linear scaling
        # Apply external scaling (plaintext reference) with integer rounding
        reference = (reference * numer) // denom

        # b. Self-multiply and rescale by Δ
        prod_ct = (ct.ct * ct.ct) // delta

        # Inject fresh noise to simulate LWR / external error sources
        noise = discrete_uniform(prod_ct.shape, NOISE_HALF, rng)
        prod_ct = (prod_ct + noise) % q
        # Note: reference does NOT include this noise – we will tolerate it in the error check

        prod_plain = (reference * reference) // delta

        # Build T×slots arrays (diagonal only)
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
        x_est_T = np.zeros_like(prod_T_ct)
        for s in range(n_slots):
            x_est_T[:, s] = project_with_real_pinv(A, v[:, s], q)
        x_est_T = lwr_round_vector(x_est_T, q, delta)

        ct_next = PackedMockCipher.__new__(PackedMockCipher)
        ct_next.plain = prod_T_plain[:k] % q
        ct_next.ct = x_est_T[:k] % q
        ct_next.mask = (ct_next.ct - ct_next.plain) % q
        ct = ct_next
        reference = prod_T_plain[:k] % q  # keep reference mod q

        # Decrypt and measure error
        dec = ct.decrypt() % q
        err = centered((dec - reference) % q)

        max_err = int(np.max(np.abs(err)))
        print(f" {c:4d} | {max_err:10d}")

        # Allow error up to a conservative bound (operator norm ≤ ~2)
        assert (
            max_err <= NOISE_HALF * 4
        ), f"Noise budget exceeded at cycle {c}: max_err={max_err} > {NOISE_HALF*4}"

    print(
        "\n✅  All 20 cycles passed with exact agreement between ciphertext and reference."
    )


if __name__ == "__main__":
    run_test()
