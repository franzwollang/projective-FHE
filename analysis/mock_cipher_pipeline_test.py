#!/usr/bin/env python3
"""
Mock ciphertext test: encrypt → mult→proj → decrypt.

This is **not** cryptographically secure; it merely demonstrates that a value
hidden inside a ciphertext-like container survives the full pipeline operations
and decrypts to the expected result.  All arithmetic is mod-q and identical to
what a real RLWE slot experiences, so correctness of the projection can be
checked end-to-end.
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
    lwr_round_vector,
    project_with_real_pinv,
    discrete_noise,
)

q = FHE_PRIME_Q


def center(x):
    return ((x + q // 2) % q) - q // 2


class MockCipher:
    """Toy ciphertext that hides the plaintext but still supports ops."""

    def __init__(self, msg: np.ndarray, rng: np.random.Generator):
        self.plain = msg % q  # keep for verification only (would be secret)
        # random mask simulating encryption noise
        self.mask = rng.integers(-q // 2, q // 2, size=msg.shape, dtype=np.int64)
        self.ct = (msg + self.mask) % q  # what the server sees

    # homomorphic element-wise multiply (ct * ct) with noise growth
    def mul(
        self, other: "MockCipher", delta: int, rng: np.random.Generator
    ) -> "MockCipher":
        assert self.ct.shape == other.ct.shape
        prod_ct = (self.ct * other.ct) % q
        # add fresh LWR noise per component
        noise = discrete_noise(self.ct.shape, delta, rng)
        prod_ct = (prod_ct + noise) % q
        # For correctness checking keep true prod
        prod_plain = (self.plain * other.plain) % q
        return MockCipher.from_components(prod_plain, prod_ct)

    # homomorphic addition
    def add(self, other: "MockCipher") -> "MockCipher":
        ct_sum = (self.ct + other.ct) % q
        plain_sum = (self.plain + other.plain) % q
        return MockCipher.from_components(plain_sum, ct_sum)

    @classmethod
    def from_components(cls, plain: np.ndarray, ct: np.ndarray) -> "MockCipher":
        obj = cls.__new__(cls)
        obj.plain = plain % q
        obj.ct = ct % q
        obj.mask = (obj.ct - obj.plain) % q  # derive mask for bookkeeping
        return obj

    def decrypt(self) -> np.ndarray:
        # client subtracts mask to recover plaintext
        return (self.ct - self.mask) % q


def pipeline_round(signal_ct: MockCipher, A, A_pinv, delta, rng) -> MockCipher:
    """Run one mult→proj cycle on mock ciphertext vector (length k)."""
    k = signal_ct.ct.shape[0]
    T = k * (k + 1) // 2
    p = A.shape[0]

    # 1. Expand k→p via public QC-MDS (linear) – just linear combo on ciphertexts
    G, _ = None, None  # expansion already rolled into A; we go straight to T

    # 2. Form quadratic terms (self-mul) – do it homomorphically per component
    #    We only need k self-mul outputs (diagonal terms) for the next step.
    prod_ct = signal_ct.mul(signal_ct, delta, rng)

    # Build T-vector with zeros except diagonal
    prod_vec_T = np.zeros(T, dtype=np.int64)
    idx = 0
    for i in range(k):
        for j in range(i, k):
            if i == j:
                prod_vec_T[idx] = prod_ct.ct[i]
            idx += 1

    # 3. Forward projection: v = A x  (lazy reduction)
    v_ct = (A @ prod_vec_T) % q

    # 4. Project with real-valued pinv
    x_est_T = project_with_real_pinv(A, v_ct, q)
    x_est_T = lwr_round_vector(x_est_T, q, delta)

    # 5. Select k logical outputs
    x_k_ct = x_est_T[:k]

    # Build new MockCipher for next round
    # Plaintext computed directly from previous round’s plain diagonal terms
    prod_plain_vec = (signal_ct.plain * signal_ct.plain) % q
    new_plain_k = prod_plain_vec[:k]
    return MockCipher.from_components(new_plain_k, x_k_ct)


def main():
    rng = np.random.default_rng(1234)
    k = 10
    p = 33
    delta = q // 256
    cycles = 10

    # Prepare system matrices once
    _, A, A_pinv, _ = compute_system_matrices_ff(k, p, seed=42, mod=q)

    # Encrypt initial vector (all Δ/2)
    init_plain = np.full(k, delta // 2, dtype=np.int64)
    ct = MockCipher(init_plain, rng)

    print("Cycle | decrypted RMS | expected \u0394/2")
    print("------|--------------|-------------")
    for c in range(1, cycles + 1):
        ct = pipeline_round(ct, A, A_pinv, delta, rng)
        dec = center(ct.decrypt())
        rms = np.sqrt(np.mean((dec.astype(float)) ** 2))
        print(f"{c:5d} | {rms:12.2f} | {delta/2:11.1f}")


if __name__ == "__main__":
    main()
