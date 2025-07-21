"""
Finite field (GF(p)) matrix utilities for mult->proj PoC.
"""

import numpy as np
from scipy.linalg import circulant
from typing import Tuple

FHE_PRIME_Q = 65537

# --- Modular Arithmetic ---


def modular_inverse(a: int, m: int) -> int:
    """
    Computes the modular inverse of a modulo m using the Extended Euclidean Algorithm.
    Returns 0 if no inverse exists.
    """
    a = a % m
    if np.gcd(a, m) != 1:
        return 0  # No modular inverse exists

    old_r, r = m, a
    old_s, s = 1, 0
    old_t, t = 0, 1

    while r != 0:
        quotient = old_r // r
        old_r, r = r, old_r - quotient * r
        old_s, s = s, old_s - quotient * s
        old_t, t = t, old_t - quotient * t

    return old_t % m


def modular_pinv(A: np.ndarray, mod: int) -> np.ndarray:
    """
    Computes the Moore-Penrose pseudoinverse of A over a prime field GF(mod).
    If A is wide (p < T), it computes A+ = A.T @ (A @ A.T)^-1.
    If A is square (p = T), it computes the standard inverse.
    """

    def _inv_square(M_sq: np.ndarray) -> np.ndarray:
        """Computes inverse of a square matrix via Gaussian elimination."""
        n = M_sq.shape[0]
        M_aug = np.hstack([M_sq, np.eye(n, dtype=np.int64)])

        for i in range(n):
            pivot = i
            while pivot < n and M_aug[pivot, i] == 0:
                pivot += 1
            if pivot == n:
                rank = np.linalg.matrix_rank(M_sq.astype(float))
                msg = f"Matrix is singular (rank {rank} < {n}), cannot find inverse"
                raise ValueError(msg)
            M_aug[[i, pivot]] = M_aug[[pivot, i]]

            inv = modular_inverse(int(M_aug[i, i]), mod)
            if inv == 0:
                msg = f"Element {M_aug[i, i]} has no modular inverse."
                raise ValueError(msg)
            M_aug[i] = (M_aug[i] * inv) % mod

            for j in range(n):
                if i != j:
                    M_aug[j] = (M_aug[j] - M_aug[j, i] * M_aug[i]) % mod

        return M_aug[:, n:]

    p, T = A.shape
    if p > T:
        raise ValueError("Tall matrices (p > T) are not supported.")

    if p == T:
        return _inv_square(A).astype(np.int64)

    A_At = (A @ A.T) % mod
    try:
        A_At_inv = _inv_square(A_At)
    except ValueError as e:
        msg = "Could not invert (A @ A.T) for pseudoinverse."
        raise ValueError(msg) from e

    A_pinv = (A.T @ A_At_inv) % mod
    return A_pinv.astype(np.int64)


# --- Matrix Generation ---


def generate_qc_mds_ff(k: int, p: int, seed: int, mod: int) -> np.ndarray:
    """
    Generate a p x k QC-MDS matrix over a finite field.
    """
    rng = np.random.default_rng(seed)
    generating_vector = rng.integers(0, mod, size=p, dtype=np.int64)
    full_circulant = circulant(generating_vector)
    G = full_circulant[:, :k]
    return G


def lift_to_quadratic_ff(G: np.ndarray, mod: int) -> Tuple[np.ndarray, int]:
    """
    Apply Veronese map to lift G to quadratic space over a finite field.
    """
    p, k = G.shape
    T = k * (k + 1) // 2

    monomial_indices = []
    for i in range(k):
        for j in range(i, k):
            monomial_indices.append((i, j))

    A = np.zeros((p, T), dtype=np.int64)
    for row_idx in range(p):
        g = G[row_idx, :]
        for col_idx, (i, j) in enumerate(monomial_indices):
            A[row_idx, col_idx] = (g[i] * g[j]) % mod

    return A, T


def compute_system_matrices_ff(
    k: int, p: int, seed: int, mod: int = FHE_PRIME_Q
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Generate complete system matrices over a finite field.
    """
    G = generate_qc_mds_ff(k, p, seed, mod)
    # Quadratic lifting to get p×T system matrix A
    A, T = lift_to_quadratic_ff(G, mod)

    # scale_shift parameter removed - no longer using matrix scaling
    A_pinv = modular_pinv(A, mod)
    return G, A, A_pinv, T


def validate_matrices_ff(
    A: np.ndarray, A_pinv: np.ndarray, mod: int, p: int, T: int, verbose: bool = True
) -> bool:
    """
    Validate matrix properties over a finite field.
    """
    # Check that A has full row rank
    rank_A = np.linalg.matrix_rank(A.astype(np.float64))
    if rank_A != p:
        if verbose:
            print(f"ERROR: A rank {rank_A} != expected {p} (over reals)")
        return False

    # Check pseudoinverse property
    reconstruction = (A @ A_pinv @ A) % mod
    error = np.sum((reconstruction - A) % mod)
    if error != 0:
        if verbose:
            print(f"ERROR: Pseudoinverse reconstruction failed. Error norm: {error}")
        return False

    if verbose:
        print("Finite field matrix validations passed.")
        print(f"  A rank (over reals): {rank_A}")
    return True
