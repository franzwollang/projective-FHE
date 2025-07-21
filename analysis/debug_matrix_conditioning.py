#!/usr/bin/env python3
"""
Debug matrix conditioning calculation to understand why Σ(1/s_i²) = 0.
"""

import numpy as np
from FHE.code.simulation.ff_poc.qc_matrix_ff import (
    compute_system_matrices_ff,
    FHE_PRIME_Q,
)


def debug_matrix_conditioning():
    """Debug the singular value calculation."""
    print("=== Debugging Matrix Conditioning ===")

    k, p, seed, q = 20, 128, 2025, FHE_PRIME_Q
    T = k * (k + 1) // 2

    print(f"Parameters: k={k}, T={T}, p={p}, q={q}")

    # Generate system matrices
    G, A_ff, A_pinv_ff, _T = compute_system_matrices_ff(k, p, seed, q)

    print(f"Matrix A shape: {A_ff.shape}")
    print(f"Matrix A range: [{A_ff.min()}, {A_ff.max()}]")

    # Convert to real for SVD
    A_real = A_ff.astype(np.float64)
    print(f"A_real range: [{A_real.min():.3f}, {A_real.max():.3f}]")

    # Compute SVD
    U, s, Vt = np.linalg.svd(A_real, full_matrices=False)

    print(f"\nSingular values:")
    print(f"  Shape: {s.shape}")
    print(f"  Range: [{s.min():.6f}, {s.max():.6f}]")
    print(f"  First 10: {s[:10]}")
    print(f"  Last 10: {s[-10:]}")

    # Check for zeros or very small values
    tiny_threshold = 1e-10
    tiny_count = np.sum(s < tiny_threshold)
    print(f"  Values < {tiny_threshold}: {tiny_count}")

    if tiny_count > 0:
        print("  ❌ Some singular values are effectively zero!")
        print("     This causes 1/s_i² to be infinite or very large")

    # Compute Σ(1/s_i²) carefully
    s_nonzero = s[s > tiny_threshold]
    if len(s_nonzero) > 0:
        inv_s_squared = 1.0 / (s_nonzero**2)
        sum_inv_s_squared = np.sum(inv_s_squared)
        print(f"  Σ(1/s_i²) (nonzero only): {sum_inv_s_squared:.6f}")
        print(f"  Max 1/s_i²: {np.max(inv_s_squared):.6f}")
        print(f"  Min 1/s_i²: {np.min(inv_s_squared):.6f}")

        # Debug individual terms
        print(f"  First few 1/s_i²: {inv_s_squared[:5]}")
        print(f"  Sum of first 5: {np.sum(inv_s_squared[:5]):.10f}")
        print(f"  Sum using higher precision:")

        # Try with higher precision
        s_high_prec = s_nonzero.astype(np.float64)
        inv_s_squared_hp = 1.0 / (s_high_prec**2)
        sum_hp = np.sum(inv_s_squared_hp)
        print(f"    High precision sum: {sum_hp:.10e}")

        # Try manual calculation
        manual_sum = 0.0
        for i in range(min(5, len(s_nonzero))):
            term = 1.0 / (s_nonzero[i] ** 2)
            manual_sum += term
            print(
                f"    s[{i}] = {s_nonzero[i]:.3f}, 1/s²= {term:.2e}, sum so far = {manual_sum:.2e}"
            )

    else:
        print("  ❌ All singular values are effectively zero!")
        sum_inv_s_squared = 0.0

    # Check the matrix rank
    rank = np.linalg.matrix_rank(A_real)
    print(f"  Matrix rank: {rank} (expected: {min(p, T)} = {min(p, T)})")

    # Check if the issue is in the original computation
    print(f"\nDebugging the original computation:")
    try:
        s_orig = np.linalg.svd(A_real, compute_uv=False)
        sum_inv_s_squared_orig = np.sum(1.0 / (s_orig**2))
        print(f"  Original Σ(1/s_i²): {sum_inv_s_squared_orig:.6f}")
    except Exception as e:
        print(f"  Error in original computation: {e}")

    return A_real, s, sum_inv_s_squared


if __name__ == "__main__":
    A_real, s, sum_inv_s_squared = debug_matrix_conditioning()
