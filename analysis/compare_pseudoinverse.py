#!/usr/bin/env python3
"""
Compare finite field pseudoinverse with real-valued Moore-Penrose pseudoinverse.
"""

import numpy as np
from FHE.code.simulation.ff_poc.qc_matrix_ff import (
    compute_system_matrices_ff,
    FHE_PRIME_Q,
)


def compare_pseudoinverses():
    """Compare finite field vs real-valued pseudoinverse behavior."""
    print("=== Comparing Pseudoinverse Methods ===")

    k, p, seed, q = 20, 128, 2025, FHE_PRIME_Q
    T = k * (k + 1) // 2

    print(f"Parameters: k={k}, T={T}, p={p}, q={q}")
    print(f"Matrix is {'underdetermined' if p < T else 'overdetermined'}")

    # Generate system matrices
    G, A_ff, A_pinv_ff, _T = compute_system_matrices_ff(k, p, seed, q)

    # Convert to real-valued matrices
    A_real = A_ff.astype(np.float64)
    A_pinv_real = np.linalg.pinv(A_real)

    print(f"\nMatrix shapes:")
    print(f"  A: {A_ff.shape}")
    print(f"  A_pinv_ff: {A_pinv_ff.shape}")
    print(f"  A_pinv_real: {A_pinv_real.shape}")

    # Test vector
    rng = np.random.default_rng(seed)
    x_test = rng.integers(0, 100, size=T, dtype=np.int64)
    print(f"\nTest vector range: [{x_test.min()}, {x_test.max()}]")

    # Forward projection: x -> A @ x
    v_ff = (A_ff @ x_test) % q
    v_real = A_real @ x_test.astype(np.float64)

    print(f"\nForward projection:")
    print(f"  v_ff range: [{v_ff.min()}, {v_ff.max()}]")
    print(f"  v_real range: [{v_real.min():.1f}, {v_real.max():.1f}]")

    # Backward projection: v -> A_pinv @ v
    x_recovered_ff = (A_pinv_ff @ v_ff) % q
    x_recovered_real = A_pinv_real @ v_real

    print(f"\nBackward projection:")
    print(f"  x_recovered_ff range: [{x_recovered_ff.min()}, {x_recovered_ff.max()}]")
    print(
        f"  x_recovered_real range: [{x_recovered_real.min():.1f}, {x_recovered_real.max():.1f}]"
    )

    # Error analysis
    error_ff = (x_recovered_ff - x_test) % q
    error_ff_centered = (error_ff + q // 2) % q - q // 2
    max_error_ff = np.max(np.abs(error_ff_centered))

    error_real = x_recovered_real - x_test.astype(np.float64)
    max_error_real = np.max(np.abs(error_real))

    print(f"\nError analysis:")
    print(f"  Max error (finite field): {max_error_ff}")
    print(f"  Max error (real): {max_error_real:.6f}")
    print(f"  FF error / Real error ratio: {max_error_ff / max_error_real:.1f}x")

    # Check if the real pseudoinverse actually recovers the vector well
    if max_error_real < 1e-10:
        print("  ✅ Real pseudoinverse recovers vector accurately")
    else:
        print("  ❌ Real pseudoinverse also has significant error")
        print("     This suggests the system is ill-conditioned")

    # Matrix conditioning analysis
    s = np.linalg.svd(A_real, compute_uv=False)
    condition_number = s[0] / s[-1]
    print("\nMatrix conditioning:")
    print(f"  Condition number: {condition_number:.2e}")
    print(f"  Smallest singular value: {s[-1]:.2e}")
    print(f"  Rank: {np.linalg.matrix_rank(A_real)}")

    return max_error_ff, max_error_real


def test_simple_case():
    """Test with a simple, well-conditioned matrix."""
    print("\n=== Testing Simple Well-Conditioned Case ===")

    # Create a simple 3x5 matrix that we know should work
    q = FHE_PRIME_Q
    A_simple = np.array(
        [[1, 0, 0, 1, 2], [0, 1, 0, 2, 1], [0, 0, 1, 1, 1]], dtype=np.int64
    )

    print(f"Simple matrix A:\n{A_simple}")

    # Compute real pseudoinverse
    A_pinv_real = np.linalg.pinv(A_simple.astype(np.float64))
    print(f"Real pseudoinverse shape: {A_pinv_real.shape}")

    # Test if A @ A_pinv @ A = A for real case
    reconstruction_real = (
        A_simple.astype(np.float64) @ A_pinv_real @ A_simple.astype(np.float64)
    )
    real_reconstruction_error = np.max(
        np.abs(reconstruction_real - A_simple.astype(np.float64))
    )
    print(f"Real reconstruction error: {real_reconstruction_error:.2e}")

    # Try to compute finite field pseudoinverse using our formula
    try:
        from FHE.code.simulation.ff_poc.qc_matrix_ff import modular_pinv

        A_pinv_ff = modular_pinv(A_simple, q)
        print(f"FF pseudoinverse computed successfully")

        # Test vector recovery
        x_test = np.array([10, 20, 30, 40, 50], dtype=np.int64)
        v = (A_simple @ x_test) % q
        x_recovered = (A_pinv_ff @ v) % q

        error = (x_recovered - x_test) % q
        error_centered = (error + q // 2) % q - q // 2
        max_error = np.max(np.abs(error_centered))

        print(f"Simple case recovery error: {max_error}")

    except Exception as e:
        print(f"FF pseudoinverse failed: {e}")


if __name__ == "__main__":
    compare_pseudoinverses()
    test_simple_case()
