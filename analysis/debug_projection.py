#!/usr/bin/env python3
"""
Debug the core projection operation to see if the pseudoinverse is working correctly.
"""

import numpy as np
from FHE.code.simulation.ff_poc.qc_matrix_ff import (
    compute_system_matrices_ff,
    FHE_PRIME_Q,
)


def test_projection_accuracy():
    """Test if A @ A_pinv @ v ≈ v for the finite field pseudoinverse."""
    print("=== Testing Finite Field Projection Accuracy ===")

    k, p, seed, q = 20, 128, 2025, FHE_PRIME_Q
    T = k * (k + 1) // 2

    print(f"Parameters: k={k}, T={T}, p={p}, q={q}")

    # Generate system matrices
    G, A, A_pinv, _T = compute_system_matrices_ff(k, p, seed, q)

    # Test 1: Check if A @ A_pinv @ A = A (pseudoinverse property)
    print("\n--- Test 1: Pseudoinverse Property ---")
    reconstruction = (A @ A_pinv @ A) % q
    error = np.sum((reconstruction - A) % q)
    print(f"||A @ A_pinv @ A - A||_1 = {error}")

    if error == 0:
        print("✅ Pseudoinverse property holds exactly")
    else:
        print("❌ Pseudoinverse property FAILED")
        return False

    # Test 2: Check projection on a known vector
    print("\n--- Test 2: Projection Test ---")
    rng = np.random.default_rng(seed)

    # Create a test vector that should be exactly recoverable
    x_test = rng.integers(0, 1000, size=T, dtype=np.int64)
    print(f"Test vector x range: [{x_test.min()}, {x_test.max()}]")

    # Project it through the system: x -> A @ x -> A_pinv @ (A @ x)
    v = (A @ x_test) % q
    x_recovered = (A_pinv @ v) % q

    print(f"Projected vector v range: [{v.min()}, {v.max()}]")
    print(f"Recovered vector range: [{x_recovered.min()}, {x_recovered.max()}]")

    # Check recovery error
    recovery_error = (x_recovered - x_test) % q

    # Convert to centered representation for analysis
    recovery_error_centered = (recovery_error + q // 2) % q - q // 2
    max_error = np.max(np.abs(recovery_error_centered))
    mean_error = np.mean(np.abs(recovery_error_centered))

    print(f"Max recovery error: {max_error}")
    print(f"Mean recovery error: {mean_error}")

    if max_error < q // 1000:  # Allow small rounding errors
        print("✅ Vector recovery is accurate")
        return True
    else:
        print("❌ Vector recovery has large errors")

        # Show some examples
        print("First 10 components:")
        for i in range(min(10, len(x_test))):
            print(
                f"  [{i}]: true={x_test[i]}, recovered={x_recovered[i]}, error={recovery_error_centered[i]}"
            )
        return False


def test_with_small_values():
    """Test with small values similar to those used in the simulation."""
    print("\n=== Testing with Small Signal Values ===")

    k, p, seed, q = 20, 128, 2025, FHE_PRIME_Q
    T = k * (k + 1) // 2
    delta = 16

    G, A, A_pinv, _T = compute_system_matrices_ff(k, p, seed, q)

    # Create small signal similar to simulation
    rng = np.random.default_rng(seed + 1000)
    narrow_band_max = delta // 8  # This gives us range [-2, 2] * delta = [-32, 32]
    x_small = (
        rng.integers(-narrow_band_max, narrow_band_max + 1, size=T, dtype=np.int64)
        // delta
    ) * delta

    print(f"Small signal range: [{x_small.min()}, {x_small.max()}]")

    # Test projection
    v = (A @ x_small) % q
    x_recovered = (A_pinv @ v) % q

    print(f"After projection through A: [{v.min()}, {v.max()}]")
    print(f"After recovery through A_pinv: [{x_recovered.min()}, {x_recovered.max()}]")

    # Check if the large values are the problem
    recovery_error = (x_recovered - x_small) % q
    recovery_error_centered = (recovery_error + q // 2) % q - q // 2
    max_error = np.max(np.abs(recovery_error_centered))

    print(f"Max recovery error: {max_error}")

    # Show the scale of the problem
    signal_magnitude = np.max(np.abs(x_small))
    if signal_magnitude > 0:
        relative_error = max_error / signal_magnitude
        print(f"Relative error: {relative_error:.1f}x signal magnitude")

    return max_error < q // 100


def test_projection_without_mod():
    """Test projection without modular reduction to diagnose wrap-around."""
    print("\n=== Testing Projection Without Modular Reduction ===")

    k, p, seed, q = 20, 128, 2025, FHE_PRIME_Q
    T = k * (k + 1) // 2
    delta = 16

    G, A_ff, A_pinv_ff, _T = compute_system_matrices_ff(k, p, seed, q)

    # Create small signal similar to simulation
    rng = np.random.default_rng(seed + 1000)
    narrow_band_max = delta // 8
    x_small = (
        rng.integers(-narrow_band_max, narrow_band_max + 1, size=T, dtype=np.int64)
        // delta
    ) * delta

    print(f"Small signal range: [{x_small.min()}, {x_small.max()}]")

    # Forward: x -> A @ x (with mod)
    v = (A_ff @ x_small) % q
    print(f"Forward projection v range: [{v.min()}, {v.max()}]")

    # Method 1: Current approach (with mod at each step)
    x_recovered_mod = (A_pinv_ff @ v) % q
    error_mod = (x_recovered_mod - x_small) % q
    error_mod_centered = (error_mod + q // 2) % q - q // 2
    max_error_mod = np.max(np.abs(error_mod_centered))

    print(f"\nMethod 1 (with mod):")
    print(f"  Recovered range: [{x_recovered_mod.min()}, {x_recovered_mod.max()}]")
    print(f"  Max error: {max_error_mod}")

    # Method 2: No mod during projection (compute in int64)
    A_pinv_int64 = A_pinv_ff.astype(np.int64)
    v_int64 = v.astype(np.int64)

    # Project without mod
    x_recovered_no_mod = A_pinv_int64 @ v_int64

    # Apply centered reduction only at the end
    x_recovered_centered = ((x_recovered_no_mod + q // 2) % q) - q // 2

    error_no_mod = x_recovered_centered - x_small
    max_error_no_mod = np.max(np.abs(error_no_mod))

    print(f"\nMethod 2 (no mod during projection):")
    print(
        f"  Raw projection range: [{x_recovered_no_mod.min()}, {x_recovered_no_mod.max()}]"
    )
    print(
        f"  After centered reduction: [{x_recovered_centered.min()}, {x_recovered_centered.max()}]"
    )
    print(f"  Max error: {max_error_no_mod}")

    # Compare methods
    improvement_factor = (
        max_error_mod / max_error_no_mod if max_error_no_mod > 0 else float("inf")
    )
    print(f"\nComparison:")
    print(f"  Error reduction: {improvement_factor:.1f}x")
    print(f"  Method 1 error: {max_error_mod}")
    print(f"  Method 2 error: {max_error_no_mod}")

    # Check if method 2 gives reasonable results
    signal_magnitude = np.max(np.abs(x_small))
    if signal_magnitude > 0:
        relative_error = max_error_no_mod / signal_magnitude
        print(f"  Relative error (method 2): {relative_error:.3f}x signal magnitude")

        if relative_error < 0.1:
            print("  ✅ Method 2 gives reasonable recovery")
        else:
            print("  ❌ Method 2 still has large errors")

    return max_error_mod, max_error_no_mod, improvement_factor


def analyze_pseudoinverse_matrix():
    """Analyze the pseudoinverse matrix to understand coefficient magnitudes."""
    print("\n=== Analyzing Pseudoinverse Matrix ===")

    k, p, seed, q = 20, 128, 2025, FHE_PRIME_Q
    T = k * (k + 1) // 2

    G, A_ff, A_pinv_ff, _T = compute_system_matrices_ff(k, p, seed, q)

    print(f"Matrix A range: [{A_ff.min()}, {A_ff.max()}]")
    print(f"Matrix A_pinv range: [{A_pinv_ff.min()}, {A_pinv_ff.max()}]")

    # Convert to real for comparison
    A_real = A_ff.astype(np.float64)
    A_pinv_real = np.linalg.pinv(A_real)

    print(f"Real A_pinv range: [{A_pinv_real.min():.3f}, {A_pinv_real.max():.3f}]")

    # Check if finite field pseudoinverse matches real pseudoinverse
    A_pinv_real_rounded = np.round(A_pinv_real).astype(np.int64) % q
    matches = np.sum(A_pinv_ff == A_pinv_real_rounded)
    total = A_pinv_ff.size
    match_ratio = matches / total

    print(f"FF vs Real pseudoinverse match: {matches}/{total} ({match_ratio:.1%})")

    # Analyze singular values of A
    s = np.linalg.svd(A_real, compute_uv=False)
    print(f"Singular values of A:")
    print(f"  Largest: {s[0]:.3f}")
    print(f"  Smallest: {s[-1]:.3f}")
    print(f"  Condition number: {s[0]/s[-1]:.2e}")

    # Check the norm of the pseudoinverse
    pinv_norm = np.linalg.norm(A_pinv_real)
    pinv_max_row_norm = np.max(np.linalg.norm(A_pinv_real, axis=1))

    print(f"Real pseudoinverse norms:")
    print(f"  Frobenius norm: {pinv_norm:.3f}")
    print(f"  Max row norm: {pinv_max_row_norm:.3f}")

    # The issue might be that our finite field pseudoinverse is wrong
    # Let's check if A @ A_pinv_real @ A ≈ A
    reconstruction_real = A_real @ A_pinv_real @ A_real
    reconstruction_error = np.max(np.abs(reconstruction_real - A_real))
    print(f"Real pseudoinverse reconstruction error: {reconstruction_error:.2e}")

    return A_pinv_ff, A_pinv_real


if __name__ == "__main__":
    test1_passed = test_projection_accuracy()
    test2_passed = test_with_small_values()
    test_no_mod_error_mod, test_no_mod_error_no_mod, improvement = (
        test_projection_without_mod()
    )
    A_pinv_ff, A_pinv_real = analyze_pseudoinverse_matrix()

    print(f"\n=== Summary ===")
    if improvement > 100:
        print(f"✅ Wrap-around is the root cause! Improvement: {improvement:.1f}x")
        print("   Next step: implement no-mod projection in the main simulation")
    elif improvement > 2:
        print(
            f"⚠️  Wrap-around is a significant factor. Improvement: {improvement:.1f}x"
        )
        print("   But there may be other issues to address")
    else:
        print(f"❌ Wrap-around is not the main issue. Improvement: {improvement:.1f}x")
        print("   The finite field pseudoinverse itself may be incorrect")

    if test1_passed and test2_passed:
        print("✅ All basic projection tests passed")
    else:
        print("❌ Basic projection tests failed")
