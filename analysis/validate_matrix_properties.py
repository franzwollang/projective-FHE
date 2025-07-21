#!/usr/bin/env python3
"""
Analyzes the singular value properties of the system matrix A to verify
the noise reduction condition required by the FHE architecture.
"""
import numpy as np

from FHE.code.simulation.ff_poc.qc_matrix_ff import compute_system_matrices_ff


def main():
    """Main analysis function."""
    print("=== System Matrix 'A' Noise Reduction Property Analysis ===")

    # Use parameters from a failing SLA tier for analysis
    k = 20
    p = 128
    seed = 2025
    q = 65537

    T = k * (k + 1) // 2
    print(f"\nAnalyzing for parameters: k={k}, T={T}, p={p}")

    # Generate the system matrix A. We only need A for this analysis.
    # The finite field version is used to get the correct structure,
    # but the SVD analysis is done over the reals as per the theory.
    _G, A_ff, _A_pinv, _T = compute_system_matrices_ff(k, p, seed, q)
    A_real = A_ff.astype(np.float64)

    # Compute singular values
    s = np.linalg.svd(A_real, compute_uv=False)
    print(f"\nComputed {len(s)} singular values.")
    print(f"  min(s) = {s.min():.4f}")
    print(f"  max(s) = {s.max():.4f}")

    # Check the noise reduction condition from the design document
    sum_inv_s_squared = np.sum(1.0 / (s**2))

    print("\n--- Noise Reduction Condition ---")
    print("Theory requires: Σ(1/s_i²) < p")
    print(f"  - Σ(1/s_i²) = {sum_inv_s_squared:.4f}")
    print(f"  - p         = {p}")

    if sum_inv_s_squared < p:
        print("\n✅ PASSED: The matrix HAS the noise reduction property.")
    else:
        print("\n❌ FAILED: Matrix DOES NOT have the noise reduction property.")
        print("This is the likely cause of the runaway noise in the simulation.")


if __name__ == "__main__":
    main()
