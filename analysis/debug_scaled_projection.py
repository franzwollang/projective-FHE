#!/usr/bin/env python3
"""
Debug script for the scaled-down matrix projection approach.
Analyzes the quantization error introduced by scaling and its impact on noise.
"""

import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from simulation.ff_poc.qc_matrix_ff import compute_system_matrices_ff, FHE_PRIME_Q


def project_with_real_pinv(A_ff: np.ndarray, v: np.ndarray, q: int) -> np.ndarray:
    """
    Project using real-valued Moore-Penrose pseudoinverse, then convert to FF.
    """
    A_real = A_ff.astype(np.float64)
    A_pinv_real = np.linalg.pinv(A_real)
    v_real = v.astype(np.float64)
    x_est_real = A_pinv_real @ v_real
    x_est_ff = np.round(x_est_real).astype(np.int64)
    x_est_ff = ((x_est_ff + q // 2) % q) - q // 2
    return x_est_ff


def analyze_scaling_error(k=20, p=210, seed=42, scale_shift=10, q=FHE_PRIME_Q):
    """
    Analyze the quantization error introduced by scaling down the matrix.
    """
    print(f"=== Scaling Analysis: scale_shift={scale_shift} ===")

    # Generate matrices with and without scaling
    G_orig, A_orig, A_pinv_orig, T = compute_system_matrices_ff(
        k, p, seed, q, scale_shift=0
    )
    G_scaled, A_scaled, A_pinv_scaled, T = compute_system_matrices_ff(
        k, p, seed, q, scale_shift=scale_shift
    )

    print(f"Matrix dimensions: A is {A_orig.shape[0]}×{A_orig.shape[1]}")
    print(f"Scaling factor: 2^{scale_shift} = {2**scale_shift}")

    # Compare matrix magnitudes
    print(f"Original A range: [{np.min(A_orig)}, {np.max(A_orig)}]")
    print(f"Scaled A range: [{np.min(A_scaled)}, {np.max(A_scaled)}]")

    # Test with a known signal
    np.random.seed(seed)
    x_true = np.random.randint(-100, 101, size=T)

    # Forward projection with both matrices
    v_orig = (A_orig @ x_true) % q
    v_scaled = (A_scaled @ x_true) % q

    print(f"Forward projection comparison:")
    print(f"Original v range: [{np.min(v_orig)}, {np.max(v_orig)}]")
    print(f"Scaled v range: [{np.min(v_scaled)}, {np.max(v_scaled)}]")

    # Check if scaling prevents overflow
    max_forward_orig = np.max(np.abs(A_orig @ x_true))
    max_forward_scaled = np.max(np.abs(A_scaled @ x_true))

    print(f"Overflow analysis (before mod q={q}):")
    print(f"Original max |A @ x|: {max_forward_orig}")
    print(f"Scaled max |A @ x|: {max_forward_scaled}")
    print(f"Original overflow: {max_forward_orig > q}")
    print(f"Scaled overflow: {max_forward_scaled > q}")

    # Reconstruction accuracy with real pseudoinverse
    x_est_orig = project_with_real_pinv(A_orig, v_orig, q)
    x_est_scaled = project_with_real_pinv(A_scaled, v_scaled, q)

    err_orig = np.linalg.norm(x_est_orig - x_true)
    err_scaled = np.linalg.norm(x_est_scaled - x_true)

    print(f"Reconstruction error (L2 norm):")
    print(f"Original: {err_orig:.6f}")
    print(f"Scaled: {err_scaled:.6f}")
    print(f"Scaling error increase: {err_scaled / err_orig:.2f}x")

    return {
        "scale_shift": scale_shift,
        "max_forward_orig": max_forward_orig,
        "max_forward_scaled": max_forward_scaled,
        "err_orig": err_orig,
        "err_scaled": err_scaled,
        "overflow_prevented": max_forward_scaled <= q,
    }


if __name__ == "__main__":
    # Test different scale shifts
    print("Scale | Max Forward | Overflow | Recon Error")
    print("------|-------------|----------|------------")

    for scale_shift in [0, 8, 10, 12, 14]:
        try:
            result = analyze_scaling_error(scale_shift=scale_shift)
            overflow = "Yes" if result["max_forward_scaled"] > FHE_PRIME_Q else "No"
            print(
                f"{scale_shift:5d} | {result['max_forward_scaled']:11.0f} | {overflow:8s} | {result['err_scaled']:11.6f}"
            )
        except Exception as e:
            print(f"Error with scale_shift={scale_shift}: {e}")
        print()
