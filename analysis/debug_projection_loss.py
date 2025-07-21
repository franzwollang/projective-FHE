#!/usr/bin/env python3
"""
Debug why empirical projection loss is 4.9x higher than theoretical prediction.
"""

import numpy as np
from FHE.code.simulation.ff_poc.qc_matrix_ff import (
    compute_system_matrices_ff,
    FHE_PRIME_Q,
)
from FHE.code.simulation.ff_poc.mult_proj_poc_ff import (
    project_with_real_pinv,
    centered_variance,
)


def debug_projection_loss_theory():
    """Debug the projection loss calculation in detail."""
    print("=== Debugging Projection Loss Theory ===")

    k, p, seed, q = 20, 128, 2025, FHE_PRIME_Q
    T = k * (k + 1) // 2
    delta = 16

    print(f"Parameters: k={k}, T={T}, p={p}")
    print(f"p/T ratio: {p/T:.3f}")
    print(f"(T-p)/T ratio: {(T-p)/T:.3f}")

    # Generate matrices and signal
    G, A, A_pinv_ff, _T = compute_system_matrices_ff(k, p, seed, q)

    rng = np.random.default_rng(seed + 1000)
    narrow_band_max = delta // 8
    x_true = (
        rng.integers(-narrow_band_max, narrow_band_max + 1, size=T, dtype=np.int64)
        // delta
    ) * delta

    print(f"Signal range: [{x_true.min()}, {x_true.max()}]")

    # Compute actual signal statistics
    signal_var_raw = np.var(x_true.astype(np.float64))
    signal_var_delta_units = signal_var_raw / (delta**2)
    print(f"Actual signal variance (Δ² units): {signal_var_delta_units:.6f}")

    # Test projection loss with real-valued computation
    A_real = A.astype(np.float64)
    A_pinv_real = np.linalg.pinv(A_real)

    # Project in real arithmetic (no finite field effects)
    x_true_real = x_true.astype(np.float64)
    v_real = A_real @ x_true_real
    x_est_real = A_pinv_real @ v_real

    # Projection matrix and loss
    P = A_pinv_real @ A_real  # This should be the projection matrix
    proj_loss_real = x_est_real - x_true_real

    print(f"\n--- Real-Valued Analysis ---")
    print(f"Projection matrix P shape: {P.shape}")
    print(f"P diagonal mean: {np.mean(np.diag(P)):.6f}")
    print(f"P diagonal std: {np.std(np.diag(P)):.6f}")
    print(f"P off-diagonal mean: {np.mean(P[np.triu_indices_from(P, k=1)]):.6f}")

    # Check if P is actually a projection (P^2 = P)
    P_squared = P @ P
    projection_error = np.max(np.abs(P_squared - P))
    print(f"||P² - P||_∞: {projection_error:.2e} (should be ~0)")

    # Analyze projection loss variance
    proj_loss_var_real = np.var(proj_loss_real)
    proj_loss_var_delta_units = proj_loss_var_real / (delta**2)
    print(f"Real projection loss variance (Δ² units): {proj_loss_var_delta_units:.6f}")

    # Compare with theory
    theory_proj_loss = (T - p) / T * signal_var_delta_units
    print(f"Theoretical projection loss: {theory_proj_loss:.6f}")
    print(f"Real/Theory ratio: {proj_loss_var_delta_units/theory_proj_loss:.1f}x")

    # Now test with finite field effects
    print("\n--- Finite Field Effects ---")
    v_ff = (A @ x_true) % q
    x_est_ff = project_with_real_pinv(A, v_ff, q)

    proj_loss_ff = (x_est_ff - x_true) % q
    proj_loss_ff_centered = (proj_loss_ff + q // 2) % q - q // 2
    proj_loss_var_ff = centered_variance(proj_loss_ff_centered, q, delta)

    print(f"FF projection loss variance: {proj_loss_var_ff:.6f}")
    print(f"FF/Real ratio: {proj_loss_var_ff/proj_loss_var_delta_units:.1f}x")
    print(f"FF/Theory ratio: {proj_loss_var_ff/theory_proj_loss:.1f}x")

    # Investigate the theoretical assumption
    print(f"\n--- Theory Assumptions ---")

    # The theory assumes isotropic signal with uniform energy distribution
    # Check if our signal violates this
    signal_energy_per_dim = np.sum(x_true_real**2) / T
    print(f"Average signal energy per dimension: {signal_energy_per_dim:.6f}")
    print(f"Expected from variance: {signal_var_raw:.6f}")

    # Check projection matrix eigenvalues
    eigenvals = np.linalg.eigvals(P)
    eigenvals_real = eigenvals[np.isreal(eigenvals)].real
    print(f"Projection matrix eigenvalues:")
    print(f"  Count of ~1.0: {np.sum(np.abs(eigenvals_real - 1.0) < 1e-10)}")
    print(f"  Count of ~0.0: {np.sum(np.abs(eigenvals_real) < 1e-10)}")
    print(f"  Rank (trace): {np.trace(P):.1f}")

    return {
        "signal_var": signal_var_delta_units,
        "theory_proj_loss": theory_proj_loss,
        "real_proj_loss": proj_loss_var_delta_units,
        "ff_proj_loss": proj_loss_var_ff,
        "real_theory_ratio": proj_loss_var_delta_units / theory_proj_loss,
        "ff_theory_ratio": proj_loss_var_ff / theory_proj_loss,
    }


if __name__ == "__main__":
    results = debug_projection_loss_theory()
