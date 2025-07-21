#!/usr/bin/env python3
"""
Debug script for detailed analysis of a single mult->proj simulation run.
This helps identify where the theory-vs-practice discrepancy comes from.
"""

import numpy as np
from FHE.code.simulation.ff_poc.mult_proj_poc_ff import (
    run_ff_experiment,
    compute_analytical_proj_err_var,
    centered_variance,
    discrete_noise,
    lwr_round_vector,
)
from FHE.code.simulation.ff_poc.qc_matrix_ff import (
    compute_system_matrices_ff,
    FHE_PRIME_Q,
)


def debug_single_cycle():
    """Run a single cycle with detailed logging."""
    print("=== Debug: Single Mult->Proj Cycle ===")

    # Use Standard Interactive tier parameters
    k = 20
    p = 128
    t_param = 4096
    seed = 2025
    q = FHE_PRIME_Q
    delta = q // t_param

    T = k * (k + 1) // 2
    print(f"Parameters: k={k}, T={T}, p={p}, q={q}, Δ={delta}")

    # Generate system matrices
    G, A, A_pinv, _T = compute_system_matrices_ff(k, p, seed, q)
    print(f"Matrix shapes: G{G.shape}, A{A.shape}, A_pinv{A_pinv.shape}")

    # Check matrix conditioning
    s = np.linalg.svd(A.astype(np.float64), compute_uv=False)
    sum_inv_s_squared = np.sum(1.0 / (s**2))
    print(f"Matrix conditioning: Σ(1/s_i²) = {sum_inv_s_squared:.6f} (vs p = {p})")

    # Generate true signal
    rng = np.random.default_rng(seed + 1000)
    narrow_band_max = delta // 8
    x_true_full = (
        rng.integers(-narrow_band_max, narrow_band_max + 1, size=T, dtype=np.int64)
        // delta
    ) * delta
    x_true_k = x_true_full[:k]

    print(f"\nTrue signal statistics:")
    print(f"  x_true_full range: [{x_true_full.min()}, {x_true_full.max()}]")
    print(f"  x_true_k range: [{x_true_k.min()}, {x_true_k.max()}]")

    # Calculate actual signal variance in Δ² units
    signal_var_raw = np.var(x_true_full.astype(np.float64))
    signal_var_delta_units = signal_var_raw / (delta**2)
    print(f"  Actual signal variance (raw): {signal_var_raw:.6f}")
    print(f"  Actual signal variance (Δ² units): {signal_var_delta_units:.6f}")
    print(f"  Theoretical assumption (Δ² units): {1/12:.6f}")
    print(f"  Scale factor: {signal_var_delta_units / (1/12):.1f}x")

    # Start with zero noise
    e_prev_full = np.zeros(T, dtype=np.int64)

    print(f"\n--- Cycle 1 ---")

    # Step 1: Measure initial noise
    noise_before = centered_variance(e_prev_full, q, delta)
    print(f"1. Initial noise variance: {noise_before:.6f}")

    # Step 2: Generate multiplication noise
    e_mult = discrete_noise((p,), delta, rng)
    mult_noise_var = centered_variance(e_mult, q, delta)
    print(f"2. Multiplication noise variance: {mult_noise_var:.6f}")

    # Step 3: Simulate encrypted computation
    v_noisy = (A @ ((x_true_full + e_prev_full) % q) + e_mult) % q
    print(f"3. Noisy observation v_noisy range: [{v_noisy.min()}, {v_noisy.max()}]")

    # Step 4: Project back using real-valued pseudoinverse
    from FHE.code.simulation.ff_poc.mult_proj_poc_ff import project_with_real_pinv

    x_est_high_precision = project_with_real_pinv(A, v_noisy, q)
    print(
        f"4. High-precision estimate range: [{x_est_high_precision.min()}, {x_est_high_precision.max()}]"
    )

    # Step 5: LWR rounding
    x_est_full = lwr_round_vector(x_est_high_precision, q, delta)
    print(f"5. After LWR rounding range: [{x_est_full.min()}, {x_est_full.max()}]")

    # Step 6: Selection
    x_est_k = x_est_full[:k]
    print(f"6. Selected k components range: [{x_est_k.min()}, {x_est_k.max()}]")

    # Step 7: Error analysis
    e_next_k = (x_est_k - x_true_k) % q
    noise_after = centered_variance(e_next_k, q, delta)
    print(f"7. Final noise variance (k components): {noise_after:.6f}")

    # Compare with analytical prediction
    is_underdetermined = p < T
    analytical_pred = compute_analytical_proj_err_var(
        A, delta, k, p, T, is_underdetermined
    )
    print(f"8. Analytical prediction (legacy): {analytical_pred:.6f}")

    # Try improved analytical prediction with detailed breakdown
    from FHE.code.simulation.ff_poc.mult_proj_poc_ff import (
        compute_analytical_proj_err_var_improved,
    )

    analytical_pred_improved = compute_analytical_proj_err_var_improved(
        A, delta, k, p, T, is_underdetermined, x_true_full
    )
    print(f"8b. Analytical prediction (improved): {analytical_pred_improved:.6f}")

    # Break down the improved prediction
    sigma_signal_sq_raw = np.var(x_true_full.astype(np.float64))
    sigma_signal_sq_delta_units = sigma_signal_sq_raw / (delta**2)
    projection_loss_var = (T - p) / T * sigma_signal_sq_delta_units

    A_real = A.astype(np.float64)
    s = np.linalg.svd(A_real, compute_uv=False)
    sum_inv_s_squared = np.sum(1.0 / (s**2))
    sigma_mult_sq_delta_units = 1.0 / 12.0
    transformed_noise_var = (sum_inv_s_squared * sigma_mult_sq_delta_units) / (T - p)

    print(f"   Breakdown of improved prediction:")
    print(f"   - Signal variance (Δ² units): {sigma_signal_sq_delta_units:.6f}")
    print(f"   - Projection loss term: {projection_loss_var:.6f}")
    print(f"   - Σ(1/s_i²): {sum_inv_s_squared:.6f}")
    print(f"   - Transformed noise term: {transformed_noise_var:.6f}")
    print(f"   - Total: {projection_loss_var + transformed_noise_var:.6f}")

    print(f"9. Empirical/Analytical ratio (legacy): {noise_after/analytical_pred:.1f}x")
    print(
        f"9b. Empirical/Analytical ratio (improved): {noise_after/analytical_pred_improved:.1f}x"
    )

    # Detailed error breakdown
    print(f"\n--- Error Breakdown ---")
    e_full = (x_est_full - x_true_full) % q
    noise_full = centered_variance(e_full, q, delta)
    print(f"Full T-dimensional error variance: {noise_full:.6f}")

    # Check if error is concentrated in the selected k components
    e_discarded = e_full[k:]
    noise_discarded = (
        centered_variance(e_discarded, q, delta) if len(e_discarded) > 0 else 0
    )
    print(f"Discarded (T-k) error variance: {noise_discarded:.6f}")

    return {
        "noise_before": noise_before,
        "noise_after": noise_after,
        "analytical_pred": analytical_pred,
        "mult_noise_var": mult_noise_var,
        "ratio": noise_after / analytical_pred,
    }


def debug_multi_cycle():
    """Run multiple cycles to see convergence behavior."""
    print("\n=== Debug: Multi-Cycle Convergence ===")

    # Run a short simulation with detailed output
    logger, analytical_eq = run_ff_experiment(
        k=20, p=128, t_param=4096, num_cycles=20, seed=2025, verbose=True
    )

    print(f"\nAnalytical equilibrium: {analytical_eq:.6f}")

    if len(logger.data) >= 10:
        final_10 = [row[2] for row in logger.data[-10:]]  # after_proj values
        mean_final = np.mean(final_10)
        std_final = np.std(final_10)
        print(f"Final 10 cycles: mean={mean_final:.6f}, std={std_final:.6f}")
        print(f"Convergence ratio: {mean_final/analytical_eq:.1f}x")

    return logger


if __name__ == "__main__":
    single_result = debug_single_cycle()
    multi_result = debug_multi_cycle()
