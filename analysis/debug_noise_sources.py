#!/usr/bin/env python3
"""
Debug missing noise sources: isolate projection loss, LWR rounding, T->k selection.
"""

import numpy as np
from FHE.code.simulation.ff_poc.qc_matrix_ff import (
    compute_system_matrices_ff,
    FHE_PRIME_Q,
)
from FHE.code.simulation.ff_poc.mult_proj_poc_ff import (
    project_with_real_pinv,
    lwr_round_vector,
    centered_variance,
    discrete_noise,
)


def debug_noise_breakdown():
    """Break down noise sources step by step."""
    print("=== Debugging Noise Sources ===")

    # Use standard parameters
    k, p, seed, q = 20, 128, 2025, FHE_PRIME_Q
    T = k * (k + 1) // 2
    delta = 16

    print(f"Parameters: k={k}, T={T}, p={p}, q={q}, Δ={delta}")
    print(f"Underdetermined: p < T = {p < T}")

    # Generate matrices and signal
    G, A, A_pinv_ff, _T = compute_system_matrices_ff(k, p, seed, q)

    rng = np.random.default_rng(seed + 1000)
    narrow_band_max = delta // 8
    x_true_full = (
        rng.integers(-narrow_band_max, narrow_band_max + 1, size=T, dtype=np.int64)
        // delta
    ) * delta
    x_true_k = x_true_full[:k]

    print(f"Signal range: [{x_true_full.min()}, {x_true_full.max()}]")

    # Step 1: Perfect projection (no noise)
    print(f"\n--- Step 1: Perfect Projection (No Noise) ---")
    v_perfect = (A @ x_true_full) % q
    x_est_perfect = project_with_real_pinv(A, v_perfect, q)

    # Error from projection loss only
    proj_loss_error = (x_est_perfect - x_true_full) % q
    proj_loss_error_centered = (proj_loss_error + q // 2) % q - q // 2
    proj_loss_var = centered_variance(proj_loss_error_centered, q, delta)

    print(f"Projection loss variance: {proj_loss_var:.6f}")

    # Theoretical projection loss
    sigma_signal_sq_raw = np.var(x_true_full.astype(np.float64))
    sigma_signal_sq_delta = sigma_signal_sq_raw / (delta**2)
    theory_proj_loss = (T - p) / T * sigma_signal_sq_delta
    print(f"Theoretical projection loss: {theory_proj_loss:.6f}")
    print(f"Empirical/Theory ratio: {proj_loss_var/theory_proj_loss:.1f}x")

    # Step 2: Add multiplication noise
    print(f"\n--- Step 2: Add Multiplication Noise ---")
    e_mult = discrete_noise((p,), delta, rng)
    mult_noise_var = centered_variance(e_mult, q, delta)
    print(f"Multiplication noise variance: {mult_noise_var:.6f}")

    v_noisy = (v_perfect + e_mult) % q
    x_est_with_mult_noise = project_with_real_pinv(A, v_noisy, q)

    mult_error = (x_est_with_mult_noise - x_true_full) % q
    mult_error_centered = (mult_error + q // 2) % q - q // 2
    total_var_after_mult = centered_variance(mult_error_centered, q, delta)

    print(f"Total variance after mult noise: {total_var_after_mult:.6f}")
    print(f"Increase from mult noise: {total_var_after_mult - proj_loss_var:.6f}")

    # Step 3: Add LWR rounding
    print(f"\n--- Step 3: Add LWR Rounding ---")
    x_est_rounded = lwr_round_vector(x_est_with_mult_noise, q, delta)

    rounding_error = (x_est_rounded - x_true_full) % q
    rounding_error_centered = (rounding_error + q // 2) % q - q // 2
    total_var_after_rounding = centered_variance(rounding_error_centered, q, delta)

    print(f"Total variance after LWR rounding: {total_var_after_rounding:.6f}")
    print(
        f"Increase from LWR rounding: {total_var_after_rounding - total_var_after_mult:.6f}"
    )

    # Step 4: T->k selection
    print(f"\n--- Step 4: T->k Selection ---")
    x_est_k = x_est_rounded[:k]

    selection_error = (x_est_k - x_true_k) % q
    selection_error_centered = (selection_error + q // 2) % q - q // 2
    final_var = centered_variance(selection_error_centered, q, delta)

    print(f"Final variance (k components): {final_var:.6f}")
    print(f"Change from T->k selection: {final_var - total_var_after_rounding:.6f}")

    # Step 5: Compare with full simulation
    print(f"\n--- Step 5: Compare with Full Simulation ---")
    from FHE.code.simulation.ff_poc.mult_proj_poc_ff import run_ff_experiment

    logger, _analytical = run_ff_experiment(
        k=k, p=p, t_param=4096, num_cycles=5, seed=seed, verbose=False
    )
    sim_final_var = logger.data[-1][2]  # after_proj from last cycle

    print(f"Full simulation final variance: {sim_final_var:.6f}")
    print(f"Step-by-step vs simulation: {final_var:.6f} vs {sim_final_var:.6f}")

    # Summary
    print(f"\n--- Noise Breakdown Summary ---")
    print(f"1. Projection loss:        {proj_loss_var:.6f}")
    print(f"2. + Multiplication noise: {total_var_after_mult - proj_loss_var:.6f}")
    print(
        f"3. + LWR rounding:         {total_var_after_rounding - total_var_after_mult:.6f}"
    )
    print(f"4. + T->k selection:       {final_var - total_var_after_rounding:.6f}")
    print(f"Total step-by-step:        {final_var:.6f}")
    print(f"Full simulation:           {sim_final_var:.6f}")

    return {
        "proj_loss": proj_loss_var,
        "mult_noise_increase": total_var_after_mult - proj_loss_var,
        "lwr_increase": total_var_after_rounding - total_var_after_mult,
        "selection_change": final_var - total_var_after_rounding,
        "total_stepwise": final_var,
        "simulation": sim_final_var,
    }


if __name__ == "__main__":
    results = debug_noise_breakdown()
