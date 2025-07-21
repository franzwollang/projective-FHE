#!/usr/bin/env python3
"""
Finite Field Proof-of-Concept for the mult→proj pipeline (fully determined
mode: p = T).
"""
import argparse
import numpy as np
import os
import sys
from typing import Tuple

from .qc_matrix_ff import (
    compute_system_matrices_ff,
    validate_matrices_ff,
    FHE_PRIME_Q,
)

# Add parent directory to path for utils import
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from utils import NoiseLogger  # noqa: E402

# ---------------- Noise helpers ---------------- #


def discrete_noise(
    shape: Tuple[int, ...], delta: int, rng: np.random.Generator
) -> np.ndarray:
    """Uniform integer noise in [−Δ/2, +Δ/2] inclusive."""
    half = delta // 2
    return rng.integers(-half, half + 1, size=shape, dtype=np.int64)


def centered_variance(vec: np.ndarray, mod: int, delta: int) -> float:
    """Return variance in **Δ^2 units** of centered residues in [−q/2, q/2]."""
    centered = (vec + mod // 2) % mod - mod // 2
    return float(np.mean((centered / delta) ** 2))


def compute_analytical_proj_err_var_improved(
    A: np.ndarray,
    delta: int,
    k: int,
    p: int,
    T: int,
    is_underdetermined: bool,
    x_true_sample: np.ndarray,
) -> float:
    """
    Improved analytical projection error variance in Δ^2 units.

    Uses actual signal variance from sample and includes both:
    - Projection loss: (T-p)/T * σ_signal²
    - Transformed noise: Σ(1/s_i²) * σ_mult² / (T-p) for underdetermined
    """
    # Measure actual signal variance from the sample
    sigma_signal_sq_raw = np.var(x_true_sample.astype(np.float64))
    sigma_signal_sq_delta_units = sigma_signal_sq_raw / (delta**2)

    if is_underdetermined:
        # Term 1: Projection loss (dominant term)
        projection_loss_var = (T - p) / T * sigma_signal_sq_delta_units

        # Term 2: Transformed multiplication noise
        # Use real-valued SVD to get singular values
        A_real = A.astype(np.float64)
        s = np.linalg.svd(A_real, compute_uv=False)
        sum_inv_s_squared = np.sum(1.0 / (s**2))

        # Multiplication noise variance (in Δ² units)
        sigma_mult_sq_delta_units = 1.0 / 12.0  # uniform on [-Δ/2, Δ/2]

        # Transformed noise variance per component
        transformed_noise_var = (sum_inv_s_squared * sigma_mult_sq_delta_units) / (
            T - p
        )

        total_var = projection_loss_var + transformed_noise_var

        return float(total_var)

    else:
        # Fully determined case: only transformed multiplication noise
        A_real = A.astype(np.float64)
        s = np.linalg.svd(A_real, compute_uv=False)
        sum_inv_s_squared = np.sum(1.0 / (s**2))

        sigma_mult_sq_delta_units = 1.0 / 12.0
        var_per_component = (sum_inv_s_squared * sigma_mult_sq_delta_units) / T

        return float(var_per_component)


def compute_analytical_proj_err_var(
    A: np.ndarray, delta: int, k: int, p: int, T: int, is_underdetermined: bool
) -> float:
    """
    Legacy analytical prediction function (kept for compatibility).
    Uses hardcoded σ_signal² = 1/12 assumption.
    """
    if is_underdetermined:
        return float((T - p) / T * (1 / 12))

    # Fully determined case:
    s = np.linalg.svd(A.astype(np.float64), compute_uv=False)
    sum_inv_s_squared = np.sum(1.0 / (s**2))
    var_e_mult_component = delta**2 / 12.0
    total_power = sum_inv_s_squared * var_e_mult_component
    var_per_component = total_power / T
    return float(var_per_component / (delta**2))


def lwr_round_vector(vec: np.ndarray, q: int, delta: int) -> np.ndarray:
    """Rounds each component of a vector to the nearest multiple of Δ."""
    # val_rounded = floor(val / Δ) * Δ
    # A more robust way that handles negative numbers correctly:
    # val_rounded = floor(val / Δ + 0.5) * Δ
    return (np.floor(vec / delta + 0.5) * delta).astype(np.int64)


def project_with_real_pinv(A_ff: np.ndarray, v: np.ndarray, q: int) -> np.ndarray:
    """
    Project using real-valued Moore-Penrose pseudoinverse, then convert to FF.

    This avoids the issue where the FF pseudoinverse has wrong coefficients
    because the real pseudoinverse has fractional values that can't be
    represented as integers mod q.
    """
    # Compute real-valued pseudoinverse
    A_real = A_ff.astype(np.float64)
    A_pinv_real = np.linalg.pinv(A_real)

    # Project in real arithmetic
    v_real = v.astype(np.float64)
    x_est_real = A_pinv_real @ v_real

    # Convert back to finite field with centered reduction
    x_est_ff = np.round(x_est_real).astype(np.int64)
    x_est_ff = ((x_est_ff + q // 2) % q) - q // 2

    return x_est_ff


# ---------------- Experiment ---------------- #


def run_ff_experiment(
    k: int = 20,
    p: int = 0,  # Default 0 means p=T
    t_param: int = 256,
    num_cycles: int = 1000,
    seed: int = 42,
    lazy_mod: bool = True,
    verbose: bool = True,
) -> Tuple[NoiseLogger, float]:
    q = FHE_PRIME_Q
    delta = q // t_param

    # Determine T and p
    T = k * (k + 1) // 2
    if p == 0:
        p = T  # Default to fully determined
    is_underdetermined = p < T
    mode_str = "Underdetermined" if is_underdetermined else "Fully Determined"

    if verbose:
        print(f"=== Finite-Field mult→proj PoC ({mode_str}) ===")
        print(
            f"k={k}, T={T}, p={p}, q={q}, Δ={delta}, "
            f"cycles={num_cycles}, seed={seed}"
        )
        print()

    # Generate system matrices (square when p = T)
    G, A, A_pinv, T = compute_system_matrices_ff(k, p, seed, q)

    if not validate_matrices_ff(A, A_pinv, q, p, T, verbose=verbose):
        raise ValueError("Matrix validation failed")

    rng = np.random.default_rng(seed + 123)
    logger = NoiseLogger()

    # True plaintext vector (fixed for this demo) - clamped to narrow band
    narrow_band_max = delta // 8
    x_true_unaligned = rng.integers(
        -narrow_band_max, narrow_band_max + 1, size=T, dtype=np.int64
    )
    x_true_full = (x_true_unaligned // delta) * delta  # ensure LWR-aligned

    # We only care about the error in the k logical outputs
    x_true_k = x_true_full[:k]

    # Initial noise (start at 0)
    e_prev_full = np.zeros(T, dtype=np.int64)

    # Rank-failure counter
    rank_fail = 0

    # Analytical projection error
    analytical_proj_err = compute_analytical_proj_err_var_improved(
        A, delta, k, p, T, is_underdetermined, x_true_unaligned
    )
    if verbose:
        msg = "Analytical Equilibrium/Error Var (Δ^2 units):"
        print(f"\n{msg} {analytical_proj_err:.4f}\n")

    for cycle in range(1, num_cycles + 1):
        noise_before = centered_variance(e_prev_full, q, delta)

        # fresh multiplication noise in p coords
        e_mult = discrete_noise((p,), delta, rng)
        mult_noise_var = centered_variance(e_mult, q, delta)

        # --- "Encrypted" Domain Update ---
        # This simulates the full operation on the noisy state vector
        v_noisy = (A @ ((x_true_full + e_prev_full) % q) + e_mult) % q
        # Use real-valued pseudoinverse to avoid FF representation issues
        x_est_high_precision = project_with_real_pinv(A, v_noisy, q)

        # --- LWR Rounding Step (CRITICAL) ---
        x_est_full = lwr_round_vector(x_est_high_precision, q, delta)

        # --- T->k Selection Step (CRITICAL) ---
        # The program only uses the k logical outputs for the next cycle.
        # The error from the other T-k terms is discarded.
        x_est_k = x_est_full[:k]

        # --- Cumulative Error Calculation (on k components) ---
        e_next_k = (x_est_k - x_true_k) % q
        noise_after = centered_variance(e_next_k, q, delta)

        # --- Projection Error Calculation (on T components) ---
        # This measures the quality of the full projection
        A_plus_e_mult = (A_pinv @ e_mult) % q
        proj_err_var = centered_variance(A_plus_e_mult, q, delta)

        logger.record_cycle(
            cycle, noise_before, noise_after, mult_noise_var, proj_err_var
        )

        # --- Prepare for next cycle ---
        # Error from k outputs becomes input error for next cycle
        e_prev_full = np.zeros(T, dtype=np.int64)
        e_prev_full[:k] = e_next_k
        noise_before = centered_variance(e_prev_full, q, delta)

        # quick rank check every 50 cycles (mod q determinant)
        if cycle % 50 == 0:
            rank = np.linalg.matrix_rank(A.astype(float))
            if rank < p:
                rank_fail += 1

        if verbose and (cycle <= 10 or cycle % 100 == 0 or cycle == num_cycles):
            print(
                f"Cycle {cycle:4d}: before={noise_before:8.4f}  "
                f"after={noise_after:8.4f}  projErr={proj_err_var:8.4f}  "
                f"multNoise={mult_noise_var:6.2f}"
            )

    if verbose:
        print("\nRank failures (out of checks):", rank_fail)

    return logger, analytical_proj_err


# ---------------- CLI ---------------- #


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(description="Finite-field mult→proj PoC")
    parser.add_argument("--cycles", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument(
        "--p",
        type=int,
        default=0,
        help="Redundant rows. Default 0 sets p=T.",
    )
    parser.add_argument("--t", type=int, default=256)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    logger, analytical_proj_err = run_ff_experiment(
        k=args.k,
        p=args.p,
        t_param=args.t,
        num_cycles=args.cycles,
        seed=args.seed,
        verbose=not args.quiet,
    )

    # Print summary (now with an analytical target)
    print()
    T = args.k * (args.k + 1) // 2
    p = args.p if args.p != 0 else T
    is_underdetermined = p < T
    summary_target = "Equilibrium" if is_underdetermined else "Proj Err Var"
    print(f"Target metric: Analytical {summary_target}")
    logger.print_summary(analytical_proj_err)


if __name__ == "__main__":
    main()
