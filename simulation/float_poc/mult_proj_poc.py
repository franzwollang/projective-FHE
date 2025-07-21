#!/usr/bin/env python3
"""
Minimal Proof-of-Concept for the mult->proj pipeline.

This script runs 50 cycles of the mult->proj operation, measuring noise
before and after each step to validate the theoretical equilibrium
derived in noise_equilibrium_proof.md.
"""

import argparse
import numpy as np
import time
from qc_matrix import (
    compute_system_matrices,
    validate_matrices,
    compute_theoretical_equilibrium,
)
from utils import (
    gaussian_vector,
    noise_variance_per_component,
    NoiseLogger,
    print_singular_values,
)


def run_mult_proj_experiment(
    k: int = 20,
    p: int = 100,
    sigma_signal: float = 1.0,
    sigma_mult: float = 0.01,
    num_cycles: int = 50,
    seed: int = 42,
    verbose: bool = True,
    use_full_formula: bool = True,
) -> NoiseLogger:
    """
    Run the complete mult->proj experiment.

    Args:
        k: Logical width (number of source components)
        p: Number of redundant rows
        sigma_signal: Standard deviation of signal components
        sigma_mult: Standard deviation of multiplication noise
        num_cycles: Number of mult->proj cycles to run
        seed: Random seed for reproducibility
        verbose: Whether to print progress and details
        use_full_formula: Use full recurrence formula vs approximation

    Returns:
        NoiseLogger containing all recorded measurements
    """
    if verbose:
        print("=== Mult->Proj Pipeline Proof of Concept ===")
        print(
            f"Parameters: k={k}, p={p}, σ_signal={sigma_signal}, "
            f"σ_mult={sigma_mult}"
        )
        print(f"Running {num_cycles} cycles with seed={seed}")
        formula_type = "full recurrence" if use_full_formula else "approximation"
        print(f"Using {formula_type} for theoretical equilibrium")
        print()

    # Generate system matrices
    start_time = time.time()
    G, A, A_pinv, T = compute_system_matrices(k, p, seed)
    matrix_time = time.time() - start_time

    if verbose:
        print(f"Matrix generation took {matrix_time:.3f}s")
        print(f"Dimensions: G{G.shape}, A{A.shape}, " f"A_pinv{A_pinv.shape}, T={T}")

    # Validate matrices
    if not validate_matrices(G, A, A_pinv, k, p, T, verbose=verbose):
        raise ValueError("Matrix validation failed!")

    if verbose:
        print()
        print_singular_values(A, "System matrix A")
        print()

    # Compute theoretical equilibrium
    theoretical_eq = compute_theoretical_equilibrium(
        A, sigma_signal, sigma_mult, T, p, use_full_formula=use_full_formula
    )

    if verbose:
        print(f"Theoretical equilibrium noise variance: " f"{theoretical_eq:.6f}")
        print()

    # Initialize RNG for experiment
    rng = np.random.default_rng(seed + 1000)  # Different seed for experiment

    # Generate fixed true signal (synthetic data)
    x_true = gaussian_vector(T, sigma_signal, rng)

    if verbose:
        x_true_var = noise_variance_per_component(x_true)
        print(f"Generated true signal with ||x_true||^2/T = " f"{x_true_var:.6f}")
        print()

    # Initialize noise logger
    logger = NoiseLogger()

    # Initialize noise vector (start with small initial noise)
    e_prev = gaussian_vector(T, sigma_mult, rng)

    if verbose:
        print("Starting mult->proj cycles...")
        print()

    # Run the main experiment loop
    for cycle in range(1, num_cycles + 1):
        # === MULTIPLICATION STEP ===
        # Current noisy state: x_true + e_prev
        # Server computes: v = A * (x_true + e_prev) + e_mult

        # Record noise before multiplication
        noise_before_mult = noise_variance_per_component(e_prev)

        # Generate fresh multiplication noise
        e_mult = gaussian_vector(p, sigma_mult, rng)
        mult_noise = noise_variance_per_component(e_mult)

        # Server computation: v = A * (x_true + e_prev) + e_mult
        v = A @ (x_true + e_prev) + e_mult

        # === PROJECTION STEP ===
        # Server computes: x_est = A_pinv * v
        x_est = A_pinv @ v

        # Compute noise after projection
        e_next = x_est - x_true
        noise_after_proj = noise_variance_per_component(e_next)

        # Record measurements
        logger.record_cycle(cycle, noise_before_mult, noise_after_proj, mult_noise)

        # Update for next cycle
        e_prev = e_next

        # Print progress occasionally
        if verbose and (cycle <= 10 or cycle % 10 == 0 or cycle == num_cycles):
            print(
                f"Cycle {cycle:2d}: before={noise_before_mult:.6f}, "
                f"after={noise_after_proj:.6f}, mult={mult_noise:.6f}"
            )

    if verbose:
        print()
        print("=== EXPERIMENT COMPLETE ===")

    return logger


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(description="Mult->Proj Pipeline PoC")
    parser.add_argument(
        "--cycles", type=int, default=50, help="Number of cycles to run"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--k", type=int, default=20, help="Logical width")
    parser.add_argument("--p", type=int, default=100, help="Number of redundant rows")
    parser.add_argument(
        "--sigma-signal", type=float, default=1.0, help="Signal std dev"
    )
    parser.add_argument(
        "--sigma-mult", type=float, default=0.01, help="Mult noise std dev"
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    parser.add_argument("--save-csv", type=str, help="Save results to CSV file")
    parser.add_argument(
        "--plot", action="store_true", help="Generate plot (requires matplotlib)"
    )
    parser.add_argument(
        "--approx",
        action="store_true",
        help="Use approximation formula instead of full recurrence",
    )

    args = parser.parse_args()

    # Run experiment
    logger = run_mult_proj_experiment(
        k=args.k,
        p=args.p,
        sigma_signal=args.sigma_signal,
        sigma_mult=args.sigma_mult,
        num_cycles=args.cycles,
        seed=args.seed,
        verbose=not args.quiet,
        use_full_formula=not args.approx,
    )

    # Compute theoretical equilibrium for summary
    G, A, A_pinv, T = compute_system_matrices(args.k, args.p, args.seed)
    theoretical_eq = compute_theoretical_equilibrium(
        A,
        args.sigma_signal,
        args.sigma_mult,
        T,
        args.p,
        use_full_formula=not args.approx,
    )

    # Print summary
    print()
    logger.print_summary(theoretical_eq)

    # Save to CSV if requested
    if args.save_csv:
        logger.save_to_csv(args.save_csv)

    # Generate plot if requested
    if args.plot:
        try:
            import matplotlib.pyplot as plt

            cycles = [row[0] for row in logger.data]
            after_proj = [row[2] for row in logger.data]

            plt.figure(figsize=(10, 6))
            plt.plot(
                cycles, after_proj, "b-", label="Empirical noise variance", alpha=0.7
            )
            plt.axhline(
                y=theoretical_eq,
                color="r",
                linestyle="--",
                label=f"Theoretical equilibrium " f"({theoretical_eq:.6f})",
            )
            plt.xlabel("Cycle")
            plt.ylabel("Noise variance per component")
            plt.title("Mult->Proj Noise Evolution")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            plot_filename = (
                f"mult_proj_noise_k{args.k}_p{args.p}_" f"cycles{args.cycles}.png"
            )
            plt.savefig(plot_filename, dpi=150)
            print(f"Plot saved to {plot_filename}")

        except ImportError:
            print("WARNING: matplotlib not available, skipping plot generation")


if __name__ == "__main__":
    main()
