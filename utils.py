"""
Utility functions for the mult->proj PoC experiment.
"""

import numpy as np
import csv
from typing import List, Tuple, Optional


def gaussian_vector(
    dim: int, sigma: float, rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Generate a dim-dimensional vector with components drawn from N(0, sigma^2).

    Args:
        dim: Vector dimension
        sigma: Standard deviation
        rng: Random number generator (uses default if None)

    Returns:
        numpy array of shape (dim,)
    """
    if rng is None:
        rng = np.random.default_rng()
    return rng.normal(0.0, sigma, size=dim)


def noise_power(vec: np.ndarray) -> float:
    """
    Compute the total noise power (sum of squared components) of a vector.

    Args:
        vec: Input vector

    Returns:
        ||vec||^2 = sum of squared components
    """
    return float(np.dot(vec, vec))


def noise_variance_per_component(vec: np.ndarray) -> float:
    """
    Compute the noise variance per component: ||vec||^2 / len(vec).

    Args:
        vec: Input vector

    Returns:
        Noise variance per component
    """
    return noise_power(vec) / len(vec)


class NoiseLogger:
    """
    Simple logger for recording noise measurements during mult->proj cycles.
    """

    def __init__(self):
        self.data: List[Tuple[int, float, float, float, float]] = []

    def record_cycle(
        self,
        cycle: int,
        noise_before_mult: float,
        noise_after_proj: float,
        mult_noise: float,
        proj_err_var: float,
    ):
        """
        Record noise measurements for one cycle.

        Args:
            cycle: Cycle number (1-indexed)
            noise_before_mult: ||e_prev||^2 / T
            noise_after_proj: ||e_next||^2 / T
            mult_noise: ||e_mult||^2 / p
            proj_err_var: Variance of projection error (x_est - state_true)
        """
        self.data.append(
            (cycle, noise_before_mult, noise_after_proj, mult_noise, proj_err_var)
        )

    def print_summary(self, theoretical_equilibrium: float, num_header_rows: int = 10):
        """
        Print a summary of the recorded data.

        Args:
            theoretical_equilibrium: Expected equilibrium noise variance
            num_header_rows: Number of initial rows to print
        """
        if not self.data:
            print("No data recorded.")
            return

        print(
            f"{'Cycle':>5} {'Before Mult':>12} {'After Proj':>12} "
            f"{'Proj Err':>12} {'Mult Noise':>12}"
        )
        print("-" * 64)

        # Print first num_header_rows
        for i, (cycle, before, after, mult, proj_err) in enumerate(self.data):
            if i < num_header_rows:
                print(
                    f"{cycle:>5} {before:>12.6f} {after:>12.6f} "
                    f"{proj_err:>12.6f} {mult:>12.6f}"
                )

        if len(self.data) > num_header_rows:
            print("...")
            # Print last row
            cycle, before, after, mult, proj_err = self.data[-1]
            print(
                f"{cycle:>5} {before:>12.6f} {after:>12.6f} "
                f"{proj_err:>12.6f} {mult:>12.6f}"
            )

        print("-" * 64)
        print(f"Theoretical equilibrium: {theoretical_equilibrium:.6f}")

        # Compute convergence statistics from last 10 cycles
        if len(self.data) >= 10:
            last_10_after_proj = [row[2] for row in self.data[-10:]]
            mean_final = np.mean(last_10_after_proj)
            std_final = np.std(last_10_after_proj)

            if theoretical_equilibrium > 0:
                error_pct = (
                    abs(mean_final - theoretical_equilibrium)
                    / theoretical_equilibrium
                    * 100
                )
                print(
                    f"Final 10 cycles - Mean: {mean_final:.6f}, "
                    f"Std: {std_final:.6f}, Error: {error_pct:.2f}%"
                )
            else:
                print(
                    f"Final 10 cycles - Mean: {mean_final:.6f}, "
                    f"Std: {std_final:.6f}"
                )

    def save_to_csv(self, filename: str):
        """
        Save recorded data to a CSV file.

        Args:
            filename: Output CSV filename
        """
        with open(filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [
                    "Cycle",
                    "Noise_Before_Mult",
                    "Noise_After_Proj",
                    "Mult_Noise",
                    "Proj_Err_Var",
                ]
            )
            writer.writerows(self.data)
        print(f"Data saved to {filename}")


def print_matrix_stats(matrix: np.ndarray, name: str):
    """
    Print basic statistics about a matrix.

    Args:
        matrix: Input matrix
        name: Name for display
    """
    print(f"{name} shape: {matrix.shape}")
    print(f"{name} norm: {np.linalg.norm(matrix):.6f}")
    print(f"{name} condition number: {np.linalg.cond(matrix):.6f}")


def print_singular_values(matrix: np.ndarray, name: str, num_to_show: int = 5):
    """
    Print the largest and smallest singular values of a matrix.

    Args:
        matrix: Input matrix
        name: Name for display
        num_to_show: Number of largest/smallest values to show
    """
    s = np.linalg.svd(matrix, compute_uv=False)
    print(f"{name} singular values:")
    print(f"  Largest {num_to_show}: {s[:num_to_show]}")
    print(f"  Smallest {num_to_show}: {s[-num_to_show:]}")
    print(f"  Min: {s[-1]:.6f}, Max: {s[0]:.6f}, " f"Condition: {s[0]/s[-1]:.6f}")
