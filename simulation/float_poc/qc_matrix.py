"""
QC matrix generation and quadratic lifting for the mult->proj PoC.
"""

import numpy as np
from scipy.linalg import circulant
from typing import Tuple


def generate_qc_mds(k: int, p: int, seed: int) -> np.ndarray:
    """
    Generate a p x k Quasi-Cyclic MDS matrix over the reals.

    This creates a matrix with circulant structure, normalized by 1/sqrt(p)
    for proper conditioning. Each row is generated from a pseudorandom
    generating vector.

    Args:
        k: Number of columns (logical width)
        p: Number of rows (redundant rows)
        seed: Random seed for deterministic generation

    Returns:
        numpy array of shape (p, k) representing the QC-MDS matrix G
    """
    rng = np.random.default_rng(seed)

    # Generate a circulant matrix by creating the first row
    generating_vector = rng.normal(0.0, 1.0, size=p) / np.sqrt(p)

    # Create full p x p circulant matrix
    full_circulant = circulant(generating_vector)

    # Take the first k columns to get p x k matrix
    G = full_circulant[:, :k]

    return G


def lift_to_quadratic(G: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Apply the Veronese degree-2 map to lift each row of G to quadratic space.

    Each row g_i of G (which has k components) is mapped to a vector a_i
    in T-dimensional space where T = k(k+1)/2. The components of a_i are
    all unique quadratic monomials: g_i1^2, g_i1*g_i2, ..., g_ik^2.

    Args:
        G: Input matrix of shape (p, k)

    Returns:
        Tuple of (A, T) where:
        - A: System matrix of shape (p, T)
        - T: Dimension of quadratic space = k(k+1)/2
    """
    p, k = G.shape
    T = k * (k + 1) // 2

    # Pre-compute the index mapping for quadratic monomials
    # Canonical ordering: x1^2, x1*x2, ..., x1*xk, x2^2, x2*x3, ..., xk^2
    monomial_indices = []
    for i in range(k):
        for j in range(i, k):
            monomial_indices.append((i, j))

    assert len(monomial_indices) == T

    # Apply the lifting to each row
    A = np.zeros((p, T))
    for row_idx in range(p):
        g = G[row_idx, :]
        for col_idx, (i, j) in enumerate(monomial_indices):
            A[row_idx, col_idx] = g[i] * g[j]

    return A, T


def compute_system_matrices(
    k: int, p: int, seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Generate the complete system: G matrix, lifted A matrix, and pseudoinverse.

    Args:
        k: Logical width
        p: Number of redundant rows
        seed: Random seed

    Returns:
        Tuple of (G, A, A_pinv, T) where:
        - G: QC-MDS matrix of shape (p, k)
        - A: System matrix of shape (p, T)
        - A_pinv: Moore-Penrose pseudoinverse of shape (T, p)
        - T: Quadratic space dimension
    """
    # Generate QC-MDS matrix
    G = generate_qc_mds(k, p, seed)

    # Lift to quadratic space
    A, T = lift_to_quadratic(G)

    # Compute pseudoinverse
    A_pinv = np.linalg.pinv(A)

    return G, A, A_pinv, T


def validate_matrices(
    G: np.ndarray,
    A: np.ndarray,
    A_pinv: np.ndarray,
    k: int,
    p: int,
    T: int,
    verbose: bool = True,
) -> bool:
    """
    Validate that the generated matrices have the expected properties.

    Args:
        G: QC-MDS matrix
        A: System matrix
        A_pinv: Pseudoinverse
        k, p, T: Expected dimensions
        verbose: Whether to print validation results

    Returns:
        True if all validations pass
    """
    success = True

    # Check shapes
    if G.shape != (p, k):
        if verbose:
            print(f"ERROR: G shape {G.shape} != expected ({p}, {k})")
        success = False

    if A.shape != (p, T):
        if verbose:
            print(f"ERROR: A shape {A.shape} != expected ({p}, {T})")
        success = False

    if A_pinv.shape != (T, p):
        if verbose:
            print(f"ERROR: A_pinv shape {A_pinv.shape} != " f"expected ({T}, {p})")
        success = False

    # Check that T = k(k+1)/2
    expected_T = k * (k + 1) // 2
    if T != expected_T:
        if verbose:
            print(f"ERROR: T={T} != expected {expected_T}")
        success = False

    # Check that A has full row rank (assuming p <= T)
    if p <= T:
        rank_A = np.linalg.matrix_rank(A)
        if rank_A != p:
            if verbose:
                print(f"ERROR: A rank {rank_A} != expected {p}")
            success = False

    # Check pseudoinverse property: A * A_pinv * A should equal A
    reconstruction_error = np.linalg.norm(A @ A_pinv @ A - A)
    if reconstruction_error > 1e-10:
        if verbose:
            print(
                f"ERROR: Pseudoinverse reconstruction error too large: "
                f"{reconstruction_error}"
            )
        success = False

    if verbose and success:
        print("All matrix validations passed")

        # Print some statistics
        singular_values = np.linalg.svd(A, compute_uv=False)
        condition_number = singular_values[0] / singular_values[-1]

        print("Matrix statistics:")
        print(f"  G condition number: {np.linalg.cond(G):.6f}")
        print(f"  A condition number: {condition_number:.6f}")
        print(
            f"  A singular values - min: {singular_values[-1]:.6f}, "
            f"max: {singular_values[0]:.6f}"
        )
        print(f"  A rank: {np.linalg.matrix_rank(A)}")

    return success


def compute_theoretical_equilibrium(
    A: np.ndarray,
    sigma_signal: float,
    sigma_mult: float,
    T: int,
    p: int,
    use_full_formula: bool = True,
) -> float:
    """
    Compute the theoretical equilibrium noise variance from the formal analysis.

    From noise_equilibrium_proof.md, the full equilibrium formula is:
    E_eq = ((T-p)*σ_signal² + (Σ 1/s_i²)*σ_mult²) / (1 - p/T)

    The simplified approximation (projection loss only) is:
    σ_eq² ≈ ((T-p)/T) * σ_signal²

    Args:
        A: System matrix
        sigma_signal: Standard deviation of signal components
        sigma_mult: Standard deviation of multiplication noise
        T: Quadratic space dimension
        p: Number of rows in A
        use_full_formula: If True, use complete recurrence; if False, use approximation

    Returns:
        Theoretical equilibrium noise variance per component
    """
    if use_full_formula:
        # Compute singular values
        singular_values = np.linalg.svd(A, compute_uv=False)

        # Compute the sum of 1/s_i^2
        sum_inv_s_squared = np.sum(1.0 / (singular_values**2))

        # Apply the full equilibrium formula
        numerator = (T - p) * (sigma_signal**2) + sum_inv_s_squared * (sigma_mult**2)
        denominator = 1.0 - (p / T)

        E_eq_total = numerator / denominator
        sigma_eq_squared = E_eq_total / T  # Per component
    else:
        # Simplified approximation (projection loss dominance)
        sigma_eq_squared = ((T - p) / T) * (sigma_signal**2)

    return sigma_eq_squared


if __name__ == "__main__":
    # Quick test
    k, p, seed = 20, 128, 42

    print(f"Testing QC matrix generation with k={k}, p={p}, seed={seed}")
    G, A, A_pinv, T = compute_system_matrices(k, p, seed)

    print(
        f"Generated matrices: G{G.shape}, A{A.shape}, " f"A_pinv{A_pinv.shape}, T={T}"
    )

    success = validate_matrices(G, A, A_pinv, k, p, T)

    if success:
        # Compute both theoretical equilibrium values
        sigma_signal, sigma_mult = 1.0, 0.01
        theory_eq_full = compute_theoretical_equilibrium(
            A, sigma_signal, sigma_mult, T, p, use_full_formula=True
        )
        theory_eq_approx = compute_theoretical_equilibrium(
            A, sigma_signal, sigma_mult, T, p, use_full_formula=False
        )

        print(f"Theoretical equilibrium (full): {theory_eq_full:.6f}")
        print(f"Theoretical equilibrium (approx): {theory_eq_approx:.6f}")
    else:
        print("Validation failed!")
