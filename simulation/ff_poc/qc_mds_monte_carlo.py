import numpy as np
from numpy.linalg import svd
import pandas as pd
import time
from scipy.linalg import circulant
import os

# Note: SymPy is no longer needed for the high-performance implementation

# Monte Carlo simulation parameters
num_trials = 10000000
p = 20
T_values = [128, 210]

# --- Matrix Generators ---


# def generate_partial_random_circulant_matrix(T, p):
#     """
#     Generates a T x p partial random circulant matrix. This is Model 1.
#     It's formed by taking the first p columns of a T x T circulant matrix.
#     """
#     generating_vector = np.random.randn(T) / np.sqrt(T)
#     full_circulant_matrix = circulant(generating_vector)
#     return full_circulant_matrix[:, :p]


# def generate_true_qc_matrix(T, p, block_size, seed):
#     """
#     Generates a T x p "true" block Quasi-Cyclic matrix. This is Model 2.
#     It's formed by a grid of smaller (block_size x block_size) circulant
#     matrices. The generating vectors for these blocks are produced
#     deterministically from the seed, simulating a PRF.
#     """
#     # Use a seeded RNG to simulate a PRF for deterministic generation
#     rng = np.random.RandomState(seed)

#     num_block_rows = T // block_size
#     num_block_cols = p // block_size

#     # Generate all the random numbers needed for the matrix at once
#     num_vectors = num_block_rows * num_block_cols
#     generating_vectors = rng.randn(num_vectors, block_size) / np.sqrt(T)

#     # Assemble the matrix from blocks
#     block_matrix = []
#     vec_idx = 0
#     for _ in range(num_block_rows):
#         row_of_blocks = []
#         for _ in range(num_block_cols):
#             gen_vec = generating_vectors[vec_idx]
#             row_of_blocks.append(circulant(gen_vec))
#             vec_idx += 1
#         block_matrix.append(row_of_blocks)

#     return np.block(block_matrix)


# # --- Simulation Execution ---

# print("Starting Monte Carlo simulation for QC matrix conditioning...")
# print(f"Number of trials per configuration: {num_trials:,}")
# start_time = time.time()

# # --- RUN 1: Partial Circulant Model ---
# print("\n--- Running Simulation for Model 1: Partial Circulant ---")
# results_pc = []
# for T in T_values:
#     print(f"\nRunning for matrix size T={T}, p={p}...")
#     print_interval = num_trials // 20 if num_trials >= 20 else 1
#     for i in range(num_trials):
#         A_T = generate_partial_random_circulant_matrix(T, p)
#         singular_vals = svd(A_T, compute_uv=False)
#         results_pc.append(
#             {
#                 "T": T,
#                 "s_max": singular_vals[0],
#                 "s_min": singular_vals[-1],
#                 "condition_number": (
#                     singular_vals[0] / singular_vals[-1]
#                     if singular_vals[-1] > 0
#                     else np.inf
#                 ),
#             }
#         )
#         if (i + 1) % print_interval == 0:
#             print(f"  ... Progress: {(i + 1) / num_trials:.0%}")

# # --- RUN 2: True Block QC Model ---
# print("\n--- Running Simulation for Model 2: True Block QC ---")
# results_qc = []
# block_sizes = {128: 4, 210: 10}  # Largest common divisors with p=20

# # Validate block sizes before starting the simulation loop
# for T_val in T_values:
#     if T_val % block_sizes[T_val] != 0 or p % block_sizes[T_val] != 0:
#         raise ValueError(f"Block size for T={T_val} must divide T and p")

# for T in T_values:
#     block_s = block_sizes[T]
#     print(
#         f"\nRunning for matrix size T={T}, p={p} "
#         f"(block size {block_s}x{block_s})..."
#     )
#     print_interval = num_trials // 20 if num_trials >= 20 else 1
#     for i in range(num_trials):
#         # Generate a new random seed for each trial to simulate a new PRF key
#         seed = int.from_bytes(os.urandom(4), "big")
#         A_T = generate_true_qc_matrix(T, p, block_s, seed)
#         singular_vals = svd(A_T, compute_uv=False)
#         results_qc.append(
#             {
#                 "T": T,
#                 "s_max": singular_vals[0],
#                 "s_min": singular_vals[-1],
#                 "condition_number": (
#                     singular_vals[0] / singular_vals[-1]
#                     if singular_vals[-1] > 0
#                     else np.inf
#                 ),
#             }
#         )
#         if (i + 1) % print_interval == 0:
#             print(f"  ... Progress: {(i + 1) / num_trials:.0%}")

# total_duration = time.time() - start_time
# print(f"\nTotal simulation finished in {total_duration:.2f} seconds.")

# # --- Display Results ---
# df_results_pc = pd.DataFrame(results_pc)
# df_results_qc = pd.DataFrame(results_qc)

# print("\n--- Results for Model 1: Partial Circulant ---")
# print(df_results_pc.describe())


# print("\n--- Results for Model 2: True Block QC ---")
# print(df_results_qc.describe())


# # --- RUN 3: Finite Field Rank Analysis (High-Performance) ---

# FHE_PRIME_Q = 65537


# def modular_inverse(a, m):
#     """Computes the modular inverse of a modulo m using the Extended Euclidean Algorithm."""
#     a = a % m
#     if np.gcd(a, m) != 1:
#         # No modular inverse exists
#         return 0

#     old_r, r = m, a
#     old_s, s = 1, 0
#     old_t, t = 0, 1

#     while r != 0:
#         quotient = old_r // r
#         old_r, r = r, old_r - quotient * r
#         old_s, s = s, old_s - quotient * s
#         old_t, t = t, old_t - quotient * t

#     return old_t % m


# def rank_mod_p(matrix, p):
#     """
#     Calculates the rank of a matrix over a finite field GF(p) using
#     NumPy for high-performance Gaussian elimination.
#     """
#     mat = np.copy(matrix).astype(np.int64)
#     rows, cols = mat.shape
#     rank = 0
#     pivot_row = 0
#     for j in range(cols):
#         if pivot_row < rows:
#             pivot = pivot_row
#             while pivot < rows and mat[pivot, j] == 0:
#                 pivot += 1

#             if pivot < rows:
#                 mat[[pivot_row, pivot]] = mat[[pivot, pivot_row]]
#                 inv = modular_inverse(mat[pivot_row, j], p)
#                 mat[pivot_row] = (mat[pivot_row] * inv) % p
#                 for i in range(rows):
#                     if i != pivot_row:
#                         mat[i] = (mat[i] - mat[i, j] * mat[pivot_row]) % p
#                 pivot_row += 1
#                 rank += 1
#     return rank


# def generate_numpy_qc_matrix(rows, cols, seed):
#     """Generates a NumPy matrix for finite field analysis."""
#     rng = np.random.RandomState(seed)
#     generating_vector = rng.randint(0, FHE_PRIME_Q, size=rows)
#     return circulant(generating_vector)[:, :cols]


# target_rank = p

# print(
#     f"\n--- Running High-Performance Simulation for QC matrix rank over GF({FHE_PRIME_Q}) ---"
# )
# print(f"Number of trials per configuration: {num_trials:,}")
# start_time_ff = time.time()

# results_ff = []
# for T in T_values:
#     print(f"\nRunning for matrix size T={T}, p={p}...")
#     rank_deficient_count = 0
#     print_interval = num_trials // 20 if num_trials >= 20 else 1
#     for i in range(num_trials):
#         seed = int.from_bytes(os.urandom(4), "big")
#         A_ff = generate_numpy_qc_matrix(T, p, seed)
#         rank = rank_mod_p(A_ff, FHE_PRIME_Q)
#         if rank < target_rank:
#             rank_deficient_count += 1
#         if (i + 1) % print_interval == 0:
#             print(f"  ... Progress: {(i + 1) / num_trials:.0%}")
#     results_ff.append({"T": T, "p": p, "deficient_count": rank_deficient_count})

# duration_ff = time.time() - start_time_ff
# print(f"\nFinite field simulation finished in {duration_ff:.2f} seconds.")

# print("\n--- Finite Field Simulation Results ---")
# for result in results_ff:
#     print(f"\nConfiguration: T={result['T']}, p={result['p']}")
#     print(f"  Matrices tested: {num_trials:,}")
#     print(f"  Target rank: {target_rank}")
#     print(f"  Rank-deficient matrices found: {result['deficient_count']}")
#     if result["deficient_count"] == 0:
#         print("  Conclusion: All generated matrices were full-rank.")
#     else:
#         failure_rate = result["deficient_count"] / num_trials
#         print(
#             "  Conclusion: Rank-deficient matrices were found with a "
#             f"failure rate of {failure_rate:.4%}."
#         )

# --- RUN 4: Initial Encoder (G) Conditioning Analysis (Normalized) ---


def generate_normalized_qc_matrix(rows, cols, seed):
    """
    Generates a NumPy matrix with normalized entries for SVD analysis.
    The entries are drawn from a standard normal distribution and scaled.
    """
    rng = np.random.RandomState(seed)
    # This normalization is standard for random matrix theory analysis
    generating_vector = rng.randn(rows) / np.sqrt(rows)
    return circulant(generating_vector)[:, :cols]


p_g = 128
k_g = 20

print(f"\n--- Running SVD Analysis for the (Normalized) p x k Encoder Matrix G ---")
print(f"Number of trials: {num_trials:,}")
print(f"Matrix dimensions: p={p_g}, k={k_g}")
start_time_g = time.time()

results_g = []
print_interval_g = num_trials // 20 if num_trials >= 20 else 1
for i in range(num_trials):
    seed = int.from_bytes(os.urandom(4), "big")
    # Generate a normalized G matrix (p x k)
    A_g = generate_normalized_qc_matrix(p_g, k_g, seed)

    # Perform SVD over the real numbers
    singular_vals = svd(A_g, compute_uv=False)

    results_g.append(
        {
            "p": p_g,
            "k": k_g,
            "s_max": singular_vals[0],
            "s_min": singular_vals[-1],
            "condition_number": (
                singular_vals[0] / singular_vals[-1]
                if singular_vals[-1] > 0
                else np.inf
            ),
        }
    )
    if (i + 1) % print_interval_g == 0:
        print(f"  ... Progress: {(i + 1) / num_trials:.0%}")

duration_g = time.time() - start_time_g
print(f"\nInitial Encoder simulation finished in {duration_g:.2f} seconds.")

df_results_g = pd.DataFrame(results_g)
print("\n--- Initial Encoder (G) Simulation Results (Normalized) ---")
print(df_results_g.describe())
