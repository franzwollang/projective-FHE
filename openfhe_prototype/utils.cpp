/**
 * Common utilities for the OpenFHE prototype implementation.
 */

#ifndef MOCK_OPENFHE
#include "openfhe.h"
#else
#include "mock_openfhe.h"
#endif

#include "utils.h"
#include "params.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <random>
#include <limits>
#include <Eigen/Dense>
#include <Eigen/SVD>

// Factory function implementation
#ifndef MOCK_OPENFHE
CCParams<CryptoContextBFVRNS> CreateBFVParams(RingMode mode) {
    FHEParams fhe_params(mode);
    
    CCParams<CryptoContextBFVRNS> parameters;
    parameters.SetPlaintextModulus(fhe_params.GetPlaintextModulus());
    parameters.SetMultiplicativeDepth(1);  // Single level for mult->project
    parameters.SetStandardDeviation(3.2);  // Standard BFV noise
    parameters.SetRingDim(fhe_params.GetRingDimension());
    parameters.SetSecurityLevel(fhe_params.GetSecurityLevel());
    
    return parameters;
}
#endif

std::vector<std::vector<double>> generate_qc_mds_matrix(int p, int k) {
    std::cout << "Generating " << p << "x" << k << " PRF-based QC-MDS matrix..." << std::endl;
    
    // PRF-based QC generator per qc_mds_formal_analysis.md
    const uint64_t q = 65537;  // Same modulus as FHE context
    
    // Generate cryptographic seed
    std::random_device rd;
    std::mt19937_64 prng(rd());
    uint64_t seed = prng();
    
    // Use seed to initialize PRF (MT19937 with fixed seed for reproducibility)
    std::mt19937_64 prf(seed);
    
    // Block size: use k for optimal conditioning (per formal analysis)
    int block_size = k;
    int num_blocks_p = (p + block_size - 1) / block_size;
    int num_blocks_k = (k + block_size - 1) / block_size;
    
    std::cout << "  Block structure: " << block_size << "x" << block_size 
              << " blocks (" << num_blocks_p << "x" << num_blocks_k << " layout)" << std::endl;
    
    // Generate base circulant elements using field arithmetic
    // Each block's first row is generated as powers of a primitive element
    std::vector<std::vector<uint64_t>> block_generators(num_blocks_p * num_blocks_k);
    
    for (int block_idx = 0; block_idx < num_blocks_p * num_blocks_k; block_idx++) {
        block_generators[block_idx].resize(block_size);
        
        // Generate primitive element for this block using PRF
        uint64_t primitive = (prf() % (q - 2)) + 2;  // Range [2, q-1]
        
        // Generate first row as powers: [α^0, α^1, α^2, ..., α^(block_size-1)]
        uint64_t power = 1;
        for (int i = 0; i < block_size; i++) {
            block_generators[block_idx][i] = power;
            power = (power * primitive) % q;
        }
    }
    
    // Assemble p×k matrix from block-circulant structure
    std::vector<std::vector<double>> matrix(p, std::vector<double>(k));
    
    for (int i = 0; i < p; i++) {
        for (int j = 0; j < k; j++) {
            // Determine which block and position within block
            int block_i = i / block_size;
            int block_j = j / block_size;
            int local_i = i % block_size;
            int local_j = j % block_size;
            
            // Block index in the generator array
            int block_idx = block_i * num_blocks_k + block_j;
            if (block_idx >= block_generators.size()) {
                // Padding for incomplete blocks
                matrix[i][j] = (i == j) ? 1.0 : 0.0;
                continue;
            }
            
            // Circulant indexing: shift by local_i
            int circ_idx = (local_j + local_i) % block_size;
            uint64_t field_elem = block_generators[block_idx][circ_idx];
            
            // Convert to double and normalize to prevent overflow
            // Scale to range [0.1, 2.0] to maintain good conditioning
            double normalized = 0.1 + (1.9 * field_elem) / (q - 1);
            matrix[i][j] = normalized;
        }
    }
    
    std::cout << "  PRF seed: 0x" << std::hex << seed << std::dec << std::endl;
    
    // Health check: verify conditioning meets formal analysis requirements
    if (!verify_qc_matrix_health(matrix, 8.0)) {
        std::cout << "  WARNING: Matrix conditioning exceeds formal analysis bounds!" << std::endl;
        std::cout << "  Consider regenerating with different seed or adjusting parameters." << std::endl;
    }
    
    return matrix;
}

std::vector<std::vector<double>> compute_pseudoinverse(const std::vector<std::vector<double>>& A) {
    int m = A.size();    // rows
    int n = A[0].size(); // columns
    
    std::cout << "Computing SVD-based pseudoinverse of " << m << "x" << n << " matrix..." << std::endl;
    
    // Convert std::vector to Eigen matrix
    Eigen::MatrixXd A_eigen(m, n);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            A_eigen(i, j) = A[i][j];
        }
    }
    
    // Compute SVD: A = U * Σ * V^T
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A_eigen, Eigen::ComputeFullU | Eigen::ComputeFullV);
    
    // Get SVD components
    Eigen::MatrixXd U = svd.matrixU();
    Eigen::MatrixXd V = svd.matrixV();
    Eigen::VectorXd singular_values = svd.singularValues();
    
    // Compute pseudoinverse: A^+ = V * Σ^+ * U^T
    // where Σ^+ has 1/σᵢ for σᵢ > tolerance, 0 otherwise
    const double tolerance = 1e-10;
    
    // Create the pseudoinverse manually to handle dimension matching
    int rank = std::min(m, n);
    Eigen::MatrixXd A_pinv_eigen = Eigen::MatrixXd::Zero(n, m);
    
    for (int i = 0; i < rank; i++) {
        if (singular_values(i) > tolerance) {
            double sigma_inv = 1.0 / singular_values(i);
            // A^+ += σ^+_i * v_i * u_i^T
            A_pinv_eigen += sigma_inv * V.col(i) * U.col(i).transpose();
        }
    }
    
    // Convert back to std::vector
    std::vector<std::vector<double>> A_pinv(n, std::vector<double>(m));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            A_pinv[i][j] = A_pinv_eigen(i, j);
        }
    }
    
    // Log some diagnostic info
    std::cout << "  Rank: " << (singular_values.array() > tolerance).count() 
              << "/" << std::min(m, n) << std::endl;
    
    // Compute condition number safely
    double min_singular = singular_values(std::min(m,n)-1);
    if (min_singular > tolerance) {
        std::cout << "  Condition number: " << singular_values(0) / min_singular << std::endl;
    } else {
        std::cout << "  Condition number: Inf (near-singular matrix)" << std::endl;
    }
    
    return A_pinv;
}

// Overload for (int, int) signature used in projection.cpp
std::vector<std::vector<double>> compute_pseudoinverse(int p, int k) {
    // Generate a simple matrix and return its pseudoinverse
    auto A = generate_qc_mds_matrix(p, k);
    return compute_pseudoinverse(A);
}

double compute_rms(const std::vector<double>& vec) {
    if (vec.empty()) return 0.0;
    
    double sum_sq = 0.0;
    for (double val : vec) {
        sum_sq += val * val;
    }
    
    return std::sqrt(sum_sq / vec.size());
}

// Overload for int64_t vector
double compute_rms(const std::vector<int64_t>& vec) {
    if (vec.empty()) return 0.0;
    
    double sum_sq = 0.0;
    for (int64_t val : vec) {
        double d_val = static_cast<double>(val);
        sum_sq += d_val * d_val;
    }
    
    return std::sqrt(sum_sq / vec.size());
}

void center_coefficients(std::vector<double>& vec) {
    if (vec.empty()) return;
    
    // Compute mean
    double mean = 0.0;
    for (double val : vec) {
        mean += val;
    }
    mean /= vec.size();
    
    // Subtract mean from each element
    for (double& val : vec) {
        val -= mean;
    }
}

// Return centered copy for int64_t vectors
std::vector<double> center_coefficients(const std::vector<int64_t>& vec) {
    std::vector<double> result;
    result.reserve(vec.size());
    
    if (vec.empty()) return result;
    
    // Convert to double and compute mean
    double mean = 0.0;
    for (int64_t val : vec) {
        double d_val = static_cast<double>(val);
        result.push_back(d_val);
        mean += d_val;
    }
    mean /= vec.size();
    
    // Subtract mean
    for (double& val : result) {
        val -= mean;
    }
    
    return result;
}

void print_vector_stats(const std::vector<double>& vec, const std::string& name) {
    if (vec.empty()) {
        std::cout << name << ": (empty)" << std::endl;
        return;
    }
    
    double min_val = *std::min_element(vec.begin(), vec.end());
    double max_val = *std::max_element(vec.begin(), vec.end());
    double rms = compute_rms(vec);
    
    std::cout << name << ": min=" << min_val << ", max=" << max_val 
              << ", rms=" << rms << ", size=" << vec.size() << std::endl;
}

// Extended Euclidean Algorithm for modular inverse
uint64_t mod_inverse(uint64_t a, uint64_t m) {
    if (m == 1) return 0;
    
    int64_t m0 = m, x0 = 0, x1 = 1;
    
    while (a > 1) {
        int64_t q = a / m;
        int64_t t = m;
        
        m = a % m;
        a = t;
        t = x0;
        
        x0 = x1 - q * x0;
        x1 = t;
    }
    
    if (x1 < 0) x1 += m0;
    
    return static_cast<uint64_t>(x1);
}

// QC matrix health check per qc_mds_formal_analysis.md
double check_matrix_conditioning(const std::vector<std::vector<double>>& A) {
    int m = A.size();
    int n = A[0].size();
    
    // Convert to Eigen matrix
    Eigen::MatrixXd A_eigen(m, n);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            A_eigen(i, j) = A[i][j];
        }
    }
    
    // Compute SVD for condition number
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A_eigen, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::VectorXd singular_values = svd.singularValues();
    
    if (singular_values.size() == 0) return std::numeric_limits<double>::infinity();
    
    double sigma_max = singular_values(0);
    double sigma_min = singular_values(singular_values.size() - 1);
    
    if (sigma_min < 1e-15) return std::numeric_limits<double>::infinity();
    
    return sigma_max / sigma_min;
}

bool verify_qc_matrix_health(const std::vector<std::vector<double>>& A, double max_condition) {
    double condition = check_matrix_conditioning(A);
    
    std::cout << "  Matrix health check: κ = " << condition;
    if (condition <= max_condition) {
        std::cout << " ✓ PASS (≤ " << max_condition << ")" << std::endl;
        return true;
    } else {
        std::cout << " ✗ FAIL (> " << max_condition << ")" << std::endl;
        return false;
    }
} 