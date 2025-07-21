/**
 * Common utilities for the OpenFHE prototype implementation.
 * 
 * Includes matrix generation, pseudoinverse computation, and statistical functions.
 */

#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <string>

// QC-MDS matrix generation
std::vector<std::vector<double>> generate_qc_mds_matrix(int p, int k);

// Pseudoinverse computation (two overloads)
std::vector<std::vector<double>> compute_pseudoinverse(const std::vector<std::vector<double>>& A);
std::vector<std::vector<double>> compute_pseudoinverse(int p, int k);

// RMS computation (two overloads)
double compute_rms(const std::vector<double>& vec);
double compute_rms(const std::vector<int64_t>& vec);

// Center coefficients (two overloads)
void center_coefficients(std::vector<double>& vec);
std::vector<double> center_coefficients(const std::vector<int64_t>& vec);

// Vector statistics
void print_vector_stats(const std::vector<double>& vec, const std::string& name);

// Modular arithmetic
uint64_t mod_inverse(uint64_t a, uint64_t m);

// QC matrix health check per qc_mds_formal_analysis.md
double check_matrix_conditioning(const std::vector<std::vector<double>>& A);
bool verify_qc_matrix_health(const std::vector<std::vector<double>>& A, double max_condition = 8.0);

#endif // UTILS_H 