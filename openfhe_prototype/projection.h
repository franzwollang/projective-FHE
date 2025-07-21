#pragma once

#ifndef MOCK_OPENFHE
#include "openfhe.h"
#else
#include "mock_openfhe.h"
#endif

#include <vector>

using namespace lbcrypto;

class ProjectionEvaluator {
private:
    CryptoContext<DCRTPoly> context;
    int k, p;
    std::vector<std::vector<Plaintext>> A_pinv_rows;  // Pre-encoded pseudoinverse rows
    uint64_t scale_inv;  // Precomputed S^-1 mod q for final scaling

public:
    ProjectionEvaluator(CryptoContext<DCRTPoly> ctx, int p_val, int k_val);
    
    // Project p input ciphertexts to T output ciphertexts using A_pinv
    std::vector<Ciphertext<DCRTPoly>> project(const std::vector<Ciphertext<DCRTPoly>>& input_cts);
    
    // Alternative implementation using EvalLinearWSum if available
    std::vector<Ciphertext<DCRTPoly>> project_optimized(const std::vector<Ciphertext<DCRTPoly>>& input_cts);
}; 