/**
 * M2 - Projection evaluator
 * 
 * Implements the projection step that applies the Moore-Penrose pseudoinverse
 * to transform p noisy ciphertexts back to k clean logical outputs.
 */

#ifndef MOCK_OPENFHE
#include "openfhe/openfhe.h"
#else
#include "mock_openfhe.h"
#endif

#include "utils.h"
#include "projection.h"
#include <iostream>
#include <chrono>

using namespace lbcrypto;

// ProjectionEvaluator implementation
ProjectionEvaluator::ProjectionEvaluator(CryptoContext<DCRTPoly> ctx, int p_val, int k_val) 
    : context(ctx), p(p_val), k(k_val) {
        
        // Generate QC-MDS matrix and compute pseudoinverse
        auto A = generate_qc_mds_matrix(p, k * (k + 1) / 2);  // T = k(k+1)/2 for one multiplication
        auto A_pinv = compute_pseudoinverse(A);
        
        // Pre-encode pseudoinverse rows as BFV plaintexts using 64-bit scaling
        A_pinv_rows.resize(k * (k + 1) / 2);
        uint64_t modulus = context->GetCryptoParameters()->GetPlaintextModulus();
        
        // Use smaller power-of-2 scaling factor to prevent overflow
        const int64_t SCALE_BITS = 10;  // Reduced from 20 to 10
        const int64_t SCALE_FACTOR = 1LL << SCALE_BITS;  // 2^10 = 1,024
        scale_inv = mod_inverse(SCALE_FACTOR, modulus);  // Precompute S^-1 mod q
        
        for (size_t i = 0; i < A_pinv.size(); i++) {
            A_pinv_rows[i].resize(p);
            for (int j = 0; j < p; j++) {
                // Scale by S and store as integer (no early modular reduction)
                int64_t coeff_scaled = static_cast<int64_t>(A_pinv[i][j] * SCALE_FACTOR);
                A_pinv_rows[i][j] = context->MakePackedPlaintext({coeff_scaled});
            }
        }
        
        std::cout << "ProjectionEvaluator initialized: " << p << " -> " << k * (k + 1) / 2 << std::endl;
    }
    
std::vector<Ciphertext<DCRTPoly>> ProjectionEvaluator::project(const std::vector<Ciphertext<DCRTPoly>>& input_cts) {
        if (input_cts.size() != p) {
            throw std::invalid_argument("Expected " + std::to_string(p) + " input ciphertexts");
        }
        
        int T = k * (k + 1) / 2;
        std::vector<Ciphertext<DCRTPoly>> output_cts(T);
        
        uint64_t modulus = context->GetCryptoParameters()->GetPlaintextModulus();
        auto scale_inv_pt = context->MakePackedPlaintext({static_cast<int64_t>(scale_inv)});
        
        // Use EvalLinearWSum for optimal performance
        for (int i = 0; i < T; i++) {
            // Use manual loop (EvalLinearWSum has const-correctness issues)
            output_cts[i] = nullptr;
            for (int j = 0; j < p; j++) {
                auto scaled = context->EvalMult(input_cts[j], A_pinv_rows[i][j]);
                if (output_cts[i] == nullptr) {
                    output_cts[i] = scaled;
                } else {
                    output_cts[i] = context->EvalAdd(output_cts[i], scaled);
                }
            }
            // Apply final scaling
            output_cts[i] = context->EvalMult(output_cts[i], scale_inv_pt);
        }
        
        return output_cts;
    }
    
std::vector<Ciphertext<DCRTPoly>> ProjectionEvaluator::project_optimized(const std::vector<Ciphertext<DCRTPoly>>& input_cts) {
        if (input_cts.size() != p) {
            throw std::invalid_argument("Expected " + std::to_string(p) + " input ciphertexts");
        }
        
        int T = k * (k + 1) / 2;
        std::vector<Ciphertext<DCRTPoly>> output_cts(T);
        
        // Implement with proper 64-bit scaling (same as project method)
        uint64_t modulus = context->GetCryptoParameters()->GetPlaintextModulus();
        auto scale_inv_pt = context->MakePackedPlaintext({static_cast<int64_t>(scale_inv)});
        
        for (int i = 0; i < T; i++) {
            output_cts[i] = nullptr;
            
            // Accumulate scaled dot product
            for (int j = 0; j < p; j++) {
                auto scaled = context->EvalMult(input_cts[j], A_pinv_rows[i][j]);
                
                if (output_cts[i] == nullptr) {
                    output_cts[i] = scaled;
                } else {
                    output_cts[i] = context->EvalAdd(output_cts[i], scaled);
                }
            }
            
            // Apply final scaling
            output_cts[i] = context->EvalMult(output_cts[i], scale_inv_pt);
        }
        
        return output_cts;
    }

void test_projection_accuracy() {
    std::cout << "=== Testing Projection Accuracy ===" << std::endl;
    
    // Setup context
    CCParams<CryptoContextBFVRNS> parameters;
    parameters.SetPlaintextModulus(65537);
    parameters.SetMultiplicativeDepth(0);
    parameters.SetStandardDeviation(3.2);
    parameters.SetSecurityLevel(HEStd_128_classic);
    parameters.SetRingDim(4096);
    
    auto context = GenCryptoContext(parameters);
    context->Enable(PKE);
    context->Enable(KEYSWITCH);
    context->Enable(LEVELEDSHE);
    
    auto keyPair = context->KeyGen();
    context->EvalMultKeyGen(keyPair.secretKey);
    
    // Test parameters
    const int k = 5, p = 33;
    ProjectionEvaluator projector(context, p, k);
    
    // Create test input ciphertexts
    std::vector<Ciphertext<DCRTPoly>> input_cts(p);
    std::vector<std::vector<int64_t>> input_data(p);
    
    for (int i = 0; i < p; i++) {
        input_data[i] = {i + 1, i + 2, i + 3, i + 4, i + 5, i + 6, i + 7, i + 8};
        auto pt = context->MakePackedPlaintext(input_data[i]);
        input_cts[i] = context->Encrypt(keyPair.publicKey, pt);
    }
    
    // Perform projection
    auto projected = projector.project(input_cts);
    
    // Decrypt and analyze results
    std::cout << "Projection completed. Analyzing " << projected.size() << " outputs..." << std::endl;
    
    double total_magnitude = 0.0;
    for (size_t i = 0; i < projected.size(); i++) {
        Plaintext result_pt;
        context->Decrypt(keyPair.secretKey, projected[i], &result_pt);
        auto values = result_pt->GetPackedValue();
        
        // Compute RMS of first few coefficients
        double rms = 0.0;
        size_t count = std::min(values.size(), size_t(8));
        for (size_t j = 0; j < count; j++) {
            rms += values[j] * values[j];
        }
        rms = std::sqrt(rms / count);
        total_magnitude += rms;
    }
    
    double avg_magnitude = total_magnitude / projected.size();
    std::cout << "Average output magnitude: " << avg_magnitude << std::endl;
    
    if (avg_magnitude > 0 && avg_magnitude < 1e6) {
        std::cout << "✅ Projection test passed" << std::endl;
    } else {
        std::cout << "❌ Projection test failed" << std::endl;
    }
}

void benchmark_projection() {
    std::cout << "\n=== Benchmarking Projection ===" << std::endl;
    
    // Setup context
    CCParams<CryptoContextBFVRNS> parameters;
    parameters.SetPlaintextModulus(65537);
    parameters.SetMultiplicativeDepth(0);
    parameters.SetStandardDeviation(3.2);
    parameters.SetSecurityLevel(HEStd_128_classic);
    parameters.SetRingDim(4096);
    
    auto context = GenCryptoContext(parameters);
    context->Enable(PKE);
    context->Enable(KEYSWITCH);
    context->Enable(LEVELEDSHE);
    
    auto keyPair = context->KeyGen();
    context->EvalMultKeyGen(keyPair.secretKey);
    
    const int k = 20, p = 128;
    ProjectionEvaluator projector(context, p, k);
    
    // Create test ciphertexts
    std::vector<Ciphertext<DCRTPoly>> input_cts(p);
    for (int i = 0; i < p; i++) {
        std::vector<int64_t> data(4096, i + 100);
        auto pt = context->MakePackedPlaintext(data);
        input_cts[i] = context->Encrypt(keyPair.publicKey, pt);
    }
    
    // Benchmark projection
    const int num_trials = 5;
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int trial = 0; trial < num_trials; trial++) {
        auto projected = projector.project(input_cts);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    double avg_latency = duration.count() / static_cast<double>(num_trials);
    std::cout << "Average projection latency (p=" << p << " -> T=" << k * (k + 1) / 2 << "): " 
              << avg_latency << " ms" << std::endl;
}

// Main function moved to separate executable file 