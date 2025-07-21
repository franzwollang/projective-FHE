/**
 * M1 - Encoding / QC-MDS rotations
 * 
 * Implements QC-MDS encoding using OpenFHE's EvalMult and EvalAdd operations
 * for linear combinations, avoiding the complexity of EvalAtIndexBatch rotations
 * in this prototype version.
 */

#ifndef MOCK_OPENFHE
#include "openfhe/openfhe.h"
#else
#include "mock_openfhe.h"
#endif

#include "utils.h"
#include "encoding.h"
#include <iostream>
#include <chrono>

using namespace lbcrypto;

// QCMDSEncoder implementation
QCMDSEncoder::QCMDSEncoder(CryptoContext<DCRTPoly> ctx, int p_val, int k_val) 
    : context(ctx), k(k_val), p(p_val) {
        
        // Generate QC-MDS matrix
        G = generate_qc_mds_matrix(p, k);
        G_pinv = compute_pseudoinverse(G);
        
        // Cache plaintext coefficients for performance
        uint64_t modulus = context->GetCryptoParameters()->GetPlaintextModulus();
        const int64_t SCALE_FACTOR = 1000;  // Fixed scaling for consistency
        
        // Cache G matrix coefficients as plaintexts
        G_plaintexts.resize(p);
        for (int i = 0; i < p; i++) {
            G_plaintexts[i].resize(k);
            for (int j = 0; j < k; j++) {
                int64_t coeff_int = static_cast<int64_t>(G[i][j] * SCALE_FACTOR) % modulus;
                G_plaintexts[i][j] = context->MakePackedPlaintext({coeff_int});
            }
        }
        
        // Cache G_pinv matrix coefficients as plaintexts
        G_pinv_plaintexts.resize(k);
        for (int i = 0; i < k; i++) {
            G_pinv_plaintexts[i].resize(p);
            for (int j = 0; j < p; j++) {
                int64_t coeff_int = static_cast<int64_t>(G_pinv[i][j] * SCALE_FACTOR) % modulus;
                G_pinv_plaintexts[i][j] = context->MakePackedPlaintext({coeff_int});
            }
        }
        
        std::cout << "QCMDSEncoder initialized: " << p << "x" << k << " matrix" << std::endl;
    }
    
std::vector<Ciphertext<DCRTPoly>> QCMDSEncoder::encode(const std::vector<Ciphertext<DCRTPoly>>& logical_cts) {
        if (logical_cts.size() != k) {
            throw std::invalid_argument("Expected " + std::to_string(k) + " logical ciphertexts");
        }
        
        std::vector<Ciphertext<DCRTPoly>> encoded_cts(p);
        
        // Use cached plaintexts and EvalLinearWSum for optimal performance
        for (int i = 0; i < p; i++) {
            // Single fused operation: ct_i = sum(G_plaintexts[i][j] * logical_cts[j])
            // OpenFHE EvalLinearWSum requires non-const vector, so fallback to manual loop for now
            // TODO: Investigate if we can cast to ReadOnlyCiphertext safely
            encoded_cts[i] = nullptr;
            for (int j = 0; j < k; j++) {
                auto scaled = context->EvalMult(logical_cts[j], G_plaintexts[i][j]);
                if (encoded_cts[i] == nullptr) {
                    encoded_cts[i] = scaled;
                } else {
                    encoded_cts[i] = context->EvalAdd(encoded_cts[i], scaled);
                }
            }
        }
        
        return encoded_cts;
    }
    
std::vector<Ciphertext<DCRTPoly>> QCMDSEncoder::decode(const std::vector<Ciphertext<DCRTPoly>>& encoded_cts) {
        if (encoded_cts.size() != p) {
            throw std::invalid_argument("Expected " + std::to_string(p) + " encoded ciphertexts");
        }
        
        std::vector<Ciphertext<DCRTPoly>> logical_cts(k);
        
        // Use cached plaintexts for optimal performance
         for (int j = 0; j < k; j++) {
             // Single fused operation: ct_j = sum(G_pinv_plaintexts[j][i] * encoded_cts[i])
            // Use manual loop (EvalLinearWSum has const-correctness issues)
                logical_cts[j] = nullptr;
                for (int i = 0; i < p; i++) {
                    auto scaled = context->EvalMult(encoded_cts[i], G_pinv_plaintexts[j][i]);
                    if (logical_cts[j] == nullptr) {
                        logical_cts[j] = scaled;
                    } else {
                        logical_cts[j] = context->EvalAdd(logical_cts[j], scaled);
                    }
                }
         }
        
        return logical_cts;
    }

void test_qc_mds_roundtrip() {
    std::cout << "=== Testing QC-MDS Encode/Decode Roundtrip ===" << std::endl;
    
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
    
    // Create test data
    const int k = 5, p = 10;
    QCMDSEncoder encoder(context, p, k);
    
    std::vector<Ciphertext<DCRTPoly>> logical_cts(k);
    std::vector<std::vector<int64_t>> original_data(k);
    
    for (int i = 0; i < k; i++) {
        original_data[i] = {i+1, i+2, i+3, i+4, i+5, i+6, i+7, i+8};
        auto pt = context->MakePackedPlaintext(original_data[i]);
        logical_cts[i] = context->Encrypt(keyPair.publicKey, pt);
    }
    
    // Test encode -> decode
    auto encoded = encoder.encode(logical_cts);
    auto decoded = encoder.decode(encoded);
    
    // Verify roundtrip accuracy
    std::cout << "Checking roundtrip accuracy..." << std::endl;
    double max_error = 0.0;
    
    for (int i = 0; i < k; i++) {
        Plaintext original_pt;
        context->Decrypt(keyPair.secretKey, logical_cts[i], &original_pt);
        Plaintext decoded_pt;
        context->Decrypt(keyPair.secretKey, decoded[i], &decoded_pt);
        
        auto orig_vals = original_pt->GetPackedValue();
        auto dec_vals = decoded_pt->GetPackedValue();
        
        for (size_t j = 0; j < std::min(orig_vals.size(), dec_vals.size()); j++) {
            double error = std::abs(static_cast<double>(orig_vals[j] - dec_vals[j]));
            max_error = std::max(max_error, error);
        }
    }
    
    std::cout << "Maximum roundtrip error: " << max_error << std::endl;
    if (max_error < 100) { // Allow some quantization error
        std::cout << "✅ QC-MDS roundtrip test passed" << std::endl;
    } else {
        std::cout << "❌ QC-MDS roundtrip test failed" << std::endl;
    }
}

void benchmark_qc_mds_encoding() {
    std::cout << "\n=== Benchmarking QC-MDS Encoding ===" << std::endl;
    
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
    QCMDSEncoder encoder(context, p, k);
    
    // Create test ciphertexts
    std::vector<Ciphertext<DCRTPoly>> logical_cts(k);
    for (int i = 0; i < k; i++) {
        std::vector<int64_t> data(4096, i + 1);
        auto pt = context->MakePackedPlaintext(data);
        logical_cts[i] = context->Encrypt(keyPair.publicKey, pt);
    }
    
    // Benchmark encoding
    const int num_trials = 10;
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int trial = 0; trial < num_trials; trial++) {
        auto encoded = encoder.encode(logical_cts);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    double avg_latency = duration.count() / static_cast<double>(num_trials);
    std::cout << "Average encoding latency (k=" << k << ", p=" << p << "): " 
              << avg_latency << " ms" << std::endl;
}

// Main function moved to separate executable file 
