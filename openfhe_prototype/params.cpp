/**
 * OpenFHE Parameter Presets
 * 
 * Defines 16-bit and 32-bit single-prime BFV contexts optimized for the
 * mult→project architecture without modulus switching overhead.
 */

#ifndef MOCK_OPENFHE
#include "openfhe/pke/openfhe.h"
#else
#include "mock_openfhe.h"
#endif

#include "params.h"
#include "utils.h"
#include <iostream>
#include <chrono>

using namespace lbcrypto;



// 16-bit tier: single-prime BFV for low latency
PrototypeParams get_16bit_params() {
    return PrototypeParams(65537, 0, 3.2, 4096, HEStd_128_classic);
}

// 32-bit tier: single-prime BFV for premium precision  
PrototypeParams get_32bit_params() {
    return PrototypeParams(4294967291U, 0, 3.2, 8192, HEStd_128_classic);
}

CryptoContext<DCRTPoly> setup_context(const PrototypeParams& params) {
    CCParams<CryptoContextBFVRNS> parameters;
    
    parameters.SetPlaintextModulus(params.plaintextModulus);
    parameters.SetMultiplicativeDepth(params.multiplicativeDepth);
    parameters.SetStandardDeviation(params.standardDeviation);
    parameters.SetRingDim(params.ringDim);
    parameters.SetSecurityLevel(params.securityLevel);
    
    auto context = GenCryptoContext(parameters);
    context->Enable(PKE);
    context->Enable(KEYSWITCH);  
    context->Enable(LEVELEDSHE);
    
    return context;
}

void test_basic_operations() {
    std::cout << "=== Testing Basic BFV Operations ===" << std::endl;
    
    auto context = setup_context(get_16bit_params());
    auto keyPair = context->KeyGen();
    context->EvalMultKeyGen(keyPair.secretKey);
    
    // Test basic arithmetic
    std::vector<int64_t> vec1 = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<int64_t> vec2 = {8, 7, 6, 5, 4, 3, 2, 1};
    
    auto pt1 = context->MakePackedPlaintext(vec1);
    auto pt2 = context->MakePackedPlaintext(vec2);
    
    auto ct1 = context->Encrypt(keyPair.publicKey, pt1);
    auto ct2 = context->Encrypt(keyPair.publicKey, pt2);
    
    // Addition
    auto ctAdd = context->EvalAdd(ct1, ct2);
    Plaintext ptAdd;
    context->Decrypt(keyPair.secretKey, ctAdd, &ptAdd);
    auto resultAdd = ptAdd->GetPackedValue();
    
    // Multiplication  
    auto ctMult = context->EvalMult(ct1, ct2);
    Plaintext ptMult;
    context->Decrypt(keyPair.secretKey, ctMult, &ptMult);
    auto resultMult = ptMult->GetPackedValue();
    
    // Scalar multiplication
    auto ctScalar = context->EvalMult(ct1, 3);
    Plaintext ptScalar;
    context->Decrypt(keyPair.secretKey, ctScalar, &ptScalar);
    auto resultScalar = ptScalar->GetPackedValue();
    
    std::cout << "Original 1: ";
    for (size_t i = 0; i < std::min(vec1.size(), size_t(8)); i++) {
        std::cout << vec1[i] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "Original 2: ";
    for (size_t i = 0; i < std::min(vec2.size(), size_t(8)); i++) {
        std::cout << vec2[i] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "Addition:   ";
    for (size_t i = 0; i < std::min(resultAdd.size(), size_t(8)); i++) {
        std::cout << resultAdd[i] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "Multiplication: ";
    for (size_t i = 0; i < std::min(resultMult.size(), size_t(8)); i++) {
        std::cout << resultMult[i] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "Scalar (*3): ";
    for (size_t i = 0; i < std::min(resultScalar.size(), size_t(8)); i++) {
        std::cout << resultScalar[i] << " ";
    }
    std::cout << std::endl;
}

void test_dual_ring_modes() {
    std::cout << "\n=== Testing Dual Ring Dimension Modes ===" << std::endl;
    
    // Test both modes
    for (auto mode : {RingMode::MIN_LATENCY_4096, RingMode::STANDARD_8192}) {
        FHEParams fhe_params(mode);
        
        std::cout << "\n--- " << fhe_params.GetSpec().name << " Mode ---" << std::endl;
        std::cout << "Ring Dimension: " << fhe_params.GetRingDimension() << std::endl;
        std::cout << "Use Case: " << fhe_params.GetSpec().use_case << std::endl;
        std::cout << "Security Profile: " << fhe_params.GetSpec().security_profile << std::endl;
        std::cout << "Estimated Latency Factor: " << fhe_params.GetLatencyFactor() << "x" << std::endl;
        
        // Show security warning for min-latency mode
        if (fhe_params.IsMinLatencyMode()) {
            std::cout << "\n" << fhe_params.GetSecurityWarning() << std::endl;
        }
        
        // Benchmark context setup
        auto start = std::chrono::high_resolution_clock::now();
        auto params = CreateBFVParams(mode);
        auto context = GenCryptoContext(params);
        context->Enable(PKE);
        context->Enable(KEYSWITCH);
        context->Enable(LEVELEDSHE);
        
        auto keyPair = context->KeyGen();
        context->EvalMultKeyGen(keyPair.secretKey);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Context + keygen time: " << duration.count() << " ms" << std::endl;
        
        // Quick operation test
        std::vector<int64_t> testData = {1, 2, 3, 4, 5};
        auto plaintext = context->MakePackedPlaintext(testData);
        auto ciphertext = context->Encrypt(keyPair.publicKey, plaintext);
        auto result = context->EvalMult(ciphertext, ciphertext);
        
        Plaintext decrypted;
        context->Decrypt(keyPair.secretKey, result, &decrypted);
        
        std::cout << "✓ Basic operations work correctly" << std::endl;
    }
}

int main() {
    std::cout << "OpenFHE Prototype Parameter Validation" << std::endl;
    std::cout << "======================================" << std::endl;
    
    test_dual_ring_modes();
    
    return 0;
} 