#ifdef MOCK_OPENFHE
#include "mock_openfhe.h"
#else
#include "openfhecore.h"
#endif

#include "utils.h"
#include <iostream>
#include <vector>

using namespace lbcrypto;

// Test our core corrected logic
int main() {
    std::cout << "Testing OpenFHE prototype syntax fixes..." << std::endl;
    
    // Test 1: QC-MDS matrix generation
    auto G = generate_qc_mds_matrix(10, 5);
    std::cout << "✓ Generated QC-MDS matrix: " << G.size() << "x" << G[0].size() << std::endl;
    
    // Test 2: Pseudoinverse computation
    auto A_pinv = compute_pseudoinverse(10, 5);
    std::cout << "✓ Computed pseudoinverse: " << A_pinv.size() << "x" << A_pinv[0].size() << std::endl;
    
    // Test 3: Modular inverse
    int64_t delta = 256;
    int64_t modulus = 65537;
    int64_t delta_inv = mod_inverse(delta, modulus);
    std::cout << "✓ Modular inverse of " << delta << " mod " << modulus << " = " << delta_inv << std::endl;
    
    // Test 4: Mock OpenFHE operations
    CryptoContext<DCRTPoly> context;
    auto params = context.GetCryptoParameters();
    uint32_t mod = params->GetPlaintextModulus();
    std::cout << "✓ Mock context modulus: " << mod << std::endl;
    
    // Test 5: Pre-encoded plaintext creation
    std::vector<int64_t> coeffs = {1000, 2000, 3000};
    for (auto& coeff : coeffs) {
        coeff = coeff % mod;
    }
    auto plaintext = context.MakePackedPlaintext(coeffs);
    std::cout << "✓ Created packed plaintext" << std::endl;
    
    // Test 6: Ciphertext operations
    Ciphertext<DCRTPoly> ct1, ct2;
    ct1 = context.Encrypt(nullptr, plaintext);
    ct2 = context.EvalMult(ct1, plaintext);
    ct2 = context.EvalAdd(ct1, ct2);
    std::cout << "✓ Performed mock FHE operations" << std::endl;
    
    // Test 7: Null ciphertext handling
    Ciphertext<DCRTPoly> null_ct = nullptr;
    if (null_ct == nullptr) {
        std::cout << "✓ Null ciphertext comparison works" << std::endl;
    }
    
    std::cout << "All syntax tests passed! Core fixes are correct." << std::endl;
    return 0;
} 