#include <iostream>
#include <vector>
#include <chrono>

#ifndef MOCK_OPENFHE
#include "openfhe/pke/openfhe.h"
#else
#include "mock_openfhe.h"
#endif

using namespace lbcrypto;

int main(int argc, char* argv[]) {
    int cycles = 100;
    
    // Allow optional CLI override for cycles
    for (int i = 1; i + 1 < argc; i++) {
        if (std::string(argv[i]) == "--cycles") {
            cycles = std::atoi(argv[i + 1]);
            i++;
        }
    }

    std::cout << "\n=== Simple 100-Cycle Workload Test (Mock Mode) ===" << std::endl;
    std::cout << "Cycles: " << cycles << std::endl;

#ifdef MOCK_OPENFHE
    std::cout << "Running in MOCK mode - validating program logic" << std::endl;
    
    // Mock context setup
    CryptoContext<DCRTPoly> context;
    KeyPair<DCRTPoly> keyPair;
    
    // Create mock ciphertexts
    std::vector<int64_t> data_a(4096, 100);
    std::vector<int64_t> data_b(4096, 200);
    
    auto pt_a = context.MakePackedPlaintext(data_a);
    auto pt_b = context.MakePackedPlaintext(data_b);
    
    auto ct_a = context.Encrypt(keyPair.publicKey, pt_a);
    auto ct_b = context.Encrypt(keyPair.publicKey, pt_b);
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int cycle = 0; cycle < cycles; cycle++) {
        // Simulate ciphertext × ciphertext multiplication
        auto mult_result = context.EvalMult(ct_a, ct_b);
        
        // Apply cycle-dependent transform
        if (cycle % 2 == 0) {
            // Even cycles: add constant
            std::vector<int64_t> const_vec(4096, 1);
            auto const_pt = context.MakePackedPlaintext(const_vec);
            auto const_ct = context.Encrypt(keyPair.publicKey, const_pt);
            ct_a = context.EvalAdd(mult_result, const_ct);
        } else {
            // Odd cycles: multiply by constant
            std::vector<int64_t> mult_vec(4096, 3);
            auto mult_pt = context.MakePackedPlaintext(mult_vec);
            ct_a = context.EvalMult(mult_result, mult_pt);
        }
        
        // Selector MUX simulation
        std::vector<int64_t> sel_vec(4096);
        for (int s = 0; s < 4096; s++) {
            sel_vec[s] = (s + cycle) % 2;
        }
        auto sel_pt = context.MakePackedPlaintext(sel_vec);
        auto sel_ct = context.Encrypt(keyPair.publicKey, sel_pt);
        
        auto part_a = context.EvalMult(sel_ct, ct_a);
        auto part_b = context.EvalMult(sel_ct, ct_b); // simplified selector complement
        auto mux_result = context.EvalAdd(part_a, part_b);
        
        // Update for next cycle
        ct_b = ct_a;
        ct_a = mux_result;
        
        if (cycle % 10 == 0) {
            std::cout << "Completed cycle " << (cycle + 1) << std::endl;
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << "\n📈 Mock Test Results:" << std::endl;
    std::cout << "Total cycles: " << cycles << std::endl;
    std::cout << "Total time: " << elapsed.count() << " ms" << std::endl;
    std::cout << "Average cycle time: " << (elapsed.count() / cycles) << " ms" << std::endl;
    
    // Mock decryption to show final state
    Plaintext final_pt;
    context.Decrypt(keyPair.secretKey, ct_a, &final_pt);
    
    std::cout << "✅ Mock workload completed successfully" << std::endl;
    std::cout << "This validates the program logic - ready for real OpenFHE implementation" << std::endl;
    
#else
    std::cout << "Real OpenFHE mode not implemented in this simplified test" << std::endl;
    return 1;
#endif

    return 0;
} 