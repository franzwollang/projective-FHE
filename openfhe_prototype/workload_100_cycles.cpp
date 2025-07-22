#include <iostream>
#include <vector>
#include <chrono>

#ifndef MOCK_OPENFHE
#include "openfhe/pke/openfhe.h"
#else
#include "mock_openfhe.h"
#endif

#include "encoding.h"
#include "projection.h"
#include "utils.h"
#include "monitor.h"
#include "params.h"

using namespace lbcrypto;

// Helper: create a packed plaintext filled with a constant value
static Plaintext MakeConstantPackedPT(CryptoContext<DCRTPoly> ctx, int64_t value) {
    std::vector<int64_t> vec(4096, value);
    return ctx->MakePackedPlaintext(vec);
}

int main(int argc, char* argv[]) {
    int cycles = 100;
    int k = 10;
    int p = 34;
    RingMode mode = RingMode::STANDARD_8192;

    // Allow optional CLI override for cycles
    for (int i = 1; i + 1 < argc; i++) {
        if (std::string(argv[i]) == "--cycles") {
            cycles = std::atoi(argv[i + 1]);
            i++;
        }
    }

    std::cout << "\n=== 100-Cycle Ciphertext×Ciphertext Workload Test ===" << std::endl;
    std::cout << "Parameters: k=" << k << ", p=" << p << ", cycles=" << cycles << std::endl;

    // Build BFV context
    auto params = CreateBFVParams(mode);
    auto context = GenCryptoContext(params);
    context->Enable(PKE);
    context->Enable(KEYSWITCH);
    context->Enable(LEVELEDSHE);

    auto keyPair = context->KeyGen();
    context->EvalMultKeyGen(keyPair.secretKey);

    uint32_t modulus = context->GetCryptoParameters()->GetPlaintextModulus();

    // Utilities
    NoiseMonitor monitor("workload_noise_log.csv", k, p, modulus);
    QCMDSEncoder encoder(context, p, k);
    ProjectionEvaluator projector(context, p, k);

    // Pre-compute common plaintext constants
    uint64_t delta_inv = mod_inverse(256, modulus);
    Plaintext delta_inv_pt = MakeConstantPackedPT(context, static_cast<int64_t>(delta_inv));

    uint64_t inv2 = mod_inverse(2, modulus);
    Plaintext inv2_pt = MakeConstantPackedPT(context, static_cast<int64_t>(inv2));

    Plaintext three_pt = MakeConstantPackedPT(context, 3);
    Plaintext one_pt   = MakeConstantPackedPT(context, 1);
    Plaintext minus_one_pt = MakeConstantPackedPT(context, static_cast<int64_t>(modulus - 1)); // −1 mod q

    // Constant ciphertext of all ones for selector complement
    Ciphertext<DCRTPoly> ones_ct = context->Encrypt(keyPair.publicKey, one_pt);

    // Initialise two logical ciphertext registers A and B with distinct data
    std::vector<Ciphertext<DCRTPoly>> ctA(k), ctB(k);
    for (int i = 0; i < k; i++) {
        std::vector<int64_t> dataA(4096, 50 + i * 2);   // 50,52,54,...
        std::vector<int64_t> dataB(4096, 75 + i * 3);   // 75,78,81,...
        ctA[i] = context->Encrypt(keyPair.publicKey, context->MakePackedPlaintext(dataA));
        ctB[i] = context->Encrypt(keyPair.publicKey, context->MakePackedPlaintext(dataB));
    }

    auto start_total = std::chrono::high_resolution_clock::now();
    std::vector<DetailedNoiseStats> all_stats;

    for (int cycle = 0; cycle < cycles; cycle++) {
        std::cout << "\n--- Cycle " << (cycle + 1) << " ---" << std::endl;

        // 1) QC-MDS expand both logical vectors to p lanes
        auto expA = encoder.encode(ctA);
        auto expB = encoder.encode(ctB);

        // 2) Ciphertext×Ciphertext lane-wise multiplication
        std::vector<Ciphertext<DCRTPoly>> mult_cts;
        mult_cts.reserve(p);
        for (int i = 0; i < p; i++) {
            mult_cts.push_back(context->EvalMult(expA[i], expB[i]));
        }

        // 3) Δ⁻¹ rescale (simulate scale management)
        std::vector<Ciphertext<DCRTPoly>> rescaled_cts;
        rescaled_cts.reserve(p);
        for (const auto& ct : mult_cts) {
            rescaled_cts.push_back(context->EvalMult(ct, delta_inv_pt));
        }

        // 4) Projection p → T
        auto T_cts = projector.project(rescaled_cts);

        // 5) Select first k diagonal terms as logical outputs
        std::vector<Ciphertext<DCRTPoly>> logical_cts(k);
        for (int i = 0; i < k; i++) {
            logical_cts[i] = T_cts[i];
        }

        // 6) Apply cycle-dependent linear transform to each ciphertext
        if (cycle % 2 == 0) { // Even: divide by 2 then add 1
            for (auto& ct : logical_cts) {
                ct = context->EvalAdd(context->EvalMult(ct, inv2_pt), one_pt);
            }
        } else {             // Odd: multiply by 3 then subtract 1
            for (auto& ct : logical_cts) {
                ct = context->EvalAdd(context->EvalMult(ct, three_pt), minus_one_pt);
            }
        }

        // 7) Encrypted selector MUX: alternate every cycle which source is kept
        //    Build selector plaintext (pattern 0/1 across slots)
        std::vector<int64_t> sel_vec(4096);
        for (int s = 0; s < 4096; s++) {
            sel_vec[s] = ( (s + cycle) % 2 ); // shift pattern each cycle for uniqueness
        }
        Plaintext sel_pt = context->MakePackedPlaintext(sel_vec);
        Ciphertext<DCRTPoly> sel_ct = context->Encrypt(keyPair.publicKey, sel_pt);
        Ciphertext<DCRTPoly> comp_sel_ct = context->EvalSub(ones_ct, sel_ct);

        std::vector<Ciphertext<DCRTPoly>> mux_results(k);
        for (int i = 0; i < k; i++) {
            auto part_main = context->EvalMult(sel_ct, logical_cts[i]);
            auto part_aux  = context->EvalMult(comp_sel_ct, ctB[i]);
            mux_results[i] = context->EvalAdd(part_main, part_aux);
        }

        // 8) Noise analysis on first ciphertext
        auto stats = monitor.analyze_ciphertext(mux_results[0], context, keyPair, cycle + 1);
        monitor.log_stats(stats);
        all_stats.push_back(stats);

        // 9) Rotate registers for next cycle: A ← mux_results,  B ← previous A
        ctB.swap(ctA);
        ctA = mux_results;
    }

    auto end_total = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end_total - start_total);
    monitor.log_performance_summary(cycles, elapsed, all_stats);

    // Decrypt and print first 8 coefficients of first ciphertext as proof of non-trivial output
    Plaintext final_pt;
    context->Decrypt(keyPair.secretKey, ctA[0], &final_pt);
    auto vals = final_pt->GetPackedValue();
    std::cout << "\nSample output coefficients (first 8 slots): ";
    for (int i = 0; i < 8; i++) {
        std::cout << vals[i] << (i < 7 ? ", " : "\n");
    }

    std::cout << "\n✅ Workload test completed successfully." << std::endl;
    return 0;
} 