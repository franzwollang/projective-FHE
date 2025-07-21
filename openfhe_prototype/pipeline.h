#pragma once

#ifndef MOCK_OPENFHE
#include "openfhe.h"
#else
#include "mock_openfhe.h"
#endif

#include <vector>
#include <memory>

using namespace lbcrypto;

// Forward declarations
class QCMDSEncoder;
class ProjectionEvaluator;
class NoiseMonitor;

struct NoiseStats {
    double rms;
    double max_coeff;
    double snr;
    
    void print() const;
};

class FHEPipeline {
private:
    CryptoContext<DCRTPoly> context;
    KeyPair<DCRTPoly> keyPair;
    int k, p;
    uint32_t modulus;
    RingMode mode;
    
    std::unique_ptr<QCMDSEncoder> encoder;
    std::unique_ptr<ProjectionEvaluator> projector;
#ifdef ENABLE_DIAGNOSTICS
    std::unique_ptr<NoiseMonitor> monitor;
#endif

public:
    FHEPipeline(int k_val, int p_val, RingMode mode_val = RingMode::STANDARD_8192, uint32_t seed = 42);
    
    void setup_context();
    std::vector<Ciphertext<DCRTPoly>> inject_noise(const std::vector<Ciphertext<DCRTPoly>>& cts, int noise_level = 32);
    std::vector<Ciphertext<DCRTPoly>> self_multiply(const std::vector<Ciphertext<DCRTPoly>>& cts);
    std::vector<Ciphertext<DCRTPoly>> delta_rescale(const std::vector<Ciphertext<DCRTPoly>>& cts);
    std::vector<Ciphertext<DCRTPoly>> expand_to_T(const std::vector<Ciphertext<DCRTPoly>>& k_cts);
    std::vector<Ciphertext<DCRTPoly>> select_k_outputs(const std::vector<Ciphertext<DCRTPoly>>& T_cts);
    std::vector<Ciphertext<DCRTPoly>> project_T_to_k(const std::vector<Ciphertext<DCRTPoly>>& T_cts);
    
    // NEW CORRECTED METHODS FOR PROPER MULT->PROJECT CYCLE
    std::vector<Ciphertext<DCRTPoly>> mds_expand_k_to_p(const std::vector<Ciphertext<DCRTPoly>>& logical_cts);
    std::vector<Ciphertext<DCRTPoly>> self_multiply_p_lanes(const std::vector<Ciphertext<DCRTPoly>>& expanded_cts);
    std::vector<Ciphertext<DCRTPoly>> project_p_to_T(const std::vector<Ciphertext<DCRTPoly>>& p_cts);
    std::vector<Ciphertext<DCRTPoly>> select_k_from_T(const std::vector<Ciphertext<DCRTPoly>>& T_cts);
    
    void run_pipeline(int num_cycles = 10);
    NoiseStats analyze_noise(const std::vector<Ciphertext<DCRTPoly>>& cts, int cycle);
    void simulate_noise_reduction(std::vector<Ciphertext<DCRTPoly>>& cts);
    void stress_test(int num_cycles = 100);
}; 