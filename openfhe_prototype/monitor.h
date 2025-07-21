#pragma once

#ifndef MOCK_OPENFHE
#include "openfhe.h"
#else
#include "mock_openfhe.h"
#endif

#include <vector>
#include <chrono>
#include <string>
#include <fstream>

using namespace lbcrypto;

struct DetailedNoiseStats {
    int cycle;
    double rms_noise;
    double max_coeff;
    double min_coeff;
    double snr_db;
    double theoretical_noise;
    double noise_ratio;  // empirical/theoretical
    double signal_power;
    int num_coeffs;
    std::chrono::milliseconds timestamp;
    
    void print() const;
};

class NoiseMonitor {
private:
    std::string log_file;
    std::ofstream csv_stream;
    std::chrono::high_resolution_clock::time_point start_time;
    int k, p;
    uint32_t modulus;
    double delta;
    
public:
    NoiseMonitor(const std::string& filename, int k_val, int p_val, uint32_t mod);
    ~NoiseMonitor();
    
    DetailedNoiseStats analyze_ciphertext(const Ciphertext<DCRTPoly>& ct, 
                                         CryptoContext<DCRTPoly> context,
                                         const KeyPair<DCRTPoly>& keyPair,
                                         int cycle);
    
    void log_stats(const DetailedNoiseStats& stats);
    
    void log_performance_summary(int total_cycles, 
                                std::chrono::milliseconds total_time,
                                const std::vector<DetailedNoiseStats>& all_stats);
};

// Test function
void test_noise_monitoring(); 