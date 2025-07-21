/**
 * M4 - Noise & SNR Instrumentation
 * 
 * Comprehensive monitoring system for the mult->project FHE pipeline.
 * Provides real-time noise analysis, CSV logging, and performance metrics.
 */

#ifndef MOCK_OPENFHE
#include "openfhe.h"
#else
#include "mock_openfhe.h"
#endif

#include "utils.h"
#include "monitor.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cmath>

using namespace lbcrypto;

// Implementation of DetailedNoiseStats::print()
void DetailedNoiseStats::print() const {
    std::cout << "📊 Cycle " << cycle 
              << " | RMS: " << rms_noise 
              << " | SNR: " << snr_db << " dB"
              << " | Ratio: " << noise_ratio << "x theory"
              << " | Max: " << max_coeff << std::endl;
}

// DetailedNoiseStats is declared in monitor.h
// NoiseMonitor implementation

NoiseMonitor::NoiseMonitor(const std::string& filename, int k_val, int p_val, uint32_t mod) 
    : log_file(filename), k(k_val), p(p_val), modulus(mod) {
    
    delta = static_cast<double>(modulus) / 256.0;  // Assuming t=256
    start_time = std::chrono::high_resolution_clock::now();
    
    // Open CSV file and write header
    csv_stream.open(log_file);
    csv_stream << "cycle,timestamp_ms,rms_noise,max_coeff,min_coeff,snr_db,"
               << "theoretical_noise,noise_ratio,signal_power,num_coeffs,"
               << "k,p,T,modulus,delta" << std::endl;
    
    std::cout << "📈 NoiseMonitor initialized: " << filename << std::endl;
    std::cout << "   Parameters: k=" << k << ", p=" << p << ", q=" << modulus 
              << ", Δ=" << delta << std::endl;
}

NoiseMonitor::~NoiseMonitor() {
    if (csv_stream.is_open()) {
        csv_stream.close();
    }
}

DetailedNoiseStats NoiseMonitor::analyze_ciphertext(const Ciphertext<DCRTPoly>& ct, 
                                     CryptoContext<DCRTPoly> context,
                                     const KeyPair<DCRTPoly>& keyPair,
                                     int cycle) {
        DetailedNoiseStats stats;
        stats.cycle = cycle;
        
        // Get timestamp
        auto now = std::chrono::high_resolution_clock::now();
        stats.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time);
        
        // Decrypt to analyze noise
        Plaintext pt;
        context->Decrypt(keyPair.secretKey, ct, &pt);
        auto values = pt->GetPackedValue();
        
        // Center the coefficients (remove DC component for noise analysis)
        auto centered = center_coefficients(values);
        
        // Compute statistics
        stats.rms_noise = compute_rms(centered);
        stats.num_coeffs = centered.size();
        
        double sum = 0.0;
        double min_val = centered[0];
        double max_val = centered[0];
        
        for (const auto& val : centered) {
            double dval = static_cast<double>(val);
            sum += dval * dval;
            min_val = std::min(min_val, dval);
            max_val = std::max(max_val, dval);
        }
        
        stats.min_coeff = min_val;
        stats.max_coeff = max_val;
        stats.signal_power = sum / centered.size();
        
        // Estimate theoretical noise based on FHE v3 model
        int T = k * (k + 1) / 2;  // Number of product terms
        double p_over_T = static_cast<double>(p) / T;
        
        // Simplified theoretical model: projection loss dominates
        double sigma_signal = 1000.0;  // Assume normalized signal
        stats.theoretical_noise = std::sqrt((T - p) / static_cast<double>(T)) * sigma_signal;
        stats.noise_ratio = stats.rms_noise / stats.theoretical_noise;
        
        // Compute SNR
        double signal_rms = std::sqrt(stats.signal_power);
        stats.snr_db = 20.0 * std::log10(signal_rms / stats.rms_noise);
        
        return stats;
    }

void NoiseMonitor::log_stats(const DetailedNoiseStats& stats) {
        // Write to CSV
        int T = k * (k + 1) / 2;
        csv_stream << stats.cycle << ","
                   << stats.timestamp.count() << ","
                   << stats.rms_noise << ","
                   << stats.max_coeff << ","
                   << stats.min_coeff << ","
                   << stats.snr_db << ","
                   << stats.theoretical_noise << ","
                   << stats.noise_ratio << ","
                   << stats.signal_power << ","
                   << stats.num_coeffs << ","
                   << k << "," << p << "," << T << ","
                   << modulus << "," << delta << std::endl;
        
        csv_stream.flush();  // Ensure data is written immediately
        
        // Print to console
        stats.print();
    }

void NoiseMonitor::log_performance_summary(int total_cycles, 
                            std::chrono::milliseconds total_time,
                            const std::vector<DetailedNoiseStats>& all_stats) {
        
        std::cout << "\n📈 PERFORMANCE SUMMARY" << std::endl;
        std::cout << "======================" << std::endl;
        std::cout << "Total cycles: " << total_cycles << std::endl;
        std::cout << "Total time: " << total_time.count() << " ms" << std::endl;
        std::cout << "Average cycle time: " << (total_time.count() / total_cycles) << " ms" << std::endl;
        std::cout << "Throughput: " << (1000.0 * total_cycles / total_time.count()) << " cycles/sec" << std::endl;
        
        if (!all_stats.empty()) {
            // Compute noise statistics
            double sum_rms = 0.0, sum_snr = 0.0, sum_ratio = 0.0;
            double min_rms = all_stats[0].rms_noise;
            double max_rms = all_stats[0].rms_noise;
            
            for (const auto& stat : all_stats) {
                sum_rms += stat.rms_noise;
                sum_snr += stat.snr_db;
                sum_ratio += stat.noise_ratio;
                min_rms = std::min(min_rms, stat.rms_noise);
                max_rms = std::max(max_rms, stat.rms_noise);
            }
            
            double avg_rms = sum_rms / all_stats.size();
            double avg_snr = sum_snr / all_stats.size();
            double avg_ratio = sum_ratio / all_stats.size();
            
            std::cout << "\nNOISE ANALYSIS:" << std::endl;
            std::cout << "Average RMS noise: " << avg_rms << std::endl;
            std::cout << "RMS range: [" << min_rms << ", " << max_rms << "]" << std::endl;
            std::cout << "Average SNR: " << avg_snr << " dB" << std::endl;
            std::cout << "Average noise ratio: " << avg_ratio << "x theoretical" << std::endl;
            
            // Noise equilibrium analysis
            if (all_stats.size() >= 5) {
                double early_avg = 0.0, late_avg = 0.0;
                int early_count = std::min(3, static_cast<int>(all_stats.size()) / 3);
                int late_count = std::min(3, static_cast<int>(all_stats.size()) / 3);
                
                for (int i = 0; i < early_count; i++) {
                    early_avg += all_stats[i].rms_noise;
                }
                for (int i = all_stats.size() - late_count; i < all_stats.size(); i++) {
                    late_avg += all_stats[i].rms_noise;
                }
                
                early_avg /= early_count;
                late_avg /= late_count;
                
                double equilibrium_change = std::abs(late_avg - early_avg) / early_avg;
                
                std::cout << "Equilibrium stability: " << (equilibrium_change * 100) << "% change" << std::endl;
                std::cout << (equilibrium_change < 0.1 ? "✅ Stable equilibrium achieved" 
                                                       : "⚠️ Noise still settling") << std::endl;
            }
        }
        
        std::cout << "\n📊 Detailed log saved to: " << log_file << std::endl;
    }

// Example usage function
void test_noise_monitoring() {
    std::cout << "Testing NoiseMonitor functionality..." << std::endl;
    
    // This would be called from the main pipeline
    NoiseMonitor monitor("noise_analysis.csv", 10, 34, 65537);
    
    std::cout << "✅ NoiseMonitor test completed" << std::endl;
} 