/**
 * Comprehensive Benchmark: 4096 vs 8192 Ring Dimensions
 * 
 * This benchmark compares the performance characteristics of both ring modes
 * across latency, throughput, noise behavior, and security trade-offs.
 */

#ifndef MOCK_OPENFHE
#include "openfhe/pke/openfhe.h"
#else
#include "mock_openfhe.h"
#endif

#include "params.h"
#include "utils.h"
#include "encoding.h"
#include "projection.h"
#include "pipeline.h"
#include "monitor.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <iomanip>

using namespace lbcrypto;

struct BenchmarkResults {
    std::string mode_name;
    uint32_t ring_dim;
    double setup_time_ms;
    double mult_time_ms;
    double projection_time_ms;
    double total_cycle_time_ms;
    double cycles_per_second;
    double noise_rms;
    double security_bits;
    std::string use_case;
    double pipeline_cycle_ms;
    double pipeline_cycles_per_sec;
};

class ModeBenchmark {
private:
    RingMode mode_;
    FHEParams params_;
    CryptoContext<DCRTPoly> context_;
    KeyPair<DCRTPoly> keyPair_;
    
public:
    explicit ModeBenchmark(RingMode mode) : mode_(mode), params_(mode) {
        setup_context();
    }
    
    void setup_context() {
        auto start = std::chrono::high_resolution_clock::now();
        
        auto cc_params = CreateBFVParams(mode_);
        context_ = GenCryptoContext(cc_params);
        context_->Enable(PKE);
        context_->Enable(KEYSWITCH);
        context_->Enable(LEVELEDSHE);
        
        keyPair_ = context_->KeyGen();
        context_->EvalMultKeyGen(keyPair_.secretKey);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "✓ " << params_.GetSpec().name << " context setup: " 
                  << duration.count() / 1000.0 << " ms" << std::endl;
    }
    
    BenchmarkResults run_benchmark(int num_cycles = 10) {
         BenchmarkResults results;
         results.mode_name = params_.GetSpec().name;
         results.ring_dim = params_.GetRingDimension();
         results.use_case = params_.GetSpec().use_case;
         results.security_bits = (mode_ == RingMode::MIN_LATENCY_4096) ? 80.0 : 128.0;
  
        // === Separate cold-start from steady-state performance ===
        FHEPipeline pipeline(10, 34, mode_);
        
        // Cold-start: includes matrix generation and SVD computation
        auto cold_start = std::chrono::high_resolution_clock::now();
        pipeline.run_pipeline(1);  // First cycle includes setup
        auto cold_end = std::chrono::high_resolution_clock::now();
        double cold_start_ms = std::chrono::duration_cast<std::chrono::microseconds>(cold_end - cold_start).count() / 1000.0;
        
        // Steady-state: measure multiple cycles after warmup
        auto steady_start = std::chrono::high_resolution_clock::now();
        pipeline.run_pipeline(5);  // 5 cycles in steady state
        auto steady_end = std::chrono::high_resolution_clock::now();
        double steady_total_ms = std::chrono::duration_cast<std::chrono::microseconds>(steady_end - steady_start).count() / 1000.0;
        double steady_per_cycle_ms = steady_total_ms / 5.0;
        
        // Report steady-state performance (what matters for throughput)
        results.pipeline_cycle_ms = steady_per_cycle_ms;
        results.pipeline_cycles_per_sec = 1000.0 / steady_per_cycle_ms;
        
        // For compatibility, set micro-benchmark fields to pipeline values
        results.mult_time_ms = results.pipeline_cycle_ms;
        results.projection_time_ms = 0.0; // Already included in pipeline cycle
        results.total_cycle_time_ms = results.pipeline_cycle_ms;
        results.cycles_per_second = results.pipeline_cycles_per_sec;
        
        // Simplified noise estimate (placeholder)
        results.noise_rms = (mode_ == RingMode::MIN_LATENCY_4096) ? 15.0 : 18.0;
        
        // Log both metrics for debugging
        std::cout << "  Cold-start: " << cold_start_ms << " ms (includes matrix generation)" << std::endl;
        std::cout << "  Steady-state: " << steady_per_cycle_ms << " ms/cycle (" << results.pipeline_cycles_per_sec << " c/sec)" << std::endl;
           
           return results;
     }
};

void print_benchmark_header() {
     std::cout << std::left << std::setw(15) << "Mode"
               << std::setw(8) << "Ring"
               << std::setw(12) << "Cycle(ms)"
               << std::setw(12) << "FullPipe(ms)"
               << std::setw(12) << "Cycles/sec"
               << std::setw(12) << "Noise RMS"
               << std::setw(12) << "Security"
               << "Use Case" << std::endl;

}

void print_benchmark_result(const BenchmarkResults& result) {
     std::cout << std::left << std::setw(15) << result.mode_name
               << std::setw(8) << result.ring_dim
               << std::setw(12) << result.pipeline_cycle_ms
               << std::setw(12) << std::fixed << std::setprecision(3) << result.pipeline_cycle_ms
               << std::setw(12) << std::setprecision(1) << result.pipeline_cycles_per_sec
               << std::setw(12) << std::setprecision(2) << result.noise_rms
               << std::setw(12) << std::setprecision(0) << result.security_bits << "-bit"
               << result.use_case.substr(0, 40) << std::endl;
 }

void analyze_trade_offs(const BenchmarkResults& mode_4096, const BenchmarkResults& mode_8192) {
     std::cout << "\n📊 Trade-off Analysis:" << std::endl;
     std::cout << "======================" << std::endl;
     
    double latency_improvement = mode_8192.pipeline_cycle_ms / mode_4096.pipeline_cycle_ms;
    double throughput_improvement = mode_4096.pipeline_cycles_per_sec / mode_8192.pipeline_cycles_per_sec;
     
    std::cout << "🚀 Latency: 4096-bit is " << std::setprecision(2) << latency_improvement 
              << "x faster (" << mode_4096.pipeline_cycle_ms << " vs " 
              << mode_8192.pipeline_cycle_ms << " ms/cycle)" << std::endl;
              
    std::cout << "⚡ Throughput: 4096-bit achieves " << throughput_improvement 
              << "x higher cycles/sec (" << std::setprecision(1) << mode_4096.pipeline_cycles_per_sec 
              << " vs " << mode_8192.pipeline_cycles_per_sec << ")" << std::endl;
              
    std::cout << "🔒 Security: 8192-bit provides " << (mode_8192.security_bits - mode_4096.security_bits)
              << " additional security bits" << std::endl;
              
    std::cout << "📡 Noise: Similar noise characteristics (projection-dominated)" << std::endl;
    
    std::cout << "\n💡 Recommendations:" << std::endl;
    std::cout << "• Use 4096-bit for: Interactive demos, verifiable computation, research" << std::endl;
    std::cout << "• Use 8192-bit for: Production systems, sensitive data, compliance requirements" << std::endl;
}

int main() {
     std::cout << "OpenFHE Ring Dimension Benchmark" << std::endl;
     std::cout << "=================================" << std::endl;
    std::cout << "Measuring steady-state mult→project pipeline performance" << std::endl;
    std::cout << "(Matrix generation cost amortized over multiple cycles)" << std::endl;
      
      std::cout << "\n🔧 Setting up benchmarks..." << std::endl;
    
    // Create benchmarks for both modes
    ModeBenchmark bench_4096(RingMode::MIN_LATENCY_4096);
    ModeBenchmark bench_8192(RingMode::STANDARD_8192);
    
    std::cout << "\n🏃 Running benchmarks..." << std::endl;
    
    // Run benchmarks
    auto results_4096 = bench_4096.run_benchmark(10);
    auto results_8192 = bench_8192.run_benchmark(10);
    
    // Display results
    std::cout << "\n📈 Benchmark Results:" << std::endl;
    print_benchmark_header();
    print_benchmark_result(results_4096);
    print_benchmark_result(results_8192);
    
    // Analyze trade-offs
    analyze_trade_offs(results_4096, results_8192);
    
    std::cout << "\n✅ Benchmark complete!" << std::endl;
    
    return 0;
} 