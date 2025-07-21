#pragma once

#include "openfhe.h"
#include <string>

using namespace lbcrypto;

/**
 * FHE Ring Dimension Modes for Different Use Cases
 * 
 * This system supports two ring dimension configurations optimized for
 * different security/performance trade-offs:
 */

enum class RingMode {
    MIN_LATENCY_4096,  // Minimum latency mode (4096-bit ring)
    STANDARD_8192      // Standard security mode (8192-bit ring)  
};

/**
 * Ring Mode Specifications
 */
struct RingModeSpec {
    uint32_t ring_dim;
    std::string name;
    std::string use_case;
    std::string security_profile;
    double estimated_latency_factor;  // Relative to 4096 baseline
    
    static RingModeSpec get_spec(RingMode mode) {
        switch (mode) {
            case RingMode::MIN_LATENCY_4096:
                return {
                    .ring_dim = 4096,
                    .name = "Min-Latency",
                    .use_case = "Low-value/transient computations, verifiability-focused distributed networks",
                    .security_profile = "Verifiability via MACs/SNARKs, not cryptographic privacy",
                    .estimated_latency_factor = 1.0
                };
                
            case RingMode::STANDARD_8192:
                return {
                    .ring_dim = 8192,
                    .name = "Standard-Security", 
                    .use_case = "Production FHE applications, cryptographic privacy guarantees",
                    .security_profile = "128-bit cryptographic security (OpenFHE validated)",
                    .estimated_latency_factor = 2.2  // ~2.2x slower due to larger NTT
                };
                
            default:
                throw std::invalid_argument("Invalid ring mode");
        }
    }
};

/**
 * Use Case Guidelines:
 * 
 * MIN_LATENCY_4096:
 * - Distributed computation networks where privacy is NOT the primary concern
 * - Verifiable computation via cryptographic commitments (MACs, SNARKs)
 * - Interactive applications requiring sub-100ms response times
 * - Proof-of-concept demonstrations and research benchmarks
 * - Client explicitly opts-in understanding reduced security model
 * 
 * STANDARD_8192:
 * - Production applications requiring cryptographic privacy
 * - Compliance with standard FHE security requirements
 * - Comparison benchmarks against other FHE implementations
 * - Applications handling sensitive data (financial, medical, personal)
 * - Default mode for security-critical deployments
 */

/**
 * Parameter Configuration for Ring Modes
 */
class FHEParams {
private:
    RingMode mode_;
    RingModeSpec spec_;
    
public:
    explicit FHEParams(RingMode mode = RingMode::STANDARD_8192) 
        : mode_(mode), spec_(RingModeSpec::get_spec(mode)) {}
    
    // Core FHE parameters
    uint32_t GetRingDimension() const { return spec_.ring_dim; }
    uint32_t GetPlaintextModulus() const { return 65537; }  // Same for both modes
    SecurityLevel GetSecurityLevel() const { 
        return (mode_ == RingMode::MIN_LATENCY_4096) ? 
               HEStd_NotSet :  // Bypass validation for research mode
               HEStd_128_classic; 
    }
    
    // Mode information
    RingMode GetMode() const { return mode_; }
    RingModeSpec GetSpec() const { return spec_; }
    std::string GetModeDescription() const {
        return spec_.name + " (" + std::to_string(spec_.ring_dim) + "-bit): " + spec_.use_case;
    }
    
    // Performance estimates
    double GetLatencyFactor() const { return spec_.estimated_latency_factor; }
    
    // Validation
    bool IsMinLatencyMode() const { return mode_ == RingMode::MIN_LATENCY_4096; }
    bool IsStandardSecurityMode() const { return mode_ == RingMode::STANDARD_8192; }
    
    // Warning for min-latency mode
    std::string GetSecurityWarning() const {
        if (IsMinLatencyMode()) {
            return "⚠️  MIN-LATENCY MODE: Reduced cryptographic security. "
                   "Suitable only for verifiable computation where privacy is not required. "
                   "Use MACs/SNARKs for integrity verification.";
        }
        return "";
    }
};

/**
 * Factory function to create CCParams based on ring mode
 */
CCParams<CryptoContextBFVRNS> CreateBFVParams(RingMode mode = RingMode::STANDARD_8192); 