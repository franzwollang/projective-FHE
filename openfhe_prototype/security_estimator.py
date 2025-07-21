#!/usr/bin/env python3
"""
Security Parameter Estimator for OpenFHE Prototype

This script estimates the security level of RLWE parameters used in the
mult->project FHE architecture using the lattice estimator framework.

Usage:
    python3 security_estimator.py
"""

import sys
import math
from typing import Dict


def estimate_rlwe_security(n: int, q: int, sigma: float) -> Dict[str, float]:
    """
    Estimate RLWE security using simplified lattice attack models.

    This is a simplified estimator for preliminary analysis.
    For production use, integrate with malb/lattice-estimator.

    Args:
        n: Ring dimension
        q: Modulus
        sigma: Noise standard deviation

    Returns:
        Dictionary with security estimates
    """

    # Convert to alpha parameter (noise rate)
    alpha = sigma / q

    # Simplified security estimation based on known attack complexities
    # These are rough estimates - use proper lattice estimator for production

    # Primal attack complexity (simplified BKZ model)
    # Based on: log2(operations) ≈ 0.292 * beta + 16.4 + log2(8*d)
    # where beta ≈ sqrt(n * log(q) / log(delta)) for target root Hermite
    # factor delta

    log_q = math.log2(q)

    # Estimate required beta for different attack models
    # These formulas are simplified approximations

    # Classical primal attack
    delta_classical = 1.0045  # Target root Hermite factor
    beta_classical = math.sqrt(n * log_q / math.log(delta_classical))
    classical_primal = 0.292 * beta_classical + 16.4 + math.log2(8 * (n + 1))

    # Quantum primal attack (more aggressive reduction)
    delta_quantum = 1.0035
    beta_quantum = math.sqrt(n * log_q / math.log(delta_quantum))
    quantum_primal = 0.265 * beta_quantum  # Quantum BKZ cost model

    # Dual attack estimate
    # Simplified: similar complexity but different constants
    classical_dual = classical_primal - 5  # Typically slightly easier
    quantum_dual = quantum_primal - 3

    # Take minimum (most efficient attack)
    classical_security = min(classical_primal, classical_dual)
    quantum_security = min(quantum_primal, quantum_dual)

    return {
        "classical_primal": classical_primal,
        "classical_dual": classical_dual,
        "quantum_primal": quantum_primal,
        "quantum_dual": quantum_dual,
        "classical_min": classical_security,
        "quantum_min": quantum_security,
    }


def analyze_parameters():
    """Analyze security of current and proposed parameter sets."""

    print("=" * 60)
    print("SECURITY ANALYSIS: OpenFHE Prototype Parameters")
    print("=" * 60)
    print()

    # Parameter sets to analyze
    params = [
        {
            "name": "Current (4096-bit)",
            "n": 4096,
            "q": 65537,
            "sigma": 256 / 2,  # LWR with Delta ≈ 256, sigma ≈ Delta/2
        },
        {"name": "Proposed (8192-bit)", "n": 8192, "q": 65537, "sigma": 256 / 2},
    ]

    results = []

    for param in params:
        print(f"Parameter Set: {param['name']}")
        print(f"  Ring dimension (n): {param['n']}")
        print(f"  Modulus (q): {param['q']}")
        print(f"  Noise std dev (σ): {param['sigma']:.1f}")
        print(f"  Noise rate (α): {param['sigma']/param['q']:.2e}")
        print()

        security = estimate_rlwe_security(param["n"], param["q"], param["sigma"])
        results.append((param["name"], security))

        print("Security Estimates (bits):")
        print(f"  Classical Primal: {security['classical_primal']:.1f}")
        print(f"  Classical Dual:   {security['classical_dual']:.1f}")
        print(f"  Quantum Primal:   {security['quantum_primal']:.1f}")
        print(f"  Quantum Dual:     {security['quantum_dual']:.1f}")
        print()
        print(f"  Classical Security: {security['classical_min']:.1f} bits")
        print(f"  Quantum Security:   {security['quantum_min']:.1f} bits")
        print()

        # Security assessment
        classical_ok = security["classical_min"] >= 128
        quantum_ok = security["quantum_min"] >= 128

        print("Security Assessment:")
        status_classical = "✅ PASS" if classical_ok else "❌ FAIL"
        status_quantum = "✅ PASS" if quantum_ok else "❌ FAIL"

        print(f"  Classical (≥128 bits): {status_classical}")
        print(f"  Quantum (≥128 bits):   {status_quantum}")

        if classical_ok and quantum_ok:
            print(f"  Overall: ✅ SECURE for production")
        else:
            print(f"  Overall: ❌ INSECURE for production")

        print("=" * 60)
        print()

    # Comparative analysis
    print("COMPARATIVE ANALYSIS")
    print("=" * 60)

    current = results[0][1]
    proposed = results[1][1]

    classical_improvement = proposed["classical_min"] - current["classical_min"]
    quantum_improvement = proposed["quantum_min"] - current["quantum_min"]

    print(f"Security Improvement (8192 vs 4096):")
    print(f"  Classical: +{classical_improvement:.1f} bits")
    print(f"  Quantum:   +{quantum_improvement:.1f} bits")
    print()

    # Performance impact estimate
    perf_ratio = 8192 / 4096 * math.log2(8192) / math.log2(4096)
    print(f"Performance Impact:")
    print(f"  NTT/FFT operations: ~{perf_ratio:.1f}x slower")
    print(f"  Memory usage: 2x larger")
    print(f"  Estimated pipeline: ~{perf_ratio:.1f}x slower")
    print()

    # Recommendations
    print("RECOMMENDATIONS")
    print("=" * 60)

    if current["classical_min"] < 120 or current["quantum_min"] < 120:
        print("🚨 CRITICAL: 4096-bit parameters provide insufficient security")
        print("   Recommendation: UPGRADE to 8192-bit for production")
        print("   Rationale: Falls below industry standard 128-bit requirement")
    else:
        print("ℹ️  4096-bit parameters may be acceptable for research use")
        print("   However, 8192-bit recommended for production deployment")

    print()
    print("Note: These are simplified estimates. For production deployment,")
    print("use the full lattice-estimator (github.com/malb/lattice-estimator)")
    print("with conservative attack models and expert cryptographic review.")


def main():
    """Main entry point."""
    try:
        analyze_parameters()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
