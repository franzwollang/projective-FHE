# FHE Code Development

This directory contains the implementation and testing of a Fully Homomorphic Encryption (FHE) system based on the architecture described in `../reference/fhe_v3.md`. The core innovation is a noise management approach using frequent projection after every homomorphic multiplication.

## Project Structure

### Simulation Tracks

#### 1. Finite-Field Simulation (`simulation/ff_poc/`)

**Status: STABLE** - Primary development focus

The finite-field simulation implements the complete `mult->project` cycle over GF(65537) using:

- **Lazy-reduction arithmetic**: Forward projection uses 64-bit accumulators to prevent overflow
- **Real-valued Moore-Penrose pseudoinverse**: Computes projection in floating-point then converts back to finite field
- **LWR noise model**: Learning With Rounding using deterministic quantization

**Key Files:**

- `mult_proj_poc_ff.py`: Main simulation loop
- `qc_matrix_ff.py`: QC-MDS matrix generation and finite-field utilities
- `run_sla_validation.py`: Validation across multiple SLA tiers

**Current Performance:**

- Empirical noise after projection: ~0.5-0.7 Δ² (underdetermined), ~0.02 Δ² (fully determined)
- No overflow warnings with lazy-reduction approach
- All unit tests passing

#### 2. Floating-Point Simulation (`simulation/float_poc/`)

**Status: ARCHIVED** - Served as initial proof-of-concept

These simulations validated the basic projection mathematics in floating-point arithmetic before tackling finite-field implementation challenges.

#### 3. TenSEAL Integration (`experiments/tenseal/`)

**Status: EXPERIMENTAL** - Future work

Planned integration with the TenSEAL library for practical FHE operations.

### Analysis Tools (`analysis/`)

Debug scripts for investigating noise sources, matrix conditioning, and projection accuracy:

- `debug_single_run.py`: Step-by-step analysis of individual cycles
- `debug_projection.py`: Projection accuracy testing
- `debug_matrix_conditioning.py`: Numerical precision analysis

### Testing (`tests/`)

Comprehensive unit test suite covering:

- Projection recovery accuracy
- Noise convergence and stability
- Matrix rank preservation
- Empirical vs theoretical noise bounds

## Key Technical Findings

### Projection Strategy

After extensive investigation, we determined that **real-valued Moore-Penrose pseudoinverse with lazy-reduction** is the optimal approach for noise filtering:

1. **Pure finite-field pseudoinverse**: While algebraically correct (satisfies A·A⁺ = I), it provides no noise reduction. The lack of an inner product in finite fields means the projection doesn't minimize energy, resulting in noise levels ~10⁶ Δ² (unusable).

2. **Scaled-down matrices**: Right-shifting matrix coefficients prevents overflow but introduces quantization noise that becomes the dominant error source (~50-170 Δ² vs theoretical ~0.1 Δ²).

3. **Lazy-reduction + real pseudoinverse**: Uses 64-bit accumulators for forward projection to prevent overflow, then applies real-valued Moore-Penrose pseudoinverse for optimal noise filtering. Final rounding back to finite field introduces minimal additional noise.

### Current Noise Performance

The system achieves noise levels close to theoretical predictions:

- **Underdetermined case** (p < T): ~0.5-0.7 Δ² (vs ~0.1 Δ² theoretical projection loss)
- **Fully determined case** (p = T): ~0.02 Δ² (near-optimal)

The gap between empirical and theoretical is due to unavoidable rounding noise:

- Float-to-integer conversion in pseudoinverse computation
- LWR quantization noise
- Numerical precision limits

### Matrix Generation

QC-MDS matrices are generated using cryptographically strong pseudorandom seeds and provide:

- Guaranteed MDS property with overwhelming probability (failure rate < 2^-1744)
- FFT-accelerated projection via block-circulant structure
- Fresh matrix per computational epoch for security

## Development Roadmap

### Completed ✅

- [x] Finite-field overflow resolution via lazy-reduction
- [x] Real-valued pseudoinverse integration for optimal noise filtering
- [x] Comprehensive unit test suite with appropriate thresholds
- [x] Matrix conditioning and numerical precision analysis
- [x] SLA tier validation across different parameter sets

### Current Priorities

- [ ] Performance optimization of projection kernels
- [ ] Integration with hardware-accelerated FFT libraries
- [ ] TenSEAL backend implementation
- [ ] Circuit depth analysis for complex applications

### Future Work

- [ ] GPU kernel implementation for production deployment
- [ ] Integration with existing FHE frameworks
- [ ] Security analysis and parameter recommendations
- [ ] Benchmarking against state-of-the-art FHE schemes

## Running the Code

### Basic Simulation

```bash
# Run default configuration (k=20, p=210, 1000 cycles)
python simulation/ff_poc/mult_proj_poc_ff.py

# Custom parameters
python simulation/ff_poc/mult_proj_poc_ff.py --k 10 --p 33 --cycles 100
```

### Testing

```bash
# Run all tests
python -m pytest tests/

# Run specific test suite
python -m pytest tests/test_projection.py -v
```

### SLA Validation

```bash
# Validate across multiple tiers
python simulation/ff_poc/run_sla_validation.py
```

## Architecture Notes

This implementation validates the mathematical foundations described in `fhe_v3.md` while revealing practical constraints:

1. **Overflow management** is critical for 16-bit modulus systems
2. **Real-domain computation** is necessary for effective noise filtering
3. **Lazy-reduction** provides the optimal balance of performance and accuracy
4. **QC-MDS structure** enables practical FFT acceleration while maintaining security

The system demonstrates that frequent projection can indeed maintain low noise levels (~0.5 Δ²) across thousands of multiplication cycles, validating the core architectural premise.

## Upgrade to 32bit?

For the SLA tiers in the current spec (σ_signal ≤ 2¹⁰, depth ≤ 3 000 cycles) the 16-bit design with the 0.5-0.7 Δ² noise floor is already safe and keeps latency and cost low → ship Tier 1-4 on 16-bit.

Introduce a “High-Precision” premium tier that uses a 32-bit modulus (or CRT packing) only if:
 – clients need > 11-bit plaintext range, or
 – very deep circuits (> 10 k mults) without bootstrapping.

This keeps the mainstream service lean while offering an upgrade path for edge cases.

### OpenFHE Prototype Roadmap (research PoC)

> Goal: demonstrate the mult→project architecture on a real RLWE backend **without** CKKS modulus-switch overhead. OpenFHE (ex-PALISADE) already supports single-prime BFV parameters, fixed modulus and GPU NTT kernels, making it the most direct fit.

1. **Parameter preset**
   • Use OpenFHE’s BFVrns scheme with **one 16-bit prime** (or 32-bit for the premium tier). No levels, no mod-switch.
2. **QC-MDS slot rotations**
   • Implement `encode()` / `decode()` with `EvalAtIndexBatch` rotations and plaintext constants.
3. **Projection evaluator**
   • Encode each row of `A⁺` as a BFV plaintext and evaluate the dot-product `∑ P_i · ct_i` in one fused call.
4. **Pipeline skeleton**
   1. Encode client data → `k` logical ciphertexts
   2. For each step: external scale (public constant) → self-mult → Δ⁻¹ (public constant) → projection → T→k selection
   3. Decrypt on client
5. **Noise & SNR instrumentation**
   • After every projection decrypt a probe ciphertext, compute centred RMS, log to CSV.
6. **Unit-test parity**
   • Port `random_external_matrix_test` to OpenFHE; assert noise ≤ theoretical + 3 σ.
7. **Stress test**
   • 1 000 sequential mults on single-prime backend; confirm no precision loss.

Custom CUDA/Metal kernels stay out-of-scope until funded/picked up.

**✅ UPDATE: OpenFHE prototype successfully compiles!** All core components (`libopenfhe_utils.dylib`, `params`, `pipeline`) build correctly. Currently debugging runtime projection dimension mismatch. See `openfhe_prototype/README.md` for details.

---
