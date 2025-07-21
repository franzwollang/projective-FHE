#!/usr/bin/env python3
"""
Runs the finite-field mult->proj simulation across a matrix of parameters
to validate the service tiers proposed in the FHE architecture document.
"""

import pandas as pd

from FHE.code.simulation.ff_poc.mult_proj_poc_ff import run_ff_experiment

# Define the SLA tiers from the FHEv3 design document (Table 5.3)
# We exclude Tier 5 as it uses a different FHE backend.
SLA_TIERS = {
    "1. Micro-Latency": {"k": 10, "p_ratio_pct": 61, "sigma_bits": 10},
    "2. Standard Interactive": {"k": 20, "p_ratio_pct": 61, "sigma_bits": 10},
    "3. High-Throughput": {"k": 30, "p_ratio_pct": 61, "sigma_bits": 10},
    "4. High-Precision": {"k": 20, "p_ratio_pct": 90, "sigma_bits": 11},
}

# Common simulation parameters
NUM_CYCLES = 200
SEED = 2025


def main():
    """
    Main function to run the SLA validation sweep.
    """
    print("=== Running SLA Validation Sweep ===")
    results = []

    for tier_name, params in SLA_TIERS.items():
        k = params["k"]
        T = k * (k + 1) // 2
        # Calculate p from the required p/T ratio
        p = int(T * (params["p_ratio_pct"] / 100.0))

        print(f"\n--- Running Tier: {tier_name} ---")
        print(f"  k={k}, T={T}, p={p}")
        print(f"  (target ratio: {params['p_ratio_pct']}%)")

        # The 't_param' is derived from sigma_bits to set the noise ceiling.
        # 2^(sigma_bits) gives the data range, and t should be > 2*range.
        t_param = 2 ** (params["sigma_bits"] + 2)

        # Run the experiment
        logger, analytical_eq = run_ff_experiment(
            k=k,
            p=p,
            t_param=t_param,
            num_cycles=NUM_CYCLES,
            seed=SEED,
            verbose=False,  # Keep the output clean
        )

        # Extract the final converged noise value
        if len(logger.data) >= 10:
            final_noise_variance = logger.data[-1][2]  # after_proj from last cycle
        else:
            final_noise_variance = float("nan")

        results.append(
            {
                "Tier": tier_name,
                "k": k,
                "T": T,
                "p": p,
                "p/T Ratio (%)": f"{(p/T)*100:.1f}",
                "Sigma Bits": params["sigma_bits"],
                "t_param": t_param,
                "Analytical Eq (Δ^2)": f"{analytical_eq:.6f}",
                "Empirical Eq (Δ^2)": f"{final_noise_variance:.6f}",
            }
        )
        print(f"  ... Done. Final noise: {final_noise_variance:.6f}")

    print("\n\n=== SLA Validation Results ===")
    df = pd.DataFrame(results)
    print(df.to_string())

    # Save to CSV
    output_filename = "FHE/code/sla_validation_results.csv"
    df.to_csv(output_filename, index=False)
    print(f"\nResults saved to {output_filename}")


if __name__ == "__main__":
    main()
