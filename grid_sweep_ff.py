#!/usr/bin/env python3
"""
Grid sweep utility for the finite field mult->proj PoC.

Runs the experiment over a grid of parameters and saves results to CSV.
"""

import argparse
import csv
import subprocess


def run_single_experiment(k, p, t, cycles, seed):
    """
    Run a single PoC experiment and parse its output.
    """
    cmd = [
        "python",
        "mult_proj_poc_ff.py",
        "--k",
        str(k),
        "--p",
        str(p),
        "--t",
        str(t),
        "--cycles",
        str(cycles),
        "--seed",
        str(seed),
        "--quiet",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"ERROR running with k={k}, p={p}, t={t}:")
        print(result.stderr)
        return None

    # Parse output
    lines = result.stdout.strip().split("\n")
    last_var, max_var, last_proj_err, rank_fails = -1.0, -1.0, -1.0, -1

    try:
        data_lines = [
            l.split() for l in lines if l.strip().startswith("Cycle") and "before" in l
        ]

        if data_lines:
            # Get data from last line
            last_line_parts = data_lines[-1]
            last_var = float(last_line_parts[3])
            last_proj_err = float(last_line_parts[5])

            # Get max variance from all data lines
            max_var = max(float(parts[3]) for parts in data_lines)

        # Rank failures
        rank_fail_line = next(l for l in lines if "Rank failures" in l)
        rank_fails = int(rank_fail_line.split(":")[1].strip())
    except (StopIteration, IndexError, ValueError) as e:
        print(f"    -> Parsing failed for k={k},p={p},t={t}: {e}")

    return {
        "k": k,
        "p": p,
        "t": t,
        "cycles": cycles,
        "maxVar": max_var,
        "lastVar": last_var,
        "lastProjErr": last_proj_err,
        "rankFails": rank_fails,
    }


def main():
    parser = argparse.ArgumentParser(description="Grid sweep for FF PoC")
    parser.add_argument("--cycles", type=int, default=200, help="Cycles per run")
    parser.add_argument(
        "--outfile", type=str, default="sweep_results.csv", help="Output CSV file"
    )
    args = parser.parse_args()

    param_grid = {"k": [20], "p": [210, 128, 96, 64], "t": [128, 256]}

    results = []

    print("Starting grid sweep...")
    for k in param_grid["k"]:
        for p in param_grid["p"]:
            for t in param_grid["t"]:
                print(f"  Running k={k}, p={p}, t={t}...")
                result = run_single_experiment(k, p, t, args.cycles, 42)
                if result:
                    results.append(result)

    # Write to CSV
    if results:
        fieldnames = [
            "k",
            "p",
            "t",
            "cycles",
            "maxVar",
            "lastVar",
            "lastProjErr",
            "rankFails",
        ]
        with open(args.outfile, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nGrid sweep complete. Results saved to {args.outfile}")


if __name__ == "__main__":
    main()
