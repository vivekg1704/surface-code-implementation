"""Run the full study: threshold sweep, decoding-time benchmark, then figures.

    python run.py

Every setting is written out below. Comment out a step to skip it. The threshold
sweep resumes from results/threshold.csv, so rerunning with the same settings
reuses the data already collected rather than resampling.
"""

import numpy as np

from surface_code import plot_threshold, plot_timing, run_threshold_sweep, run_timing_benchmark


def main():
    # 1. Threshold: logical error rate vs physical error rate p for d = 3, 5, 7.
    #    Each point runs until 1000 logical errors (~3% relative error) or 10M shots.
    run_threshold_sweep(
        distances=[3, 5, 7],
        error_rates=np.geomspace(0.002, 0.015, 12),  # 0.2% to 1.5%, evenly spaced in log(p)
        max_shots=10_000_000,
        max_errors=1_000,
        out_path="results/threshold.csv",
    )

    # 2. Decoding time: single-threaded PyMatching time per shot and per round vs d.
    #    Runs after the sweep so it has the machine to itself.
    run_timing_benchmark(
        distances=[3, 5, 7, 9, 11, 13, 15, 17, 19, 21],
        error_rates=[0.001, 0.003, 0.005],
        shots=20_000,
        repeats=5,
        out_path="results/decode_timing.csv",
    )

    # 3. Figures.
    plot_threshold(csv_path="results/threshold.csv", path="figures/threshold.png")
    plot_timing(csv_path="results/decode_timing.csv", path="figures/decode_timing.png")


if __name__ == "__main__":
    main()
