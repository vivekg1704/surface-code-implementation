"""Decoding time vs code distance for PyMatching (MWPM).

Run on an otherwise idle machine: sharing cores skews the timings.
"""

import csv
import os
import platform
import time

import numpy as np
import pymatching

from .circuits import memory_circuit


def time_decoding(d, p, shots, repeats, seed=0):
    """Time single-threaded `decode_batch` on `shots` syndromes from a d-round memory experiment.

    Sampling is excluded from the timing. Returns the median over `repeats` runs,
    per shot and per syndrome round, plus the mean number of detection events
    (defects) per shot, since matching cost is driven by how many there are to pair up.
    """
    circuit = memory_circuit(d, p)
    matching = pymatching.Matching.from_detector_error_model(circuit.detector_error_model(decompose_errors=True))
    dets, obs = circuit.compile_detector_sampler(seed=seed).sample(shots, separate_observables=True)

    matching.decode_batch(dets[: min(shots, 1000)])  # warm-up
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        pred = matching.decode_batch(dets)
        times.append(time.perf_counter() - t0)

    per_shot = np.median(times) / shots
    return {
        "d": d,
        "p": p,
        "rounds": d,
        "shots": shots,
        "num_detectors": circuit.num_detectors,
        "mean_defects": float(dets.sum(axis=1).mean()),
        "logical_error_rate": float(np.any(pred != obs, axis=1).mean()),
        "us_per_shot": per_shot * 1e6,
        "us_per_round": per_shot * 1e6 / d,
        "us_per_shot_min": min(times) / shots * 1e6,
        "us_per_shot_max": max(times) / shots * 1e6,
    }


def run_timing_benchmark(distances, error_rates, shots, repeats, out_path):
    print(f"pymatching {pymatching.__version__} | {platform.processor() or platform.machine()} | {platform.platform()}")
    rows = []
    for p in error_rates:
        for d in distances:
            row = time_decoding(d, p, shots, repeats)
            rows.append(row)
            print(
                f"p={p:.3f} d={d:>2}  defects/shot={row['mean_defects']:7.1f}  "
                f"{row['us_per_shot']:9.2f} us/shot  {row['us_per_round']:7.2f} us/round  P_L={row['logical_error_rate']:.2e}",
                flush=True,
            )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out_path}")
    return rows
