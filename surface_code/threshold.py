"""Threshold sweep: logical error rate vs physical error rate across code distances."""

import os

import sinter

from .circuits import memory_circuit


def run_threshold_sweep(distances, error_rates, max_shots, max_errors, out_path, workers=os.cpu_count()):
    """Sample and decode a memory experiment for every (d, p) pair.

    Stim samples the syndromes, PyMatching (MWPM) decodes them, and sinter spreads
    the work over `workers` processes. Each (d, p) point stops once it has seen
    `max_errors` logical errors or `max_shots` shots, whichever comes first.

    Results are appended to `out_path`; rerunning with the same settings picks up
    the existing data instead of starting again.
    """
    tasks = [
        sinter.Task(circuit=memory_circuit(d, p), json_metadata={"d": d, "p": float(p), "rounds": d})
        for d in distances
        for p in error_rates
    ]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    stats = sinter.collect(
        num_workers=workers,
        tasks=tasks,
        decoders=["pymatching"],
        max_shots=max_shots,
        max_errors=max_errors,
        save_resume_filepath=out_path,
        print_progress=True,
    )

    for s in sorted(stats, key=lambda s: (s.json_metadata["d"], s.json_metadata["p"])):
        m = s.json_metadata
        print(f"d={m['d']}  p={m['p']:.4f}  shots={s.shots:>9}  errors={s.errors:>5}  P_L={s.errors / s.shots:.3e}")
    return stats
