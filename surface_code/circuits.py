"""Rotated surface code memory circuits with circuit-level depolarising noise."""

import stim


def memory_circuit(distance: int, p: float, rounds: int | None = None, basis: str = "z") -> stim.Circuit:
    """Rotated surface code memory experiment with uniform circuit-level noise.

    Every noisy location gets the same strength p:
      - two-qubit depolarising channel after each CNOT
      - single-qubit depolarising on data qubits at the start of each round (idling)
      - bit-flip before each measurement and after each reset

    Defaults to `distance` rounds of stabiliser measurement, so time-like and
    space-like errors are both on the same footing.
    """
    return stim.Circuit.generated(
        f"surface_code:rotated_memory_{basis}",
        distance=distance,
        rounds=rounds if rounds is not None else distance,
        after_clifford_depolarization=p,
        before_round_data_depolarization=p,
        before_measure_flip_probability=p,
        after_reset_flip_probability=p,
    )
