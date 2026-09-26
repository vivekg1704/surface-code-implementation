import numpy as np
import pymatching
import pytest

from surface_code import memory_circuit


def logical_error_rate(d, p, shots=20_000):
    circuit = memory_circuit(d, p)
    matching = pymatching.Matching.from_detector_error_model(circuit.detector_error_model(decompose_errors=True))
    dets, obs = circuit.compile_detector_sampler(seed=1).sample(shots, separate_observables=True)
    return np.any(matching.decode_batch(dets) != obs, axis=1).mean()


@pytest.mark.parametrize("d", [3, 5, 7])
def test_layout(d):
    circuit = memory_circuit(d, 0.001)
    # (d^2 - 1)/2 Z checks in the first round, all d^2 - 1 checks in rounds 2..d,
    # and (d^2 - 1)/2 more from the final data-qubit readout: d(d^2 - 1) in total.
    assert circuit.num_detectors == (d**2 - 1) * d
    assert circuit.num_observables == 1


def test_noiseless_circuit_never_fails():
    assert logical_error_rate(5, 0.0, shots=1_000) == 0


def test_graphlike_distance_matches_code_distance():
    for d in (3, 5, 7):
        dem = memory_circuit(d, 0.001).detector_error_model(decompose_errors=True)
        assert len(dem.shortest_graphlike_error()) == d


def test_below_threshold_larger_code_is_better():
    assert logical_error_rate(5, 0.003) < logical_error_rate(3, 0.003)


def test_above_threshold_larger_code_is_worse():
    assert logical_error_rate(5, 0.012) > logical_error_rate(3, 0.012)
