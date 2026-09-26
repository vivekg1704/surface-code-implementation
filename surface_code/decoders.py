"""Decoders compared in this study: PyMatching (MWPM) and union-find (the union_find_decoder package).

Both are built from the same Stim detector error model and share the
decode_batch interface, so the timing and threshold code treats them alike.
"""

import pymatching
import stim
import union_find_decoder

# Decoder names usable in sinter sweeps, beyond sinter's built-in "pymatching":
# "union_find" and "union_find_unweighted".
CUSTOM_DECODERS = union_find_decoder.sinter_decoders()


def mwpm(dem: stim.DetectorErrorModel) -> pymatching.Matching:
    return pymatching.Matching.from_detector_error_model(dem)


def union_find(dem: stim.DetectorErrorModel, weight_resolution: int = 16) -> union_find_decoder.UnionFind:
    return union_find_decoder.UnionFind.from_detector_error_model(dem, weight_resolution=weight_resolution)
