"""
Compute tempo metrics
"""
import numpy as np


def acc1(reference_tempo, estimated_tempo, tolerance=0.04, factor=1.0):
    return np.abs(reference_tempo * factor - estimated_tempo)\
        <= (reference_tempo * factor * tolerance)


def acc2(reference_tempo, estimated_tempo, tolerance=0.04):
    return (
        (acc1(reference_tempo, estimated_tempo, tolerance, 1.0))
        | (acc1(reference_tempo, estimated_tempo, tolerance, 2.0))
        | (acc1(reference_tempo, estimated_tempo, tolerance, 3.0))
        | (acc1(reference_tempo, estimated_tempo, tolerance, 1.0 / 2.0))
        | (acc1(reference_tempo, estimated_tempo, tolerance, 1.0 / 3.0))
    )


def oe1(reference_tempo, estimated_tempo, octave_factor=1.0):
    return np.log2((estimated_tempo * octave_factor) / reference_tempo)


def oe2(reference_tempo, estimated_tempo):
    factors = [1 / 3, 1 / 2, 1, 2, 3]
    oe = np.zeros_like(factors)

    for idx, factor in enumerate(factors):
        oe[idx] = oe1(reference_tempo, estimated_tempo, factor)

    return oe.min()


def aoe1(reference_tempo, estimated_tempo):
    return np.abs(oe1(reference_tempo, estimated_tempo))


def aoe2(reference_tempo, estimated_tempo):
    return
