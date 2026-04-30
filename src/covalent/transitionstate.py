import psi4
import numpy as np


def is_saddle_point(wfn: psi4.core.Wavefunction) -> bool:
    """
    Check if the optimized geometry is a saddle point (TS) by counting imaginary frequencies.

    Returns
    -------
    bool
        True if exactly one imaginary frequency is present, indicating a TS.
    """
    freqs = wfn.frequencies().to_array()
    num_imaginary = np.sum(freqs < 0)
    return num_imaginary == 1


def is_local_minimum(wfn: psi4.core.Wavefunction) -> bool:
    """
    Check if the optimized geometry is a minimum by ensuring no imaginary frequencies.

    Returns
    -------
    bool
        True if all frequencies are real (≥ 0), indicating a minimum.
    """
    freqs = wfn.frequencies().to_array()
    num_imaginary = np.sum(freqs < 0)
    return num_imaginary == 0