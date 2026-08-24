"""Shunt capacitor model for external capacitance loading."""

import numpy as np


def admittance_with_capacitor(
    y_coil: np.ndarray,
    C_ext: float,
    freq: np.ndarray,
) -> np.ndarray:
    """
    Coil admittance with external shunt capacitor.

    Y_total = Y_coil + j·ω·C_ext

    Args:
        y_coil: Coil admittance (complex)
        C_ext: External capacitance (F)
        freq: Frequency array (Hz)

    Returns:
        Y_total: Total admittance
    """
    omega = 2 * np.pi * freq
    return y_coil + 1j * omega * C_ext
