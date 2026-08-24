"""
One-port circuit: S11 and impedance transformations.
"""

import numpy as np


def s11_to_z(s11: np.ndarray, z0: float = 50.0) -> np.ndarray:
    """
    Convert S11 to impedance.

    Z = Z₀ · (1 + S11) / (1 - S11)

    Args:
        s11: Complex S-parameter, shape (...,)
        z0: Reference impedance (Ω)

    Returns:
        Z: Complex impedance
    """
    return z0 * (1 + s11) / (1 - s11 + 1e-16)


def z_to_s11(z: np.ndarray, z0: float = 50.0) -> np.ndarray:
    """
    Convert impedance to S11.

    S11 = (Z - Z₀) / (Z + Z₀)

    Args:
        z: Complex impedance
        z0: Reference impedance (Ω)

    Returns:
        S11: Complex S-parameter
    """
    return (z - z0) / (z + z0 + 1e-16)


def extract_rlc_low_frequency(
    freq: np.ndarray,
    z: np.ndarray,
    fmax: float = 1e7,
) -> dict:
    """
    Extract R, L, C from low-frequency impedance using linear regression.

    Below first resonance: Z ≈ R + jωL (capacitance negligible)
    Fit: Z_imag = ω·L_fit

    Args:
        freq: Frequency array (Hz)
        z: Complex impedance array
        fmax: Maximum frequency for low-freq fitting (Hz)

    Returns:
        params: dict with 'R', 'L_low', 'C_p', 'f_resonance'
    """
    # Find low-frequency region
    mask = freq <= fmax
    freq_low = freq[mask]
    z_low = z[mask]

    omega = 2 * np.pi * freq_low

    # Real part: mean (DC resistance)
    R = np.real(z_low[0])

    # Imaginary part: linear fit to ω·L
    L_imag = np.imag(z_low)
    coeffs = np.polyfit(omega, L_imag, 1)
    L_low = coeffs[0]

    # Distributed capacity estimate (optional)
    C_p = 0.0

    params = {
        'R': R,
        'L_low': L_low,
        'C_p': C_p,
    }

    return params


def find_resonance_peak(
    freq: np.ndarray,
    s11_mag: np.ndarray,
) -> dict:
    """
    Find first resonance (minimum |S11| near SRF).

    Args:
        freq: Frequency array
        s11_mag: |S11| magnitude

    Returns:
        resonance: dict with 'f_srf', 'index', 'mag'
    """
    # Find minimum in |S11|
    idx_min = np.argmin(s11_mag)
    f_srf = freq[idx_min]

    resonance = {
        'f_srf': f_srf,
        'index': idx_min,
        'magnitude': s11_mag[idx_min],
    }

    return resonance


if __name__ == '__main__':
    # Test conversions
    print("One-port circuit tests:\n")

    # Test 1: S11 ↔ Z conversion
    s11_test = 0.5 + 0.3j
    z_conv = s11_to_z(s11_test, z0=50.0)
    s11_back = z_to_s11(z_conv, z0=50.0)

    print(f"S11 test: {s11_test}")
    print(f"Z conversion: {z_conv:.4f} Ω")
    print(f"S11 back: {s11_back}")
    print(f"Roundtrip error: {np.abs(s11_test - s11_back):.2e}")
    print()

    # Test 2: LRC extraction
    freq = np.logspace(6, 9, 100)
    L_true = 10e-9  # 10 nH
    R_true = 1.0    # 1 Ω
    z_synthetic = R_true + 1j * 2 * np.pi * freq * L_true

    params = extract_rlc_low_frequency(freq, z_synthetic)
    print(f"LRC extraction:")
    print(f"  R_true = {R_true:.3f} Ω,  R_fit = {params['R']:.3f} Ω")
    print(f"  L_true = {L_true*1e9:.2f} nH, L_fit = {params['L_low']*1e9:.2f} nH")
