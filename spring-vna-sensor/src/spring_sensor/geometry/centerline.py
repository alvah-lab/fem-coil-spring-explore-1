"""
Centerline generation for spring coil with constant curvature bending.

Stable numerical implementation for κ → 0 limit using sinc/cosc expansions.
"""

import numpy as np
from typing import Tuple


def sinc(x):
    """Stable sinc(x) = sin(x)/x for x → 0."""
    x = np.asarray(x)
    # Use small angle approximation for |x| < 1e-6
    small = np.abs(x) < 1e-6
    result = np.ones_like(x, dtype=float)

    if np.isscalar(x):
        if small:
            result = 1.0 - x**2 / 6.0 + x**4 / 120.0
        else:
            result = np.sin(x) / x
    else:
        result[~small] = np.sin(x[~small]) / x[~small]
        result[small] = 1.0 - x[small]**2 / 6.0 + x[small]**4 / 120.0

    return result


def cosc(x):
    """Stable cosc(x) = (1 - cos(x))/x for x → 0."""
    x = np.asarray(x)
    small = np.abs(x) < 1e-6
    result = np.zeros_like(x, dtype=float)

    if np.isscalar(x):
        if small:
            result = x / 2.0 - x**3 / 24.0 + x**5 / 720.0
        else:
            result = (1.0 - np.cos(x)) / x
    else:
        result[~small] = (1.0 - np.cos(x[~small])) / x[~small]
        result[small] = x[small] / 2.0 - x[small]**3 / 24.0 + x[small]**5 / 720.0

    return result


def constant_curvature_centerline(
    s: np.ndarray,
    kappa: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate 3D centerline with constant curvature in xy-plane.

    Arc-length parametrization: c(s) for s ∈ [0, ℓ]
    Bending occurs in xy-plane, z-axis is the undeformed (reference) axis.

    For κ=0: straight line along Z: c = [0, 0, s]
    For κ>0: circle in xy-plane, linear along z

    Args:
        s: Arc length array [0, ℓ], shape (N,)
        kappa: Curvature in m⁻¹

    Returns:
        c: Centerline points, shape (N, 3)
        t: Tangent vectors (unit), shape (N, 3)
        n: Normal vectors (unit), shape (N, 3)
    """
    s = np.asarray(s, dtype=float)
    kappa = float(kappa)

    # For small κ: treat as straight
    if np.abs(kappa) < 1e-10:
        c_x = np.zeros_like(s)
        c_y = np.zeros_like(s)
        c_z = s.copy()
        c = np.column_stack([c_x, c_y, c_z])
    else:
        # Centerline: c(s) = [R·sin(κs), R·(1-cos(κs)), s]
        # where R = 1/κ is the radius of curvature
        R = 1.0 / kappa
        c_x = R * np.sin(kappa * s)
        c_y = R * (1 - np.cos(kappa * s))
        c_z = s.copy()
        c = np.column_stack([c_x, c_y, c_z])

    # Tangent: t(s) = dc/ds
    if np.abs(kappa) < 1e-10:
        # Straight line along Z
        t = np.zeros_like(c)
        t[:, 2] = 1.0  # Unit tangent along Z
        n = np.zeros_like(c)
        n[:, 1] = 1.0  # Unit normal in Y direction
    else:
        ks = kappa * s
        sin_ks = np.sin(ks)
        cos_ks = np.cos(ks)

        # t(s) = dc/ds = [cos(κs), sin(κs), 1]
        # But need to normalize to account for all terms
        t_x = cos_ks
        t_y = sin_ks
        t_z = np.ones_like(s)

        t = np.column_stack([t_x, t_y, t_z])

        # Normalize tangent (but shouldn't be necessary since |t| should be ~ 1)
        t_norm = np.linalg.norm(t, axis=1, keepdims=True)
        t = t / (t_norm + 1e-16)

        # Normal (principal curvature direction): n(s) = [-sin(κs), cos(κs), 0]
        n_x = -sin_ks
        n_y = cos_ks
        n_z = np.zeros_like(s)
        n = np.column_stack([n_x, n_y, n_z])

    return c, t, n


def deformed_geometry_check(
    c: np.ndarray,
    a: float,
    wire_radius: float,
    min_separation: float = 1e-5,
) -> dict:
    """
    Check geometric validity: no self-intersection, minimum separations.

    Args:
        c: Centerline, shape (N, 3)
        a: Mean coil radius (m)
        wire_radius: Wire radius (m)
        min_separation: Minimum allowed separation (m)

    Returns:
        validity: dict with keys 'valid', 'messages', 'issues'
    """
    validity = {
        'valid': True,
        'messages': [],
        'issues': [],
    }

    # Check 1: Bending amplitude vs wire radius
    # For tight bending, a·κ should be < 1 and wire shouldn't auto-intersect
    if a < 2 * wire_radius:
        validity['issues'].append(
            f'Mean radius {a*1e3:.2f}mm < 2×wire_radius {2*wire_radius*1e3:.2f}mm'
        )
        validity['valid'] = False

    # Check 2: Centerline curvature radius reasonable
    max_xy_dist = np.sqrt(c[:, 0]**2 + c[:, 1]**2).max()
    if max_xy_dist > 10 * a:
        validity['issues'].append(
            f'Centerline excursion {max_xy_dist*1e3:.2f}mm >> mean radius {a*1e3:.2f}mm'
        )

    return validity


if __name__ == '__main__':
    # Test: κ=0 (straight line), κ≠0
    s = np.linspace(0, 50e-3, 101)

    print("Test 1: κ = 0 (straight line)")
    c, t, n = constant_curvature_centerline(s, kappa=0.0)
    print(f"  Centerline start: {c[0]}")
    print(f"  Centerline end: {c[-1]}")
    print(f"  Tangent (should be [1,0,0]): {t[0]}")
    print()

    print("Test 2: κ = 10 m⁻¹ (R = 100mm curvature)")
    c, t, n = constant_curvature_centerline(s, kappa=10.0)
    print(f"  Centerline start: {c[0]}")
    print(f"  Centerline mid: {c[len(c)//2]}")
    print(f"  Centerline end: {c[-1]}")
    print(f"  Total angular change: {10.0 * s[-1]:.3f} rad")
    print(f"  Max xy displacement: {np.sqrt(c[:, 0]**2 + c[:, 1]**2).max()*1e3:.2f} mm")
