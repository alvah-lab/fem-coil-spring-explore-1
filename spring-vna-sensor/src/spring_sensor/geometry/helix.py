"""
Helix geometry generation with centerline bending and compression.
"""

import numpy as np
from .centerline import constant_curvature_centerline


def helical_coil_wireframe(
    s: np.ndarray,
    a: float,
    turns: int,
    c_centerline: np.ndarray,
    n_centerline: np.ndarray,
    wire_radius: float,
    segs_per_turn: int = 16,
) -> np.ndarray:
    """
    Generate wire centerline trajectory for helical coil with bending.

    r(s) = c(s) + a[cos(qs)·n(s) + sin(qs)·b]

    Args:
        s: Arc length nodes along centerline
        a: Mean coil radius (m)
        turns: Number of turns
        c_centerline: Centerline points
        n_centerline: Normal vectors of bent centerline
        wire_radius: Wire cross-section radius
        segs_per_turn: Discretization per turn (unused)

    Returns:
        r: Wire centerline, shape (N, 3)
    """
    ℓ = s[-1]
    q = 2 * np.pi * turns / ℓ  # Helical phase

    qs = q * s
    cos_qs = np.cos(qs)
    sin_qs = np.sin(qs)

    # Binormal (constant in global frame for centerline in xy-plane)
    b = np.zeros((len(s), 3))
    b[:, 2] = 1.0

    # Wire centerline: r(s) = c(s) + a[cos(qs)·n(s) + sin(qs)·b]
    r = (
        c_centerline
        + a * np.outer(cos_qs, np.ones(3)) * n_centerline
        + a * np.outer(sin_qs, np.ones(3)) * b
    )

    return r


def create_spring_coil_state(
    epsilon: float,
    kappa: float,
    turns: int,
    mean_radius: float,
    wire_radius: float,
    initial_length: float,
    axis_points: int = 101,
) -> dict:
    """
    Generate complete parametric spring coil state.

    Args:
        epsilon: Compression strain [0, 1)
        kappa: Curvature (m⁻¹)
        turns: Number of turns
        mean_radius: Coil mean radius (m)
        wire_radius: Wire radius (m)
        initial_length: Initial (uncompressed) axial length (m)
        axis_points: Discretization points along axis

    Returns:
        state: dict with 'centerline', 'wire_trajectory', 'parameters', etc.
    """
    # Deformed length
    ℓ = initial_length * (1 - epsilon)

    # Arc length array
    s = np.linspace(0, ℓ, axis_points)

    # Centerline with bending
    c, t, n = constant_curvature_centerline(s, kappa=kappa)

    # Wire helix
    r = helical_coil_wireframe(s, mean_radius, turns, c, n, wire_radius)

    state = {
        'epsilon': epsilon,
        'kappa': kappa,
        'eta': (mean_radius * kappa)**2,  # Bending metric
        'length': ℓ,
        'centerline': c,
        'tangent': t,
        'normal': n,
        'wire_trajectory': r,
        'parameters': {
            'turns': turns,
            'mean_radius': mean_radius,
            'wire_radius': wire_radius,
            'initial_length': initial_length,
        },
    }

    return state


if __name__ == '__main__':
    # Test: Create a sample spring coil
    print("Generating sample spring coil state...\n")

    state_0 = create_spring_coil_state(
        epsilon=0.0,  # No compression
        kappa=0.0,    # No bending
        turns=20,
        mean_radius=5e-3,
        wire_radius=0.25e-3,
        initial_length=50e-3,
        axis_points=51,
    )

    print(f"Undeformed straight coil:")
    print(f"  Length: {state_0['length']*1e3:.2f} mm")
    print(f"  Wire trajectory shape: {state_0['wire_trajectory'].shape}")
    print()

    state_1 = create_spring_coil_state(
        epsilon=0.1,   # 10% compression
        kappa=10.0,    # 10 m⁻¹ curvature
        turns=20,
        mean_radius=5e-3,
        wire_radius=0.25e-3,
        initial_length=50e-3,
        axis_points=51,
    )

    print(f"Deformed coil (ε=10%, κ=10 m⁻¹):")
    print(f"  Length: {state_1['length']*1e3:.2f} mm")
    print(f"  η = (a·κ)² = {state_1['eta']:.6f}")
    print(f"  Wire max xy excursion: {np.sqrt(state_1['wire_trajectory'][:, 0]**2 + state_1['wire_trajectory'][:, 1]**2).max()*1e3:.2f} mm")
