"""
Global uniqueness and identifiability region analysis.
"""

import numpy as np
from typing import Callable


class GlobalIdentifiability:
    """Assess global uniqueness of state recovery in measurement space."""

    def __init__(
        self,
        observation_func: Callable,
        noise_covariance: np.ndarray = None,
    ):
        """
        Initialize global identifiability checker.

        Args:
            observation_func: y = f(state) where state = [ε, η]
            noise_covariance: Noise covariance Σ, shape (n_obs, n_obs)
        """
        self.observation_func = observation_func
        self.noise_cov = noise_covariance

    def mahalanobis_distance(
        self,
        state_a: np.ndarray,
        state_b: np.ndarray,
    ) -> float:
        """
        Compute Mahalanobis distance between two states in observation space.

        d² = [y_a - y_b]^T Σ^(-1) [y_a - y_b]

        Args:
            state_a, state_b: State vectors

        Returns:
            distance: Mahalanobis distance
        """
        y_a = self.observation_func(state_a)
        y_b = self.observation_func(state_b)

        Δy = y_a - y_b

        if self.noise_cov is not None:
            Σ_inv = np.linalg.inv(self.noise_cov + 1e-16 * np.eye(len(y_a)))
        else:
            Σ_inv = np.eye(len(y_a))

        dist_sq = Δy @ Σ_inv @ Δy
        return np.sqrt(np.abs(dist_sq))

    def state_grid_distances(
        self,
        epsilon_grid: np.ndarray,
        eta_grid: np.ndarray,
        min_separation: dict = None,
    ) -> dict:
        """
        Compute pairwise distances in state grid.

        Only compare states differing by at least min_separation.

        Args:
            epsilon_grid: Compression values
            eta_grid: Bending values
            min_separation: {'epsilon': Δε_min, 'eta': Δη_min}

        Returns:
            result: dict with 'distance_matrix', 'confusion_pairs', 'global_min'
        """
        if min_separation is None:
            min_separation = {'epsilon': 0.01, 'eta': 0.001}

        n_eps = len(epsilon_grid)
        n_eta = len(eta_grid)
        total = n_eps * n_eta

        # Grid indices
        idx_map = {}
        grid_points = []
        for i, eps in enumerate(epsilon_grid):
            for j, eta in enumerate(eta_grid):
                idx = i * n_eta + j
                idx_map[(i, j)] = idx
                grid_points.append(np.array([eps, eta]))

        # Pairwise distances
        distances = np.full((total, total), np.nan)
        confusion_pairs = []

        for idx_i in range(total):
            for idx_j in range(idx_i + 1, total):
                state_i = grid_points[idx_i]
                state_j = grid_points[idx_j]

                # Check separation requirement
                Δε = np.abs(state_i[0] - state_j[0])
                Δη = np.abs(state_i[1] - state_j[1])

                if (Δε >= min_separation['epsilon'] or
                    Δη >= min_separation['eta']):
                    d = self.mahalanobis_distance(state_i, state_j)
                    distances[idx_i, idx_j] = d
                    distances[idx_j, idx_i] = d

                    # Flag confusion if distance is small
                    if d < 6.0:  # Mahalanobis < 6σ is confusing
                        confusion_pairs.append({
                            'state_i': state_i,
                            'state_j': state_j,
                            'distance': d,
                        })

        # Global minimum
        valid_distances = distances[~np.isnan(distances)]
        global_min_dist = np.min(valid_distances) if len(valid_distances) > 0 else np.nan

        result = {
            'distance_matrix': distances,
            'confusion_pairs': confusion_pairs,
            'global_min_distance': global_min_dist,
            'num_confusing_pairs': len(confusion_pairs),
        }

        return result


class BlindInversion:
    """Blind inversion with surrogate model."""

    def __init__(
        self,
        observation_func: Callable,
        epsilon_grid: np.ndarray,
        eta_grid: np.ndarray,
        noise_model: dict = None,
    ):
        """
        Initialize blind inversion.

        Args:
            observation_func: y = f(state)
            epsilon_grid, eta_grid: Training grids
            noise_model: Noise configuration
        """
        self.observation_func = observation_func
        self.epsilon_grid = epsilon_grid
        self.eta_grid = eta_grid
        self.noise_model = noise_model or {'type': 'gaussian', 'snr_db': 60}

    def _build_surrogate_lut(self) -> np.ndarray:
        """Build lookup table of observations."""
        n_eps = len(self.epsilon_grid)
        n_eta = len(self.eta_grid)
        sample_obs = self.observation_func(
            np.array([self.epsilon_grid[0], self.eta_grid[0]])
        )
        n_obs = len(sample_obs)

        lut = np.zeros((n_eps, n_eta, n_obs))

        for i, eps in enumerate(self.epsilon_grid):
            for j, eta in enumerate(self.eta_grid):
                lut[i, j, :] = self.observation_func(np.array([eps, eta]))

        return lut

    def invert_nearest_neighbor(
        self,
        y_meas: np.ndarray,
    ) -> dict:
        """
        Simple nearest-neighbor inversion in measurement space.

        Args:
            y_meas: Measured observation vector

        Returns:
            result: dict with 'epsilon_est', 'eta_est', 'distance'
        """
        lut = self._build_surrogate_lut()

        n_eps, n_eta, n_obs = lut.shape

        # L2 distance to each grid point
        distances = np.zeros((n_eps, n_eta))
        for i in range(n_eps):
            for j in range(n_eta):
                distances[i, j] = np.linalg.norm(lut[i, j, :] - y_meas)

        # Minimum
        i_min, j_min = np.unravel_index(np.argmin(distances), distances.shape)

        result = {
            'epsilon_est': self.epsilon_grid[i_min],
            'eta_est': self.eta_grid[j_min],
            'distance': distances[i_min, j_min],
            'grid_index': (i_min, j_min),
        }

        return result


if __name__ == '__main__':
    # Test: Global uniqueness
    def test_obs(state):
        eps, eta = state
        freq = np.logspace(6, 9, 31)
        s11_real = 0.2 * eps + 0.1 * eta * np.sin(2 * np.pi * freq / 1e9)
        s11_imag = 0.3 * (1 - eps) - 0.2 * eta
        return np.concatenate([s11_real, s11_imag])

    print("Global identifiability test:\n")

    epsilon_grid = np.linspace(0, 0.2, 5)
    eta_grid = np.linspace(0, 0.01, 5)

    gi = GlobalIdentifiability(test_obs, noise_covariance=1e-3 * np.eye(62))
    result = gi.state_grid_distances(epsilon_grid, eta_grid)

    print(f"Confusion pairs (d < 6σ): {result['num_confusing_pairs']}")
    print(f"Global minimum distance: {result['global_min_distance']:.3f}")
    print()

    # Blind inversion test
    print("Blind inversion test:")
    bi = BlindInversion(test_obs, epsilon_grid, eta_grid)

    # Synthetic "truth" state
    state_true = np.array([0.08, 0.0035])
    y_true = test_obs(state_true)
    print(f"True state: ε={state_true[0]:.3f}, η={state_true[1]:.6f}")

    result = bi.invert_nearest_neighbor(y_true)
    print(f"Estimated: ε={result['epsilon_est']:.3f}, η={result['eta_est']:.6f}")
    print(f"Distance: {result['distance']:.6f}")
