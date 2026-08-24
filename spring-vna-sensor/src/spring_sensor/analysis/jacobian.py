"""
Jacobian/sensitivity matrix computation for identifiability analysis.
"""

import numpy as np
from scipy.optimize import approx_fprime


def jacobian_central_difference(
    observation_func,
    state: np.ndarray,
    state_scale: np.ndarray,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """
    Compute Jacobian J = ∂y/∂x using central difference.

    J[i,j] = ∂y_i/∂x_j

    Args:
        observation_func: Function y = f(state) returning observation vector
        state: State vector [ε, η], shape (2,)
        state_scale: Scaling for normalized states [Δε, Δη]
        epsilon: Step size for finite difference

    Returns:
        J: Jacobian matrix, shape (n_obs, 2)
    """
    n_obs = len(observation_func(state))
    J = np.zeros((n_obs, 2))

    for j in range(2):
        state_plus = state.copy()
        state_minus = state.copy()

        # Perturbation in scaled state
        h = epsilon * state_scale[j]

        state_plus[j] += h
        state_minus[j] -= h

        y_plus = observation_func(state_plus)
        y_minus = observation_func(state_minus)

        J[:, j] = (y_plus - y_minus) / (2 * h)

    return J


class SensitivityAnalysis:
    """Sensitivity matrix computation and analysis."""

    def __init__(
        self,
        observation_func,
        state_scaling: np.ndarray,
        noise_covariance: np.ndarray = None,
    ):
        """
        Initialize sensitivity analysis.

        Args:
            observation_func: y = f(q) where q = [ε, η]
            state_scaling: Scaling vector [Δε_target, Δη_target]
            noise_covariance: Noise covariance Σ, shape (n_obs, n_obs)
        """
        self.observation_func = observation_func
        self.state_scaling = state_scaling
        self.noise_cov = noise_covariance

    def jacobian_at_state(self, state: np.ndarray) -> np.ndarray:
        """
        Compute Jacobian at given state.

        Args:
            state: [ε, η]

        Returns:
            J: Jacobian, shape (n_obs, 2)
        """
        return jacobian_central_difference(
            self.observation_func,
            state,
            self.state_scaling,
            epsilon=1e-6,
        )

    def whitened_jacobian(self, J: np.ndarray) -> np.ndarray:
        """
        Compute noise-whitened Jacobian.

        J_w = Σ^(-1/2) · J

        Args:
            J: Jacobian, shape (n_obs, 2)

        Returns:
            J_w: Whitened Jacobian
        """
        if self.noise_cov is None:
            return J

        # Cholesky decomposition: Σ = L·L^T
        L = np.linalg.cholesky(self.noise_cov)
        L_inv = np.linalg.inv(L)

        return L_inv @ J

    def svd_analysis(self, J: np.ndarray) -> dict:
        """
        SVD analysis of Jacobian.

        Args:
            J: Jacobian or whitened Jacobian

        Returns:
            analysis: dict with 'U', 'sigma', 'V', 'rank', 'condition_number'
        """
        U, sigma, Vt = np.linalg.svd(J, full_matrices=False)

        rank = np.sum(sigma > 1e-10 * sigma[0])
        cond = sigma[0] / (sigma[-1] + 1e-16)

        analysis = {
            'U': U,
            'sigma': sigma,
            'V': Vt.T,
            'rank': rank,
            'condition_number': cond,
            'sigma_min': sigma[-1] if len(sigma) > 1 else sigma[0],
        }

        return analysis

    def identifiability_at_state(self, state: np.ndarray) -> dict:
        """
        Full identifiability assessment at given state.

        Args:
            state: [ε, η]

        Returns:
            result: dict with 'jacobian', 'whitened_jacobian', 'svd', 'identifiable'
        """
        J = self.jacobian_at_state(state)
        J_w = self.whitened_jacobian(J)
        svd_result = self.svd_analysis(J_w)

        # Identifiability criterion: rank = 2
        identifiable = svd_result['rank'] == 2

        result = {
            'state': state,
            'jacobian': J,
            'whitened_jacobian': J_w,
            'svd': svd_result,
            'identifiable': identifiable,
            'condition_number': svd_result['condition_number'],
        }

        return result


if __name__ == '__main__':
    # Test: Synthetic observation function
    def test_observation(state):
        """Synthetic full-rank observation."""
        eps, eta = state
        freq = np.logspace(6, 9, 31)
        omega = 2 * np.pi * freq

        # Synthetic S11: varies with both ε and η
        s11_real = 0.3 * eps + 0.1 * np.sqrt(eta) * np.sin(omega / 1e8)
        s11_imag = -0.5 + 0.2 * eps + 0.15 * eta * np.cos(omega / 1e8)

        # Concatenate
        y = np.concatenate([s11_real, s11_imag])
        return y

    print("Jacobian and SVD analysis test:\n")

    # Setup
    state_scale = np.array([0.005, 0.0025])  # [Δε, Δη]
    noise_cov = 1e-3 * np.eye(62)

    analysis = SensitivityAnalysis(test_observation, state_scale, noise_cov)

    # Test state
    state_test = np.array([0.05, 0.01])
    result = analysis.identifiability_at_state(state_test)

    print(f"State: ε={result['state'][0]:.3f}, η={result['state'][1]:.4f}")
    print(f"Identifiable: {result['identifiable']}")
    print(f"Condition number: {result['condition_number']:.2f}")
    print(f"Singular values: {result['svd']['sigma']}")
