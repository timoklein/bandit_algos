from __future__ import annotations
import numpy as np


class ThompsonSampling:
    """Implementation of Thompson Sampling with known prior variance."""

    k: int
    selection_counts: np.ndarray
    step_counts: np.ndarray
    prior_vars: np.ndarray
    posterior_means: np.ndarray
    posterior_vars: np.ndarray

    def __init__(
        self, k: int, prior_mean: float = 0.0, prior_variance: float = 1.0
    ) -> None:
        self.k = k
        # NOTE: We must track selection counts and rewards for each arm individually
        self.selection_counts = np.zeros(k, dtype=int)
        self.rewards = np.zeros(k, dtype=float)
        self.prior_vars = np.tile(float(prior_variance), k)
        # Prior mean is not used in the update so we can directly put it into the posterior
        self.posterior_means = np.tile(float(prior_mean), k)
        self.posterior_vars = np.ones(k, dtype=float)

    def select_action(self) -> int:  # type: ignore[return-value]
        """Sample an action from the approximate posterior."""
        posterior_stddevs = np.sqrt(self.posterior_vars)
        return np.random.normal(self.posterior_means, posterior_stddevs).argmax()  # type: ignore[return-value]

    def update(self, action, reward) -> None:
        """Update the posterior (closed form as prior is also Normal)."""
        self.rewards[action] += reward
        self.selection_counts[action] += 1
        # Update rule for Normal prior with known variance, see here https://en.wikipedia.org/wiki/Conjugate_prior
        var_updated = 1 / (
            1 / self.posterior_vars[action]
            + self.selection_counts[action] / self.prior_vars[action]
        )
        self.posterior_means[action] = var_updated * (
            self.posterior_means[action] / self.posterior_vars[action]
            + np.sum(self.rewards[action]) / self.prior_vars[action]
        )
        self.posterior_vars[action] = var_updated
