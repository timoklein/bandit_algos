from __future__ import annotations
import numpy as np


class GradientBandit:
    """Gradient-based bandit algorithm using the Boltzmann distribution for action selection."""

    k: int
    lr: float
    use_baseline: bool
    step_count: int
    mean_reward: float
    preferences: np.ndarray

    def __init__(self, k: int, lr: float = 0.1, use_baseline: bool = False) -> None:
        # Create a vector of counters for easier computation later
        self.k = k
        self.lr = lr
        self.use_baseline = use_baseline

        self.step_count = 0
        self.mean_reward = 0
        self.preferences = np.zeros(k)

    def select_action(self) -> int:
        """Compute the softmax over preferences and sample an action."""
        # Compute max used for stable softmax
        C = np.max(self.preferences)
        exp_sum = np.sum(np.exp(self.preferences - C))
        # Cache the probabilities for the update
        self.probs = np.exp(self.preferences - C) / exp_sum
        return np.random.multinomial(n=1, pvals=self.probs).argmax()  # type: ignore[return-value]

    def update(self, action, reward) -> None:
        """Gradient update for the preferences."""
        assert hasattr(self, "probs"), "Must select action before update."
        self.step_count += 1
        self.mean_reward += (reward - self.mean_reward) / self.step_count
        if self.use_baseline:
            reward_diff = reward - self.mean_reward
        else:
            # Don't use any baseline
            reward_diff = reward
        # Update for the selected action
        self.preferences[action] += self.lr * reward_diff * (1 - self.probs[action])

        # Update for the not-chosen actions
        idx_vec = np.arange(self.k) != action
        self.preferences[idx_vec] -= self.lr * reward_diff * self.probs[idx_vec]
