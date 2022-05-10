import numpy as np


class KArmedBandit:
    """A simple k-armed bandit implementation."""

    def __init__(self, k: int, is_stationary: bool = True) -> None:
        self.k = k
        self.is_stationary = is_stationary
        self.means = np.random.randn(k)

    def pull_arm(self, idx: int) -> np.ndarray:
        """Pull an arm of the bandit."""
        if self.is_stationary:
            return np.random.normal(loc=self.means[idx], scale=1)
        else:
            self.means += 0.01 * np.random.randn(self.k)
            return np.random.normal(loc=self.means[idx], scale=1)

    def best_arm(self) -> int:
        """Return the index of the arm with highest mean return."""
        return self.means.argmax()  # type: ignore[return-value]

    def optimal_return(self) -> np.ndarray:
        """Best long run return -> Expectation of highest return arm."""
        return self.means.max()
