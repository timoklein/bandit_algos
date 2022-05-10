from __future__ import annotations
import numpy as np


class UCB:
    """Simple upper confidence bound (UCB) implementation."""

    k: int
    confidence: float
    q_values: np.ndarray

    def __init__(self, k: int, confidence: float = 2.0) -> None:
        # Create a vector of counters for easier computation later
        self.step_counter = np.zeros(k)
        self.k = k
        self.c = confidence

        self.q_values = np.zeros(k, dtype=float)
        self.action_counts = np.zeros(k, dtype=int)

    def select_action(self) -> int:
        """Perform action selection according to the UCB rule."""
        if (0 == self.action_counts).any():  # type: ignore[attr-defined]
            # There is an unexplored action -> Select it first
            indices = np.where(self.action_counts == 0)
            return np.random.choice(indices[0])
        else:
            ucb = self.q_values + self.c * np.sqrt(
                np.log(self.step_counter) / self.action_counts
            )
            return ucb.argmax()  # type: ignore[return-value]

    def update(self, action, reward) -> None:
        """Update step counter, action counts and action q values."""
        self.step_counter += 1
        self.action_counts[action] += 1
        self.q_values[action] += (1 / self.action_counts[action]) * (
            reward - self.q_values[action]
        )
