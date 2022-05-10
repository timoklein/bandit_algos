from __future__ import annotations
import numpy as np


class EGreedy:
    """Basic epsilon-greedy implementation."""

    epsilon: float
    k: int
    lr: float | None
    q_values: np.ndarray
    action_counts: np.ndarray

    def __init__(
        self, k: int, eps: float, optimistic: bool = False, lr: float | None = None
    ) -> None:
        self.epsilon = eps
        self.k = k
        self.lr = lr

        init_value = 5 if optimistic else 0

        self.q_values = np.tile(float(init_value), k)
        self.action_counts = np.zeros(k)

    def select_action(self) -> int:
        """Take the epsilon greedy action."""
        if np.random.rand() < self.epsilon:
            # Return index of random arm
            return np.random.randint(0, self.k)
        else:
            # Return index of best arm
            # Will always break ties by selecting first arm, which doesn't matter for this case
            return self.q_values.argmax()  # type: ignore[return-value]

    def update(self, action, reward) -> None:
        """Update the action-value estimates according to the received reward."""
        self.action_counts[action] += 1
        step_size = self.lr if self.lr is not None else (1 / self.action_counts[action])
        self.q_values[action] += step_size * (reward - self.q_values[action])
