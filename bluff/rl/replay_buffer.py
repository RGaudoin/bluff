"""Replay buffer for DQN training."""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class Transition:
    """A single transition in the replay buffer."""

    obs: np.ndarray           # Flattened observation
    action: int               # Action taken
    reward: float             # Reward received
    next_obs: np.ndarray      # Next flattened observation
    action_mask: np.ndarray   # Valid action mask for next_obs
    terminated: bool          # Episode naturally ended
    truncated: bool           # Episode was cut off


class ReplayBuffer:
    """
    Simple replay buffer for DQN.

    Stores transitions and samples random minibatches for training.
    Uses a circular buffer for memory efficiency.
    """

    def __init__(self, capacity: int = 100000):
        """
        Initialize replay buffer.

        Args:
            capacity: Maximum number of transitions to store
        """
        self.capacity = capacity
        self.buffer: List[Optional[Transition]] = [None] * capacity
        self.position = 0
        self.size = 0

    def push(self, transition: Transition) -> None:
        """
        Add a transition to the buffer.

        Args:
            transition: Transition to add
        """
        self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a random batch of transitions.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Tuple of (obs, actions, rewards, next_obs, action_masks, terminated, truncated)
            Each is a numpy array with batch_size as first dimension
        """
        indices = np.random.choice(self.size, size=batch_size, replace=False)
        transitions = [self.buffer[i] for i in indices]

        obs = np.stack([t.obs for t in transitions])
        actions = np.array([t.action for t in transitions], dtype=np.int64)
        rewards = np.array([t.reward for t in transitions], dtype=np.float32)
        next_obs = np.stack([t.next_obs for t in transitions])
        action_masks = np.stack([t.action_mask for t in transitions])
        terminated = np.array([t.terminated for t in transitions], dtype=np.float32)
        truncated = np.array([t.truncated for t in transitions], dtype=np.float32)

        return obs, actions, rewards, next_obs, action_masks, terminated, truncated

    def __len__(self) -> int:
        """Return current buffer size."""
        return self.size

    def can_sample(self, batch_size: int) -> bool:
        """Check if we have enough samples for a batch."""
        return self.size >= batch_size
