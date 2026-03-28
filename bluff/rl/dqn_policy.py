"""DQN policy with action masking for Bluff."""

from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from bluff.agents.wrappers import EnvPolicy
from bluff.rl.obs_utils import flatten_obs


class DQNNetwork(nn.Module):
    """
    Simple MLP Q-network for DQN.

    Architecture: obs -> hidden layers -> Q-values for each action
    """

    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        hidden_dims: tuple = (128, 128),
    ):
        """
        Initialize Q-network.

        Args:
            obs_dim: Dimension of flattened observation
            num_actions: Number of discrete actions
            hidden_dims: Tuple of hidden layer sizes
        """
        super().__init__()

        layers = []
        prev_dim = obs_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_actions))

        self.network = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Compute Q-values for all actions.

        Args:
            obs: Batch of flattened observations (batch_size, obs_dim)

        Returns:
            Q-values (batch_size, num_actions)
        """
        return self.network(obs)


class DQNPolicy(EnvPolicy):
    """
    DQN policy that implements EnvPolicy interface.

    Uses epsilon-greedy exploration with action masking.
    """

    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        num_players: int,
        num_faces: int,
        max_dice: int,
        max_tracked_players: int,
        hidden_dims: tuple = (128, 128),
        epsilon: float = 0.1,
        device: str = "cpu",
    ):
        """
        Initialize DQN policy.

        Args:
            obs_dim: Dimension of flattened observation
            num_actions: Number of discrete actions
            num_players: Number of players (for obs flattening)
            num_faces: Number of die faces (for obs flattening)
            max_dice: Maximum total dice (for obs flattening)
            max_tracked_players: Max tracked players (for obs flattening)
            hidden_dims: Hidden layer sizes for Q-network
            epsilon: Exploration rate for epsilon-greedy
            device: Device to run on ("cpu" or "cuda")
        """
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.num_players = num_players
        self.num_faces = num_faces
        self.max_dice = max_dice
        self.max_tracked_players = max_tracked_players
        self.epsilon = epsilon
        self.device = torch.device(device)

        # Q-network and target network
        self.q_network = DQNNetwork(obs_dim, num_actions, hidden_dims).to(self.device)
        self.target_network = DQNNetwork(obs_dim, num_actions, hidden_dims).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Random number generator
        self.rng = np.random.default_rng()

    def select_action(self, obs: Dict[str, Any], action_mask: np.ndarray) -> int:
        """
        Select action using epsilon-greedy with action masking.

        Args:
            obs: Dict observation from BluffEnv
            action_mask: Boolean mask of valid actions

        Returns:
            Selected action integer
        """
        valid_actions = np.flatnonzero(action_mask)

        if len(valid_actions) == 0:
            raise ValueError("No valid actions available")

        # Epsilon-greedy exploration
        if self.rng.random() < self.epsilon:
            return int(self.rng.choice(valid_actions))

        # Greedy action selection
        flat_obs = flatten_obs(
            obs,
            self.num_players,
            self.num_faces,
            self.max_dice,
            self.max_tracked_players,
        )

        with torch.no_grad():
            obs_tensor = torch.tensor(flat_obs, dtype=torch.float32, device=self.device)
            obs_tensor = obs_tensor.unsqueeze(0)  # Add batch dim

            q_values = self.q_network(obs_tensor).squeeze(0)  # (num_actions,)

            # Mask invalid actions with -inf
            mask_tensor = torch.tensor(action_mask, dtype=torch.bool, device=self.device)
            q_values[~mask_tensor] = float("-inf")

            action = q_values.argmax().item()

        return int(action)

    def get_q_values(
        self,
        obs_batch: np.ndarray,
        use_target: bool = False,
    ) -> torch.Tensor:
        """
        Get Q-values for a batch of observations.

        Args:
            obs_batch: Batch of flattened observations (batch_size, obs_dim)
            use_target: Whether to use target network

        Returns:
            Q-values tensor (batch_size, num_actions)
        """
        obs_tensor = torch.tensor(obs_batch, dtype=torch.float32, device=self.device)
        network = self.target_network if use_target else self.q_network
        return network(obs_tensor)

    def update_target_network(self) -> None:
        """Copy Q-network weights to target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def soft_update_target_network(self, tau: float = 0.005) -> None:
        """
        Soft update target network weights.

        Args:
            tau: Interpolation factor (0 = no update, 1 = hard copy)
        """
        for target_param, q_param in zip(
            self.target_network.parameters(), self.q_network.parameters()
        ):
            target_param.data.copy_(tau * q_param.data + (1 - tau) * target_param.data)

    def set_epsilon(self, epsilon: float) -> None:
        """Set exploration rate."""
        self.epsilon = epsilon

    def train_mode(self) -> None:
        """Set networks to training mode."""
        self.q_network.train()
        self.target_network.train()

    def eval_mode(self) -> None:
        """Set networks to evaluation mode."""
        self.q_network.eval()
        self.target_network.eval()

    def save(self, path: str, dice_per_player: int = 3) -> None:
        """Save model weights and metadata."""
        torch.save({
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "metadata": {
                "num_players": self.num_players,
                "num_faces": self.num_faces,
                "dice_per_player": dice_per_player,
                "max_dice": self.max_dice,
                "max_tracked_players": self.max_tracked_players,
                "obs_dim": self.obs_dim,
                "num_actions": self.num_actions,
            },
        }, path)

    def load(self, path: str) -> None:
        """Load model weights."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])

    @staticmethod
    def load_metadata(path: str) -> Optional[Dict[str, Any]]:
        """Load just the metadata from a checkpoint without loading weights."""
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            return checkpoint.get("metadata")
        except Exception:
            return None
