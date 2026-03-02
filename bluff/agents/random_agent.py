"""Random agent that selects uniformly from valid actions."""

import hashlib
from typing import List, Optional

import numpy as np

from bluff.agents.base import BaseAgent
from bluff.game.types import Action
from bluff.game.game_state import PlayerObservation


class RandomAgent(BaseAgent):
    """Agent that selects uniformly random valid actions."""

    def __init__(
        self,
        policy_id: Optional[str] = None,
        player_id: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        # Auto-generate policy_id if not provided
        if policy_id is None:
            # Use hash for uniqueness, include seed if provided for readability
            random_hash = hashlib.sha256(np.random.bytes(8)).hexdigest()[:6]
            if seed is not None:
                policy_id = f"rand_{seed}_{random_hash}"
            else:
                policy_id = f"rand_{random_hash}"

        super().__init__(policy_id, player_id, seed)
        self.rng = np.random.default_rng(seed)

    def select_action(
        self,
        observation: PlayerObservation,
        valid_actions: List[Action],
    ) -> Action:
        """Select a random valid action."""
        idx = self.rng.integers(0, len(valid_actions))
        return valid_actions[idx]
