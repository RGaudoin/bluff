"""Abstract base class for Bluff game agents."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

from bluff.game.types import Action, RoundResult
from bluff.game.game_state import PlayerObservation


class BaseAgent(ABC):
    """
    Abstract base class for Bluff game agents.

    Agents receive observations (not full state) and select actions.

    Attributes:
        policy_id: Permanent identifier for the policy/model (e.g., "dqn_v2")
        player_id: Visible identifier for a session (defaults to policy_id)
        seat: Current position at the table (0, 1, 2...) - changes each game
    """

    def __init__(
        self,
        policy_id: str,
        player_id: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize agent.

        Args:
            policy_id: Permanent identifier for the policy/model
            player_id: Visible session identifier (defaults to policy_id)
            seed: Random seed for reproducibility
        """
        self.policy_id = policy_id
        self.player_id = player_id if player_id is not None else policy_id
        self.seat: Optional[int] = None  # Set when joining a game
        self.seed = seed

    def set_seat(self, seat: int) -> None:
        """
        Set the seat (table position) for the current game.

        Args:
            seat: Position at the table (0, 1, 2, ...)
        """
        self.seat = seat

    @abstractmethod
    def select_action(
        self,
        observation: PlayerObservation,
        valid_actions: List[Action],
    ) -> Action:
        """
        Select an action given the current observation.

        Args:
            observation: What the agent can see
            valid_actions: List of legal actions

        Returns:
            Selected Action
        """
        pass

    def on_round_end(
        self,
        revealed_dice: Dict[int, Tuple[int, ...]],
        result: RoundResult,
        seat_to_player_id: Dict[int, str],
    ) -> None:
        """
        Called when a round ends, revealing all dice.

        Agents can use this to update beliefs/models about opponents.

        Args:
            revealed_dice: Dict mapping seat to their dice tuple
            result: The round outcome
            seat_to_player_id: Dict mapping seat to player_id
        """
        pass

    def on_game_start(self, seat_to_player_id: Dict[int, str]) -> None:
        """
        Called when a new game starts.

        Allows agents to recognize opponents across games in a session.

        Args:
            seat_to_player_id: Dict mapping seat to player_id for all players
        """
        pass

    def on_game_end(self, winner_seat: int, seat_to_player_id: Dict[int, str]) -> None:
        """
        Called when the game ends.

        Args:
            winner_seat: The winning player's seat
            seat_to_player_id: Dict mapping seat to player_id
        """
        pass
