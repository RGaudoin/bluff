"""
Wrappers to allow agents to work in both BluffGame and BluffEnv contexts.

Two adapters:
1. EnvPolicyWrapper: Wraps a BaseAgent to work with BluffEnv
2. GameAgentWrapper: Wraps an env-based policy to work with run_game/BluffGame
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from bluff.agents.base import BaseAgent
from bluff.game.types import Action, ActionType, Bid
from bluff.game.game_state import PlayerObservation
from bluff.gym_env.spaces import encode_bid, decode_action


class EnvPolicy(ABC):
    """
    Abstract base class for policies that work with BluffEnv observations.

    These policies receive dict observations (numpy arrays) and return
    discrete action integers.
    """

    @abstractmethod
    def select_action(self, obs: Dict[str, Any], action_mask: np.ndarray) -> int:
        """
        Select an action given env observation.

        Args:
            obs: Dictionary observation from BluffEnv
            action_mask: Boolean mask of valid actions

        Returns:
            Action integer for the discrete action space
        """
        pass


class EnvPolicyWrapper(EnvPolicy):
    """
    Wraps a BaseAgent to work with BluffEnv.

    Converts:
    - Dict observation → PlayerObservation (for agent)
    - Action object → action int (for env)

    Usage:
        agent = HeuristicAgent("heuristic")
        policy = EnvPolicyWrapper(agent, num_faces=6, max_dice=9)

        # In env loop:
        action_int = policy.select_action(obs, obs["action_mask"])
        env.step(action_int)
    """

    def __init__(
        self,
        agent: BaseAgent,
        num_faces: int = 6,
        max_dice: int = 9,
    ):
        """
        Initialize wrapper.

        Args:
            agent: BaseAgent to wrap
            num_faces: Number of faces per die
            max_dice: Maximum total dice (for action encoding)
        """
        self.agent = agent
        self.num_faces = num_faces
        self.max_dice = max_dice

    @property
    def policy_id(self) -> str:
        return self.agent.policy_id

    @property
    def player_id(self) -> str:
        return self.agent.player_id

    def select_action(self, obs: Dict[str, Any], action_mask: np.ndarray) -> int:
        """Convert env observation to PlayerObservation, get action, convert back."""
        # Convert dict obs to PlayerObservation
        player_obs = self._obs_to_player_observation(obs)

        # Get valid actions as Action objects
        valid_actions = self._mask_to_actions(action_mask, player_obs.seat)

        # Let agent select
        action = self.agent.select_action(player_obs, valid_actions)

        # Convert Action back to int
        return self._action_to_int(action)

    def _obs_to_player_observation(self, obs: Dict[str, Any]) -> PlayerObservation:
        """Convert env dict observation to PlayerObservation dataclass."""
        private = obs["private"]
        public_round = obs["public_round"]

        # Reconstruct own_dice from dice_counts
        dice_counts = private["dice_counts"]
        own_dice = []
        for face, count in enumerate(dice_counts, start=1):
            own_dice.extend([face] * int(count))
        own_dice = tuple(own_dice)

        my_seat = int(public_round["my_seat"])

        # Reconstruct other_seats_dice_counts
        dice_per_seat = public_round["dice_per_seat"]
        other_counts = tuple(
            (seat, int(dice_per_seat[seat]))
            for seat in range(len(dice_per_seat))
            if seat != my_seat
        )

        # Reconstruct seat_to_player_id (use indices as player_ids if not available)
        seat_to_player_idx = public_round["seat_to_player_idx"]
        seat_to_player_id = tuple(
            (seat, f"player_{int(seat_to_player_idx[seat])}")
            for seat in range(len(seat_to_player_idx))
        )

        # Current bid
        current_bid = None
        bidder_seat = None
        if obs["public_round"]["bid_exists"]:
            bid_arr = public_round["current_bid"]
            if bid_arr[0] > 0:
                current_bid = Bid(int(bid_arr[0]), int(bid_arr[1]))
                # We don't have bidder_seat in the simplified obs, estimate from context
                # In practice, this might need enhancement
                bidder_seat = int(public_round["current_seat"]) - 1
                if bidder_seat < 0:
                    bidder_seat = len(dice_per_seat) - 1

        # Active seats
        active_mask = public_round["active_mask"]
        active_seats = tuple(i for i, active in enumerate(active_mask) if active)

        # Total dice
        total_dice = int(sum(dice_per_seat))

        return PlayerObservation(
            seat=my_seat,
            player_id=f"player_{my_seat}",
            own_dice=own_dice,
            own_num_dice=len(own_dice),
            other_seats_dice_counts=other_counts,
            seat_to_player_id=seat_to_player_id,
            current_bid=current_bid,
            bidder_seat=bidder_seat,
            current_seat=int(public_round["current_seat"]),
            round_number=int(public_round.get("round_number", 0)),
            num_faces=self.num_faces,
            active_seats=active_seats,
            total_dice=total_dice,
        )

    def _mask_to_actions(self, mask: np.ndarray, seat: int) -> List[Action]:
        """Convert action mask to list of valid Action objects."""
        valid_actions = []
        valid_indices = np.flatnonzero(mask)

        for action_id in valid_indices:
            action_type, count, face = decode_action(
                int(action_id), self.max_dice, self.num_faces
            )
            if action_type == "call":
                valid_actions.append(Action(ActionType.CALL, None, seat))
            else:
                valid_actions.append(Action(ActionType.BID, Bid(count, face), seat))

        return valid_actions

    def _action_to_int(self, action: Action) -> int:
        """Convert Action object to action integer."""
        if action.action_type == ActionType.CALL:
            return self.max_dice * self.num_faces
        else:
            return encode_bid(action.bid.count, action.bid.face_value, self.num_faces)


class GameAgentWrapper(BaseAgent):
    """
    Wraps an EnvPolicy to work with run_game/BluffGame.

    Converts:
    - PlayerObservation → Dict observation (for policy)
    - action int → Action object (for game)

    Usage:
        policy = MyDQNPolicy(...)  # Works with env observations
        agent = GameAgentWrapper(policy, "dqn_agent", num_faces=6)

        # In run_game:
        winner = run_game(game, [agent, other_agent], verbose=True)
    """

    def __init__(
        self,
        policy: EnvPolicy,
        policy_id: str,
        player_id: Optional[str] = None,
        seed: Optional[int] = None,
        num_faces: int = 6,
        max_dice: int = 9,
        num_players: int = 3,
        max_tracked_players: int = 10,
    ):
        """
        Initialize wrapper.

        Args:
            policy: EnvPolicy to wrap
            policy_id: Identifier for this agent
            player_id: Visible session identifier
            seed: Random seed
            num_faces: Number of faces per die
            max_dice: Maximum total dice
            num_players: Number of players in game
            max_tracked_players: Max players for stats arrays
        """
        super().__init__(policy_id, player_id, seed)
        self.policy = policy
        self.num_faces = num_faces
        self.max_dice = max_dice
        self.num_players = num_players
        self.max_tracked_players = max_tracked_players

    def select_action(
        self,
        observation: PlayerObservation,
        valid_actions: List[Action],
    ) -> Action:
        """Convert PlayerObservation to dict, get action int, convert back."""
        # Convert to dict observation
        obs_dict = self._player_observation_to_obs(observation)

        # Create action mask from valid_actions
        action_mask = self._actions_to_mask(valid_actions)

        # Get action from policy
        action_int = self.policy.select_action(obs_dict, action_mask)

        # Convert back to Action
        return self._int_to_action(action_int, observation.seat, valid_actions)

    def _player_observation_to_obs(self, obs: PlayerObservation) -> Dict[str, Any]:
        """Convert PlayerObservation to env dict format."""
        # Private: dice counts
        dice_counts = np.zeros(self.num_faces, dtype=np.int32)
        for die in obs.own_dice:
            dice_counts[die - 1] += 1

        # Public round: dice per seat
        dice_per_seat = np.zeros(self.num_players, dtype=np.int32)
        dice_per_seat[obs.seat] = obs.own_num_dice
        for seat, count in obs.other_seats_dice_counts:
            if seat < self.num_players:
                dice_per_seat[seat] = count

        # Active mask
        active_mask = np.zeros(self.num_players, dtype=np.int8)
        for seat in obs.active_seats:
            if seat < self.num_players:
                active_mask[seat] = 1

        # Current bid
        if obs.current_bid:
            current_bid = np.array(
                [obs.current_bid.count, obs.current_bid.face_value],
                dtype=np.int32
            )
            bid_exists = 1
        else:
            current_bid = np.array([0, 0], dtype=np.int32)
            bid_exists = 0

        # Seat to player idx (simplified: use seat as player idx)
        seat_to_player_idx = np.arange(self.num_players, dtype=np.int32)

        # Public player stats (zeros - not available from PlayerObservation)
        rounds_played = np.zeros(self.max_tracked_players, dtype=np.int32)
        dice_remaining = np.zeros(self.max_tracked_players, dtype=np.int32)
        bluff_rate = np.full(self.max_tracked_players, 0.5, dtype=np.float32)
        call_rate = np.full(self.max_tracked_players, 0.5, dtype=np.float32)
        aggression = np.zeros(self.max_tracked_players, dtype=np.float32)

        # Fill in dice_remaining from what we know
        dice_remaining[obs.seat] = obs.own_num_dice
        for seat, count in obs.other_seats_dice_counts:
            if seat < self.max_tracked_players:
                dice_remaining[seat] = count

        # Action mask (computed separately, but include for completeness)
        action_mask = self._create_action_mask(obs)

        return {
            "private": {
                "dice_counts": dice_counts,
            },
            "public_round": {
                "dice_per_seat": dice_per_seat,
                "active_mask": active_mask,
                "current_bid": current_bid,
                "bid_exists": bid_exists,
                "my_seat": obs.seat,
                "current_seat": obs.current_seat,
                "seat_to_player_idx": seat_to_player_idx,
                "round_number": obs.round_number,
            },
            "public_player": {
                "rounds_played": rounds_played,
                "dice_remaining": dice_remaining,
                "bluff_rate": bluff_rate,
                "call_rate": call_rate,
                "aggression": aggression,
            },
            "action_mask": action_mask,
        }

    def _create_action_mask(self, obs: PlayerObservation) -> np.ndarray:
        """Create action mask from PlayerObservation."""
        from bluff.gym_env.spaces import get_action_mask

        current_bid = (0, 0)
        if obs.current_bid:
            current_bid = (obs.current_bid.count, obs.current_bid.face_value)

        return get_action_mask(
            current_bid=current_bid,
            total_dice=obs.total_dice,
            num_faces=self.num_faces,
            max_dice=self.max_dice,
        )

    def _actions_to_mask(self, valid_actions: List[Action]) -> np.ndarray:
        """Convert list of valid Action objects to mask."""
        mask = np.zeros(self.max_dice * self.num_faces + 1, dtype=np.int8)

        for action in valid_actions:
            if action.action_type == ActionType.CALL:
                mask[-1] = 1
            else:
                action_id = encode_bid(
                    action.bid.count, action.bid.face_value, self.num_faces
                )
                mask[action_id] = 1

        return mask

    def _int_to_action(
        self,
        action_int: int,
        seat: int,
        valid_actions: List[Action],
    ) -> Action:
        """Convert action integer to Action object."""
        action_type, count, face = decode_action(
            action_int, self.max_dice, self.num_faces
        )

        if action_type == "call":
            return Action(ActionType.CALL, None, seat)
        else:
            return Action(ActionType.BID, Bid(count, face), seat)


# Convenience function to wrap any BaseAgent for env use
def wrap_for_env(
    agent: BaseAgent,
    num_faces: int = 6,
    max_dice: int = 9,
) -> EnvPolicyWrapper:
    """
    Wrap a BaseAgent to work with BluffEnv.

    Args:
        agent: BaseAgent instance
        num_faces: Number of faces per die
        max_dice: Maximum total dice

    Returns:
        EnvPolicyWrapper that can be used with BluffEnv
    """
    return EnvPolicyWrapper(agent, num_faces=num_faces, max_dice=max_dice)


# Convenience function to wrap any EnvPolicy for game use
def wrap_for_game(
    policy: EnvPolicy,
    policy_id: str,
    player_id: Optional[str] = None,
    num_faces: int = 6,
    max_dice: int = 9,
    num_players: int = 3,
    max_tracked_players: int = 10,
) -> GameAgentWrapper:
    """
    Wrap an EnvPolicy to work with run_game/BluffGame.

    Args:
        policy: EnvPolicy instance
        policy_id: Identifier for this agent
        player_id: Visible session identifier
        num_faces: Number of faces per die
        max_dice: Maximum total dice
        num_players: Number of players
        max_tracked_players: Max players for stats arrays (must match training)

    Returns:
        GameAgentWrapper that can be used with run_game
    """
    return GameAgentWrapper(
        policy,
        policy_id=policy_id,
        player_id=player_id,
        num_faces=num_faces,
        max_dice=max_dice,
        num_players=num_players,
        max_tracked_players=max_tracked_players,
    )
