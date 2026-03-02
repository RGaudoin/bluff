"""Heuristic agent that uses probability calculations to make decisions."""

from typing import List, Optional

import numpy as np
from scipy import stats

from bluff.agents.base import BaseAgent
from bluff.game.types import Action, ActionType, Bid
from bluff.game.game_state import PlayerObservation


class HeuristicAgent(BaseAgent):
    """
    Agent that uses probability calculations to decide when to bid or call.

    Parameters control the agent's style:
    - call_threshold: P(bid true) below which we call (lower = more aggressive caller)
    - bid_aggression: How far above/below expected to bid (-1 to 1, 0 = neutral)
    - bluff_probability: Chance to intentionally overbid (0 to 1)

    This creates a family of agents from conservative (high threshold, negative
    aggression, low bluff) to aggressive (low threshold, positive aggression,
    high bluff).
    """

    def __init__(
        self,
        policy_id: str,
        player_id: Optional[str] = None,
        seed: Optional[int] = None,
        call_threshold: float = 0.3,
        bid_aggression: float = 0.0,
        bluff_probability: float = 0.1,
    ):
        """
        Initialize heuristic agent.

        Args:
            policy_id: Permanent identifier for the policy
            player_id: Visible session identifier (defaults to policy_id)
            seed: Random seed for reproducibility
            call_threshold: P(bid true) below which to call (default 0.3)
            bid_aggression: Bid offset from expected, -1 to 1 (default 0.0)
            bluff_probability: Chance to make a bluff bid (default 0.1)
        """
        super().__init__(policy_id, player_id, seed)
        self.call_threshold = call_threshold
        self.bid_aggression = bid_aggression
        self.bluff_probability = bluff_probability
        self.rng = np.random.default_rng(seed)

    def select_action(
        self,
        observation: PlayerObservation,
        valid_actions: List[Action],
    ) -> Action:
        """
        Select action based on probability calculations.

        Decision logic:
        1. If no bid exists, make an opening bid
        2. If P(current bid is true) < call_threshold, call
        3. Otherwise, raise with a calculated bid
        """
        # Separate valid actions
        call_action = None
        bid_actions = []
        for action in valid_actions:
            if action.action_type == ActionType.CALL:
                call_action = action
            else:
                bid_actions.append(action)

        current_bid = observation.current_bid

        # If no bid exists, make opening bid
        if current_bid is None:
            return self._make_opening_bid(observation, bid_actions)

        # Calculate probability that current bid is true
        p_true = self._probability_bid_true(observation, current_bid)

        # Decision: call or raise
        if p_true < self.call_threshold and call_action is not None:
            return call_action

        # Raise - find best bid
        return self._make_raise_bid(observation, bid_actions, current_bid)

    def _probability_bid_true(
        self,
        observation: PlayerObservation,
        bid: Bid,
    ) -> float:
        """
        Calculate P(at least `bid.count` dice show `bid.face_value`).

        Uses binomial distribution for unknown dice.
        Own dice are known exactly.
        """
        # Count own dice matching the bid face
        own_matching = sum(1 for d in observation.own_dice if d == bid.face_value)

        # Unknown dice = total - own
        unknown_dice = observation.total_dice - observation.own_num_dice

        # Need at least (bid.count - own_matching) from unknown dice
        needed_from_unknown = bid.count - own_matching

        if needed_from_unknown <= 0:
            # We already have enough - bid is definitely true
            return 1.0

        if needed_from_unknown > unknown_dice:
            # Impossible to reach the bid count
            return 0.0

        # P(X >= needed) where X ~ Binomial(unknown_dice, 1/num_faces)
        # Use survival function for better numerical stability than 1 - CDF
        p_each = 1.0 / observation.num_faces
        p_at_least = stats.binom.sf(needed_from_unknown - 1, unknown_dice, p_each)

        return float(p_at_least)

    def _expected_count(
        self,
        observation: PlayerObservation,
        face_value: int,
    ) -> float:
        """
        Calculate expected count of a face value across all dice.

        Expected = own_matching + unknown_dice / num_faces
        """
        own_matching = sum(1 for d in observation.own_dice if d == face_value)
        unknown_dice = observation.total_dice - observation.own_num_dice
        expected_from_unknown = unknown_dice / observation.num_faces
        return own_matching + expected_from_unknown

    def _make_opening_bid(
        self,
        observation: PlayerObservation,
        bid_actions: List[Action],
    ) -> Action:
        """
        Make the opening bid of a round.

        Strategy: Bid on the face we have most of, with count based on
        expected value adjusted by aggression.
        """
        # Find face we have most of
        face_counts = {}
        for face in range(1, observation.num_faces + 1):
            face_counts[face] = sum(1 for d in observation.own_dice if d == face)

        best_face = max(face_counts, key=lambda f: face_counts[f])

        # Calculate target count
        expected = self._expected_count(observation, best_face)

        # Aggression adjusts the bid: positive = bid higher, negative = lower
        # Scale aggression by expected value (so it's relative)
        aggression_offset = self.bid_aggression * max(1, expected * 0.5)
        target_count = int(round(expected + aggression_offset))
        target_count = max(1, target_count)  # At least 1

        # Maybe bluff - bid even higher
        if self.rng.random() < self.bluff_probability:
            target_count += self.rng.integers(1, 3)  # Add 1-2 extra

        # Find closest valid bid
        return self._find_closest_bid(bid_actions, target_count, best_face)

    def _make_raise_bid(
        self,
        observation: PlayerObservation,
        bid_actions: List[Action],
        current_bid: Bid,
    ) -> Action:
        """
        Make a raise over the current bid.

        Strategy: Find the minimum valid raise that we're comfortable with,
        adjusted by aggression.
        """
        if not bid_actions:
            # No valid bids (shouldn't happen if we got here)
            raise ValueError("No valid bid actions available")

        # Score each valid bid by P(true) - prefer high probability bids
        scored_bids = []
        for action in bid_actions:
            bid = action.bid
            p_true = self._probability_bid_true(observation, bid)

            # Aggression affects our comfort level
            # Positive aggression: accept lower probability bids
            comfort_threshold = 0.5 - (self.bid_aggression * 0.3)

            # Maybe bluff
            if self.rng.random() < self.bluff_probability:
                comfort_threshold -= 0.3  # Accept even riskier bids

            # Score combines probability and how much above current bid
            # Prefer smaller raises that are still likely true
            raise_amount = bid.count - current_bid.count
            score = p_true - (raise_amount * 0.05)  # Small penalty for big raises

            scored_bids.append((action, p_true, score))

        # Sort by score (highest first)
        scored_bids.sort(key=lambda x: x[2], reverse=True)

        # Find first bid above comfort threshold, or take best available
        comfort_threshold = 0.5 - (self.bid_aggression * 0.3)
        if self.rng.random() < self.bluff_probability:
            comfort_threshold -= 0.3

        for action, p_true, score in scored_bids:
            if p_true >= comfort_threshold:
                return action

        # All bids below comfort - take the one with highest probability
        best_p_action = max(scored_bids, key=lambda x: x[1])
        return best_p_action[0]

    def _find_closest_bid(
        self,
        bid_actions: List[Action],
        target_count: int,
        target_face: int,
    ) -> Action:
        """Find the valid bid closest to our target."""
        if not bid_actions:
            raise ValueError("No valid bid actions")

        def distance(action: Action) -> float:
            bid = action.bid
            # Prefer matching face, then closest count
            face_penalty = 0 if bid.face_value == target_face else 10
            count_diff = abs(bid.count - target_count)
            return face_penalty + count_diff

        return min(bid_actions, key=distance)

    def __repr__(self) -> str:
        return (
            f"HeuristicAgent(policy_id={self.policy_id!r}, "
            f"call_threshold={self.call_threshold}, "
            f"bid_aggression={self.bid_aggression}, "
            f"bluff_probability={self.bluff_probability})"
        )
