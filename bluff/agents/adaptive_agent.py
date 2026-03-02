"""Adaptive agent that uses opponent statistics for decision making."""

from typing import Dict, List, Optional, Tuple

import numpy as np

from bluff.agents.heuristic_agent import HeuristicAgent
from bluff.game.types import Action, ActionType, RoundResult
from bluff.game.game_state import PlayerObservation


class AdaptiveAgent(HeuristicAgent):
    """
    Agent that adapts its strategy based on opponent statistics.

    Extends HeuristicAgent by:
    - Adjusting call threshold based on opponent's bluff_rate
    - Tracking opponent behavior via on_round_end callbacks
    - Using cold_start_rounds to avoid trusting unreliable early stats

    Parameters (in addition to HeuristicAgent params):
    - opponent_trust: Weight given to opponent stats (0=ignore, 1=fully trust)
    - cold_start_rounds: Min rounds before trusting a player's stats
    """

    def __init__(
        self,
        policy_id: str,
        player_id: Optional[str] = None,
        seed: Optional[int] = None,
        call_threshold: float = 0.3,
        bid_aggression: float = 0.0,
        bluff_probability: float = 0.1,
        opponent_trust: float = 0.5,
        cold_start_rounds: int = 10,
    ):
        """
        Initialize adaptive agent.

        Args:
            policy_id: Permanent identifier for the policy
            player_id: Visible session identifier (defaults to policy_id)
            seed: Random seed for reproducibility
            call_threshold: Base P(bid true) threshold for calling (default 0.3)
            bid_aggression: Bid offset from expected, -1 to 1 (default 0.0)
            bluff_probability: Chance to make a bluff bid (default 0.1)
            opponent_trust: How much to adjust based on opponent stats (default 0.5)
            cold_start_rounds: Rounds before trusting opponent stats (default 10)
        """
        super().__init__(
            policy_id=policy_id,
            player_id=player_id,
            seed=seed,
            call_threshold=call_threshold,
            bid_aggression=bid_aggression,
            bluff_probability=bluff_probability,
        )
        self.opponent_trust = opponent_trust
        self.cold_start_rounds = cold_start_rounds

        # Track opponent statistics ourselves (supplements env stats)
        # Maps player_id -> stats dict
        self._opponent_stats: Dict[str, Dict] = {}

    def select_action(
        self,
        observation: PlayerObservation,
        valid_actions: List[Action],
    ) -> Action:
        """
        Select action, adjusting for opponent behavior.

        Key adaptation: If the current bidder has high bluff_rate,
        we're more likely to call their bid.
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

        # If no bid exists, make opening bid (use parent logic)
        if current_bid is None:
            return self._make_opening_bid(observation, bid_actions)

        # Get adjusted call threshold based on bidder's bluff rate
        adjusted_threshold = self._get_adjusted_call_threshold(observation)

        # Calculate probability that current bid is true
        p_true = self._probability_bid_true(observation, current_bid)

        # Decision: call or raise
        if p_true < adjusted_threshold and call_action is not None:
            return call_action

        # Raise - find best bid
        return self._make_raise_bid(observation, bid_actions, current_bid)

    def _get_adjusted_call_threshold(self, observation: PlayerObservation) -> float:
        """
        Adjust call threshold based on current bidder's bluff rate.

        If opponent bluffs often, raise our call threshold (call more).
        If opponent is honest, lower threshold (call less).
        """
        if observation.bidder_seat is None:
            return self.call_threshold

        # Get bidder's player_id
        bidder_player_id = observation.get_player_id(observation.bidder_seat)
        if bidder_player_id is None:
            return self.call_threshold

        # Get their bluff rate from our tracked stats
        opponent_stats = self._opponent_stats.get(bidder_player_id, {})
        rounds_played = opponent_stats.get("rounds_played", 0)
        bluff_rate = opponent_stats.get("bluff_rate", 0.5)  # Default: 50%

        # Cold start check - don't trust stats with few observations
        if rounds_played < self.cold_start_rounds:
            # Interpolate from default (0.5) based on how many rounds we've seen
            confidence = rounds_played / self.cold_start_rounds
            bluff_rate = 0.5 * (1 - confidence) + bluff_rate * confidence

        # Adjust threshold:
        # - High bluff_rate (e.g., 0.7) -> increase threshold (call more often)
        # - Low bluff_rate (e.g., 0.2) -> decrease threshold (trust them more)
        # Neutral point is 0.5 bluff_rate (no adjustment)
        bluff_adjustment = (bluff_rate - 0.5) * self.opponent_trust

        adjusted = self.call_threshold + bluff_adjustment

        # Clamp to reasonable range
        return max(0.1, min(0.9, adjusted))

    def on_round_end(
        self,
        revealed_dice: Dict[int, Tuple[int, ...]],
        result: RoundResult,
        seat_to_player_id: Dict[int, str],
    ) -> None:
        """
        Update opponent statistics when a round ends.

        Track each player's bluff rate based on whether their bids
        were true when called.
        """
        # Get the bidder's player_id (the one whose bid was called)
        # result contains winner_seat and loser_seat
        # If bid_was_true: bidder won, caller lost
        # If bid_was_false: bidder lost, caller won

        # We need to identify the bidder - they're the loser if bid was false,
        # winner if bid was true
        if result.bid_was_true:
            bidder_seat = result.winner_seat
        else:
            bidder_seat = result.loser_seat

        bidder_player_id = seat_to_player_id.get(bidder_seat)
        if bidder_player_id is None:
            return

        # Update their stats
        if bidder_player_id not in self._opponent_stats:
            self._opponent_stats[bidder_player_id] = {
                "rounds_played": 0,
                "bids_called": 0,
                "bids_were_bluffs": 0,
                "bluff_rate": 0.5,
            }

        stats = self._opponent_stats[bidder_player_id]
        stats["rounds_played"] += 1
        stats["bids_called"] += 1

        if not result.bid_was_true:
            stats["bids_were_bluffs"] += 1

        # Update bluff rate (exponential moving average for smoothness)
        if stats["bids_called"] > 0:
            raw_rate = stats["bids_were_bluffs"] / stats["bids_called"]
            # Smooth with exponential moving average
            alpha = 0.3  # Learning rate for EMA
            stats["bluff_rate"] = (1 - alpha) * stats["bluff_rate"] + alpha * raw_rate

    def on_game_start(self, seat_to_player_id: Dict[int, str]) -> None:
        """
        Called when a new game starts.

        Initialize stats for any new opponents.
        """
        for seat, player_id in seat_to_player_id.items():
            if player_id not in self._opponent_stats:
                self._opponent_stats[player_id] = {
                    "rounds_played": 0,
                    "bids_called": 0,
                    "bids_were_bluffs": 0,
                    "bluff_rate": 0.5,  # Prior: assume 50% bluff rate
                }

    def __repr__(self) -> str:
        return (
            f"AdaptiveAgent(policy_id={self.policy_id!r}, "
            f"call_threshold={self.call_threshold}, "
            f"bid_aggression={self.bid_aggression}, "
            f"bluff_probability={self.bluff_probability}, "
            f"opponent_trust={self.opponent_trust}, "
            f"cold_start_rounds={self.cold_start_rounds})"
        )
