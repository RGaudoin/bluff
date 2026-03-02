"""Main game engine for Bluff (Liar's Dice)."""

from typing import List, Optional, Tuple

import numpy as np

from bluff.game.types import Action, ActionType, Bid, RoundResult
from bluff.game.player import Player
from bluff.game.game_state import GameState


class BluffGame:
    """
    Game engine for Bluff (Liar's Dice).

    Manages game flow, validates actions, and updates state.
    Uses immutable state pattern for Gymnasium compatibility.
    """

    def __init__(
        self,
        num_players: int = 2,
        dice_per_player: int = 5,
        num_faces: int = 6,
        seed: Optional[int] = None,
    ):
        """
        Initialize game configuration.

        Args:
            num_players: Number of players (>= 2)
            dice_per_player: Starting dice per player (>= 1)
            num_faces: Faces on each die (default 6)
            seed: Random seed for reproducibility
        """
        if num_players < 2:
            raise ValueError(f"Need at least 2 players, got {num_players}")
        if dice_per_player < 1:
            raise ValueError(f"Need at least 1 die per player, got {dice_per_player}")
        if num_faces < 2:
            raise ValueError(f"Need at least 2 faces per die, got {num_faces}")

        self.num_players = num_players
        self.dice_per_player = dice_per_player
        self.num_faces = num_faces
        self.rng = np.random.default_rng(seed)

    def reset(self, starting_seat: int = 0) -> GameState:
        """
        Reset game to initial state and start first round.

        Args:
            starting_seat: Seat of player who goes first

        Returns:
            Initial GameState with dice rolled
        """
        players = tuple(
            Player(seat, self.dice_per_player)
            for seat in range(self.num_players)
        )

        # Roll dice for all players
        players = tuple(p.roll(self.rng, self.num_faces) for p in players)

        return GameState(
            players=players,
            current_seat=starting_seat,
            current_bid=None,
            bidder_seat=None,
            round_number=0,
            num_faces=self.num_faces,
            is_game_over=False,
            winner_seat=None,
        )

    def get_valid_actions(self, state: GameState) -> List[Action]:
        """
        Get all valid actions for the current player.

        Args:
            state: Current game state

        Returns:
            List of valid Action objects
        """
        if state.is_game_over:
            return []

        seat = state.current_seat
        valid_actions = []

        # Can call if there's an existing bid
        if state.current_bid is not None:
            valid_actions.append(Action(ActionType.CALL, None, seat))

        # Generate all valid bids
        for count in range(1, state.total_dice + 1):
            for face in range(1, self.num_faces + 1):
                bid = Bid(count, face)
                if bid.is_higher_than(state.current_bid):
                    valid_actions.append(Action(ActionType.BID, bid, seat))

        return valid_actions

    def is_valid_action(self, state: GameState, action: Action) -> bool:
        """
        Check if an action is valid in the current state.

        Args:
            state: Current game state
            action: Proposed action

        Returns:
            True if action is valid
        """
        if state.is_game_over:
            return False
        if action.seat != state.current_seat:
            return False
        if action.action_type == ActionType.CALL:
            return state.current_bid is not None
        # BID action
        if action.bid is None:
            return False
        if action.bid.face_value > self.num_faces:
            return False
        if action.bid.count > state.total_dice:
            return False
        return action.bid.is_higher_than(state.current_bid)

    def step(
        self, state: GameState, action: Action
    ) -> Tuple[GameState, Optional[RoundResult]]:
        """
        Apply an action and return the new state.

        Args:
            state: Current game state
            action: Action to apply (must be valid)

        Returns:
            Tuple of (new_state, round_result)
            round_result is None if round continues, otherwise contains outcome

        Raises:
            ValueError: If action is invalid
        """
        if not self.is_valid_action(state, action):
            raise ValueError(f"Invalid action: {action}")

        if action.action_type == ActionType.BID:
            return self._handle_bid(state, action), None
        else:
            return self._handle_call(state, action)

    def _handle_bid(self, state: GameState, action: Action) -> GameState:
        """Process a bid action."""
        next_seat = self._next_active_seat(state, state.current_seat)

        return GameState(
            players=state.players,
            current_seat=next_seat,
            current_bid=action.bid,
            bidder_seat=action.seat,
            round_number=state.round_number,
            num_faces=state.num_faces,
            is_game_over=False,
            winner_seat=None,
        )

    def _handle_call(
        self, state: GameState, action: Action
    ) -> Tuple[GameState, RoundResult]:
        """Process a call action and resolve the round."""
        caller_seat = action.seat
        bidder_seat = state.bidder_seat
        bid = state.current_bid

        # Count actual dice matching the bid
        actual_count = sum(
            p.count_face(bid.face_value)
            for p in state.players
            if p.is_active
        )

        bid_was_true = actual_count >= bid.count

        if bid_was_true:
            winner_seat = bidder_seat
            loser_seat = caller_seat
        else:
            winner_seat = caller_seat
            loser_seat = bidder_seat

        result = RoundResult(
            winner_seat=winner_seat,
            loser_seat=loser_seat,
            bid_was_true=bid_was_true,
            actual_count=actual_count,
            called_bid=bid,
        )

        # Loser loses a die
        new_players = list(state.players)
        new_players[loser_seat] = new_players[loser_seat].lose_die()

        # Check for game over
        active_players = [p for p in new_players if p.is_active]
        if len(active_players) == 1:
            return GameState(
                players=tuple(new_players),
                current_seat=active_players[0].seat,
                current_bid=None,
                bidder_seat=None,
                round_number=state.round_number + 1,
                num_faces=state.num_faces,
                is_game_over=True,
                winner_seat=active_players[0].seat,
            ), result

        # Start new round - winner starts, re-roll all dice
        new_players = tuple(
            p.roll(self.rng, self.num_faces) for p in new_players
        )

        return GameState(
            players=new_players,
            current_seat=winner_seat,
            current_bid=None,
            bidder_seat=None,
            round_number=state.round_number + 1,
            num_faces=state.num_faces,
            is_game_over=False,
            winner_seat=None,
        ), result

    def _next_active_seat(self, state: GameState, current_seat: int) -> int:
        """Find the next active seat in turn order."""
        n = len(state.players)
        for i in range(1, n + 1):
            next_seat = (current_seat + i) % n
            if state.players[next_seat].is_active:
                return next_seat
        raise RuntimeError("No active players found")
