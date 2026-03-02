"""Game state representation for the Bluff game."""

from dataclasses import dataclass
from typing import Optional, Tuple

from bluff.game.types import Bid
from bluff.game.player import Player


@dataclass(frozen=True)
class PlayerObservation:
    """
    What a player can observe about the game state.

    This is the information available to an agent for decision-making.
    Notably excludes other players' actual dice values.

    Attributes:
        seat: This player's table position (0, 1, 2...)
        player_id: This player's visible session identifier
        own_dice: This player's dice values
        own_num_dice: Number of dice this player has
        other_seats_dice_counts: Tuple of (seat, num_dice) for other players
        seat_to_player_id: Tuple of (seat, player_id) for all players
        current_bid: Current bid on table (None at round start)
        bidder_seat: Seat of who made the current bid (None at round start)
        current_seat: Whose turn it is (seat number)
        round_number: Current round number
        num_faces: Faces per die
        active_seats: Seats of players still in the game
        total_dice: Total dice in play
    """
    seat: int
    player_id: str
    own_dice: Tuple[int, ...]
    own_num_dice: int
    other_seats_dice_counts: Tuple[Tuple[int, int], ...]  # (seat, num_dice)
    seat_to_player_id: Tuple[Tuple[int, str], ...]  # (seat, player_id)
    current_bid: Optional[Bid]
    bidder_seat: Optional[int]
    current_seat: int
    round_number: int
    num_faces: int
    active_seats: Tuple[int, ...]
    total_dice: int

    @property
    def is_my_turn(self) -> bool:
        """Check if it's this player's turn."""
        return self.current_seat == self.seat

    @property
    def can_call(self) -> bool:
        """Check if calling is a valid action (requires existing bid)."""
        return self.current_bid is not None

    def get_player_id(self, seat: int) -> Optional[str]:
        """Get the player_id for a given seat."""
        for s, pid in self.seat_to_player_id:
            if s == seat:
                return pid
        return None


@dataclass(frozen=True)
class GameState:
    """
    Immutable snapshot of the complete game state.

    Attributes:
        players: Tuple of Player objects (indexed by seat)
        current_seat: Seat of player whose turn it is
        current_bid: The current bid on the table (None at round start)
        bidder_seat: Seat of player who made the current bid (None at round start)
        round_number: Current round (starts at 0)
        num_faces: Number of faces on each die
        is_game_over: True if game has ended
        winner_seat: Seat of player who won the game (None if not over)
    """
    players: Tuple[Player, ...]
    current_seat: int
    current_bid: Optional[Bid]
    bidder_seat: Optional[int]
    round_number: int
    num_faces: int
    is_game_over: bool = False
    winner_seat: Optional[int] = None

    @property
    def active_seats(self) -> Tuple[int, ...]:
        """Seats of players still in the game (have dice)."""
        return tuple(p.seat for p in self.players if p.is_active)

    @property
    def num_active_players(self) -> int:
        """Number of players still in the game."""
        return len(self.active_seats)

    @property
    def total_dice(self) -> int:
        """Total dice across all active players."""
        return sum(p.num_dice for p in self.players)

    def get_observation(
        self,
        seat: int,
        player_id: str,
        seat_to_player_id: Tuple[Tuple[int, str], ...],
    ) -> PlayerObservation:
        """
        Get the observation for a specific player.

        Players can only see their own dice, not others'.

        Args:
            seat: The player's table position
            player_id: The player's visible session identifier
            seat_to_player_id: Tuple of (seat, player_id) for all players

        Returns:
            PlayerObservation with visible information
        """
        player = self.players[seat]
        other_counts = tuple(
            (p.seat, p.num_dice)
            for p in self.players
            if p.seat != seat
        )

        return PlayerObservation(
            seat=seat,
            player_id=player_id,
            own_dice=player.dice,
            own_num_dice=player.num_dice,
            other_seats_dice_counts=other_counts,
            seat_to_player_id=seat_to_player_id,
            current_bid=self.current_bid,
            bidder_seat=self.bidder_seat,
            current_seat=self.current_seat,
            round_number=self.round_number,
            num_faces=self.num_faces,
            active_seats=self.active_seats,
            total_dice=self.total_dice,
        )

    def __str__(self) -> str:
        lines = [f"Round {self.round_number} | Total dice: {self.total_dice}"]
        for p in self.players:
            lines.append(f"  {p}")
        if self.current_bid:
            lines.append(f"Current bid: {self.current_bid} by Seat {self.bidder_seat}")
        else:
            lines.append("No bid yet")
        lines.append(f"Current seat: {self.current_seat}")
        if self.is_game_over:
            lines.append(f"GAME OVER - Winner: Seat {self.winner_seat}")
        return "\n".join(lines)
