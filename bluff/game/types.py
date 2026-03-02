"""Core data types for the Bluff game."""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional


class ActionType(Enum):
    """Type of action a player can take."""
    BID = auto()
    CALL = auto()


@dataclass(frozen=True)
class Bid:
    """
    A bid claiming at least `count` dice show `face_value` across all players.

    Attributes:
        count: Minimum number of dice claimed (>= 1)
        face_value: Die face being bid on (1 to num_faces)
    """
    count: int
    face_value: int

    def __post_init__(self) -> None:
        if self.count < 1:
            raise ValueError(f"Bid count must be >= 1, got {self.count}")
        if self.face_value < 1:
            raise ValueError(f"Face value must be >= 1, got {self.face_value}")

    def is_higher_than(self, other: Optional["Bid"]) -> bool:
        """
        Check if this bid is strictly higher than another.

        Standard Liar's Dice rules: a bid is higher if:
        - Count increases (any face value allowed), OR
        - Same count with higher face value

        This allows strategic plays like 3x6 -> 4x1.

        Args:
            other: The previous bid (None if first bid of round)

        Returns:
            True if this bid is a valid raise over other
        """
        if other is None:
            return True
        return (
            self.count > other.count or
            (self.count == other.count and self.face_value > other.face_value)
        )

    def __str__(self) -> str:
        return f"{self.count}x{self.face_value}s"


@dataclass(frozen=True)
class Action:
    """
    An action taken by a player.

    Attributes:
        action_type: BID or CALL
        bid: The bid details (required for BID, None for CALL)
        seat: Table position of the player taking the action
    """
    action_type: ActionType
    bid: Optional[Bid]
    seat: int

    def __post_init__(self) -> None:
        if self.action_type == ActionType.BID and self.bid is None:
            raise ValueError("BID action requires a bid")
        if self.action_type == ActionType.CALL and self.bid is not None:
            raise ValueError("CALL action should not have a bid")

    def __str__(self) -> str:
        if self.action_type == ActionType.CALL:
            return f"Seat {self.seat} CALLS"
        return f"Seat {self.seat} bids {self.bid}"


@dataclass(frozen=True)
class RoundResult:
    """
    Result of a round after a call is made.

    Attributes:
        winner_seat: Seat of player who won the round
        loser_seat: Seat of player who lost the round
        bid_was_true: Whether the called bid was actually true
        actual_count: Actual count of the bid face value
        called_bid: The bid that was called
    """
    winner_seat: int
    loser_seat: int
    bid_was_true: bool
    actual_count: int
    called_bid: Bid

    def __str__(self) -> str:
        outcome = "TRUE" if self.bid_was_true else "FALSE"
        return (
            f"Bid {self.called_bid} was {outcome} "
            f"(actual: {self.actual_count}). "
            f"Winner: Seat {self.winner_seat}, Loser: Seat {self.loser_seat}"
        )
