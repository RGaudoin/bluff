"""Player state management for the Bluff game."""

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class Player:
    """
    Represents a player's state in the game.

    Attributes:
        seat: Table position (0, 1, 2, ...)
        num_dice: Current number of dice the player holds
        dice: Current dice values (tuple of ints, empty until rolled)
    """
    seat: int
    num_dice: int
    dice: Tuple[int, ...] = field(default_factory=tuple)

    @property
    def is_active(self) -> bool:
        """Player is active if they have at least one die."""
        return self.num_dice > 0

    def roll(self, rng: np.random.Generator, num_faces: int = 6) -> "Player":
        """
        Roll dice for this player.

        Args:
            rng: Random number generator
            num_faces: Number of faces on each die

        Returns:
            New Player instance with rolled dice
        """
        if self.num_dice == 0:
            return Player(self.seat, 0, tuple())
        dice = tuple(int(x) for x in rng.integers(1, num_faces + 1, size=self.num_dice))
        return Player(self.seat, self.num_dice, dice)

    def lose_die(self) -> "Player":
        """
        Remove one die from this player.

        Returns:
            New Player instance with one fewer die (dice cleared)
        """
        new_num_dice = max(0, self.num_dice - 1)
        return Player(self.seat, new_num_dice, tuple())

    def count_face(self, face_value: int) -> int:
        """Count how many dice show the given face value."""
        return sum(1 for d in self.dice if d == face_value)

    def __str__(self) -> str:
        if not self.is_active:
            return f"Seat {self.seat}: OUT"
        dice_str = ",".join(str(d) for d in self.dice) if self.dice else "not rolled"
        return f"Seat {self.seat}: {self.num_dice} dice [{dice_str}]"
