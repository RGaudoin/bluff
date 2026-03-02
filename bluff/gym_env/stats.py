"""Player statistics tracking for opponent modeling."""

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class PlayerStats:
    """
    Aggregated statistics for a player across rounds/games.

    Tracked per player_id (session identifier), not seat, enabling
    opponent modeling across multiple games.

    Attributes:
        player_id: Session-visible identifier for this player
        rounds_played: Number of rounds this player has participated in
        total_bids: Total number of bids made
        called_bids: Times this player's bid was called
        false_bids: Times a called bid was false (bluff detected)
        total_calls: Times this player called another's bid
        aggression_sum: Sum of aggression scores across bids
        aggression_count: Number of bids with computed aggression
    """

    player_id: str
    rounds_played: int = 0
    total_bids: int = 0
    called_bids: int = 0
    false_bids: int = 0
    total_calls: int = 0
    aggression_sum: float = 0.0
    aggression_count: int = 0

    @property
    def bluff_rate(self) -> float:
        """
        Fraction of called bids that were false.

        Returns 0.0 if no bids have been called yet.
        """
        if self.called_bids == 0:
            return 0.0
        return self.false_bids / self.called_bids

    @property
    def call_rate(self) -> float:
        """
        Fraction of actions that were calls (vs bids).

        Returns 0.0 if no actions have been taken.
        """
        total_actions = self.total_calls + self.total_bids
        if total_actions == 0:
            return 0.0
        return self.total_calls / total_actions

    @property
    def aggression(self) -> float:
        """
        Average aggression score across all bids.

        Aggression = (bid_count - expected_count) / expected_count
        Positive means bidding above expected (aggressive/bluffing).
        Negative means bidding below expected (conservative).

        Returns 0.0 if no aggression data yet.
        """
        if self.aggression_count == 0:
            return 0.0
        return self.aggression_sum / self.aggression_count

    def record_bid(self, aggression_score: float) -> None:
        """Record a bid action with its aggression score."""
        self.total_bids += 1
        self.aggression_sum += aggression_score
        self.aggression_count += 1

    def record_call(self) -> None:
        """Record a call action."""
        self.total_calls += 1

    def record_bid_called(self, was_false: bool) -> None:
        """Record that this player's bid was called."""
        self.called_bids += 1
        if was_false:
            self.false_bids += 1

    def record_round_end(self) -> None:
        """Record end of a round."""
        self.rounds_played += 1


class StatsTracker:
    """
    Manages PlayerStats across multiple players.

    Maps player_id to PlayerStats and provides consistent indexing
    for observation space arrays.
    """

    def __init__(self, max_players: int = 32):
        """
        Initialize tracker.

        Args:
            max_players: Maximum number of unique players to track
        """
        self.max_players = max_players
        self._stats: Dict[str, PlayerStats] = {}
        self._player_to_idx: Dict[str, int] = {}
        self._next_idx = 0

    def get_or_create(self, player_id: str) -> PlayerStats:
        """Get stats for a player, creating if needed."""
        if player_id not in self._stats:
            if self._next_idx >= self.max_players:
                raise ValueError(
                    f"Cannot track more than {self.max_players} players"
                )
            self._stats[player_id] = PlayerStats(player_id=player_id)
            self._player_to_idx[player_id] = self._next_idx
            self._next_idx += 1
        return self._stats[player_id]

    def get_player_idx(self, player_id: str) -> int:
        """Get the index for a player_id in observation arrays."""
        if player_id not in self._player_to_idx:
            # Create stats to assign index
            self.get_or_create(player_id)
        return self._player_to_idx[player_id]

    def get_stats(self, player_id: str) -> PlayerStats:
        """Get stats for a player. Raises KeyError if not found."""
        return self._stats[player_id]

    def all_stats(self) -> Dict[str, PlayerStats]:
        """Get all tracked stats."""
        return self._stats.copy()

    def reset_current_game(self) -> None:
        """
        Reset per-game state while preserving cross-game statistics.

        Currently a no-op since all stats are cumulative.
        Could be extended to track per-game vs lifetime stats.
        """
        pass

    @property
    def num_tracked(self) -> int:
        """Number of players currently being tracked."""
        return len(self._stats)
