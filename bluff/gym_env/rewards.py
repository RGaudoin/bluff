"""Composable reward configuration for Bluff environment."""

from dataclasses import dataclass


@dataclass
class RewardConfig:
    """
    Configuration for composable reward structure.

    Rewards are organized in three layers:
    - Layer 1 (Game): Sparse rewards at game end
    - Layer 2 (Round): Semi-sparse rewards when rounds end
    - Layer 3 (Step): Dense rewards for specific actions/outcomes

    Note on reward magnitudes:
        With gamma=0.99 over ~30 steps, game-end rewards discount to ~0.74.
        Step rewards should be small (<10% of discounted game reward) to avoid
        agents optimizing for immediate rewards over winning.

    Example conservative config:
        game_win=1.0, round_win=0.1, successful_call=0.05
    """

    # === Layer 1: Game-level (sparse) ===
    game_win: float = 1.0
    """Reward for winning the entire game."""

    elimination: float = -1.0
    """Penalty for being eliminated from the game."""

    # === Layer 2: Round-level (semi-sparse) ===
    round_win: float = 0.0
    """Reward for winning a round (opponent loses a die)."""

    round_loss: float = 0.0
    """Penalty for losing a round (you lose a die)."""

    survive_round: float = 0.0
    """Reward for surviving when another player loses (N>2 games)."""

    dice_fraction_scale: float = 0.0
    """Scale factor for delta in dice fraction at round end.

    Reward = scale * (own_dice/total_dice_after - own_dice/total_dice_before)

    This provides non-linear weighting by game phase: late-game rounds
    have larger deltas (each die is worth more as total shrinks).
    Suggested values: 0.5-2.0 to experiment.
    """

    # === Layer 3: Step-level (dense) ===
    # Call outcomes
    successful_call: float = 0.0
    """Reward for correctly calling a bluff (opponent was lying)."""

    failed_call: float = 0.0
    """Penalty for incorrectly calling (opponent was truthful)."""

    got_called_truthful: float = 0.0
    """Reward when your bid was called but it was true."""

    got_called_bluffing: float = 0.0
    """Penalty when your bid was called and it was false (caught bluffing)."""

    # Invalid actions
    invalid_action: float = -1.0
    """Penalty for submitting an invalid action."""

    @classmethod
    def sparse(cls) -> "RewardConfig":
        """Create config with only game-end rewards (default)."""
        return cls()

    @classmethod
    def round_based(cls) -> "RewardConfig":
        """Create config with game + round rewards."""
        return cls(round_win=0.1, round_loss=-0.1)

    @classmethod
    def dense(cls) -> "RewardConfig":
        """Create config with all reward layers active (conservative magnitudes)."""
        return cls(
            round_win=0.1,
            round_loss=-0.1,
            survive_round=0.02,
            successful_call=0.05,
            failed_call=-0.05,
            got_called_truthful=0.02,
            got_called_bluffing=-0.02,
        )
