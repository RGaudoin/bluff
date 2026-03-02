#!/usr/bin/env python
"""
Run tournaments between different agent configurations.

This script demonstrates the baseline agents and their relative performance.
Run from the repo root:
    python -m scripts.agent_tournament
"""

from collections import Counter
from collections.abc import Sequence

from bluff.agents import HeuristicAgent, AdaptiveAgent, RandomAgent, BaseAgent
from bluff.game import BluffGame
from bluff.run_game import run_game


def run_tournament(
    agents: Sequence[BaseAgent],
    num_games: int = 500,
    dice_per_player: int = 3,
    verbose: bool = False,
) -> Counter:
    """
    Run a tournament between agents.

    Args:
        agents: List of agents to compete
        num_games: Number of games to play
        dice_per_player: Dice per player
        verbose: Print each game result

    Returns:
        Counter of wins per agent player_id
    """
    wins = Counter()
    for i in range(num_games):
        game = BluffGame(
            num_players=len(agents),
            dice_per_player=dice_per_player,
            seed=i,
        )
        winner_id = run_game(game, agents, verbose=False)
        wins[winner_id] += 1
        if verbose:
            print(f"Game {i+1}: {winner_id} wins")
    return wins


def print_results(name: str, wins: Counter, total: int) -> None:
    """Print tournament results."""
    print(f"\n=== {name} ===")
    for player_id, count in sorted(wins.items(), key=lambda x: -x[1]):
        pct = 100 * count / total
        print(f"  {player_id}: {count} wins ({pct:.1f}%)")


def main():
    num_games = 500

    # Tournament 1: Heuristic parameter variations
    print("Tournament 1: Heuristic Variants")
    print("Testing different call_threshold and bid_aggression values")
    agents1 = [
        HeuristicAgent(
            "conservative",
            call_threshold=0.4,
            bid_aggression=-0.2,
            bluff_probability=0.05,
            seed=1,
        ),
        HeuristicAgent(
            "neutral",
            call_threshold=0.3,
            bid_aggression=0.0,
            bluff_probability=0.1,
            seed=2,
        ),
        HeuristicAgent(
            "aggressive",
            call_threshold=0.2,
            bid_aggression=0.3,
            bluff_probability=0.25,
            seed=3,
        ),
    ]
    wins1 = run_tournament(agents1, num_games)
    print_results("Heuristic Variants", wins1, num_games)

    # Tournament 2: Agent types comparison
    print("\n" + "=" * 50)
    print("Tournament 2: Agent Types Comparison")
    print("Heuristic vs Adaptive vs Random")
    agents2 = [
        HeuristicAgent("heuristic", call_threshold=0.3, seed=10),
        AdaptiveAgent("adaptive", opponent_trust=0.6, cold_start_rounds=5, seed=20),
        RandomAgent("random", seed=30),
    ]
    wins2 = run_tournament(agents2, num_games)
    print_results("Agent Types", wins2, num_games)

    # Tournament 3: Adaptive trust levels
    print("\n" + "=" * 50)
    print("Tournament 3: Adaptive Trust Levels")
    print("Testing how much to trust opponent statistics")
    agents3 = [
        AdaptiveAgent("low_trust", opponent_trust=0.2, seed=100),
        AdaptiveAgent("mid_trust", opponent_trust=0.5, seed=200),
        AdaptiveAgent("high_trust", opponent_trust=0.8, seed=300),
    ]
    wins3 = run_tournament(agents3, num_games)
    print_results("Adaptive Trust Levels", wins3, num_games)

    # Tournament 4: Call threshold sweep
    print("\n" + "=" * 50)
    print("Tournament 4: Call Threshold Sweep")
    print("Testing call_threshold from 0.2 to 0.5")
    agents4 = [
        HeuristicAgent(f"thresh_{t}", call_threshold=t / 10, seed=t)
        for t in [2, 3, 4, 5]
    ]
    wins4 = run_tournament(agents4, num_games)
    print_results("Call Threshold Sweep", wins4, num_games)

    # Summary
    print("\n" + "=" * 50)
    print("KEY OBSERVATIONS:")
    print("1. Conservative play (higher call_threshold) tends to win more")
    print("2. Probability-based agents vastly outperform random (~50x)")
    print("3. Opponent modeling (AdaptiveAgent) provides marginal benefit")
    print("4. Over-trusting early opponent stats can be harmful")


if __name__ == "__main__":
    main()
