"""Simple game runner to test the Bluff simulation."""

from collections.abc import Sequence
from typing import Dict, Tuple

from bluff.game.game import BluffGame
from bluff.agents.base import BaseAgent
from bluff.agents.random_agent import RandomAgent


def run_game(
    game: BluffGame,
    agents: Sequence[BaseAgent],
    verbose: bool = False,
) -> str:
    """
    Run a complete game with the given agents.

    Args:
        game: Configured BluffGame instance
        agents: List of agents (will be assigned seats 0, 1, 2, ...)
        verbose: Print game progress

    Returns:
        Winner's player_id (visible session identifier)
    """
    if len(agents) != game.num_players:
        raise ValueError(
            f"Expected {game.num_players} agents, got {len(agents)}"
        )

    # Assign seats to agents
    for seat, agent in enumerate(agents):
        agent.set_seat(seat)

    # Create seat -> player_id mapping
    seat_to_player_id: Tuple[Tuple[int, str], ...] = tuple(
        (agent.seat, agent.player_id) for agent in agents
    )
    seat_to_player_id_dict: Dict[int, str] = dict(seat_to_player_id)

    # Create lookup by seat
    agents_by_seat: Dict[int, BaseAgent] = {
        agent.seat: agent for agent in agents
    }

    state = game.reset()

    # Notify agents of game start
    for agent in agents:
        agent.on_game_start(seat_to_player_id_dict)

    if verbose:
        print("=== GAME START ===")
        print(f"Seat assignments: {seat_to_player_id_dict}")
        print(state)
        print()

    while not state.is_game_over:
        current_agent = agents_by_seat[state.current_seat]
        observation = state.get_observation(
            seat=state.current_seat,
            player_id=current_agent.player_id,
            seat_to_player_id=seat_to_player_id,
        )
        valid_actions = game.get_valid_actions(state)

        action = current_agent.select_action(observation, valid_actions)

        if verbose:
            print(f"  [{current_agent.player_id}] {action}")

        state, result = game.step(state, action)

        if result is not None:
            # Round ended - reveal dice to all agents
            revealed = {p.seat: p.dice for p in state.players}
            for agent in agents:
                agent.on_round_end(revealed, result, seat_to_player_id_dict)

            if verbose:
                print(f"\n  >>> {result}")
                print(f"\n{state}\n")

    # Game ended
    winner_player_id = seat_to_player_id_dict[state.winner_seat]
    for agent in agents:
        agent.on_game_end(state.winner_seat, seat_to_player_id_dict)

    if verbose:
        print(f"=== GAME OVER: {winner_player_id} (seat {state.winner_seat}) wins! ===")

    return winner_player_id


def main():
    """Run a sample game."""
    # Create game with 3 players, 3 dice each
    game = BluffGame(num_players=3, dice_per_player=3, seed=42)

    # Create random agents
    # policy_id is the model identifier, player_id defaults to policy_id (visible in session)
    agents = [
        RandomAgent("alice", seed=100),
        RandomAgent("bob", seed=101),
        RandomAgent("charlie", seed=102),
    ]

    # Run game with verbose output
    winner = run_game(game, agents, verbose=True)

    # Run multiple games to check statistics
    print("\n" + "=" * 50)
    print("Running 1000 games to check win distribution...")
    print("(Same agents, randomly assigned seats each game)")

    # Create persistent agents
    persistent_agents = [
        RandomAgent("alice", seed=100),
        RandomAgent("bob", seed=101),
        RandomAgent("charlie", seed=102),
    ]

    wins = {agent.player_id: 0 for agent in persistent_agents}

    import numpy as np
    rng = np.random.default_rng(42)

    for i in range(1000):
        game_i = BluffGame(num_players=3, dice_per_player=3, seed=i)
        # Shuffle seat positions each game
        shuffled = persistent_agents.copy()
        rng.shuffle(shuffled)
        winner = run_game(game_i, shuffled, verbose=False)
        wins[winner] += 1

    print(f"Win counts: {wins}")
    rates = {k: f"{v/10:.1f}%" for k, v in wins.items()}
    print(f"Win rates: {rates}")


if __name__ == "__main__":
    main()
