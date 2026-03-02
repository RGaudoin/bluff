#!/usr/bin/env python
"""
Test script for agent wrappers.

Tests that:
1. BaseAgents can be wrapped to work with BluffEnv
2. EnvPolicies can be wrapped to work with BluffGame

Run from the repo root:
    python -m scripts.test_wrappers
"""

import numpy as np
from collections import Counter

from bluff.agents import (
    HeuristicAgent,
    RandomAgent,
    wrap_for_env,
    wrap_for_game,
    EnvPolicy,
)
from bluff.game import BluffGame
from bluff.run_game import run_game
from bluff.gym_env.bluff_env import BluffEnv


class SimpleEnvPolicy(EnvPolicy):
    """
    Simple env-based policy that picks a random valid action.

    This simulates what an RL policy would do - receive dict observations
    and return action integers.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def select_action(self, obs: dict, action_mask: np.ndarray) -> int:
        """Select random valid action from mask."""
        valid_indices = np.flatnonzero(action_mask)
        return int(self.rng.choice(valid_indices))


def test_env_wrapper():
    """Test wrapping BaseAgent to work with BluffEnv."""
    print("\n=== Test 1: BaseAgent wrapped for BluffEnv ===")

    # Game parameters
    num_players = 2
    dice_per_player = 3
    num_faces = 6
    max_dice = num_players * dice_per_player  # Must match env!

    # Create agents
    agent1 = HeuristicAgent("heuristic", seed=1)
    agent2 = RandomAgent("random", seed=2)

    # Wrap them for env use (max_dice must match env's max_dice)
    policy1 = wrap_for_env(agent1, num_faces=num_faces, max_dice=max_dice)
    policy2 = wrap_for_env(agent2, num_faces=num_faces, max_dice=max_dice)
    policies = {"player_0": policy1, "player_1": policy2}

    # Create env
    env = BluffEnv(num_players=num_players, dice_per_player=dice_per_player)

    # Run a few episodes
    wins = Counter()
    for episode in range(10):
        env.reset(seed=42 + episode)

        for agent_name in env.agent_iter():
            obs, reward, term, trunc, info = env.last()

            if term or trunc:
                action = None
            else:
                policy = policies[agent_name]
                action_mask = obs["action_mask"]
                action = policy.select_action(obs, action_mask)

            env.step(action)

        # Check who won (has positive cumulative reward)
        for agent_name in ["player_0", "player_1"]:
            if env._cumulative_rewards.get(agent_name, 0) > 0:
                wins[agent_name] += 1

    print(f"  Ran 10 episodes in BluffEnv")
    print(f"  Results: {dict(wins)}")
    print("  SUCCESS: BaseAgents work in BluffEnv via wrapper")
    return True


def test_game_wrapper():
    """Test wrapping EnvPolicy to work with BluffGame."""
    print("\n=== Test 2: EnvPolicy wrapped for BluffGame ===")

    # Create env-based policies
    env_policy1 = SimpleEnvPolicy(seed=1)
    env_policy2 = SimpleEnvPolicy(seed=2)

    # Wrap them for game use
    agent1 = wrap_for_game(env_policy1, "env_random_1", num_players=2)
    agent2 = wrap_for_game(env_policy2, "env_random_2", num_players=2)

    # Run games
    wins = Counter()
    for i in range(10):
        game = BluffGame(num_players=2, dice_per_player=3, seed=i)
        winner_id = run_game(game, [agent1, agent2], verbose=False)
        wins[winner_id] += 1

    print(f"  Ran 10 games in BluffGame")
    print(f"  Results: {dict(wins)}")
    print("  SUCCESS: EnvPolicies work in BluffGame via wrapper")
    return True


def test_mixed_tournament():
    """Test tournament with both native and wrapped agents."""
    print("\n=== Test 3: Mixed Tournament ===")
    print("Native HeuristicAgent vs Wrapped EnvPolicy (random)")

    # Native game agent
    native_agent = HeuristicAgent("native_heuristic", seed=1)

    # Env-based policy wrapped for game
    env_policy = SimpleEnvPolicy(seed=2)
    wrapped_agent = wrap_for_game(env_policy, "wrapped_random", num_players=2)

    # Run tournament
    wins = Counter()
    num_games = 100
    for i in range(num_games):
        game = BluffGame(num_players=2, dice_per_player=3, seed=i)
        winner_id = run_game(game, [native_agent, wrapped_agent], verbose=False)
        wins[winner_id] += 1

    print(f"  Ran {num_games} games")
    for player_id, count in sorted(wins.items(), key=lambda x: -x[1]):
        pct = 100 * count / num_games
        print(f"    {player_id}: {count} wins ({pct:.1f}%)")

    # Heuristic should significantly outperform random
    if wins["native_heuristic"] > wins["wrapped_random"]:
        print("  SUCCESS: Native heuristic beats wrapped random as expected")
        return True
    else:
        print("  WARNING: Unexpected result - random beat heuristic")
        return False


def test_roundtrip():
    """Test wrapping an agent, running in env, then wrapping result for game."""
    print("\n=== Test 4: Roundtrip Wrapper Test ===")

    # Start with a native agent
    original_agent = HeuristicAgent("original", call_threshold=0.3, seed=1)

    # Wrap for env
    env_policy = wrap_for_env(original_agent, num_faces=6, max_dice=9)

    # Wrap back for game
    roundtrip_agent = wrap_for_game(env_policy, "roundtrip", num_players=2)

    # Compare against fresh native agent
    fresh_agent = HeuristicAgent("fresh", call_threshold=0.3, seed=2)

    # Run tournament - both should perform similarly (same algorithm)
    wins = Counter()
    num_games = 100
    for i in range(num_games):
        game = BluffGame(num_players=2, dice_per_player=3, seed=i)
        winner_id = run_game(game, [roundtrip_agent, fresh_agent], verbose=False)
        wins[winner_id] += 1

    print(f"  Ran {num_games} games: roundtrip vs fresh heuristic")
    for player_id, count in sorted(wins.items(), key=lambda x: -x[1]):
        pct = 100 * count / num_games
        print(f"    {player_id}: {count} wins ({pct:.1f}%)")

    # Should be roughly 50/50 (same algorithm, different seeds)
    ratio = min(wins.values()) / max(wins.values()) if max(wins.values()) > 0 else 0
    if ratio > 0.3:  # At least 30/70 split
        print("  SUCCESS: Roundtrip wrapper preserves agent behavior")
        return True
    else:
        print("  WARNING: Large disparity suggests wrapper issues")
        return False


def main():
    print("=" * 60)
    print("Testing Agent Wrappers")
    print("=" * 60)

    results = []
    results.append(("EnvWrapper", test_env_wrapper()))
    results.append(("GameWrapper", test_game_wrapper()))
    results.append(("MixedTournament", test_mixed_tournament()))
    results.append(("Roundtrip", test_roundtrip()))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        all_passed = all_passed and passed

    if all_passed:
        print("\nAll tests passed!")
    else:
        print("\nSome tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
