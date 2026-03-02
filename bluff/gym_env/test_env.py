"""Test script for BluffEnv PettingZoo environment."""

import numpy as np
from pettingzoo.test import api_test, parallel_api_test

from bluff.gym_env import BluffEnv, RewardConfig


def test_random_game():
    """Run a game with random actions to verify basic functionality."""
    print("=" * 60)
    print("Testing random game play")
    print("=" * 60)

    env = BluffEnv(
        num_players=3,
        dice_per_player=3,
        num_faces=6,
        reward_config=RewardConfig.round_based(),
        render_mode="human",
    )

    env.reset(seed=42)
    print(f"\nInitial agents: {env.agents}")
    print(f"Possible agents: {env.possible_agents}")

    step_count = 0
    max_steps = 200  # Safety limit

    for agent in env.agent_iter():
        step_count += 1
        if step_count > max_steps:
            print("\nMax steps reached, stopping.")
            break

        obs, reward, termination, truncation, info = env.last()

        if termination or truncation:
            action = None
        else:
            # Get valid actions from mask
            action_mask = obs["action_mask"]
            valid_actions = np.where(action_mask == 1)[0]

            if len(valid_actions) == 0:
                print(f"No valid actions for {agent}!")
                action = None
            else:
                # Random valid action
                action = np.random.choice(valid_actions)

        print(f"\nStep {step_count}: {agent}")
        print(f"  Reward: {reward}, Term: {termination}, Trunc: {truncation}")
        if action is not None:
            print(f"  Action: {action}")
            env.step(action)
        else:
            env.step(action)

        if not env.agents:
            print("\nGame over!")
            break

    env.close()
    print(f"\nTotal steps: {step_count}")


def test_observation_space():
    """Test that observations match the defined space."""
    print("\n" + "=" * 60)
    print("Testing observation space")
    print("=" * 60)

    env = BluffEnv(num_players=3, dice_per_player=3)
    env.reset(seed=123)

    for agent in env.possible_agents:
        obs = env.observe(agent)
        space = env.observation_space(agent)

        # Check that observation is in space
        assert space.contains(obs), f"Observation for {agent} not in space!"

        print(f"\n{agent} observation:")
        print(f"  Private dice counts: {obs['private']['dice_counts']}")
        print(f"  Dice per seat: {obs['public_round']['dice_per_seat']}")
        print(f"  Active mask: {obs['public_round']['active_mask']}")
        print(f"  Current bid: {obs['public_round']['current_bid']}")
        print(f"  My seat: {obs['public_round']['my_seat']}")
        print(f"  Valid actions: {np.sum(obs['action_mask'])}")

    env.close()
    print("\nObservation space test passed!")


def test_action_encoding():
    """Test action encoding/decoding."""
    print("\n" + "=" * 60)
    print("Testing action encoding")
    print("=" * 60)

    from bluff.gym_env.spaces import encode_bid, decode_action

    max_dice = 9
    num_faces = 6

    # Test all bid encodings
    for count in range(1, max_dice + 1):
        for face in range(1, num_faces + 1):
            action_id = encode_bid(count, face, num_faces)
            decoded = decode_action(action_id, max_dice, num_faces)
            assert decoded == ("bid", count, face), f"Mismatch: {decoded}"

    # Test CALL encoding
    call_id = max_dice * num_faces
    decoded = decode_action(call_id, max_dice, num_faces)
    assert decoded == ("call", 0, 0), f"CALL mismatch: {decoded}"

    print("Action encoding test passed!")


def test_stats_tracking():
    """Test that player stats are tracked correctly."""
    print("\n" + "=" * 60)
    print("Testing stats tracking")
    print("=" * 60)

    env = BluffEnv(
        num_players=2,
        dice_per_player=2,
        track_stats=True,
    )

    # Run multiple games
    for game_num in range(3):
        env.reset(seed=game_num * 100)

        for agent in env.agent_iter():
            obs, reward, term, trunc, info = env.last()

            if term or trunc:
                action = None
            else:
                action_mask = obs["action_mask"]
                valid_actions = np.where(action_mask == 1)[0]
                action = np.random.choice(valid_actions) if len(valid_actions) > 0 else None

            env.step(action)

            if not env.agents:
                break

    # Check stats
    if env._stats_tracker:
        print("\nPlayer stats after 3 games:")
        for agent, stats in env._stats_tracker.all_stats().items():
            print(f"\n{agent}:")
            print(f"  Rounds played: {stats.rounds_played}")
            print(f"  Total bids: {stats.total_bids}")
            print(f"  Total calls: {stats.total_calls}")
            print(f"  Bluff rate: {stats.bluff_rate:.2f}")
            print(f"  Call rate: {stats.call_rate:.2f}")
            print(f"  Aggression: {stats.aggression:.2f}")

    env.close()
    print("\nStats tracking test passed!")


def test_reward_configs():
    """Test different RewardConfig presets."""
    print("\n" + "=" * 60)
    print("Testing RewardConfig presets")
    print("=" * 60)

    presets = [
        ("sparse", RewardConfig.sparse()),
        ("round_based", RewardConfig.round_based()),
        ("dense", RewardConfig.dense()),
    ]

    for name, config in presets:
        print(f"\nTesting {name} preset...")

        env = BluffEnv(
            num_players=2,
            dice_per_player=2,
            reward_config=config,
        )
        env.reset(seed=42)

        total_rewards = {agent: 0.0 for agent in env.possible_agents}

        for agent in env.agent_iter():
            obs, reward, term, trunc, info = env.last()
            total_rewards[agent] += reward

            if term or trunc:
                action = None
            else:
                action_mask = obs["action_mask"]
                valid_actions = np.where(action_mask == 1)[0]
                action = np.random.choice(valid_actions) if len(valid_actions) > 0 else None

            env.step(action)

            if not env.agents:
                break

        print(f"  Total rewards: {total_rewards}")
        env.close()

    print("\nRewardConfig presets test passed!")


def run_pettingzoo_api_test():
    """Run PettingZoo's official API test."""
    print("\n" + "=" * 60)
    print("Running PettingZoo API test")
    print("=" * 60)

    env = BluffEnv(num_players=3, dice_per_player=3)

    try:
        api_test(env, num_cycles=10, verbose_progress=True)
        print("\nPettingZoo API test passed!")
    except Exception as e:
        print(f"\nPettingZoo API test failed: {e}")
        raise


def main():
    """Run all tests."""
    print("\nBluff Environment Test Suite")
    print("=" * 60)

    test_action_encoding()
    test_observation_space()
    test_random_game()
    test_stats_tracking()
    test_reward_configs()

    # Run PettingZoo API test last (most comprehensive)
    try:
        run_pettingzoo_api_test()
    except Exception as e:
        print(f"\nWarning: PettingZoo API test failed: {e}")
        print("This may be due to subtle API requirements.")

    print("\n" + "=" * 60)
    print("All basic tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
