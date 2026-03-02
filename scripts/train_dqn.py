#!/usr/bin/env python
"""
Train a DQN agent for Bluff.

Usage:
    python -m scripts.train_dqn

This script trains a DQN agent against heuristic opponents,
then evaluates its performance.
"""

from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from bluff.agents import HeuristicAgent, wrap_for_env
from bluff.gym_env import BluffEnv, RewardConfig
from bluff.rl import (
    DQNPolicy,
    ReplayBuffer,
    Transition,
    flatten_obs,
    get_flat_obs_dim,
)


def collect_episode(
    env: BluffEnv,
    dqn_policy: DQNPolicy,
    opponent_policy,
    dqn_buffer: ReplayBuffer,
    opponent_buffer: ReplayBuffer = None,
    dqn_seats: list = None,
    seed: int = None,
) -> dict:
    """
    Collect one episode of experience.

    DQN plays at dqn_seats, opponent plays at other seats.
    All DQN seats share the same policy (self-play style).

    Args:
        env: BluffEnv instance
        dqn_policy: DQN policy being trained
        opponent_policy: Wrapped opponent policy
        dqn_buffer: Replay buffer for DQN agent transitions
        opponent_buffer: Optional replay buffer for opponent transitions (off-policy learning)
        dqn_seats: List of seats where DQN plays (e.g., [0, 2] for seats 0 and 2)
        seed: Random seed for episode

    Returns:
        Dict with episode stats (winner_seat, num_steps, dqn_seats, dqn_won, etc.)
    """
    if dqn_seats is None:
        dqn_seats = [0]
    dqn_seats_set = set(dqn_seats)
    num_players = env.num_players

    env.reset(seed=seed)

    # Track previous state/action for all agents (to support off-policy learning from opponents)
    prev_state = {seat: {
        "obs": None,
        "action": None,
        "flat_obs": None,
        "action_mask": None,
    } for seat in range(num_players)}

    num_steps = 0

    for agent in env.agent_iter():
        obs, reward, term, trunc, info = env.last()
        seat = int(agent.split("_")[1])

        # Store transition for this agent (if we have a previous state)
        if prev_state[seat]["obs"] is not None:
            flat_obs = flatten_obs(
                obs,
                dqn_policy.num_players,
                dqn_policy.num_faces,
                dqn_policy.max_dice,
                dqn_policy.max_tracked_players,
            )
            transition = Transition(
                obs=prev_state[seat]["flat_obs"],
                action=prev_state[seat]["action"],
                reward=reward,
                next_obs=flat_obs,
                action_mask=obs["action_mask"].copy(),
                terminated=term,
                truncated=trunc,
            )
            # Route to appropriate buffer
            if seat in dqn_seats_set:
                dqn_buffer.push(transition)
            elif opponent_buffer is not None:
                opponent_buffer.push(transition)

        if term or trunc:
            action = None
        else:
            if seat in dqn_seats_set:
                # DQN agent
                action = dqn_policy.select_action(obs, obs["action_mask"])
            else:
                # Opponent
                action = opponent_policy.select_action(obs, obs["action_mask"])

            # Track state/action for all agents (for off-policy learning)
            prev_state[seat]["obs"] = obs
            prev_state[seat]["action"] = action
            prev_state[seat]["flat_obs"] = flatten_obs(
                obs,
                dqn_policy.num_players,
                dqn_policy.num_faces,
                dqn_policy.max_dice,
                dqn_policy.max_tracked_players,
            )
            prev_state[seat]["action_mask"] = obs["action_mask"].copy()

        env.step(action)
        num_steps += 1

    # Determine winner seat
    winner_seat = None
    for agent_name in env.possible_agents:
        if env._cumulative_rewards.get(agent_name, 0) > 0:
            winner_seat = int(agent_name.split("_")[1])
            break

    # Store final terminal transitions for all agents.
    # NOTE: This is necessary because PettingZoo's agent_iter() exits when the game
    # ends rather than cycling back to terminated agents with term=True. The main
    # loop above captures mid-game transitions (reward from env.last()), but terminal
    # rewards are only available in env._cumulative_rewards after the iterator exits.
    for seat in range(num_players):
        if prev_state[seat]["obs"] is not None:
            agent_name = f"player_{seat}"
            final_reward = env._cumulative_rewards.get(agent_name, 0.0)
            transition = Transition(
                obs=prev_state[seat]["flat_obs"],
                action=prev_state[seat]["action"],
                reward=final_reward,
                next_obs=prev_state[seat]["flat_obs"],  # Dummy, doesn't matter for terminal
                action_mask=prev_state[seat]["action_mask"],  # Dummy
                terminated=True,
                truncated=False,
            )
            # Route to appropriate buffer
            if seat in dqn_seats_set:
                dqn_buffer.push(transition)
            elif opponent_buffer is not None:
                opponent_buffer.push(transition)

    # DQN wins if any DQN seat won
    dqn_won = winner_seat in dqn_seats_set

    return {
        "winner_seat": winner_seat,
        "num_steps": num_steps,
        "dqn_seats": dqn_seats,
        "dqn_won": dqn_won,
    }


def train_step(
    dqn_policy: DQNPolicy,
    optimizer: torch.optim.Optimizer,
    dqn_buffer: ReplayBuffer,
    batch_size: int,
    gamma: float = 0.99,
    opponent_buffer: ReplayBuffer = None,
    opponent_ratio: float = 0.0,
) -> float:
    """
    Perform one DQN training step.

    Args:
        dqn_policy: DQN policy to train
        optimizer: PyTorch optimizer
        dqn_buffer: Replay buffer with DQN-generated transitions
        batch_size: Batch size for training
        gamma: Discount factor
        opponent_buffer: Optional replay buffer with opponent transitions
        opponent_ratio: Fraction of batch to sample from opponent buffer (0.0 to 1.0)

    Returns:
        Loss value
    """
    # Calculate how many samples from each buffer
    if opponent_buffer is not None and opponent_ratio > 0.0:
        opponent_batch_size = int(batch_size * opponent_ratio)
        dqn_batch_size = batch_size - opponent_batch_size
    else:
        dqn_batch_size = batch_size
        opponent_batch_size = 0

    # Check if we can sample enough from DQN buffer
    if not dqn_buffer.can_sample(dqn_batch_size):
        return 0.0

    # Sample from DQN buffer
    dqn_samples = dqn_buffer.sample(dqn_batch_size)

    # Sample from opponent buffer if available and has enough samples
    if opponent_batch_size > 0 and opponent_buffer.can_sample(opponent_batch_size):
        opp_samples = opponent_buffer.sample(opponent_batch_size)
        # Concatenate samples from both buffers
        obs = np.concatenate([dqn_samples[0], opp_samples[0]], axis=0)
        actions = np.concatenate([dqn_samples[1], opp_samples[1]], axis=0)
        rewards = np.concatenate([dqn_samples[2], opp_samples[2]], axis=0)
        next_obs = np.concatenate([dqn_samples[3], opp_samples[3]], axis=0)
        action_masks = np.concatenate([dqn_samples[4], opp_samples[4]], axis=0)
        terminated = np.concatenate([dqn_samples[5], opp_samples[5]], axis=0)
        truncated = np.concatenate([dqn_samples[6], opp_samples[6]], axis=0)
    else:
        obs, actions, rewards, next_obs, action_masks, terminated, truncated = dqn_samples

    device = dqn_policy.device

    # Convert to tensors
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
    actions_t = torch.tensor(actions, dtype=torch.int64, device=device)
    rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
    next_obs_t = torch.tensor(next_obs, dtype=torch.float32, device=device)
    action_masks_t = torch.tensor(action_masks, dtype=torch.bool, device=device)
    terminated_t = torch.tensor(terminated, dtype=torch.float32, device=device)
    # truncated_t = torch.tensor(truncated, dtype=torch.float32, device=device)

    # Current Q-values for taken actions
    q_values = dqn_policy.q_network(obs_t)
    q_values = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

    # Target Q-values (Double DQN: use online network to select, target to evaluate)
    with torch.no_grad():
        # Get Q-values from online network for action selection
        next_q_online = dqn_policy.q_network(next_obs_t)
        # Mask invalid actions
        next_q_online[~action_masks_t] = float("-inf")
        # Select best actions
        best_actions = next_q_online.argmax(dim=1)

        # Get Q-values from target network for evaluation
        next_q_target = dqn_policy.target_network(next_obs_t)
        next_q_values = next_q_target.gather(1, best_actions.unsqueeze(1)).squeeze(1)

        # TD target: r + gamma * Q(s', a') * (1 - terminated)
        # Note: We don't bootstrap on terminated (game over), but we could on truncated
        # For simplicity, treat both as terminal for now
        targets = rewards_t + gamma * next_q_values * (1 - terminated_t)

    # Huber loss (smooth L1)
    loss = F.smooth_l1_loss(q_values, targets)

    # Optimize
    optimizer.zero_grad()
    loss.backward()
    # Gradient clipping for stability
    torch.nn.utils.clip_grad_norm_(dqn_policy.q_network.parameters(), max_norm=10.0)
    optimizer.step()

    return loss.item()


def evaluate(
    env: BluffEnv,
    dqn_policy: DQNPolicy,
    opponent_policy,
    num_players: int,
    num_games: int = 100,
    num_dqn: int = 1,
    randomize_seats: bool = True,
) -> dict:
    """
    Evaluate DQN policy against opponents.

    Args:
        env: BluffEnv instance
        dqn_policy: DQN policy to evaluate
        opponent_policy: Opponent policy
        num_players: Number of players in game
        num_games: Number of games to play
        num_dqn: Number of DQN seats (all share the same policy)
        randomize_seats: Whether to randomize DQN seats each game

    Returns:
        Dict with evaluation stats including per-agent win counts
    """
    old_epsilon = dqn_policy.epsilon
    dqn_policy.set_epsilon(0.0)  # Greedy for evaluation
    dqn_policy.eval_mode()

    rng = np.random.default_rng(seed=9999)
    dqn_wins = 0
    heuristic_wins = 0

    for i in range(num_games):
        # Randomize or fix DQN seats
        if randomize_seats:
            dqn_seats = set(rng.choice(num_players, size=num_dqn, replace=False))
        else:
            dqn_seats = set(range(num_dqn))

        env.reset(seed=1000 + i)

        for agent in env.agent_iter():
            obs, reward, term, trunc, info = env.last()

            if term or trunc:
                action = None
            else:
                seat = int(agent.split("_")[1])
                if seat in dqn_seats:
                    action = dqn_policy.select_action(obs, obs["action_mask"])
                else:
                    action = opponent_policy.select_action(obs, obs["action_mask"])

            env.step(action)

        # Check winner
        winner_seat = None
        for agent_name in env.possible_agents:
            if env._cumulative_rewards.get(agent_name, 0) > 0:
                winner_seat = int(agent_name.split("_")[1])
                break

        if winner_seat in dqn_seats:
            dqn_wins += 1
        else:
            heuristic_wins += 1

    dqn_policy.set_epsilon(old_epsilon)
    dqn_policy.train_mode()

    # Expected win rate: num_dqn out of num_players
    expected_rate = num_dqn / num_players

    return {
        "dqn_win_rate": dqn_wins / num_games,
        "dqn_wins": dqn_wins,
        "heuristic_wins": heuristic_wins,
        "games": num_games,
        "expected_rate": expected_rate,
        "num_dqn": num_dqn,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train DQN for Bluff")
    parser.add_argument("--num-players", type=int, default=2, help="Number of players")
    parser.add_argument("--dice-per-player", type=int, default=3, help="Dice per player")
    parser.add_argument("--num-dqn", type=int, default=1, help="Number of DQN seats (all share same policy)")
    parser.add_argument("--episodes", type=int, default=5000, help="Training episodes")
    parser.add_argument("--opponent-ratio", type=float, default=0.2,
                        help="Fraction of training batch from opponent buffer (0.0-1.0)")
    # Reward configuration
    parser.add_argument("--reward-preset", type=str, default="sparse",
                        choices=["sparse", "round", "dense"],
                        help="Reward preset: sparse (game-end only), round (+round rewards), dense (+step rewards)")
    parser.add_argument("--reward-round-win", type=float, default=None,
                        help="Override round_win reward (default: 0 for sparse, 0.1 for round/dense)")
    parser.add_argument("--reward-successful-call", type=float, default=None,
                        help="Override successful_call reward (default: 0 for sparse/round, 0.05 for dense)")
    parser.add_argument("--dice-fraction-scale", type=float, default=None,
                        help="Scale for delta dice-fraction reward (non-linear by game phase, try 0.5-2.0)")
    parser.add_argument("--game-win", type=float, default=None,
                        help="Override game_win reward (default: 1.0)")
    parser.add_argument("--elimination", type=float, default=None,
                        help="Override elimination penalty (default: -1.0)")
    args = parser.parse_args()

    # Validate num_dqn
    if args.num_dqn < 1 or args.num_dqn >= args.num_players:
        raise ValueError(f"num_dqn must be between 1 and {args.num_players - 1} (need at least 1 opponent)")

    # Validate opponent_ratio
    if args.opponent_ratio < 0.0 or args.opponent_ratio >= 1.0:
        raise ValueError("opponent_ratio must be in [0.0, 1.0)")

    # Build reward config from preset + overrides
    if args.reward_preset == "sparse":
        reward_config = RewardConfig.sparse()
    elif args.reward_preset == "round":
        reward_config = RewardConfig.round_based()
    else:  # dense
        reward_config = RewardConfig.dense()

    # Apply any CLI overrides
    if args.reward_round_win is not None:
        reward_config.round_win = args.reward_round_win
        reward_config.round_loss = -args.reward_round_win  # Keep symmetric
    if args.reward_successful_call is not None:
        reward_config.successful_call = args.reward_successful_call
        reward_config.failed_call = -args.reward_successful_call  # Keep symmetric
    if args.dice_fraction_scale is not None:
        reward_config.dice_fraction_scale = args.dice_fraction_scale
    if args.game_win is not None:
        reward_config.game_win = args.game_win
    if args.elimination is not None:
        reward_config.elimination = args.elimination

    # Hyperparameters
    num_players = args.num_players
    dice_per_player = args.dice_per_player
    num_dqn = args.num_dqn
    opponent_ratio = args.opponent_ratio
    num_faces = 6
    max_tracked_players = 32

    num_episodes = args.episodes
    batch_size = 64
    learning_rate = 1e-3
    gamma = 0.99
    buffer_capacity = 50000
    target_update_freq = 100
    eval_freq = 500
    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay_episodes = 2000

    # Derived values
    max_dice = num_players * dice_per_player
    num_actions = max_dice * num_faces + 1
    obs_dim = get_flat_obs_dim(num_players, num_faces, max_tracked_players)

    print("=" * 60)
    print("DQN Training for Bluff")
    print("=" * 60)
    print(f"Players: {num_players}, Dice/player: {dice_per_player}")
    print(f"DQN seats: {num_dqn}, Heuristic seats: {num_players - num_dqn}")
    print(f"Observation dim: {obs_dim}, Actions: {num_actions}")
    print(f"Episodes: {num_episodes}, Batch size: {batch_size}")
    print(f"Opponent buffer ratio: {opponent_ratio:.0%}")
    print(f"Reward preset: {args.reward_preset}")
    if reward_config.game_win != 1.0 or reward_config.elimination != -1.0:
        print(f"  game_win={reward_config.game_win}, elimination={reward_config.elimination}")
    if reward_config.round_win != 0 or reward_config.successful_call != 0 or reward_config.dice_fraction_scale != 0:
        print(f"  round_win={reward_config.round_win}, successful_call={reward_config.successful_call}, dice_fraction_scale={reward_config.dice_fraction_scale}")
    print()

    # Create environment
    env = BluffEnv(
        num_players=num_players,
        dice_per_player=dice_per_player,
        num_faces=num_faces,
        max_tracked_players=max_tracked_players,
        reward_config=reward_config,
    )

    # Create DQN policy
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    dqn_policy = DQNPolicy(
        obs_dim=obs_dim,
        num_actions=num_actions,
        num_players=num_players,
        num_faces=num_faces,
        max_dice=max_dice,
        max_tracked_players=max_tracked_players,
        hidden_dims=(128, 128),
        epsilon=epsilon_start,
        device=device,
    )

    # Create opponent (heuristic agent wrapped for env)
    heuristic_agent = HeuristicAgent("heuristic_opponent", seed=42)
    opponent_policy = wrap_for_env(heuristic_agent, num_faces=num_faces, max_dice=max_dice)

    # Optimizer and buffers
    optimizer = Adam(dqn_policy.q_network.parameters(), lr=learning_rate)
    dqn_buffer = ReplayBuffer(capacity=buffer_capacity)
    opponent_buffer = ReplayBuffer(capacity=buffer_capacity) if opponent_ratio > 0 else None

    # Training loop
    losses = []
    # Track wins during training (rolling window)
    recent_wins = {"dqn": [], "heuristic": []}
    train_rng = np.random.default_rng(seed=12345)

    print("Starting training...")
    expected_dqn_rate = num_dqn / num_players * 100
    print(f"Expected DQN win rate if equal skill: {expected_dqn_rate:.1f}% ({num_dqn}/{num_players} seats)")
    print()

    for episode in range(num_episodes):
        # Epsilon decay
        epsilon = max(
            epsilon_end,
            epsilon_start - (epsilon_start - epsilon_end) * episode / epsilon_decay_episodes,
        )
        dqn_policy.set_epsilon(epsilon)

        # Randomize which seats are DQN (select num_dqn seats randomly)
        dqn_seats = list(train_rng.choice(num_players, size=num_dqn, replace=False))

        # Collect episode
        ep_info = collect_episode(
            env=env,
            dqn_policy=dqn_policy,
            opponent_policy=opponent_policy,
            dqn_buffer=dqn_buffer,
            opponent_buffer=opponent_buffer,
            dqn_seats=dqn_seats,
            seed=episode,
        )

        # Track wins
        recent_wins["dqn"].append(1 if ep_info["dqn_won"] else 0)
        recent_wins["heuristic"].append(0 if ep_info["dqn_won"] else 1)
        # Keep last 100 episodes
        for key in recent_wins:
            if len(recent_wins[key]) > 100:
                recent_wins[key].pop(0)

        # Train
        if dqn_buffer.can_sample(batch_size):
            loss = train_step(
                dqn_policy, optimizer, dqn_buffer, batch_size, gamma,
                opponent_buffer=opponent_buffer,
                opponent_ratio=opponent_ratio,
            )
            losses.append(loss)

            # Update target network
            if (episode + 1) % target_update_freq == 0:
                dqn_policy.soft_update_target_network(tau=0.01)

        # Logging
        if (episode + 1) % 100 == 0:
            recent_losses = losses[-100:] if losses else [0]
            avg_loss = sum(recent_losses) / len(recent_losses)
            dqn_rate = sum(recent_wins["dqn"]) / len(recent_wins["dqn"]) * 100
            heur_rate = sum(recent_wins["heuristic"]) / len(recent_wins["heuristic"]) * 100
            print(
                f"Episode {episode + 1:5d} | "
                f"Eps: {epsilon:.3f} | "
                f"Loss: {avg_loss:.4f} | "
                f"DQN: {dqn_rate:.1f}% | Heur: {heur_rate:.1f}%"
            )

        # Evaluation
        if (episode + 1) % eval_freq == 0:
            eval_results = evaluate(
                env=env,
                dqn_policy=dqn_policy,
                opponent_policy=opponent_policy,
                num_players=num_players,
                num_games=100,
                num_dqn=num_dqn,
                randomize_seats=True,
            )
            print(
                f"  >> Eval: DQN {eval_results['dqn_win_rate']*100:.1f}% "
                f"({eval_results['dqn_wins']}/{eval_results['games']}) | "
                f"Heuristic {eval_results['heuristic_wins']}/{eval_results['games']} | "
                f"Expected: {eval_results['expected_rate']*100:.1f}%"
            )
            print()

    print()
    print("=" * 60)
    print("Training Complete!")
    print("=" * 60)

    # Final evaluation
    print(f"\nFinal Evaluation (500 games, {num_dqn} DQN vs {num_players - num_dqn} Heuristic, randomized seats):")
    final_eval = evaluate(
        env=env,
        dqn_policy=dqn_policy,
        opponent_policy=opponent_policy,
        num_players=num_players,
        num_games=500,
        num_dqn=num_dqn,
        randomize_seats=True,
    )
    print(f"  DQN win rate: {final_eval['dqn_win_rate']*100:.1f}% ({final_eval['dqn_wins']}/500)")
    print(f"  Heuristic wins: {final_eval['heuristic_wins']}/500")
    print(f"  Expected if equal: {final_eval['expected_rate']*100:.1f}%")

    # Save model with metadata
    save_path = Path(__file__).resolve().parent.parent / "models"
    save_path.mkdir(exist_ok=True)
    model_path = save_path / f"dqn_{num_players}p_{dice_per_player}d.pt"
    dqn_policy.save(str(model_path), dice_per_player=dice_per_player)
    print(f"\nModel saved to: {model_path}")
    print(f"  Config: {num_players} players, {dice_per_player} dice each")
    print(f"  Training: {num_dqn} DQN seats vs {num_players - num_dqn} Heuristic seats")


if __name__ == "__main__":
    main()
