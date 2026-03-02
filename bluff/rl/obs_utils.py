"""Observation utilities for RL training."""

from typing import Any, Dict

import numpy as np


def flatten_obs(
    obs: Dict[str, Any],
    num_players: int,
    num_faces: int,
    max_dice: int,
    max_tracked_players: int,
) -> np.ndarray:
    """
    Flatten nested dict observation into a single vector for neural network input.

    The flattening order is deterministic and documented here for reproducibility:
    1. private/dice_counts: (num_faces,)
    2. public_round/dice_per_seat: (num_players,)
    3. public_round/active_mask: (num_players,)
    4. public_round/current_bid: (2,) normalized by max_dice and num_faces
    5. public_round/bid_exists: (1,)
    6. public_round/my_seat: (num_players,) one-hot
    7. public_round/current_seat: (num_players,) one-hot
    8. public_round/round_number: (1,) normalized
    9. public_player/bluff_rate: (max_tracked_players,)
    10. public_player/call_rate: (max_tracked_players,)
    11. public_player/aggression: (max_tracked_players,) clipped and normalized

    Note: seat_to_player_idx, rounds_played, dice_remaining are excluded as they're
    either redundant or the agent should learn to use bluff_rate/call_rate directly.

    Args:
        obs: Nested dict observation from BluffEnv
        num_players: Number of players in game
        num_faces: Number of die faces
        max_dice: Maximum total dice
        max_tracked_players: Max players for stats arrays

    Returns:
        Flat numpy array of shape (flat_dim,)
    """
    parts = []

    # 1. Private dice counts - normalize by max possible
    dice_counts = obs["private"]["dice_counts"].astype(np.float32)
    parts.append(dice_counts / max(max_dice, 1))

    # 2. Dice per seat - normalize
    dice_per_seat = obs["public_round"]["dice_per_seat"].astype(np.float32)
    parts.append(dice_per_seat / max(max_dice, 1))

    # 3. Active mask - already 0/1
    active_mask = obs["public_round"]["active_mask"].astype(np.float32)
    parts.append(active_mask)

    # 4. Current bid - normalize count and face separately
    current_bid = obs["public_round"]["current_bid"].astype(np.float32)
    bid_normalized = np.array([
        current_bid[0] / max(max_dice, 1),  # count
        current_bid[1] / max(num_faces, 1),  # face
    ], dtype=np.float32)
    parts.append(bid_normalized)

    # 5. Bid exists - scalar to array
    bid_exists = np.array([float(obs["public_round"]["bid_exists"])], dtype=np.float32)
    parts.append(bid_exists)

    # 6. My seat - one-hot encode
    my_seat = int(obs["public_round"]["my_seat"])
    my_seat_onehot = np.zeros(num_players, dtype=np.float32)
    my_seat_onehot[my_seat] = 1.0
    parts.append(my_seat_onehot)

    # 7. Current seat - one-hot encode
    current_seat = int(obs["public_round"]["current_seat"])
    current_seat_onehot = np.zeros(num_players, dtype=np.float32)
    current_seat_onehot[current_seat] = 1.0
    parts.append(current_seat_onehot)

    # 8. Round number - normalize
    round_number = float(obs["public_round"]["round_number"])
    round_normalized = np.array([round_number / max(max_dice, 1)], dtype=np.float32)
    parts.append(round_normalized)

    # 9. Bluff rate - already 0-1
    bluff_rate = obs["public_player"]["bluff_rate"].astype(np.float32)
    parts.append(bluff_rate)

    # 10. Call rate - already 0-1
    call_rate = obs["public_player"]["call_rate"].astype(np.float32)
    parts.append(call_rate)

    # 11. Aggression - clip to [-2, 2] and normalize to [-1, 1]
    aggression = obs["public_player"]["aggression"].astype(np.float32)
    aggression_normalized = np.clip(aggression, -2.0, 2.0) / 2.0
    parts.append(aggression_normalized)

    return np.concatenate(parts)


def get_flat_obs_dim(
    num_players: int,
    num_faces: int,
    max_tracked_players: int,
) -> int:
    """
    Calculate the dimension of the flattened observation vector.

    Args:
        num_players: Number of players in game
        num_faces: Number of die faces
        max_tracked_players: Max players for stats arrays

    Returns:
        Total dimension of flattened observation
    """
    dim = 0
    dim += num_faces           # dice_counts
    dim += num_players         # dice_per_seat
    dim += num_players         # active_mask
    dim += 2                   # current_bid (count, face)
    dim += 1                   # bid_exists
    dim += num_players         # my_seat one-hot
    dim += num_players         # current_seat one-hot
    dim += 1                   # round_number
    dim += max_tracked_players # bluff_rate
    dim += max_tracked_players # call_rate
    dim += max_tracked_players # aggression
    return dim
