"""Observation and action space definitions for Bluff environment."""

from typing import Dict, Tuple

import numpy as np
from gymnasium import spaces


def create_observation_space(
    num_players: int,
    num_faces: int,
    max_dice: int,
    max_tracked_players: int,
) -> spaces.Dict:
    """
    Create the hierarchical observation space for Bluff.

    Args:
        num_players: Number of players in the game
        num_faces: Number of faces per die (typically 6)
        max_dice: Maximum total dice in the game
        max_tracked_players: Maximum number of unique players to track stats for

    Returns:
        Dictionary observation space with private, public_round, and public_player info
    """
    return spaces.Dict(
        {
            # Private information - only visible to this agent
            "private": spaces.Dict(
                {
                    # Count of each face value in own dice
                    # e.g., [2, 0, 1, 0, 0, 1] means 2x1s, 1x3, 1x6
                    "dice_counts": spaces.Box(
                        low=0,
                        high=max_dice,
                        shape=(num_faces,),
                        dtype=np.int32,
                    ),
                }
            ),
            # Public round information - visible to all
            "public_round": spaces.Dict(
                {
                    # Dice count per seat
                    "dice_per_seat": spaces.Box(
                        low=0,
                        high=max_dice,
                        shape=(num_players,),
                        dtype=np.int32,
                    ),
                    # Active players mask (1 = still in game)
                    "active_mask": spaces.MultiBinary(num_players),
                    # Current bid as (count, face), (0, 0) if no bid
                    "current_bid": spaces.Box(
                        low=0,
                        high=max(max_dice, num_faces),
                        shape=(2,),
                        dtype=np.int32,
                    ),
                    # Whether a bid exists (0 = no, 1 = yes)
                    "bid_exists": spaces.Discrete(2),
                    # This agent's seat
                    "my_seat": spaces.Discrete(num_players),
                    # Whose turn it is
                    "current_seat": spaces.Discrete(num_players),
                    # Maps seat -> player stats index
                    # Allows agent to correlate seat with player behavior
                    "seat_to_player_idx": spaces.Box(
                        low=0,
                        high=max_tracked_players - 1,
                        shape=(num_players,),
                        dtype=np.int32,
                    ),
                    # Current round number (0-indexed)
                    # Useful for learning time-based strategies
                    "round_number": spaces.Discrete(max_dice),
                }
            ),
            # Per-player aggregated statistics (indexed by player_idx)
            "public_player": spaces.Dict(
                {
                    # Rounds played (for cold-start detection)
                    "rounds_played": spaces.Box(
                        low=0,
                        high=np.iinfo(np.int32).max,
                        shape=(max_tracked_players,),
                        dtype=np.int32,
                    ),
                    # Current dice count per player
                    "dice_remaining": spaces.Box(
                        low=0,
                        high=max_dice,
                        shape=(max_tracked_players,),
                        dtype=np.int32,
                    ),
                    # Bluff rate (fraction of called bids that were false)
                    "bluff_rate": spaces.Box(
                        low=0.0,
                        high=1.0,
                        shape=(max_tracked_players,),
                        dtype=np.float32,
                    ),
                    # Call rate (fraction of actions that were calls)
                    "call_rate": spaces.Box(
                        low=0.0,
                        high=1.0,
                        shape=(max_tracked_players,),
                        dtype=np.float32,
                    ),
                    # Aggression (average bid vs expected)
                    "aggression": spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=(max_tracked_players,),
                        dtype=np.float32,
                    ),
                }
            ),
            # Valid action mask
            "action_mask": spaces.MultiBinary(max_dice * num_faces + 1),
        }
    )


def create_action_space(max_dice: int, num_faces: int) -> spaces.Discrete:
    """
    Create the discrete action space for Bluff.

    Action encoding:
    - Bids: action_id = (count - 1) * num_faces + (face - 1)
      where count in [1, max_dice], face in [1, num_faces]
    - CALL: action_id = max_dice * num_faces

    Args:
        max_dice: Maximum total dice in the game
        num_faces: Number of faces per die

    Returns:
        Discrete action space
    """
    num_actions = max_dice * num_faces + 1  # +1 for CALL
    return spaces.Discrete(num_actions)


def encode_bid(count: int, face: int, num_faces: int) -> int:
    """
    Encode a bid as an action id.

    Args:
        count: Number of dice claimed (1-indexed)
        face: Face value claimed (1-indexed)
        num_faces: Number of faces per die

    Returns:
        Action id
    """
    return (count - 1) * num_faces + (face - 1)


def decode_action(
    action_id: int, max_dice: int, num_faces: int
) -> Tuple[str, int, int]:
    """
    Decode an action id to action type and parameters.

    Args:
        action_id: The action id from the discrete space
        max_dice: Maximum total dice in the game
        num_faces: Number of faces per die

    Returns:
        Tuple of (action_type, count, face) where:
        - action_type: "call" or "bid"
        - count: Dice count for bid (0 for call)
        - face: Face value for bid (0 for call)
    """
    call_action = max_dice * num_faces
    if action_id == call_action:
        return ("call", 0, 0)

    count = action_id // num_faces + 1
    face = action_id % num_faces + 1
    return ("bid", count, face)


def get_action_mask(
    current_bid: Tuple[int, int],
    total_dice: int,
    num_faces: int,
    max_dice: int,
) -> np.ndarray:
    """
    Generate valid action mask given current game state.

    Args:
        current_bid: (count, face) of current bid, or (0, 0) if no bid
        total_dice: Total dice currently in play
        num_faces: Number of faces per die
        max_dice: Maximum action space size

    Returns:
        Boolean mask where True = valid action
    """
    bid_count, bid_face = current_bid

    # Build 2D mask: rows = counts (0-indexed), cols = faces (0-indexed)
    mask_2d = np.zeros((max_dice, num_faces), dtype=np.int8)

    if bid_count == 0:
        # No current bid - all bids up to total_dice are valid
        mask_2d[:total_dice, :] = 1
    else:
        # Standard Liar's Dice rules:
        # - Higher count: any face value is valid
        # - Same count: only higher face values are valid
        if bid_count < total_dice:
            # All higher counts with any face
            mask_2d[bid_count:total_dice, :] = 1
        # For same count, only faces > bid_face are valid
        if bid_count <= total_dice:
            mask_2d[bid_count - 1, bid_face:] = 1

    # Flatten 2D mask and append CALL action
    mask = np.zeros(max_dice * num_faces + 1, dtype=np.int8)
    mask[:-1] = mask_2d.ravel()

    # CALL is valid only if there's an existing bid
    if bid_count > 0:
        mask[-1] = 1

    return mask
