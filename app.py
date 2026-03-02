#!/usr/bin/env python
"""
Bluff (Liar's Dice) - Streamlit App

Home/Setup page. Run with: streamlit run app.py
"""

from pathlib import Path

import streamlit as st
import random

from bluff.game.game import BluffGame
from bluff.agents import list_agent_types, create_agent, is_agent_available


# --- Page Config ---
st.set_page_config(
    page_title="Bluff (Liar's Dice)",
    page_icon="🎲",
    layout="centered",
)

# Directory for DQN models
MODELS_DIR = Path(__file__).resolve().parent / "models"


def get_model_path(num_players: int, dice_per_player: int) -> Path:
    """Get path to DQN model for specific game configuration."""
    # Try config-specific model first
    config_path = MODELS_DIR / f"dqn_{num_players}p_{dice_per_player}d.pt"
    if config_path.exists():
        return config_path
    # Fall back to legacy baseline (2p 3d)
    if num_players == 2 and dice_per_player == 3:
        legacy_path = MODELS_DIR / "dqn_baseline.pt"
        if legacy_path.exists():
            return legacy_path
    return config_path  # Return expected path even if doesn't exist


def get_available_agent_types() -> list:
    """Get list of agent types that are currently available for current game config."""
    all_types = list_agent_types()
    available = []
    num_players = st.session_state.get("num_players", 2)
    dice_per_player = st.session_state.get("dice_per_player", 3)

    for agent_type in all_types:
        # Check if agent is available (e.g., DQN needs model file + game config)
        config = {
            "num_players": num_players,
            "dice_per_player": dice_per_player,
        }
        if "DQN" in agent_type:
            config["model_path"] = str(get_model_path(num_players, dice_per_player))
        if is_agent_available(agent_type, config):
            available.append(agent_type)
    return available


def init_session_state():
    """Initialize session state variables."""
    defaults = {
        "game": None,
        "state": None,
        "agents": {},
        "human_seat": 0,
        "history": [],
        "per_player_history": {},
        "game_phase": "setup",  # setup, playing, reveal, confirm_reveal, round_over, game_over
        "round_result": None,
        "spectator_mode": False,
        "show_all_dice": False,
        "game_configured": False,
        "num_players": 2,
        "dice_per_player": 3,
        "seat_agent_types": {},  # seat -> agent type
        # Stats tracking (persists across games in session)
        "completed_games": [],  # List of completed game records
        "agent_stats": {},  # agent_type -> {wins, games, rounds_won, rounds_lost}
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def start_new_game():
    """Start a new game with current settings."""
    cfg = st.session_state

    game = BluffGame(
        num_players=cfg.num_players,
        dice_per_player=cfg.dice_per_player,
        num_faces=6,
    )
    state = game.reset()

    # Create AI agents based on per-seat selection
    agents = {}
    for seat in range(cfg.num_players):
        agent_type = cfg.seat_agent_types.get(seat, "Heuristic")

        # Skip human seat (no agent needed)
        if agent_type == "Human":
            continue

        # Build config for agent
        config = {"seed": random.randint(0, 100000)}
        if "DQN" in agent_type:
            config["model_path"] = str(get_model_path(cfg.num_players, cfg.dice_per_player))
            config["num_players"] = cfg.num_players
            config["dice_per_player"] = cfg.dice_per_player
            config["max_tracked_players"] = 32

        agent = create_agent(agent_type, seat, config)
        if agent is not None:
            agents[seat] = agent

    # Update session state
    st.session_state.game = game
    st.session_state.state = state
    st.session_state.agents = agents
    st.session_state.history = []
    st.session_state.per_player_history = {i: [] for i in range(cfg.num_players)}
    st.session_state.game_phase = "playing"
    st.session_state.round_result = None
    st.session_state.game_configured = True
    # Clear per-game action logs (stats persist)
    st.session_state.action_log = []
    st.session_state.round_results_log = []
    st.session_state.game_result_recorded = False


def main():
    init_session_state()

    st.title("🎲 Bluff (Liar's Dice)")
    st.markdown("*A game of deception and probability*")

    st.divider()

    # --- Game Setup ---
    st.header("Game Setup")

    col1, col2 = st.columns(2)

    with col1:
        num_players = st.slider(
            "Number of players",
            min_value=2,
            max_value=6,
            value=st.session_state.num_players,
            help="Total players including you"
        )
        st.session_state.num_players = num_players

        dice_per_player = st.slider(
            "Dice per player",
            min_value=1,
            max_value=6,
            value=st.session_state.dice_per_player,
        )
        st.session_state.dice_per_player = dice_per_player

    with col2:
        st.session_state.show_all_dice = st.checkbox(
            "🔓 Show all dice",
            value=st.session_state.show_all_dice,
            help="See all players' dice (debug mode)"
        )

    st.divider()

    # --- Per-seat Agent Selection ---
    st.subheader("Player Configuration")

    available_types = get_available_agent_types()

    # Ensure seat_agent_types has entries for all seats
    for seat in range(num_players):
        if seat not in st.session_state.seat_agent_types:
            # Default: seat 0 is Human, others are Heuristic
            st.session_state.seat_agent_types[seat] = "Human" if seat == 0 else "Heuristic"

    # Create columns for each seat
    cols = st.columns(min(num_players, 4))  # Max 4 columns per row
    for seat in range(min(num_players, 4)):
        with cols[seat]:
            current_type = st.session_state.seat_agent_types.get(seat, "Heuristic")
            if current_type not in available_types:
                current_type = "Heuristic"

            try:
                default_idx = available_types.index(current_type)
            except ValueError:
                default_idx = 0

            agent_type = st.selectbox(
                f"Seat {seat}",
                options=available_types,
                index=default_idx,
                key=f"agent_type_{seat}",
            )
            st.session_state.seat_agent_types[seat] = agent_type

            # Show indicator
            if agent_type == "Human":
                st.caption("👤 You")
            elif "DQN" in agent_type:
                st.caption("🤖 Neural Network")
            elif agent_type == "Heuristic":
                st.caption("🧠 Probability-based")
            else:
                st.caption("🎲 Random")

    # Extra row for more than 4 players
    if num_players > 4:
        cols2 = st.columns(num_players - 4)
        for seat in range(4, num_players):
            with cols2[seat - 4]:
                current_type = st.session_state.seat_agent_types.get(seat, "Heuristic")
                if current_type not in available_types:
                    current_type = "Heuristic"

                try:
                    default_idx = available_types.index(current_type)
                except ValueError:
                    default_idx = 0

                agent_type = st.selectbox(
                    f"Seat {seat}",
                    options=available_types,
                    index=default_idx,
                    key=f"agent_type_{seat}",
                )
                st.session_state.seat_agent_types[seat] = agent_type

                if agent_type == "Human":
                    st.caption("👤 You")
                elif "DQN" in agent_type:
                    st.caption("🤖 Neural Network")
                elif agent_type == "Heuristic":
                    st.caption("🧠 Probability-based")
                else:
                    st.caption("🎲 Random")

    # Show info if DQN is not available
    model_path = get_model_path(num_players, dice_per_player)
    dqn_config = {"model_path": str(model_path), "num_players": num_players, "dice_per_player": dice_per_player}
    if not is_agent_available("DQN (trained)", dqn_config):
        st.info(f"ℹ️ DQN agent not available for {num_players}p/{dice_per_player}d (no trained model)")

    # Validate: 0 or 1 human allowed (0 = spectator mode, 1 = playing)
    human_count = sum(1 for s in range(num_players) if st.session_state.seat_agent_types.get(s) == "Human")
    if human_count > 1:
        st.warning(f"⚠️ Maximum one Human seat allowed (currently: {human_count})")
        can_start = False
    else:
        can_start = True

    # Derive spectator mode from human count
    st.session_state.spectator_mode = (human_count == 0)
    if human_count == 0:
        st.info("👁️ Spectator mode: watching AI agents play")

    # Find human seat (0 if spectator mode)
    st.session_state.human_seat = next(
        (s for s in range(num_players) if st.session_state.seat_agent_types.get(s) == "Human"),
        0
    )

    st.divider()

    # Start button
    if st.button("🎮 Start Game", type="primary", use_container_width=True, disabled=not can_start):
        start_new_game()
        st.switch_page("pages/1_Game.py")

    # Continue existing game
    if st.session_state.game_configured and st.session_state.game_phase != "setup":
        if st.button("▶️ Continue Game", use_container_width=True):
            st.switch_page("pages/1_Game.py")

    # View stats (if any games have been played)
    if st.session_state.get("completed_games"):
        if st.button("📊 View Stats", use_container_width=True):
            st.switch_page("pages/2_Stats.py")

    st.divider()

    # Rules
    with st.expander("📜 How to Play", expanded=False):
        st.markdown("""
        **Objective:** Be the last player with dice remaining.

        **Each Round:**
        1. All players roll their dice (hidden from others)
        2. Players take turns making bids or calling bluff
        3. A **bid** claims a minimum count of a face value across ALL dice
        4. Each bid must be higher than the previous (more dice, or same count + higher face)
        5. **Call** if you think the current bid is false

        **Resolution:**
        - If bid was TRUE: caller loses a die
        - If bid was FALSE: bidder loses a die

        **Winning:** Last player with dice wins!

        **Example:** "3 fours" means you claim there are at least 3 dice showing 4 among all players.
        """)

    # Agent info
    with st.expander("🤖 About AI Agents", expanded=False):
        st.markdown("""
        **Available Agents:**

        - **Human** — You play this seat
        - **Random** — Makes random valid moves
        - **Heuristic** — Uses probability calculations to decide
        - **DQN (trained)** — Deep Q-Network trained via reinforcement learning

        *More agent types may be added in the future.*
        """)


if __name__ == "__main__":
    main()
