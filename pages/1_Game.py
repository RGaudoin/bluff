#!/usr/bin/env python
"""
Bluff Game Page - Step-through gameplay with reveal phase.
"""

from datetime import datetime

import streamlit as st
from bluff.game.game import BluffGame
from bluff.game.types import Action, ActionType, Bid

# --- Page Config ---
st.set_page_config(
    page_title="Bluff - Game",
    page_icon="🎲",
    layout="wide",
)


def get_seat_to_player_id(num_players: int) -> tuple:
    """Create seat to player_id mapping."""
    return tuple((seat, f"player_{seat}") for seat in range(num_players))


def get_agent_label(seat: int) -> str:
    """Get display label for agent at seat."""
    agent_types = st.session_state.get("seat_agent_types", {})
    agent_type = agent_types.get(seat, "AI")

    if agent_type == "Human":
        return "You"
    elif "DQN" in agent_type:
        return "DQN"
    elif agent_type == "Heuristic":
        return "Heur"
    elif agent_type == "Random":
        return "Rand"
    else:
        return "AI"


def render_dice(dice: tuple, large: bool = False) -> str:
    """Render dice as emoji string."""
    dice_emoji = {1: "⚀", 2: "⚁", 3: "⚂", 4: "⚃", 5: "⚄", 6: "⚅"}
    size = "##" if large else ""
    return f"{size} " + " ".join(dice_emoji.get(d, str(d)) for d in sorted(dice))


def render_dice_counts(dice: tuple) -> str:
    """Render dice as count summary."""
    counts = {}
    for d in dice:
        counts[d] = counts.get(d, 0) + 1
    return ", ".join(f"{v}×{k}s" for k, v in sorted(counts.items()) if v > 0)


def log_action(seat: int, action: Action, state):
    """Log an action with detailed metrics tracking."""
    agent_label = get_agent_label(seat)
    agent_type = st.session_state.get("seat_agent_types", {}).get(seat, "Unknown")
    player = state.players[seat]

    # Initialize action_log if needed
    if "action_log" not in st.session_state:
        st.session_state.action_log = []

    # Build detailed action record
    action_record = {
        "seat": seat,
        "agent_type": agent_type,
        "action_type": action.action_type.name,
        "player_dice_count": player.num_dice,
        "total_dice": state.total_dice,
    }

    if action.action_type == ActionType.CALL:
        st.session_state.history.append(f"Seat {seat} ({agent_label}): **CALL!**")
        st.session_state.per_player_history[seat].append("CALL")
        # Track bid being called
        action_record["called_bid"] = str(state.current_bid) if state.current_bid else None
        action_record["bidder_seat"] = state.bidder_seat
    else:
        st.session_state.history.append(f"Seat {seat} ({agent_label}): {action.bid}")
        st.session_state.per_player_history[seat].append(str(action.bid))
        action_record["bid_count"] = action.bid.count
        action_record["bid_face"] = action.bid.face_value
        # Track how aggressive the bid is relative to total dice
        action_record["bid_ratio"] = action.bid.count / state.total_dice if state.total_dice > 0 else 0

    st.session_state.action_log.append(action_record)


def log_round_result(round_result, state):
    """Log round result for metrics tracking."""
    if "round_results_log" not in st.session_state:
        st.session_state.round_results_log = []

    # Derive caller and bidder from result
    if round_result.bid_was_true:
        caller_seat = round_result.loser_seat
        bidder_seat = round_result.winner_seat
    else:
        caller_seat = round_result.winner_seat
        bidder_seat = round_result.loser_seat

    caller_type = st.session_state.get("seat_agent_types", {}).get(caller_seat, "Unknown")
    bidder_type = st.session_state.get("seat_agent_types", {}).get(bidder_seat, "Unknown")

    result_record = {
        "caller_seat": caller_seat,
        "caller_type": caller_type,
        "bidder_seat": bidder_seat,
        "bidder_type": bidder_type,
        "bid_was_true": round_result.bid_was_true,
        "called_bid": str(round_result.called_bid),
        "actual_count": round_result.actual_count,
    }

    st.session_state.round_results_log.append(result_record)


def execute_one_ai_turn():
    """Execute exactly one AI turn."""
    game = st.session_state.game
    state = st.session_state.state
    num_players = len(state.players)
    seat_to_player_id = get_seat_to_player_id(num_players)

    seat = state.current_seat
    agent = st.session_state.agents.get(seat)

    if agent is None:
        return False

    # Get agent's observation and valid actions
    player_id = f"player_{seat}"
    obs = state.get_observation(seat, player_id, seat_to_player_id)
    valid_actions = game.get_valid_actions(state)

    # Agent selects action
    action = agent.select_action(obs, valid_actions)

    # Log the action with detailed tracking
    log_action(seat, action, state)

    # Execute action
    new_state, round_result = game.step(state, action)
    st.session_state.state = new_state

    # Handle round result - go to confirm reveal phase
    if round_result:
        log_round_result(round_result, state)
        st.session_state.round_result = round_result
        # Store all dice before they change
        st.session_state.dice_at_call = tuple(p.dice for p in state.players)
        st.session_state.game_phase = "confirm_reveal"
        return True

    if new_state.is_game_over:
        st.session_state.game_phase = "game_over"
        return True

    return True


def human_action(action: Action):
    """Process human player's action."""
    game = st.session_state.game
    state = st.session_state.state
    seat = st.session_state.human_seat

    # Log the action with detailed tracking
    log_action(seat, action, state)

    # Store all dice before they change (in case of call)
    st.session_state.dice_at_call = tuple(p.dice for p in state.players)

    # Execute action
    new_state, round_result = game.step(state, action)
    st.session_state.state = new_state

    # Handle round result
    if round_result:
        log_round_result(round_result, state)
        st.session_state.round_result = round_result
        st.session_state.game_phase = "confirm_reveal"
    elif new_state.is_game_over:
        st.session_state.game_phase = "game_over"


def proceed_to_reveal():
    """Move from confirm_reveal to reveal phase."""
    st.session_state.game_phase = "reveal"


def continue_after_reveal():
    """Continue game after reveal phase."""
    state = st.session_state.state
    if state.is_game_over:
        st.session_state.game_phase = "game_over"
    else:
        st.session_state.game_phase = "playing"
        st.session_state.round_result = None
        st.session_state.history.append("---")
        st.session_state.history.append("*New round*")


def is_human_turn() -> bool:
    """Check if it's the human's turn."""
    if st.session_state.spectator_mode:
        return False
    state = st.session_state.state
    return state.current_seat == st.session_state.human_seat


def should_show_dice(seat: int) -> bool:
    """Determine if we should show dice for a seat."""
    spectator_mode = st.session_state.get("spectator_mode", False)
    show_all_dice = st.session_state.get("show_all_dice", False)
    human_seat = st.session_state.get("human_seat", 0)

    # Always show in spectator mode or show_all_dice mode
    if spectator_mode or show_all_dice:
        return True

    # Show human's own dice
    if seat == human_seat:
        return True

    return False


def compute_agent_metrics():
    """Compute detailed metrics from action_log and round_results_log."""
    action_log = st.session_state.get("action_log", [])
    round_results_log = st.session_state.get("round_results_log", [])

    # Metrics per agent type
    metrics = {}

    # Process actions
    for action in action_log:
        agent_type = action.get("agent_type", "Unknown")
        if agent_type not in metrics:
            metrics[agent_type] = {
                "bids": 0,
                "calls": 0,
                "bid_ratios": [],  # For average aggressiveness
            }

        if action["action_type"] == "CALL":
            metrics[agent_type]["calls"] += 1
        else:
            metrics[agent_type]["bids"] += 1
            if "bid_ratio" in action:
                metrics[agent_type]["bid_ratios"].append(action["bid_ratio"])

    # Process round results
    for result in round_results_log:
        caller_type = result.get("caller_type", "Unknown")
        bidder_type = result.get("bidder_type", "Unknown")

        # Initialize if needed
        for agent_type in [caller_type, bidder_type]:
            if agent_type not in metrics:
                metrics[agent_type] = {
                    "bids": 0,
                    "calls": 0,
                    "bid_ratios": [],
                }

        # Track call accuracy (caller perspective)
        if "successful_calls" not in metrics[caller_type]:
            metrics[caller_type]["successful_calls"] = 0
            metrics[caller_type]["failed_calls"] = 0

        if result["bid_was_true"]:
            metrics[caller_type]["failed_calls"] += 1
        else:
            metrics[caller_type]["successful_calls"] += 1

        # Track bid truthfulness (bidder perspective)
        if "true_bids_called" not in metrics[bidder_type]:
            metrics[bidder_type]["true_bids_called"] = 0
            metrics[bidder_type]["false_bids_called"] = 0

        if result["bid_was_true"]:
            metrics[bidder_type]["true_bids_called"] += 1
        else:
            metrics[bidder_type]["false_bids_called"] += 1

    return metrics


def record_game_result(winner_seat: int):
    """Record the completed game result for stats tracking."""
    num_players = st.session_state.get("num_players", 2)
    seat_agent_types = st.session_state.get("seat_agent_types", {})

    # Compute metrics from this game
    game_metrics = compute_agent_metrics()

    # Build game record
    game_record = {
        "timestamp": datetime.now().isoformat(),
        "num_players": num_players,
        "dice_per_player": st.session_state.get("dice_per_player", 3),
        "winner_seat": winner_seat,
        "winner_type": seat_agent_types.get(winner_seat, "Unknown"),
        "players": {seat: seat_agent_types.get(seat, "Unknown") for seat in range(num_players)},
        "history": st.session_state.history.copy(),
        "metrics": game_metrics,
    }

    # Append to completed games
    if "completed_games" not in st.session_state:
        st.session_state.completed_games = []
    st.session_state.completed_games.append(game_record)

    # Update agent stats
    if "agent_stats" not in st.session_state:
        st.session_state.agent_stats = {}

    for seat in range(num_players):
        agent_type = seat_agent_types.get(seat, "Unknown")
        if agent_type not in st.session_state.agent_stats:
            st.session_state.agent_stats[agent_type] = {
                "wins": 0,
                "games": 0,
                "total_bids": 0,
                "total_calls": 0,
                "successful_calls": 0,
                "failed_calls": 0,
                "true_bids_called": 0,
                "false_bids_called": 0,
                "bid_ratios": [],
            }

        stats = st.session_state.agent_stats[agent_type]
        stats["games"] += 1
        if seat == winner_seat:
            stats["wins"] += 1

        # Merge game metrics into cumulative stats
        if agent_type in game_metrics:
            gm = game_metrics[agent_type]
            stats["total_bids"] += gm.get("bids", 0)
            stats["total_calls"] += gm.get("calls", 0)
            stats["successful_calls"] += gm.get("successful_calls", 0)
            stats["failed_calls"] += gm.get("failed_calls", 0)
            stats["true_bids_called"] += gm.get("true_bids_called", 0)
            stats["false_bids_called"] += gm.get("false_bids_called", 0)
            stats["bid_ratios"].extend(gm.get("bid_ratios", []))


def main():
    # Check if game is configured
    if not st.session_state.get("game_configured", False):
        st.warning("No game in progress. Please set up a game first.")
        if st.button("← Back to Setup"):
            st.switch_page("app.py")
        return

    state = st.session_state.state
    game = st.session_state.game
    human_seat = st.session_state.human_seat
    spectator_mode = st.session_state.spectator_mode
    show_all_dice = st.session_state.get("show_all_dice", False)
    phase = st.session_state.game_phase

    # --- Header ---
    col_title, col_nav = st.columns([3, 1])
    with col_title:
        if spectator_mode:
            mode_str = "👁️ Spectator Mode"
        elif show_all_dice:
            mode_str = f"🔓 Playing as Seat {human_seat} (all dice visible)"
        else:
            mode_str = f"🎮 Playing as Seat {human_seat}"
        st.markdown(f"## 🎲 Bluff — {mode_str}")
    with col_nav:
        if st.button("← Setup"):
            st.switch_page("app.py")

    # --- Game Over ---
    if phase == "game_over":
        winner_seat = state.winner_seat

        # Record result (only once per game)
        if not st.session_state.get("game_result_recorded", False):
            record_game_result(winner_seat)
            st.session_state.game_result_recorded = True

        st.balloons()
        winner_label = get_agent_label(winner_seat)
        if spectator_mode:
            st.success(f"🏆 **Seat {winner_seat} ({winner_label}) wins!**")
        elif winner_seat == human_seat:
            st.success("🎉 **You won!**")
        else:
            st.error(f"💀 **Seat {winner_seat} ({winner_label}) wins!**")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 New Game", type="primary", use_container_width=True):
                st.session_state.game_phase = "setup"
                st.session_state.game_result_recorded = False
                st.switch_page("app.py")
        with col2:
            if st.button("📊 View Stats", use_container_width=True):
                st.switch_page("pages/2_Stats.py")
        return

    # --- Confirm Reveal Phase ---
    if phase == "confirm_reveal":
        result = st.session_state.round_result

        st.markdown("### 🚨 Call Made!")
        # Derive caller and bidder from winner/loser based on bid_was_true
        # If bid was true: caller loses (loser=caller, winner=bidder)
        # If bid was false: bidder loses (loser=bidder, winner=caller)
        if result.bid_was_true:
            caller_seat = result.loser_seat
            bidder_seat = result.winner_seat
        else:
            caller_seat = result.winner_seat
            bidder_seat = result.loser_seat

        caller_label = get_agent_label(caller_seat)
        bidder_label = get_agent_label(bidder_seat)

        st.markdown(f"**Seat {caller_seat} ({caller_label})** called **Seat {bidder_seat} ({bidder_label})**'s bid of **{result.called_bid}**")

        st.divider()
        if st.button("🔍 Reveal All Dice", type="primary", use_container_width=True):
            proceed_to_reveal()
            st.rerun()
        return

    # --- Reveal Phase ---
    if phase == "reveal":
        st.markdown("### 🔍 Reveal!")
        result = st.session_state.round_result

        # Show all dice (from before the call)
        st.markdown("**All dice revealed:**")
        dice_at_call = st.session_state.get("dice_at_call", tuple(p.dice for p in state.players))
        cols = st.columns(len(state.players))
        for i, col in enumerate(zip(cols)):
            col = cols[i]
            with col:
                label = f"Seat {i}"
                agent_label = get_agent_label(i)
                if agent_label == "You":
                    label += " (You)"
                else:
                    label += f" ({agent_label})"
                st.markdown(f"**{label}**")
                if i < len(dice_at_call):
                    st.markdown(render_dice(dice_at_call[i]))
                    st.caption(render_dice_counts(dice_at_call[i]))

        # Show result
        st.divider()
        bid = result.called_bid
        actual = result.actual_count
        was_true = result.bid_was_true

        if was_true:
            st.success(f"**Bid {bid} was TRUE!** (Actual: {actual} dice showing {bid.face_value})")
            loser_label = get_agent_label(result.loser_seat)
            st.markdown(f"Seat {result.loser_seat} ({loser_label}) loses a die.")
        else:
            st.error(f"**Bid {bid} was FALSE!** (Actual: {actual} dice showing {bid.face_value})")
            loser_label = get_agent_label(result.loser_seat)
            st.markdown(f"Seat {result.loser_seat} ({loser_label}) loses a die.")

        st.divider()
        if st.button("▶️ Continue", type="primary", use_container_width=True):
            continue_after_reveal()
            st.rerun()
        return

    # --- Normal Playing Phase ---

    # Player status row (compact)
    st.markdown("### Players")
    cols = st.columns(len(state.players))
    for i, (col, player) in enumerate(zip(cols, state.players)):
        with col:
            is_current = (state.current_seat == i)
            is_human = (i == human_seat) and not spectator_mode
            is_active = player.is_active
            agent_label = get_agent_label(i)

            # Header
            if is_current and is_active:
                marker = "→ "
            else:
                marker = ""

            if is_human:
                label = f"{marker}Seat {i} (You)"
            else:
                label = f"{marker}Seat {i} ({agent_label})"

            if not is_active:
                st.markdown(f"~~{label}~~")
                st.caption("Eliminated")
            else:
                st.markdown(f"**{label}**")

                # Show dice based on visibility rules
                if should_show_dice(i):
                    st.markdown(render_dice(player.dice))
                    st.caption(render_dice_counts(player.dice))
                else:
                    st.markdown(f"🎲 × {player.num_dice}")

    st.divider()

    # Current bid
    col_bid, col_action = st.columns([1, 1])

    with col_bid:
        st.markdown("### Current Bid")
        if state.current_bid:
            st.markdown(f"## {state.current_bid}")
            bidder_label = get_agent_label(state.bidder_seat)
            st.caption(f"by Seat {state.bidder_seat} ({bidder_label})")
        else:
            st.markdown("*No bid yet*")
            st.caption("First player must bid")

    with col_action:
        st.markdown("### Action")

        # Determine what action to show
        if not spectator_mode and is_human_turn() and state.players[human_seat].is_active:
            # Human's turn - show controls
            st.markdown("**Your turn!**")

            # Bid controls
            c1, c2 = st.columns(2)
            with c1:
                min_count = state.current_bid.count if state.current_bid else 1
                count = st.selectbox(
                    "Count",
                    options=list(range(min_count, state.total_dice + 1)),
                    key="bid_count"
                )
            with c2:
                if state.current_bid and count == state.current_bid.count:
                    min_face = state.current_bid.face_value + 1
                else:
                    min_face = 1
                faces = list(range(min_face, 7))
                if faces:
                    face = st.selectbox("Face", options=faces, key="bid_face")
                else:
                    face = None
                    st.caption("Must increase count")

            c1, c2 = st.columns(2)
            with c1:
                if face is not None:
                    if st.button(f"Bid {count}×{face}s", type="primary", use_container_width=True):
                        action = Action(ActionType.BID, Bid(count, face), human_seat)
                        human_action(action)
                        st.rerun()
            with c2:
                if state.current_bid:
                    if st.button("🚨 Call!", type="secondary", use_container_width=True):
                        action = Action(ActionType.CALL, None, human_seat)
                        human_action(action)
                        st.rerun()

        else:
            # AI's turn or spectator mode
            current_seat = state.current_seat
            if state.players[current_seat].is_active:
                current_label = get_agent_label(current_seat)
                st.markdown(f"**Seat {current_seat}'s turn** ({current_label})")
                if st.button("▶️ Next Turn", type="primary", use_container_width=True):
                    execute_one_ai_turn()
                    st.rerun()
            else:
                st.warning("Unexpected state")

    # History - Sequential view
    st.divider()
    with st.expander("📜 Round History", expanded=True):
        history = st.session_state.history
        if history:
            # Show history in reverse order (most recent first), skip separators for display
            display_items = []
            for item in history:
                if item == "---":
                    continue
                display_items.append(item)

            # Show last 10 items, most recent at top
            for item in reversed(display_items[-10:]):
                st.markdown(item)
        else:
            st.caption("No bids yet")


if __name__ == "__main__":
    main()
