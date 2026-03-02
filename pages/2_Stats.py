#!/usr/bin/env python
"""
Bluff Stats Page - View session statistics and game history.
"""

import streamlit as st

# --- Page Config ---
st.set_page_config(
    page_title="Bluff - Stats",
    page_icon="📊",
    layout="wide",
)


def main():
    st.markdown("## 📊 Session Statistics")

    # Navigation
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("← Back to Setup"):
            st.switch_page("app.py")

    st.divider()

    # Get session data
    agent_stats = st.session_state.get("agent_stats", {})
    completed_games = st.session_state.get("completed_games", [])

    # --- Agent Stats Section ---
    st.markdown("### Agent Performance")

    if agent_stats:
        # Create columns for each agent type
        agent_types = list(agent_stats.keys())
        cols = st.columns(len(agent_types))

        for col, agent_type in zip(cols, agent_types):
            with col:
                stats = agent_stats[agent_type]
                wins = stats.get("wins", 0)
                games = stats.get("games", 0)
                win_rate = (wins / games * 100) if games > 0 else 0

                st.markdown(f"**{agent_type}**")
                st.metric("Win Rate", f"{win_rate:.1f}%")
                st.caption(f"{wins} wins / {games} games")

        # Detailed metrics section
        st.markdown("#### Detailed Metrics")

        for agent_type in agent_types:
            stats = agent_stats[agent_type]
            total_bids = stats.get("total_bids", 0)
            total_calls = stats.get("total_calls", 0)
            successful_calls = stats.get("successful_calls", 0)
            failed_calls = stats.get("failed_calls", 0)
            true_bids_called = stats.get("true_bids_called", 0)
            false_bids_called = stats.get("false_bids_called", 0)
            bid_ratios = stats.get("bid_ratios", [])

            total_actions = total_bids + total_calls

            with st.expander(f"{agent_type} — Detailed Stats"):
                if total_actions == 0:
                    st.caption("No actions recorded yet")
                    continue

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown("**Action Breakdown**")
                    call_rate = (total_calls / total_actions * 100) if total_actions > 0 else 0
                    st.write(f"Total actions: {total_actions}")
                    st.write(f"Bids: {total_bids}")
                    st.write(f"Calls: {total_calls}")
                    st.write(f"Call rate: {call_rate:.1f}%")

                with col2:
                    st.markdown("**Call Accuracy**")
                    total_calls_made = successful_calls + failed_calls
                    if total_calls_made > 0:
                        accuracy = successful_calls / total_calls_made * 100
                        st.write(f"Successful: {successful_calls}")
                        st.write(f"Failed: {failed_calls}")
                        st.metric("Accuracy", f"{accuracy:.1f}%")
                    else:
                        st.caption("No calls resolved yet")

                with col3:
                    st.markdown("**Bid Truthfulness**")
                    total_bids_called = true_bids_called + false_bids_called
                    if total_bids_called > 0:
                        truthfulness = true_bids_called / total_bids_called * 100
                        st.write(f"True bids: {true_bids_called}")
                        st.write(f"False bids: {false_bids_called}")
                        st.metric("Truthfulness", f"{truthfulness:.1f}%")
                    else:
                        st.caption("No bids called yet")

                # Aggressiveness
                if bid_ratios:
                    avg_ratio = sum(bid_ratios) / len(bid_ratios) * 100
                    st.markdown("**Bid Aggressiveness**")
                    st.write(f"Average bid as % of total dice: {avg_ratio:.1f}%")
                    st.caption("Higher = more aggressive bidding")
    else:
        st.info("No games played yet. Complete a game to see stats here.")

    st.divider()

    # --- Game History Section ---
    st.markdown("### Game History")

    if completed_games:
        st.caption(f"{len(completed_games)} completed game(s) this session")

        # List games in reverse order (most recent first)
        for i, game in enumerate(reversed(completed_games)):
            game_idx = len(completed_games) - i
            timestamp = game.get("timestamp", "Unknown")
            # Parse timestamp for display
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(timestamp)
                time_str = dt.strftime("%H:%M:%S")
            except Exception:
                time_str = timestamp

            winner_seat = game.get("winner_seat", 0)
            winner_type = game.get("winner_type", "Unknown")
            num_players = game.get("num_players", 2)
            dice_per_player = game.get("dice_per_player", 3)

            with st.expander(
                f"Game {game_idx}: {winner_type} won (Seat {winner_seat}) — {time_str}"
            ):
                # Game config
                st.markdown(f"**Config:** {num_players} players, {dice_per_player} dice each")

                # Player types
                players = game.get("players", {})
                player_str = ", ".join(
                    f"Seat {s}: {t}" for s, t in sorted(players.items(), key=lambda x: int(x[0]))
                )
                st.markdown(f"**Players:** {player_str}")

                st.divider()

                # Game history
                st.markdown("**Bid History:**")
                history = game.get("history", [])
                if history:
                    # Display history in order
                    for item in history:
                        if item == "---":
                            st.markdown("---")
                        else:
                            st.markdown(item)
                else:
                    st.caption("No history recorded")
    else:
        st.info("No completed games yet. Finish a game to see its history here.")

    st.divider()

    # --- Quick Actions ---
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🎮 New Game", type="primary", use_container_width=True):
            st.switch_page("app.py")
    with col2:
        # Continue game if one is in progress
        if st.session_state.get("game_configured") and st.session_state.get("game_phase") not in ["setup", "game_over"]:
            if st.button("▶️ Continue Game", use_container_width=True):
                st.switch_page("pages/1_Game.py")


if __name__ == "__main__":
    main()
