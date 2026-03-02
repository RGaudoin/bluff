"""PettingZoo AECEnv wrapper for Bluff (Liar's Dice)."""

from typing import Any, Dict, List, Optional

import numpy as np
from pettingzoo import AECEnv
from pettingzoo.utils.agent_selector import AgentSelector

from bluff.game.game import BluffGame
from bluff.game.game_state import GameState
from bluff.game.types import Action, ActionType, Bid
from bluff.gym_env.rewards import RewardConfig
from bluff.gym_env.stats import StatsTracker
from bluff.gym_env.spaces import (
    create_observation_space,
    create_action_space,
    decode_action,
    get_action_mask,
)


class BluffEnv(AECEnv):
    """
    PettingZoo AECEnv for Bluff (Liar's Dice).

    This is a turn-based, imperfect information game where players
    make bids about the total dice showing a certain face value
    across all players' hidden dice.

    Attributes:
        metadata: Environment metadata including render modes
    """

    metadata = {
        "render_modes": ["human", "ansi"],
        "name": "bluff_v0",
        "is_parallelizable": False,
    }

    def __init__(
        self,
        num_players: int = 3,
        dice_per_player: int = 5,
        num_faces: int = 6,
        reward_config: Optional[RewardConfig] = None,
        track_stats: bool = True,
        max_tracked_players: int = 32,
        render_mode: Optional[str] = None,
    ):
        """
        Initialize the Bluff environment.

        Args:
            num_players: Number of players (2+)
            dice_per_player: Starting dice per player
            num_faces: Faces per die (default 6)
            reward_config: RewardConfig instance for composable rewards.
                          Defaults to RewardConfig() (sparse rewards).
                          Use RewardConfig.sparse(), .round_based(), or .dense() presets.
            track_stats: Whether to track player statistics
            max_tracked_players: Max unique players to track
            render_mode: "human", "ansi", or None
        """
        super().__init__()

        self.num_players = num_players
        self.dice_per_player = dice_per_player
        self.num_faces = num_faces
        self.track_stats = track_stats
        self.max_tracked_players = max_tracked_players
        self.render_mode = render_mode

        # Reward configuration (default to sparse)
        self._reward_config = reward_config if reward_config is not None else RewardConfig()

        # Maximum dice across all players
        self.max_dice = num_players * dice_per_player

        # Create underlying game engine
        self._game = BluffGame(
            num_players=num_players,
            dice_per_player=dice_per_player,
            num_faces=num_faces,
        )

        # Agent names (player_0, player_1, ...)
        self.possible_agents = [f"player_{i}" for i in range(num_players)]

        # Create observation and action spaces
        self._observation_space = create_observation_space(
            num_players=num_players,
            num_faces=num_faces,
            max_dice=self.max_dice,
            max_tracked_players=max_tracked_players,
        )
        self._action_space = create_action_space(
            max_dice=self.max_dice,
            num_faces=num_faces,
        )

        # Stats tracker for opponent modeling
        if track_stats:
            self._stats_tracker = StatsTracker(max_players=max_tracked_players)
        else:
            self._stats_tracker = None

        # Will be set in reset()
        self._state: Optional[GameState] = None
        self.agents: List[str] = []
        self._agent_selector: Optional[AgentSelector] = None
        self.agent_selection: str = ""
        # Maps seat -> player_id for opponent modeling (set in reset)
        self._seat_to_player_id: Dict[int, str] = {}

        # Reward/termination tracking per agent
        self.rewards: Dict[str, float] = {}
        self.terminations: Dict[str, bool] = {}
        self.truncations: Dict[str, bool] = {}
        self.infos: Dict[str, Dict] = {}

        # Cumulative rewards
        self._cumulative_rewards: Dict[str, float] = {}

    def observation_space(self, agent: str):
        """Get observation space for an agent."""
        return self._observation_space

    def action_space(self, agent: str):
        """Get action space for an agent."""
        return self._action_space

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Reset the environment to start a new game.

        Args:
            seed: Random seed for reproducibility
            options: Additional options:
                - seat_to_player_id: Dict[int, str] mapping seats to player IDs
                  for opponent modeling. Stats are tracked by player_id, enabling
                  persistent modeling across games even when players change seats.
                  Defaults to seat-based names ("player_0", "player_1", ...).
        """
        if seed is not None:
            self._game.rng = np.random.default_rng(seed)

        # Reset game state
        self._state = self._game.reset()

        # Set up seat -> player_id mapping for opponent modeling
        # If not provided, default to seat-based names (backward compatible)
        if options and "seat_to_player_id" in options:
            self._seat_to_player_id = options["seat_to_player_id"]
        else:
            self._seat_to_player_id = {
                i: f"player_{i}" for i in range(self.num_players)
            }

        # Reset agent tracking
        # AgentSelector kept for PettingZoo API compatibility, but actual turn
        # order is driven by game engine's current_seat (see _sync_agent_selection)
        self.agents = list(self.possible_agents)
        self._agent_selector = AgentSelector(self.agents)
        self.agent_selection = self._agent_selector.reset()

        # Reset rewards and termination flags for ALL possible agents
        self.rewards = {agent: 0.0 for agent in self.possible_agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.possible_agents}
        self.terminations = {agent: False for agent in self.possible_agents}
        self.truncations = {agent: False for agent in self.possible_agents}
        self.infos = {agent: {} for agent in self.possible_agents}

        # Ensure stats are set up for all player_ids
        if self._stats_tracker:
            for seat in range(self.num_players):
                player_id = self._get_player_id(seat)
                self._stats_tracker.get_or_create(player_id)

    def step(self, action: int) -> None:
        """
        Take a step in the environment.

        Args:
            action: Action from the discrete action space
        """
        agent = self.agent_selection

        # Check for terminated/truncated agent
        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            return

        # Clear cumulative rewards for agent taking action (they're about to act)
        self._cumulative_rewards[agent] = 0.0

        # Clear step rewards for all agents
        self.rewards = {a: 0.0 for a in self.possible_agents}

        # Decode and validate action
        action_type, count, face = decode_action(
            action, self.max_dice, self.num_faces
        )
        seat = self._agent_to_seat(agent)

        # Build game Action
        if action_type == "call":
            game_action = Action(ActionType.CALL, None, seat)
        else:
            game_action = Action(ActionType.BID, Bid(count, face), seat)

        # Validate action - if invalid, penalize and play random valid action
        if not self._game.is_valid_action(self._state, game_action):
            self.rewards[agent] = self._reward_config.invalid_action

            # Get valid actions and sample one randomly
            action_mask = get_action_mask(
                current_bid=(
                    (self._state.current_bid.count, self._state.current_bid.face_value)
                    if self._state.current_bid
                    else (0, 0)
                ),
                total_dice=self._state.total_dice,
                num_faces=self.num_faces,
                max_dice=self.max_dice,
            )
            valid_actions = np.flatnonzero(action_mask)
            action = self._game.rng.choice(valid_actions)

            # Decode the random valid action
            action_type, count, face = decode_action(
                action, self.max_dice, self.num_faces
            )
            if action_type == "call":
                game_action = Action(ActionType.CALL, None, seat)
            else:
                game_action = Action(ActionType.BID, Bid(count, face), seat)

        # Record stats by player_id (not seat) for cross-game opponent modeling
        player_id = self._get_player_id(seat)
        if self._stats_tracker and action_type == "bid":
            aggression = self._compute_aggression(agent, count, face)
            self._stats_tracker.get_or_create(player_id).record_bid(aggression)
        elif self._stats_tracker and action_type == "call":
            self._stats_tracker.get_or_create(player_id).record_call()

        # Capture bidder_seat and caller_seat before state transition
        # (bidder_seat is cleared in new round state after a CALL)
        old_bidder_seat = self._state.bidder_seat
        caller_seat = seat if action_type == "call" else None

        # Execute action
        new_state, round_result = self._game.step(self._state, game_action)
        self._state = new_state

        # Process round result if round ended
        if round_result is not None:
            self._process_round_result(round_result, old_bidder_seat, caller_seat)

        # Check for game over
        if self._state.is_game_over:
            self._process_game_over()
        else:
            # Update active agents
            self._update_agents()

        # Accumulate rewards for all agents
        self._accumulate_rewards()

        # Sync agent selection with game state
        self._sync_agent_selection()

    def _accumulate_rewards(self) -> None:
        """Add step rewards to cumulative rewards for all agents."""
        for agent in self.possible_agents:
            self._cumulative_rewards[agent] += self.rewards[agent]

    def _sync_agent_selection(self) -> None:
        """Sync agent_selection with game engine's current_seat."""
        if not self.agents or self._state.is_game_over:
            return

        current_seat = self._state.current_seat
        self.agent_selection = self._seat_to_agent(current_seat)

    def observe(self, agent: str) -> Dict[str, Any]:
        """
        Get observation for an agent.

        Args:
            agent: Agent name

        Returns:
            Observation dictionary matching observation_space
        """
        if self._state is None:
            raise RuntimeError("Environment not reset")

        seat = self._agent_to_seat(agent)
        player = self._state.players[seat]

        # Private: own dice counts
        dice_counts = np.zeros(self.num_faces, dtype=np.int32)
        for die in player.dice:
            dice_counts[die - 1] += 1

        # Public round info
        dice_per_seat = np.array(
            [p.num_dice for p in self._state.players], dtype=np.int32
        )
        active_mask = np.array(
            [int(p.is_active) for p in self._state.players], dtype=np.int8
        )

        if self._state.current_bid:
            current_bid = np.array(
                [self._state.current_bid.count, self._state.current_bid.face_value],
                dtype=np.int32,
            )
            bid_exists = 1
        else:
            current_bid = np.array([0, 0], dtype=np.int32)
            bid_exists = 0

        # Seat to player index mapping (maps seat -> stats index via player_id)
        seat_to_player_idx = np.zeros(self.num_players, dtype=np.int32)
        if self._stats_tracker:
            for s in range(self.num_players):
                player_id = self._get_player_id(s)
                seat_to_player_idx[s] = self._stats_tracker.get_player_idx(player_id)

        # Public player stats arrays (indexed by player stats index)
        rounds_played = np.zeros(self.max_tracked_players, dtype=np.int32)
        dice_remaining = np.zeros(self.max_tracked_players, dtype=np.int32)
        bluff_rate = np.zeros(self.max_tracked_players, dtype=np.float32)
        call_rate = np.zeros(self.max_tracked_players, dtype=np.float32)
        aggression = np.zeros(self.max_tracked_players, dtype=np.float32)

        if self._stats_tracker:
            # Fill stats for players at current seats
            for s in range(self.num_players):
                player_id = self._get_player_id(s)
                idx = self._stats_tracker.get_player_idx(player_id)
                stats = self._stats_tracker.get_or_create(player_id)
                rounds_played[idx] = stats.rounds_played
                bluff_rate[idx] = stats.bluff_rate
                call_rate[idx] = stats.call_rate
                aggression[idx] = stats.aggression
                dice_remaining[idx] = self._state.players[s].num_dice

        # Action mask
        action_mask = get_action_mask(
            current_bid=(
                (self._state.current_bid.count, self._state.current_bid.face_value)
                if self._state.current_bid
                else (0, 0)
            ),
            total_dice=self._state.total_dice,
            num_faces=self.num_faces,
            max_dice=self.max_dice,
        )

        return {
            "private": {
                "dice_counts": dice_counts,
            },
            "public_round": {
                "dice_per_seat": dice_per_seat,
                "active_mask": active_mask,
                "current_bid": current_bid,
                "bid_exists": bid_exists,
                "my_seat": seat,
                "current_seat": self._state.current_seat,
                "seat_to_player_idx": seat_to_player_idx,
                "round_number": self._state.round_number,
            },
            "public_player": {
                "rounds_played": rounds_played,
                "dice_remaining": dice_remaining,
                "bluff_rate": bluff_rate,
                "call_rate": call_rate,
                "aggression": aggression,
            },
            "action_mask": action_mask,
        }

    def render(self) -> Optional[str]:
        """
        Render the current game state.

        Returns:
            String representation if render_mode is "ansi", else None
        """
        if self._state is None:
            return None

        if self.render_mode == "ansi":
            return str(self._state)
        elif self.render_mode == "human":
            print(self._state)
            return None
        return None

    def close(self) -> None:
        """Clean up resources."""
        pass

    # Helper methods

    def _agent_to_seat(self, agent: str) -> int:
        """Convert agent name to seat index."""
        return int(agent.split("_")[1])

    def _seat_to_agent(self, seat: int) -> str:
        """Convert seat index to agent name."""
        return f"player_{seat}"

    def _get_player_id(self, seat: int) -> str:
        """Get player_id for a seat (for opponent modeling/stats)."""
        return self._seat_to_player_id.get(seat, f"player_{seat}")

    def _compute_aggression(self, agent: str, count: int, face: int) -> float:
        """
        Compute aggression score for a bid.

        aggression = (bid_count - expected) / expected
        where expected = own_matching + unknown_dice / num_faces
        """
        seat = self._agent_to_seat(agent)
        player = self._state.players[seat]

        # Count own dice matching this face
        own_matching = player.count_face(face)

        # Unknown dice (others' dice)
        unknown_dice = self._state.total_dice - player.num_dice

        # Expected count
        expected = own_matching + unknown_dice / self.num_faces

        if expected == 0:
            return float(count)  # Avoid division by zero

        return (count - expected) / expected

    def _process_round_result(
        self, result, bidder_seat: Optional[int], caller_seat: Optional[int] = None
    ) -> None:
        """
        Process end of round, assign rewards, update stats.

        Args:
            result: RoundResult from game engine
            bidder_seat: Seat of the player who made the bid that was called
            caller_seat: Seat of the player who called (current player when CALL happened)
        """
        winner_seat = result.winner_seat
        loser_seat = result.loser_seat
        winner_agent = self._seat_to_agent(winner_seat)
        loser_agent = self._seat_to_agent(loser_seat)
        rc = self._reward_config

        # Update stats by player_id (not seat) for cross-game modeling
        if self._stats_tracker:
            # Record that bidder's bid was called (bidder_seat passed from caller
            # since it's cleared in the new state after a CALL)
            if bidder_seat is not None:
                bidder_player_id = self._get_player_id(bidder_seat)
                self._stats_tracker.get_or_create(bidder_player_id).record_bid_called(
                    not result.bid_was_true
                )

            # Record round end for all active players
            for seat in self._state.active_seats:
                player_id = self._get_player_id(seat)
                self._stats_tracker.get_or_create(player_id).record_round_end()

        # === Layer 2: Round-level rewards ===
        self.rewards[winner_agent] += rc.round_win
        self.rewards[loser_agent] += rc.round_loss

        # Survival reward for observers (multi-player games, N > 2)
        if rc.survive_round != 0:
            for seat in self._state.active_seats:
                if seat != winner_seat and seat != loser_seat:
                    agent = self._seat_to_agent(seat)
                    self.rewards[agent] += rc.survive_round

        # Dice fraction delta reward (non-linear by game phase)
        # Loser lost 1 die, so we can reconstruct before/after fractions
        if rc.dice_fraction_scale != 0:
            total_after = self._state.total_dice
            total_before = total_after + 1  # One die was just lost

            for seat in self._state.active_seats:
                agent = self._seat_to_agent(seat)
                dice_after = self._state.players[seat].num_dice

                # Reconstruct dice_before (loser had +1 die)
                if seat == loser_seat:
                    dice_before = dice_after + 1
                else:
                    dice_before = dice_after

                # Delta in fraction
                frac_before = dice_before / total_before
                frac_after = dice_after / total_after
                delta = frac_after - frac_before

                self.rewards[agent] += rc.dice_fraction_scale * delta

            # Also give reward to eliminated player (loser with 0 dice)
            # They went from dice_before/total_before to 0
            if self._state.players[loser_seat].num_dice == 0:
                dice_before = 1  # They had 1 die before losing it
                frac_before = dice_before / total_before
                delta = 0 - frac_before  # Fraction goes to 0
                self.rewards[loser_agent] += rc.dice_fraction_scale * delta

        # === Layer 3: Step-level rewards (call outcomes) ===
        # Caller is the one who made the CALL action
        # Bidder is the one whose bid was called
        if caller_seat is not None and bidder_seat is not None:
            caller_agent = self._seat_to_agent(caller_seat)
            bidder_agent = self._seat_to_agent(bidder_seat)

            if result.bid_was_true:
                # Bid was true: caller loses (failed call), bidder survives
                self.rewards[caller_agent] += rc.failed_call
                self.rewards[bidder_agent] += rc.got_called_truthful
            else:
                # Bid was false (bluff): caller wins (successful call), bidder caught
                self.rewards[caller_agent] += rc.successful_call
                self.rewards[bidder_agent] += rc.got_called_bluffing

    def _process_game_over(self) -> None:
        """Process game over, assign final rewards."""
        winner_seat = self._state.winner_seat
        winner_agent = self._seat_to_agent(winner_seat)
        rc = self._reward_config

        # Assign game-end rewards
        for agent in self.possible_agents:
            if agent == winner_agent:
                self.rewards[agent] += rc.game_win
            elif not self.terminations[agent]:
                # Final loser (just eliminated) - they didn't go through _update_agents
                # Already-terminated agents got their penalty earlier
                self.rewards[agent] += rc.elimination

            self.terminations[agent] = True

        # Clear agents list
        self.agents = []

    def _update_agents(self) -> None:
        """Update list of active agents based on game state."""
        active_seats = set(self._state.active_seats)
        newly_eliminated = []

        for agent in list(self.agents):
            seat = self._agent_to_seat(agent)
            if seat not in active_seats:
                newly_eliminated.append(agent)
                self.terminations[agent] = True
                self.rewards[agent] += self._reward_config.elimination

        for agent in newly_eliminated:
            self.agents.remove(agent)

        # Update selector with remaining agents
        if self.agents:
            self._agent_selector = AgentSelector(self.agents)
            self._agent_selector.reset()

    def _was_dead_step(self, action: int) -> None:
        """Handle step for terminated agent (PettingZoo requirement)."""
        self._cumulative_rewards[self.agent_selection] = 0
        self._sync_agent_selection()
