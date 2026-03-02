# Bluff Game Architecture

This document describes how the core components interact in the Bluff (Liar's Dice) simulation.

## Component Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        BluffEnv (gym_env/)                      │
│                   PettingZoo AECEnv wrapper                     │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                     BluffGame (game/)                     │  │
│  │                    Game engine/rules                      │  │
│  │  ┌─────────────────────────────────────────────────────┐  │  │
│  │  │                    GameState                        │  │  │
│  │  │  ┌─────────┐ ┌─────────┐ ┌─────────┐              │  │  │
│  │  │  │ Player  │ │ Player  │ │ Player  │  ...         │  │  │
│  │  │  │ (seat 0)│ │ (seat 1)│ │ (seat 2)│              │  │  │
│  │  │  └─────────┘ └─────────┘ └─────────┘              │  │  │
│  │  └─────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
        ▲                                           ▲
        │ select_action()                           │ step(action)
        │ observe()                                 │
┌───────┴───────┐                         ┌────────┴────────┐
│   BaseAgent   │                         │  External RL    │
│ (agents/)     │                         │  Algorithm      │
│ - RandomAgent │                         │  (e.g., PPO)    │
└───────────────┘                         └─────────────────┘
```

## Core Game Components (`bluff/game/`)

### Player
A frozen dataclass representing a player's state at a point in time.

```python
@dataclass(frozen=True)
class Player:
    seat: int              # Table position (0, 1, 2, ...)
    num_dice: int          # Current dice count
    dice: Tuple[int, ...]  # Current dice values (hidden from others)
```

- **Immutable**: Each modification (roll, lose_die) returns a new Player instance
- **seat**: Fixed table position within a game
- **is_active**: True if player has dice remaining

### GameState
Immutable snapshot of the complete game at any moment.

```python
@dataclass(frozen=True)
class GameState:
    players: Tuple[Player, ...]  # Indexed by seat
    current_seat: int            # Whose turn
    current_bid: Optional[Bid]   # Current bid on table
    bidder_seat: Optional[int]   # Who made the bid
    round_number: int
    is_game_over: bool
    winner_seat: Optional[int]
```

### BluffGame
The game engine that enforces rules and produces state transitions.

```python
class BluffGame:
    def reset() -> GameState           # Start new game
    def get_valid_actions(state) -> List[Action]
    def is_valid_action(state, action) -> bool
    def step(state, action) -> (GameState, Optional[RoundResult])
```

- **Stateless**: Takes state in, returns new state out
- **Deterministic**: Same state + action = same result (except dice rolls)

## Game Flow

### Terminology
- **Game**: Complete match from start until one player remains
- **Round**: Sequence of bids ending with a CALL; loser loses one die
- **Turn**: Single player's action (BID or CALL)

### Round Flow
```
1. All players roll dice (hidden)
2. Round starter makes first bid
3. Bidding continues clockwise:
   - Each player must BID higher OR CALL
   - BID: claim at least N dice show face F
   - "Higher" means: more dice, OR same dice + higher face
4. When someone CALLs:
   - Reveal all dice
   - Count matching dice
   - If bid true: caller loses a die
   - If bid false: bidder loses a die
5. Winner starts next round
6. Player with 0 dice is eliminated
7. Last player standing wins the game
```

## Agent System (`bluff/agents/`)

### Three-Tier Identity System
```
policy_id  : Permanent identifier for the policy/model (e.g., "dqn_v2_trained")
player_id  : Session-visible identifier (defaults to policy_id)
seat       : Table position for current game (0, 1, 2, ...)
```

This separation enables:
- Same policy playing multiple seats across games
- Opponent modeling across sessions (track by player_id)
- Seat-based game logic (turn order, etc.)
- Self-play: same policy_id can fill multiple seats with different player_ids

### Identity Mapping for Training

BluffEnv (PettingZoo) uses seat-based agent names: `player_0`, `player_1`, etc.
These are **positional**, not persistent identities.

Training code maintains two mappings:

1. **seat → player_id**: Passed to BluffEnv for opponent modeling (stats persist by player_id)
2. **player_id → policy_id**: Managed by learner for routing experiences

```python
# Define mappings
seat_to_player_id = {0: "alice", 1: "bob", 2: "charlie"}
player_id_to_policy_id = {"alice": "dqn_v2", "bob": "ppo_v1", "charlie": "dqn_v2"}

# Pass player_ids to BluffEnv for opponent modeling
env.reset(options={"seat_to_player_id": seat_to_player_id})

# Route rewards from BluffEnv (seat-based) to policy buffers
for seat_name, reward in env.rewards.items():
    seat = int(seat_name.split("_")[1])  # "player_0" -> 0
    player_id = seat_to_player_id[seat]  # 0 -> "alice"
    policy_id = player_id_to_policy_id[player_id]  # "alice" -> "dqn_v2"
    replay_buffers[policy_id].add_reward(reward)

# Next game: rotate seats, keep player_ids
seat_to_player_id = {0: "bob", 1: "charlie", 2: "alice"}
```

The relationship:
```
BluffEnv          seat_to_player_id      player_id_to_policy_id
─────────         ─────────────────      ──────────────────────
player_0  ──seat 0──► "alice" ──────────► policy_id "dqn_v2"
player_1  ──seat 1──► "bob"   ──────────► policy_id "ppo_v1"
player_2  ──seat 2──► "charlie" ────────► policy_id "dqn_v2" (self-play!)
              │
              └── Stats tracked by player_id (persists across games)
```

### BaseAgent
Abstract base class for all agents.

```python
class BaseAgent(ABC):
    def __init__(policy_id, player_id=None, seed=None)
    def set_seat(seat)  # Called before each game

    @abstractmethod
    def select_action(observation, valid_actions) -> Action

    # Optional callbacks
    def on_game_start(seat_to_player_id)
    def on_round_end(revealed_dice, result, seat_to_player_id)
    def on_game_end(winner_seat, seat_to_player_id)
```

### Implemented Agents

| Agent | Description | Key Parameters |
|-------|-------------|----------------|
| `RandomAgent` | Uniform random action selection | `seed` |
| `HeuristicAgent` | Probability-based decisions using binomial distribution | `call_threshold`, `bid_aggression`, `bluff_probability` |
| `AdaptiveAgent` | Extends Heuristic with opponent modeling | `opponent_trust`, `cold_start_rounds` |

**HeuristicAgent** calculates P(bid is true) using:
```
P(≥k dice show face j) = 1 - BinomialCDF(k-1, unknown_dice, 1/num_faces)
```
- Calls when P(bid true) < `call_threshold`
- Bids based on expected count ± `bid_aggression`
- Random bluffs with probability `bluff_probability`

**AdaptiveAgent** adjusts call threshold based on opponent bluff rates:
- Tracks opponent bluff rates via `on_round_end` callbacks
- High bluff rate → increase call threshold (call more often)
- Uses `cold_start_rounds` to avoid trusting early stats

### Agent Performance (Tournament Results)

Run `python -m scripts.agent_tournament` to reproduce.

**Heuristic Variants (500 games):**
| Agent | call_threshold | bid_aggression | Win Rate |
|-------|----------------|----------------|----------|
| conservative | 0.4 | -0.2 | ~43% |
| neutral | 0.3 | 0.0 | ~29% |
| aggressive | 0.2 | 0.3 | ~28% |

**Agent Types Comparison (500 games):**
| Agent | Win Rate |
|-------|----------|
| AdaptiveAgent | ~50% |
| HeuristicAgent | ~48% |
| RandomAgent | ~2% |

**Key Observations:**
1. Conservative play (higher call_threshold) outperforms aggressive
2. Probability-based agents vastly outperform random (~50x better)
3. Opponent modeling provides marginal benefit over pure heuristics

### PlayerObservation
What an agent sees (excludes other players' dice):

```python
@dataclass
class PlayerObservation:
    seat: int                    # My position
    own_dice: Tuple[int, ...]    # My dice values
    other_seats_dice_counts: ... # Others' dice counts (not values!)
    current_bid: Optional[Bid]
    bidder_seat: Optional[int]
    active_seats: Tuple[int, ...]
    total_dice: int
```

## Multi-Agent Environment (`bluff/gym_env/`)

### BluffEnv (PettingZoo AECEnv)
Wraps BluffGame for reinforcement learning.

```python
class BluffEnv(AECEnv):
    def __init__(
        num_players=3,
        dice_per_player=5,
        reward_config=RewardConfig(),  # Or .sparse(), .round_based(), .dense()
        track_stats=True,              # Enable opponent modeling
    )

    def reset(seed=None)
    def step(action: int)      # Discrete action
    def observe(agent) -> dict # Hierarchical observation
    def render()
```

### Action Space
Discrete space: `max_dice * num_faces + 1` actions

```
Encoding:
- BID(count, face): action_id = (count-1) * num_faces + (face-1)
- CALL: action_id = max_dice * num_faces

Example (3 players, 3 dice each, 6 faces):
- 54 bid actions (1-9 dice × 6 faces)
- 1 call action
- Total: 55 actions
- Invalid actions masked via observation["action_mask"]
```

### Observation Space
Hierarchical dictionary:

```python
{
    "private": {
        "dice_counts": [2, 0, 1, 0, 0, 1]  # 2×1s, 1×3, 1×6
    },
    "public_round": {
        "dice_per_seat": [3, 2, 3],
        "active_mask": [1, 1, 1],
        "current_bid": [4, 3],  # 4×3s
        "bid_exists": 1,
        "my_seat": 0,
        "current_seat": 1,
        "seat_to_player_idx": [0, 1, 2],
    },
    "public_player": {  # Per-player stats for opponent modeling
        "rounds_played": [...],
        "bluff_rate": [...],
        "call_rate": [...],
        "aggression": [...],
    },
    "action_mask": [0, 0, 1, 1, ...]  # Valid actions
}
```

### Reward Configuration
Use `RewardConfig` to configure rewards. Presets available:

```python
RewardConfig.sparse()      # Game-end only: +1 win, -1 elimination (default)
RewardConfig.round_based() # + round rewards: +0.1 round win, -0.1 round loss
RewardConfig.dense()       # + step rewards: call outcomes, survival bonuses
```

Custom configuration:
```python
RewardConfig(
    game_win=1.0, elimination=-1.0,           # Layer 1: Game-level
    round_win=0.1, round_loss=-0.1,           # Layer 2: Round-level
    successful_call=0.05, failed_call=-0.05,  # Layer 3: Step-level
)
```

### Stats Tracking
`StatsTracker` maintains per-player statistics across episodes:

```python
class PlayerStats:
    rounds_played: int     # For cold-start detection
    bluff_rate: float      # Fraction of called bids that were false
    call_rate: float       # How often they call vs raise
    aggression: float      # Bid amount vs expected
```

Aggression metric:
```
expected = own_matching_dice + (unknown_dice / num_faces)
aggression = (bid_count - expected) / expected
```

## Usage Patterns

### Direct Game Simulation (with BaseAgent)
```python
from bluff.game import BluffGame
from bluff.agents import RandomAgent
from bluff.run_game import run_game

game = BluffGame(num_players=3, dice_per_player=3)
agents = [RandomAgent(seed=i) for i in range(3)]
winner = run_game(game, agents, verbose=True)
```

### RL Training (with BluffEnv)
```python
from bluff.gym_env import BluffEnv

env = BluffEnv(num_players=3, reward_mode="round")
env.reset(seed=42)

for agent in env.agent_iter():
    obs, reward, term, trunc, info = env.last()
    if term or trunc:
        action = None
    else:
        mask = obs["action_mask"]
        action = my_policy(obs, mask)
    env.step(action)
```

For detailed guidance on reward accumulation, replay buffer handling, and proper bootstrapping in the training loop, see [RL_TRAINING_LOOP.md](RL_TRAINING_LOOP.md).

## File Structure
```
bluff/                        # Importable package
├── game/                     # Core game logic
│   ├── types.py              # Bid, Action, ActionType, RoundResult
│   ├── player.py             # Player dataclass
│   ├── game_state.py         # GameState, PlayerObservation
│   └── game.py               # BluffGame engine
├── gym_env/                  # PettingZoo wrapper
│   ├── bluff_env.py          # BluffEnv(AECEnv)
│   ├── spaces.py             # Observation/action space helpers
│   └── stats.py              # PlayerStats, StatsTracker
├── agents/                   # Agent implementations
│   ├── base.py               # BaseAgent ABC
│   ├── random_agent.py       # RandomAgent
│   ├── heuristic_agent.py    # HeuristicAgent (probability-based)
│   └── adaptive_agent.py     # AdaptiveAgent (opponent modeling)
├── rl/                       # RL training components
│   ├── dqn_policy.py         # DQN policy and network
│   ├── replay_buffer.py      # Experience replay buffer
│   └── obs_utils.py          # Observation flattening utilities
└── run_game.py               # Simple game runner
scripts/
├── train_dqn.py              # DQN training script
├── agent_tournament.py       # Run tournaments between agents
└── test_wrappers.py          # Wrapper integration tests
```

## Streamlit App Stats Tracking (`pages/`)

### Current Implementation

The Streamlit app tracks statistics by **agent_type** (e.g., "Heuristic", "DQN (trained)", "Human"):

```python
# Tracked per agent_type
agent_stats = {
    "Heuristic": {
        "wins": 5, "games": 10,
        "total_bids": 47, "total_calls": 12,
        "successful_calls": 8, "failed_calls": 4,
        "bid_ratios": [0.33, 0.40, ...],  # For aggressiveness
    },
    "DQN (trained)": {...},
}
```

This aggregates all instances of the same type, which is useful for comparing agent classes.

### Identity Model Considerations

The app simplifies the full identity model:

```
Full model:            App's simplification:
─────────────          ─────────────────────
seat                   seat (used for game logic)
  ↓                      ↓
player_id              (skipped - not tracked)
  ↓                      ↓
policy_id              agent_type (all instances pooled)
```

**Current limitations:**
- Two Heuristic agents in the same game both contribute to "Heuristic" stats
- Can't track individual policy instances across sessions
- Can't analyze seat position effects on performance

**Future enhancements could include:**

1. **Track by policy_id**: Each agent instance gets its own stats
   ```python
   # Instead of agent_type, use policy_id from agent creation
   policy_id = f"{agent_type}_{seat}_{game_id}"
   ```

2. **Track by (agent_type, seat)**: Analyze positional effects
   ```python
   stats_key = f"{agent_type}_seat{seat}"
   ```

3. **Keep raw logs, aggregate on demand**: Store action_log and round_results_log
   persistently, compute stats views dynamically in the Stats page

4. **Use the engine's player_id**: If agents had persistent player_ids across games,
   track stats by player_id for true opponent modeling

The raw action logs already capture `seat`, so adding `policy_id` to `log_action()` would
enable these more granular aggregations without breaking existing functionality.
