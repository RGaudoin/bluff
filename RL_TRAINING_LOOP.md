# RL Training Loop for BluffEnv

This document explains the PettingZoo AEC (Agent Environment Cycle) training loop, focusing on reward accumulation and proper replay buffer handling for multi-agent turn-based games.

## The Core Challenge

In single-agent RL, rewards are immediate: agent acts, environment returns reward. In multi-agent turn-based games like Bluff:

1. Agent A acts
2. Agents B, C, D take their turns
3. A round might end, assigning rewards to A
4. Eventually it's A's turn again

The reward for A's action is **delayed** — it arrives while other agents act. How do we ensure A receives the correct reward for their action?

## Cumulative vs Step Rewards

BluffEnv maintains two reward structures:

```python
self.rewards: Dict[str, float]              # Immediate step rewards
self._cumulative_rewards: Dict[str, float]  # Accumulated since last action
```

### Flow Through a Turn

```
1. Agent A's turn begins
   └─► _cumulative_rewards["player_0"] = 0.0  (cleared - line 208)

2. A takes action, game advances
   └─► rewards assigned (e.g., round ends, A wins → rewards["player_0"] = +0.1)

3. _accumulate_rewards() called (line 279)
   └─► _cumulative_rewards["player_0"] += rewards["player_0"]

4. Turns pass: B acts, C acts...
   └─► More rewards may accrue for A during these steps
   └─► Each step: _cumulative_rewards[A] += rewards[A]

5. A's turn comes again
   └─► env.last() returns _cumulative_rewards["player_0"]
   └─► This is the TOTAL reward for A's previous action
   └─► Then cleared, cycle repeats
```

## The Training Loop

```python
env.reset(seed=42, options={"seat_to_player_id": seat_to_player_id})

# Track previous state/action per agent for replay buffer
prev_obs = {}
prev_action = {}

for agent in env.agent_iter():
    obs, reward, term, trunc, info = env.last()

    # reward is cumulative since this agent's LAST action
    # Store transition for PREVIOUS (state, action) pair
    if agent in prev_obs:
        replay_buffer.add(
            agent=agent,
            obs=prev_obs[agent],
            action=prev_action[agent],
            reward=reward,           # Reward for previous action
            next_obs=obs,
            terminated=term,         # Store separately!
            truncated=trunc,         # Store separately!
        )

    if term or trunc:
        action = None
    else:
        action = policy[agent].select_action(obs)

    # Store current for next iteration
    prev_obs[agent] = obs
    prev_action[agent] = action

    env.step(action)
```

### Key Insight: Reward Attribution

When `last()` returns `reward`, it's the consequence of what the agent did **last time**, not what they're about to do. The training loop must:

1. Call `last()` → get `(obs, reward, term, trunc, info)`
2. Log `reward` with the **previous** `(state, action)` pair
3. Choose new action based on current `obs`
4. Store current `(obs, action)` for the next iteration

## Dynamic Agent Ordering

`agent_iter()` yields agents based on game state, not a fixed order:

```python
for agent in env.agent_iter():  # Order determined dynamically
    ...
```

In Bluff:
- Turn order follows clockwise from round starter
- Round loser starts the next round
- Eliminated players are skipped
- `agent_selection` updates via `_sync_agent_selection()` after each step

The iterator yields whoever `env.agent_selection` points to until the game ends.

## Policy Per Agent

Different agents may use different policies:

```python
# Separate policies per agent
policies = {
    "player_0": DQNPolicy(...),
    "player_1": PPOPolicy(...),
    "player_2": DQNPolicy(...),  # Could share weights with player_0
}
action = policies[agent].select_action(obs)

# Or shared policy with agent context
action = shared_policy.select_action(obs, policy_id=agent)

# Or self-play (same policy, different seats)
action = policy.select_action(obs)
```

With the identity mapping system (see ARCHITECTURE.md), you route by `player_id` or `policy_id`:

```python
seat = int(agent.split("_")[1])           # "player_0" → 0
player_id = seat_to_player_id[seat]       # 0 → "alice"
policy_id = player_id_to_policy_id[player_id]  # "alice" → "dqn_v2"
action = policies[policy_id].select_action(obs)
```

## Why player_id Exists: Cross-Game Opponent Modeling

At first glance, `player_id` may seem redundant — why not just use seat or policy_id? The answer: **opponent modeling stats must persist across games even when players switch seats**.

### The Problem

Without `player_id`:
- Game 1: Alice sits at seat 0, builds up bluff_rate=0.4
- Game 2: Alice moves to seat 1
- Her stats are lost (tied to seat 0) or incorrectly attributed to whoever now sits at seat 0

### The Solution

Stats are tracked by `player_id`, not seat. The observation includes:

```python
"public_round": {
    ...
    "seat_to_player_idx": [0, 2, 1],  # seat → stats array index
},
"public_player": {
    "rounds_played": [50, 30, 45],    # indexed by player_idx
    "bluff_rate": [0.4, 0.2, 0.35],
    "call_rate": [0.3, 0.5, 0.25],
    "aggression": [0.1, -0.2, 0.05],
}
```

### How the Agent Uses It

To look up the bluff rate of the player at seat 2:

```python
player_idx = obs["public_round"]["seat_to_player_idx"][2]  # e.g., 1
bluff_rate = obs["public_player"]["bluff_rate"][player_idx]  # 0.2
```

This works regardless of seat rotation:
- Game 1: Alice at seat 0 → player_idx 0 → bluff_rate 0.4
- Game 2: Alice at seat 1 → player_idx 0 → bluff_rate 0.4 (same stats follow her)

### Identity Hierarchy

```
seat        : Position at the table (0, 1, 2) — changes between games
player_id   : Persistent identity ("alice") — stats tracked here
policy_id   : Policy/model identifier ("dqn_v2") — for routing to replay buffers
```

Multiple `player_id`s can share a `policy_id` (self-play), and the same `player_id` can occupy different seats across games while retaining their opponent modeling history.

## Termination vs Truncation

These must be stored **separately** in the replay buffer for correct bootstrapping:

| | Terminated | Truncated |
|---|---|---|
| **Meaning** | Natural end (game over, eliminated) | Artificial cutoff (step limit) |
| **Future value** | Zero — no more rewards possible | Non-zero — just unobserved |
| **Bootstrap?** | No | Yes |

### Why It Matters

```python
# During learning - compute TD target
if terminated:
    # Game truly ended, no future value
    target = reward
elif truncated:
    # Episode cut off, estimate continuation value
    target = reward + gamma * value_net(next_obs)
else:
    # Normal transition
    target = reward + gamma * value_net(next_obs)
```

If you combine them (`done = term or trunc`), truncated episodes incorrectly get zero future value — the agent learns "hitting step limit = losing" instead of properly estimating unobserved future rewards.

### Handling Dead Agents

When an agent is terminated (eliminated), they still appear in `agent_iter()` (PettingZoo requirement):

```python
if term or trunc:
    action = None  # Required by PettingZoo
env.step(action)   # Internally calls _was_dead_step()
```

The iterator continues until the game fully ends. `_was_dead_step()` clears the dead agent's cumulative rewards and advances to the next agent.

## Complete Example

```python
from bluff.gym_env import BluffEnv

env = BluffEnv(num_players=3, reward_config=RewardConfig.round_based(), track_stats=True)

# Identity mappings
seat_to_player_id = {0: "alice", 1: "bob", 2: "charlie"}
player_id_to_policy_id = {"alice": "dqn", "bob": "dqn", "charlie": "ppo"}

# Replay buffers per policy_id
buffers = {"dqn": ReplayBuffer(), "ppo": ReplayBuffer()}
policies = {"dqn": DQNPolicy(), "ppo": PPOPolicy()}

env.reset(seed=42, options={"seat_to_player_id": seat_to_player_id})

prev_obs, prev_action = {}, {}

for agent in env.agent_iter():
    obs, reward, term, trunc, info = env.last()

    # Map to policy_id
    seat = int(agent.split("_")[1])
    player_id = seat_to_player_id[seat]
    policy_id = player_id_to_policy_id[player_id]

    # Store transition for previous action
    if agent in prev_obs:
        buffers[policy_id].add(
            obs=prev_obs[agent],
            action=prev_action[agent],
            reward=reward,
            next_obs=obs,
            terminated=term,
            truncated=trunc,
        )

    if term or trunc:
        action = None
    else:
        action = policies[policy_id].select_action(obs)

    prev_obs[agent] = obs
    prev_action[agent] = action
    env.step(action)
```

## Training Architecture: Env vs Game API

The codebase provides two interfaces for interacting with Bluff:

| Interface | Primary Use | Key Characteristics |
|-----------|-------------|---------------------|
| `BluffEnv` (PettingZoo) | RL Training | Numpy arrays, action masks, standard RL library integration |
| `BluffGame` | MCTS, Apps | Immutable states, clean game logic, human-readable types |

### Which to Use When

**RL Training → BluffEnv**
- Observations are numpy arrays ready for neural networks
- Action masks pre-computed for masked policy outputs
- Integrates with Stable Baselines3, RLlib, CleanRL via PettingZoo
- Handles multi-agent turn-taking automatically

**MCTS / Tree Search → BluffGame**
- `GameState` is immutable (frozen dataclass) — perfect for tree nodes
- `game.step(state, action)` returns new state without mutation
- Fast rollouts without numpy conversion overhead
- Easy state comparison and hashing for transposition tables

**Apps / UI → BluffGame**
- Cleaner state display (`Bid`, `Action` have `__str__` methods)
- Human-readable types for rendering
- Simpler debugging and logging

### Bridging with Wrappers

The `wrappers.py` module enables interoperability:

```python
from bluff.agents import wrap_for_env, wrap_for_game

# Train a policy with BluffEnv
trained_policy = train_with_ppo(BluffEnv(...))

# Use it in an app (BluffGame-based)
app_agent = wrap_for_game(trained_policy, "ppo_agent")
winner = run_game(game, [app_agent, heuristic_agent])

# Or use heuristic agents during RL training
heuristic = HeuristicAgent("opponent")
env_policy = wrap_for_env(heuristic)
# Use as opponent in BluffEnv training loop
```

### Self-Play and Population Training

For self-play or league training:

```python
# All agents share the same policy (different seats)
policy = PPOPolicy(...)
env.reset(seed=42)

for agent in env.agent_iter():
    obs, reward, term, trunc, info = env.last()
    if not (term or trunc):
        action = policy.select_action(obs, obs["action_mask"])
    else:
        action = None
    env.step(action)
```

For population-based training with different policies:

```python
# Map seats to different policies
policies = {
    "player_0": current_best_policy,
    "player_1": historical_policy_v1,
    "player_2": historical_policy_v2,
}
```

## Summary

| Concept | Purpose |
|---------|---------|
| `rewards` | Immediate step rewards, cleared each step |
| `_cumulative_rewards` | Accumulates between agent's actions |
| `last()` returns cumulative | Agent receives total reward for previous action |
| Clear on action | Reset accumulator when agent acts |
| Separate term/trunc | Correct bootstrapping in value estimation |
| Dynamic agent order | Game rules determine turn sequence |
| `BluffEnv` | Primary interface for RL training |
| `BluffGame` | Primary interface for MCTS/apps |
| Wrappers | Bridge between the two interfaces |
