# Bluff (Liar's Dice)

[![Licence: Apache 2.0](https://img.shields.io/badge/Licence-Apache_2.0-blue.svg)](LICENCE)

A simulation of the strategic imperfect-information game **Bluff** (also known as Liar's Dice), with AI agents and a Streamlit web app for interactive play.

## Features

- **Game engine** — configurable players (2-6), dice per player, and face count
- **AI agents** — random, probability-based (heuristic), and opponent-modelling (adaptive)
- **DQN training** — Deep Q-Network RL agent trained via PettingZoo AEC environment
- **Streamlit app** — play against AI agents or watch them compete
- **Bidirectional wrappers** — agents work in both the game engine and the PettingZoo env

## Installation

```bash
pip install -e .

# For RL training (requires PyTorch):
pip install -e ".[rl]"
```

## Quick Start

### Play in the browser

```bash
streamlit run app.py
```

### Run an agent tournament

```bash
python -m scripts.agent_tournament
```

### Train a DQN agent

```bash
python -m scripts.train_dqn --num-players 2 --dice-per-player 3 --episodes 5000
```

## Game Rules

- There are *n* players with *m* dice each, sitting in a specified order
- At the start of a round the dice are rolled and stay fixed until the round ends
- Each player only sees their own dice (allowing for bluffing)
- Bidding proceeds in order; a bid claims at least *k* dice of face value *j* exist among all players
- Each bid must be higher than the previous (more dice, or same count with a higher face) — or the player calls
- When a call happens all dice are revealed:
    - If the bid was true the caller loses a die; if false the bidder loses a die
    - The winner starts the next round
- The last player with dice wins

## Agents

| Agent | Description | Key Parameters |
|-------|-------------|----------------|
| `RandomAgent` | Uniform random valid actions | `seed` |
| `HeuristicAgent` | Binomial probability calculations | `call_threshold`, `bid_aggression`, `bluff_probability` |
| `AdaptiveAgent` | Heuristic + opponent bluff-rate tracking | `opponent_trust`, `cold_start_rounds` |
| `DQN (trained)` | Deep Q-Network with action masking | `model_path` |

Conservative play (higher call threshold) consistently outperforms aggressive strategies. Probability-based agents beat random by ~50x. See `scripts/agent_tournament.py` for details.

## Project Structure

```
bluff/                   # Importable package
├── game/                # Core game engine (types, player, game_state, game)
├── agents/              # Agent implementations + factory + wrappers
├── gym_env/             # PettingZoo AECEnv wrapper, spaces, rewards, stats
├── rl/                  # DQN policy, replay buffer, observation utilities
└── run_game.py          # Simple game runner
app.py                   # Streamlit app entry point
pages/                   # Streamlit multi-page app
scripts/                 # Training and tournament scripts
models/                  # Trained model weights (not tracked in git)
```

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) — component design, observation/action spaces, identity system
- [RL_TRAINING_LOOP.md](RL_TRAINING_LOOP.md) — training loop details, reward accumulation, replay buffer handling
- [CHANGELOG_ALGORITHM.md](CHANGELOG_ALGORITHM.md) — algorithm design decisions and experiments

## Design Notes

The observation space supports multiple learning approaches (policy gradient, value-based RL, ISMCTS, CFR, Bayesian inference). The action space uses a flat discrete encoding with validity masks. See [ARCHITECTURE.md](ARCHITECTURE.md) for the full specification.
