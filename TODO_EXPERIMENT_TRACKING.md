# Experiment Tracking & TensorBoard Logging - Implementation Plan

> **Status**: Research complete, ready for implementation
> **Priority**: Medium (nice to have for systematic hyperparameter tuning)
> **Estimated effort**: 1-2 hours for basic setup, 4-6 hours for full integration

## Summary of Options

| Tool | Best For | Cost | Setup | Recommendation |
|------|----------|------|-------|----------------|
| **TensorBoard** | Local debugging, real-time viz | Free | 5 min | Phase 1 |
| **Weights & Biases** | Hyperparameter sweeps, collaboration | Free tier | 5 min | Phase 2 |
| **MLflow** | Full ML lifecycle, deployment | Free (self-host) | Hours | Phase 3 |
| **Neptune.ai** | Large-scale training | Free tier | 10 min | Alternative |
| **CSV/JSON** | Full control, offline | Free | 2 min | Backup |

## Recommended Approach

### Phase 1: TensorBoard (Immediate)

Add basic TensorBoard logging to `train_dqn.py`:

```python
from torch.utils.tensorboard import SummaryWriter

def main():
    # ... existing setup ...

    # Initialize TensorBoard
    run_name = f"dqn_{num_players}p_{dice_per_player}d_{args.reward_preset}"
    writer = SummaryWriter(f'runs/{run_name}')

    # Log hyperparameters
    writer.add_hparams({
        'num_players': num_players,
        'dice_per_player': dice_per_player,
        'learning_rate': learning_rate,
        'gamma': gamma,
        'reward_preset': args.reward_preset,
        'dice_fraction_scale': reward_config.dice_fraction_scale,
        'opponent_ratio': opponent_ratio,
    }, {})

    # In training loop
    if losses:
        writer.add_scalar('Loss/train', np.mean(losses[-100:]), episode)
    writer.add_scalar('WinRate/dqn', dqn_rate, episode)
    writer.add_scalar('WinRate/heuristic', heur_rate, episode)
    writer.add_scalar('Exploration/epsilon', dqn_policy.epsilon, episode)

    # Optional: Q-value distribution
    writer.add_histogram('Q_values', q_values, episode)

    writer.close()
```

**To run TensorBoard:**
```bash
tensorboard --logdir=bluff/runs
# Open http://localhost:6006
```

### Phase 2: Weights & Biases (For Hyperparameter Tuning)

When ready to systematically tune hyperparameters:

```python
import wandb

def main():
    wandb.init(
        project="bluff-dqn",
        config={
            "num_players": num_players,
            "dice_per_player": dice_per_player,
            "num_dqn": num_dqn,
            "learning_rate": learning_rate,
            "gamma": gamma,
            "batch_size": batch_size,
            "reward_preset": args.reward_preset,
            "dice_fraction_scale": reward_config.dice_fraction_scale,
            "game_win": reward_config.game_win,
            "elimination": reward_config.elimination,
            "opponent_ratio": opponent_ratio,
        }
    )

    # In training loop
    wandb.log({
        "episode": episode,
        "loss": np.mean(losses[-100:]) if losses else 0,
        "dqn_win_rate": dqn_rate,
        "epsilon": dqn_policy.epsilon,
    })

    # During evaluation
    wandb.log({
        "eval/dqn_win_rate": eval_win_rate,
        "eval/games": eval_games,
    })
```

**Hyperparameter sweep config (`sweep.yaml`):**
```yaml
program: bluff/scripts/train_dqn.py
method: bayes
metric:
  name: eval/dqn_win_rate
  goal: maximize
parameters:
  learning_rate:
    min: 0.0001
    max: 0.01
  gamma:
    values: [0.95, 0.99, 0.999]
  dice_fraction_scale:
    values: [0.0, 0.5, 1.0, 2.0]
  opponent_ratio:
    min: 0.0
    max: 0.5
```

## Metrics to Track

### Episode-Level (every episode)
- Episode return (cumulative reward)
- Episode length (steps)
- Win/loss outcome
- Per-seat win distribution

### Training (every 10-100 episodes)
- TD loss (mean, std)
- Average Q-values
- Replay buffer size
- Gradient norms (optional)

### Exploration
- Epsilon value
- Action distribution
- Valid action mask utilization

### Game-Specific (Bluff)
- Bluff rate (bids that were false)
- Call accuracy (correct vs incorrect)
- Bid aggression (bid count vs expected)
- Round survival rate

### Evaluation (every 500 episodes)
- Win rate vs heuristic (greedy policy)
- Win rate vs random baseline
- Confidence intervals (if running multiple seeds)

## Implementation Tasks

1. **Basic TensorBoard** (1 hour)
   - [ ] Add SummaryWriter to train_dqn.py
   - [ ] Log loss, win rates, epsilon
   - [ ] Add `--tensorboard` CLI flag
   - [ ] Add `--run-name` CLI flag

2. **Enhanced Metrics** (1 hour)
   - [ ] Track Q-value distributions
   - [ ] Track action distributions
   - [ ] Add game-specific metrics (bluff rate, call accuracy)

3. **W&B Integration** (2 hours)
   - [ ] Add wandb initialization
   - [ ] Create sweep configuration
   - [ ] Add `--wandb` CLI flag
   - [ ] Test hyperparameter sweep

4. **Experiment Management** (2 hours)
   - [ ] Create experiment naming convention
   - [ ] Add checkpoint saving with config
   - [ ] Add experiment comparison script

## CLI Changes

```python
# Add to argparse
parser.add_argument("--tensorboard", action="store_true",
                    help="Enable TensorBoard logging")
parser.add_argument("--wandb", action="store_true",
                    help="Enable Weights & Biases logging")
parser.add_argument("--run-name", type=str, default=None,
                    help="Name for this experiment run")
parser.add_argument("--seed", type=int, default=None,
                    help="Random seed for reproducibility")
```

## Directory Structure

```
bluff/
├── runs/                    # TensorBoard logs
│   └── dqn_3p_3d_sparse_20251219/
├── models/                  # Saved checkpoints
│   └── dqn_3p_3d.pt
├── experiments/             # Experiment configs & results
│   ├── sweep_reward_config.yaml
│   └── results/
└── scripts/
    └── train_dqn.py
```

## Agents to Use for Implementation

When implementing, consider using these Claude Code agents:

1. **Plan agent** (`subagent_type=Plan`)
   - Use when designing the full logging architecture
   - Get step-by-step implementation plan

2. **analytics-expert agent** (`subagent_type=analytics-expert`)
   - Use for designing metrics and statistical analysis
   - Validate metric calculations

3. **data-engineering-expert agent** (`subagent_type=data-engineering-expert`)
   - Use for efficient data logging patterns
   - Design experiment data pipelines

## References

- [TensorBoard PyTorch Tutorial](https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html)
- [Weights & Biases RL Tutorial](https://wandb.ai/yashkotadia/rl-example/reports/Track-and-Tune-Your-Reinforcement-Learning-Models-With-Weights-Biases--VmlldzoyMjgxMzc)
- [CleanRL DQN with W&B](https://docs.cleanrl.dev/rl-algorithms/dqn/)
- [RLiable: Reliable RL Evaluation](https://research.google/blog/rliable-towards-reliable-evaluation-reporting-in-reinforcement-learning/)

---

*Generated: 2025-12-19*
