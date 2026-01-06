# UAV Delivery System with MOPSO Scheduler and PPO

This repository implements a UAV delivery scheduling system with:
- **MOPSO (Multi-Objective Particle Swarm Optimization)** for order assignment
- **PPO (Proximal Policy Optimization)** for routing control (heading + speed)

## Features

### Environment
- **UAV_ENVIRONMENT_6.py**: Main simulation environment
  - Multi-objective rewards (throughput, cost, quality)
  - Action space: (N, 3) where N=number of drones
    - (hx, hy): heading direction
    - u: speed multiplier
  - Snapshot interfaces for external scheduling

### MOPSO Dispatcher
- **algorithms/mopso_dispatcher.py**: Order assignment scheduler
  - Parameters: M=200 (orders), K=10 (max per drone), P=30 (particles), I=10 (iterations)
  - Multi-objective fitness:
    - f0: Planned distance (minimize)
    - f1: Expected timeout orders (minimize)
    - f2: Planned energy consumption (minimize)
  - Pareto archive with crowding distance
  - Cross-merchant interleaved stop construction

### PPO Training
- **ppo/train_ppo.py**: Training script using Stable-Baselines3
  - Integrates MOPSO for assignment at each step
  - PPO controls routing via (direction, speed) actions
  - Multi-objective reward mode (reward_output_mode="zero")

### Evaluation
- **ppo/eval_ppo.py**: Evaluation and inference script
  - Reports daily statistics and metrics
  - Multi-objective return analysis

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or install manually:
pip install numpy pandas matplotlib scikit-learn gymnasium stable-baselines3 torch
```

## Usage

### 1. Test the Integration

```bash
python test_integration.py
```

This runs basic tests to verify:
- Environment creation with (N, 3) action space
- Snapshot interfaces work correctly
- MOPSO dispatcher generates valid plans
- Action wrappers function properly
- Full integration (MOPSO + PPO)

### 2. Train a PPO Agent

```bash
# Basic training (100K steps)
python ppo/train_ppo.py --total-steps 100000

# With custom parameters
python ppo/train_ppo.py \
    --total-steps 500000 \
    --seed 42 \
    --lr 3e-4 \
    --n-steps 2048 \
    --batch-size 64 \
    --log-dir ./logs \
    --model-dir ./models
```

Training parameters:
- `--total-steps`: Total training timesteps
- `--seed`: Random seed
- `--lr`: Learning rate (default: 3e-4)
- `--n-steps`: Steps per rollout (default: 2048)
- `--batch-size`: Minibatch size (default: 64)
- `--n-epochs`: PPO epochs (default: 10)
- `--gamma`: Discount factor (default: 0.99)
- `--save-freq`: Model save frequency (default: 10000)
- `--log-dir`: Tensorboard logs directory
- `--model-dir`: Model checkpoints directory

### 3. Evaluate a Trained Model

```bash
# Basic evaluation
python ppo/eval_ppo.py --model models/ppo_uav_final.zip --episodes 10

# With custom parameters
python ppo/eval_ppo.py \
    --model models/ppo_uav_final.zip \
    --episodes 50 \
    --seed 123 \
    --enable-random-events
```

Evaluation parameters:
- `--model`: Path to trained model file
- `--episodes`: Number of episodes to evaluate
- `--seed`: Random seed
- `--enable-random-events`: Enable random events (disabled by default for deterministic eval)
- `--quiet`: Suppress episode-by-episode details

## Architecture

### Flow
1. **Reset**: Environment initializes, MOPSO assigns orders to idle drones
2. **Each Step**:
   - MOPSO assigns new orders to idle drones (every step)
   - PPO produces (hx, hy, u) actions for all drones
   - Environment executes drone movements and deliveries
   - Multi-objective rewards computed
3. **Episode End**: Daily statistics aggregated

### MOPSO Details
- **Encoding**: Random Keys for order assignment
- **Decoding**: Greedy assignment based on sorted keys, respecting capacity constraints
- **Stop Construction**:
  1. Group orders by merchant
  2. Sort merchants by earliest deadline
  3. For each merchant: P(merchant) → D(most_urgent_order)
  4. Append remaining deliveries sorted by deadline
- **Pareto Archive**: Maintains non-dominated solutions, truncated by crowding distance

### PPO Details
- **Policy**: MultiInputPolicy for Dict observation space
- **Action**: Box(N*3,) flattened for SB3, reshaped to (N, 3) for environment
- **Reward**: Zero (multi-objective training) - gradients from episode returns in info
- **Integration**: MOPSOWrapper calls dispatcher before each step

## File Structure

```
UAV/
├── UAV_ENVIRONMENT_6.py          # Main environment with snapshot interfaces
├── algorithms/
│   ├── mopso_utils.py            # Pareto utilities (dominance, crowding, archive)
│   └── mopso_dispatcher.py       # MOPSO planner (M=200, K=10, P=30, I=10)
├── env_wrappers/
│   └── flatten_action.py         # Action space wrapper for SB3
├── ppo/
│   ├── train_ppo.py              # PPO training script
│   └── eval_ppo.py               # Evaluation script
├── test_integration.py           # Integration tests
├── requirements.txt              # Dependencies
└── README.md                     # This file
```

## Key Parameters

### MOPSO
- M = 200: Maximum candidate READY orders per scheduling call
- K = 10: Maximum orders assigned per drone
- P = 30: Number of particles
- I = 10: Number of PSO iterations

### Environment
- N = 6: Number of drones (default)
- Grid size: 16x16
- Action space: (6, 3) for (hx, hy, u)
- Observation: Dict with orders, drones, merchants, etc.

### PPO
- Learning rate: 3e-4
- Rollout steps: 2048
- Batch size: 64
- Epochs: 10
- Gamma: 0.99

## Notes

- The environment uses fallback data if CSV/Excel files are not found
- Random events can be disabled during evaluation for deterministic results
- Multi-objective returns are tracked in `info['r_vec']` and `info['episode_r_vec']`
- MOPSO runs every step to assign orders to newly idle drones
- PPO controls routing without affecting order assignment

## Example Output

Training:
```
============================================================
PPO Training with MOPSO Dispatcher
============================================================
Total timesteps: 100000
Seed: 42
Learning rate: 0.0003
Reward mode: zero (multi-objective)
MOPSO: M=200, K=10, P=30, I=10
============================================================
```

Evaluation:
```
============================================================
Evaluation Results
============================================================
Episodes: 10

Episode Metrics:
  Mean reward: 0.00 ± 0.00
  Mean length: 64.0 ± 0.0

Multi-Objective Returns (mean ± std):
  f0 (throughput): 45.23 ± 5.12
  f1 (cost):       -123.45 ± 15.32
  f2 (quality):    32.10 ± 4.56

Aggregate Daily Statistics:
  Total orders completed: 450
  Completion rate: 89.5%
  On-time rate: 78.3%
============================================================
```
