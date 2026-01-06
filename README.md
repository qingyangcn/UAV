# UAV Delivery System with MOPSO Scheduler and PPO

This repository implements a UAV delivery scheduling system with:
- **MOPSO (Multi-Objective Particle Swarm Optimization)** for order assignment
- **PPO (Proximal Policy Optimization)** for routing control (heading + speed)
- **Diagnostics and Training Stabilizers** to achieve non-zero on-time deliveries

## Quick Start

Training with recommended stabilizers (solves 0% on-time delivery problem):
```bash
python ppo/train_ppo.py --total-steps 100000 -K 5 \
  --throttle-warmup-steps 16 --eta-speed-scale 0.7
```

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
  - Parameters: M=200 (orders), K=5 (default, max per drone), P=30 (particles), I=10 (iterations)
  - Multi-objective fitness:
    - f0: Planned distance (minimize)
    - f1: Expected timeout orders (minimize with conservative ETA)
    - f2: Planned energy consumption (minimize)
  - Pareto archive with crowding distance
  - Cross-merchant interleaved stop construction
  - Conservative ETA mode to prevent over-commitment

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

#### Quick Start with Recommended Defaults (Stabilized Training)

```bash
# Recommended for initial training with stabilizers enabled
python ppo/train_ppo.py --total-steps 100000

# This uses safe defaults:
# - K=5 (reduced from 10 for less over-commitment)
# - Throttle warmup: 16 steps at min speed 1.0
# - Conservative ETA: speed_scale=0.7, service=1 step
```

#### Advanced Training with Custom Parameters

```bash
# Full control over training and stabilizers
python ppo/train_ppo.py \
    --total-steps 500000 \
    --seed 42 \
    --lr 3e-4 \
    --n-steps 2048 \
    --batch-size 64 \
    -K 10 \
    --throttle-warmup-steps 32 \
    --throttle-warmup-min-u 1.0 \
    --eta-speed-scale 0.6 \
    --eta-service-steps 2 \
    --log-dir ./logs \
    --model-dir ./models
```

#### Training Parameters

**PPO Parameters:**
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

**Training Stabilizers:**
- `-K, --max-orders-per-drone`: Max orders per drone for MOPSO (default: 5, use 10 for aggressive)
- `--throttle-warmup-steps`: Steps to enforce minimum speed (default: 16, use 0 to disable)
- `--throttle-warmup-min-u`: Minimum speed multiplier during warmup (default: 1.0)
- `--eta-speed-scale`: Conservative speed assumption for MOPSO ETA (default: 0.7)
- `--eta-service-steps`: Service time per stop for MOPSO (default: 1)

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
- K = 5: Maximum orders assigned per drone (default for stability, can use 10 for aggressive)
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

## Diagnostics and Training Stabilizers

### Diagnostics

The environment tracks and exposes detailed diagnostics to help identify performance issues:

**Lateness Tracking:**
- Tracks `lateness_steps` for each delivered order (delivery_time - deadline)
- Aggregated metrics in `info['diagnostics']`:
  - `lateness_mean`: Average lateness across all deliveries
  - `lateness_p90`: 90th percentile lateness
  - `lateness_max`: Maximum lateness observed
  - `on_time_count`: Number of on-time deliveries
  - `delivered_count`: Total deliveries

**Slack Tracking:**
- Tracks `assign_slack_steps` when orders transition READY → ASSIGNED
- Shows how much time buffer exists at assignment
- Aggregated metrics:
  - `assign_slack_mean`: Average slack at assignment
  - `assign_slack_p10`: 10th percentile slack (worst case)
  - `assign_slack_min`: Minimum slack observed

**Speed Scale Tracking:**
- Tracks PPO's speed multiplier (u parameter) per step
- Helps identify if PPO is throttling too much early in training
- Aggregated metrics:
  - `speed_scale_mean`: Average speed multiplier
  - `speed_scale_p10`: 10th percentile (conservative end)
  - `speed_scale_p90`: 90th percentile (aggressive end)

All diagnostics are printed at end-of-day and available in `info['diagnostics']`.

### Training Stabilizers

To achieve non-zero on-time deliveries, especially early in training:

**1. Throttle Warmup:**
- Forces drones to fly at or near max speed during initial steps
- Prevents PPO from learning to go too slow before understanding the environment
- Configure with `throttle_warmup_steps` and `throttle_warmup_min_u`
- Default: 16 steps at min speed 1.0

**2. Conservative ETA for MOPSO:**
- MOPSO uses pessimistic speed assumptions to avoid over-promising
- Reduces predicted timeout count in fitness calculation
- Configure with `eta_speed_scale_assumption` (default 0.7) and `eta_stop_service_steps` (default 1)

**3. Reduced Over-Commitment:**
- Lower K (max orders per drone) reduces risk of late deliveries
- Default K=5 for stability, can increase to 10 once training stabilizes

### Recommended Staged Training

**Stage 1: Stabilization (0-100K steps)**
```bash
python ppo/train_ppo.py \
    --total-steps 100000 \
    -K 5 \
    --throttle-warmup-steps 16 \
    --eta-speed-scale 0.7
```
Goal: Achieve >0% on-time delivery rate

**Stage 2: Performance (100K-500K steps)**
```bash
python ppo/train_ppo.py \
    --total-steps 500000 \
    -K 7 \
    --throttle-warmup-steps 8 \
    --eta-speed-scale 0.75
```
Goal: Improve on-time rate to 50%+

**Stage 3: Optimization (500K+ steps)**
```bash
python ppo/train_ppo.py \
    --total-steps 1000000 \
    -K 10 \
    --throttle-warmup-steps 0 \
    --eta-speed-scale 0.8
```
Goal: Achieve 70%+ on-time rate

### Interpreting Diagnostics

**If on-time deliveries = 0:**
- Check `assign_slack_mean`: If negative, deadlines are too tight before assignment
- Check `lateness_mean`: Large positive values indicate systematic lateness
- Check `speed_scale_mean`: If < 0.8, PPO may be throttling too much

**If slack is consistently negative:**
- Increase `eta_speed_scale_assumption` (more optimistic)
- Reduce K to assign fewer orders per drone
- Consider looser SLA in environment config

**If speed_scale is too low:**
- Increase `throttle_warmup_steps` to enforce faster flight longer
- May indicate PPO hasn't learned that speed matters for on-time delivery

## Notes

- The environment uses fallback data if CSV/Excel files are not found
- Random events can be disabled during evaluation for deterministic results
- Multi-objective returns are tracked in `info['r_vec']` and `info['episode_r_vec']`
- MOPSO runs every step to assign orders to newly idle drones
- PPO controls routing without affecting order assignment

## Example Output

Training with stabilizers:
```
============================================================
PPO Training with MOPSO Dispatcher
============================================================
Total timesteps: 100000
Seed: 42
Learning rate: 0.0003
Reward mode: zero (multi-objective)
MOPSO: M=200, K=5, P=30, I=10
Throttle warmup: 16 steps, min_u=1.0
Conservative ETA: speed_scale=0.7, service=1
============================================================
```

End-of-day diagnostics:
```
=== 结束 ===
今日统计:
  生成订单: 120
  完成订单: 85
  取消订单: 35
  准时交付: 42

诊断指标 - 送达延迟:
  平均延迟: 2.3 steps
  P90延迟: 8.5 steps
  最大延迟: 15.0 steps
  准时率: 42/85 (49.4%)

诊断指标 - 分配时余裕:
  平均余裕: 12.5 steps
  P10余裕: 3.2 steps
  最小余裕: -2.0 steps

诊断指标 - PPO速度倍数:
  平均速度: 0.982
  P10速度: 0.850
  P90速度: 1.150

  未完成订单: 0 已取消
=== 当日营业结束 ===
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
