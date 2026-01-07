# U7 Quick Start Guide

## What Changed?

### Action Space
**Before (U6):**
- Action: `(num_drones, 3)` = [heading_x, heading_y, speed]
- PPO controls both heading and speed

**After (U7):**
- Action: `(num_drones, 2)` = [task_choice, speed]
- PPO chooses which order to serve next + speed
- Environment flies drone straight to target

### Observation Space
**New component:**
- `candidates`: `(num_drones, 20, 12)` - 20 candidate orders per drone with 12 features each

### MOPSO Role
**Before (U6):**
- Assigns orders AND builds route plans (planned_stops)

**After (U7):**
- Only assigns orders (READY → ASSIGNED)
- No route planning

### PPO Role
**Before (U6):**
- Controls heading (direction) and speed
- Follows fixed route from MOPSO

**After (U7):**
- Selects next task (which order to serve) from K=20 candidates
- Controls speed
- Can change task at decision points only

## Quick Examples

### Minimum Working Example
```python
from U7_train import make_env

# Create and run
env = make_env(seed=42, reward_output_mode='scalar')
obs, info = env.reset()

for step in range(100):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    if done:
        break
```

### With Stable-Baselines3
```python
from U7_train import make_env
from stable_baselines3 import PPO

env = make_env(seed=42, reward_output_mode='scalar')
model = PPO('MultiInputPolicy', env, verbose=1)
model.learn(total_timesteps=100000)
model.save("u7_ppo_model")
```

### Custom Training Loop
```python
import numpy as np
from UAV_ENVIRONMENT_7 import ThreeObjectiveDroneDeliveryEnv
from U7_mopso_dispatcher import U7MOPSOAssigner, apply_mopso_assignment

# Setup
env = ThreeObjectiveDroneDeliveryEnv(
    num_drones=6,
    num_candidates=20,
    reward_output_mode='scalar'
)
assigner = U7MOPSOAssigner(n_particles=30, n_iterations=10)

# Training loop
obs, info = env.reset(seed=42)
for episode in range(100):
    episode_reward = 0
    obs, info = env.reset()
    
    for step in range(1000):
        # MOPSO assigns READY orders to drones
        apply_mopso_assignment(env, assigner)
        
        # PPO chooses task and speed
        # action[:, 0] = task choice in [-1, 1]
        # action[:, 1] = speed in [-1, 1]
        action = your_policy(obs)  # Shape: (6, 2)
        
        # Environment step
        obs, reward, done, truncated, info = env.step(action)
        episode_reward += reward
        
        if done:
            break
    
    print(f"Episode {episode}: Reward = {episode_reward:.2f}")
```

## Testing

```bash
# Run all tests
python test_u7_implementation.py
python test_u7_integration.py

# Test training script
python U7_train.py --total-steps 1000 --seed 42
```

## Key Features

### Candidate Features (12 dimensions)
1. **Validity** (0/1): Is this a real order or padding?
2-6. **Status** (one-hot): Order status encoding
7. **Type** (0-1): Normal/Free/Complex order type
8. **Age** (0-1): Time since order creation
9. **Urgency** (0/1): Is order marked urgent?
10. **Deadline slack** (0-1): Time until deadline
11-12. **Location** (0-1): Merchant location (normalized)

### Decision Points
PPO can change target ONLY when:
- ✓ Drone is IDLE
- ✓ Just completed pickup at merchant
- ✓ Just completed delivery at customer
- ✗ Never mid-flight

### Speed Multiplier
- Input range: [-1, 1]
- Mapped to: [0.5, 1.5]
- Affects: Movement distance per step
- Formula: `(action + 1) / 2 * (1.5 - 0.5) + 0.5`

## Debugging Tips

### Check if MOPSO is assigning
```python
assignment_counts = apply_mopso_assignment(env, assigner)
print(f"Assigned {sum(assignment_counts.values())} orders")
```

### Check candidate validity
```python
obs, _ = env.reset()
for drone_id in range(env.num_drones):
    valid_count = sum(obs['candidates'][drone_id, :, 0])
    print(f"Drone {drone_id}: {valid_count} valid candidates")
```

### Check decision points
```python
for drone_id in range(env.num_drones):
    is_decision = env._is_at_decision_point(drone_id)
    status = env.drones[drone_id]['status'].name
    print(f"Drone {drone_id}: {status} -> Decision: {is_decision}")
```

### Monitor order flow
```python
from collections import Counter
statuses = [env.orders[oid]['status'].name for oid in env.active_orders]
print(Counter(statuses))
# Example output: {'READY': 10, 'ASSIGNED': 5, 'PICKED_UP': 2}
```

## Common Issues

### "No valid candidates"
- Orders may still be ACCEPTED (not READY yet)
- Wait a few steps for order preparation
- Check: `sum(1 for oid in env.active_orders if env.orders[oid]['status'].name == 'READY')`

### "MOPSO not assigning"
- Drones may be at capacity
- No READY orders available
- Check: `env.get_ready_orders_snapshot()`

### "Action space mismatch"
- Use `U7FlattenActionWrapper` for SB3
- Direct use requires `(num_drones, 2)` shape
- Flattened version is `(num_drones * 2,)`

## Performance

Typical metrics (6 drones, 200 max orders):
- **MOPSO assignments/step**: 10-30 orders when READY available
- **Orders completed/100 steps**: 50-100 (varies by policy)
- **Step time**: ~10-20ms (including MOPSO)
- **Observation size**: ~50KB (Dict observation)
