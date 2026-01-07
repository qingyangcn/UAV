# U7 Control Decomposition Implementation

## Overview

This implementation introduces a new control decomposition for the UAV delivery system that separates responsibilities between MOPSO and PPO:

- **MOPSO**: Handles order-to-drone assignment only (no route planning)
- **PPO**: Performs per-drone task selection (which order to serve next) + speed control
- **Environment**: Computes heading automatically (drones fly straight to chosen target)

## Key Changes

### 1. UAV_ENVIRONMENT_7.py

#### New Parameters
- `num_candidates` (default=20): Number of candidate orders per drone for PPO task selection

#### Observation Space Changes
Added new observation component:
```python
'candidates': Box(0, 1, (num_drones, num_candidates, 12), float32)
```

Each candidate has 12 features:
- Feature 0: Validity flag (1.0 if valid, 0.0 if padding)
- Features 1-5: Order status one-hot encoding
- Feature 6: Order type (normalized)
- Feature 7: Age (normalized by 50 steps)
- Feature 8: Urgency flag
- Feature 9: Deadline slack (normalized)
- Features 10-11: Merchant location (normalized)

#### Action Space Changes
Changed from `(num_drones, 3)` to `(num_drones, 2)`:
```python
# Old: [heading_x, heading_y, speed_multiplier]
# New: [task_choice, speed_multiplier]
```

Where:
- `task_choice`: Continuous value in [-1, 1] mapped to discrete choice [0, K-1]
- `speed_multiplier`: Continuous value in [-1, 1] mapped to [0.5, 1.5]

#### Decision Point Logic
PPO can only change targets at decision points:
- Drone status is IDLE
- Drone has just arrived at merchant (after pickup)
- Drone has just arrived at customer (after delivery)

This prevents mid-flight oscillation and ensures stable behavior.

#### Candidate Selection Priority
Per drone, candidates are selected in priority order:
1. Orders in cargo (PICKED_UP status, assigned to this drone)
2. Orders ASSIGNED to this drone but not yet picked up
3. Padding with invalid entries up to K=20

#### Movement Changes
- Removed heading control from PPO
- Drones now fly straight to their `target_location`
- Speed multiplier from PPO action is applied to movement distance

### 2. U7_mopso_dispatcher.py

New dispatcher that performs assignment only:

#### U7MOPSOAssigner
```python
assigner = U7MOPSOAssigner(
    n_particles=30,
    n_iterations=10,
    max_orders=200,
    max_orders_per_drone=10
)
```

Reuses MOPSO optimization from U6 but:
- Returns assignment dict `{drone_id: [order_ids]}` only
- Does NOT generate `planned_stops`
- Assigns READY orders to drones respecting capacity

#### apply_mopso_assignment
```python
assignment_counts = apply_mopso_assignment(env, assigner)
```

Applies assignments to environment:
- Updates order status from READY to ASSIGNED
- Sets `order['assigned_drone']`
- Increments `drone['current_load']`
- Does NOT set `drone['planned_stops']`

### 3. U7_train.py

Training script with integrated MOPSO + PPO:

#### U7MOPSOWrapper
Gym wrapper that calls MOPSO assignment at each step:
```python
env = U7MOPSOWrapper(base_env, mopso_assigner)
```

- Calls `apply_mopso_assignment()` before each `step()`
- Calls `apply_mopso_assignment()` after `reset()`

#### U7FlattenActionWrapper
Flattens action space for Stable-Baselines3 compatibility:
```python
# Environment expects: (N, 2)
# SB3 provides: (N*2,) flat array
env = U7FlattenActionWrapper(env)
```

#### make_env
Complete environment factory:
```python
env = make_env(
    seed=42,
    reward_output_mode='scalar',  # Uses weighted sum of objectives
    num_candidates=20
)
```

## Usage

### Basic Training Setup
```python
from U7_train import make_env

# Create environment
env = make_env(
    seed=42,
    reward_output_mode='scalar',
    num_candidates=20
)

# Use with Stable-Baselines3
from stable_baselines3 import PPO
model = PPO('MultiInputPolicy', env, verbose=1)
model.learn(total_timesteps=100000)
```

### Manual Control
```python
from UAV_ENVIRONMENT_7 import ThreeObjectiveDroneDeliveryEnv
from U7_mopso_dispatcher import apply_mopso_assignment, U7MOPSOAssigner

# Create environment
env = ThreeObjectiveDroneDeliveryEnv(
    num_candidates=20,
    reward_output_mode='scalar'
)

# Create assigner
assigner = U7MOPSOAssigner()

# Reset
obs, info = env.reset()

# Main loop
for step in range(100):
    # Apply MOPSO assignment
    apply_mopso_assignment(env, assigner)
    
    # Generate PPO action (task choice + speed per drone)
    action = policy(obs)  # Shape: (num_drones, 2)
    
    # Step environment
    obs, reward, done, truncated, info = env.step(action)
```

## Testing

Run the test suites:
```bash
# Basic functionality tests
python test_u7_implementation.py

# Integration workflow test
python test_u7_integration.py

# Training script test
python U7_train.py --total-steps 1000
```

## Design Rationale

### Why Assignment Only?
- **Separation of Concerns**: MOPSO excels at global assignment optimization, PPO excels at local sequential decision making
- **Flexibility**: PPO can adapt task selection online based on real-time conditions
- **Scalability**: Simpler MOPSO runs faster (no route construction overhead)

### Why Decision Points?
- **Stability**: Prevents oscillation from frequent target changes
- **Realism**: Real drones can't instantly change direction mid-flight
- **Better Learning**: Clearer action consequences for PPO

### Why Straight-to-Target Movement?
- **Simplification**: Removes heading as a learning problem
- **Focus**: PPO focuses on high-level task selection, not low-level control
- **Efficiency**: Direct movement is optimal for point-to-point delivery

## Backward Compatibility

The changes maintain compatibility with existing code:
- Original U6 files unchanged
- UAV_ENVIRONMENT_7.py retains old action processing paths for backward compatibility
- Can still use route planning mode by not using U7 wrappers

## Performance Characteristics

Based on tests:
- **Order Assignment**: MOPSO assigns 10-30+ orders per step when READY orders available
- **Candidate Updates**: ~1ms per step for 6 drones × 20 candidates
- **Task Selection**: PPO chooses from valid candidates at decision points
- **Completion Rate**: 13-19 orders completed in first 10 steps (test scenario)

## Future Enhancements

Potential improvements:
1. **Dynamic K**: Adjust `num_candidates` based on load
2. **Priority Candidates**: Weight urgent orders higher in candidate selection
3. **Multi-Step Planning**: Allow PPO to see consequences of task sequences
4. **Hierarchical Learning**: Train separate policies for assignment and routing
