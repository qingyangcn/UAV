# UAV Environment 6 - P0+P1 API Documentation

## Overview

This document describes the new APIs and features added in the P0+P1 implementation for external PSO integration and experimental clarity.

## P0-1: Snapshot Read Interfaces

### `get_ready_orders_snapshot() -> List[Dict]`

Returns an immutable snapshot of READY & unassigned orders for external PSO planning.

**Returns:** List of order dictionaries with the following fields:
- `order_id` (int): Unique order identifier
- `merchant_id` (str): Merchant identifier
- `merchant_location` (tuple): (x, y) grid coordinates
- `customer_location` (tuple): (x, y) grid coordinates
- `creation_time` (int): Step when order was created
- `age_steps` (int): Current age of order in steps
- `urgent` (bool): Whether order is urgent
- `promised_steps` (int): SLA promised delivery steps
- `deadline_step` (int): Absolute deadline step (creation_time + promised_steps * timeout_factor)
- `order_type` (int): Order type enum value
- `preparation_time` (int): Preparation time in steps

**Example:**
```python
env = ThreeObjectiveDroneDeliveryEnv(...)
obs, info = env.reset()

# Get snapshot of available orders
ready_orders = env.get_ready_orders_snapshot()

for order in ready_orders:
    print(f"Order {order['order_id']}: "
          f"merchant={order['merchant_id']}, "
          f"deadline={order['deadline_step']}")
```

### `get_drones_snapshot() -> List[Dict]`

Returns an immutable snapshot of all drones.

**Returns:** List of drone dictionaries with the following fields:
- `drone_id` (int): Drone identifier
- `location` (tuple): Current (x, y) position
- `status` (int): DroneStatus enum value
- `status_name` (str): Human-readable status name
- `battery_level` (float): Current battery percentage
- `base_id` (int): Home base identifier
- `assigned_load` (int): Number of committed orders (not yet completed)
- `cargo_load` (int): Number of picked-up orders (in transit)
- `has_route` (bool): Whether a route plan is installed
- `max_capacity` (int): Maximum cargo capacity
- `speed` (float): Drone speed
- `reliability` (float): Reliability factor

**Example:**
```python
drones = env.get_drones_snapshot()

for drone in drones:
    if drone['status_name'] == 'IDLE' and drone['battery_level'] > 50:
        print(f"Drone {drone['drone_id']} is available for assignment")
```

### `get_merchants_snapshot() -> List[Dict]`

Returns an immutable snapshot of all merchants.

**Returns:** List of merchant dictionaries with the following fields:
- `merchant_id` (str): Merchant identifier
- `location` (tuple): (x, y) grid coordinates
- `queue_len` (int): Current queue length
- `efficiency` (float): Preparation efficiency factor
- `business_type` (str): Type of business
- `landing_zone` (bool): Whether merchant has landing zone

**Example:**
```python
merchants = env.get_merchants_snapshot()

# Find merchants with high queue length
busy_merchants = [m for m in merchants if m['queue_len'] > 5]
```

### `get_route_plan_constraints() -> Dict`

Returns constraint declaration for external PSO to generate legal route plans.

**Returns:** Dictionary with constraint information:
- `ready_only_policy` (bool): Whether only READY orders can be planned
- `description` (str): Policy description
- `stop_types` (list): Valid stop types ['P', 'D']
- `stop_format` (dict): Format specification for each stop type
- `speed_multiplier_range` (list): [min, max] speed multiplier (future use)
- `max_stops_per_route` (int): Maximum stops per route
- `max_capacity` (int): Drone capacity
- `timeout_factor` (float): SLA timeout multiplier
- `retry_strategy` (str): Retry strategy description
- `grid_size` (int): Environment grid size
- `num_drones` (int): Number of drones
- `num_bases` (int): Number of bases

**Example:**
```python
constraints = env.get_route_plan_constraints()
print(f"Max capacity: {constraints['max_capacity']}")
print(f"Timeout factor: {constraints['timeout_factor']}")
```

## P0-2: Structured apply_route_plan Results

### `apply_route_plan(drone_id, planned_stops, commit_orders=None, allow_busy=True) -> Dict`

Apply a cross-merchant interleaved route plan with structured return.

**Arguments:**
- `drone_id` (int): Target drone
- `planned_stops` (List[dict]): Route plan as list of stops
  - Pickup stop: `{'type': 'P', 'merchant_id': 'M1'}`
  - Delivery stop: `{'type': 'D', 'order_id': 123}`
- `commit_orders` (Optional[List[int]]): Explicit list of order IDs to commit (defaults to all D stops)
- `allow_busy` (bool): Whether to allow dispatching to busy drones

**Returns:** Dictionary with:
- `success` (bool): Whether route plan was successfully applied
- `committed_orders` (List[int]): Successfully committed order IDs
- `rejected_orders` (Dict[int, str]): Rejected order_id -> reason mapping
- `route_installed` (bool): Whether route was installed
- `reason` (str): Explanation of result

**Atomic Behavior:** If no orders can be committed, no state changes occur.

**Example:**
```python
# Plan a multi-merchant route
planned_stops = [
    {'type': 'P', 'merchant_id': 'M1'},
    {'type': 'D', 'order_id': 101},
    {'type': 'P', 'merchant_id': 'M2'},
    {'type': 'D', 'order_id': 102},
]

result = env.apply_route_plan(
    drone_id=0,
    planned_stops=planned_stops,
    allow_busy=True
)

if result['success']:
    print(f"✓ Committed {len(result['committed_orders'])} orders")
else:
    print(f"✗ Failed: {result['reason']}")
    for oid, reason in result['rejected_orders'].items():
        print(f"  Order {oid}: {reason}")
```

## P0-3: Zombie Order/Deadlock Recovery

The environment now includes automatic stuck order reconciliation that runs every step.

### Configuration Parameters

Set in `__init__`:
- `timeout_factor` (float): SLA timeout multiplier (default: 2.0)
- `stuck_assigned_reset_threshold` (int): Steps before resetting ASSIGNED → READY (default: 20)
- `stuck_assigned_cancel_threshold` (int): Steps before canceling stuck ASSIGNED (default: 50)
- `stuck_picked_up_retry_threshold` (int): Steps before retrying PICKED_UP delivery (default: 30)
- `stuck_picked_up_force_threshold` (int): Steps before force-completing PICKED_UP (default: 80)

### Statistics

Available in `info['stuck_order_stats']`:
- `stuck_assigned_reset_ready`: ASSIGNED orders reset to READY
- `stuck_assigned_timeout_cancel`: ASSIGNED orders canceled due to timeout
- `stuck_picked_up_force_complete`: PICKED_UP orders force-completed
- `stuck_picked_up_retry`: PICKED_UP orders flagged for retry

**Example:**
```python
obs, reward, done, truncated, info = env.step(action)

stats = info['stuck_order_stats']
print(f"Stuck orders handled: {sum(stats.values())}")
```

## P1-1: assigned_load vs cargo_load

The environment now clearly distinguishes:
- **assigned_load** (`drone['current_load']`): Number of committed orders (ASSIGNED + PICKED_UP)
- **cargo_load** (`len(drone['cargo'])`): Number of picked-up orders currently in transit

Both are exposed in `get_drones_snapshot()`.

## P1-2: PSO vs PPO Metrics

The environment now tracks and reports separate metrics for PSO (planning) and PPO (execution).

### Available in info

**PSO Metrics** (`info['pso_metrics']`):
- `total_route_plans`: Total number of route plans applied
- `recent_success_rate`: Success rate of recent 100 plans
- `avg_committed_per_route`: Average orders committed per route
- `avg_rejected_per_route`: Average orders rejected per route
- `avg_stops_per_route`: Average number of stops per route
- `avg_planned_distance`: Average planned route distance

**PPO Metrics** (`info['ppo_metrics']`):
- `total_actual_flight_distance`: Total actual distance flown
- `total_energy_consumed`: Total energy consumed
- `avg_energy_per_order`: Average energy per delivered order
- `avg_distance_per_order`: Average distance per delivered order
- `speed_stats`: Speed multiplier statistics (placeholder)

**Example:**
```python
obs, reward, done, truncated, info = env.step(action)

pso = info['pso_metrics']
ppo = info['ppo_metrics']

print(f"PSO success rate: {pso['recent_success_rate']:.1%}")
print(f"PPO distance efficiency: {ppo['avg_distance_per_order']:.2f}")
```

## P1-3: Consistent SLA/Deadline Fields

All deadline calculations now use a consistent formula:
```
deadline_step = creation_time + promised_steps * timeout_factor
```

Where:
- `promised_steps = preparation_steps + SLA_steps`
- `SLA_steps = _minutes_to_steps(15)` (15-minute SLA)
- `timeout_factor = 2.0` (configurable)

This deadline is exposed in `get_ready_orders_snapshot()` and can be used by external PSO for deadline-aware planning.

## Complete Integration Example

```python
import numpy as np
from UAV_ENVIRONMENT_6 import ThreeObjectiveDroneDeliveryEnv

# Create environment
env = ThreeObjectiveDroneDeliveryEnv(
    grid_size=16,
    num_drones=6,
    max_orders=100,
    drone_max_capacity=10,
    reward_output_mode="zero",  # For PSO+PPO training
)

obs, info = env.reset(seed=42)

for step in range(1000):
    # External PSO: Get current state
    ready_orders = env.get_ready_orders_snapshot()
    drones = env.get_drones_snapshot()
    constraints = env.get_route_plan_constraints()
    
    # External PSO: Generate route plans (your logic here)
    # ... PSO planning logic ...
    
    # Apply route plan for available drone
    if len(ready_orders) > 0:
        available_drones = [d for d in drones if d['status_name'] == 'IDLE']
        if available_drones:
            drone_id = available_drones[0]['drone_id']
            order = ready_orders[0]
            
            planned_stops = [
                {'type': 'P', 'merchant_id': order['merchant_id']},
                {'type': 'D', 'order_id': order['order_id']},
            ]
            
            result = env.apply_route_plan(drone_id, planned_stops)
            if result['success']:
                print(f"✓ Step {step}: Assigned order {order['order_id']} to drone {drone_id}")
    
    # PPO: Generate heading actions
    ppo_action = np.random.randn(env.num_drones, 2)  # Replace with PPO policy
    
    # Step environment
    obs, reward, done, truncated, info = env.step(ppo_action)
    
    # Check metrics
    if step % 100 == 0:
        print(f"Step {step}:")
        print(f"  PSO success rate: {info['pso_metrics']['recent_success_rate']:.1%}")
        print(f"  PPO avg distance: {info['ppo_metrics']['avg_distance_per_order']:.2f}")
        print(f"  Stuck orders: {sum(info['stuck_order_stats'].values())}")
    
    if done:
        break
```

## Reproducibility (P0-4)

All new features use the same random source set by `env.reset(seed=...)`:
- Order generation
- Merchant/drone initialization
- Stuck order reconciliation (deterministic thresholds)

No uncontrolled randomness has been introduced.

## Testing

Run the provided test suite:
```bash
python test_snapshot_interfaces.py
```

This tests:
- Snapshot interface serialization
- apply_route_plan atomicity and rollback
- Stuck order recovery
- Metrics separation
- Deadline consistency
