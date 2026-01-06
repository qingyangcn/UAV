# UAV Environment Route Planning Guide

## Overview

`UAV_ENVIRONMENT_6.py` has been enhanced with comprehensive route planning support for external PSO scheduling. This guide explains how to use the new features.

## Key Features

### 1. Route-Plan-Only Dispatch Mode

By default, the environment operates in `route_plan_only` mode, which means:
- Internal auto-assignment is disabled
- External schedulers (PSO) must use `apply_route_plan()` to assign routes
- Only READY orders should be included in route plans

```python
env = ThreeObjectiveDroneDeliveryEnv(
    dispatch_mode='route_plan_only',  # Default
    # ... other parameters
)
```

### 2. Route Plan Format

Each route plan is a list of stops:

```python
route_plan = [
    {'type': 'P', 'merchant_id': 'merchant_1'},  # Pickup stop
    {'type': 'D', 'order_id': 123},              # Delivery stop
    {'type': 'P', 'merchant_id': 'merchant_2'},
    {'type': 'D', 'order_id': 124},
    # ... more stops
]
```

**Constraints:**
- Each stop must have `type` in ('P', 'D')
- P stops require `merchant_id`
- D stops require `order_id` (must be READY)
- P(merchant) must appear before D(order) for that order's merchant
- Each order can appear at most once
- Total stops ≤ `max_planned_stops` (default: 20)

### 3. Applying Route Plans

```python
# Get READY orders
ready_orders = [
    oid for oid in env.active_orders 
    if env.orders[oid]['status'] == OrderStatus.READY
]

# Build route plan (PSO decides assignment, batching, sequencing)
route_plan = build_route_with_pso(ready_orders, drone_id)

# Apply route plan
success = env.apply_route_plan(
    drone_id=0,
    planned_stops=route_plan,
    commit_orders=None,  # Auto-derived from D stops
    allow_busy=True      # Can assign to busy drones
)

if success:
    print("Route plan accepted and applied")
else:
    print("Route plan rejected (validation failed)")
```

### 4. Retry Mechanism

Failed pickups/deliveries are automatically retried:

```python
env = ThreeObjectiveDroneDeliveryEnv(
    pickup_retry_policy='back',  # 'back' or 'front'
    max_stop_retries=3,          # Max retry attempts
)
```

**Behavior:**
- If pickup fails at P stop: retry up to `max_stop_retries` times
- If delivery fails at D stop: retry (order not picked up yet)
- After max retries: cancel order or reset to READY
- `pickup_retry_policy` controls where retried stop is inserted:
  - `'front'`: Insert at front of queue (retry immediately)
  - `'back'`: Insert at back of queue (retry after other stops)

### 5. Timeout and Cancellation

Orders automatically timeout to prevent indefinite hanging:

```python
env = ThreeObjectiveDroneDeliveryEnv(
    timeout_factor=2.0,  # Timeout multiplier
)
```

**Timeout Calculation:**
```
deadline = creation_time + promised_delivery_steps * timeout_factor
```

Where `promised_delivery_steps = preparation_steps + SLA_steps`

**Cancellation Reasons:**
All cancellations are tracked with reasons:
```python
print(env.metrics['cancellation_reasons'])
# {'timeout': 42, 'pickup_retry_exceeded': 5, ...}
```

### 6. Random Events Control

For deterministic evaluation:

```python
env = ThreeObjectiveDroneDeliveryEnv(
    enable_random_events=False,  # Disable random cancellations
)

env.reset(seed=42)  # Fixed seed for reproducibility
```

### 7. PPO Heading Control

PPO outputs heading direction for drones:

```python
env = ThreeObjectiveDroneDeliveryEnv(
    heading_guidance_alpha=0.8,  # 0.8 = 80% PPO, 20% target
)

# In training/evaluation loop
action = ppo_policy(obs)  # shape: (num_drones, 2)
obs, reward, done, truncated, info = env.step(action)
```

**Alpha values:**
- `0.0`: Pure target-directed flight (no PPO)
- `0.5`: Balanced blend
- `1.0`: Pure PPO control (PPO-primary mode)

## Complete Example

```python
from UAV_ENVIRONMENT_6 import ThreeObjectiveDroneDeliveryEnv, OrderStatus
import numpy as np

# Create environment
env = ThreeObjectiveDroneDeliveryEnv(
    grid_size=16,
    num_drones=6,
    dispatch_mode='route_plan_only',
    pickup_retry_policy='back',
    max_stop_retries=3,
    timeout_factor=2.0,
    max_planned_stops=20,
    enable_random_events=False,  # For evaluation
    heading_guidance_alpha=0.8,   # PPO-dominant
)

# Reset with seed for reproducibility
obs, info = env.reset(seed=42)

# Main loop
for step in range(1000):
    # 1. Get READY orders
    ready_orders = [
        oid for oid in env.active_orders 
        if env.orders[oid]['status'] == OrderStatus.READY
    ]
    
    # 2. PSO assigns routes (your PSO logic here)
    for drone_id in range(env.num_drones):
        if env.drones[drone_id]['status'] == DroneStatus.IDLE:
            # Build route plan with PSO
            route_plan = your_pso_scheduler(
                ready_orders, 
                drone_id, 
                env
            )
            
            # Apply route plan
            if route_plan:
                success = env.apply_route_plan(drone_id, route_plan)
                if not success:
                    print(f"Route plan rejected for drone {drone_id}")
    
    # 3. PPO controls heading
    action = ppo_policy(obs)
    
    # 4. Step environment
    obs, reward, done, truncated, info = env.step(action)
    
    # 5. Check metrics
    if step % 100 == 0:
        print(f"Step {step}: Completed={info['metrics']['completed_orders']}")
```

## Validation Rules

Route plans are validated before application:

1. **Structure**: Each stop must be `{'type':'P'|'D', ...}`
2. **P stops**: Must have valid `merchant_id`
3. **D stops**: Must have valid `order_id` (READY status)
4. **P-before-D**: Each order's merchant P must appear before its D
5. **Uniqueness**: Each order appears at most once (one D stop)
6. **Consistency**: All D-stop orders must be in commit_orders
7. **Capacity**: Total committed orders ≤ drone capacity
8. **Length**: Total stops ≤ max_planned_stops

## Troubleshooting

**Q: Route plan rejected with "P-before-D" error**
A: Ensure pickup stop for each order's merchant appears before delivery stop

**Q: Order stuck in ASSIGNED, never picked up**
A: Check if P stop for that merchant is in the route plan

**Q: D stop keeps retrying**
A: Order hasn't been picked up yet; ensure P stop executed successfully

**Q: Orders cancelled with "timeout"**
A: Increase `timeout_factor` or deliver orders faster

**Q: Legacy assignment warning**
A: External code is calling `_process_batch_assignment`; use `apply_route_plan()` instead

## Performance Tips

1. **Batch route planning**: Plan routes for all idle drones at once
2. **Filter READY orders**: Only include READY orders in PSO
3. **Merchant grouping**: Group orders by merchant for efficient pickup
4. **Capacity awareness**: Don't exceed drone capacity in route plans
5. **Distance optimization**: PSO should minimize total route distance
6. **Timeout awareness**: Prioritize older orders to avoid timeouts

## Metrics

Track performance with:
```python
print(env.metrics['completed_orders'])
print(env.metrics['cancelled_orders'])
print(env.metrics['cancellation_reasons'])
print(env.metrics['on_time_deliveries'])
print(env.daily_stats)
```

## References

- Main environment: `UAV_ENVIRONMENT_6.py`
- Test examples: See test scripts in repository
- Module documentation: See docstrings in code
