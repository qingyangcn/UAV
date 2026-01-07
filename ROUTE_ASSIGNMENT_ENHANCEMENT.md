# UAV Route Assignment Enhancement - Documentation

## Overview

This document describes the enhancement to the UAV environment that enables dispatch/route assignment to drones that have not reached their load capacity, not only idle drones.

## Changes Made

### 1. New Method: `append_route_plan()`

**Location**: `UAV_ENVIRONMENT_6.py`

**Purpose**: Allows assigning additional orders to drones that are currently busy but still have remaining capacity.

**Signature**:
```python
def append_route_plan(self,
                      drone_id: int,
                      planned_stops: List[dict],
                      commit_orders: Optional[List[int]] = None) -> bool
```

**Parameters**:
- `drone_id`: ID of the drone to extend
- `planned_stops`: List of stops to append (format: `[{'type':'P','merchant_id':mid}, {'type':'D','order_id':oid}, ...]`)
- `commit_orders`: Optional list of order IDs to commit (if None, derived from planned_stops)

**Returns**: `True` if route was successfully extended, `False` otherwise

**Key Features**:
- **Non-destructive**: Appends to existing `planned_stops` instead of replacing them
- **Cargo preservation**: Does NOT reset `drone['cargo']` or clear existing picked-up orders
- **Capacity check**: Only accepts new orders if `current_load < max_capacity`
- **State preservation**: Does NOT reset `current_stop` or interrupt current target
- **Safe order commitment**: Only commits READY orders to ASSIGNED state

**Constraints**:
- Drone must have remaining capacity (`current_load < max_capacity`)
- Only READY orders can be committed
- Does not interrupt current route execution
- Filters out invalid delivery stops

**Example Usage**:
```python
# Drone is busy with 2 orders but has capacity for 5
drone_id = 0
additional_stops = [
    {'type': 'P', 'merchant_id': 'merchant_3'},
    {'type': 'D', 'order_id': 123},
]
success = env.append_route_plan(drone_id, additional_stops)
```

### 2. Enhanced Method: `get_drones_snapshot()`

**Location**: `UAV_ENVIRONMENT_6.py`

**Change**: Added new field `can_accept_more` to drone snapshot

**New Field**:
- `can_accept_more`: Boolean indicating if drone can accept more orders (`current_load < max_capacity`)

**Purpose**: Helps dispatchers quickly identify eligible drones for additional assignments

**Example Output**:
```python
{
    'drone_id': 0,
    'location': (5.2, 3.8),
    'base': 0,
    'status': DroneStatus.FLYING_TO_CUSTOMER,
    'battery_level': 85.0,
    'current_load': 2,
    'max_capacity': 5,
    'speed': 4.2,
    'battery_consumption_rate': 0.3,
    'has_route': True,
    'can_accept_more': True,  # <-- NEW FIELD
}
```

## Backward Compatibility

### Existing `apply_route_plan()` behavior is unchanged:
- Still replaces entire route when called
- Still resets cargo and planned_stops
- Still works as before for idle drones
- No changes to method signature or semantics

### New functionality is additive:
- `append_route_plan()` is a new method - existing code unaffected
- `can_accept_more` is a new field - existing code can ignore it
- No breaking changes to any existing APIs

## Implementation Details

### Safety Mechanisms

1. **Capacity Validation**:
   ```python
   if drone['current_load'] >= drone['max_capacity']:
       return False
   ```

2. **Order State Validation**:
   - Only READY orders can be committed
   - Orders already assigned to another drone are rejected
   - Invalid stops are filtered out

3. **Route Preservation**:
   ```python
   # Initialize if needed (don't reset!)
   if 'planned_stops' not in drone or drone['planned_stops'] is None:
       drone['planned_stops'] = deque()
   
   # Append to existing route
   for stop in filtered_stops:
       drone['planned_stops'].append(stop)
   ```

4. **Execution Management**:
   - Only starts route execution if drone was idle
   - If drone is already executing, new stops are simply queued
   - Does not interrupt current target location

### Differences from `apply_route_plan()`

| Aspect | `apply_route_plan()` | `append_route_plan()` |
|--------|---------------------|----------------------|
| Route handling | Replaces entire route | Appends to existing route |
| Cargo | Resets to empty set | Preserves existing cargo |
| Current stop | Resets to None | Preserves current stop |
| Execution | Always starts new route | Only starts if idle |
| Use case | Initial assignment to idle drones | Adding orders to busy drones |
| Capacity check | Via commit loop | Explicit upfront check |

## Testing

A comprehensive test is provided in `test_append_route.py` that validates:

1. ✅ `get_drones_snapshot()` includes `can_accept_more` field
2. ✅ `apply_route_plan()` works for idle drones (existing behavior)
3. ✅ `append_route_plan()` extends routes for busy drones
4. ✅ `append_route_plan()` preserves existing cargo and stops
5. ✅ Drones with remaining capacity can accept additional orders

**Run the test**:
```bash
cd /home/runner/work/UAV/UAV
python3 test_append_route.py
```

## Use Cases

### Use Case 1: Multi-Wave Dispatch
A dispatcher can assign initial orders to drones, then as more READY orders become available, append them to drones that still have capacity:

```python
# Initial dispatch
env.apply_route_plan(drone_0, initial_route_plan_0)
env.apply_route_plan(drone_1, initial_route_plan_1)

# Later, more orders become ready
snapshot = env.get_drones_snapshot()
for drone_snap in snapshot:
    if drone_snap['can_accept_more']:
        # Drone has capacity - add more orders
        env.append_route_plan(
            drone_snap['drone_id'],
            additional_stops
        )
```

### Use Case 2: Dynamic Rebalancing
During peak hours, efficiently utilize drone capacity by continuously adding orders:

```python
ready_orders = env.get_ready_orders_snapshot(limit=100)
drones = env.get_drones_snapshot()

# Filter drones that can accept more
available_drones = [d for d in drones if d['can_accept_more']]

# Assign orders to available drones
for drone in available_drones:
    if ready_orders:
        # Build route for this drone
        route = build_route_for_drone(drone, ready_orders)
        success = env.append_route_plan(drone['drone_id'], route)
```

### Use Case 3: Opportunistic Assignment
When a drone completes one delivery in a multi-order route, check if nearby orders can be added:

```python
# Monitor drone progress
if drone['status'] == DroneStatus.FLYING_TO_CUSTOMER:
    if drone['current_load'] < drone['max_capacity']:
        # Check for nearby ready orders
        nearby_orders = find_nearby_ready_orders(drone['location'])
        if nearby_orders:
            route = build_route(nearby_orders)
            env.append_route_plan(drone_id, route)
```

## API Reference

### append_route_plan()

Append additional stops to a drone's existing route plan.

**Arguments**:
- `drone_id` (int): Drone identifier
- `planned_stops` (List[dict]): Stops to append
  - Format: `[{'type':'P','merchant_id':...}, {'type':'D','order_id':...}, ...]`
- `commit_orders` (Optional[List[int]]): Order IDs to commit (auto-derived if None)

**Returns**: bool
- `True`: Route successfully extended
- `False`: Failed (drone at capacity, no valid orders, etc.)

**Raises**: No exceptions - returns False on error

### get_drones_snapshot()

Get snapshot of all drones for scheduling.

**Arguments**: None

**Returns**: List[dict]
- List of drone snapshots with fields:
  - `drone_id`: Drone identifier
  - `location`: Current (x, y) position
  - `base`: Base station ID
  - `status`: DroneStatus enum
  - `battery_level`: Current battery percentage
  - `current_load`: Number of assigned orders
  - `max_capacity`: Maximum order capacity
  - `speed`: Flight speed
  - `battery_consumption_rate`: Battery drain rate
  - `has_route`: Whether drone has committed route
  - `can_accept_more`: **NEW** - Whether drone can accept more orders

## Migration Guide

If you're using an external dispatcher/scheduler:

1. **Check drone capacity before assignment**:
   ```python
   # OLD: Only check if idle
   if drone['status'] == DroneStatus.IDLE:
       env.apply_route_plan(drone_id, route)
   
   # NEW: Also check capacity for busy drones
   snapshot = env.get_drones_snapshot()
   drone_snap = snapshot[drone_id]
   if drone_snap['can_accept_more']:
       env.append_route_plan(drone_id, route)
   ```

2. **Use the new snapshot field**:
   ```python
   # Filter drones that can accept orders
   available = [d for d in env.get_drones_snapshot() 
                if d['can_accept_more']]
   ```

3. **Choose the right method**:
   - Use `apply_route_plan()` for idle drones or full route replacement
   - Use `append_route_plan()` to extend routes of busy drones

## Performance Considerations

- `append_route_plan()` is lightweight - O(n) where n = number of new stops
- No route recomputation is triggered
- Existing route execution continues without interruption
- Orders are validated before commitment to avoid invalid states

## Future Enhancements

Potential future improvements:
1. Support for route optimization when appending stops
2. Configurable insertion points (not just append to end)
3. Route merging and consolidation
4. Dynamic capacity adjustment based on battery level
5. Priority-based insertion for urgent orders

## Questions & Support

For questions or issues related to this enhancement, please refer to:
- Test file: `test_append_route.py`
- Implementation: `UAV_ENVIRONMENT_6.py` (lines ~1996-2110)
- This documentation: `ROUTE_ASSIGNMENT_ENHANCEMENT.md`
