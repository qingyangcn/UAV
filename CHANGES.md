# MOPSO Dispatcher Capacity-Based Dispatch Enhancement

## Overview
Updated the MOPSO (Multi-Objective Particle Swarm Optimization) dispatcher to enable dispatching orders to drones based on remaining capacity rather than only IDLE status. This allows the dispatcher to assign additional orders to busy drones that have capacity available.

## Problem Statement
Previously, the MOPSO dispatcher could only dispatch orders to drones that were:
- In IDLE status
- Did not have an active route (`has_route=False`)

This limitation prevented efficient utilization of drones that were already executing routes but had remaining capacity to accept more orders.

## Solution
The dispatcher now:
1. **Filters drones by capacity** rather than IDLE status
2. **Safely appends to existing routes** using `env.append_route_plan()` for busy drones
3. **Maintains backward compatibility** with older environments

## Changes Made

### 1. U6_mopso_dispatcher.py

#### MOPSOPlanner class
- **Updated class docstring** to reflect capacity-based dispatch capability
- **Changed drone filtering logic** in `mopso_dispatch()`:
  - Old: `if d['status'].name == 'IDLE' and not d.get('has_route', False)`
  - New: `if d.get('can_accept_more', d.get('current_load', 0) < max(d.get('max_capacity', 1), 1))`
- **Renamed variables**: `idle_drones` → `eligible_drones`
- **Updated docstrings**: `_run_mopso()` now documents it works with eligible drones

#### apply_mopso_dispatch function
- **Added smart route application logic**:
  ```python
  if has_route and has_append_method:
      # Drone has existing route - append to it
      success = env.append_route_plan(drone_id, planned_stops, commit_orders)
  else:
      # Drone has no route or env doesn't support append - use apply with allow_busy
      success = env.apply_route_plan(drone_id, planned_stops, commit_orders, allow_busy=True)
  ```
- **Backward compatibility**: Checks for `append_route_plan` existence before using it

### 2. test_mopso_capacity_dispatch.py (New File)
Created comprehensive integration tests validating:
- Capacity-based filtering excludes drones at max capacity
- `append_route_plan` is used for drones with existing routes
- `apply_route_plan` with `allow_busy=True` is used for drones without routes
- Backward compatibility with environments lacking `append_route_plan`

### 3. .gitignore (New File)
Added Python cache files to prevent committing `__pycache__` directories.

## Technical Details

### Capacity-Based Filtering
The dispatcher now uses the `can_accept_more` field from drone snapshots, with a fallback to calculating `current_load < max_capacity`. This allows:
- IDLE drones with capacity to be selected
- FLYING drones with remaining capacity to be selected
- Drones at max capacity to be excluded (regardless of status)

### Safe Route Modification
- **For drones with `has_route=True`**: Uses `append_route_plan()` which extends the existing route without clearing cargo or disrupting execution
- **For drones with `has_route=False`**: Uses `apply_route_plan()` with `allow_busy=True` which installs a new route

### Backward Compatibility
The dispatcher checks `hasattr(env, 'append_route_plan')` before using the append method. If not available, it falls back to `apply_route_plan()` with `allow_busy=True`.

## Testing

All tests pass:
```
=== Test 1: Capacity-based filtering ===
✓ Only eligible drones (with remaining capacity) got plans

=== Test 2: Append vs Apply logic ===
✓ Drone 1 (has_route=True) used append_route_plan
✓ Drone 0 (has_route=False) used apply_route_plan with allow_busy=True
✓ All appended plans went to drones with existing routes
✓ All applied plans used correct allow_busy flag

=== Test 3: Backward compatibility ===
✓ Backward compatibility maintained
```

## Security
- CodeQL security scan: **0 alerts** ✓
- Code review: All feedback addressed ✓

## Acceptance Criteria

All acceptance criteria from the problem statement are met:

✅ **Dispatcher selects drones using capacity rather than IDLE status**
- Filters using `can_accept_more` or `current_load < max_capacity`

✅ **Plans are applied safely**
- Route replacement only when drone has no active route
- Append when drone has active route

✅ **No disruption of busy drones' in-progress route/cargo**
- Uses `append_route_plan()` which doesn't clear `cargo` set
- Doesn't reset `planned_stops` or `current_stop`

✅ **Backward compatibility maintained**
- Checks for `append_route_plan` existence
- Falls back gracefully to `apply_route_plan` with `allow_busy=True`

## Impact
This change enables more efficient drone utilization by allowing the MOPSO dispatcher to:
- Assign additional orders to drones already in flight
- Better utilize available capacity across the fleet
- Reduce order wait times by leveraging busy drones with capacity

## Files Modified
- `U6_mopso_dispatcher.py` - Core dispatcher logic
- `test_mopso_capacity_dispatch.py` - Integration tests (new)
- `.gitignore` - Python exclusions (new)
