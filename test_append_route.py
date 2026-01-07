#!/usr/bin/env python3
"""
Test script to validate append_route_plan functionality.

This test demonstrates that drones with remaining capacity can accept additional orders
without interrupting their current route execution.

Test Scenarios:
1. Verify get_drones_snapshot includes 'can_accept_more' field
2. Test apply_route_plan for idle drones (existing behavior)
3. Test append_route_plan for busy drones (new functionality)
4. Validate cargo and planned stops preservation during append
5. Confirm drones with capacity can accept additional orders
"""

import sys
import numpy as np
from UAV_ENVIRONMENT_6 import ThreeObjectiveDroneDeliveryEnv, OrderStatus, DroneStatus

def test_append_route_plan():
    """Test that append_route_plan can add orders to a busy drone with remaining capacity."""
    print("=" * 60)
    print("Testing append_route_plan functionality")
    print("=" * 60)
    
    # Create environment with small grid and few drones
    env = ThreeObjectiveDroneDeliveryEnv(
        grid_size=10,
        num_drones=2,
        max_orders=20,
        steps_per_hour=4,
        drone_max_capacity=5,  # Allow multiple orders per drone
        high_load_factor=0.5,  # Reduce order generation
        enable_random_events=False,  # Disable random events for deterministic testing
        debug_state_warnings=True,
    )
    
    # Reset environment
    obs = env.reset()
    print(f"\nEnvironment initialized with {env.num_drones} drones")
    print(f"Drone max capacity: {env.drone_max_capacity}")
    
    # Step through a few times to generate some orders
    print("\nGenerating orders...")
    for i in range(10):
        obs, reward, done, truncated, info = env.step(np.zeros((env.num_drones, 3)))
        if i % 3 == 0:
            print(f"Step {i}: Active orders = {len(env.active_orders)}")
    
    # Find READY orders
    ready_orders = [oid for oid in env.active_orders 
                   if env.orders[oid]['status'] == OrderStatus.READY]
    
    if len(ready_orders) < 3:
        print("\n⚠ Not enough READY orders generated. Generating more...")
        for _ in range(20):
            obs, reward, done, truncated, info = env.step(np.zeros((env.num_drones, 3)))
        ready_orders = [oid for oid in env.active_orders 
                       if env.orders[oid]['status'] == OrderStatus.READY]
    
    print(f"\nFound {len(ready_orders)} READY orders")
    
    if len(ready_orders) < 3:
        print("❌ Test failed: Not enough READY orders to test")
        return False
    
    # Select first drone and verify it's idle
    drone_id = 0
    drone = env.drones[drone_id]
    print(f"\nDrone {drone_id} status: {drone['status']}")
    print(f"Drone {drone_id} capacity: {drone['current_load']}/{drone['max_capacity']}")
    
    # Get snapshot and verify can_accept_more field
    snapshot = env.get_drones_snapshot()
    drone_snapshot = snapshot[drone_id]
    print(f"\nDrone snapshot includes 'can_accept_more': {'can_accept_more' in drone_snapshot}")
    print(f"can_accept_more value: {drone_snapshot.get('can_accept_more', 'N/A')}")
    
    if 'can_accept_more' not in drone_snapshot:
        print("❌ Test failed: can_accept_more field missing from snapshot")
        return False
    
    # Test 1: Apply initial route with 2 orders
    print("\n" + "=" * 60)
    print("TEST 1: Apply initial route plan with 2 orders")
    print("=" * 60)
    
    order1 = env.orders[ready_orders[0]]
    order2 = env.orders[ready_orders[1]]
    
    # Build route plan for first 2 orders
    initial_stops = [
        {'type': 'P', 'merchant_id': order1['merchant_id']},
        {'type': 'D', 'order_id': ready_orders[0]},
        {'type': 'P', 'merchant_id': order2['merchant_id']},
        {'type': 'D', 'order_id': ready_orders[1]},
    ]
    
    success = env.apply_route_plan(drone_id, initial_stops)
    print(f"apply_route_plan result: {success}")
    
    if not success:
        print("❌ Test failed: Could not apply initial route plan")
        return False
    
    print(f"Drone {drone_id} current_load after initial route: {drone['current_load']}")
    print(f"Drone {drone_id} planned_stops count: {len(drone['planned_stops'])}")
    print(f"Drone {drone_id} status: {drone['status']}")
    
    # Step a few times to let drone start executing
    print("\nExecuting route for a few steps...")
    for i in range(5):
        obs, reward, done, truncated, info = env.step(np.zeros((env.num_drones, 3)))
        print(f"  Step {i}: Drone status = {drone['status']}, location = ({drone['location'][0]:.2f}, {drone['location'][1]:.2f})")
    
    # Test 2: Append additional order while drone is busy
    print("\n" + "=" * 60)
    print("TEST 2: Append additional order to busy drone")
    print("=" * 60)
    
    print(f"Drone {drone_id} status before append: {drone['status']}")
    print(f"Drone {drone_id} current_load before append: {drone['current_load']}/{drone['max_capacity']}")
    print(f"Drone {drone_id} can_accept_more: {drone['current_load'] < drone['max_capacity']}")
    
    if drone['current_load'] >= drone['max_capacity']:
        print("❌ Test failed: Drone already at capacity")
        return False
    
    # Find another READY order
    remaining_ready = [oid for oid in ready_orders[2:] 
                      if oid in env.orders and env.orders[oid]['status'] == OrderStatus.READY]
    
    if not remaining_ready:
        print("❌ Test failed: No more READY orders available")
        return False
    
    order3 = env.orders[remaining_ready[0]]
    
    # Build additional stops
    additional_stops = [
        {'type': 'P', 'merchant_id': order3['merchant_id']},
        {'type': 'D', 'order_id': remaining_ready[0]},
    ]
    
    # Save state before append
    stops_before = len(drone['planned_stops'])
    cargo_before = drone['cargo'].copy()
    
    # Append route plan
    success = env.append_route_plan(drone_id, additional_stops)
    print(f"append_route_plan result: {success}")
    
    if not success:
        print("❌ Test failed: Could not append route plan")
        return False
    
    print(f"Drone {drone_id} current_load after append: {drone['current_load']}")
    print(f"Drone {drone_id} planned_stops count: before={stops_before}, after={len(drone['planned_stops'])}")
    print(f"Cargo preserved: {cargo_before.issubset(drone['cargo'])}")
    
    # Verify that stops were appended, not replaced
    if len(drone['planned_stops']) <= stops_before:
        print("❌ Test failed: Stops were not appended (route was replaced or cleared)")
        return False
    
    # Verify that cargo was preserved (existing cargo should still be there)
    if not cargo_before.issubset(drone['cargo']):
        print("❌ Test failed: Cargo was not preserved (existing cargo was lost)")
        return False
    
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)
    print("\nSummary:")
    print("- get_drones_snapshot includes 'can_accept_more' field")
    print("- apply_route_plan works for idle drones")
    print("- append_route_plan extends routes for busy drones")
    print("- append_route_plan preserves existing cargo and stops")
    print("- Drones with remaining capacity can accept additional orders")
    
    return True

if __name__ == "__main__":
    try:
        success = test_append_route_plan()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
