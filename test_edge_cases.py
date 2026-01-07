#!/usr/bin/env python3
"""
Comprehensive edge case tests for append_route_plan functionality.
Tests various edge cases and error conditions.
"""

import sys
import numpy as np
from UAV_ENVIRONMENT_6 import ThreeObjectiveDroneDeliveryEnv, OrderStatus, DroneStatus

def test_edge_cases():
    """Test edge cases and error conditions for append_route_plan."""
    print("=" * 60)
    print("Testing append_route_plan edge cases")
    print("=" * 60)
    
    env = ThreeObjectiveDroneDeliveryEnv(
        grid_size=10,
        num_drones=3,
        max_orders=30,
        steps_per_hour=4,
        drone_max_capacity=3,  # Small capacity for testing limits
        high_load_factor=0.5,
        enable_random_events=False,
        debug_state_warnings=True,
    )
    
    obs = env.reset()
    print(f"\nEnvironment initialized")
    
    # Generate orders
    for _ in range(20):
        obs, reward, done, truncated, info = env.step(np.zeros((env.num_drones, 3)))
    
    ready_orders = [oid for oid in env.active_orders 
                   if env.orders[oid]['status'] == OrderStatus.READY]
    
    print(f"Generated {len(ready_orders)} READY orders")
    
    if len(ready_orders) < 5:
        print("❌ Not enough orders for edge case testing")
        return False
    
    # Test 1: Append to drone at full capacity should fail
    print("\n" + "=" * 60)
    print("TEST 1: Append to drone at full capacity")
    print("=" * 60)
    
    drone_id = 0
    drone = env.drones[drone_id]
    
    # Fill drone to capacity
    orders_to_assign = ready_orders[:drone['max_capacity']]
    stops = []
    for oid in orders_to_assign:
        order = env.orders[oid]
        stops.extend([
            {'type': 'P', 'merchant_id': order['merchant_id']},
            {'type': 'D', 'order_id': oid},
        ])
    
    success = env.apply_route_plan(drone_id, stops)
    print(f"Initial assignment: {success}, load={drone['current_load']}/{drone['max_capacity']}")
    
    if drone['current_load'] < drone['max_capacity']:
        print("❌ Drone not at capacity, cannot test this case")
        return False
    
    # Try to append - should fail
    extra_order = [oid for oid in ready_orders if oid not in orders_to_assign][0]
    extra_stops = [
        {'type': 'P', 'merchant_id': env.orders[extra_order]['merchant_id']},
        {'type': 'D', 'order_id': extra_order},
    ]
    
    success = env.append_route_plan(drone_id, extra_stops)
    print(f"Append at capacity result: {success}")
    
    if success:
        print("❌ Test failed: append_route_plan should fail when drone at capacity")
        return False
    
    print("✅ Correctly rejected append when at capacity")
    
    # Test 2: Append to idle drone should work
    print("\n" + "=" * 60)
    print("TEST 2: Append to idle drone")
    print("=" * 60)
    
    drone_id = 1
    drone = env.drones[drone_id]
    print(f"Drone {drone_id} status: {drone['status']}, load: {drone['current_load']}")
    
    if drone['status'] != DroneStatus.IDLE:
        print("⚠ Drone not idle, skipping this test")
    else:
        remaining_orders = [oid for oid in ready_orders 
                           if env.orders[oid]['status'] == OrderStatus.READY 
                           and oid not in orders_to_assign][:2]
        
        if len(remaining_orders) >= 2:
            stops = []
            for oid in remaining_orders:
                order = env.orders[oid]
                stops.extend([
                    {'type': 'P', 'merchant_id': order['merchant_id']},
                    {'type': 'D', 'order_id': oid},
                ])
            
            success = env.append_route_plan(drone_id, stops)
            print(f"Append to idle drone: {success}, load={drone['current_load']}")
            
            if not success:
                print("❌ Test failed: append should work on idle drone")
                return False
            
            print("✅ Successfully appended to idle drone")
    
    # Test 3: Append with empty stops list
    print("\n" + "=" * 60)
    print("TEST 3: Append with empty stops")
    print("=" * 60)
    
    drone_id = 2
    success = env.append_route_plan(drone_id, [])
    print(f"Append with empty stops: {success}")
    
    if success:
        print("❌ Test failed: append should fail with empty stops")
        return False
    
    print("✅ Correctly rejected empty stops")
    
    # Test 4: Append with non-READY orders
    print("\n" + "=" * 60)
    print("TEST 4: Append with non-READY orders")
    print("=" * 60)
    
    # Find an assigned or delivered order
    non_ready_order = None
    for oid in env.active_orders:
        if env.orders[oid]['status'] != OrderStatus.READY:
            non_ready_order = oid
            break
    
    if non_ready_order:
        order = env.orders[non_ready_order]
        print(f"Order {non_ready_order} status: {order['status']}")
        
        stops = [
            {'type': 'P', 'merchant_id': order['merchant_id']},
            {'type': 'D', 'order_id': non_ready_order},
        ]
        
        drone_id = 2
        success = env.append_route_plan(drone_id, stops)
        print(f"Append non-READY order: {success}")
        
        if success:
            print("❌ Test failed: should not append non-READY orders")
            return False
        
        print("✅ Correctly rejected non-READY orders")
    else:
        print("⚠ No non-READY orders available, skipping this test")
    
    # Test 5: Verify snapshot field is consistent
    print("\n" + "=" * 60)
    print("TEST 5: Snapshot consistency")
    print("=" * 60)
    
    snapshot = env.get_drones_snapshot()
    for drone_snap in snapshot:
        drone = env.drones[drone_snap['drone_id']]
        expected_can_accept = drone['current_load'] < drone['max_capacity']
        actual_can_accept = drone_snap['can_accept_more']
        
        if expected_can_accept != actual_can_accept:
            print(f"❌ Inconsistency for drone {drone_snap['drone_id']}: "
                  f"expected={expected_can_accept}, actual={actual_can_accept}")
            return False
    
    print("✅ All snapshot fields consistent")
    
    # Test 6: Invalid drone ID
    print("\n" + "=" * 60)
    print("TEST 6: Invalid drone ID")
    print("=" * 60)
    
    success = env.append_route_plan(999, [{'type': 'P', 'merchant_id': 'test'}])
    print(f"Append to invalid drone: {success}")
    
    if success:
        print("❌ Test failed: should reject invalid drone ID")
        return False
    
    print("✅ Correctly rejected invalid drone ID")
    
    print("\n" + "=" * 60)
    print("✅ All edge case tests passed!")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    try:
        success = test_edge_cases()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
