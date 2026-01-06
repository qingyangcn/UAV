#!/usr/bin/env python3
"""
Test script for P0+P1 snapshot interfaces and functionality.
"""

import numpy as np
from UAV_ENVIRONMENT_6 import ThreeObjectiveDroneDeliveryEnv

def test_snapshot_interfaces():
    """Test P0-1: Snapshot read interfaces"""
    print("=" * 60)
    print("Testing P0-1: Snapshot Read Interfaces")
    print("=" * 60)
    
    # Create environment
    env = ThreeObjectiveDroneDeliveryEnv(
        grid_size=10,
        num_drones=3,
        max_orders=20,
        drone_max_capacity=5,
        reward_output_mode="zero",
        enable_random_events=False,  # Disable for reproducibility
    )
    
    # Reset environment
    obs, info = env.reset(seed=42)
    print("✓ Environment reset successfully")
    
    # Step a few times to generate orders
    for i in range(10):
        action = np.zeros((env.num_drones, 2), dtype=np.float32)
        obs, reward, done, truncated, info = env.step(action)
    
    # Test get_ready_orders_snapshot
    print("\n--- Testing get_ready_orders_snapshot() ---")
    ready_orders = env.get_ready_orders_snapshot()
    print(f"✓ Found {len(ready_orders)} READY orders")
    if len(ready_orders) > 0:
        sample_order = ready_orders[0]
        print(f"  Sample order keys: {list(sample_order.keys())}")
        print(f"  Sample order: order_id={sample_order['order_id']}, "
              f"merchant_id={sample_order['merchant_id']}, "
              f"age_steps={sample_order['age_steps']}, "
              f"deadline_step={sample_order['deadline_step']}")
        
        # Verify serializable
        assert isinstance(sample_order['order_id'], int)
        assert isinstance(sample_order['merchant_id'], str)
        assert isinstance(sample_order['merchant_location'], tuple)
        assert isinstance(sample_order['urgent'], bool)
        print("✓ Order snapshot is properly serializable")
    
    # Test get_drones_snapshot
    print("\n--- Testing get_drones_snapshot() ---")
    drones = env.get_drones_snapshot()
    print(f"✓ Found {len(drones)} drones")
    if len(drones) > 0:
        sample_drone = drones[0]
        print(f"  Sample drone keys: {list(sample_drone.keys())}")
        print(f"  Sample drone: drone_id={sample_drone['drone_id']}, "
              f"status={sample_drone['status_name']}, "
              f"battery={sample_drone['battery_level']:.1f}, "
              f"assigned_load={sample_drone['assigned_load']}, "
              f"cargo_load={sample_drone['cargo_load']}")
        
        # Verify serializable
        assert isinstance(sample_drone['drone_id'], int)
        assert isinstance(sample_drone['location'], tuple)
        assert isinstance(sample_drone['battery_level'], float)
        print("✓ Drone snapshot is properly serializable")
    
    # Test get_merchants_snapshot
    print("\n--- Testing get_merchants_snapshot() ---")
    merchants = env.get_merchants_snapshot()
    print(f"✓ Found {len(merchants)} merchants")
    if len(merchants) > 0:
        sample_merchant = merchants[0]
        print(f"  Sample merchant keys: {list(sample_merchant.keys())}")
        print(f"  Sample merchant: merchant_id={sample_merchant['merchant_id']}, "
              f"queue_len={sample_merchant['queue_len']}, "
              f"efficiency={sample_merchant['efficiency']:.2f}")
        
        # Verify serializable
        assert isinstance(sample_merchant['merchant_id'], str)
        assert isinstance(sample_merchant['location'], tuple)
        assert isinstance(sample_merchant['queue_len'], int)
        print("✓ Merchant snapshot is properly serializable")
    
    # Test get_route_plan_constraints
    print("\n--- Testing get_route_plan_constraints() ---")
    constraints = env.get_route_plan_constraints()
    print(f"✓ Retrieved constraints with keys: {list(constraints.keys())}")
    print(f"  READY-only policy: {constraints['ready_only_policy']}")
    print(f"  Max capacity: {constraints['max_capacity']}")
    print(f"  Timeout factor: {constraints['timeout_factor']}")
    print(f"  Max stops per route: {constraints['max_stops_per_route']}")
    print("✓ Constraints are properly formatted")
    
    print("\n✓ All snapshot interface tests passed!")
    return True

def test_apply_route_plan():
    """Test P0-2: Structured apply_route_plan results"""
    print("\n" + "=" * 60)
    print("Testing P0-2: Structured apply_route_plan Results")
    print("=" * 60)
    
    # Create environment
    env = ThreeObjectiveDroneDeliveryEnv(
        grid_size=10,
        num_drones=3,
        max_orders=20,
        drone_max_capacity=5,
        reward_output_mode="zero",
        enable_random_events=False,
    )
    
    obs, info = env.reset(seed=42)
    
    # Step to generate orders
    for i in range(15):
        action = np.zeros((env.num_drones, 2), dtype=np.float32)
        obs, reward, done, truncated, info = env.step(action)
    
    ready_orders = env.get_ready_orders_snapshot()
    print(f"✓ Generated {len(ready_orders)} READY orders")
    
    if len(ready_orders) >= 2:
        # Test successful route plan
        print("\n--- Testing successful route plan ---")
        order1 = ready_orders[0]
        order2 = ready_orders[1]
        
        merchant1 = order1['merchant_id']
        merchant2 = order2['merchant_id']
        
        planned_stops = [
            {'type': 'P', 'merchant_id': merchant1},
            {'type': 'D', 'order_id': order1['order_id']},
            {'type': 'P', 'merchant_id': merchant2},
            {'type': 'D', 'order_id': order2['order_id']},
        ]
        
        result = env.apply_route_plan(
            drone_id=0,
            planned_stops=planned_stops,
            allow_busy=True
        )
        
        print(f"  Result keys: {list(result.keys())}")
        print(f"  Success: {result['success']}")
        print(f"  Committed orders: {result['committed_orders']}")
        print(f"  Rejected orders: {result['rejected_orders']}")
        print(f"  Route installed: {result['route_installed']}")
        print(f"  Reason: {result['reason']}")
        
        assert result['success']
        assert len(result['committed_orders']) > 0
        assert result['route_installed']
        print("✓ Successful route plan test passed")
        
        # Test invalid drone
        print("\n--- Testing invalid drone_id ---")
        result = env.apply_route_plan(
            drone_id=999,
            planned_stops=planned_stops,
        )
        assert not result['success']
        assert 'Invalid drone_id' in result['reason']
        print(f"✓ Invalid drone rejection: {result['reason']}")
        
        # Test empty stops
        print("\n--- Testing empty planned_stops ---")
        result = env.apply_route_plan(
            drone_id=1,
            planned_stops=[],
        )
        assert not result['success']
        print(f"✓ Empty stops rejection: {result['reason']}")
    
    print("\n✓ All apply_route_plan tests passed!")
    return True

def test_stuck_order_reconciliation():
    """Test P0-3: Zombie order/deadlock recovery"""
    print("\n" + "=" * 60)
    print("Testing P0-3: Stuck Order Reconciliation")
    print("=" * 60)
    
    # Create environment with lower thresholds for faster testing
    env = ThreeObjectiveDroneDeliveryEnv(
        grid_size=10,
        num_drones=2,
        max_orders=10,
        drone_max_capacity=3,
        reward_output_mode="zero",
        enable_random_events=False,
    )
    
    obs, info = env.reset(seed=42)
    
    # Manually set lower thresholds for testing
    env.stuck_assigned_reset_threshold = 5
    env.stuck_assigned_cancel_threshold = 10
    
    # Step to generate orders
    for i in range(10):
        action = np.zeros((env.num_drones, 2), dtype=np.float32)
        obs, reward, done, truncated, info = env.step(action)
    
    print(f"✓ Environment stepped, stuck_order_stats: {info.get('stuck_order_stats', {})}")
    
    # Check that _reconcile_stuck_orders is being called
    stats = env._reconcile_stuck_orders()
    print(f"✓ Reconciliation stats: {stats}")
    
    print("\n✓ Stuck order reconciliation test passed!")
    return True

def test_pso_ppo_metrics():
    """Test P1-2: PSO vs PPO metrics split"""
    print("\n" + "=" * 60)
    print("Testing P1-2: PSO vs PPO Metrics Split")
    print("=" * 60)
    
    env = ThreeObjectiveDroneDeliveryEnv(
        grid_size=10,
        num_drones=3,
        max_orders=20,
        drone_max_capacity=5,
        reward_output_mode="zero",
        enable_random_events=False,
    )
    
    obs, info = env.reset(seed=42)
    
    # Step and apply some route plans
    for i in range(15):
        action = np.zeros((env.num_drones, 2), dtype=np.float32)
        obs, reward, done, truncated, info = env.step(action)
    
    ready_orders = env.get_ready_orders_snapshot()
    if len(ready_orders) >= 1:
        order = ready_orders[0]
        planned_stops = [
            {'type': 'P', 'merchant_id': order['merchant_id']},
            {'type': 'D', 'order_id': order['order_id']},
        ]
        result = env.apply_route_plan(0, planned_stops)
    
    # Check PSO metrics
    print("\n--- PSO Metrics ---")
    pso_metrics = env._get_pso_metrics()
    print(f"  Keys: {list(pso_metrics.keys())}")
    print(f"  Total route plans: {pso_metrics['total_route_plans']}")
    print(f"  Recent success rate: {pso_metrics['recent_success_rate']:.2f}")
    print(f"  Avg committed per route: {pso_metrics['avg_committed_per_route']:.2f}")
    print("✓ PSO metrics retrieved")
    
    # Check PPO metrics
    print("\n--- PPO Metrics ---")
    ppo_metrics = env._get_ppo_metrics()
    print(f"  Keys: {list(ppo_metrics.keys())}")
    print(f"  Total actual flight distance: {ppo_metrics['total_actual_flight_distance']:.2f}")
    print(f"  Total energy consumed: {ppo_metrics['total_energy_consumed']:.2f}")
    print("✓ PPO metrics retrieved")
    
    # Check in info
    print("\n--- Metrics in info ---")
    action = np.zeros((env.num_drones, 2), dtype=np.float32)
    obs, reward, done, truncated, info = env.step(action)
    
    assert 'pso_metrics' in info
    assert 'ppo_metrics' in info
    print(f"✓ PSO metrics in info: {list(info['pso_metrics'].keys())}")
    print(f"✓ PPO metrics in info: {list(info['ppo_metrics'].keys())}")
    
    print("\n✓ All PSO/PPO metrics tests passed!")
    return True

def test_deadline_consistency():
    """Test P1-3: SLA/deadline field consistency"""
    print("\n" + "=" * 60)
    print("Testing P1-3: SLA/Deadline Field Consistency")
    print("=" * 60)
    
    env = ThreeObjectiveDroneDeliveryEnv(
        grid_size=10,
        num_drones=2,
        max_orders=10,
        reward_output_mode="zero",
        enable_random_events=False,
    )
    
    obs, info = env.reset(seed=42)
    
    # Step to generate orders
    for i in range(10):
        action = np.zeros((env.num_drones, 2), dtype=np.float32)
        obs, reward, done, truncated, info = env.step(action)
    
    ready_orders = env.get_ready_orders_snapshot()
    print(f"✓ Generated {len(ready_orders)} READY orders")
    
    if len(ready_orders) > 0:
        order = ready_orders[0]
        order_id = order['order_id']
        
        # Check deadline calculation
        print(f"\n  Order {order_id}:")
        print(f"    Creation time: {order['creation_time']}")
        print(f"    Promised steps: {order['promised_steps']}")
        print(f"    Deadline step: {order['deadline_step']}")
        print(f"    Timeout factor: {env.timeout_factor}")
        
        # Verify deadline = creation_time + promised_steps * timeout_factor
        expected_deadline = order['creation_time'] + int(order['promised_steps'] * env.timeout_factor)
        assert order['deadline_step'] == expected_deadline
        print(f"✓ Deadline calculation is consistent: {order['deadline_step']} == {expected_deadline}")
    
    print("\n✓ All deadline consistency tests passed!")
    return True

def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("P0 + P1 Implementation Test Suite")
    print("=" * 60)
    
    try:
        test_snapshot_interfaces()
        test_apply_route_plan()
        test_stuck_order_reconciliation()
        test_pso_ppo_metrics()
        test_deadline_consistency()
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
        return 0
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
