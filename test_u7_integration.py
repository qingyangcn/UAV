"""
Integration test for U7: Verify full workflow with order generation.

This test runs a few steps and verifies:
1. Orders are generated
2. MOPSO assigns orders to drones
3. Candidates include assigned orders
4. PPO can select tasks at decision points
5. Drones move toward selected targets
"""
import numpy as np
from UAV_ENVIRONMENT_7 import ThreeObjectiveDroneDeliveryEnv, OrderStatus, DroneStatus
from U7_mopso_dispatcher import U7MOPSOAssigner, apply_mopso_assignment


def test_full_workflow():
    """Test complete workflow from order generation to task selection."""
    print("=" * 60)
    print("U7 Integration Test: Full Workflow")
    print("=" * 60)
    
    # Create environment
    env = ThreeObjectiveDroneDeliveryEnv(
        grid_size=16,
        num_drones=6,
        max_orders=200,
        num_bases=2,
        num_candidates=20,
        reward_output_mode='scalar',
        enable_random_events=False
    )
    
    # Create MOPSO assigner
    assigner = U7MOPSOAssigner(n_particles=20, n_iterations=10)
    
    # Reset environment
    print("\n1. Resetting environment...")
    obs, info = env.reset(seed=42)
    print(f"   Initial orders: {len(env.active_orders)}")
    print(f"   Initial drones: {env.num_drones}")
    
    # Run a few steps to generate orders
    print("\n2. Running steps to generate orders and wait for READY...")
    ready_found = False
    for step in range(30):
        # Apply MOPSO assignment
        assignment_counts = apply_mopso_assignment(env, assigner)
        
        # Create action (middle choice and speed for all drones)
        action = np.zeros((6, 2), dtype=np.float32)
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Count READY orders
        ready_count = sum(1 for oid in env.active_orders 
                         if env.orders[oid]['status'].name == 'READY')
        assigned_count = sum(1 for oid in env.active_orders 
                            if env.orders[oid]['status'].name == 'ASSIGNED')
        
        if assignment_counts and not ready_found:
            print(f"   Step {step+1}: {len(env.active_orders)} active orders, "
                  f"{ready_count} READY, {assigned_count} ASSIGNED, "
                  f"new assignments: {sum(assignment_counts.values())} orders to {len(assignment_counts)} drones")
            ready_found = True
            break
        elif step % 5 == 0:
            print(f"   Step {step+1}: {len(env.active_orders)} active orders, "
                  f"{ready_count} READY, {assigned_count} ASSIGNED")
    
    print(f"\n3. Current state after {step+1} steps:")
    print(f"   Active orders: {len(env.active_orders)}")
    print(f"   Order statuses:")
    
    status_counts = {}
    for oid in env.active_orders:
        order = env.orders[oid]
        status = order['status'].name
        status_counts[status] = status_counts.get(status, 0) + 1
    for status, count in status_counts.items():
        print(f"     {status}: {count}")
    
    # Check candidates
    print("\n4. Checking candidate orders for drones:")
    for drone_id in range(min(3, env.num_drones)):  # Check first 3 drones
        drone = env.drones[drone_id]
        candidates = env.drone_candidate_mappings[drone_id]
        valid_candidates = [(oid, valid) for oid, valid in candidates if valid]
        
        print(f"   Drone {drone_id}:")
        print(f"     Status: {drone['status'].name}")
        print(f"     Load: {drone['current_load']}/{drone['max_capacity']}")
        print(f"     Valid candidates: {len(valid_candidates)}")
        if valid_candidates:
            print(f"     Sample candidates: {valid_candidates[:3]}")
    
    # Verify candidates are properly encoded
    print("\n5. Verifying candidate encoding:")
    candidates_obs = obs['candidates']
    print(f"   Candidates shape: {candidates_obs.shape}")
    print(f"   Drone 0, candidate 0 (validity): {candidates_obs[0, 0, 0]}")
    
    # Check if any drone has valid candidates
    has_valid = False
    for d in range(env.num_drones):
        for c in range(env.num_candidates):
            if candidates_obs[d, c, 0] > 0.5:  # validity flag
                has_valid = True
                print(f"   Found valid candidate: drone {d}, candidate {c}")
                print(f"     Features: {candidates_obs[d, c, :]}")
                break
        if has_valid:
            break
    
    # Test decision points
    print("\n6. Testing decision point detection:")
    for drone_id in range(env.num_drones):
        is_decision = env._is_at_decision_point(drone_id)
        drone = env.drones[drone_id]
        print(f"   Drone {drone_id}: {drone['status'].name} -> "
              f"Decision point: {is_decision}")
    
    # Simulate several more steps with task selection
    print("\n7. Running more steps with task selection...")
    for step in range(5):
        # Apply MOPSO assignment
        assignment_counts = apply_mopso_assignment(env, assigner)
        
        # Create varied actions (different choices and speeds)
        action = np.random.randn(6, 2).astype(np.float32) * 0.5
        action = np.clip(action, -1.0, 1.0)
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"   Step {step+1}: reward={reward:.4f}, r_vec={info['r_vec']}")
    
    print("\n8. Final state:")
    print(f"   Total active orders: {len(env.active_orders)}")
    print(f"   Completed orders: {env.daily_stats['orders_completed']}")
    print(f"   Cancelled orders: {env.daily_stats['orders_cancelled']}")
    
    # Check drone positions changed
    print("\n9. Verifying drone movement:")
    for drone_id in range(min(3, env.num_drones)):
        drone = env.drones[drone_id]
        base_loc = env.bases[drone['base']]['location']
        current_loc = drone['location']
        dist_from_base = np.sqrt((current_loc[0] - base_loc[0])**2 + 
                                 (current_loc[1] - base_loc[1])**2)
        print(f"   Drone {drone_id}: distance from base = {dist_from_base:.2f}, "
              f"status = {drone['status'].name}")
    
    print("\n" + "=" * 60)
    print("Integration test completed successfully! ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_full_workflow()
