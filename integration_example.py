#!/usr/bin/env python3
"""
Integration example: External PSO + PPO with UAV Environment 6

Demonstrates:
- Using snapshot interfaces to get environment state
- External PSO-style route planning
- Applying route plans with structured results
- PPO-style path control
- Monitoring PSO vs PPO metrics
"""

import numpy as np
from UAV_ENVIRONMENT_6 import ThreeObjectiveDroneDeliveryEnv


def simple_greedy_pso(ready_orders, drones, merchants, constraints):
    """
    Simple greedy PSO-style scheduler.
    For each idle drone, assign the oldest ready order within capacity.
    
    Returns: List of (drone_id, planned_stops) tuples
    """
    route_plans = []
    max_capacity = constraints['max_capacity']
    
    # Get idle drones
    idle_drones = [d for d in drones if d['status_name'] == 'IDLE' and d['battery_level'] > 20]
    
    if not idle_drones or not ready_orders:
        return route_plans
    
    # Sort orders by age (oldest first)
    sorted_orders = sorted(ready_orders, key=lambda o: o['age_steps'], reverse=True)
    
    assigned_order_ids = set()
    
    for drone in idle_drones:
        drone_id = drone['drone_id']
        available_capacity = max_capacity - drone['assigned_load']
        
        if available_capacity <= 0:
            continue
        
        # Select orders for this drone
        selected_orders = []
        for order in sorted_orders:
            if order['order_id'] in assigned_order_ids:
                continue
            if len(selected_orders) >= available_capacity:
                break
            selected_orders.append(order)
            assigned_order_ids.add(order['order_id'])
        
        if not selected_orders:
            continue
        
        # Build route plan: group by merchant, then interleave P/D stops
        merchant_groups = {}
        for order in selected_orders:
            mid = order['merchant_id']
            if mid not in merchant_groups:
                merchant_groups[mid] = []
            merchant_groups[mid].append(order)
        
        # Simple route: visit each merchant, then deliver all from that merchant
        planned_stops = []
        for merchant_id, orders in merchant_groups.items():
            planned_stops.append({'type': 'P', 'merchant_id': merchant_id})
            for order in orders:
                planned_stops.append({'type': 'D', 'order_id': order['order_id']})
        
        route_plans.append((drone_id, planned_stops))
    
    return route_plans


def simple_ppo_policy(obs, env):
    """
    Simple PPO-style policy: random heading for demonstration.
    In real use, this would be a trained neural network.
    """
    # Random heading (normalized)
    action = np.random.randn(env.num_drones, 2).astype(np.float32)
    # Normalize to unit vectors
    for i in range(env.num_drones):
        norm = np.linalg.norm(action[i])
        if norm > 1e-6:
            action[i] /= norm
    return action


def main():
    print("=" * 70)
    print("Integration Example: External PSO + PPO with UAV Environment 6")
    print("=" * 70)
    
    # Create environment
    env = ThreeObjectiveDroneDeliveryEnv(
        grid_size=16,
        num_drones=6,
        max_orders=50,
        drone_max_capacity=5,
        reward_output_mode="zero",
        enable_random_events=False,  # Disable for reproducibility
    )
    
    obs, info = env.reset(seed=42)
    print(f"\n✓ Environment initialized:")
    print(f"  Grid size: {env.grid_size}x{env.grid_size}")
    print(f"  Drones: {env.num_drones}")
    print(f"  Max capacity: {env.drone_max_capacity}")
    print(f"  Timeout factor: {env.timeout_factor}")
    
    # Get constraints
    constraints = env.get_route_plan_constraints()
    print(f"\n✓ Route plan constraints:")
    print(f"  READY-only policy: {constraints['ready_only_policy']}")
    print(f"  Max stops per route: {constraints['max_stops_per_route']}")
    
    total_planned_routes = 0
    total_successful_routes = 0
    
    # Main loop
    for step in range(200):
        # === External PSO: Plan routes ===
        ready_orders = env.get_ready_orders_snapshot()
        drones = env.get_drones_snapshot()
        merchants = env.get_merchants_snapshot()
        
        # Apply PSO planning every 5 steps
        if step % 5 == 0 and ready_orders:
            route_plans = simple_greedy_pso(ready_orders, drones, merchants, constraints)
            
            for drone_id, planned_stops in route_plans:
                result = env.apply_route_plan(
                    drone_id=drone_id,
                    planned_stops=planned_stops,
                    allow_busy=True
                )
                
                total_planned_routes += 1
                if result['success']:
                    total_successful_routes += 1
                    if step < 50:  # Print first few for visibility
                        print(f"\n  Step {step}: ✓ Drone {drone_id} assigned "
                              f"{len(result['committed_orders'])} orders")
                        print(f"    Stops: {len(planned_stops)}")
                else:
                    if step < 50:
                        print(f"\n  Step {step}: ✗ Drone {drone_id} plan failed: {result['reason']}")
        
        # === PPO: Generate heading actions ===
        ppo_action = simple_ppo_policy(obs, env)
        
        # Step environment
        obs, reward, done, truncated, info = env.step(ppo_action)
        
        # Print metrics every 50 steps
        if step % 50 == 0 and step > 0:
            print(f"\n{'='*70}")
            print(f"Step {step} Metrics:")
            print(f"{'='*70}")
            
            # PSO metrics
            pso_metrics = info['pso_metrics']
            print(f"\nPSO (Planning) Metrics:")
            print(f"  Total route plans: {pso_metrics['total_route_plans']}")
            print(f"  Success rate: {pso_metrics['recent_success_rate']:.1%}")
            print(f"  Avg committed/route: {pso_metrics['avg_committed_per_route']:.2f}")
            print(f"  Avg planned distance: {pso_metrics['avg_planned_distance']:.2f}")
            
            # PPO metrics
            ppo_metrics = info['ppo_metrics']
            print(f"\nPPO (Execution) Metrics:")
            print(f"  Total actual flight: {ppo_metrics['total_actual_flight_distance']:.2f}")
            print(f"  Total energy: {ppo_metrics['total_energy_consumed']:.2f}")
            print(f"  Avg distance/order: {ppo_metrics['avg_distance_per_order']:.2f}")
            print(f"  Avg energy/order: {ppo_metrics['avg_energy_per_order']:.2f}")
            
            # Stuck order stats
            stuck_stats = info['stuck_order_stats']
            total_stuck = sum(stuck_stats.values())
            if total_stuck > 0:
                print(f"\nStuck Order Recovery:")
                for reason, count in stuck_stats.items():
                    if count > 0:
                        print(f"  {reason}: {count}")
            
            # General stats
            print(f"\nGeneral Stats:")
            print(f"  Active orders: {info['backlog_size']}")
            print(f"  Completed: {info['metrics']['completed_orders']}")
            print(f"  Canceled: {info['metrics']['cancelled_orders']}")
            print(f"  On-time rate: {info.get('on_time_rate', 0):.1%}")
        
        if done:
            print(f"\n✓ Episode completed at step {step}")
            break
    
    # Final summary
    print(f"\n{'='*70}")
    print("Final Summary")
    print(f"{'='*70}")
    
    print(f"\nRoute Planning (PSO):")
    print(f"  Total routes planned: {total_planned_routes}")
    print(f"  Successful routes: {total_successful_routes}")
    if total_planned_routes > 0:
        print(f"  Overall success rate: {total_successful_routes/total_planned_routes:.1%}")
    
    final_metrics = info['metrics']
    print(f"\nDelivery Performance:")
    print(f"  Total orders: {final_metrics['total_orders']}")
    print(f"  Completed: {final_metrics['completed_orders']}")
    print(f"  Canceled: {final_metrics['cancelled_orders']}")
    
    if final_metrics['completed_orders'] > 0:
        completion_rate = final_metrics['completed_orders'] / final_metrics['total_orders']
        print(f"  Completion rate: {completion_rate:.1%}")
    
    print(f"\nDistance & Energy:")
    print(f"  Total flight distance: {final_metrics['total_flight_distance']:.2f}")
    print(f"  Optimal distance: {final_metrics['optimal_flight_distance']:.2f}")
    if final_metrics['optimal_flight_distance'] > 0:
        efficiency = final_metrics['optimal_flight_distance'] / max(final_metrics['total_flight_distance'], 0.1)
        print(f"  Distance efficiency: {efficiency:.1%}")
    print(f"  Total energy consumed: {final_metrics['energy_consumed']:.2f}")
    
    print(f"\n✓ Integration example completed successfully!")


if __name__ == "__main__":
    main()
