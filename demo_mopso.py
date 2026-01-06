"""
Quick demo showing MOPSO dispatcher in action without requiring stable-baselines3.

This script demonstrates:
1. Environment creation
2. MOPSO dispatcher assigning orders to drones
3. Route plan application
4. Basic simulation loop
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from UAV_ENVIRONMENT_6 import ThreeObjectiveDroneDeliveryEnv
from algorithms.mopso_dispatcher import MOPSOPlanner, apply_mopso_dispatch


def main():
    print("=" * 60)
    print("MOPSO Dispatcher Demo")
    print("=" * 60)
    print()
    
    # Create environment
    print("Creating environment...")
    env = ThreeObjectiveDroneDeliveryEnv(
        grid_size=16,
        num_drones=6,
        max_orders=100,
        reward_output_mode="zero",
        enable_random_events=True,
    )
    
    # Reset
    print("Resetting environment...")
    obs, info = env.reset(seed=42)
    print(f"✓ Environment ready")
    print()
    
    # Create MOPSO planner
    print("Creating MOPSO planner...")
    planner = MOPSOPlanner(
        n_particles=30,
        n_iterations=10,
        max_orders=200,
        max_orders_per_drone=10,
        seed=42
    )
    print(f"✓ MOPSO planner created (P=30, I=10, M=200, K=10)")
    print()
    
    # Run simulation
    print("Running simulation for 20 steps...")
    print("-" * 60)
    
    total_assigned = 0
    total_completed = 0
    
    for step in range(20):
        # Apply MOPSO dispatch
        plans = apply_mopso_dispatch(env, planner)
        
        # Count assignments
        step_assigned = sum(len(orders) for _, orders in plans.values())
        total_assigned += step_assigned
        
        # Random action (in real use, this would come from PPO)
        action = np.random.uniform(-1, 1, size=(6, 3)).astype(np.float32)
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Get statistics
        completed = info['daily_stats']['orders_completed']
        cancelled = info['daily_stats']['orders_cancelled']
        active = len(env.active_orders)
        
        if step_assigned > 0 or step % 5 == 0:
            print(f"Step {step:2d}: "
                  f"Assigned={step_assigned:2d}, "
                  f"Active={active:3d}, "
                  f"Completed={completed:3d}, "
                  f"Cancelled={cancelled:2d}")
        
        total_completed = completed
        
        if terminated or truncated:
            print(f"Episode ended at step {step}")
            break
    
    print("-" * 60)
    print()
    
    # Final statistics
    print("Final Statistics:")
    print(f"  Total orders assigned by MOPSO: {total_assigned}")
    print(f"  Total orders completed: {total_completed}")
    print(f"  Active orders: {len(env.active_orders)}")
    print(f"  Cancelled orders: {info['daily_stats']['orders_cancelled']}")
    
    if total_completed > 0:
        on_time = info['daily_stats']['on_time_deliveries']
        print(f"  On-time rate: {on_time / total_completed:.1%}")
        print(f"  Total distance: {info['daily_stats']['total_flight_distance']:.1f}")
        print(f"  Energy consumed: {info['daily_stats']['energy_consumed']:.1f}")
    
    print()
    print("=" * 60)
    print("✓ Demo completed successfully!")
    print("=" * 60)
    print()
    print("This demonstrates:")
    print("  - MOPSO dispatcher assigns orders to idle drones")
    print("  - Route plans are applied via apply_route_plan()")
    print("  - Drones execute assigned routes")
    print("  - Multi-objective rewards are computed")
    print()
    print("To train with PPO:")
    print("  1. Install: pip install stable-baselines3 torch")
    print("  2. Run: python ppo/train_ppo.py --total-steps 100000")
    print()
    
    env.close()


if __name__ == "__main__":
    main()
