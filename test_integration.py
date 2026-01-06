"""
Simple test to verify MOPSO and PPO integration works correctly.
"""
import sys
import os
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from UAV_ENVIRONMENT_6 import ThreeObjectiveDroneDeliveryEnv
from algorithms.mopso_dispatcher import MOPSOPlanner, apply_mopso_dispatch
from env_wrappers.flatten_action import FlattenActionWrapper
from ppo.train_ppo import MOPSOWrapper


def test_environment_creation():
    """Test basic environment creation."""
    print("=" * 60)
    print("Test 1: Environment Creation")
    print("=" * 60)
    
    env = ThreeObjectiveDroneDeliveryEnv(
        grid_size=16,
        num_drones=6,
        max_orders=100,
        reward_output_mode="zero",
    )
    
    print(f"✓ Environment created")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    print(f"  Action space shape: {env.action_space.shape}")
    
    assert env.action_space.shape == (6, 3), f"Expected action shape (6, 3), got {env.action_space.shape}"
    print(f"✓ Action space is (6, 3) as expected")
    
    obs, info = env.reset(seed=42)
    print(f"✓ Environment reset successful")
    print(f"  Observation keys: {list(obs.keys())}")
    
    env.close()
    print()


def test_snapshot_interfaces():
    """Test snapshot interface methods."""
    print("=" * 60)
    print("Test 2: Snapshot Interfaces")
    print("=" * 60)
    
    env = ThreeObjectiveDroneDeliveryEnv(
        grid_size=16,
        num_drones=6,
        max_orders=100,
        reward_output_mode="zero",
    )
    env.reset(seed=42)
    
    # Test snapshot methods
    ready_orders = env.get_ready_orders_snapshot(limit=200)
    print(f"✓ get_ready_orders_snapshot: {len(ready_orders)} orders")
    
    drones = env.get_drones_snapshot()
    print(f"✓ get_drones_snapshot: {len(drones)} drones")
    
    merchants = env.get_merchants_snapshot()
    print(f"✓ get_merchants_snapshot: {len(merchants)} merchants")
    
    constraints = env.get_route_plan_constraints()
    print(f"✓ get_route_plan_constraints: {list(constraints.keys())}")
    
    env.close()
    print()


def test_mopso_dispatcher():
    """Test MOPSO dispatcher."""
    print("=" * 60)
    print("Test 3: MOPSO Dispatcher")
    print("=" * 60)
    
    env = ThreeObjectiveDroneDeliveryEnv(
        grid_size=16,
        num_drones=6,
        max_orders=100,
        reward_output_mode="zero",
    )
    env.reset(seed=42)
    
    # Create MOPSO planner
    planner = MOPSOPlanner(
        n_particles=30,
        n_iterations=10,
        max_orders=200,
        max_orders_per_drone=10,
        seed=42
    )
    print(f"✓ MOPSO planner created (P=30, I=10, M=200, K=10)")
    
    # Run dispatch
    plans = planner.mopso_dispatch(env)
    print(f"✓ MOPSO dispatch completed")
    print(f"  Generated plans for {len(plans)} drones")
    
    for drone_id, (stops, orders) in plans.items():
        print(f"  Drone {drone_id}: {len(stops)} stops, {len(orders)} orders")
    
    env.close()
    print()


def test_action_wrapper():
    """Test action wrapper."""
    print("=" * 60)
    print("Test 4: Action Wrapper")
    print("=" * 60)
    
    env = ThreeObjectiveDroneDeliveryEnv(
        grid_size=16,
        num_drones=6,
        max_orders=100,
        reward_output_mode="zero",
    )
    
    # Wrap with flatten action
    env = FlattenActionWrapper(env)
    print(f"✓ FlattenActionWrapper applied")
    print(f"  New action space shape: {env.action_space.shape}")
    
    assert env.action_space.shape == (18,), f"Expected flat shape (18,), got {env.action_space.shape}"
    print(f"✓ Flattened action space is (18,) = 6 drones × 3 dims")
    
    obs, info = env.reset(seed=42)
    
    # Test random action
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"✓ Step with flattened action successful")
    
    env.close()
    print()


def test_mopso_wrapper():
    """Test MOPSO wrapper integration."""
    print("=" * 60)
    print("Test 5: MOPSO Wrapper Integration")
    print("=" * 60)
    
    env = ThreeObjectiveDroneDeliveryEnv(
        grid_size=16,
        num_drones=6,
        max_orders=100,
        reward_output_mode="zero",
    )
    
    # Wrap with MOPSO
    env = MOPSOWrapper(env)
    print(f"✓ MOPSOWrapper applied")
    
    # Wrap with action flattener
    env = FlattenActionWrapper(env)
    print(f"✓ FlattenActionWrapper applied")
    
    obs, info = env.reset(seed=42)
    print(f"✓ Environment reset with wrappers")
    
    # Run a few steps
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    
    print(f"✓ Completed {i+1} steps with MOPSO+PPO integration")
    
    env.close()
    print()


def test_action_processing():
    """Test that (N,3) actions are correctly processed."""
    print("=" * 60)
    print("Test 6: Action Processing (hx, hy, u)")
    print("=" * 60)
    
    env = ThreeObjectiveDroneDeliveryEnv(
        grid_size=16,
        num_drones=6,
        max_orders=100,
        reward_output_mode="zero",
    )
    
    obs, info = env.reset(seed=42)
    
    # Create action with specific values
    action = np.zeros((6, 3), dtype=np.float32)
    action[:, 0] = 0.5  # hx
    action[:, 1] = 0.5  # hy
    action[:, 2] = 0.0  # u (speed multiplier in [-1, 1], mapped to [0.5, 1.5])
    # Note: u=0.0 maps to speed multiplier 1.0 via: (0+1)/2 * (1.5-0.5) + 0.5 = 0.5 * 1.0 + 0.5 = 1.0
    
    print(f"✓ Created action with shape {action.shape}")
    print(f"  Sample action[0]: hx={action[0,0]}, hy={action[0,1]}, u={action[0,2]} (->speed 1.0x)")
    
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"✓ Step with (N,3) action successful")
    
    env.close()
    print()


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("MOPSO + PPO Integration Tests")
    print("=" * 60)
    print()
    
    try:
        test_environment_creation()
        test_snapshot_interfaces()
        test_mopso_dispatcher()
        test_action_wrapper()
        test_mopso_wrapper()
        test_action_processing()
        
        print("=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        print()
        print("Next steps:")
        print("1. To train: python ppo/train_ppo.py --total-steps 100000")
        print("2. To evaluate: python ppo/eval_ppo.py --model models/ppo_uav_final.zip")
        print()
        
    except Exception as e:
        print()
        print("=" * 60)
        print("✗ Test failed!")
        print("=" * 60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
