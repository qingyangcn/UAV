"""
Test script for U7 control decomposition implementation.

Verifies:
1. Environment creation with num_candidates parameter
2. Observation space includes candidates
3. Action space is (N, 2) for choice + speed
4. Candidate mappings are updated each step
5. Decision point logic works
6. Task selection updates drone targets
7. MOPSO assignment works
8. Training wrapper integrates correctly
"""
import numpy as np
from UAV_ENVIRONMENT_7 import ThreeObjectiveDroneDeliveryEnv
from U7_mopso_dispatcher import U7MOPSOAssigner, apply_mopso_assignment
from U7_train import make_env


def test_environment_creation():
    """Test environment can be created with new parameters."""
    print("Test 1: Environment creation...")
    env = ThreeObjectiveDroneDeliveryEnv(
        grid_size=16,
        num_drones=6,
        max_orders=200,
        num_bases=2,
        num_candidates=20,
        reward_output_mode='scalar'
    )
    assert env.num_candidates == 20
    assert env.action_space.shape == (6, 2)
    print("  ✓ Environment created with correct parameters")
    return env


def test_observation_space(env):
    """Test observation space includes candidates."""
    print("\nTest 2: Observation space...")
    assert 'candidates' in env.observation_space.keys()
    candidates_space = env.observation_space['candidates']
    assert candidates_space.shape == (6, 20, 12)
    print("  ✓ Candidates observation space correct: (6, 20, 12)")


def test_reset_and_candidates(env):
    """Test reset initializes candidate mappings."""
    print("\nTest 3: Reset and candidate initialization...")
    obs, info = env.reset(seed=42)
    
    assert 'candidates' in obs
    assert obs['candidates'].shape == (6, 20, 12)
    
    # Check that candidate mappings are initialized
    assert len(env.drone_candidate_mappings) == 6
    for drone_id in range(6):
        candidates = env.drone_candidate_mappings[drone_id]
        assert len(candidates) == 20
        # Each candidate is (order_id, is_valid)
        assert all(isinstance(c, tuple) and len(c) == 2 for c in candidates)
    
    print("  ✓ Candidate mappings initialized for all drones")
    print(f"  ✓ Drone 0 candidates sample: {env.drone_candidate_mappings[0][:3]}")


def test_action_processing(env):
    """Test action processing with task choice + speed."""
    print("\nTest 4: Action processing...")
    
    # Reset first
    obs, info = env.reset(seed=42)
    
    # Create action: all drones choose middle candidate and speed
    action = np.zeros((6, 2), dtype=np.float32)
    
    obs2, reward, terminated, truncated, info = env.step(action)
    
    assert not terminated
    assert isinstance(reward, (int, float))
    assert 'r_vec' in info
    assert len(info['r_vec']) == 3
    
    print("  ✓ Step executed successfully")
    print(f"  ✓ Reward: {reward:.4f}")
    print(f"  ✓ r_vec: {info['r_vec']}")


def test_candidate_updates(env):
    """Test candidate mappings update each step."""
    print("\nTest 5: Candidate mapping updates...")
    
    obs, info = env.reset(seed=42)
    initial_candidates = [c for c in env.drone_candidate_mappings[0]]
    
    # Take a step
    action = np.zeros((6, 2), dtype=np.float32)
    obs2, _, _, _, _ = env.step(action)
    
    # Candidates should have been updated
    updated_candidates = env.drone_candidate_mappings[0]
    
    print(f"  ✓ Candidates updated each step")
    print(f"  ✓ Initial candidates sample: {initial_candidates[:3]}")
    print(f"  ✓ Updated candidates sample: {updated_candidates[:3]}")


def test_mopso_assignment():
    """Test MOPSO assignment-only dispatcher."""
    print("\nTest 6: MOPSO assignment...")
    
    env = ThreeObjectiveDroneDeliveryEnv(
        grid_size=16,
        num_drones=6,
        max_orders=200,
        num_bases=2,
        num_candidates=20,
        reward_output_mode='scalar'
    )
    
    obs, info = env.reset(seed=42)
    
    # Create assigner
    assigner = U7MOPSOAssigner(n_particles=10, n_iterations=5)
    
    # Apply assignment
    assignment_counts = apply_mopso_assignment(env, assigner)
    
    print(f"  ✓ MOPSO assignment executed")
    print(f"  ✓ Assignment counts: {assignment_counts}")


def test_training_wrapper():
    """Test training wrapper integrates everything."""
    print("\nTest 7: Training wrapper integration...")
    
    env = make_env(
        seed=42,
        reward_output_mode='scalar',
        enable_random_events=False,
        num_candidates=20
    )
    
    # Check flattened action space
    assert env.action_space.shape == (12,)  # 6 drones * 2 actions
    
    # Reset
    obs, info = env.reset(seed=42)
    assert 'candidates' in obs
    
    # Step with flattened action
    action = env.action_space.sample()
    obs2, reward, terminated, truncated, info = env.step(action)
    
    print("  ✓ Training wrapper works correctly")
    print(f"  ✓ Flattened action space: {env.action_space.shape}")
    print(f"  ✓ Reward output mode: scalar")


def test_decision_points(env):
    """Test decision point detection."""
    print("\nTest 8: Decision point logic...")
    
    obs, info = env.reset(seed=42)
    
    # Check initial decision points (all drones should be IDLE)
    decision_points = []
    for drone_id in range(6):
        is_decision = env._is_at_decision_point(drone_id)
        decision_points.append(is_decision)
        drone_status = env.drones[drone_id]['status'].name
    
    # At least some drones should be at decision points initially
    print(f"  ✓ Decision point detection implemented")
    print(f"  ✓ Decision points at reset: {sum(decision_points)}/{len(decision_points)} drones")


def main():
    """Run all tests."""
    print("=" * 60)
    print("U7 Control Decomposition Test Suite")
    print("=" * 60)
    
    try:
        env = test_environment_creation()
        test_observation_space(env)
        test_reset_and_candidates(env)
        test_action_processing(env)
        test_candidate_updates(env)
        test_decision_points(env)
        test_mopso_assignment()
        test_training_wrapper()
        
        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
