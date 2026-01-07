"""
U7 PPO Training Script with MOPSO Assignment Integration.

This script trains a PPO agent for UAV delivery task selection while using MOPSO
for order assignment at each step.

Key differences from U6:
- MOPSO does assignment only (no route planning)
- PPO does task selection (which order to serve next) + speed control
- No heading control - drones fly straight to chosen target
- Action space: (choice, speed) per drone
- Reward output mode: "scalar" (dot product with objective weights)
"""
import argparse
import os
import sys
from typing import Optional

import numpy as np
import gymnasium as gym

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from UAV_ENVIRONMENT_7 import ThreeObjectiveDroneDeliveryEnv
from U7_mopso_dispatcher import U7MOPSOAssigner, apply_mopso_assignment


class U7MOPSOWrapper(gym.Wrapper):
    """
    Wrapper that calls MOPSO assignment dispatcher before each step.

    This ensures READY orders get assigned to drones via MOPSO,
    while PPO controls task selection (which order to serve next) and speed.
    """

    def __init__(self, env, mopso_assigner: Optional[U7MOPSOAssigner] = None,
                 max_orders_per_drone: int = 10,
                 eta_speed_scale_assumption: float = 0.7,
                 eta_stop_service_steps: int = 1):
        super().__init__(env)
        self.mopso_assigner = mopso_assigner or U7MOPSOAssigner(
            n_particles=30,
            n_iterations=10,
            max_orders=200,
            max_orders_per_drone=max_orders_per_drone,
            eta_speed_scale_assumption=eta_speed_scale_assumption,
            eta_stop_service_steps=eta_stop_service_steps
        )

    def step(self, action):
        # Apply MOPSO assignment before stepping
        apply_mopso_assignment(self.env, self.mopso_assigner)

        # Step with PPO action (task choice + speed)
        return self.env.step(action)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # Apply MOPSO after reset
        apply_mopso_assignment(self.env, self.mopso_assigner)

        return obs, info


class U7FlattenActionWrapper(gym.Wrapper):
    """
    Wrapper to flatten/unflatten action space for SB3 compatibility.
    
    Environment expects: (N, 2) array where N is number of drones
    Each drone has: [choice, speed]
    SB3 provides: (N*2,) flat array
    
    This wrapper handles the conversion.
    """
    
    def __init__(self, env):
        """
        Initialize wrapper.
        
        Args:
            env: Gymnasium environment with Box(N, 2) action space
        """
        super().__init__(env)
        
        # Get original action space
        orig_space = env.action_space
        
        if not isinstance(orig_space, gym.spaces.Box):
            raise ValueError(f"Expected Box action space, got {type(orig_space)}")
        
        if len(orig_space.shape) != 2:
            raise ValueError(f"Expected 2D action space (N, 2), got shape {orig_space.shape}")
        
        self.n_drones = orig_space.shape[0]
        self.action_dim = orig_space.shape[1]
        
        if self.action_dim != 2:
            raise ValueError(f"Expected action dimension 2 (choice, speed), got {self.action_dim}")
        
        # Create flattened action space
        flat_size = self.n_drones * self.action_dim
        
        # Properly flatten bounds to maintain per-dimension constraints
        low_flat = orig_space.low.flatten()
        high_flat = orig_space.high.flatten()
        
        self.action_space = gym.spaces.Box(
            low=low_flat,
            high=high_flat,
            shape=(flat_size,),
            dtype=orig_space.dtype
        )
    
    def step(self, action):
        """
        Step with flattened action.
        
        Args:
            action: Flat array of shape (N*2,)
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Reshape flat action to (N, 2)
        action_reshaped = action.reshape(self.n_drones, self.action_dim)
        
        # Call environment step
        return self.env.step(action_reshaped)
    
    def reset(self, **kwargs):
        """Reset environment."""
        return self.env.reset(**kwargs)


def make_env(seed: int = 0,
             reward_output_mode: str = "scalar",
             enable_random_events: bool = False,
             max_orders_per_drone: int = 10,
             eta_speed_scale_assumption: float = 0.7,
             eta_stop_service_steps: int = 1,
             num_candidates: int = 20):
    """
    Create wrapped environment for training.

    Args:
        seed: Random seed
        reward_output_mode: Reward mode ("scalar" for weighted sum)
        enable_random_events: Whether to enable random events
        max_orders_per_drone: K parameter for MOPSO (max orders per drone)
        eta_speed_scale_assumption: Conservative speed assumption for MOPSO ETA
        eta_stop_service_steps: Service time per stop for MOPSO ETA
        num_candidates: Number of candidate orders per drone for PPO

    Returns:
        Wrapped environment
    """
    # Create base environment
    env = ThreeObjectiveDroneDeliveryEnv(
        grid_size=16,
        num_drones=6,
        max_orders=200,
        num_bases=2,
        steps_per_hour=4,
        drone_max_capacity=10,
        reward_output_mode=reward_output_mode,
        enable_random_events=enable_random_events,
        fixed_objective_weights=(0.5, 0.3, 0.2),
        num_candidates=num_candidates,
    )

    # Wrap with MOPSO assignment
    env = U7MOPSOWrapper(
        env,
        max_orders_per_drone=max_orders_per_drone,
        eta_speed_scale_assumption=eta_speed_scale_assumption,
        eta_stop_service_steps=eta_stop_service_steps
    )

    # Flatten action space for SB3
    env = U7FlattenActionWrapper(env)

    return env


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="U7 PPO Training with MOPSO Assignment")
    parser.add_argument("--total-steps", type=int, default=100000,
                        help="Total training steps")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--eval-episodes", type=int, default=5,
                        help="Number of evaluation episodes")
    parser.add_argument("--save-path", type=str, default="./u7_ppo_model",
                        help="Path to save model")
    parser.add_argument("--no-random-events", action="store_true",
                        help="Disable random events")
    parser.add_argument("--num-candidates", type=int, default=20,
                        help="Number of candidate orders per drone")
    
    args = parser.parse_args()

    print("=" * 60)
    print("U7 PPO Training with MOPSO Assignment")
    print("=" * 60)
    print(f"Total steps: {args.total_steps}")
    print(f"Seed: {args.seed}")
    print(f"Num candidates per drone: {args.num_candidates}")
    print(f"Random events: {not args.no_random_events}")
    print(f"Reward output mode: scalar")
    print("=" * 60)

    # Create environment
    env = make_env(
        seed=args.seed,
        reward_output_mode="scalar",
        enable_random_events=not args.no_random_events,
        num_candidates=args.num_candidates
    )

    print("\nEnvironment created successfully!")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Test environment
    print("\nTesting environment reset...")
    obs, info = env.reset(seed=args.seed)
    print(f"Reset successful. Observation keys: {obs.keys()}")
    print(f"Candidates shape: {obs['candidates'].shape}")

    # Test step
    print("\nTesting environment step...")
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step successful. Reward: {reward:.4f}")
    print(f"Info r_vec: {info.get('r_vec', 'N/A')}")

    print("\n" + "=" * 60)
    print("Environment tests passed!")
    print("=" * 60)
    print("\nTo train with Stable-Baselines3, use:")
    print("  from stable_baselines3 import PPO")
    print("  model = PPO('MultiInputPolicy', env, verbose=1)")
    print("  model.learn(total_timesteps=args.total_steps)")
    print("  model.save(args.save_path)")


if __name__ == "__main__":
    main()
