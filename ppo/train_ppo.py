"""
PPO Training Script with MOPSO Dispatcher Integration.

This script trains a PPO agent for UAV delivery routing while using MOPSO
for order assignment at each step.

Usage:
    python ppo/train_ppo.py --total-steps 100000 --seed 42
"""
import argparse
import os
import sys
from typing import Optional

import numpy as np
import gymnasium as gym

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from UAV_ENVIRONMENT_6 import ThreeObjectiveDroneDeliveryEnv
from algorithms.mopso_dispatcher import MOPSOPlanner, apply_mopso_dispatch
from env_wrappers.flatten_action import FlattenActionWrapper


class MOPSOWrapper(gym.Wrapper):
    """
    Wrapper that calls MOPSO dispatcher before each step.
    
    This ensures idle drones get assigned orders via MOPSO,
    while PPO controls the routing (heading + speed).
    """
    
    def __init__(self, env, mopso_planner: Optional[MOPSOPlanner] = None):
        super().__init__(env)
        self.mopso_planner = mopso_planner or MOPSOPlanner(
            n_particles=30,
            n_iterations=10,
            max_orders=200,
            max_orders_per_drone=10
        )
    
    def step(self, action):
        # Apply MOPSO dispatch before stepping
        apply_mopso_dispatch(self.env, self.mopso_planner)
        
        # Step with PPO action
        return self.env.step(action)
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        # Apply MOPSO after reset
        apply_mopso_dispatch(self.env, self.mopso_planner)
        
        return obs, info


def make_env(seed: int = 0, 
             reward_output_mode: str = "zero",
             enable_random_events: bool = True):
    """
    Create wrapped environment for training.
    
    Args:
        seed: Random seed
        reward_output_mode: Reward mode ("zero" for multi-objective)
        enable_random_events: Whether to enable random events
    
    Returns:
        Wrapped environment
    """
    # Create base environment
    env = ThreeObjectiveDroneDeliveryEnv(
        grid_size=16,
        num_drones=6,
        max_orders=100,
        steps_per_hour=4,
        drone_max_capacity=10,
        reward_output_mode=reward_output_mode,
        enable_random_events=enable_random_events,
        fixed_objective_weights=(0.5, 0.3, 0.2),
    )
    
    # Wrap with MOPSO dispatcher
    env = MOPSOWrapper(env)
    
    # Flatten action space for SB3
    env = FlattenActionWrapper(env)
    
    return env


def train_ppo(total_timesteps: int = 100000,
              seed: int = 42,
              learning_rate: float = 3e-4,
              n_steps: int = 2048,
              batch_size: int = 64,
              n_epochs: int = 10,
              gamma: float = 0.99,
              gae_lambda: float = 0.95,
              clip_range: float = 0.2,
              save_freq: int = 10000,
              log_dir: str = "./logs",
              model_dir: str = "./models"):
    """
    Train PPO agent.
    
    Args:
        total_timesteps: Total training timesteps
        seed: Random seed
        learning_rate: Learning rate
        n_steps: Steps per rollout
        batch_size: Minibatch size
        n_epochs: Number of epochs per update
        gamma: Discount factor
        gae_lambda: GAE lambda
        clip_range: PPO clip range
        save_freq: Model save frequency
        log_dir: Tensorboard log directory
        model_dir: Model save directory
    """
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        from stable_baselines3.common.callbacks import CheckpointCallback
    except ImportError:
        print("Error: stable-baselines3 not installed.")
        print("Install with: pip install stable-baselines3")
        sys.exit(1)
    
    # Create directories
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    print("=" * 60)
    print("PPO Training with MOPSO Dispatcher")
    print("=" * 60)
    print(f"Total timesteps: {total_timesteps}")
    print(f"Seed: {seed}")
    print(f"Learning rate: {learning_rate}")
    print(f"Reward mode: zero (multi-objective)")
    print(f"MOPSO: M=200, K=10, P=30, I=10")
    print("=" * 60)
    
    # Create environment
    def env_fn():
        return make_env(seed=seed, reward_output_mode="zero")
    
    env = DummyVecEnv([env_fn])
    
    # Create PPO model
    model = PPO(
        "MultiInputPolicy",  # For Dict observation space
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        verbose=1,
        tensorboard_log=log_dir,
        seed=seed,
    )
    
    # Create checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=model_dir,
        name_prefix="ppo_uav",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    
    print("\nStarting training...")
    print(f"Logs will be saved to: {log_dir}")
    print(f"Models will be saved to: {model_dir}")
    print()
    
    # Train
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        progress_bar=True,
    )
    
    # Save final model
    final_path = os.path.join(model_dir, "ppo_uav_final")
    model.save(final_path)
    print(f"\nTraining complete! Final model saved to: {final_path}")
    
    # Close environment
    env.close()
    
    return model


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train PPO with MOPSO dispatcher")
    parser.add_argument("--total-steps", type=int, default=100000,
                       help="Total training timesteps")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--lr", type=float, default=3e-4,
                       help="Learning rate")
    parser.add_argument("--n-steps", type=int, default=2048,
                       help="Steps per rollout")
    parser.add_argument("--batch-size", type=int, default=64,
                       help="Minibatch size")
    parser.add_argument("--n-epochs", type=int, default=10,
                       help="Number of epochs")
    parser.add_argument("--gamma", type=float, default=0.99,
                       help="Discount factor")
    parser.add_argument("--save-freq", type=int, default=10000,
                       help="Model save frequency")
    parser.add_argument("--log-dir", type=str, default="./logs",
                       help="Tensorboard log directory")
    parser.add_argument("--model-dir", type=str, default="./models",
                       help="Model save directory")
    
    args = parser.parse_args()
    
    # Train
    train_ppo(
        total_timesteps=args.total_steps,
        seed=args.seed,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        save_freq=args.save_freq,
        log_dir=args.log_dir,
        model_dir=args.model_dir,
    )


if __name__ == "__main__":
    main()
