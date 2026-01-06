"""
PPO Evaluation/Inference Script.

Evaluates a trained PPO model with MOPSO dispatcher and reports daily statistics.

Usage:
    python ppo/eval_ppo.py --model models/ppo_uav_final.zip --episodes 10
"""
import argparse
import os
import sys
from typing import Optional

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from UAV_ENVIRONMENT_6 import ThreeObjectiveDroneDeliveryEnv
from algorithms.mopso_dispatcher import MOPSOPlanner, apply_mopso_dispatch
from env_wrappers.flatten_action import FlattenActionWrapper
from ppo.train_ppo import MOPSOWrapper


def make_eval_env(seed: int = 0, enable_random_events: bool = False):
    """
    Create environment for evaluation.
    
    Args:
        seed: Random seed
        enable_random_events: Whether to enable random events (usually False for eval)
    
    Returns:
        Wrapped environment
    """
    env = ThreeObjectiveDroneDeliveryEnv(
        grid_size=16,
        num_drones=6,
        max_orders=100,
        steps_per_hour=4,
        drone_max_capacity=10,
        reward_output_mode="zero",
        enable_random_events=enable_random_events,
        fixed_objective_weights=(0.5, 0.3, 0.2),
    )
    
    # Wrap with MOPSO dispatcher
    env = MOPSOWrapper(env)
    
    # Flatten action space
    env = FlattenActionWrapper(env)
    
    return env


def evaluate_model(model_path: str,
                  n_episodes: int = 10,
                  seed: int = 42,
                  enable_random_events: bool = False,
                  verbose: bool = True):
    """
    Evaluate trained PPO model.
    
    Args:
        model_path: Path to saved model
        n_episodes: Number of episodes to evaluate
        seed: Random seed
        enable_random_events: Whether to enable random events
        verbose: Whether to print episode details
    
    Returns:
        Dictionary of evaluation metrics
    """
    try:
        from stable_baselines3 import PPO
    except ImportError:
        print("Error: stable-baselines3 not installed.")
        print("Install with: pip install stable-baselines3")
        sys.exit(1)
    
    # Load model
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        sys.exit(1)
    
    print("=" * 60)
    print("PPO Evaluation with MOPSO Dispatcher")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Episodes: {n_episodes}")
    print(f"Seed: {seed}")
    print(f"Random events: {enable_random_events}")
    print("=" * 60)
    print()
    
    model = PPO.load(model_path)
    
    # Create environment
    env = make_eval_env(seed=seed, enable_random_events=enable_random_events)
    
    # Storage for metrics
    episode_rewards = []
    episode_lengths = []
    episode_r_vecs = []
    daily_stats_list = []
    
    # Run episodes
    for episode in range(n_episodes):
        obs, info = env.reset(seed=seed + episode)
        
        episode_reward = 0.0
        episode_length = 0
        done = False
        
        while not done:
            # Get action from model
            action, _ = model.predict(obs, deterministic=True)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
        
        # Get episode stats
        r_vec = info.get('episode_r_vec', np.zeros(3))
        daily_stats = info.get('daily_stats', {})
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_r_vecs.append(r_vec)
        daily_stats_list.append(daily_stats)
        
        if verbose:
            print(f"Episode {episode + 1}/{n_episodes}:")
            print(f"  Reward: {episode_reward:.2f}")
            print(f"  Length: {episode_length}")
            print(f"  R_vec: [{r_vec[0]:.2f}, {r_vec[1]:.2f}, {r_vec[2]:.2f}]")
            print(f"  Orders completed: {daily_stats.get('orders_completed', 0)}")
            print(f"  Orders cancelled: {daily_stats.get('orders_cancelled', 0)}")
            print(f"  On-time rate: {daily_stats.get('on_time_deliveries', 0) / max(1, daily_stats.get('orders_completed', 1)):.2%}")
            print()
    
    # Compute statistics
    episode_rewards = np.array(episode_rewards)
    episode_lengths = np.array(episode_lengths)
    episode_r_vecs = np.array(episode_r_vecs)
    
    # Aggregate daily stats
    total_completed = sum(s.get('orders_completed', 0) for s in daily_stats_list)
    total_cancelled = sum(s.get('orders_cancelled', 0) for s in daily_stats_list)
    total_generated = sum(s.get('orders_generated', 0) for s in daily_stats_list)
    total_on_time = sum(s.get('on_time_deliveries', 0) for s in daily_stats_list)
    total_distance = sum(s.get('total_flight_distance', 0) for s in daily_stats_list)
    total_energy = sum(s.get('energy_consumed', 0) for s in daily_stats_list)
    
    print("=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Episodes: {n_episodes}")
    print()
    print("Episode Metrics:")
    print(f"  Mean reward: {episode_rewards.mean():.2f} ± {episode_rewards.std():.2f}")
    print(f"  Mean length: {episode_lengths.mean():.1f} ± {episode_lengths.std():.1f}")
    print()
    print("Multi-Objective Returns (mean ± std):")
    print(f"  f0 (throughput): {episode_r_vecs[:, 0].mean():.2f} ± {episode_r_vecs[:, 0].std():.2f}")
    print(f"  f1 (cost):       {episode_r_vecs[:, 1].mean():.2f} ± {episode_r_vecs[:, 1].std():.2f}")
    print(f"  f2 (quality):    {episode_r_vecs[:, 2].mean():.2f} ± {episode_r_vecs[:, 2].std():.2f}")
    print()
    print("Aggregate Daily Statistics:")
    print(f"  Total orders generated: {total_generated}")
    print(f"  Total orders completed: {total_completed}")
    print(f"  Total orders cancelled: {total_cancelled}")
    print(f"  Completion rate: {total_completed / max(1, total_generated):.2%}")
    print(f"  On-time rate: {total_on_time / max(1, total_completed):.2%}")
    print(f"  Total flight distance: {total_distance:.1f}")
    print(f"  Total energy consumed: {total_energy:.1f}")
    if total_completed > 0:
        print(f"  Avg distance per order: {total_distance / total_completed:.2f}")
        print(f"  Avg energy per order: {total_energy / total_completed:.2f}")
    print("=" * 60)
    
    env.close()
    
    return {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'episode_r_vecs': episode_r_vecs,
        'daily_stats': daily_stats_list,
        'mean_reward': episode_rewards.mean(),
        'std_reward': episode_rewards.std(),
        'mean_r_vec': episode_r_vecs.mean(axis=0),
        'std_r_vec': episode_r_vecs.std(axis=0),
        'total_completed': total_completed,
        'total_cancelled': total_cancelled,
        'total_generated': total_generated,
        'completion_rate': total_completed / max(1, total_generated),
        'on_time_rate': total_on_time / max(1, total_completed),
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Evaluate PPO with MOPSO dispatcher")
    parser.add_argument("--model", type=str, required=True,
                       help="Path to model file (e.g., models/ppo_uav_final.zip)")
    parser.add_argument("--episodes", type=int, default=10,
                       help="Number of episodes to evaluate")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--enable-random-events", action="store_true",
                       help="Enable random events during evaluation")
    parser.add_argument("--quiet", action="store_true",
                       help="Suppress episode details")
    
    args = parser.parse_args()
    
    # Evaluate
    evaluate_model(
        model_path=args.model,
        n_episodes=args.episodes,
        seed=args.seed,
        enable_random_events=args.enable_random_events,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
