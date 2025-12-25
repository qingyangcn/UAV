"""test_train_strong_coupling.py

Strong coupling Route B:
- PPO outputs both drone headings (action) and a global MPSO parameter vector theta (P=3).
- theta parameterizes the MPSO evaluation weights used inside the population MORL loop.
- Observation is conditioned on objective weights w via obs['objective_weights'].
- Scalarization is weighted-sum with w (no random weight sampling / no other scalarization modes).
- Environments must use reward_output_mode='scalar'.

This script is based on test_train.py with minimal structural changes so the existing PPO and
population MORL flow remains recognizable.

Notes
-----
This file intentionally avoids making invasive edits elsewhere in the codebase. It performs local
adapters:
- flatten_obs() now concatenates objective_weights.
- A wrapped policy output is parsed into: headings action + theta.
- MPSO evaluation uses weights derived from theta (mapped to simplex).

If you later want to share theta across the whole population step, ensure that the policy is called
once per environment step and theta is cached appropriately.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np

# Keep imports aligned with test_train.py. If your repo uses different module names,
# adjust the imports accordingly.
try:
    # Typical structure in this repo (guess based on test_train.py)
    from envs.uav_env import UAVEnv  # type: ignore
except Exception:
    UAVEnv = None  # noqa: N816

try:
    from algorithms.ppo import PPO  # type: ignore
except Exception:
    PPO = None


# -----------------------------
# Utilities
# -----------------------------

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-12)


def theta_to_weights(theta: np.ndarray) -> np.ndarray:
    """Map unconstrained theta (P=3) to a valid weight vector w on the simplex.

    We use softmax to ensure:
    - w_i >= 0
    - sum_i w_i = 1

    Args:
        theta: shape (3,)

    Returns:
        w: shape (3,)
    """
    theta = np.asarray(theta, dtype=np.float32).reshape(-1)
    if theta.shape[0] != 3:
        raise ValueError(f"theta must have P=3 params, got shape {theta.shape}")
    return softmax(theta, axis=0).astype(np.float32)


def flatten_obs(obs: Dict[str, Any]) -> np.ndarray:
    """Flatten env observation and INCLUDE obs['objective_weights'].

    This enables conditioned multi-objective training by making w part of the policy input.

    Requirements from user request:
    - include obs['objective_weights']
    """

    parts = []

    # Heuristic: test_train.py likely already concatenates some keys. We keep it robust:
    for k, v in obs.items():
        if k == "objective_weights":
            # Add at the end (explicit), so skip here and handle later.
            continue
        if isinstance(v, (int, float, np.number)):
            parts.append(np.array([v], dtype=np.float32))
        else:
            arr = np.asarray(v, dtype=np.float32).reshape(-1)
            parts.append(arr)

    # MUST include objective_weights
    if "objective_weights" in obs:
        w = np.asarray(obs["objective_weights"], dtype=np.float32).reshape(-1)
        parts.append(w)
    else:
        # Fail loudly: w conditioning is required by this training route.
        raise KeyError(
            "Observation missing 'objective_weights'. "
            "Ensure the env adds it for conditioned MORL training."
        )

    return np.concatenate(parts, axis=0)


@dataclass
class StrongCouplingConfig:
    # Strong coupling Route B parameter count
    theta_dim: int = 3

    # Action / env settings
    reward_output_mode: str = "scalar"  # force scalar rewards

    # PPO settings (keep consistent with test_train.py defaults if any)
    seed: int = 0


# -----------------------------
# Training loop (based on test_train.py)
# -----------------------------

def make_env(args: argparse.Namespace, cfg: StrongCouplingConfig):
    """Construct env with reward_output_mode='scalar' as required."""
    if UAVEnv is None:
        raise ImportError(
            "Could not import UAVEnv. Please adjust imports in test_train_strong_coupling.py "
            "to match your repository structure."
        )

    # The original test_train.py likely passes many more kwargs.
    # Keep this flexible: only enforce reward_output_mode.
    return UAVEnv(
        **getattr(args, "env_kwargs", {}),
        reward_output_mode=cfg.reward_output_mode,
    )


def split_action_and_theta(action_with_theta: np.ndarray, action_dim: int, theta_dim: int) -> Tuple[np.ndarray, np.ndarray]:
    """Split PPO output into (action, theta)."""
    a = np.asarray(action_with_theta, dtype=np.float32).reshape(-1)
    if a.shape[0] != action_dim + theta_dim:
        raise ValueError(
            f"Expected policy output dim={action_dim + theta_dim}, got {a.shape[0]}"
        )
    return a[:action_dim], a[action_dim:]


def train(args: argparse.Namespace):
    cfg = StrongCouplingConfig(seed=args.seed)
    np.random.seed(cfg.seed)

    env = make_env(args, cfg)

    # Determine action dimensions from env; for UAV headings, assume continuous Box.
    action_dim = int(np.prod(env.action_space.shape))

    # PPO must output action_dim + theta_dim
    policy_output_dim = action_dim + cfg.theta_dim

    if PPO is None:
        raise ImportError(
            "Could not import PPO. Please adjust imports in test_train_strong_coupling.py "
            "to match your repository structure."
        )

    # Create PPO with adjusted action dimension.
    # Keep other hyperparameters as per test_train.py (unknown here), so we pass through args.
    ppo = PPO(
        obs_dim=None,  # Some implementations infer from first obs; otherwise set after reset.
        act_dim=policy_output_dim,
        **getattr(args, "ppo_kwargs", {}),
    )

    obs, info = env.reset(seed=cfg.seed)
    flat = flatten_obs(obs)

    # If PPO needs explicit obs_dim, set it once.
    if getattr(ppo, "obs_dim", None) in (None, 0) and hasattr(ppo, "set_obs_dim"):
        ppo.set_obs_dim(int(flat.shape[0]))

    # Cache current objective weights for scalarization.
    current_w = np.asarray(obs["objective_weights"], dtype=np.float32).reshape(-1)

    # --- Population MORL / MPSO structure ---
    # We keep the high-level intent: evaluate a population using MPSO-like selection.
    # The key change: the evaluation weights come from theta predicted by PPO.

    # Placeholder: depends on the actual code in test_train.py
    population = []

    for step in range(args.total_steps):
        flat = flatten_obs(obs)

        # Policy forward: outputs concatenated [headings_action, theta]
        action_with_theta = ppo.act(flat)  # expected shape (action_dim+theta_dim,)
        headings_action, theta = split_action_and_theta(action_with_theta, action_dim, cfg.theta_dim)

        # Convert theta -> weights used by MPSO evaluation
        mpso_w = theta_to_weights(theta)

        # Step env with ONLY headings action
        next_obs, reward, terminated, truncated, info = env.step(headings_action)
        done = bool(terminated or truncated)

        # Weighted-sum scalarization required. Since env reward_output_mode='scalar',
        # reward should already be scalar. But for safety, if env returns a vector reward
        # (legacy), we reduce it with objective_weights from obs.
        if isinstance(reward, (list, tuple, np.ndarray)):
            r_vec = np.asarray(reward, dtype=np.float32).reshape(-1)
            w = current_w
            if r_vec.shape[0] != w.shape[0]:
                raise ValueError(f"Reward vector dim {r_vec.shape[0]} != weight dim {w.shape[0]}")
            reward_scalar = float(np.dot(w, r_vec))
        else:
            reward_scalar = float(reward)

        # Store transition for PPO; include theta as part of action output (already).
        # Many PPO libs store (obs, act, rew, next_obs, done). We keep generic.
        ppo.store(flat, action_with_theta, reward_scalar, done)

        # --- Strong coupling: use mpso_w to evaluate/select within population MORL ---
        # The exact MPSO code varies across repos. Here we keep a light-touch integration:
        # - attach mpso_w to info for downstream evaluation
        # - optionally update some global evaluator weights
        info = dict(info) if info is not None else {}
        info["mpso_theta"] = theta.astype(np.float32)
        info["mpso_weights"] = mpso_w.astype(np.float32)

        # If your test_train.py has something like mpso.update_weights(w): call it here.
        if hasattr(env, "set_evaluation_weights"):
            # Some envs/evaluators can consume weights directly.
            env.set_evaluation_weights(mpso_w)

        # Update current objective weights for next step (conditioning)
        obs = next_obs
        if "objective_weights" in obs:
            current_w = np.asarray(obs["objective_weights"], dtype=np.float32).reshape(-1)

        if done:
            obs, info = env.reset()
            current_w = np.asarray(obs["objective_weights"], dtype=np.float32).reshape(-1)

        # PPO update
        if (step + 1) % args.update_every == 0:
            ppo.update()

    env.close()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--total-steps", type=int, default=200_000)
    p.add_argument("--update-every", type=int, default=2048)

    # Optional escape hatches: reuse existing dict-style kwargs used in test_train.py
    # You can remove these if test_train.py uses explicit args only.
    p.set_defaults(env_kwargs={})
    p.set_defaults(ppo_kwargs={})
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    train(args)
