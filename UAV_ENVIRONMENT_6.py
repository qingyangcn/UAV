# -*- coding: utf-8 -*-
"""UAV_ENVIRONMENT_6.py

Updated UAV environment supporting multi-merchant *batch* route plans.

Key changes vs UAV_ENVIRONMENT_5:
- "batch_orders" refactored into "route_plan" with explicit route_idx.
- READY-only dispatching (only dispatch UAVs in READY state).
- Disallow mid-flight target re-routing; once a plan is dispatched, the UAV
  follows it until completion (or failure) and cannot be assigned a new target.
- Adds deadline_step on plan nodes / orders.
- Adds dispatch_plans(plans) API.
- Adds simulate_plan(...) pure evaluator (no mutation) to score/validate plans.
- Adds RL observation fields: to_target, dist_to_target, traffic_patch.

This file is designed to be a drop-in replacement for UAV_ENVIRONMENT_5.py
and follows the same general structure, while adding compatibility helpers.

NOTE:
This implementation intentionally keeps dependencies minimal and does not
require the rest of the repo beyond typical constants and map definitions.
If UAV_ENVIRONMENT_5.py defines additional helpers/constants, integrate those
as needed.

Author: qingyangcn
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import copy
import math
import numpy as np


# ---------------------------
# Data structures
# ---------------------------


@dataclass
class RouteNode:
    """A single node in a UAV route plan.

    node_type: 'PICKUP' or 'DROP'
    merchant_id: int identifies merchant.
    order_id: int identifies order.
    pos: (x, y) grid coordinate.
    deadline_step: optional deadline (step index) by which service should occur.

    READY-only dispatching means plans can be assigned only to UAVs in READY;
    once assigned, node sequence is immutable.
    """

    node_type: str
    merchant_id: int
    order_id: int
    pos: Tuple[int, int]
    deadline_step: Optional[int] = None


@dataclass
class RoutePlan:
    """A batch route plan for one UAV.

    route_idx: which active plan slot this refers to (for training integration).
    uav_id: target UAV.
    nodes: list of RouteNode in sequence PICKUP/DROP/...

    The environment will not allow changing nodes after dispatch.
    """

    route_idx: int
    uav_id: int
    nodes: List[RouteNode] = field(default_factory=list)


@dataclass
class UAVState:
    """Internal UAV status."""

    uav_id: int
    pos: np.ndarray  # shape (2,)
    battery: float
    status: str = "READY"  # READY, FLYING, CHARGING, BROKEN, DONE
    # Current route following state
    active_plan: Optional[RoutePlan] = None
    active_node_idx: int = 0
    # Target bookkeeping
    target_pos: Optional[np.ndarray] = None


# ---------------------------
# Utilities
# ---------------------------


def manhattan(a: Union[np.ndarray, Sequence[int]], b: Union[np.ndarray, Sequence[int]]) -> int:
    ax, ay = int(a[0]), int(a[1])
    bx, by = int(b[0]), int(b[1])
    return abs(ax - bx) + abs(ay - by)


def clip_int(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


# ---------------------------
# Environment
# ---------------------------


class UAV_ENVIRONMENT_6:
    """Multi-merchant batch route planning UAV environment.

    This environment is intended for RL training and simulation.
    - step(): advances simulation by one tick.
    - dispatch_plans(plans): assign RoutePlan(s) to READY UAV(s).

    Observation additions:
    - to_target: vector from UAV position to current target.
    - dist_to_target: L1 distance to current target.
    - traffic_patch: local map patch around UAV (e.g., congestion grid).

    Mid-flight reroute is disallowed:
    - dispatch_plans will reject plans for UAVs not in READY.
    - dispatch_plans will reject if UAV already has an active_plan.
    """

    def __init__(
        self,
        grid_size: Tuple[int, int] = (100, 100),
        num_uav: int = 10,
        max_steps: int = 2000,
        traffic_map: Optional[np.ndarray] = None,
        traffic_patch_radius: int = 3,
        uav_speed: int = 1,
        battery_capacity: float = 1000.0,
        battery_cost_per_step: float = 1.0,
        seed: Optional[int] = None,
    ):
        self.grid_w, self.grid_h = int(grid_size[0]), int(grid_size[1])
        self.num_uav = int(num_uav)
        self.max_steps = int(max_steps)
        self.traffic_patch_radius = int(traffic_patch_radius)
        self.uav_speed = int(uav_speed)
        self.battery_capacity = float(battery_capacity)
        self.battery_cost_per_step = float(battery_cost_per_step)

        self.rng = np.random.RandomState(seed if seed is not None else 0)

        if traffic_map is None:
            self.traffic_map = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        else:
            assert traffic_map.shape == (self.grid_h, self.grid_w)
            self.traffic_map = traffic_map.astype(np.float32)

        self.step_count = 0

        # UAV States
        self.uavs: List[UAVState] = []
        self.reset()

    # ---------------------------
    # Reset / Observation
    # ---------------------------

    def reset(self) -> Dict[str, Any]:
        self.step_count = 0
        self.uavs = []
        for i in range(self.num_uav):
            x = int(self.rng.randint(0, self.grid_w))
            y = int(self.rng.randint(0, self.grid_h))
            self.uavs.append(
                UAVState(
                    uav_id=i,
                    pos=np.array([x, y], dtype=np.int32),
                    battery=self.battery_capacity,
                    status="READY",
                    active_plan=None,
                    active_node_idx=0,
                    target_pos=None,
                )
            )
        return self.get_obs()

    def get_obs(self) -> Dict[str, Any]:
        """Return observation dict suitable for RL."""
        uav_obs = []
        for u in self.uavs:
            target = u.target_pos if u.target_pos is not None else u.pos
            to_target = (target - u.pos).astype(np.int32)
            dist_to_target = int(manhattan(u.pos, target))
            traffic_patch = self._get_traffic_patch(u.pos)

            uav_obs.append(
                {
                    "uav_id": u.uav_id,
                    "pos": u.pos.copy(),
                    "battery": float(u.battery),
                    "status": u.status,
                    "has_plan": u.active_plan is not None,
                    "active_node_idx": int(u.active_node_idx),
                    "target_pos": target.copy(),
                    "to_target": to_target,
                    "dist_to_target": dist_to_target,
                    "traffic_patch": traffic_patch,
                }
            )

        return {
            "step": int(self.step_count),
            "uav_obs": uav_obs,
            "traffic_map": self.traffic_map,
        }

    def _get_traffic_patch(self, pos: np.ndarray) -> np.ndarray:
        r = self.traffic_patch_radius
        x, y = int(pos[0]), int(pos[1])
        x0, x1 = x - r, x + r + 1
        y0, y1 = y - r, y + r + 1

        patch = np.zeros((2 * r + 1, 2 * r + 1), dtype=np.float32)
        for yy in range(y0, y1):
            for xx in range(x0, x1):
                if 0 <= xx < self.grid_w and 0 <= yy < self.grid_h:
                    patch[yy - y0, xx - x0] = self.traffic_map[yy, xx]
        return patch

    # ---------------------------
    # Dispatching API
    # ---------------------------

    def dispatch_plans(self, plans: Sequence[RoutePlan]) -> Dict[str, Any]:
        """Dispatch multiple RoutePlan(s).

        READY-only:
        - Only UAVs with status READY and no active_plan can be dispatched.

        Mid-flight reroute disallowed:
        - UAV in FLYING/CHARGING/etc. will be rejected.

        Returns a dict with accepted/rejected plans.
        """
        accepted = []
        rejected = []

        for p in plans:
            if p.uav_id < 0 or p.uav_id >= self.num_uav:
                rejected.append((p, "INVALID_UAV_ID"))
                continue
            u = self.uavs[p.uav_id]
            if u.status != "READY":
                rejected.append((p, f"UAV_NOT_READY({u.status})"))
                continue
            if u.active_plan is not None:
                rejected.append((p, "UAV_ALREADY_HAS_PLAN"))
                continue
            if not p.nodes:
                rejected.append((p, "EMPTY_PLAN"))
                continue
            ok, reason = self._validate_plan_for_uav(u, p)
            if not ok:
                rejected.append((p, reason))
                continue

            # Accept: freeze plan and set first target
            u.active_plan = copy.deepcopy(p)
            u.active_node_idx = 0
            u.target_pos = np.array(u.active_plan.nodes[0].pos, dtype=np.int32)
            u.status = "FLYING"  # leaving READY
            accepted.append(p)

        return {"accepted": accepted, "rejected": rejected}

    def _validate_plan_for_uav(self, uav: UAVState, plan: RoutePlan) -> Tuple[bool, str]:
        for n in plan.nodes:
            if n.node_type not in ("PICKUP", "DROP"):
                return False, f"BAD_NODE_TYPE({n.node_type})"
            x, y = int(n.pos[0]), int(n.pos[1])
            if not (0 <= x < self.grid_w and 0 <= y < self.grid_h):
                return False, "NODE_OUT_OF_BOUNDS"
            if n.deadline_step is not None and int(n.deadline_step) < self.step_count:
                return False, "DEADLINE_ALREADY_MISSED"

        # Basic feasibility under battery, using pure simulation evaluator.
        sim = self.simulate_plan(
            uav_pos=uav.pos,
            uav_battery=uav.battery,
            start_step=self.step_count,
            nodes=plan.nodes,
        )
        if not sim["feasible"]:
            return False, f"INFEASIBLE({sim['reason']})"
        return True, "OK"

    # ---------------------------
    # Pure evaluator
    # ---------------------------

    def simulate_plan(
        self,
        uav_pos: Union[np.ndarray, Sequence[int]],
        uav_battery: float,
        start_step: int,
        nodes: Sequence[RouteNode],
    ) -> Dict[str, Any]:
        """Pure evaluator for a route plan.

        Computes travel steps/cost and deadline satisfaction without mutating env.
        Movement model: Manhattan distance, constant speed (cells per step).
        Battery cost: per step.

        Returns dict:
            feasible: bool
            reason: str
            finish_step: int
            total_steps: int
            battery_left: float
            deadline_violations: list of (node_idx, deadline_step, arrive_step)
        """
        pos = np.array(uav_pos, dtype=np.int32).copy()
        battery = float(uav_battery)
        t = int(start_step)

        violations = []
        total_steps = 0

        for i, n in enumerate(nodes):
            target = np.array(n.pos, dtype=np.int32)
            d = manhattan(pos, target)
            # convert distance to steps based on speed
            steps_needed = int(math.ceil(d / max(1, self.uav_speed)))
            cost = steps_needed * self.battery_cost_per_step
            if battery < cost:
                return {
                    "feasible": False,
                    "reason": "INSUFFICIENT_BATTERY",
                    "finish_step": t,
                    "total_steps": total_steps,
                    "battery_left": battery,
                    "deadline_violations": violations,
                }

            t_arrive = t + steps_needed
            if n.deadline_step is not None and t_arrive > int(n.deadline_step):
                violations.append((i, int(n.deadline_step), t_arrive))

            # apply
            battery -= cost
            t = t_arrive
            total_steps += steps_needed
            pos = target

        return {
            "feasible": True,
            "reason": "OK",
            "finish_step": t,
            "total_steps": total_steps,
            "battery_left": battery,
            "deadline_violations": violations,
        }

    # ---------------------------
    # Simulation step
    # ---------------------------

    def step(self) -> Tuple[Dict[str, Any], Dict[str, float], bool, Dict[str, Any]]:
        """Advance environment by one step.

        This is a simplified kinematic step function:
        - Each UAV with an active_plan moves toward its current target_pos.
        - On reaching a node, increments active_node_idx and sets next target.
        - When plan completes, UAV becomes READY again.

        Returns: obs, reward_dict, done, info
        """
        self.step_count += 1
        reward = {"completed_nodes": 0.0, "deadline_penalty": 0.0}
        info: Dict[str, Any] = {"events": []}

        for u in self.uavs:
            if u.active_plan is None:
                # READY or other idle statuses: no movement
                continue

            if u.target_pos is None:
                # should not happen
                u.target_pos = np.array(u.active_plan.nodes[u.active_node_idx].pos, dtype=np.int32)

            # move one step toward target using Manhattan (axis-aligned)
            if (u.pos == u.target_pos).all():
                self._on_reach_node(u, reward, info)
            else:
                self._move_toward(u, u.target_pos)
                u.battery -= self.battery_cost_per_step
                if u.battery < 0:
                    u.battery = 0
                    u.status = "BROKEN"
                    info["events"].append(("BATTERY_DEPLETED", u.uav_id))

                # Check arrival after moving
                if (u.pos == u.target_pos).all():
                    self._on_reach_node(u, reward, info)

        done = self.step_count >= self.max_steps
        obs = self.get_obs()
        return obs, reward, done, info

    def _move_toward(self, u: UAVState, target: np.ndarray) -> None:
        # Move up to uav_speed cells per step in Manhattan manner.
        steps = max(1, int(self.uav_speed))
        for _ in range(steps):
            if (u.pos == target).all():
                return
            dx = int(target[0] - u.pos[0])
            dy = int(target[1] - u.pos[1])
            if dx != 0:
                u.pos[0] += 1 if dx > 0 else -1
            elif dy != 0:
                u.pos[1] += 1 if dy > 0 else -1
            u.pos[0] = clip_int(int(u.pos[0]), 0, self.grid_w - 1)
            u.pos[1] = clip_int(int(u.pos[1]), 0, self.grid_h - 1)

    def _on_reach_node(self, u: UAVState, reward: Dict[str, float], info: Dict[str, Any]) -> None:
        assert u.active_plan is not None
        idx = int(u.active_node_idx)
        node = u.active_plan.nodes[idx]

        # Node completion reward / deadline penalty
        reward["completed_nodes"] += 1.0
        if node.deadline_step is not None and self.step_count > int(node.deadline_step):
            reward["deadline_penalty"] -= 1.0
            info["events"].append(("DEADLINE_MISSED", u.uav_id, idx))

        info["events"].append(("NODE_DONE", u.uav_id, idx, node.node_type, node.order_id))

        # Advance to next node or finish
        u.active_node_idx += 1
        if u.active_node_idx >= len(u.active_plan.nodes):
            # Plan finished
            info["events"].append(("PLAN_DONE", u.uav_id, u.active_plan.route_idx))
            u.active_plan = None
            u.active_node_idx = 0
            u.target_pos = None
            # Back to READY
            if u.status != "BROKEN":
                u.status = "READY"
        else:
            u.target_pos = np.array(u.active_plan.nodes[u.active_node_idx].pos, dtype=np.int32)
            # Status remains FLYING unless broken
            if u.status != "BROKEN":
                u.status = "FLYING"


# Backwards-compatible alias
UAV_ENVIRONMENT = UAV_ENVIRONMENT_6
