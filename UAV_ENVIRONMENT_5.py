"""UAV_ENVIRONMENT_5.py

Restored and updated ThreeObjectiveDroneDeliveryEnv-based implementation.

Key fixes:
- Enforce unified time scale: one environment step equals one real hour (steps_per_hour=1).
- Provide a single authoritative absolute hour index (abs_hour_index) and hour-of-day via DailyTimeSystem.
- Fix episode_r_vec double counting.
- Ensure drones move only once per step (remove immediate position updates during assignment).
- Replace force_complete_order synchronization with proximity-checked delivery; otherwise cancel as
  lost_after_pickup with penalties.
- Implement SLA-based expiration / auto-cancel for overdue orders.

This module is intentionally self-contained and avoids relying on the previously rewritten logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import math
import numpy as np


# -----------------------------
# Time system
# -----------------------------


class DailyTimeSystem:
    """Maintain absolute time in hours.

    One environment step == one hour. The env stores abs_hour_index and provides hour-of-day.
    """

    def __init__(self, start_abs_hour: int = 0):
        self.abs_hour_index = int(start_abs_hour)

    @property
    def hour_of_day(self) -> int:
        return int(self.abs_hour_index % 24)

    def step(self, hours: int = 1) -> None:
        self.abs_hour_index += int(hours)


# -----------------------------
# Data structures
# -----------------------------


@dataclass
class Order:
    order_id: int
    pickup_xy: np.ndarray  # shape (2,)
    dropoff_xy: np.ndarray  # shape (2,)

    created_abs_hour: int
    sla_hours: int

    status: str = "pending"  # pending->assigned->picked_up->delivered; or cancelled/expired/lost_after_pickup
    assigned_drone_id: Optional[int] = None

    picked_abs_hour: Optional[int] = None
    delivered_abs_hour: Optional[int] = None
    cancelled_abs_hour: Optional[int] = None

    cancel_reason: Optional[str] = None

    @property
    def due_abs_hour(self) -> int:
        return int(self.created_abs_hour + self.sla_hours)

    def is_overdue(self, now_abs_hour: int) -> bool:
        return now_abs_hour >= self.due_abs_hour


@dataclass
class Drone:
    drone_id: int
    pos_xy: np.ndarray  # shape (2,)
    speed_km_per_hour: float

    carrying_order_id: Optional[int] = None
    target_xy: Optional[np.ndarray] = None  # immediate target (pickup or dropoff)

    def distance_to(self, xy: np.ndarray) -> float:
        return float(np.linalg.norm(self.pos_xy - xy))


# -----------------------------
# Environment
# -----------------------------


class ThreeObjectiveDroneDeliveryEnv:
    """A minimal ThreeObjectiveDroneDeliveryEnv style environment.

    Note: this file historically existed in the repo with a ThreeObjectiveDroneDeliveryEnv-based
    implementation. This version restores that style and applies fixes requested.

    Reward vector convention (episode_r_vec accumulates EACH once per step):
      r_vec = [profit_like, service_quality, penalty_like]

    where service_quality is positive for on-time delivery; penalty_like is negative penalties.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        map_size_km: float = 20.0,
        n_drones: int = 5,
        drone_speed_km_per_hour: float = 60.0,
        delivery_radius_km: float = 0.25,
        pickup_radius_km: float = 0.25,
        order_sla_hours: int = 6,
        max_episode_hours: int = 24 * 7,
        seed: int = 0,
    ):
        # Time scale enforcement
        self.steps_per_hour = 1  # enforced
        self._hours_per_step = 1

        self.rng = np.random.default_rng(seed)

        self.map_size_km = float(map_size_km)
        self.delivery_radius_km = float(delivery_radius_km)
        self.pickup_radius_km = float(pickup_radius_km)
        self.default_sla_hours = int(order_sla_hours)
        self.max_episode_hours = int(max_episode_hours)

        self.time = DailyTimeSystem(start_abs_hour=0)

        self.drones: List[Drone] = []
        for i in range(int(n_drones)):
            p0 = self._random_xy()
            self.drones.append(
                Drone(
                    drone_id=i,
                    pos_xy=p0,
                    speed_km_per_hour=float(drone_speed_km_per_hour),
                )
            )

        self.orders: Dict[int, Order] = {}
        self._next_order_id = 0

        self.episode_step = 0
        self.episode_r_vec = np.zeros(3, dtype=np.float64)

        # Accounting for debugging
        self.last_step_r_vec = np.zeros(3, dtype=np.float64)

    # -----------------------------
    # Utilities
    # -----------------------------

    def _random_xy(self) -> np.ndarray:
        # positions are in a square [0, map_size]x[0, map_size]
        return self.rng.random(2, dtype=np.float64) * self.map_size_km

    def _clamp_xy(self, xy: np.ndarray) -> np.ndarray:
        return np.clip(xy, 0.0, self.map_size_km)

    def _move_towards(self, src: np.ndarray, dst: np.ndarray, max_dist: float) -> np.ndarray:
        v = dst - src
        d = float(np.linalg.norm(v))
        if d <= 1e-12:
            return src.copy()
        if d <= max_dist:
            return dst.copy()
        return src + (v / d) * max_dist

    # -----------------------------
    # Order generation / lifecycle
    # -----------------------------

    def spawn_order(self, pickup_xy: Optional[np.ndarray] = None, dropoff_xy: Optional[np.ndarray] = None, sla_hours: Optional[int] = None) -> int:
        if pickup_xy is None:
            pickup_xy = self._random_xy()
        if dropoff_xy is None:
            dropoff_xy = self._random_xy()
        if sla_hours is None:
            sla_hours = self.default_sla_hours

        oid = self._next_order_id
        self._next_order_id += 1

        o = Order(
            order_id=oid,
            pickup_xy=np.array(pickup_xy, dtype=np.float64),
            dropoff_xy=np.array(dropoff_xy, dtype=np.float64),
            created_abs_hour=int(self.time.abs_hour_index),
            sla_hours=int(sla_hours),
        )
        self.orders[oid] = o
        return oid

    def _auto_expire_or_cancel_overdue_orders(self) -> np.ndarray:
        """Auto-cancel overdue orders.

        - If pending or assigned but not picked up, cancel as "expired".
        - If picked_up but not delivered, cancel as "lost_after_pickup".

        Returns r_vec penalties applied this step.
        """
        r_vec = np.zeros(3, dtype=np.float64)
        now = int(self.time.abs_hour_index)
        for o in self.orders.values():
            if o.status in ("delivered", "cancelled", "expired", "lost_after_pickup"):
                continue
            if not o.is_overdue(now):
                continue

            if o.status in ("pending", "assigned"):
                o.status = "expired"
                o.cancel_reason = "sla_expired"
                o.cancelled_abs_hour = now
                # penalty for failing before pickup
                r_vec[2] -= 5.0
                # unassign drone if any
                if o.assigned_drone_id is not None:
                    d = self.drones[o.assigned_drone_id]
                    if d.carrying_order_id == o.order_id:
                        d.carrying_order_id = None
                        d.target_xy = None
                    elif d.target_xy is not None:
                        # if it was heading to pickup
                        d.target_xy = None
                o.assigned_drone_id = None

            elif o.status == "picked_up":
                o.status = "lost_after_pickup"
                o.cancel_reason = "sla_expired_after_pickup"
                o.cancelled_abs_hour = now
                # stronger penalty because the parcel was onboard
                r_vec[2] -= 12.0
                if o.assigned_drone_id is not None:
                    d = self.drones[o.assigned_drone_id]
                    if d.carrying_order_id == o.order_id:
                        d.carrying_order_id = None
                        d.target_xy = None
                o.assigned_drone_id = None

        return r_vec

    # -----------------------------
    # Assignment / action interface
    # -----------------------------

    def assign_order_to_drone(self, order_id: int, drone_id: int) -> bool:
        """Assign order to drone without moving the drone immediately.

        Fix: previous rewrite updated drone position immediately during assignment, causing double
        movement in a single step. This function only sets targets.
        """
        if order_id not in self.orders:
            return False
        if drone_id < 0 or drone_id >= len(self.drones):
            return False

        o = self.orders[order_id]
        d = self.drones[drone_id]

        if o.status != "pending":
            return False
        if d.carrying_order_id is not None:
            return False

        o.status = "assigned"
        o.assigned_drone_id = drone_id

        # set target to pickup; do not move yet
        d.target_xy = o.pickup_xy.copy()
        return True

    # -----------------------------
    # Step / dynamics
    # -----------------------------

    def reset(self, *, seed: Optional[int] = None) -> Tuple[dict, dict]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.time = DailyTimeSystem(start_abs_hour=0)
        self.episode_step = 0
        self.episode_r_vec = np.zeros(3, dtype=np.float64)
        self.last_step_r_vec = np.zeros(3, dtype=np.float64)
        self.orders.clear()
        self._next_order_id = 0
        for d in self.drones:
            d.pos_xy = self._random_xy()
            d.carrying_order_id = None
            d.target_xy = None

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def _deliver_if_close_else_cancel_if_impossible(self) -> np.ndarray:
        """Synchronization logic without force_complete_order.

        For each drone:
        - If it has assigned order and is close enough to pickup, pick it up.
        - If carrying and close enough to dropoff, deliver.

        If carrying but not close enough, do nothing (no teleporting).

        If a drone is carrying an order that has been cancelled/expired/lost, drop it.
        """
        r_vec = np.zeros(3, dtype=np.float64)
        now = int(self.time.abs_hour_index)

        for d in self.drones:
            # if targetless, nothing
            if d.carrying_order_id is not None:
                oid = d.carrying_order_id
                o = self.orders.get(oid)
                if o is None or o.status in ("cancelled", "expired", "lost_after_pickup"):
                    d.carrying_order_id = None
                    d.target_xy = None
                    continue

                # attempt delivery if in radius
                dist = d.distance_to(o.dropoff_xy)
                if dist <= self.delivery_radius_km:
                    o.status = "delivered"
                    o.delivered_abs_hour = now
                    d.carrying_order_id = None
                    d.target_xy = None

                    # rewards
                    r_vec[0] += 10.0  # profit-like
                    # service quality by lateness
                    lateness = max(0, now - o.due_abs_hour)
                    if lateness == 0:
                        r_vec[1] += 5.0
                    else:
                        r_vec[1] -= 1.0 * float(lateness)
                    continue

            # not carrying: see if heading to pickup for some assigned order
            # find assigned order for this drone
            assigned_order: Optional[Order] = None
            for o in self.orders.values():
                if o.assigned_drone_id == d.drone_id and o.status in ("assigned", "pending"):
                    assigned_order = o
                    break
            if assigned_order is None:
                continue

            o = assigned_order
            # If order became overdue, it will be expired by auto-cancel before we get here.
            if o.status == "assigned":
                dist = d.distance_to(o.pickup_xy)
                if dist <= self.pickup_radius_km:
                    o.status = "picked_up"
                    o.picked_abs_hour = now
                    d.carrying_order_id = o.order_id
                    d.target_xy = o.dropoff_xy.copy()
                    # small pickup reward
                    r_vec[0] += 1.0

        return r_vec

    def _move_drones_once(self) -> None:
        """Move each drone at most once for the step."""
        max_dist = 0.0
        for d in self.drones:
            max_dist = d.speed_km_per_hour * self._hours_per_step
            if d.target_xy is None:
                continue
            new_xy = self._move_towards(d.pos_xy, d.target_xy, max_dist)
            d.pos_xy = self._clamp_xy(new_xy)

    def step(self, action=None) -> Tuple[dict, np.ndarray, bool, bool, dict]:
        """Advance the env by exactly one hour.

        action: optional external dispatcher; kept for compatibility.
        """
        # Enforce step-hour mapping
        self._hours_per_step = 1
        self.steps_per_hour = 1

        self.last_step_r_vec = np.zeros(3, dtype=np.float64)

        # 1) Auto-cancel overdue orders BEFORE movement/attempts
        r_exp = self._auto_expire_or_cancel_overdue_orders()
        self.last_step_r_vec += r_exp

        # 2) Move drones ONCE
        self._move_drones_once()

        # 3) Pickup/deliver if within radius (no force completion)
        r_sync = self._deliver_if_close_else_cancel_if_impossible()
        self.last_step_r_vec += r_sync

        # 4) Advance time by one hour
        self.time.step(hours=1)
        self.episode_step += 1

        # 5) Accumulate episode vector ONCE (fix double counting)
        self.episode_r_vec += self.last_step_r_vec

        terminated = False
        truncated = self.time.abs_hour_index >= self.max_episode_hours

        obs = self._get_obs()
        info = self._get_info()
        return obs, self.last_step_r_vec.copy(), terminated, truncated, info

    # -----------------------------
    # Observations / info
    # -----------------------------

    def _get_obs(self) -> dict:
        """Lightweight dict observation."""
        return {
            "abs_hour_index": int(self.time.abs_hour_index),
            "hour_of_day": int(self.time.hour_of_day),
            "drones": np.array([d.pos_xy for d in self.drones], dtype=np.float64),
            "n_pending": int(sum(1 for o in self.orders.values() if o.status == "pending")),
            "n_assigned": int(sum(1 for o in self.orders.values() if o.status == "assigned")),
            "n_picked_up": int(sum(1 for o in self.orders.values() if o.status == "picked_up")),
            "n_delivered": int(sum(1 for o in self.orders.values() if o.status == "delivered")),
        }

    def _get_info(self) -> dict:
        return {
            "episode_step": int(self.episode_step),
            "abs_hour_index": int(self.time.abs_hour_index),
            "hour_of_day": int(self.time.hour_of_day),
            "episode_r_vec": self.episode_r_vec.copy(),
        }
