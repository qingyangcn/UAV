# UAV_ENVIRONMENT_5.py
# Fully revised version
# Fixes:
# 1) Unified hour-based time scale for 08:00-19:00 simulation using abs_hour_index and t_hours;
#    order probability and weather indexing use hours, not steps.
# 2) Fix episode_r_vec double counting by accumulating only once after final_bonus.
# 3) Remove double movement by eliminating immediate position updates; positions update only once per step;
#    avoid duplicate merchant preparation updates.
# 4) Optimize state synchronization: avoid force-completing orders without being near customer; instead cancel
#    as lost_after_pickup with penalties, only complete if drone is near customer.
# 5) Add backlog control: order expiration (READY wait timeout and total time > promised*multiplier),
#    merchant queue overload rejection, and improved backlog penalty using late backlog.
#
# Commit: "Fix time scale, reward accounting, movement consistency, and backlog/SLA handling"

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# -----------------------------
# Helper dataclasses
# -----------------------------


@dataclass
class Order:
    oid: int
    merchant_id: int
    customer_id: int

    created_t: float  # hours since 08:00
    promised_t: float  # hours since 08:00 (delivery promise)

    prep_start_t: Optional[float] = None
    ready_t: Optional[float] = None

    pickup_t: Optional[float] = None
    delivered_t: Optional[float] = None

    state: str = "CREATED"  # CREATED -> PREPARING -> READY -> ONROUTE -> DELIVERED/CANCELLED

    assigned_drone: Optional[int] = None

    cancel_reason: Optional[str] = None

    def is_active(self) -> bool:
        return self.state not in ("DELIVERED", "CANCELLED")


@dataclass
class Merchant:
    mid: int
    xy: np.ndarray
    prep_queue: List[int] = field(default_factory=list)  # list of order ids in prep/ready pipeline


@dataclass
class Customer:
    cid: int
    xy: np.ndarray


@dataclass
class Drone:
    did: int
    xy: np.ndarray
    v_kmh: float
    battery: float = 1.0

    carrying_oid: Optional[int] = None


# -----------------------------
# Environment
# -----------------------------


class UAV_ENVIRONMENT_5:
    """A simplified but consistent UAV delivery simulator.

    Notes on revisions:
    - Time uses hours (t_hours) from 08:00 to 19:00.
    - All processes (order generation, weather, prep, deadline) use hour-based time.
    - Movement is applied once per step (no immediate update helpers).
    - Delivery completion requires proximity to customer.
    - Backlog controls (expiration and queue overload) added.
    """

    # Simulation window
    DAY_START_HOUR = 8
    DAY_END_HOUR = 19  # exclusive end for indices; 08:00-19:00 => 11 hours
    SIM_HOURS = DAY_END_HOUR - DAY_START_HOUR

    def __init__(
        self,
        n_drones: int = 10,
        n_merchants: int = 20,
        n_customers: int = 50,
        area_size_km: float = 10.0,
        dt_minutes: float = 1.0,
        seed: int = 0,
        # Demand / weather arrays can be supplied per hour
        hourly_order_prob: Optional[List[float]] = None,
        hourly_weather: Optional[List[int]] = None,
        # Prep model
        prep_time_mean_min: float = 12.0,
        prep_time_std_min: float = 4.0,
        # SLA / backlog controls
        promised_mean_min: float = 35.0,
        promised_std_min: float = 8.0,
        promised_multiplier_expire: float = 1.8,
        ready_wait_timeout_min: float = 20.0,
        merchant_queue_limit: int = 8,
        # Proximity thresholds
        pickup_radius_km: float = 0.10,
        drop_radius_km: float = 0.10,
    ):
        self.rng = random.Random(seed)
        np.random.seed(seed)

        self.n_drones = n_drones
        self.n_merchants = n_merchants
        self.n_customers = n_customers
        self.area_size_km = area_size_km

        self.dt_hours = dt_minutes / 60.0
        self.steps_per_hour = int(round(1.0 / self.dt_hours))

        # Hour-indexed inputs (len = SIM_HOURS)
        if hourly_order_prob is None:
            # Default: midday peak
            base = [0.04] * self.SIM_HOURS
            for h in range(self.SIM_HOURS):
                if 3 <= h <= 6:  # 11:00-14:00
                    base[h] = 0.08
            hourly_order_prob = base
        if hourly_weather is None:
            # 0=clear,1=rain,2=windy etc.
            hourly_weather = [0] * self.SIM_HOURS
        if len(hourly_order_prob) != self.SIM_HOURS:
            raise ValueError("hourly_order_prob must have length SIM_HOURS")
        if len(hourly_weather) != self.SIM_HOURS:
            raise ValueError("hourly_weather must have length SIM_HOURS")

        self.hourly_order_prob = list(hourly_order_prob)
        self.hourly_weather = list(hourly_weather)

        # Prep model
        self.prep_mu_h = prep_time_mean_min / 60.0
        self.prep_sigma_h = prep_time_std_min / 60.0

        # SLA
        self.promised_mu_h = promised_mean_min / 60.0
        self.promised_sigma_h = promised_std_min / 60.0
        self.promised_multiplier_expire = promised_multiplier_expire
        self.ready_wait_timeout_h = ready_wait_timeout_min / 60.0
        self.merchant_queue_limit = merchant_queue_limit

        # Radii
        self.pickup_radius_km = pickup_radius_km
        self.drop_radius_km = drop_radius_km

        # World entities
        self.merchants: Dict[int, Merchant] = {}
        self.customers: Dict[int, Customer] = {}
        self.drones: Dict[int, Drone] = {}

        # Orders
        self.orders: Dict[int, Order] = {}
        self.next_oid = 1

        # Time
        self.t_hours: float = 0.0  # 0 means 08:00
        self.step_i: int = 0

        # Episode reward accounting
        self.episode_r: float = 0.0
        self.episode_r_vec: List[float] = []  # store per-step reward only once

        # Stats
        self.n_created = 0
        self.n_rejected_overload = 0
        self.n_expired = 0
        self.n_cancel_lost_after_pickup = 0
        self.n_delivered = 0

        self.reset()

    # -------------------------
    # Utilities
    # -------------------------

    def _abs_hour_index(self) -> int:
        """Absolute hour index in [0, SIM_HOURS-1] based on t_hours."""
        idx = int(math.floor(self.t_hours))
        return max(0, min(self.SIM_HOURS - 1, idx))

    def _dist(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))

    def _sample_xy(self) -> np.ndarray:
        return np.array(
            [self.rng.random() * self.area_size_km, self.rng.random() * self.area_size_km],
            dtype=np.float32,
        )

    def _weather_speed_factor(self, w: int) -> float:
        # Example mapping
        if w == 0:
            return 1.0
        if w == 1:
            return 0.85
        if w == 2:
            return 0.75
        return 0.9

    # -------------------------
    # Reset
    # -------------------------

    def reset(self):
        self.merchants = {i: Merchant(i, self._sample_xy()) for i in range(self.n_merchants)}
        self.customers = {i: Customer(i, self._sample_xy()) for i in range(self.n_customers)}
        self.drones = {
            i: Drone(i, self._sample_xy(), v_kmh=30.0)
            for i in range(self.n_drones)
        }

        self.orders.clear()
        self.next_oid = 1

        self.t_hours = 0.0
        self.step_i = 0

        self.episode_r = 0.0
        self.episode_r_vec = []

        self.n_created = 0
        self.n_rejected_overload = 0
        self.n_expired = 0
        self.n_cancel_lost_after_pickup = 0
        self.n_delivered = 0

        return self._get_obs()

    # -------------------------
    # Observation
    # -------------------------

    def _get_obs(self) -> Dict:
        # Minimal state; adapt as needed.
        hour_idx = self._abs_hour_index()
        w = self.hourly_weather[hour_idx]
        return {
            "t_hours": self.t_hours,
            "abs_hour_index": hour_idx,
            "weather": w,
            "n_active_orders": sum(1 for o in self.orders.values() if o.is_active()),
            "drones": {
                d.did: {
                    "xy": d.xy.copy(),
                    "carrying": d.carrying_oid,
                }
                for d in self.drones.values()
            },
        }

    # -------------------------
    # Order lifecycle
    # -------------------------

    def _create_order(self):
        mid = self.rng.randrange(self.n_merchants)
        cid = self.rng.randrange(self.n_customers)

        m = self.merchants[mid]

        # Backlog control: reject if merchant queue overloaded
        if len(m.prep_queue) >= self.merchant_queue_limit:
            self.n_rejected_overload += 1
            # Small penalty for rejection/backlog
            return -0.2

        created_t = self.t_hours
        promised_dt = max(0.15, self.rng.gauss(self.promised_mu_h, self.promised_sigma_h))
        promised_t = created_t + promised_dt

        oid = self.next_oid
        self.next_oid += 1

        o = Order(
            oid=oid,
            merchant_id=mid,
            customer_id=cid,
            created_t=created_t,
            promised_t=promised_t,
            state="PREPARING",
            prep_start_t=self.t_hours,
        )
        self.orders[oid] = o

        # enqueue at merchant
        m.prep_queue.append(oid)

        self.n_created += 1
        return 0.0

    def _advance_prep_queues(self):
        """Advance merchant preparation once per step (no duplication)."""
        for m in self.merchants.values():
            # For each order in queue, check if ready time should be set
            for oid in list(m.prep_queue):
                o = self.orders.get(oid)
                if o is None or not o.is_active():
                    if oid in m.prep_queue:
                        m.prep_queue.remove(oid)
                    continue

                if o.state == "PREPARING":
                    if o.ready_t is None:
                        prep_duration = max(0.03, self.rng.gauss(self.prep_mu_h, self.prep_sigma_h))
                        o.ready_t = o.prep_start_t + prep_duration
                    if self.t_hours >= o.ready_t:
                        o.state = "READY"

                # Once picked up/delivered/cancelled, remove from merchant queue
                if o.state in ("ONROUTE", "DELIVERED", "CANCELLED"):
                    if oid in m.prep_queue:
                        m.prep_queue.remove(oid)

    def _expire_orders(self) -> float:
        """Cancel orders that violate wait/SLA constraints. Returns penalty."""
        penalty = 0.0
        for o in self.orders.values():
            if not o.is_active():
                continue

            total_age = self.t_hours - o.created_t
            max_age = (o.promised_t - o.created_t) * self.promised_multiplier_expire

            # Total time hard expire (SLA)
            if total_age > max_age:
                o.state = "CANCELLED"
                o.cancel_reason = "expired_total_time"
                self.n_expired += 1
                penalty -= 1.0
                continue

            # READY wait timeout (food getting cold)
            if o.state == "READY" and o.ready_t is not None:
                ready_wait = self.t_hours - o.ready_t
                if ready_wait > self.ready_wait_timeout_h:
                    o.state = "CANCELLED"
                    o.cancel_reason = "expired_ready_wait"
                    self.n_expired += 1
                    penalty -= 0.8

        return penalty

    # -------------------------
    # Movement and completion
    # -------------------------

    def _move_drone_towards(self, d: Drone, target_xy: np.ndarray, speed_factor: float):
        """Move drone once per step towards a target."""
        v = d.v_kmh * speed_factor
        max_dist = v * self.dt_hours
        vec = target_xy - d.xy
        dist = float(np.linalg.norm(vec))
        if dist <= 1e-9:
            return
        if dist <= max_dist:
            d.xy = target_xy.copy()
        else:
            d.xy = d.xy + (vec / dist) * max_dist

    def _attempt_pickup_and_drop(self) -> float:
        """Pickup READY orders if assigned and near merchant; drop if near customer.

        No force completion: only deliver if within drop radius.
        If carrying too long past SLA, cancel as lost_after_pickup.
        """
        r = 0.0
        for d in self.drones.values():
            # Pickup
            if d.carrying_oid is None:
                # Find an assigned READY order for this drone
                assigned = None
                for o in self.orders.values():
                    if o.is_active() and o.assigned_drone == d.did and o.state == "READY":
                        assigned = o
                        break
                if assigned is not None:
                    mxy = self.merchants[assigned.merchant_id].xy
                    if self._dist(d.xy, mxy) <= self.pickup_radius_km:
                        assigned.state = "ONROUTE"
                        assigned.pickup_t = self.t_hours
                        d.carrying_oid = assigned.oid

            # Drop / cancel after pickup
            if d.carrying_oid is not None:
                o = self.orders.get(d.carrying_oid)
                if o is None or not o.is_active():
                    d.carrying_oid = None
                    continue

                cxy = self.customers[o.customer_id].xy
                if self._dist(d.xy, cxy) <= self.drop_radius_km:
                    o.state = "DELIVERED"
                    o.delivered_t = self.t_hours
                    d.carrying_oid = None
                    self.n_delivered += 1

                    # Reward: on-time bonus, late penalty
                    lateness = max(0.0, o.delivered_t - o.promised_t)
                    r += 2.0 - 4.0 * lateness
                else:
                    # If way past promise after pickup, cancel as lost_after_pickup
                    if o.pickup_t is not None:
                        if (self.t_hours - o.created_t) > (o.promised_t - o.created_t) * self.promised_multiplier_expire:
                            o.state = "CANCELLED"
                            o.cancel_reason = "lost_after_pickup"
                            d.carrying_oid = None
                            self.n_cancel_lost_after_pickup += 1
                            r -= 2.0

        return r

    # -------------------------
    # Policy hooks
    # -------------------------

    def assign_orders(self):
        """Simple heuristic assignment: assign unassigned READY orders to nearest idle drone."""
        for o in self.orders.values():
            if not o.is_active() or o.state != "READY" or o.assigned_drone is not None:
                continue

            mxy = self.merchants[o.merchant_id].xy
            best_d = None
            best_dist = 1e18
            for d in self.drones.values():
                if d.carrying_oid is not None:
                    continue
                # only consider drones not already assigned to another READY order
                already = any(
                    oo.is_active() and oo.state == "READY" and oo.assigned_drone == d.did
                    for oo in self.orders.values()
                )
                if already:
                    continue
                dist = self._dist(d.xy, mxy)
                if dist < best_dist:
                    best_dist = dist
                    best_d = d
            if best_d is not None:
                o.assigned_drone = best_d.did

    def _move_drones(self):
        hour_idx = self._abs_hour_index()
        speed_factor = self._weather_speed_factor(self.hourly_weather[hour_idx])

        for d in self.drones.values():
            # If carrying, go to customer
            if d.carrying_oid is not None:
                o = self.orders.get(d.carrying_oid)
                if o is None or not o.is_active():
                    d.carrying_oid = None
                    continue
                target = self.customers[o.customer_id].xy
                self._move_drone_towards(d, target, speed_factor)
            else:
                # If assigned READY order, go to merchant
                target = None
                for o in self.orders.values():
                    if o.is_active() and o.state == "READY" and o.assigned_drone == d.did:
                        target = self.merchants[o.merchant_id].xy
                        break
                if target is not None:
                    self._move_drone_towards(d, target, speed_factor)

    # -------------------------
    # Backlog penalty
    # -------------------------

    def _late_backlog_penalty(self) -> float:
        """Penalty proportional to number of active orders that are already late."""
        late = 0
        for o in self.orders.values():
            if not o.is_active():
                continue
            if self.t_hours > o.promised_t:
                late += 1
        # mild per-step pressure
        return -0.02 * late

    # -------------------------
    # Step
    # -------------------------

    def step(self, action=None) -> Tuple[Dict, float, bool, Dict]:
        """Advance env by one dt.

        action is ignored in this reference implementation.
        """
        reward = 0.0

        # 1) Hour-based demand generation (uses abs_hour_index)
        hour_idx = self._abs_hour_index()
        p = self.hourly_order_prob[hour_idx]
        # interpret p as probability of one new order each minute-step (kept simple)
        if self.rng.random() < p:
            reward += self._create_order()

        # 2) Advance preparation once per step
        self._advance_prep_queues()

        # 3) Assign orders
        self.assign_orders()

        # 4) Move drones once per step
        self._move_drones()

        # 5) Attempt pickup/drop and handle lost_after_pickup
        reward += self._attempt_pickup_and_drop()

        # 6) Expire stale orders
        reward += self._expire_orders()

        # 7) Backlog penalty (late backlog)
        reward += self._late_backlog_penalty()

        # 8) Advance time
        self.step_i += 1
        self.t_hours += self.dt_hours

        done = self.t_hours >= self.SIM_HOURS

        info = {
            "n_created": self.n_created,
            "n_delivered": self.n_delivered,
            "n_expired": self.n_expired,
            "n_rejected_overload": self.n_rejected_overload,
            "n_cancel_lost_after_pickup": self.n_cancel_lost_after_pickup,
        }

        # 9) Final bonus once at end, and prevent double-counting in episode_r_vec
        if done:
            final_bonus = 0.1 * self.n_delivered - 0.2 * self.n_expired
            reward += final_bonus

        self.episode_r += reward
        self.episode_r_vec.append(reward)

        return self._get_obs(), reward, done, info
