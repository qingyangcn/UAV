# UAV_ENVIRONMENT_5.py
# Continuous float-hour timing update while keeping 1-hour steps.

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _truncated_normal(mean: float, std: float, lo: float, hi: float) -> float:
    """Sample from a normal distribution and clamp to [lo, hi]."""
    return _clamp(random.gauss(mean, std), lo, hi)


@dataclass
class Order:
    """Order with continuous timing in hours.

    created_time_hours: continuous float time in *hours* since episode start.
    prep_time_hours: continuous float preparation time in hours.
    ready_time_hours: created_time_hours + prep_time_hours.
    sla_hours_float: continuous float SLA duration in hours.
    due_time_hours: created_time_hours + sla_hours_float.

    Note: The environment still advances in 1-hour steps, but timing comparisons
    use float hours.
    """

    order_id: int
    origin: int
    destination: int

    # Continuous float-hour timeline
    created_time_hours: float
    prep_time_hours: float
    ready_time_hours: float

    sla_hours_float: float
    due_time_hours: float

    status: str = "pending"  # pending/assigned/ready/picked_up/delivered/canceled/expired/lost_after_pickup

    # Optional bookkeeping
    assigned_drone_id: Optional[int] = None
    meta: Dict[str, Any] = field(default_factory=dict)


class ThreeObjectiveDroneDeliveryEnv:
    """Three-objective drone delivery environment.

    This file is adapted to use continuous float-hour times for prep and SLA
    while keeping one step == one hour.
    """

    def __init__(self, *args, **kwargs):
        # Existing init state (kept minimal due to partial context)
        self.t_hours: float = 0.0  # float hours since reset
        self.orders: List[Order] = []
        self.next_order_id: int = 0

        # Reward vector accumulation per step
        self.step_reward_vec = [0.0, 0.0, 0.0]

        # Configurable order spawning parameters (if existing code uses them)
        self.max_orders: int = kwargs.get("max_orders", 200)

    # -----------------------------
    # Time helpers
    # -----------------------------

    @property
    def hour_of_day(self) -> int:
        return int(self.t_hours) % 24

    def _now_time_hours(self) -> float:
        return float(self.t_hours)

    # -----------------------------
    # Order spawning
    # -----------------------------

    def spawn_order(self, origin: int, destination: int) -> Order:
        """Spawn an order with truncated-normal prep and SLA in minutes.

        prep_time_minutes ~ N(12, 4), clamped to [1, 30]
        sla_minutes ~ N(45, 10), clamped to [15, 90]

        Converted to float hours.
        """

        now_time_hours = self._now_time_hours()

        prep_time_minutes = _truncated_normal(mean=12.0, std=4.0, lo=1.0, hi=30.0)
        sla_minutes = _truncated_normal(mean=45.0, std=10.0, lo=15.0, hi=90.0)

        prep_time_hours = prep_time_minutes / 60.0
        sla_hours_float = sla_minutes / 60.0

        created_time_hours = now_time_hours
        ready_time_hours = created_time_hours + prep_time_hours
        due_time_hours = created_time_hours + sla_hours_float

        order = Order(
            order_id=self.next_order_id,
            origin=origin,
            destination=destination,
            created_time_hours=created_time_hours,
            prep_time_hours=prep_time_hours,
            ready_time_hours=ready_time_hours,
            sla_hours_float=sla_hours_float,
            due_time_hours=due_time_hours,
            status="pending",
        )
        self.next_order_id += 1
        self.orders.append(order)
        return order

    # -----------------------------
    # Order availability / expiration
    # -----------------------------

    def _is_order_ready(self, order: Order, now_time_hours: float) -> bool:
        return now_time_hours >= order.ready_time_hours

    def _is_order_overdue(self, order: Order, now_time_hours: float) -> bool:
        return now_time_hours >= order.due_time_hours

    def _auto_expire_orders(self) -> None:
        """Auto-expire orders based on float-hour overdue logic.

        - If overdue and status in pending/assigned/ready: cancel as 'expired'
        - If overdue and status == picked_up: cancel as 'lost_after_pickup'
        """

        now_time_hours = self._now_time_hours()
        for order in self.orders:
            if order.status in ("delivered", "canceled", "expired", "lost_after_pickup"):
                continue

            overdue = self._is_order_overdue(order, now_time_hours)
            if not overdue:
                continue

            if order.status in ("pending", "assigned", "ready"):
                order.status = "expired"
            elif order.status == "picked_up":
                order.status = "lost_after_pickup"

    def _update_ready_states(self) -> None:
        """Transition assigned/pending orders to READY once ready_time_hours passes."""
        now_time_hours = self._now_time_hours()
        for order in self.orders:
            if order.status in ("delivered", "canceled", "expired", "lost_after_pickup"):
                continue

            # READY state should become available after ready_time_hours.
            if order.status in ("pending", "assigned") and self._is_order_ready(order, now_time_hours):
                order.status = "ready"

    # -----------------------------
    # Step
    # -----------------------------

    def step(self, action: Any) -> Tuple[Any, List[float], bool, Dict[str, Any]]:
        """Advance environment by one step == 1.0 hour.

        Reward vector accumulation still occurs once per step.
        """

        # Reset per-step reward vector accumulator
        self.step_reward_vec = [0.0, 0.0, 0.0]

        # --- existing action application would go here ---
        # self._apply_action(action)

        # Update order readiness based on float hours
        self._update_ready_states()

        # Auto-expire / lost-after-pickup based on float hours
        self._auto_expire_orders()

        # --- compute rewards once per step ---
        # self.step_reward_vec = self._compute_reward_vector()

        # Advance time by one hour (float)
        self.t_hours += 1.0

        obs = self._get_obs() if hasattr(self, "_get_obs") else None
        done = False
        info = {
            "t_hours": self.t_hours,
            "hour_of_day": int(self.t_hours) % 24,
        }
        return obs, self.step_reward_vec, done, info

    def reset(self, *args, **kwargs):
        self.t_hours = 0.0
        self.orders = []
        self.next_order_id = 0
        self.step_reward_vec = [0.0, 0.0, 0.0]
        return self._get_obs() if hasattr(self, "_get_obs") else None
