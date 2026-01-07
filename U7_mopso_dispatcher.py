"""
U7 MOPSO Dispatcher: Assignment-only (no route planning).

This dispatcher uses MOPSO to assign READY orders to drones,
but does NOT generate planned_stops. PPO handles task selection and routing.

Key differences from U6:
- Only assigns orders to drones (READY -> ASSIGNED)
- Does not create planned_stops
- Respects capacity constraints
- Uses existing MOPSO logic for assignment optimization
"""
import math
import numpy as np
from typing import Dict, List, Tuple, Optional

# NOTE: This module depends on U6_mopso_dispatcher for the core MOPSO optimization logic.
# We reuse MOPSOPlanner's _run_mopso method to avoid code duplication while changing
# only the assignment application (no route planning in U7).
from U6_mopso_dispatcher import MOPSOPlanner

# Import OrderStatus from environment for proper enum usage
try:
    from UAV_ENVIRONMENT_7 import OrderStatus
except ImportError:
    # Fallback if import fails
    OrderStatus = None


class U7MOPSOAssigner:
    """
    Assignment-only MOPSO dispatcher for U7.

    Uses MOPSO to optimize order-to-drone assignment without
    generating route plans. PPO handles task selection.
    """

    def __init__(self,
                 n_particles: int = 30,
                 n_iterations: int = 10,
                 max_orders: int = 200,
                 max_orders_per_drone: int = 10,
                 seed: Optional[int] = None,
                 **mopso_kwargs):
        """
        Initialize U7 MOPSO assigner.

        Args:
            n_particles: Number of PSO particles
            n_iterations: PSO iterations
            max_orders: Maximum orders to consider
            max_orders_per_drone: Maximum orders per drone (capacity constraint)
            seed: Random seed
            **mopso_kwargs: Additional arguments for MOPSOPlanner
        """
        # Reuse the existing MOPSO planner but only use assignment logic
        self.planner = MOPSOPlanner(
            n_particles=n_particles,
            n_iterations=n_iterations,
            max_orders=max_orders,
            max_orders_per_drone=max_orders_per_drone,
            seed=seed,
            **mopso_kwargs
        )

    def assign_orders(self, env,
                     ready_orders: Optional[List[dict]] = None,
                     drones: Optional[List[dict]] = None,
                     merchants: Optional[Dict] = None,
                     constraints: Optional[dict] = None,
                     objective_weights: Optional[np.ndarray] = None) -> Dict[int, List[int]]:
        """
        Assign READY orders to drones using MOPSO.

        Args:
            env: Environment instance
            ready_orders: List of ready order snapshots
            drones: List of drone snapshots
            merchants: Dict of merchant snapshots
            constraints: Constraint parameters
            objective_weights: Weights for solution selection

        Returns:
            Assignment dict: {drone_id: [order_ids]}
        """
        # Get snapshots if not provided
        if ready_orders is None:
            ready_orders = env.get_ready_orders_snapshot(limit=self.planner.max_orders)
        if drones is None:
            drones = env.get_drones_snapshot()
        if merchants is None:
            merchants = env.get_merchants_snapshot()
        if constraints is None:
            constraints = env.get_route_plan_constraints()
        if objective_weights is None:
            objective_weights = getattr(env, 'objective_weights', np.array([0.33, 0.33, 0.34]))

        # Filter available drones (not just IDLE, but also those with capacity)
        available_drones = []
        for d in drones:
            # Include drones that have capacity, regardless of status
            # MOPSO can assign to busy drones that will serve orders later
            current_load = d.get('current_load', 0)
            max_capacity = d.get('max_capacity', 10)
            if current_load < max_capacity:
                available_drones.append(d)

        if not available_drones or not ready_orders:
            return {}

        # Limit orders to max_orders
        ready_orders = ready_orders[:self.planner.max_orders]

        # Run MOPSO to get assignment
        assignment = self.planner._run_mopso(
            ready_orders, available_drones, merchants, constraints, objective_weights
        )

        return assignment


def apply_mopso_assignment(env, assigner: Optional[U7MOPSOAssigner] = None, **kwargs) -> Dict[int, int]:
    """
    Apply MOPSO assignment to environment (assignment only, no route planning).

    This function:
    1. Gets READY orders
    2. Uses MOPSO to assign them to drones
    3. Updates order status to ASSIGNED
    4. Does NOT create planned_stops (PPO handles routing)

    Args:
        env: Environment instance
        assigner: U7MOPSOAssigner instance (created if None)
        **kwargs: Additional arguments for assigner

    Returns:
        Dict mapping drone_id to count of newly assigned orders
    """
    if assigner is None:
        assigner = U7MOPSOAssigner(**kwargs)

    # Get assignment from MOPSO
    assignment = assigner.assign_orders(env)

    # Apply assignments to environment
    assignment_counts = {}
    total_assigned = 0

    for drone_id, order_ids in assignment.items():
        if not order_ids:
            continue

        drone = env.drones.get(drone_id)
        if not drone:
            continue

        # Check capacity
        current_load = drone.get('current_load', 0)
        max_capacity = drone.get('max_capacity', 10)
        available_capacity = max_capacity - current_load

        if available_capacity <= 0:
            continue

        # Assign orders up to capacity
        assigned_count = 0
        for order_id in order_ids[:available_capacity]:
            order = env.orders.get(order_id)
            if not order:
                continue

            # Only assign READY orders - use proper enum comparison if available
            if OrderStatus is not None:
                if order['status'] != OrderStatus.READY:
                    continue
            else:
                if order['status'].name != 'READY':
                    continue

            # Check if already assigned
            if order.get('assigned_drone') not in (None, -1):
                continue

            # Assign order using environment's internal method
            # Use _process_single_assignment if available, otherwise manual assignment
            if hasattr(env, '_process_single_assignment'):
                try:
                    env._process_single_assignment(drone_id, order_id, allow_busy=True)
                    assigned_count += 1
                except Exception as e:
                    # Fallback to manual assignment if _process_single_assignment fails
                    # This can happen if the environment state is inconsistent
                    import warnings
                    warnings.warn(f"Failed to use _process_single_assignment for order {order_id}: {e}. "
                                 f"Falling back to manual assignment.")
                    # Use imported OrderStatus if available, otherwise get from env
                    assigned_status = OrderStatus.ASSIGNED if OrderStatus else env.OrderStatus.ASSIGNED
                    env.state_manager.update_order_status(order_id, assigned_status)
                    order['assigned_drone'] = drone_id
                    drone['current_load'] += 1
                    assigned_count += 1
            else:
                # Manual assignment
                assigned_status = OrderStatus.ASSIGNED if OrderStatus else env.OrderStatus.ASSIGNED
                env.state_manager.update_order_status(order_id, assigned_status)
                order['assigned_drone'] = drone_id
                drone['current_load'] += 1
                assigned_count += 1

        if assigned_count > 0:
            assignment_counts[drone_id] = assigned_count
            total_assigned += assigned_count

    return assignment_counts
