"""
MOPSO Dispatcher: Multi-Objective Particle Swarm Optimization for UAV order scheduling.

Parameters:
- M=200: Maximum candidate orders
- K=10: Maximum orders per drone
- P=30: Number of particles  
- I=10: Number of iterations

Multi-objective fitness:
- f0: Planned distance (minimize -> maximize negative)
- f1: Expected timeout orders count (minimize -> maximize negative)
- f2: Planned energy consumption (minimize -> maximize negative)
"""
import math
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from U6_mopso_utils import (
    dominates, pareto_filter, truncate_archive,
    select_leader, select_best_solution
)

# Constants
EPSILON = 1e-6  # Small value to prevent division by zero


class MOPSOPlanner:
    """
    Multi-Objective PSO for UAV delivery scheduling.

    Generates route plans for idle drones using MOPSO with:
    - Random Keys encoding for order assignment
    - Cross-merchant interleaved stop construction
    - Multi-objective fitness evaluation
    - Pareto archive maintenance
    """

    def __init__(self,
                 n_particles: int = 30,
                 n_iterations: int = 10,
                 max_orders: int = 200,
                 max_orders_per_drone: int = 10,
                 inertia: float = 0.6,
                 c1: float = 1.4,
                 c2: float = 1.4,
                 archive_size: int = 50,
                 seed: Optional[int] = None,
                 # Conservative ETA parameters
                 eta_speed_scale_assumption: float = 0.7,
                 eta_stop_service_steps: int = 1):
        """
        Initialize MOPSO planner.

        Args:
            n_particles: Number of particles (P)
            n_iterations: Number of iterations (I)
            max_orders: Maximum candidate orders (M)
            max_orders_per_drone: Maximum orders per drone (K)
            inertia: Inertia weight for PSO
            c1: Cognitive parameter
            c2: Social parameter
            archive_size: Maximum Pareto archive size
            seed: Random seed
            eta_speed_scale_assumption: Conservative speed scale for ETA (default 0.7)
            eta_stop_service_steps: Service time per stop in steps (default 1)
        """
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.max_orders = max_orders
        self.max_orders_per_drone = max_orders_per_drone
        self.inertia = inertia
        self.c1 = c1
        self.c2 = c2
        self.archive_size = archive_size
        self.rng = np.random.RandomState(seed)

        # Conservative ETA parameters for fitness calculation
        self.eta_speed_scale_assumption = float(eta_speed_scale_assumption)
        self.eta_stop_service_steps = int(eta_stop_service_steps)

        # Validate parameters
        if self.eta_speed_scale_assumption <= 0 or self.eta_speed_scale_assumption > 2.0:
            raise ValueError(f"eta_speed_scale_assumption must be in (0, 2.0], got {self.eta_speed_scale_assumption}")

        # Pareto archive
        self.archive = []

    def mopso_dispatch(self, env,
                      ready_orders: Optional[List[dict]] = None,
                      drones: Optional[List[dict]] = None,
                      merchants: Optional[Dict] = None,
                      constraints: Optional[dict] = None,
                      objective_weights: Optional[np.ndarray] = None) -> Dict[int, Tuple[List[dict], List[int]]]:
        """
        Main dispatch function using MOPSO.

        Args:
            env: Environment instance
            ready_orders: List of ready order snapshots
            drones: List of drone snapshots
            merchants: Dict of merchant snapshots
            constraints: Constraint parameters
            objective_weights: Weights for final solution selection

        Returns:
            Dict mapping drone_id to (planned_stops, commit_orders)
        """
        # Get snapshots if not provided
        if ready_orders is None:
            ready_orders = env.get_ready_orders_snapshot(limit=self.max_orders)
        if drones is None:
            drones = env.get_drones_snapshot()
        if merchants is None:
            merchants = env.get_merchants_snapshot()
        if constraints is None:
            constraints = env.get_route_plan_constraints()
        if objective_weights is None:
            objective_weights = getattr(env, 'objective_weights', np.array([0.33, 0.33, 0.34]))

        # Filter idle drones without route
        idle_drones = [d for d in drones
                      if d['status'].name == 'IDLE' and not d.get('has_route', False)]

        if not idle_drones or not ready_orders:
            return {}

        # Limit orders to max_orders
        ready_orders = ready_orders[:self.max_orders]

        # Run MOPSO
        best_assignment = self._run_mopso(ready_orders, idle_drones, merchants, constraints, objective_weights)

        # Convert assignment to route plans
        plans = self._assignment_to_route_plans(best_assignment, ready_orders, idle_drones, merchants, constraints)

        return plans

    def _run_mopso(self, orders: List[dict], drones: List[dict],
                   merchants: Dict, constraints: dict, weights: np.ndarray) -> Dict[int, List[int]]:
        """
        Run MOPSO algorithm to find best order assignment.

        Args:
            orders: List of order snapshots
            drones: List of idle drone snapshots
            merchants: Merchant data
            constraints: Environment constraints
            weights: Objective weights for final selection

        Returns:
            Assignment dict: {drone_id: [order_ids]}
        """
        n_orders = len(orders)
        n_drones = len(drones)

        if n_orders == 0 or n_drones == 0:
            return {}

        # Initialize particles
        particles = []
        for _ in range(self.n_particles):
            # Random keys encoding: each order gets a random key in [0, 1]
            x = self.rng.uniform(0, 1, size=n_orders).astype(np.float32)
            v = self.rng.normal(0, 0.1, size=n_orders).astype(np.float32)

            # Evaluate fitness
            assignment = self._decode(x, orders, drones)
            fitness = self._evaluate(assignment, orders, drones, merchants, constraints)

            particles.append({
                'x': x,
                'v': v,
                'pbest_x': x.copy(),
                'pbest_f': fitness.copy()
            })

        # Initialize archive with non-dominated solutions
        candidates = [(p['pbest_x'].copy(), p['pbest_f'].copy()) for p in particles]
        self.archive = pareto_filter(candidates)
        self.archive = truncate_archive(self.archive, self.archive_size)

        # PSO iterations
        for iteration in range(self.n_iterations):
            for p in particles:
                # Select leader from archive
                if self.archive:
                    gbest_x = select_leader(self.archive, weights, self.rng)
                else:
                    # If no archive yet, use best pbest among all particles
                    best_idx = 0
                    best_score = -np.inf
                    for i, particle in enumerate(particles):
                        score = np.dot(weights, particle['pbest_f'])
                        if score > best_score:
                            best_score = score
                            best_idx = i
                    gbest_x = particles[best_idx]['pbest_x']

                # Update velocity
                r1 = self.rng.rand(n_orders).astype(np.float32)
                r2 = self.rng.rand(n_orders).astype(np.float32)

                p['v'] = (self.inertia * p['v'] +
                         self.c1 * r1 * (p['pbest_x'] - p['x']) +
                         self.c2 * r2 * (gbest_x - p['x']))

                # Update position
                p['x'] = p['x'] + p['v']
                p['x'] = np.clip(p['x'], 0.0, 1.0)

                # Evaluate new position
                assignment = self._decode(p['x'], orders, drones)
                fitness = self._evaluate(assignment, orders, drones, merchants, constraints)

                # Update personal best
                if dominates(fitness, p['pbest_f']):
                    p['pbest_f'] = fitness.copy()
                    p['pbest_x'] = p['x'].copy()

            # Update archive
            candidates = self.archive + [(p['pbest_x'].copy(), p['pbest_f'].copy()) for p in particles]
            self.archive = pareto_filter(candidates)
            self.archive = truncate_archive(self.archive, self.archive_size)

        # Select best solution from archive
        if self.archive:
            best_x, _ = select_best_solution(self.archive, weights, normalize=True)
            best_assignment = self._decode(best_x, orders, drones)
        else:
            best_assignment = {}

        return best_assignment

    def _decode(self, x: np.ndarray, orders: List[dict], drones: List[dict]) -> Dict[int, List[int]]:
        """
        Decode particle position to order assignment using Random Keys.

        Args:
            x: Particle position (random keys for each order)
            orders: List of orders
            drones: List of drones

        Returns:
            Assignment dict: {drone_id: [order_ids]}
        """
        if len(orders) == 0 or len(drones) == 0:
            return {}

        # Sort orders by random key
        order_indices = np.argsort(x)

        # Initialize assignment
        assignment = {d['drone_id']: [] for d in drones}
        assigned_orders = set()

        # Greedy assignment based on sorted order
        for order_idx in order_indices:
            if order_idx in assigned_orders:
                continue

            order = orders[order_idx]
            order_id = order['order_id']

            # Find best drone for this order (closest with capacity)
            best_drone = None
            best_cost = float('inf')

            for drone in drones:
                drone_id = drone['drone_id']

                # Check capacity constraint
                if len(assignment[drone_id]) >= self.max_orders_per_drone:
                    continue
                if len(assignment[drone_id]) >= drone['max_capacity'] - drone['current_load']:
                    continue

                # Calculate incremental cost (distance)
                drone_loc = drone['location']
                merchant_loc = order['merchant_location']
                customer_loc = order['customer_location']

                # Simple cost: drone -> merchant + merchant -> customer
                cost = self._euclidean_distance(drone_loc, merchant_loc)
                cost += self._euclidean_distance(merchant_loc, customer_loc)

                if cost < best_cost:
                    best_cost = cost
                    best_drone = drone_id

            # Assign to best drone
            if best_drone is not None:
                assignment[best_drone].append(order_id)
                assigned_orders.add(order_idx)

        # Remove empty assignments
        assignment = {d: oids for d, oids in assignment.items() if oids}

        return assignment

    def _evaluate(self, assignment: Dict[int, List[int]],
                  orders: List[dict], drones: List[dict],
                  merchants: Dict, constraints: dict) -> np.ndarray:
        """
        Evaluate multi-objective fitness for an assignment.

        Fitness (maximize all):
        - f0: -total_distance (minimize distance)
        - f1: -timeout_count (minimize timeout orders)
        - f2: -total_energy (minimize energy)

        Args:
            assignment: {drone_id: [order_ids]}
            orders: List of orders
            drones: List of drones
            merchants: Merchant data
            constraints: Environment constraints

        Returns:
            Fitness vector [f0, f1, f2]
        """
        if not assignment:
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # Create order lookup
        order_map = {o['order_id']: o for o in orders}
        drone_map = {d['drone_id']: d for d in drones}

        total_distance = 0.0
        timeout_count = 0
        total_energy = 0.0

        current_step = constraints['current_step']
        speed_factor = constraints.get('weather_speed_factor', 1.0)

        for drone_id, order_ids in assignment.items():
            if not order_ids:
                continue

            drone = drone_map.get(drone_id)
            if drone is None:
                continue

            # Construct stops for this drone
            stops = self._construct_stops(order_ids, order_map, merchants)

            # Calculate distance for this route
            drone_loc = drone['location']
            route_distance = 0.0

            for stop in stops:
                if stop['type'] == 'P':
                    merchant_id = stop['merchant_id']
                    if merchant_id in merchants:
                        next_loc = merchants[merchant_id]['location']
                    else:
                        continue
                elif stop['type'] == 'D':
                    order_id = stop['order_id']
                    if order_id in order_map:
                        next_loc = order_map[order_id]['customer_location']
                    else:
                        continue
                else:
                    continue

                step_dist = self._euclidean_distance(drone_loc, next_loc)
                route_distance += step_dist
                drone_loc = next_loc

            total_distance += route_distance

            # Estimate timeout orders using conservative ETA
            # Apply conservative speed scale and add service time per stop
            conservative_speed = drone['speed'] * speed_factor * self.eta_speed_scale_assumption
            num_stops = len(stops)
            service_time = num_stops * self.eta_stop_service_steps
            estimated_time = (route_distance / (conservative_speed + EPSILON)) + service_time

            for order_id in order_ids:
                order = order_map.get(order_id)
                if order is None:
                    continue

                deadline = order.get('deadline_step', current_step + 1000)
                eta = current_step + estimated_time

                if eta > deadline:
                    timeout_count += 1

            # Estimate energy (proportional to distance)
            energy_rate = drone.get('battery_consumption_rate', 0.3)
            total_energy += route_distance * energy_rate

        # Convert to maximization objectives
        f0 = -total_distance  # Minimize distance
        f1 = -float(timeout_count)  # Minimize timeouts
        f2 = -total_energy  # Minimize energy

        return np.array([f0, f1, f2], dtype=np.float32)

    def _construct_stops(self, order_ids: List[int], order_map: Dict, merchants: Dict) -> List[dict]:
        """
        Construct interleaved P/D stops for a set of orders.

        Strategy:
        1. Group orders by merchant
        2. For each merchant, sort by deadline (earliest first)
        3. Insert P(merchant) followed by D(most_urgent_order_of_merchant)
        4. Append remaining D stops sorted by deadline

        Args:
            order_ids: List of order IDs to construct stops for
            order_map: Order lookup dict
            merchants: Merchant data

        Returns:
            List of stop dicts: [{'type': 'P', 'merchant_id': ...}, {'type': 'D', 'order_id': ...}, ...]
        """
        if not order_ids:
            return []

        # Group by merchant
        merchant_orders = defaultdict(list)
        for oid in order_ids:
            order = order_map.get(oid)
            if order is None:
                continue
            mid = order['merchant_id']
            merchant_orders[mid].append(order)

        # Sort merchants by earliest deadline among their orders
        merchant_list = []
        for mid, orders in merchant_orders.items():
            orders.sort(key=lambda o: (o.get('deadline_step', 9999999), o.get('distance', 0)))
            earliest_deadline = orders[0].get('deadline_step', 9999999)
            merchant_list.append((mid, orders, earliest_deadline))

        merchant_list.sort(key=lambda x: x[2])  # Sort by earliest deadline

        # Construct stops
        stops = []
        remaining_deliveries = []

        for mid, orders, _ in merchant_list:
            # Add pickup stop
            stops.append({'type': 'P', 'merchant_id': mid})

            # Add most urgent delivery from this merchant
            most_urgent = orders[0]
            stops.append({'type': 'D', 'order_id': most_urgent['order_id']})

            # Collect remaining deliveries
            for order in orders[1:]:
                remaining_deliveries.append(order)

        # Sort remaining deliveries by deadline, then by distance (nearest neighbor approximation)
        if remaining_deliveries:
            remaining_deliveries.sort(key=lambda o: (o.get('deadline_step', 9999999),
                                                     o.get('distance', 0)))
            for order in remaining_deliveries:
                stops.append({'type': 'D', 'order_id': order['order_id']})

        return stops

    def _assignment_to_route_plans(self, assignment: Dict[int, List[int]],
                                   orders: List[dict], drones: List[dict],
                                   merchants: Dict, constraints: dict) -> Dict[int, Tuple[List[dict], List[int]]]:
        """
        Convert assignment to route plans format expected by env.apply_route_plan.

        Args:
            assignment: {drone_id: [order_ids]}
            orders: List of orders
            drones: List of drones
            merchants: Merchant data
            constraints: Environment constraints

        Returns:
            Dict mapping drone_id to (planned_stops, commit_orders)
        """
        order_map = {o['order_id']: o for o in orders}

        plans = {}
        for drone_id, order_ids in assignment.items():
            if not order_ids:
                continue

            # Construct stops
            planned_stops = self._construct_stops(order_ids, order_map, merchants)

            # Commit orders are all assigned orders
            commit_orders = list(order_ids)

            plans[drone_id] = (planned_stops, commit_orders)

        return plans

    @staticmethod
    def _euclidean_distance(loc1: Tuple[float, float], loc2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two locations."""
        return math.sqrt((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2)


def apply_mopso_dispatch(env, planner: Optional[MOPSOPlanner] = None, **kwargs):
    """
    Convenience function to apply MOPSO dispatch to environment.

    Args:
        env: Environment instance
        planner: MOPSOPlanner instance (created if None)
        **kwargs: Additional arguments for planner

    Returns:
        Dict of applied plans
    """
    if planner is None:
        planner = MOPSOPlanner(**kwargs)

    plans = planner.mopso_dispatch(env)

    # Apply plans to environment (Task D: handle return value)
    applied_count = 0
    failed_count = 0
    for drone_id, (planned_stops, commit_orders) in plans.items():
        try:
            success = env.apply_route_plan(drone_id, planned_stops, commit_orders, allow_busy=False)
            if success:
                applied_count += 1
            else:
                failed_count += 1
                if hasattr(env, 'debug_state_warnings') and env.debug_state_warnings:
                    print(f"Warning: Failed to apply plan for drone {drone_id} (returned False)")
        except Exception as e:
            failed_count += 1
            print(f"Warning: Exception applying plan for drone {drone_id}: {e}")

    return plans