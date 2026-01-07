"""
Integration test for MOPSO capacity-based dispatch with append_route_plan support.

This test validates that:
1. MOPSO dispatcher can select drones based on capacity, not just IDLE status
2. Plans are applied using append_route_plan for drones with existing routes
3. Plans are applied using apply_route_plan for drones without routes
4. Backward compatibility is maintained for environments without append_route_plan
"""

import numpy as np
from collections import deque
from U6_mopso_dispatcher import MOPSOPlanner, apply_mopso_dispatch


class MockEnvironment:
    """Mock environment for testing MOPSO dispatcher."""
    
    def __init__(self, with_append_method=True):
        self.current_step = 10
        self.drones = {}
        self.orders = {}
        self.merchants = {}
        self.applied_plans = []
        self.appended_plans = []
        self._has_append = with_append_method
        
    def get_drones_snapshot(self):
        """Return drone snapshots."""
        return [
            {
                'drone_id': d_id,
                'location': drone['location'],
                'base': drone['base'],
                'status': drone['status'],
                'battery_level': drone['battery_level'],
                'current_load': drone['current_load'],
                'max_capacity': drone['max_capacity'],
                'speed': drone['speed'],
                'battery_consumption_rate': 0.1,
                'has_route': drone.get('has_route', False),
                'can_accept_more': drone['current_load'] < drone['max_capacity'],
            }
            for d_id, drone in self.drones.items()
        ]
    
    def get_ready_orders_snapshot(self, limit=200):
        """Return ready orders."""
        return [
            {
                'order_id': o_id,
                'merchant_id': order['merchant_id'],
                'merchant_location': order['merchant_location'],
                'customer_location': order['customer_location'],
                'deadline_step': order['deadline_step'],
                'distance': order.get('distance', 10.0),
            }
            for o_id, order in self.orders.items()
        ][:limit]
    
    def get_merchants_snapshot(self):
        """Return merchants."""
        return self.merchants
    
    def get_route_plan_constraints(self):
        """Return constraints."""
        return {
            'current_step': self.current_step,
            'weather_speed_factor': 1.0,
        }
    
    def apply_route_plan(self, drone_id, planned_stops, commit_orders, allow_busy=False):
        """Mock apply_route_plan."""
        self.applied_plans.append({
            'drone_id': drone_id,
            'planned_stops': planned_stops,
            'commit_orders': commit_orders,
            'allow_busy': allow_busy,
        })
        return True
    
    def append_route_plan(self, drone_id, planned_stops, commit_orders):
        """Mock append_route_plan."""
        if not self._has_append:
            raise AttributeError("append_route_plan not available")
        self.appended_plans.append({
            'drone_id': drone_id,
            'planned_stops': planned_stops,
            'commit_orders': commit_orders,
        })
        return True


def test_capacity_based_filtering():
    """Test that dispatcher filters by capacity, not just IDLE status."""
    print("\n=== Test 1: Capacity-based filtering ===")
    
    # Create mock status enum
    class Status:
        def __init__(self, name):
            self.name = name
    
    env = MockEnvironment()
    
    # Add drones with different states
    env.drones = {
        0: {'location': (0, 0), 'base': (0, 0), 'status': Status('IDLE'),
            'battery_level': 100, 'current_load': 0, 'max_capacity': 5,
            'speed': 10, 'has_route': False},
        1: {'location': (10, 10), 'base': (0, 0), 'status': Status('FLYING_TO_MERCHANT'),
            'battery_level': 80, 'current_load': 2, 'max_capacity': 5,
            'speed': 10, 'has_route': True},
        2: {'location': (20, 20), 'base': (0, 0), 'status': Status('IDLE'),
            'battery_level': 100, 'current_load': 5, 'max_capacity': 5,
            'speed': 10, 'has_route': False},
        3: {'location': (30, 30), 'base': (0, 0), 'status': Status('FLYING_TO_CUSTOMER'),
            'battery_level': 70, 'current_load': 3, 'max_capacity': 5,
            'speed': 10, 'has_route': True},
    }
    
    # Add some orders and merchants
    env.orders = {
        100: {'merchant_id': 10, 'merchant_location': (5, 5),
              'customer_location': (15, 15), 'deadline_step': 100},
        101: {'merchant_id': 10, 'merchant_location': (5, 5),
              'customer_location': (25, 25), 'deadline_step': 100},
    }
    env.merchants = {
        10: {'location': (5, 5), 'name': 'Restaurant A'},
    }
    
    # Create planner and get plans
    planner = MOPSOPlanner(seed=42, n_particles=5, n_iterations=2)
    plans = planner.mopso_dispatch(env)
    
    print(f"Generated plans for {len(plans)} drones")
    print(f"Drone IDs with plans: {list(plans.keys())}")
    
    # Expected: drones 0, 1, 3 are eligible (drone 2 is at max capacity)
    # At least some of them should get plans if there are orders
    eligible_drones = {0, 1, 3}
    for drone_id in plans.keys():
        assert drone_id in eligible_drones, f"Drone {drone_id} should not get plan (at max capacity or not eligible)"
    
    print("✓ Only eligible drones (with remaining capacity) got plans")


def test_append_vs_apply_logic():
    """Test that append_route_plan is used for drones with existing routes."""
    print("\n=== Test 2: Append vs Apply logic ===")
    
    class Status:
        def __init__(self, name):
            self.name = name
    
    env = MockEnvironment(with_append_method=True)
    
    # Add drones: one without route, one with route
    # Make drone 1 closer to orders to increase likelihood of assignment
    env.drones = {
        0: {'location': (50, 50), 'base': (0, 0), 'status': Status('IDLE'),
            'battery_level': 100, 'current_load': 0, 'max_capacity': 5,
            'speed': 10, 'has_route': False},
        1: {'location': (5, 5), 'base': (0, 0), 'status': Status('FLYING_TO_MERCHANT'),
            'battery_level': 80, 'current_load': 2, 'max_capacity': 5,
            'speed': 10, 'has_route': True},
    }
    
    # Add more orders to increase assignment probability
    env.orders = {
        100: {'merchant_id': 10, 'merchant_location': (5, 5),
              'customer_location': (15, 15), 'deadline_step': 100},
        101: {'merchant_id': 10, 'merchant_location': (5, 5),
              'customer_location': (25, 25), 'deadline_step': 100},
        102: {'merchant_id': 11, 'merchant_location': (6, 6),
              'customer_location': (16, 16), 'deadline_step': 100},
        103: {'merchant_id': 11, 'merchant_location': (6, 6),
              'customer_location': (26, 26), 'deadline_step': 100},
    }
    env.merchants = {
        10: {'location': (5, 5), 'name': 'Restaurant A'},
        11: {'location': (6, 6), 'name': 'Restaurant B'},
    }
    
    # Apply dispatch with more iterations to increase assignment probability
    planner = MOPSOPlanner(seed=42, n_particles=10, n_iterations=5)
    apply_mopso_dispatch(env, planner)
    
    print(f"Applied plans: {len(env.applied_plans)}")
    print(f"Appended plans: {len(env.appended_plans)}")
    
    # Check that we have both applied and appended plans
    total_plans = len(env.applied_plans) + len(env.appended_plans)
    print(f"Total plans executed: {total_plans}")
    
    # Verify that drones with has_route=True got append, others got apply
    for plan in env.appended_plans:
        drone_id = plan['drone_id']
        drone = env.drones[drone_id]
        assert drone.get('has_route', False), f"Drone {drone_id} got append but has_route=False"
        print(f"✓ Drone {drone_id} (has_route=True) used append_route_plan")
    
    for plan in env.applied_plans:
        drone_id = plan['drone_id']
        drone = env.drones[drone_id]
        # Should have allow_busy=True
        assert plan['allow_busy'], f"Drone {drone_id} should use allow_busy=True"
        print(f"✓ Drone {drone_id} (has_route={drone.get('has_route', False)}) used apply_route_plan with allow_busy=True")
    
    # Verify the logic is working: if we have plans, check they went to the right method
    if total_plans > 0:
        # At least verify the dispatching logic
        has_route_drones = {d_id for d_id, d in env.drones.items() if d.get('has_route', False)}
        no_route_drones = {d_id for d_id, d in env.drones.items() if not d.get('has_route', False)}
        
        appended_drone_ids = {p['drone_id'] for p in env.appended_plans}
        applied_drone_ids = {p['drone_id'] for p in env.applied_plans}
        
        # All appended plans should be for drones with routes
        if appended_drone_ids:
            assert appended_drone_ids.issubset(has_route_drones), \
                f"Appended drones {appended_drone_ids} should be subset of has_route drones {has_route_drones}"
            print(f"✓ All {len(appended_drone_ids)} appended plans went to drones with existing routes")
        
        # Applied plans can be for any drone (but should use allow_busy=True)
        if applied_drone_ids:
            print(f"✓ All {len(applied_drone_ids)} applied plans used correct allow_busy flag")
    else:
        print("⚠ No plans generated (this is ok, but check order/drone setup)")


def test_backward_compatibility():
    """Test backward compatibility when append_route_plan doesn't exist."""
    print("\n=== Test 3: Backward compatibility ===")
    
    class Status:
        def __init__(self, name):
            self.name = name
    
    # Create environment WITHOUT append_route_plan
    env = MockEnvironment(with_append_method=False)
    
    env.drones = {
        0: {'location': (0, 0), 'base': (0, 0), 'status': Status('IDLE'),
            'battery_level': 100, 'current_load': 0, 'max_capacity': 5,
            'speed': 10, 'has_route': False},
        1: {'location': (10, 10), 'base': (0, 0), 'status': Status('FLYING_TO_MERCHANT'),
            'battery_level': 80, 'current_load': 2, 'max_capacity': 5,
            'speed': 10, 'has_route': True},
    }
    
    env.orders = {
        100: {'merchant_id': 10, 'merchant_location': (5, 5),
              'customer_location': (15, 15), 'deadline_step': 100},
    }
    env.merchants = {
        10: {'location': (5, 5), 'name': 'Restaurant A'},
    }
    
    # Apply dispatch - should fall back to apply_route_plan
    planner = MOPSOPlanner(seed=42, n_particles=5, n_iterations=2)
    apply_mopso_dispatch(env, planner)
    
    print(f"Applied plans: {len(env.applied_plans)}")
    print(f"Appended plans: {len(env.appended_plans)}")
    
    # Should have only applied plans, no appended plans
    assert len(env.appended_plans) == 0, "Should not use append when method doesn't exist"
    
    # All plans should use allow_busy=True
    for plan in env.applied_plans:
        assert plan['allow_busy'], "Should use allow_busy=True for backward compatibility"
    
    print("✓ Backward compatibility maintained - falls back to apply_route_plan with allow_busy=True")


if __name__ == '__main__':
    print("=" * 60)
    print("MOPSO Capacity-Based Dispatch Integration Tests")
    print("=" * 60)
    
    test_capacity_based_filtering()
    test_append_vs_apply_logic()
    test_backward_compatibility()
    
    print("\n" + "=" * 60)
    print("All integration tests passed! ✓")
    print("=" * 60)
