import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
from enum import Enum
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import math
from sklearn.cluster import KMeans

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ===== Constants for action processing =====
SPEED_MULTIPLIER_MIN = 0.5  # Minimum speed multiplier
SPEED_MULTIPLIER_MAX = 1.5  # Maximum speed multiplier


# Speed multiplier u ∈ [-1, 1] is mapped to [0.5, 1.5] via: (u + 1) / 2 * (max - min) + min


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


# 枚举定义
class OrderStatus(Enum):
    PENDING = 0
    ACCEPTED = 1
    PREPARING = 2
    READY = 3
    ASSIGNED = 4
    PICKED_UP = 5
    DELIVERED = 6
    CANCELLED = 7


class WeatherType(Enum):
    SUNNY = 0
    RAINY = 1
    WINDY = 2
    EXTREME = 3


class OrderType(Enum):
    NORMAL = 0
    FREE_SIMPLE = 1
    FREE_COMPLEX = 2


class DroneStatus(Enum):
    IDLE = 0
    ASSIGNED = 1
    FLYING_TO_MERCHANT = 2
    WAITING_FOR_PICKUP = 3
    FLYING_TO_CUSTOMER = 4
    DELIVERING = 5
    RETURNING_TO_BASE = 6
    CHARGING = 7


# 天级时间管理系统
class DailyTimeSystem:
    """天级时间管理系统"""

    def __init__(self, start_hour=6, end_hour=22, steps_per_hour=4):
        self.start_hour = start_hour
        self.end_hour = end_hour
        self.steps_per_hour = steps_per_hour
        self.operating_hours = end_hour - start_hour
        self.steps_per_day = self.operating_hours * steps_per_hour

        # 时间状态
        self.current_step = 0
        self.current_hour = start_hour
        self.current_minute = 0
        self.day_number = 0

    def reset(self):
        """重置时间系统"""
        self.current_step = 0
        self.current_hour = self.start_hour
        self.current_minute = 0
        self.day_number = 0
        return self.get_time_state()

    def step(self):
        """前进一个时间步"""
        self.current_step += 1
        self.current_minute += 60 // self.steps_per_hour

        if self.current_minute >= 60:
            self.current_minute = 0
            self.current_hour += 1

        # 检查是否到达当天结束
        if self.current_hour == self.end_hour and self.current_minute == 0:
            self.day_number += 1
            return True  # 表示一天结束

        return False

    def get_time_state(self):
        """获取当前时间状态"""
        return {
            'step': self.current_step,
            'hour': self.current_hour,
            'minute': self.current_minute,
            'day': self.day_number,
            'progress': self.current_step / self.steps_per_day,
            'is_peak_hour': self._is_peak_hour(),
            'is_business_hours': self._is_business_hours()
        }

    def _is_peak_hour(self):
        """判断是否是高峰时段"""
        peak_hours = [(7, 9), (11, 13), (17, 19)]  # 早中晚高峰
        for start, end in peak_hours:
            if start <= self.current_hour < end:
                return True
        return False

    def _is_business_hours(self):
        """判断是否在营业时间内"""
        return self.start_hour <= self.current_hour < self.end_hour

    def get_day_progress(self):
        """获取当天进度"""
        return self.current_step / self.steps_per_day


# 统一状态管理器
class StateManager:
    """统一状态管理器"""

    def __init__(self, env):
        self.env = env
        self.state_log = []

    def update_order_status(self, order_id, new_status, reason=""):
        """统一更新订单状态"""
        if order_id not in self.env.orders:
            return False

        order = self.env.orders[order_id]
        old_status = order['status']
        order['status'] = new_status

        # Record ready_step when transitioning to READY for the first time
        if new_status == OrderStatus.READY and old_status != OrderStatus.READY:
            if order.get('ready_step') is None:
                order['ready_step'] = self.env.time_system.current_step

        # 记录状态变更
        state_change = {
            'time': self.env.time_system.current_step,
            'order_id': order_id,
            'old_status': old_status,
            'new_status': new_status,
            'reason': reason,
            'drone_id': order.get('assigned_drone')
        }
        self.state_log.append(state_change)

        return True

    def update_drone_status(self, drone_id, new_status, target_location=None):
        """统一更新无人机状态"""
        if drone_id not in self.env.drones:
            return False

        drone = self.env.drones[drone_id]
        old_status = drone['status']
        drone['status'] = new_status

        if target_location is not None:
            drone['target_location'] = target_location
        else:
            # 若显式传 None，则移除目标避免“旧目标残留”
            drone.pop('target_location', None)

        # 记录状态变更
        state_change = {
            'time': self.env.time_system.current_step,
            'drone_id': drone_id,
            'old_status': old_status,
            'new_status': new_status,
            'target_location': target_location
        }
        self.state_log.append(state_change)

        return True

    def get_state_consistency_check(self):
        """检查状态一致性 (Route-aware for Task B)"""
        issues = []

        # 检查订单与无人机状态一致性
        for order_id, order in self.env.orders.items():
            drone_id = order.get('assigned_drone')
            if drone_id is not None and drone_id >= 0:
                if drone_id not in self.env.drones:
                    issues.append(f"订单 {order_id} 分配的无人机 {drone_id} 不存在")
                    continue

                drone = self.env.drones[drone_id]
                planned_stops = drone.get('planned_stops', [])

                # Route-aware check: when drone has planned_stops, use route logic
                if planned_stops and len(planned_stops) > 0:
                    # For route-plan mode, check if order is in the route
                    order_in_route = self._order_in_planned_stops(order_id, planned_stops)
                    order_in_cargo = order_id in drone.get('cargo', set())

                    if order['status'] == OrderStatus.ASSIGNED:
                        # ASSIGNED order should be in route or about to be picked up
                        if not order_in_route:
                            issues.append(f"[Route] 订单 {order_id} 已分配但不在无人机 {drone_id} 的路线中")

                    elif order['status'] == OrderStatus.PICKED_UP:
                        # PICKED_UP order consistency checks (route-aware)
                        has_delivery_stop = self._has_delivery_stop(order_id, planned_stops)

                        if len(planned_stops) > 0 and has_delivery_stop:
                            # Drone has active route with D stop for this order
                            if not order_in_cargo:
                                # This is a real inconsistency - order should be in cargo
                                issues.append(f"[Route] 订单 {order_id} 已取货但不在无人机 {drone_id} 的货物集合中")
                        elif len(planned_stops) > 0 and order_in_cargo and not has_delivery_stop:
                            # Order in cargo but no D stop - missing delivery stop
                            issues.append(f"[Route] 订单 {order_id} 已取货但缺少对应的 D stop")
                        # If planned_stops is empty or no D stop and not in cargo,
                        # drone is likely completing/resetting - this is OK
                else:
                    # Legacy mode: original consistency check
                    if (order['status'] == OrderStatus.ASSIGNED and
                            drone['status'] not in [DroneStatus.FLYING_TO_MERCHANT, DroneStatus.WAITING_FOR_PICKUP]):
                        issues.append(f"订单 {order_id} 已分配但无人机 {drone_id} 状态不匹配: {drone['status']}")

                    elif (order['status'] == OrderStatus.PICKED_UP and
                          drone['status'] not in [DroneStatus.FLYING_TO_CUSTOMER, DroneStatus.DELIVERING]):
                        # 如果是批量订单，并且无人机正在飞往商家（取下一个订单），那么是允许的
                        if not (drone['status'] == DroneStatus.FLYING_TO_MERCHANT and
                                'batch_orders' in drone and
                                order_id in drone['batch_orders']):
                            issues.append(f"订单 {order_id} 已取货但无人机 {drone_id} 状态不匹配: {drone['status']}")

        return issues

    def _order_in_planned_stops(self, order_id, planned_stops):
        """Check if an order appears in the planned stops (D stop)"""
        for stop in planned_stops:
            if stop.get('type') == 'D' and stop.get('order_id') == order_id:
                return True
        return False

    def _has_delivery_stop(self, order_id, planned_stops):
        """Check if a delivery stop exists for the order"""
        return self._order_in_planned_stops(order_id, planned_stops)


# 路径规划可视化类
class PathVisualizer:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.path_history = defaultdict(list)
        self.planned_paths = {}

    def update_path_history(self, drone_id, location):
        """更新无人机路径历史"""
        if drone_id not in self.path_history:
            self.path_history[drone_id] = []

        # 只记录位置变化
        if (not self.path_history[drone_id] or
                self.path_history[drone_id][-1] != location):
            self.path_history[drone_id].append(location)

        # 限制历史长度
        if len(self.path_history[drone_id]) > 100:
            self.path_history[drone_id].pop(0)

    def update_planned_path(self, drone_id, current_loc, target_loc, route_preferences=None):
        """更新规划路径"""
        if route_preferences is not None:
            # 基于路径偏好生成规划路径
            planned_path = self._generate_path_from_preferences(current_loc, target_loc, route_preferences)
            self.planned_paths[drone_id] = planned_path

    def _generate_path_from_preferences(self, start, end, preferences):
        """基于路径偏好生成路径"""
        path = [start]
        current = start

        max_steps = 20  # 最大路径规划步数
        for step in range(max_steps):
            if self._distance(current, end) < 1.0:
                break

            # 计算方向向量
            dx = end[0] - current[0]
            dy = end[1] - current[1]

            # 标准化方向
            dist = max(0.1, math.sqrt(dx * dx + dy * dy))
            dx /= dist
            dy /= dist

            # 应用路径偏好
            next_x = current[0] + dx
            next_y = current[1] + dy

            # 确保在网格范围内
            next_x = max(0, min(self.grid_size - 1, next_x))
            next_y = max(0, min(self.grid_size - 1, next_y))

            # 检查路径偏好
            x_int, y_int = int(next_x), int(next_y)
            if 0 <= x_int < self.grid_size and 0 <= y_int < self.grid_size:
                # 确保 preference 是标量值
                preference_value = preferences[x_int, y_int]
                if hasattr(preference_value, '__iter__'):
                    preference_value = float(np.mean(preference_value))
                else:
                    preference_value = float(preference_value)

                # 高偏好区域更容易被选中
                if random.random() < preference_value:
                    current = (next_x, next_y)
                    path.append(current)
                else:
                    # 随机探索其他方向
                    current = (
                        current[0] + random.uniform(-0.5, 0.5),
                        current[1] + random.uniform(-0.5, 0.5)
                    )
                    current = (
                        max(0, min(self.grid_size - 1, current[0])),
                        max(0, min(self.grid_size - 1, current[1]))
                    )
                    path.append(current)

        path.append(end)  # 确保路径到达目标
        return path

    def _distance(self, loc1, loc2):
        """计算两点距离"""
        return math.sqrt((loc1[0] - loc2[0]) ** 2 + (loc1[1] - loc2[1]) ** 2)

    def clear_paths(self):
        """清除所有路径"""
        self.path_history.clear()
        self.planned_paths.clear()


# 位置数据加载器类
class LocationDataLoader:
    def __init__(self, merchant_csv_path, user_csv_path, grid_size=10):
        self.grid_size = grid_size
        self.merchant_locations = self._load_merchant_locations(merchant_csv_path)
        self.user_locations = self._load_user_locations(user_csv_path)

        # 计算经纬度范围用于归一化到网格坐标
        self._calculate_coordinate_range()
        # 计算基站位置
        self.base_locations = []

    def _load_merchant_locations(self, csv_path):
        """加载商家位置数据"""
        try:
            df = pd.read_csv(csv_path)
            locations = []
            for _, row in df.iterrows():
                location_str = str(row['location'])
                if ',' in location_str:
                    lon, lat = map(float, location_str.split(','))
                    locations.append({
                        'id': row['id'],
                        'name': row['name'],
                        'business_type': row['business_type'],
                        'longitude': lon,
                        'latitude': lat,
                        'address': row.get('address', ''),
                        'rating': row.get('rating', 4.0),
                        'cost': row.get('cost', 30.0)
                    })
            return locations
        except Exception as e:
            print(f"加载商家位置数据失败: {e}")
            return self._create_fallback_merchant_locations()

    def _load_user_locations(self, csv_path):
        """加载用户位置数据"""
        try:
            df = pd.read_csv(csv_path)
            locations = []
            for _, row in df.iterrows():
                locations.append({
                    'user_id': row['user_id'],
                    'latitude': float(row['latitude']),
                    'longitude': float(row['longitude']),
                    'type': row.get('type', 'user')
                })
            return locations
        except Exception as e:
            print(f"加载用户位置数据失败: {e}")
            return self._create_fallback_user_locations()

    def _calculate_coordinate_range(self):
        """计算经纬度范围用于坐标归一化"""
        all_lats = []
        all_lons = []

        # 收集所有商家和用户的经纬度
        for merchant in self.merchant_locations:
            all_lats.append(merchant['latitude'])
            all_lons.append(merchant['longitude'])

        for user in self.user_locations:
            all_lats.append(user['latitude'])
            all_lons.append(user['longitude'])

        if all_lats and all_lons:
            self.min_lat = min(all_lats)
            self.max_lat = max(all_lats)
            self.min_lon = min(all_lons)
            self.max_lon = max(all_lons)

            # 计算缩放比例，保持纵横比
            lat_range = self.max_lat - self.min_lat
            lon_range = self.max_lon - self.min_lon

            # 使用较大的范围作为基准
            self.range_lat = lat_range if lat_range > 0 else 0.01
            self.range_lon = lon_range if lon_range > 0 else 0.01
        else:
            # 备用范围
            self.min_lat = 25.81
            self.max_lat = 25.82
            self.min_lon = 114.92
            self.max_lon = 114.93
            self.range_lat = 0.01
            self.range_lon = 0.01

    def _create_fallback_merchant_locations(self):
        """创建备用商家位置数据"""
        print("使用备用商家位置数据")
        locations = []
        for i in range(5):
            locations.append({
                'id': f'B{i}',
                'name': f'商家{i}',
                'business_type': '餐饮',
                'longitude': 114.92 + random.uniform(-0.005, 0.005),
                'latitude': 25.815 + random.uniform(-0.005, 0.005),
                'address': f'地址{i}',
                'rating': random.uniform(3.5, 5.0),
                'cost': random.uniform(20, 50)
            })
        return locations

    def _create_fallback_user_locations(self):
        """创建备用用户位置数据"""
        print("使用备用用户位置数据")
        locations = []
        for i in range(50):
            locations.append({
                'user_id': f'user_{i:04d}',
                'latitude': 25.815 + random.uniform(-0.01, 0.01),
                'longitude': 114.92 + random.uniform(-0.01, 0.01),
                'type': 'user'
            })
        return locations

    def convert_to_grid_coordinates(self, longitude, latitude):
        """将经纬度坐标转换为网格坐标"""
        # 归一化到 [0, 1] 范围
        norm_x = (longitude - self.min_lon) / self.range_lon
        norm_y = (latitude - self.min_lat) / self.range_lat

        # 缩放到网格大小
        grid_x = norm_x * (self.grid_size - 1)
        grid_y = norm_y * (self.grid_size - 1)

        return (grid_x, grid_y)

    def get_merchant_grid_locations(self):
        """获取商家在网格中的位置"""
        grid_locations = []
        for merchant in self.merchant_locations:
            grid_loc = self.convert_to_grid_coordinates(
                merchant['longitude'],
                merchant['latitude']
            )
            grid_locations.append({
                'id': merchant['id'],
                'name': merchant['name'],
                'grid_location': grid_loc,
                'original_location': (merchant['longitude'], merchant['latitude']),
                'business_type': merchant['business_type'],
                'rating': merchant['rating'],
                'cost': merchant['cost']
            })
        return grid_locations

    def get_random_user_grid_location(self):
        """随机获取一个用户在网格中的位置"""
        if not self.user_locations:
            return self._generate_random_grid_location()

        user = random.choice(self.user_locations)
        grid_loc = self.convert_to_grid_coordinates(
            user['longitude'],
            user['latitude']
        )
        return grid_loc

    def _generate_random_grid_location(self):
        """生成随机网格位置（备用）"""
        return (
            random.uniform(0, self.grid_size - 1),
            random.uniform(0, self.grid_size - 1)
        )

    def find_optimal_base_locations(self, num_bases, method='kmeans'):
        """使用K-means、密度或随机寻找基站位置"""
        if method == 'kmeans':
            return self._kmeans_base_locations(num_bases)
        elif method == 'centroid':
            return self._centroid_base_locations(num_bases)
        else:
            return self._random_base_locations(num_bases)

    def _kmeans_base_locations(self, num_bases):
        """使用K-means聚类确定基站位置"""
        # 收集所有商家和用户的位置
        all_locations = []

        # 添加商家位置
        for merchant in self.merchant_locations:
            grid_loc = self.convert_to_grid_coordinates(
                merchant['longitude'], merchant['latitude']
            )
            all_locations.append(grid_loc)

        # 添加用户位置（采样一部分避免过多）- 修复：按索引采样
        user_sample_size = min(100, len(self.user_locations))
        if len(self.user_locations) > 0 and user_sample_size > 0:
            idx = np.random.choice(np.arange(len(self.user_locations)), size=user_sample_size, replace=False)
            sampled_users = [self.user_locations[int(i)] for i in idx]
        else:
            sampled_users = []

        for user in sampled_users:
            grid_loc = self.convert_to_grid_coordinates(
                user['longitude'], user['latitude']
            )
            all_locations.append(grid_loc)

        if len(all_locations) < num_bases:
            # 如果位置数量不足，补充随机位置
            while len(all_locations) < num_bases:
                all_locations.append((
                    random.uniform(0, self.grid_size - 1),
                    random.uniform(0, self.grid_size - 1)
                ))

        # 转换为numpy数组
        locations_array = np.array(all_locations)

        # 使用K-means聚类
        kmeans = KMeans(n_clusters=num_bases, random_state=42, n_init=10)
        kmeans.fit(locations_array)

        # 获取聚类中心作为基站位置
        base_locations = kmeans.cluster_centers_

        # 确保在网格范围内
        base_locations = np.clip(base_locations, 0, self.grid_size - 1)

        print(f"K-means找到 {len(base_locations)} 个基站位置")
        return base_locations.tolist()

    def _centroid_base_locations(self, num_bases):
        """基于密度确定基站位置"""
        # 创建网格密度图
        density_map = np.zeros((self.grid_size, self.grid_size))

        # 统计每个网格点的位置密度
        for merchant in self.merchant_locations:
            grid_loc = self.convert_to_grid_coordinates(
                merchant['longitude'], merchant['latitude']
            )
            x, y = int(grid_loc[0]), int(grid_loc[1])
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                density_map[x, y] += 2  # 商家权重更高

        # 修复：按索引采样用户
        user_sample_size = min(200, len(self.user_locations))
        if len(self.user_locations) > 0 and user_sample_size > 0:
            idx = np.random.choice(np.arange(len(self.user_locations)), size=user_sample_size, replace=False)
            sampled_users = [self.user_locations[int(i)] for i in idx]
        else:
            sampled_users = []

        for user in sampled_users:
            grid_loc = self.convert_to_grid_coordinates(
                user['longitude'], user['latitude']
            )
            x, y = int(grid_loc[0]), int(grid_loc[1])
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                density_map[x, y] += 1

        # 找到密度最高的num_bases个位置
        base_locations = []
        for _ in range(num_bases):
            max_idx = np.unravel_index(np.argmax(density_map), density_map.shape)
            base_locations.append((max_idx[0] + 0.5, max_idx[1] + 0.5))  # 使用网格中心

            # 将周围区域密度降低，避免基站太近
            x, y = max_idx
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                        distance = np.sqrt(dx ** 2 + dy ** 2)
                        density_map[nx, ny] *= max(0, 1 - distance / 5)

        print(f"基于密度找到 {len(base_locations)} 个基站位置")
        return base_locations

    def _random_base_locations(self, num_bases):
        """随机生成基站位置（备用方法）"""
        base_locations = []
        for i in range(num_bases):
            base_locations.append((
                random.uniform(0, self.grid_size - 1),
                random.uniform(0, self.grid_size - 1)
            ))
        print(f"随机生成 {len(base_locations)} 个基站位置")
        return base_locations


# 帕累托优化器
class ParetoOptimizer:
    def __init__(self, num_objectives=3):
        self.num_objectives = num_objectives
        self.pareto_front = []
        self.solutions_history = []

    def update_pareto_front(self, solution):
        """更新帕累托前沿"""
        solution = np.array(solution)
        is_dominated = False

        # 创建前沿的副本用于迭代
        front_to_check = self.pareto_front.copy()

        # 检查是否被现有前沿中的解支配
        for front_solution in front_to_check:
            if self._dominates(front_solution, solution):
                is_dominated = True
                break

        # 移除被新解支配的旧解
        new_front = []
        for front_solution in self.pareto_front:
            if not self._dominates(solution, front_solution):
                new_front.append(front_solution)

        self.pareto_front = new_front

        # 如果未被支配，加入帕累托前沿
        if not is_dominated:
            self.pareto_front.append(solution)

        self.solutions_history.append(solution.copy())

    def _dominates(self, sol1, sol2):
        """检查sol1是否支配sol2"""
        # 对于最大化问题，sol1在所有目标上都不比sol2差，且至少在一个目标上严格更好
        all_better_or_equal = np.all(sol1 >= sol2)
        at_least_one_better = np.any(sol1 > sol2)
        return all_better_or_equal and at_least_one_better

    def get_pareto_front(self):
        """获取帕累托前沿"""
        return np.array(self.pareto_front)

    def calculate_hypervolume(self, reference_point):
        """计算超体积指标"""
        if len(self.pareto_front) == 0:
            return 0.0

        front = np.array(self.pareto_front)
        # 简化的超体积计算
        return np.sum([np.prod(solution) for solution in front])

    def get_diversity(self):
        """计算帕累托前沿的多样性"""
        if len(self.pareto_front) < 2:
            return 0.0

        front = np.array(self.pareto_front)
        distances = []
        for i in range(len(front)):
            for j in range(i + 1, len(front)):
                distances.append(np.linalg.norm(front[i] - front[j]))

        return np.mean(distances) if distances else 0.0


# 天气数据处理器
class WeatherDataProcessor:
    def __init__(self, csv_path):
        self.weather_df = self._load_weather_data(csv_path)
        self.data_length = len(self.weather_df)

    def _load_weather_data(self, csv_path):
        """加载天气数据"""
        try:
            df = pd.read_csv(csv_path)
            df['Formatted Date'] = pd.to_datetime(df['Formatted Date'], errors='coerce', utc=True)
            df = df.dropna(subset=['Formatted Date', 'Summary', 'Temperature (C)'])
            return df
        except Exception as e:
            print(f"加载天气数据失败: {e}")
            return self._create_fallback_weather_data()

    def _create_fallback_weather_data(self):
        """创建备用天气数据"""
        print("使用备用天气数据")
        dates = pd.date_range(start='2006-01-01', end='2016-12-31', freq='H')
        data = {
            'Formatted Date': dates,
            'Summary': np.random.choice(['Clear', 'Partly Cloudy', 'Cloudy', 'Rain', 'Windy', 'Fog'], len(dates)),
            'Temperature (C)': np.random.normal(15, 10, len(dates)),
            'Humidity': np.random.uniform(0.3, 0.9, len(dates)),
            'Wind Speed (km/h)': np.random.exponential(10, len(dates)),
            'Visibility (km)': np.random.uniform(5, 20, len(dates)),
            'Pressure (millibars)': np.random.normal(1013, 10, len(dates)),
            'Precip Type': np.random.choice(['rain', 'snow', 'none'], len(dates), p=[0.2, 0.05, 0.75])
        }
        return pd.DataFrame(data)

    def get_weather_at_time(self, env_time):
        """根据环境时间获取天气数据"""
        index = int(env_time) % max(1, self.data_length)
        return self.weather_df.iloc[index]

    def map_to_weather_type(self, weather_summary):
        """将天气摘要映射到WeatherType枚举"""
        summary_lower = str(weather_summary).lower()
        extreme_keywords = ['storm', 'thunderstorm', 'heavy', 'blizzard', 'hurricane', 'tornado']
        if any(keyword in summary_lower for keyword in extreme_keywords):
            return WeatherType.EXTREME
        rain_keywords = ['rain', 'drizzle', 'shower', 'precip', 'wet']
        if any(keyword in summary_lower for keyword in rain_keywords):
            return WeatherType.RAINY
        wind_keywords = ['wind', 'breezy', 'gust']
        if any(keyword in summary_lower for keyword in wind_keywords):
            return WeatherType.WINDY
        sunny_keywords = ['clear', 'sunny', 'fair']
        if any(keyword in summary_lower for keyword in sunny_keywords):
            return WeatherType.SUNNY
        return WeatherType.SUNNY


# 订单数据处理器
class OrderDataProcessor:
    def __init__(self, excel_path, grid_size=10, merchant_ids=None, time_system: Optional[DailyTimeSystem] = None):
        self.grid_size = grid_size
        self.merchant_ids = merchant_ids if merchant_ids else []
        self.num_merchants = len(merchant_ids) if merchant_ids else 5
        self.order_df = self._load_order_data(excel_path)
        self.order_patterns = self._analyze_order_patterns()
        self.time_system = time_system  # 注入时间系统，供准备时间计算
        print(f"订单模式分析完成，商家数量: {self.num_merchants}")

    def _load_order_data(self, excel_path):
        """加载订单数据"""
        try:
            df = pd.read_excel(excel_path)
            required_columns = ['order_time', 'merchant_id', 'order_type', 'preparation_time', 'distance']
            for col in required_columns:
                if col not in df.columns:
                    if col == 'order_time':
                        df[col] = pd.date_range(start='2023-01-01', periods=len(df), freq='h')
                    elif col == 'merchant_id':
                        if self.merchant_ids:
                            df[col] = np.random.choice(self.merchant_ids, len(df))
                        else:
                            df[col] = np.random.randint(0, self.num_merchants, len(df))
                    elif col == 'order_type':
                        df[col] = np.random.choice([0, 1, 2], len(df), p=[0.8, 0.15, 0.05])
                    elif col == 'preparation_time':
                        df[col] = np.random.randint(3, 10, len(df))
                    elif col == 'distance':
                        df[col] = np.random.exponential(3, len(df))
            return df
        except Exception as e:
            print(f"加载订单数据失败: {e}")
            return self._create_fallback_order_data()

    def _create_fallback_order_data(self):
        """创建备用订单数据"""
        print("使用备用订单数据")
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='H')
        data = {
            'order_time': dates,
            'merchant_id': np.random.choice(self.merchant_ids, len(dates)) if self.merchant_ids else np.random.randint(
                0, self.num_merchants, len(dates)),
            'order_type': np.random.choice([0, 1, 2], len(dates), p=[0.8, 0.15, 0.05]),
            'preparation_time': np.random.randint(3, 10, len(dates)),
            'distance': np.random.exponential(3, len(dates)),
            'is_peak': np.random.choice([0, 1], len(dates), p=[0.7, 0.3])
        }
        return pd.DataFrame(data)

    def _analyze_order_patterns(self):
        """分析订单模式"""
        if self.order_df is None or len(self.order_df) == 0:
            return self._create_default_patterns()
        hourly_pattern = self._analyze_hourly_pattern()
        merchant_distribution = self._analyze_merchant_distribution()
        order_type_distribution = self._analyze_order_type_distribution()
        return {
            'hourly_pattern': hourly_pattern,
            'merchant_distribution': merchant_distribution,
            'order_type_distribution': order_type_distribution,
            'peak_hours': [11, 12, 17, 18, 19],
            'weekend_boost': 1.3
        }

    def _analyze_hourly_pattern(self):
        """分析小时订单模式"""
        if 'order_time' not in self.order_df.columns:
            return self._create_default_hourly_pattern()
        self.order_df['hour'] = self.order_df['order_time'].dt.hour
        hourly_counts = self.order_df['hour'].value_counts().sort_index()
        pattern = np.zeros(24)
        for hour, count in hourly_counts.items():
            pattern[hour] = count
        if pattern.sum() > 0:
            pattern = pattern / pattern.max()
        return pattern

    def _analyze_merchant_distribution(self):
        """分析商家分布"""
        if 'merchant_id' not in self.order_df.columns:
            return np.ones(self.num_merchants) / self.num_merchants

        merchant_counts = self.order_df['merchant_id'].value_counts()
        distribution = np.zeros(self.num_merchants)

        if self.merchant_ids:
            for i, merchant_id in enumerate(self.merchant_ids):
                if merchant_id in merchant_counts.index:
                    distribution[i] = merchant_counts[merchant_id]
        else:
            for merchant_id, count in merchant_counts.items():
                if 0 <= merchant_id < self.num_merchants:
                    distribution[merchant_id] = count

        if distribution.sum() > 0:
            distribution = distribution / distribution.sum()
        return distribution

    def _analyze_order_type_distribution(self):
        """分析订单类型分布"""
        if 'order_type' not in self.order_df.columns:
            return [0.8, 0.15, 0.05]
        type_counts = self.order_df['order_type'].value_counts()
        distribution = np.zeros(3)
        for order_type, count in type_counts.items():
            if 0 <= order_type <= 2:
                distribution[order_type] = count
        if distribution.sum() > 0:
            distribution = distribution / distribution.sum()
        return distribution

    def _create_default_patterns(self):
        """创建默认模式"""
        return {
            'hourly_pattern': self._create_default_hourly_pattern(),
            'merchant_distribution': np.ones(self.num_merchants) / self.num_merchants,
            'order_type_distribution': [0.8, 0.15, 0.05],
            'peak_hours': [11, 12, 17, 18, 19],
            'weekend_boost': 1.3
        }

    def _create_default_hourly_pattern(self):
        """创建默认小时模式"""
        pattern = np.array([0.1, 0.05, 0.02, 0.01, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3,
                            0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2])
        return pattern / pattern.max()

    def get_order_probability(self, env_time, weather_type):
        """获取订单生成概率 - 高负载版本（移除负载抑制）"""
        hour = env_time % 24
        base_prob = self.order_patterns['hourly_pattern'][hour]
        base_prob = min(0.95, base_prob * 3.0)

        weather_impact = self._get_weather_impact(weather_type)
        time_impact = 2.5 if hour in self.order_patterns['peak_hours'] else 1.0
        day_of_week = (env_time // 24) % 7
        weekend_impact = self.order_patterns['weekend_boost'] if day_of_week >= 5 else 1.0

        final_prob = base_prob * weather_impact * time_impact * weekend_impact
        final_prob = min(0.95, max(0.2, final_prob))
        return final_prob

    def _get_weather_impact(self, weather_type):
        """调整天气影响 - 降低负面影响"""
        if weather_type == WeatherType.SUNNY:
            return 1.5
        elif weather_type == WeatherType.RAINY:
            return 1.2
        elif weather_type == WeatherType.WINDY:
            return 0.9
        elif weather_type == WeatherType.EXTREME:
            return 0.6
        else:
            return 1.0

    def generate_order_details(self, env_time, weather_type):
        """生成订单详细信息"""
        if self.merchant_ids:
            merchant_id = np.random.choice(self.merchant_ids, p=self.order_patterns['merchant_distribution'])
        else:
            merchant_id = np.random.choice(range(self.num_merchants), p=self.order_patterns['merchant_distribution'])

        order_type = np.random.choice([0, 1, 2], p=self.order_patterns['order_type_distribution'])
        self.current_order_type = order_type

        if weather_type in [WeatherType.RAINY, WeatherType.EXTREME]:
            order_type_weights = [0.6, 0.2, 0.2]
            order_type = np.random.choice([0, 1, 2], p=order_type_weights)

        preparation_time = self._generate_preparation_time(weather_type)
        urgency = self._generate_urgency(env_time % 24)

        if weather_type in [WeatherType.RAINY, WeatherType.EXTREME]:
            max_distance = self.grid_size // 2
        else:
            max_distance = self.grid_size

        return {
            'merchant_id': merchant_id,
            'order_type': order_type,
            'preparation_time': preparation_time,
            'urgency': urgency,
            'max_distance': max_distance
        }

    def _generate_preparation_time(self, weather_type):
        """生成更符合实际的准备时间（返回 step）"""
        base_time_minutes = 0

        if hasattr(self, 'current_order_type'):
            if self.current_order_type == 0:
                base_time_minutes = random.randint(1, 5)
            elif self.current_order_type == 1:
                base_time_minutes = random.randint(2, 5)
            else:
                base_time_minutes = random.randint(10, 20)
        else:
            base_time_minutes = random.randint(5, 15)

        if weather_type == WeatherType.EXTREME:
            base_time_minutes += random.randint(3, 8)
        elif weather_type == WeatherType.RAINY:
            base_time_minutes += random.randint(1, 3)

        if hasattr(self, 'time_system') and self.time_system is not None:
            minutes_per_step = 60 // self.time_system.steps_per_hour
        else:
            minutes_per_step = 15

        preparation_steps = max(1, int(math.ceil(base_time_minutes / max(minutes_per_step, 1))))
        return preparation_steps

    def _generate_urgency(self, hour):
        """生成紧急程度"""
        return 0.3 if hour in [11, 12, 17, 18, 19] else 0.1


# 三目标帕累托无人机配送环境 - 天级版本 - 高负载无限制
class ThreeObjectiveDroneDeliveryEnv(gym.Env):
    """三目标帕累托无人机餐食配送环境 - 高负载无限制版本"""

    metadata = {'render_modes': ['human', 'rgb_array', 'matplotlib']}

    def __init__(self,
                 grid_size=16,
                 num_drones=6,
                 max_orders=100,
                 steps_per_hour=4,
                 weather_csv_path='weather_dataset.csv',
                 order_excel_path='food_delivery_data.xlsx',
                 merchant_location_path='food_business_zhanggong_locations.csv',
                 user_location_path='user_zhanggong_locations.csv',
                 base_placement_method='kmeans',
                 drone_max_capacity=10,
                 operating_hours=(6, 22),
                 high_load_factor=1.5,
                 distance_reward_weight=1.0,
                 multi_objective_mode: str = "conditioned",
                 fixed_objective_weights=(0.5, 0.3, 0.2),
                 shaping_progress_k: float = 0.3,
                 shaping_distance_k: float = 0.05,
                 shaping_energy_k: float = 0.01,
                 heading_guidance_alpha: float = 0.5,
                 # ===== 新增：固定 num_bases 与 Top-K 商家观测 =====
                 num_bases: Optional[int] = None,
                 top_k_merchants: int = 100,
                 reward_output_mode: str = "zero",
                 enable_random_events: bool = True,  # 可选：评估时建议关掉随机事件
                 debug_state_warnings: bool = False,  # Task B: control state consistency warning output
                 delivery_sla_steps: int = 60,  # READY-based delivery SLA in steps
                 timeout_factor: float = 1.0,  # Multiplier for deadline calculation
                 ):
        super().__init__()

        # ====== 固定基础参数（init 一次性确定）======

        self.grid_size = int(grid_size)
        self.num_drones = int(num_drones)
        self.max_obs_orders = int(max_orders)
        self.high_load_factor = float(high_load_factor)
        self.num_objectives = 3
        self.drone_max_capacity = int(drone_max_capacity)

        self.distance_reward_weight = float(distance_reward_weight)
        self.pending_distance_reward = 0.0

        self.top_k_merchants = int(top_k_merchants)

        # ========== 多目标训练方式 ==========
        self.multi_objective_mode = multi_objective_mode
        self.fixed_objective_weights = np.array(fixed_objective_weights, dtype=np.float32)
        self.fixed_objective_weights = self.fixed_objective_weights / (self.fixed_objective_weights.sum() + 1e-8)
        self.objective_weights = self.fixed_objective_weights.copy()
        self.reward_output_mode = str(reward_output_mode)
        self.enable_random_events = bool(enable_random_events)
        self.debug_state_warnings = bool(debug_state_warnings)  # Task B: debug flag
        self.delivery_sla_steps = int(delivery_sla_steps)  # READY-based delivery SLA
        self.timeout_factor = float(timeout_factor)  # Deadline multiplier
        self.episode_r_vec = np.zeros(self.num_objectives, dtype=np.float32)
        # ========== shaping 参数 ==========
        self.shaping_progress_k = float(shaping_progress_k)
        self.shaping_distance_k = float(shaping_distance_k)
        self.shaping_energy_k = float(shaping_energy_k)
        self.heading_guidance_alpha = float(np.clip(heading_guidance_alpha, 0.0, 1.0))
        self._prev_target_dist = np.zeros(self.num_drones, dtype=np.float32)

        # ====== 时间系统（先建立，后续多个组件要用）======
        start_hour, end_hour = operating_hours
        self.time_system = DailyTimeSystem(
            start_hour=start_hour,
            end_hour=end_hour,
            steps_per_hour=steps_per_hour
        )

        # ======== 1) 先加载位置与商家列表（全量商家数）========
        self.location_loader = LocationDataLoader(
            merchant_location_path, user_location_path, self.grid_size
        )
        merchant_grid_locations = self.location_loader.get_merchant_grid_locations()
        self.total_merchants = len(merchant_grid_locations)
        self.merchant_grid_data = merchant_grid_locations
        self.merchant_ids = [m['id'] for m in merchant_grid_locations]

        # ======== 2) 固定 num_bases（init 只算一次）========
        if num_bases is None:
            self.num_bases = max(2, self.num_drones // 4)
        else:
            self.num_bases = max(1, int(num_bases))

        # ======== 3) 数据处理器（依赖 time_system / merchant_ids）========
        self.weather_processor = WeatherDataProcessor(weather_csv_path)
        self.order_processor = OrderDataProcessor(
            order_excel_path, self.grid_size, self.merchant_ids, time_system=self.time_system
        )

        # 优化器/可视化
        self.pareto_optimizer = ParetoOptimizer(self.num_objectives)
        self.path_visualizer = PathVisualizer(self.grid_size)

        # ======== 4) 初始化实体：locations/bases/merchants/drones/orders ========
        self._init_locations_fixed_bases(base_placement_method)
        self._init_bases()
        self._init_merchants()
        self._init_drones()
        self._init_orders()

        # ======== 5) 固定观测维度（obs_num_*）========
        self.obs_num_bases = int(self.num_bases)
        self.obs_num_merchants = int(min(self.top_k_merchants, len(self.merchants)))

        # 状态管理器（放在 spaces 前后都行；这里放前面便于后续 reset/生成时用）
        self.state_manager = StateManager(self)

        # 定义 spaces：只使用固定 shape
        self._define_spaces()

        # 每日统计
        self.daily_stats = {
            'day_number': 0,
            'orders_generated': 0,
            'orders_completed': 0,
            'orders_cancelled': 0,
            'revenue': 0,
            'costs': 0,
            'energy_consumed': 0,
            'on_time_deliveries': 0,
            'total_flight_distance': 0.0,
            'optimal_flight_distance': 0.0,
        }

        # 性能指标
        self.metrics = {
            'total_orders': 0,
            'completed_orders': 0,
            'cancelled_orders': 0,
            'total_delivery_time': 0,
            'total_revenue': 0,
            'total_cost': 0,
            'energy_consumed': 0,
            'collisions_avoided': 0,
            'on_time_deliveries': 0,
            'total_flight_distance': 0.0,
            'optimal_flight_distance': 0.0,
            'weather_impact_stats': {
                'sunny_deliveries': 0,
                'rainy_deliveries': 0,
                'windy_deliveries': 0,
                'extreme_deliveries': 0
            }
        }

        # 事件/历史
        self.event_queue = deque()
        self.order_history = []
        self.weather_history = []
        self.pareto_history = []
        self.air_traffic = np.zeros((self.grid_size, self.grid_size))
        self.weather = WeatherType.SUNNY
        self.weather_details = {}

        # 营业后履约
        self.allow_overtime_fulfillment = True
        self.overtime_max_steps = self.time_system.steps_per_hour * 4
        self.in_overtime = False
        self.overtime_steps = 0

        self._assigned_in_step = set()

        # 归一化上限
        self.max_queue_cap: float = 50.0
        self.max_eff_cap: float = 2.0

        # 订单ID计数器
        self.global_order_counter = 0
        self.current_obs_order_ids = []

    # ------------------ 初始化相关 ------------------

    def _init_locations_fixed_bases(self, base_method: str = "kmeans"):
        """固定 num_bases：只生成 num_bases 个 base 坐标；不再在这里修改 self.num_bases"""
        self.locations = {}

        base_locations = self.location_loader.find_optimal_base_locations(
            self.num_bases, method=base_method
        )

        for i, base_loc in enumerate(base_locations):
            self.locations[f'base_{i}'] = (float(base_loc[0]), float(base_loc[1]))

        for merchant_data in self.merchant_grid_data:
            merchant_id = merchant_data['id']
            self.locations[f'merchant_{merchant_id}'] = merchant_data['grid_location']

    def _init_bases(self):
        """初始化无人机基站"""
        self.bases = {}
        for i in range(self.num_bases):
            base_location = self.locations.get(f'base_{i}')
            if base_location is None:
                base_location = (
                    random.uniform(0, self.grid_size - 1),
                    random.uniform(0, self.grid_size - 1)
                )
                self.locations[f'base_{i}'] = base_location

            self.bases[i] = {
                'location': base_location,
                'capacity': max(1, self.num_drones // self.num_bases),
                'drones_available': [],
                'charging_stations': 5,
                'charging_queue': deque(),
                'coverage_radius': self.grid_size / 3
            }

    def _init_merchants(self):
        """初始化商家"""
        self.merchants = {}

        for merchant_data in self.merchant_grid_data:
            merchant_id = merchant_data['id']

            business_type = merchant_data['business_type']
            if '饮料' in business_type or '奶茶' in business_type or '咖啡' in business_type:
                preparation_efficiency = random.uniform(1.5, 2.0)
                base_prep_time = random.randint(2, 4)
            elif '快餐' in business_type or '小吃' in business_type:
                preparation_efficiency = random.uniform(1.2, 1.5)
                base_prep_time = random.randint(5, 8)
            else:
                preparation_efficiency = random.uniform(1.0, 1.3)
                base_prep_time = random.randint(8, 12)

            self.merchants[merchant_id] = {
                'location': merchant_data['grid_location'],
                'original_location': merchant_data['original_location'],
                'name': merchant_data['name'],
                'business_type': merchant_data['business_type'],
                'rating': merchant_data['rating'],
                'avg_cost': merchant_data['cost'],
                'queue': deque(),
                'base_preparation_time': base_prep_time,
                'efficiency': preparation_efficiency,
                'order_count': 0,
                'cancellation_rate': random.uniform(0.01, 0.03),
                'landing_zone': True
            }

    def _init_drones(self):
        """初始化无人机"""
        self.drones = {}
        for i in range(self.num_drones):
            base_id = i % self.num_bases
            base_location = self.bases[base_id]['location']

            self.drones[i] = {
                'location': base_location,
                'base': base_id,
                'status': DroneStatus.IDLE,
                'current_order': None,
                'orders_completed': 0,
                'speed': random.uniform(3.0, 5.0),
                'reliability': random.uniform(0.95, 0.99),
                'max_capacity': self.drone_max_capacity,
                'current_load': 0,
                'battery_level': 100.0,
                'max_battery': 100.0,
                'battery_consumption_rate': random.uniform(0.2, 0.4),
                'charging_rate': random.uniform(10.0, 15.0),
                'cancellation_rate': random.uniform(0.005, 0.015),
                'total_distance_today': 0.0,
                'planned_stops': deque(),
                # deque of stops: {'type':'P','merchant_id':mid} or {'type':'D','order_id':oid}
                'cargo': set(),  # picked-up orders (order_ids) not yet delivered
                'current_stop': None,
                'route_committed': False,
            }
            self.bases[base_id]['drones_available'].append(i)

    def _init_orders(self):
        """初始化订单池 - 支持无限订单"""
        self.orders = {}
        self.active_orders = set()
        self.completed_orders = set()
        self.cancelled_orders = set()
        self.global_order_counter = 0

    def _define_spaces(self):
        """定义观察和动作空间（固定 shape）"""
        self.observation_space = spaces.Dict({
            'orders': spaces.Box(low=0, high=1, shape=(self.max_obs_orders, 10), dtype=np.float32),
            'drones': spaces.Box(low=0, high=1, shape=(self.num_drones, 8), dtype=np.float32),

            # Top-K merchants
            'merchants': spaces.Box(low=0, high=1, shape=(self.obs_num_merchants, 4), dtype=np.float32),

            # 固定 num_bases
            'bases': spaces.Box(low=0, high=1, shape=(self.obs_num_bases, 3), dtype=np.float32),

            'weather': spaces.Discrete(len(WeatherType)),
            'weather_details': spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32),
            'time': spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32),
            'day_progress': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            'resource_saturation': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            'air_traffic': spaces.Box(low=0, high=1, shape=(self.grid_size, self.grid_size), dtype=np.float32),
            'order_pattern': spaces.Box(low=0, high=1, shape=(24,), dtype=np.float32),
            'pareto_info': spaces.Box(low=0, high=1, shape=(self.num_objectives + 2,), dtype=np.float32),
            'objective_weights': spaces.Box(low=0, high=1, shape=(self.num_objectives,), dtype=np.float32),
        })

        # PPO：输出 heading (hx, hy) + speed multiplier (u)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_drones, 3), dtype=np.float32,
        )

    # ------------------ 时间单位统一：minutes <-> steps ------------------

    def _minutes_to_steps(self, minutes: int) -> int:
        minutes_per_step = 60 // self.time_system.steps_per_hour
        return int(math.ceil(float(minutes) / max(float(minutes_per_step), 1.0)))

    def _get_promised_delivery_steps(self, order: dict) -> int:
        """承诺送达时间（step）：prep_steps + 15分钟SLA(转step)"""
        prep_steps = int(order.get('preparation_time', 1))
        sla_steps = self._minutes_to_steps(15)
        return prep_steps + sla_steps

    # ------------------ READY-based deadline helpers ------------------

    def _get_delivery_sla_steps(self, order: dict) -> int:
        """Get delivery SLA in steps. Returns configured delivery_sla_steps."""
        return self.delivery_sla_steps

    def _get_delivery_deadline_step(self, order: dict) -> int:
        """
        Get READY-based delivery deadline step for an order.
        Uses ready_step as start time, falls back to creation_time if not available.
        If neither is available (unlikely), uses current_step as last resort.
        """
        ready_step = order.get('ready_step')
        if ready_step is None:
            # Fallback: use creation_time if ready_step not set yet
            # (e.g., for orders not yet READY or old orders before this feature)
            ready_step = order.get('creation_time')
            if ready_step is None:
                # Last resort: use current_step (should never happen in practice)
                ready_step = self.time_system.current_step

        delivery_sla = self._get_delivery_sla_steps(order)
        deadline_step = ready_step + int(round(delivery_sla * self.timeout_factor))
        return deadline_step

    # ------------------ Top-K merchants 观测选择 ------------------

    def _select_topk_merchants_for_observation(self) -> List[str]:
        """
        选择用于观测的商家ID列表（长度=obs_num_merchants）
        默认策略：按队列长度降序 Top-K（稳定、可解释、适合高负载）。
        """
        merchant_items = list(self.merchants.items())
        merchant_items.sort(key=lambda kv: len(kv[1].get('queue', [])), reverse=True)
        top_ids = [mid for mid, _ in merchant_items[:self.obs_num_merchants]]

        if len(top_ids) < self.obs_num_merchants:
            top_set = set(top_ids)
            rest = [mid for mid in sorted(self.merchants.keys(), key=lambda x: str(x)) if mid not in top_set]
            top_ids.extend(rest[:(self.obs_num_merchants - len(top_ids))])

        return top_ids

    # ------------------ reset / step ------------------

    def reset(self, seed=None, options=None):
        """重置环境 - 开始新的一天"""
        if seed is not None:
            set_global_seed(seed)

        self.time_system.reset()
        self.daily_stats['day_number'] = self.time_system.day_number

        # 重置订单系统
        self.orders = {}
        self.active_orders = set()
        self.completed_orders = set()
        self.cancelled_orders = set()
        self.global_order_counter = 0

        self._reset_drones_and_bases()
        self._reset_daily_stats()

        self.last_stats = {
            'completed': 0,
            'energy': 0,
            'on_time': 0,
            'cancelled': 0,
            'distance': 0.0,
        }

        self.path_visualizer.clear_paths()
        self.air_traffic = np.zeros((self.grid_size, self.grid_size))

        self._update_weather_from_dataset()
        self._generate_morning_orders()

        # 重置营业后履约状态
        self.in_overtime = False
        self.overtime_steps = 0
        self._assigned_in_step = set()
        self.pending_distance_reward = 0.0

        # ====== 多目标权重：每个 episode 一个偏好 ======
        if self.multi_objective_mode == "conditioned":
            w = np.random.dirichlet(alpha=np.ones(self.num_objectives)).astype(np.float32)
            self.objective_weights = w
        else:
            self.objective_weights = self.fixed_objective_weights.copy()

        # 初始化 shaping 距离缓存
        for d in range(self.num_drones):
            self._prev_target_dist[d] = self._get_dist_to_target(d)
        self.episode_r_vec[:] = 0.0
        obs = self._get_observation()
        obs['objective_weights'] = self.objective_weights.copy()
        info = self._get_info()

        print(f"=== 开始 ===")
        print(f"营业时间: {self.time_system.start_hour}:00 - {self.time_system.end_hour}:00")
        print(f"今日天气: {self.weather_details.get('summary', 'Unknown')}")
        print(f"无人机数量: {self.num_drones}, 订单观测窗口: {self.max_obs_orders}, 商家TopK: {self.obs_num_merchants}")
        return obs, info

    def step(self, action):
        """执行一步环境动作"""
        self._assigned_in_step = set()
        self._last_route_heading = action

        day_ended = self.time_system.step()
        time_state = self.time_system.get_time_state()

        # 每小时更新一次天气：使用 step 索引
        if time_state['minute'] == 0:
            self._update_weather_from_dataset()

        # 处理动作（heading 不直接给奖励，奖励来自系统指标+shaping）
        r_vec = self._process_action(action)

        # dense shaping
        shaping_vec = self._calculate_shaping_reward(action)
        r_vec = r_vec + shaping_vec
        self.episode_r_vec = self.episode_r_vec + r_vec.astype(np.float32)
        # 动作后立即更新
        self._immediate_state_update()

        # 动态事件
        self._process_events()

        # 清理过期分配
        self._cleanup_stale_assignments()

        # 强制状态同步
        self._force_state_synchronization()

        # 更新系统状态（扩展点）
        self._update_system_state()

        # 生成新订单（高负载）
        self._generate_new_orders()

        # 监控无人机状态（可扩展）
        if self.time_system.current_step % 4 == 0:
            self._monitor_drone_status()

        # 更新帕累托前沿
        self.pareto_optimizer.update_pareto_front(r_vec)

        # ---- 原逻辑：termination + final bonus ----
        terminated = self._check_termination(day_ended)
        truncated = False

        if terminated:
            final_bonus = self._calculate_daily_final_bonus()
            r_vec = r_vec + final_bonus
            self.in_overtime = False
            self.overtime_steps = 0

        # ---- 新增：累计 episode 向量回报（用最终 r_vec）----
        self.episode_r_vec = self.episode_r_vec + r_vec.astype(np.float32)

        # ---- 标量 reward 输出模式（路线1推荐用 "zero"）----
        if self.reward_output_mode == "scalar":
            scalar_reward = float(np.dot(self.objective_weights, r_vec))
        elif self.reward_output_mode == "obj0":
            scalar_reward = float(r_vec[0])
        elif self.reward_output_mode == "zero":
            scalar_reward = 0.0
        else:
            raise ValueError(f"Unknown reward_output_mode={self.reward_output_mode}")

        obs = self._get_observation()
        obs['objective_weights'] = self.objective_weights.copy()

        info = self._get_info()
        info['r_vec'] = r_vec.copy()
        info['episode_r_vec'] = self.episode_r_vec.copy()  # <- 新增
        info['objective_weights'] = self.objective_weights.copy()
        info['scalar_reward'] = scalar_reward
        info['episode'] = {
            'r': scalar_reward,
            'l': self.time_system.current_step,
            'day_number': self.time_system.day_number,
            'daily_stats': self.daily_stats.copy(),
            'r_vec': self.episode_r_vec.copy(),  # <- 新增：整集向量回报
        }
        return obs, scalar_reward, terminated, truncated, info

    # ------------------ reset helpers ------------------

    def _reset_drones_and_bases(self):
        """重置无人机和基站状态"""
        for base_id, base in self.bases.items():
            base['drones_available'] = []
            base['charging_queue'] = deque()

        for i in range(self.num_drones):
            base_id = i % self.num_bases
            base_location = self.bases[base_id]['location']

            self.drones[i] = {
                'location': base_location,
                'base': base_id,
                'status': DroneStatus.IDLE,
                'current_order': None,
                'orders_completed': 0,
                'speed': random.uniform(3.0, 5.0),
                'reliability': random.uniform(0.95, 0.99),
                'max_capacity': self.drone_max_capacity,
                'current_load': 0,
                'battery_level': 100.0,
                'max_battery': 100.0,
                'battery_consumption_rate': random.uniform(0.2, 0.4),
                'charging_rate': random.uniform(10.0, 15.0),
                'cancellation_rate': random.uniform(0.005, 0.015),
                'total_distance_today': 0.0,
            }
            self.drones[i]['planned_stops'] = deque()
            self.drones[i]['cargo'] = set()
            self.drones[i]['current_stop'] = None
            self.drones[i]['route_committed'] = False
            keys_to_remove = ['task_start_location', 'task_start_step', 'accumulated_distance',
                              'optimal_distance', 'last_location', 'target_location',
                              'batch_orders', 'current_batch_index', 'current_delivery_index',
                              'waiting_start_time', 'route_preferences',
                              'current_task_distance', 'task_optimal_distance', 'trip_started',
                              'trip_actual_distance', 'trip_optimal_distance']
            for key in keys_to_remove:
                self.drones[i].pop(key, None)

            self.bases[base_id]['drones_available'].append(i)

    def _reset_daily_stats(self):
        """重置每日统计"""
        self.daily_stats.update({
            'orders_generated': 0,
            'orders_completed': 0,
            'orders_cancelled': 0,
            'revenue': 0,
            'costs': 0,
            'energy_consumed': 0,
            'on_time_deliveries': 0,
            'total_flight_distance': 0.0,
            'optimal_flight_distance': 0.0,
        })

    def _generate_morning_orders(self):
        """生成早晨初始订单"""
        morning_prob = 0.5
        for _ in range(5):
            if random.random() < morning_prob:
                self._generate_single_order()

    # ------------------ trip distance helpers ------------------

    def _ensure_trip_started(self, drone: dict):
        if not drone.get('trip_started', False):
            drone['trip_started'] = True
            drone['trip_actual_distance'] = 0.0
            drone['trip_optimal_distance'] = 0.0

    def _accumulate_trip_optimal_leg(self, drone: dict, start_loc, end_loc):
        leg = math.sqrt((end_loc[0] - start_loc[0]) ** 2 + (end_loc[1] - start_loc[1]) ** 2)
        drone['trip_optimal_distance'] = float(drone.get('trip_optimal_distance', 0.0)) + float(leg)

    def _settle_trip_distance_reward(self, drone_id: int, drone: dict, reason: str = ""):
        """整趟结束结算（建议归入目标0：效率）。"""
        if not drone.get('trip_started', False):
            return

        optimal = float(drone.get('trip_optimal_distance', 0.0))
        actual = float(drone.get('trip_actual_distance', 0.0))

        if optimal < 0.1 or actual < 0.1:
            r = 0.0
        else:
            efficiency = optimal / max(actual, 0.1)
            if efficiency >= 0.95:
                r = 1.0
            elif efficiency >= 0.85:
                r = 0.5
            elif efficiency >= 0.75:
                r = 0.0
            else:
                r = -0.5 * (1.0 - efficiency)

        self.pending_distance_reward += float(r) * float(self.distance_reward_weight)

    def _clear_trip_data(self, drone: dict):
        for k in ['trip_started', 'trip_actual_distance', 'trip_optimal_distance']:
            drone.pop(k, None)

    # ------------------ 强制状态同步/修复 ------------------

    def _reset_order_to_ready(self, order_id, reason=""):
        """将订单重置为READY状态，允许重新分配（走 StateManager）"""
        if order_id not in self.orders:
            return

        order = self.orders[order_id]
        old_drone = order.get('assigned_drone', -1)

        self.state_manager.update_order_status(order_id, OrderStatus.READY, reason=f"reset_to_ready:{reason}")
        order['assigned_drone'] = -1

        if old_drone is not None and old_drone >= 0 and old_drone in self.drones:
            self.drones[old_drone]['current_load'] = max(0, self.drones[old_drone]['current_load'] - 1)
            if 'batch_orders' in self.drones[old_drone]:
                if order_id in self.drones[old_drone]['batch_orders']:
                    self.drones[old_drone]['batch_orders'].remove(order_id)

    def _force_complete_order(self, order_id, drone_id):
        """强制完成订单（用于异常状态恢复）（走 StateManager）"""
        if order_id not in self.orders:
            return

        order = self.orders[order_id]
        if order['status'] == OrderStatus.DELIVERED:
            return

        self.state_manager.update_order_status(order_id, OrderStatus.DELIVERED, reason="force_complete")
        order['delivery_time'] = self.time_system.current_step
        order['assigned_drone'] = -1

        if drone_id in self.drones:
            self.drones[drone_id]['orders_completed'] += 1
            self.drones[drone_id]['current_load'] = max(0, self.drones[drone_id]['current_load'] - 1)

        self.metrics['completed_orders'] += 1
        self.daily_stats['orders_completed'] += 1

        self.active_orders.discard(order_id)
        self.completed_orders.add(order_id)

    def _force_state_synchronization(self):
        """强制状态同步：确保订单与无人机状态一致"""
        drone_real_orders = {d_id: set() for d_id in self.drones}

        for drone_id, drone in self.drones.items():
            if 'batch_orders' in drone and drone['batch_orders']:
                for order_id in drone['batch_orders']:
                    if order_id in self.orders:
                        order = self.orders[order_id]
                        if order['status'] in [OrderStatus.ASSIGNED, OrderStatus.PICKED_UP]:
                            drone_real_orders[drone_id].add(order_id)
            # --- ADD: cargo-based "real orders" (picked up but not delivered) ---
            for oid in drone.get('cargo', set()):
                if oid in self.orders:
                    order = self.orders[oid]
                    if order['status'] == OrderStatus.PICKED_UP:
                        drone_real_orders[drone_id].add(oid)

        for order_id in list(self.active_orders):
            if order_id not in self.orders:
                self.active_orders.discard(order_id)
                continue

            order = self.orders[order_id]
            if order['status'] not in [OrderStatus.ASSIGNED, OrderStatus.PICKED_UP]:
                continue

            drone_id = order.get('assigned_drone', -1)

            if drone_id is None or drone_id < 0 or drone_id not in self.drones:
                self._reset_order_to_ready(order_id, "invalid_drone_id")
                continue

            drone = self.drones[drone_id]

            if drone['status'] in [DroneStatus.IDLE, DroneStatus.CHARGING]:
                if order_id not in drone_real_orders.get(drone_id, set()):
                    if order['status'] == OrderStatus.PICKED_UP:
                        self._force_complete_order(order_id, drone_id)
                    else:
                        self._reset_order_to_ready(order_id, "drone_idle")
                    continue

            if drone['status'] == DroneStatus.RETURNING_TO_BASE:
                if order['status'] == OrderStatus.ASSIGNED:
                    self._reset_order_to_ready(order_id, "drone_returning")
                elif order['status'] == OrderStatus.PICKED_UP:
                    self._force_complete_order(order_id, drone_id)
                continue

            if drone['status'] == DroneStatus.FLYING_TO_CUSTOMER:
                # Route-aware fix: Only auto-pickup if NOT in route-plan mode
                # In route-plan mode, orders are picked up explicitly at P stops
                # Note: route_committed is set to True in apply_route_plan and cleared to False
                # in _safe_reset_drone and when drone returns to base (RETURNING_TO_BASE arrival handler)
                if not drone.get('route_committed', False):
                    # Legacy mode: auto-pickup when flying to customer
                    if order['status'] == OrderStatus.ASSIGNED:
                        self.state_manager.update_order_status(order_id, OrderStatus.PICKED_UP,
                                                               reason="sync_assigned_to_picked_up")
                        if 'pickup_time' not in order:
                            order['pickup_time'] = self.time_system.current_step

        for drone_id, drone in self.drones.items():
            if 'batch_orders' in drone:
                valid_batch = []
                for order_id in drone['batch_orders']:
                    if order_id in self.orders:
                        order = self.orders[order_id]
                        if (order['status'] in [OrderStatus.ASSIGNED, OrderStatus.PICKED_UP] and
                                order.get('assigned_drone') == drone_id):
                            valid_batch.append(order_id)

                if valid_batch:
                    drone['batch_orders'] = valid_batch
                else:
                    self._clear_drone_batch_state(drone)

            actual_load = 0
            for order_id in self.active_orders:
                if order_id in self.orders:
                    order = self.orders[order_id]
                    if (order.get('assigned_drone') == drone_id and
                            order['status'] in [OrderStatus.ASSIGNED, OrderStatus.PICKED_UP]):
                        actual_load += 1
            drone['current_load'] = actual_load

    def _clear_drone_batch_state(self, drone):
        keys_to_remove = ['batch_orders', 'current_batch_index', 'current_delivery_index', 'waiting_start_time']
        for key in keys_to_remove:
            if key in drone:
                del drone[key]

    # ------------------ overtime termination ------------------

    def _busy_drones_exist(self) -> bool:
        for d in self.drones.values():
            if d['status'] not in [DroneStatus.IDLE, DroneStatus.CHARGING]:
                return True
            if d.get('current_load', 0) > 0:
                return True
        return False

    def _overtime_done(self) -> bool:
        return len(self.active_orders) == 0 and not self._busy_drones_exist()

    def _check_termination(self, day_ended):
        if not self.in_overtime:
            if day_ended:
                if self.allow_overtime_fulfillment and (len(self.active_orders) > 0 or self._busy_drones_exist()):
                    self.in_overtime = True
                    self.overtime_steps = 0
                    return False
                else:
                    return True
            return False
        else:
            self.overtime_steps += 1
            if self._overtime_done():
                return True
            elif self.overtime_steps >= self.overtime_max_steps:
                self._handle_end_of_day()
                return True
            return False

    # ------------------ rewards ------------------

    def _process_action(self, action):
        """heading action 不直接给即时奖励；奖励来自系统指标+shaping"""
        self._last_route_heading = action
        r_vec = self._calculate_three_objective_rewards()
        return r_vec

    def _get_dist_to_target(self, drone_id: int) -> float:
        drone = self.drones[drone_id]
        if 'target_location' not in drone:
            return 0.0
        cx, cy = drone['location']
        tx, ty = drone['target_location']
        return float(math.sqrt((tx - cx) ** 2 + (ty - cy) ** 2))

    def _calculate_shaping_reward(self, action):
        r = np.zeros(self.num_objectives, dtype=np.float32)

        for d in range(self.num_drones):
            drone = self.drones[d]
            if drone['status'] not in [DroneStatus.FLYING_TO_MERCHANT, DroneStatus.FLYING_TO_CUSTOMER,
                                       DroneStatus.RETURNING_TO_BASE]:
                continue
            if 'target_location' not in drone:
                continue

            new_dist = self._get_dist_to_target(d)
            prev_dist = float(self._prev_target_dist[d])

            progress = (prev_dist - new_dist)
            r[0] += self.shaping_progress_k * float(progress)

            speed = float(drone['speed']) * float(self._get_weather_speed_factor())
            r[1] -= self.shaping_distance_k * speed

            # 能耗 shaping 也可放到目标1（-cost）
            battery_consumption = float(drone["battery_consumption_rate"]) * float(self._get_weather_battery_factor())
            r[1] -= self.shaping_energy_k * battery_consumption

            self._prev_target_dist[d] = new_dist

        return r

    def _calculate_three_objective_rewards(self):
        """
        三目标语义单一：
        - obj0: 吞吐/效率（完成数、距离效率结算、取消惩罚等）
        - obj1: -成本（距离、能耗等纯负成本）
        - obj2: 服务质量（准时、取消、积压）
        """
        rewards = np.zeros(self.num_objectives, dtype=np.float32)

        current_completed = self.daily_stats['orders_completed']
        current_energy = self.daily_stats['energy_consumed']
        current_on_time = self.daily_stats['on_time_deliveries']
        current_cancelled = self.daily_stats['orders_cancelled']
        current_distance = self.daily_stats['total_flight_distance']

        delta_completed = current_completed - self.last_stats['completed']
        delta_energy = current_energy - self.last_stats['energy']
        delta_on_time = current_on_time - self.last_stats['on_time']
        delta_cancelled = current_cancelled - self.last_stats['cancelled']
        delta_distance = current_distance - self.last_stats['distance']

        self.last_stats['completed'] = current_completed
        self.last_stats['energy'] = current_energy
        self.last_stats['on_time'] = current_on_time
        self.last_stats['cancelled'] = current_cancelled
        self.last_stats['distance'] = current_distance

        # obj0：吞吐/效率
        rewards[0] += float(delta_completed) * 2.0
        rewards[0] -= float(delta_cancelled) * 1.0

        # 整趟距离效率：放到 obj0（效率）
        if self.pending_distance_reward != 0.0:
            rewards[0] += float(self.pending_distance_reward)
            self.pending_distance_reward = 0.0

        # obj1：-成本（纯负成本）
        rewards[1] -= float(delta_energy) * 0.01
        rewards[1] -= float(delta_distance) * 0.02

        # obj2：服务质量
        rewards[2] += float(delta_on_time) * 1.5
        rewards[2] -= float(delta_cancelled) * 0.5
        backlog = len(self.active_orders)
        rewards[2] -= float(backlog) * 0.005

        return rewards

    def _calculate_daily_final_bonus(self):
        bonus = np.zeros(self.num_objectives, dtype=np.float32)

        daily_completion_rate = self.daily_stats['orders_completed'] / max(1, self.daily_stats['orders_generated'])
        bonus[0] += daily_completion_rate * 3.0
        bonus[2] += daily_completion_rate * 2.0

        if self.daily_stats['orders_completed'] > 0:
            daily_on_time_rate = self.daily_stats['on_time_deliveries'] / self.daily_stats['orders_completed']
            bonus[2] += daily_on_time_rate * 1.5

        if self.daily_stats['orders_completed'] > 0:
            energy_per_order = self.daily_stats['energy_consumed'] / self.daily_stats['orders_completed']
            energy_efficiency = max(0, 1 - energy_per_order / 30.0)
            bonus[1] += energy_efficiency * 1.2

        if len(self.active_orders) > 0:
            penalty = len(self.active_orders) * 0.1
            bonus[0] -= penalty
            bonus[2] -= penalty

        return bonus

    # ------------------ MPSO 分配相关：保留接口但不从 action 中学习 ------------------

    def apply_route_plan(self,
                         drone_id: int,
                         planned_stops: List[dict],
                         commit_orders: Optional[List[int]] = None,
                         allow_busy: bool = True) -> bool:
        """
        Apply a cross-merchant interleaved route plan.
        planned_stops:
            [{'type':'P','merchant_id':mid}, {'type':'D','order_id':oid}, ...]
        Constraints (per your requirement):
            - planned_stops must not include orders that are not READY at dispatch time.
            - Pickup is executed only when arriving at that merchant stop.
            - Delivery is executed only when arriving at that delivery stop.

        Commit semantics:
            commit_orders are moved READY -> ASSIGNED and assigned to this drone.
            Orders are NOT marked PICKED_UP at commit time.

        Returns:
            bool: True if route was successfully applied, False otherwise.
        """
        if drone_id not in self.drones:
            return False
        drone = self.drones[drone_id]

        if (not allow_busy) and drone['status'] != DroneStatus.IDLE:
            return False

        # Derive commit_orders if not provided
        if commit_orders is None:
            commit_orders = []
            for st in planned_stops:
                if st.get('type') == 'D' and 'order_id' in st:
                    commit_orders.append(int(st['order_id']))
            # unique, keep order
            commit_orders = list(dict.fromkeys(commit_orders))

        # 1) Commit orders: only READY orders are allowed
        committed = []
        for oid in commit_orders:
            o = self.orders.get(oid)
            if o is None:
                continue

            # Enforced by requirement: only READY can be in planned_stops/commit list
            if o['status'] != OrderStatus.READY:
                continue
            if o.get('assigned_drone', -1) not in (-1, None):
                continue
            if drone['current_load'] >= drone['max_capacity']:
                break

            self.state_manager.update_order_status(oid, OrderStatus.ASSIGNED, reason=f"route_committed_{drone_id}")
            o['assigned_drone'] = drone_id
            drone['current_load'] += 1
            committed.append(oid)

            # Record READY-based assignment slack for diagnostics
            deadline_step = self._get_delivery_deadline_step(o)
            assignment_slack = deadline_step - self.time_system.current_step
            if 'assignment_slack_samples' not in self.metrics:
                self.metrics['assignment_slack_samples'] = []
            self.metrics['assignment_slack_samples'].append(assignment_slack)

        if not committed:
            return False

        # Task C: Validate that orders in planned_stops are in committed set
        # and filter out D stops for non-committed orders
        committed_set = set(committed)
        filtered_stops = []
        for stop in planned_stops:
            if stop.get('type') == 'D':
                oid = stop.get('order_id')
                if oid is not None and oid not in committed_set:
                    if self.debug_state_warnings:
                        print(f"[Warning] Drone {drone_id} skipping D stop for uncommitted order {oid}")
                    continue  # Skip this D stop
            filtered_stops.append(stop)

        # If no stops remain after filtering, don't install route
        if not filtered_stops:
            if self.debug_state_warnings:
                print(f"[Warning] Drone {drone_id} route empty after filtering - no valid D stops")
            return False

        # 2) Install route plan with filtered stops
        drone['planned_stops'] = deque(filtered_stops)
        drone['cargo'] = set()  # picked-up set starts empty
        drone['current_stop'] = None
        drone['route_committed'] = True

        # Clear legacy batch state to avoid conflicts with route-plan mode
        self._clear_drone_batch_state(drone)

        # 3) Start execution: set target to the first stop
        self._set_next_target_from_plan(drone_id, drone)

        return True

    def _process_batch_assignment(self, drone_id, order_ids):
        """环境内部执行批量订单分配：给 MPSO 调用"""
        if not order_ids:
            return
        drone = self.drones[drone_id]

        can_take = max(0, drone['max_capacity'] - drone['current_load'])
        if can_take <= 0:
            return

        cand = []
        for oid in order_ids:
            o = self.orders.get(oid)
            if o is None or o['status'] != OrderStatus.READY:
                continue
            if o.get('assigned_drone', -1) not in (-1, None):
                continue
            cand.append(oid)
            if len(cand) >= can_take:
                break
        if not cand:
            return

        actually_assigned = []
        for oid in cand:
            before = drone['current_load']
            self._process_single_assignment(drone_id, oid, allow_busy=True)
            if drone['current_load'] > before:
                actually_assigned.append(oid)

        if not actually_assigned:
            return

        first_order = self.orders[actually_assigned[0]]
        self.state_manager.update_drone_status(drone_id, DroneStatus.FLYING_TO_MERCHANT,
                                               first_order['merchant_location'])

        if 'batch_orders' not in drone:
            drone['batch_orders'] = []
        drone['batch_orders'].extend(actually_assigned)
        if 'current_batch_index' not in drone:
            drone['current_batch_index'] = 0

    def _process_single_assignment(self, drone_id, order_id, allow_busy=False):
        """处理单个订单分配（记录任务起点 + 走 StateManager）"""
        if order_id not in self.orders or drone_id not in self.drones:
            return

        order = self.orders[order_id]
        drone = self.drones[drone_id]

        if order['status'] != OrderStatus.READY:
            return
        if order.get('assigned_drone', -1) not in (-1, None):
            return
        if not allow_busy and drone['status'] != DroneStatus.IDLE:
            return
        if drone['current_load'] >= drone['max_capacity']:
            return

        self.state_manager.update_order_status(order_id, OrderStatus.ASSIGNED, reason=f"assigned_to_drone_{drone_id}")
        order['assigned_drone'] = drone_id
        drone['current_load'] += 1

        # Record READY-based assignment slack for diagnostics
        deadline_step = self._get_delivery_deadline_step(order)
        assignment_slack = deadline_step - self.time_system.current_step
        if 'assignment_slack_samples' not in self.metrics:
            self.metrics['assignment_slack_samples'] = []
        self.metrics['assignment_slack_samples'].append(assignment_slack)

        target_merchant_loc = order['merchant_location']

        if drone['status'] in [DroneStatus.IDLE, DroneStatus.RETURNING_TO_BASE, DroneStatus.CHARGING]:
            self._ensure_trip_started(drone)
            self._accumulate_trip_optimal_leg(drone, drone['location'], target_merchant_loc)

            self._start_new_task(drone_id, drone, target_merchant_loc)
            self.state_manager.update_drone_status(drone_id, DroneStatus.FLYING_TO_MERCHANT, target_merchant_loc)

        elif drone['status'] == DroneStatus.FLYING_TO_MERCHANT:
            pass

        elif drone['status'] == DroneStatus.FLYING_TO_CUSTOMER:
            self._ensure_trip_started(drone)
            self._accumulate_trip_optimal_leg(drone, drone['location'], target_merchant_loc)

            self._start_new_task(drone_id, drone, target_merchant_loc)
            self.state_manager.update_drone_status(drone_id, DroneStatus.FLYING_TO_MERCHANT, target_merchant_loc)

    # ------------------ 事件处理与移动 ------------------

    def _immediate_state_update(self):
        self._update_drone_positions_immediately()
        self._update_merchant_preparation_immediately()

    def _update_drone_positions_immediately(self):
        for drone_id, drone in self.drones.items():
            if drone['status'] in [DroneStatus.FLYING_TO_MERCHANT,
                                   DroneStatus.FLYING_TO_CUSTOMER,
                                   DroneStatus.RETURNING_TO_BASE]:

                if 'target_location' in drone:
                    current_loc = drone['location']
                    target_loc = drone['target_location']

                    distance = math.sqrt((target_loc[0] - current_loc[0]) ** 2 +
                                         (target_loc[1] - current_loc[1]) ** 2)

                    if distance > 0.1:
                        dx = target_loc[0] - current_loc[0]
                        dy = target_loc[1] - current_loc[1]
                        move_distance = min(distance, 0.3)

                        if distance > 0:
                            new_x = current_loc[0] + (dx / distance) * move_distance
                            new_y = current_loc[1] + (dy / distance) * move_distance
                            drone['location'] = (new_x, new_y)

                            new_distance = math.sqrt((target_loc[0] - new_x) ** 2 +
                                                     (target_loc[1] - new_y) ** 2)
                            if new_distance < 0.1:
                                self._handle_drone_arrival(drone_id, drone)

    def _update_merchant_preparation_immediately(self):
        for merchant_id, merchant in self.merchants.items():
            ready_orders = []
            for order_id in list(merchant['queue']):
                if order_id not in self.orders:
                    continue

                order = self.orders[order_id]
                if order['status'] == OrderStatus.ACCEPTED:
                    time_elapsed = (self.time_system.current_step - order['creation_time'])
                    preparation_required = order['preparation_time']

                    if time_elapsed >= preparation_required:
                        self.state_manager.update_order_status(
                            order_id, OrderStatus.READY, reason="immediate_preparation"
                        )
                        ready_orders.append(order_id)

            for order_id in ready_orders:
                if order_id in merchant['queue']:
                    merchant['queue'].remove(order_id)

    def _process_events(self):
        self._update_merchant_preparation()
        self._update_drone_positions()
        if self.enable_random_events:
            self._handle_random_events()
        self._update_air_traffic()

        # Task B: Only print warnings if debug flag is enabled
        if self.debug_state_warnings:
            consistency_issues = self.state_manager.get_state_consistency_check()
            if consistency_issues:
                print("状态一致性警告:")
                for issue in consistency_issues:
                    print(f"  - {issue}")
        else:
            # Silently check and count issues but don't spam output
            consistency_issues = self.state_manager.get_state_consistency_check()
            if consistency_issues and self.time_system.current_step % 64 == 0:
                # Periodic summary: print count every 64 steps (4 hours at 4 steps/hour)
                print(f"[Step {self.time_system.current_step}] 状态一致性问题计数: {len(consistency_issues)}")

    def _update_merchant_preparation(self):
        for merchant_id, merchant in self.merchants.items():
            ready_orders = []
            for order_id in list(merchant['queue']):
                if order_id not in self.orders:
                    continue

                order = self.orders[order_id]
                if order['status'] == OrderStatus.ACCEPTED:
                    minutes_per_step = 60 // self.time_system.steps_per_hour

                    time_elapsed_minutes = (self.time_system.current_step - order['creation_time']) * minutes_per_step
                    preparation_required_minutes = order['preparation_time'] * minutes_per_step

                    merchant_efficiency = merchant.get('efficiency', 1.0)
                    adjusted_preparation_time = preparation_required_minutes / merchant_efficiency

                    if time_elapsed_minutes >= adjusted_preparation_time:
                        self.state_manager.update_order_status(
                            order_id, OrderStatus.READY, reason="preparation_complete"
                        )
                        ready_orders.append(order_id)

            for order_id in ready_orders:
                if order_id in merchant['queue']:
                    merchant['queue'].remove(order_id)

    def _sync_drone_status_with_route(self):
        """
        Synchronize drone status with route plan (Task A).
        Called at the start of each step to ensure drone status matches planned_stops.
        """
        for drone_id, drone in self.drones.items():
            planned_stops = drone.get('planned_stops')
            if not planned_stops or len(planned_stops) == 0:
                # No planned stops - allow IDLE or RETURNING
                continue

            # Get first stop
            stop = planned_stops[0]
            stop_type = stop.get('type')

            if stop_type == 'P':
                # Pickup stop - should be FLYING_TO_MERCHANT
                expected_status = DroneStatus.FLYING_TO_MERCHANT
                loc = self._stop_to_location(stop)
                if loc and drone['status'] != expected_status:
                    self.state_manager.update_drone_status(drone_id, expected_status, target_location=loc)
                    drone['current_merchant_id'] = stop.get('merchant_id')
                # Ensure target_location matches stop location
                if loc and drone.get('target_location') != loc:
                    drone['target_location'] = loc

            elif stop_type == 'D':
                # Delivery stop - should be FLYING_TO_CUSTOMER
                expected_status = DroneStatus.FLYING_TO_CUSTOMER
                loc = self._stop_to_location(stop)
                if loc and drone['status'] != expected_status:
                    self.state_manager.update_drone_status(drone_id, expected_status, target_location=loc)
                    drone['current_order_id'] = stop.get('order_id')
                # Ensure target_location matches stop location
                if loc and drone.get('target_location') != loc:
                    drone['target_location'] = loc

    def _update_drone_positions(self):
        # Sync drone status with route plan at the start of position update
        self._sync_drone_status_with_route()

        headings = getattr(self, "_last_route_heading", None)

        for drone_id, drone in self.drones.items():
            self.path_visualizer.update_path_history(drone_id, drone["location"])

            if drone["status"] in [
                DroneStatus.FLYING_TO_MERCHANT,
                DroneStatus.FLYING_TO_CUSTOMER,
                DroneStatus.RETURNING_TO_BASE,
            ]:
                # RETURNING_TO_BASE：完全按目标直飞，不听 PPO
                if drone["status"] == DroneStatus.RETURNING_TO_BASE:
                    alpha = 0.0
                else:
                    alpha = float(self.heading_guidance_alpha)

                if "target_location" not in drone:
                    self._reset_drone_to_base(drone_id, drone)
                    continue

                cx, cy = drone["location"]
                tx, ty = drone["target_location"]

                to_target_dx = tx - cx
                to_target_dy = ty - cy
                dist_to_target = float(math.sqrt(to_target_dx * to_target_dx + to_target_dy * to_target_dy))

                ARRIVAL_THRESHOLD = 0.5
                if dist_to_target <= ARRIVAL_THRESHOLD:
                    drone["location"] = (tx, ty)
                    self._handle_drone_arrival(drone_id, drone)
                    continue

                speed = float(drone["speed"]) * float(self._get_weather_speed_factor())
                if speed <= 1e-6:
                    continue

                if headings is None:
                    ppo_hx, ppo_hy, ppo_u = 0.0, 0.0, 1.0
                else:
                    # Extract (hx, hy, u) from action - u is speed multiplier
                    action_vec = headings[int(drone_id)]
                    ppo_hx, ppo_hy = action_vec[0], action_vec[1]
                    ppo_u = float(action_vec[2]) if len(action_vec) > 2 else 1.0

                    # Map u from [-1, 1] to [SPEED_MULTIPLIER_MIN, SPEED_MULTIPLIER_MAX]
                    # Using standard linear interpolation:
                    #   normalized = (value - old_min) / (old_max - old_min)
                    #   result = normalized * (new_max - new_min) + new_min
                    # Here: old_range=[-1,1], new_range=[0.5,1.5]
                    normalized_u = (ppo_u + 1.0) / 2.0  # Map to [0, 1]
                    ppo_u = normalized_u * (SPEED_MULTIPLIER_MAX - SPEED_MULTIPLIER_MIN) + SPEED_MULTIPLIER_MIN
                    ppo_u = np.clip(ppo_u, SPEED_MULTIPLIER_MIN, SPEED_MULTIPLIER_MAX)

                ppo_norm = math.sqrt(ppo_hx * ppo_hx + ppo_hy * ppo_hy)
                if ppo_norm > 1e-6:
                    ppo_hx /= ppo_norm
                    ppo_hy /= ppo_norm
                else:
                    ppo_hx, ppo_hy = 0.0, 0.0

                tgt_hx = to_target_dx / max(dist_to_target, 1e-6)
                tgt_hy = to_target_dy / max(dist_to_target, 1e-6)

                move_hx = alpha * ppo_hx + (1.0 - alpha) * tgt_hx
                move_hy = alpha * ppo_hy + (1.0 - alpha) * tgt_hy

                move_norm = math.sqrt(move_hx * move_hx + move_hy * move_hy)
                if move_norm > 1e-6:
                    move_hx /= move_norm
                    move_hy /= move_norm
                else:
                    move_hx, move_hy = tgt_hx, tgt_hy

                # Apply speed multiplier from PPO action
                step_len = min(speed * ppo_u, dist_to_target)
                nx = float(np.clip(cx + move_hx * step_len, 0, self.grid_size - 1))
                ny = float(np.clip(cy + move_hy * step_len, 0, self.grid_size - 1))

                if "last_location" in drone:
                    step_distance = float(
                        math.sqrt((nx - drone["last_location"][0]) ** 2 + (ny - drone["last_location"][1]) ** 2)
                    )
                else:
                    step_distance = float(math.sqrt((nx - cx) ** 2 + (ny - cy) ** 2))

                drone["last_location"] = (nx, ny)

                drone["total_distance_today"] = float(drone.get("total_distance_today", 0.0)) + step_distance
                self.daily_stats["total_flight_distance"] = float(
                    self.daily_stats.get("total_flight_distance", 0.0)) + step_distance
                self.metrics["total_flight_distance"] = float(
                    self.metrics.get("total_flight_distance", 0.0)) + step_distance

                if drone.get("trip_started", False):
                    drone["trip_actual_distance"] = float(drone.get("trip_actual_distance", 0.0)) + step_distance

                drone["location"] = (nx, ny)

                new_dist = float(math.sqrt((tx - nx) ** 2 + (ty - ny) ** 2))
                if new_dist <= ARRIVAL_THRESHOLD:
                    drone["location"] = (tx, ty)
                    self._handle_drone_arrival(drone_id, drone)

                battery_consumption = float(drone["battery_consumption_rate"]) * float(
                    self._get_weather_battery_factor())
                drone["battery_level"] = max(0.0, float(drone["battery_level"]) - battery_consumption)
                self.metrics["energy_consumed"] = float(self.metrics.get("energy_consumed", 0.0)) + battery_consumption
                self.daily_stats["energy_consumed"] = float(
                    self.daily_stats.get("energy_consumed", 0.0)) + battery_consumption

            elif drone["status"] == DroneStatus.WAITING_FOR_PICKUP:
                self._handle_waiting_pickup(drone_id, drone)

            elif drone["status"] == DroneStatus.DELIVERING:
                self._handle_delivering(drone_id, drone)

            elif drone["status"] == DroneStatus.CHARGING:
                self._handle_charging(drone_id, drone)

    # ------------------ 任务/到达处理（统一结算出口）------------------
    def _stop_to_location(self, stop: dict):
        """Map a planned stop to a concrete target location."""
        stype = stop.get('type', None)
        if stype == 'P':
            mid = stop.get('merchant_id', None)
            if mid is None or mid not in self.merchants:
                return None
            return self.merchants[mid]['location']
        if stype == 'D':
            oid = stop.get('order_id', None)
            if oid is None or oid not in self.orders:
                return None
            return self.orders[oid]['customer_location']
        return None

    def _set_next_target_from_plan(self, drone_id: int, drone: dict) -> None:
        """
        Pop invalid stops until a valid target is found; then set drone target/status.
        If no stop remains, reset/return-to-base.
        Task C: Add validation for stop/order consistency.
        """
        while drone.get('planned_stops') and len(drone['planned_stops']) > 0:
            stop = drone['planned_stops'][0]
            loc = self._stop_to_location(stop)
            if loc is None:
                if self.debug_state_warnings:
                    print(f"[Drone {drone_id}] 无效 stop (location not found): {stop}")
                drone['planned_stops'].popleft()
                continue

            # Task C: Validate D stops reference valid orders
            if stop.get('type') == 'D':
                oid = stop.get('order_id')
                if oid is not None and oid in self.orders:
                    o = self.orders[oid]
                    # Check if order is valid for delivery
                    if o.get('assigned_drone') != drone_id:
                        if self.debug_state_warnings:
                            print(f"[Drone {drone_id}] D stop {oid} 不属于该无人机，跳过")
                        drone['planned_stops'].popleft()
                        continue
                    if o['status'] in [OrderStatus.CANCELLED, OrderStatus.DELIVERED]:
                        if self.debug_state_warnings:
                            print(f"[Drone {drone_id}] D stop {oid} 订单已取消或已送达，跳过")
                        drone['planned_stops'].popleft()
                        continue

            drone['current_stop'] = stop
            self._start_new_task(drone_id, drone, loc)

            if stop.get('type') == 'P':
                self.state_manager.update_drone_status(drone_id, DroneStatus.FLYING_TO_MERCHANT, target_location=loc)
            else:
                self.state_manager.update_drone_status(drone_id, DroneStatus.FLYING_TO_CUSTOMER, target_location=loc)
            return

        # No stops left: end this route cleanly
        self._safe_reset_drone(drone_id, drone)

    def _execute_pickup_stop(self, drone_id: int, drone: dict, stop: dict) -> None:
        """
        Arrive at merchant mid: pick up only orders assigned to this drone AND belonging to this merchant.
        READY-only planning: orders should already be ASSIGNED here, but we still check status strictly.
        """
        mid = stop.get('merchant_id', None)
        if mid is None:
            return

        for oid in list(self.active_orders):
            o = self.orders.get(oid)
            if o is None:
                continue
            if o.get('assigned_drone', -1) != drone_id:
                continue
            if o.get('merchant_id', None) != mid:
                continue
            if o['status'] != OrderStatus.ASSIGNED:
                continue

            self.state_manager.update_order_status(oid, OrderStatus.PICKED_UP, reason=f"pickup_at_merchant_{mid}")
            o['pickup_time'] = self.time_system.current_step
            drone['cargo'].add(oid)

    def _execute_delivery_stop(self, drone_id: int, drone: dict, stop: dict) -> None:
        """Arrive at customer: deliver exactly the specified order (must be PICKED_UP)."""
        oid = stop.get('order_id', None)
        if oid is None or oid not in self.orders:
            return

        o = self.orders[oid]
        if o.get('assigned_drone', -1) != drone_id:
            return

        # Strict legality for READY-only planning: must have been picked up at its merchant stop
        if o['status'] != OrderStatus.PICKED_UP:
            return

        self._complete_order_delivery(oid, drone_id)

        if oid in drone.get('cargo', set()):
            drone['cargo'].remove(oid)

    def _safe_reset_drone(self, drone_id, drone):
        """唯一出口：整趟结算 + 清理 trip + 清理关联订单 + 返航"""
        self._settle_trip_distance_reward(drone_id, drone, reason="safe_reset")
        self._clear_trip_data(drone)

        for order_id, order in list(self.orders.items()):
            if order.get('assigned_drone') == drone_id:
                if order['status'] == OrderStatus.PICKED_UP:
                    self._force_complete_order(order_id, drone_id)
                elif order['status'] == OrderStatus.ASSIGNED:
                    self._reset_order_to_ready(order_id, "drone_reset")

        self._clear_drone_batch_state(drone)
        # --- ADD: clear planned route data ---
        drone['planned_stops'] = deque()
        drone['cargo'] = set()
        drone['current_stop'] = None
        drone['route_committed'] = False
        # 返航（返航不计入整趟，因为 trip 字段已清理）
        base_loc = self.bases[drone['base']]['location']
        self.state_manager.update_drone_status(drone_id, DroneStatus.RETURNING_TO_BASE, target_location=base_loc)
        drone['current_load'] = 0

    def _handle_batch_pickup(self, drone_id, drone):
        if 'batch_orders' not in drone or not drone['batch_orders']:
            self._safe_reset_drone(drone_id, drone)
            return

        picked_count = 0
        for order_id in drone['batch_orders']:
            order = self.orders.get(order_id)
            if order and order['status'] == OrderStatus.ASSIGNED:
                self.state_manager.update_order_status(order_id, OrderStatus.PICKED_UP, reason="batch_pickup")
                order['pickup_time'] = self.time_system.current_step
                picked_count += 1

        if picked_count == 0:
            self._safe_reset_drone(drone_id, drone)
            return

        self._start_batch_delivery(drone_id, drone)

    def _start_batch_delivery(self, drone_id, drone):
        if 'batch_orders' not in drone or not drone['batch_orders']:
            self._safe_reset_drone(drone_id, drone)
            return

        drone['current_delivery_index'] = 0

        for i, order_id in enumerate(drone['batch_orders']):
            order = self.orders.get(order_id)
            if order and order['status'] == OrderStatus.PICKED_UP:
                drone['current_delivery_index'] = i

                self._ensure_trip_started(drone)
                self._accumulate_trip_optimal_leg(drone, drone['location'], order['customer_location'])

                self.state_manager.update_drone_status(drone_id, DroneStatus.FLYING_TO_CUSTOMER,
                                                       target_location=order['customer_location'])
                return

        self._safe_reset_drone(drone_id, drone)

    def _handle_batch_delivery(self, drone_id, drone):
        if 'batch_orders' not in drone or not drone['batch_orders']:
            self._safe_reset_drone(drone_id, drone)
            return

        current_index = drone.get('current_delivery_index', 0)

        if current_index < len(drone['batch_orders']):
            order_id = drone['batch_orders'][current_index]
            order = self.orders.get(order_id)
            if order and order['status'] == OrderStatus.PICKED_UP:
                self._complete_order_delivery(order_id, drone_id)

        current_index += 1

        while current_index < len(drone['batch_orders']):
            order_id = drone['batch_orders'][current_index]
            order = self.orders.get(order_id)

            if order and order['status'] == OrderStatus.PICKED_UP:
                self._ensure_trip_started(drone)
                self._accumulate_trip_optimal_leg(drone, drone['location'], order['customer_location'])

                drone['current_delivery_index'] = current_index
                self.state_manager.update_drone_status(drone_id, DroneStatus.FLYING_TO_CUSTOMER,
                                                       target_location=order['customer_location'])
                return

            current_index += 1

        self._safe_reset_drone(drone_id, drone)

    def _handle_drone_arrival(self, drone_id, drone):
        """
        Arrival handler:
          - If planned_stops is present: execute interleaved multi-merchant pickup/delivery.
          - Else: fallback to legacy single/batch logic.
        """
        # -------- New route-plan logic (priority) --------
        if drone.get('planned_stops') and len(drone['planned_stops']) > 0:
            stop = drone['planned_stops'][0]
            stype = stop.get('type', None)

            if stype == 'P':
                self._execute_pickup_stop(drone_id, drone, stop)
                drone['planned_stops'].popleft()
                return self._set_next_target_from_plan(drone_id, drone)

            if stype == 'D':
                self._execute_delivery_stop(drone_id, drone, stop)
                drone['planned_stops'].popleft()
                return self._set_next_target_from_plan(drone_id, drone)

            # Unknown stop type -> drop and continue
            drone['planned_stops'].popleft()
            return self._set_next_target_from_plan(drone_id, drone)

        # -------- Legacy logic (unchanged) --------
        if drone['status'] == DroneStatus.FLYING_TO_MERCHANT:
            if 'batch_orders' in drone and drone['batch_orders']:
                self._handle_batch_pickup(drone_id, drone)
            else:
                assigned_order = self._get_drone_assigned_order(drone_id)
                if assigned_order and assigned_order['status'] == OrderStatus.ASSIGNED:
                    oid = assigned_order['id']
                    self.state_manager.update_order_status(oid, OrderStatus.PICKED_UP, reason="arrived_merchant_pickup")
                    assigned_order['pickup_time'] = self.time_system.current_step

                    self._ensure_trip_started(drone)
                    self._accumulate_trip_optimal_leg(drone, drone['location'], assigned_order['customer_location'])

                    self._start_new_task(drone_id, drone, assigned_order['customer_location'])
                    self.state_manager.update_drone_status(
                        drone_id, DroneStatus.FLYING_TO_CUSTOMER, target_location=assigned_order['customer_location']
                    )
                else:
                    self._safe_reset_drone(drone_id, drone)

        elif drone['status'] == DroneStatus.FLYING_TO_CUSTOMER:
            if 'batch_orders' in drone and drone['batch_orders']:
                self._handle_batch_delivery(drone_id, drone)
            else:
                assigned_order = self._get_drone_assigned_order(drone_id)
                if assigned_order and assigned_order['status'] == OrderStatus.PICKED_UP:
                    self._complete_order_delivery(assigned_order['id'], drone_id)
                self._safe_reset_drone(drone_id, drone)

        elif drone['status'] == DroneStatus.RETURNING_TO_BASE:
            if drone['battery_level'] < 80:
                self.state_manager.update_drone_status(drone_id, DroneStatus.CHARGING, target_location=None)
            else:
                self.state_manager.update_drone_status(drone_id, DroneStatus.IDLE, target_location=None)

            drone['current_load'] = 0
            self._clear_drone_batch_state(drone)
            self._clear_task_data(drone_id, drone)

            # --- ADD: clear route-plan state ---
            drone['planned_stops'] = deque()
            drone['cargo'] = set()
            drone['current_stop'] = None
            drone['route_committed'] = False

    # ------------------ 单段任务统计（保持原逻辑）------------------

    def _start_new_task(self, drone_id, drone, target_location):
        drone['task_start_location'] = drone['location']
        drone['task_start_step'] = self.time_system.current_step
        drone['current_task_distance'] = 0.0

        optimal_distance = math.sqrt(
            (target_location[0] - drone['location'][0]) ** 2 +
            (target_location[1] - drone['location'][1]) ** 2
        )
        drone['task_optimal_distance'] = optimal_distance

        self.daily_stats['optimal_flight_distance'] = self.daily_stats.get('optimal_flight_distance',
                                                                           0.0) + optimal_distance
        self.metrics['optimal_flight_distance'] = self.metrics.get('optimal_flight_distance', 0.0) + optimal_distance

        drone['target_location'] = target_location

    def _clear_task_data(self, drone_id, drone):
        keys_to_remove = [
            'task_start_location',
            'task_start_step',
            'current_task_distance',
            'task_optimal_distance',
            'last_location'
        ]
        for key in keys_to_remove:
            drone.pop(key, None)

    # ------------------ 订单完成/取消（统一走 StateManager）------------------

    def _complete_order_delivery(self, order_id, drone_id):
        if order_id not in self.orders:
            return

        order = self.orders[order_id]
        if order['status'] == OrderStatus.DELIVERED:
            return

        self.state_manager.update_order_status(order_id, OrderStatus.DELIVERED, reason=f"delivered_by_drone_{drone_id}")
        order['delivery_time'] = self.time_system.current_step
        order['assigned_drone'] = -1

        if drone_id in self.drones:
            drone = self.drones[drone_id]
            drone['orders_completed'] += 1
            drone['current_load'] = max(0, drone['current_load'] - 1)
            if 'batch_orders' in drone and order_id in drone['batch_orders']:
                drone['batch_orders'].remove(order_id)

        delivery_duration = order['delivery_time'] - order['creation_time']
        self.metrics['total_delivery_time'] += delivery_duration
        self.metrics['completed_orders'] += 1
        self.daily_stats['orders_completed'] += 1

        # Use READY-based SLA for on-time calculation and lateness tracking
        # Note: Lateness uses pure SLA (not timeout_factor), as timeout is for cancellation only
        # Lateness = delivery_time - (ready_step + sla_steps)
        # Use ready_step with creation_time as fallback if ready_step wasn't set
        ready_step = order.get('ready_step')
        if ready_step is None:
            ready_step = order['creation_time']

        delivery_lateness = order['delivery_time'] - ready_step - self._get_delivery_sla_steps(order)

        # Record lateness for diagnostics
        if 'ready_based_lateness_samples' not in self.metrics:
            self.metrics['ready_based_lateness_samples'] = []
        self.metrics['ready_based_lateness_samples'].append(delivery_lateness)

        if delivery_lateness <= 0:
            self.metrics['on_time_deliveries'] += 1
            self.daily_stats['on_time_deliveries'] += 1

        weather_key = f"{self.weather.name.lower()}_deliveries"
        if weather_key in self.metrics['weather_impact_stats']:
            self.metrics['weather_impact_stats'][weather_key] += 1

        self.active_orders.discard(order_id)
        self.completed_orders.add(order_id)

    def _cancel_order(self, order_id, reason):
        if order_id not in self.orders:
            return

        order = self.orders[order_id]
        if order['status'] == OrderStatus.CANCELLED:
            return

        self.state_manager.update_order_status(order_id, OrderStatus.CANCELLED, reason=reason)
        order['cancellation_reason'] = reason
        order['cancellation_time'] = self.time_system.current_step

        self.active_orders.discard(order_id)
        self.cancelled_orders.add(order_id)
        self.metrics['cancelled_orders'] += 1
        self.daily_stats['orders_cancelled'] += 1

        drone_id = order.get('assigned_drone', -1)
        if drone_id is not None and 0 <= drone_id < self.num_drones:
            drone = self.drones[drone_id]
            drone['current_load'] = max(0, drone['current_load'] - 1)
            if drone['current_load'] == 0:
                base_loc = self.bases[drone['base']]['location']
                self.state_manager.update_drone_status(drone_id, DroneStatus.RETURNING_TO_BASE,
                                                       target_location=base_loc)

    # ------------------ drone helper states ------------------

    def _get_drone_assigned_order(self, drone_id):
        for order_id in self.active_orders:
            order = self.orders[order_id]
            if order.get('assigned_drone') == drone_id:
                return order
        return None

    def _reset_drone_to_base(self, drone_id, drone):
        for order_id, order in list(self.orders.items()):
            if order.get('assigned_drone') == drone_id:
                if order['status'] == OrderStatus.ASSIGNED:
                    self._reset_order_to_ready(order_id, "reset_drone_to_base")
                elif order['status'] == OrderStatus.PICKED_UP:
                    self._complete_order_delivery(order_id, drone_id)

        if 'batch_orders' in drone:
            for batch_order_id in drone['batch_orders']:
                if batch_order_id in self.orders:
                    batch_order = self.orders[batch_order_id]
                    if batch_order['status'] == OrderStatus.ASSIGNED:
                        self._reset_order_to_ready(batch_order_id, "reset_drone_to_base_batch")
            del drone['batch_orders']

        drone.pop('current_batch_index', None)
        drone.pop('current_delivery_index', None)

        base_loc = self.bases[drone['base']]['location']
        self.state_manager.update_drone_status(drone_id, DroneStatus.RETURNING_TO_BASE, target_location=base_loc)
        drone['current_load'] = 0

    def _handle_waiting_pickup(self, drone_id, drone):
        assigned_order = self._get_drone_assigned_order(drone_id)

        if assigned_order:
            if assigned_order['status'] == OrderStatus.READY:
                self.state_manager.update_order_status(
                    assigned_order['id'], OrderStatus.ASSIGNED, reason="waiting_to_assigned"
                )
                assigned_order['assigned_drone'] = drone_id

            if assigned_order['status'] == OrderStatus.ASSIGNED:
                self.state_manager.update_order_status(
                    assigned_order['id'], OrderStatus.PICKED_UP, reason="pickup_complete"
                )
                assigned_order['pickup_time'] = self.time_system.current_step

                self.state_manager.update_drone_status(
                    drone_id, DroneStatus.FLYING_TO_CUSTOMER, target_location=assigned_order['customer_location']
                )
                return

        if 'waiting_start_time' not in drone:
            drone['waiting_start_time'] = self.time_system.current_step

        waiting_duration = self.time_system.current_step - drone['waiting_start_time']
        if waiting_duration > 10:
            if assigned_order:
                self._cancel_order(assigned_order['id'], "waiting_timeout")
            self._reset_drone_to_base(drone_id, drone)

    def _handle_delivering(self, drone_id, drone):
        assigned_order = self._get_drone_assigned_order(drone_id)
        if assigned_order and assigned_order['status'] == OrderStatus.PICKED_UP:
            self._complete_order_delivery(assigned_order['id'], drone_id)
        self._reset_drone_to_base(drone_id, drone)

    def _handle_charging(self, drone_id, drone):
        if drone['battery_level'] < 95:
            drone['battery_level'] = min(
                drone['max_battery'],
                drone['battery_level'] + drone['charging_rate']
            )
        else:
            self.state_manager.update_drone_status(drone_id, DroneStatus.IDLE, target_location=None)

    # ------------------ random events ------------------

    def _handle_random_events(self):
        cancellation_factor = self._get_weather_cancellation_factor()

        for order_id in list(self.active_orders):
            order = self.orders[order_id]

            if (order['status'] in [OrderStatus.PENDING, OrderStatus.ACCEPTED] and
                    random.random() < 0.02 * cancellation_factor):
                self._cancel_order(order_id, "user_cancellation")

            elif (order['status'] == OrderStatus.ACCEPTED and
                  random.random() < self.merchants[order['merchant_id']]['cancellation_rate']):
                self._cancel_order(order_id, "merchant_cancellation")

            elif (order['status'] == OrderStatus.ASSIGNED and
                  random.random() < self.drones[order['assigned_drone']]['cancellation_rate'] * cancellation_factor):
                self._cancel_order(order_id, "drone_cancellation")

            elif (order['status'] in [OrderStatus.ACCEPTED, OrderStatus.ASSIGNED] and
                  random.random() < 0.01):
                self._change_order_address(order_id)

    def _get_weather_cancellation_factor(self):
        if self.weather == WeatherType.SUNNY:
            return 1.0
        elif self.weather == WeatherType.RAINY:
            return 1.5
        elif self.weather == WeatherType.WINDY:
            return 1.3
        else:
            return 2.0

    def _change_order_address(self, order_id):
        order = self.orders[order_id]
        new_customer_location = self.location_loader.get_random_user_grid_location()
        order['customer_location'] = new_customer_location

        if order.get('assigned_drone', -1) >= 0 and order['status'] == OrderStatus.ASSIGNED:
            drone_id = order['assigned_drone']
            drone = self.drones[drone_id]
            if drone['status'] == DroneStatus.FLYING_TO_CUSTOMER:
                drone['target_location'] = new_customer_location

    # ------------------ air traffic / weather ------------------

    def _update_air_traffic(self):
        self.air_traffic = np.zeros((self.grid_size, self.grid_size))

        for drone_id, drone in self.drones.items():
            if drone['status'] not in [DroneStatus.IDLE, DroneStatus.CHARGING]:
                x, y = drone['location']
                x_int = int(x)
                y_int = int(y)
                if 0 <= x_int < self.grid_size and 0 <= y_int < self.grid_size:
                    self.air_traffic[x_int, y_int] += 0.3

                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            nx, ny = x_int + dx, y_int + dy
                            if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                                self.air_traffic[nx, ny] += 0.1

        self.air_traffic = np.clip(self.air_traffic, 0, 1)

    def _update_weather_from_dataset(self):
        """用 step 做索引（全局）"""
        try:
            current_weather = self.weather_processor.get_weather_at_time(self.time_system.current_step)
            self.weather = self.weather_processor.map_to_weather_type(current_weather['Summary'])

            self.weather_details = {
                'summary': current_weather.get('Summary', 'Unknown'),
                'temperature': current_weather.get('Temperature (C)', 15),
                'humidity': current_weather.get('Humidity', 0.5),
                'wind_speed': current_weather.get('Wind Speed (km/h)', 10),
                'visibility': current_weather.get('Visibility (km)', 10),
                'pressure': current_weather.get('Pressure (millibars)', 1013),
                'precip_type': current_weather.get('Precip Type', 'none')
            }
        except Exception as e:
            print(f"更新天气数据失败: {e}，使用默认天气")
            self.weather = WeatherType.SUNNY
            self.weather_details = {
                'summary': 'Sunny',
                'temperature': 20,
                'humidity': 0.5,
                'wind_speed': 5,
                'visibility': 15,
                'pressure': 1013,
                'precip_type': 'none'
            }

        self.weather_history.append({
            'time': self.time_system.current_step,
            'weather': self.weather,
            'details': self.weather_details.copy()
        })

    def _get_weather_speed_factor(self):
        if self.weather == WeatherType.SUNNY:
            return 1.0
        elif self.weather == WeatherType.RAINY:
            return 0.7
        elif self.weather == WeatherType.WINDY:
            return 0.6
        else:
            return 0.3

    def _get_weather_battery_factor(self):
        if self.weather == WeatherType.SUNNY:
            return 1.0
        elif self.weather == WeatherType.RAINY:
            return 1.3
        elif self.weather == WeatherType.WINDY:
            return 1.6
        else:
            return 2.0

    # ------------------ order generation ------------------

    def _generate_new_orders(self):
        time_state = self.time_system.get_time_state()
        if not time_state['is_business_hours']:
            return

        order_prob = self.order_processor.get_order_probability(
            env_time=self.time_system.current_step,
            weather_type=self.weather
        )

        if random.random() < order_prob:
            if order_prob > 0.8:
                base_batch = random.randint(3, 6)
            elif order_prob > 0.5:
                base_batch = random.randint(2, 4)
            else:
                base_batch = random.randint(1, 2)

            batch_size = int(base_batch * self.high_load_factor)

            if time_state['is_peak_hour']:
                batch_size += random.randint(1, 3)

            for _ in range(batch_size):
                self._generate_single_order()

    def _generate_single_order(self):
        try:
            order_id = self.global_order_counter
            self.global_order_counter += 1

            env_time = self.time_system.day_number * 24 + self.time_system.current_hour
            order_details = self.order_processor.generate_order_details(env_time, self.weather)

            if order_details['merchant_id'] not in self.merchant_ids:
                order_details['merchant_id'] = random.choice(self.merchant_ids)

            self._generate_order_with_details(order_details, order_id)

        except Exception as e:
            print(f"生成订单时出错: {e}")

    def _generate_order_with_details(self, order_details, order_id):
        try:
            merchant_id = order_details['merchant_id']
            if merchant_id not in self.merchants:
                merchant_id = random.choice(self.merchant_ids)

            merchant_loc = self.merchants[merchant_id]['location']
            max_distance = order_details['max_distance']
            customer_loc = self._generate_customer_location(merchant_loc, max_distance)

            if random.random() < 0.3:
                customer_loc = self._generate_distant_location(merchant_loc)

            weather_summary = self.weather_details.get('summary', 'Unknown')
            preparation_time = int(order_details.get('preparation_time', random.randint(2, 6)))

            order = {
                'id': order_id,
                'order_type': OrderType(order_details['order_type']),
                'merchant_id': merchant_id,
                'merchant_location': merchant_loc,
                'customer_location': customer_loc,
                'status': OrderStatus.PENDING,
                'creation_time': self.time_system.current_step,
                'assigned_drone': -1,
                'preparation_time': preparation_time,  # step
                'urgent': random.random() < order_details['urgency'],
                'weather_conditions': weather_summary
            }

            self.orders[order_id] = order
            self.active_orders.add(order_id)
            self.metrics['total_orders'] += 1
            self.daily_stats['orders_generated'] += 1

            self.merchants[merchant_id]['queue'].append(order_id)

            # PENDING -> ACCEPTED
            self.state_manager.update_order_status(order_id, OrderStatus.ACCEPTED, reason="order_created_and_accepted")

            self.order_history.append({
                'time': self.time_system.current_step,
                'weather': weather_summary,
                'order_type': order_details['order_type'],
                'distance': self._calculate_euclidean_distance(merchant_loc, customer_loc),
                'merchant_id': merchant_id
            })

        except Exception as e:
            print(f"生成订单详细时出错: {e}")

    def _generate_distant_location(self, merchant_loc):
        merchant_x, merchant_y = merchant_loc
        edge = random.choice(['top', 'bottom', 'left', 'right'])
        if edge == 'top':
            return random.uniform(0, self.grid_size - 1), self.grid_size - 1
        elif edge == 'bottom':
            return random.uniform(0, self.grid_size - 1), 0
        elif edge == 'left':
            return 0, random.uniform(0, self.grid_size - 1)
        else:
            return self.grid_size - 1, random.uniform(0, self.grid_size - 1)

    def _generate_customer_location(self, merchant_loc, max_distance):
        for _ in range(10):
            customer_loc = self.location_loader.get_random_user_grid_location()
            distance = self._calculate_euclidean_distance(merchant_loc, customer_loc)
            if distance <= max_distance:
                return customer_loc

        merchant_x, merchant_y = merchant_loc
        return (
            max(0, min(self.grid_size - 1, merchant_x + random.randint(-2, 2))),
            max(0, min(self.grid_size - 1, merchant_y + random.randint(-2, 2)))
        )

    def _calculate_euclidean_distance(self, loc1, loc2):
        return math.sqrt((loc1[0] - loc2[0]) ** 2 + (loc1[1] - loc2[1]) ** 2)

    # ------------------ stale assignment cleanup ------------------

    def _cleanup_stale_assignments(self):
        current_step = self.time_system.current_step
        stale_threshold = 50

        for order_id, order in list(self.orders.items()):
            # Check for READY-based timeout cancellation
            if order['status'] in [OrderStatus.READY, OrderStatus.ASSIGNED, OrderStatus.PICKED_UP]:
                deadline_step = self._get_delivery_deadline_step(order)
                if current_step > deadline_step:
                    self._cancel_order(order_id, "ready_based_timeout")
                    continue

            if order['status'] == OrderStatus.ASSIGNED:
                drone_id = order.get('assigned_drone', -1)

                if drone_id < 0 or drone_id not in self.drones:
                    self._reset_order_to_ready(order_id, "stale_invalid_drone")
                    continue

                drone = self.drones[drone_id]

                if drone['status'] in [DroneStatus.IDLE, DroneStatus.CHARGING]:
                    age = current_step - order['creation_time']
                    if age > stale_threshold:
                        self._reset_order_to_ready(order_id, "stale_age")

    # ------------------ end-of-day ------------------

    def _handle_end_of_day(self):
        print(f"\n=== 结束 ===")
        print(f"今日统计:")
        print(f"  生成订单: {self.daily_stats['orders_generated']}")
        print(f"  完成订单: {self.daily_stats['orders_completed']}")
        print(f"  取消订单: {self.daily_stats['orders_cancelled']}")
        print(f"  准时交付: {self.daily_stats['on_time_deliveries']}")

        unfinished_orders = list(self.active_orders)
        for order_id in unfinished_orders:
            self._cancel_order(order_id, "end_of_day")

        for drone_id, drone in self.drones.items():
            if drone['status'] not in [DroneStatus.IDLE, DroneStatus.CHARGING]:
                base_loc = self.bases[drone['base']]['location']
                self.state_manager.update_drone_status(drone_id, DroneStatus.RETURNING_TO_BASE,
                                                       target_location=base_loc)

        print(f"  未完成订单: {len(unfinished_orders)} 已取消")
        print("=== 当日营业结束 ===\n")

    # ------------------ observation encoding ------------------

    def _get_observation(self):
        order_obs = np.zeros((self.max_obs_orders, 10), dtype=np.float32)

        active_ids = list(self.active_orders)

        def sort_key(oid):
            o = self.orders[oid]
            return (not o.get('urgent', False), o['creation_time'])

        active_ids.sort(key=sort_key)
        self.current_obs_order_ids = active_ids[:self.max_obs_orders]

        for i, order_id in enumerate(self.current_obs_order_ids):
            order = self.orders[order_id]
            order_obs[i] = self._encode_order(order)

        drone_obs = np.zeros((self.num_drones, 8), dtype=np.float32)
        for drone_id, drone in self.drones.items():
            drone_obs[drone_id] = self._encode_drone(drone)

        # Top-K merchants
        merchant_obs = np.zeros((self.obs_num_merchants, 4), dtype=np.float32)
        obs_merchant_ids = self._select_topk_merchants_for_observation()
        for i, merchant_id in enumerate(obs_merchant_ids):
            merchant = self.merchants[merchant_id]
            merchant_obs[i] = self._encode_merchant(merchant)

        q_cap = max(getattr(self, "max_queue_cap", 50.0), 1e-6)
        e_cap = max(getattr(self, "max_eff_cap", 2.0), 1e-6)
        merchant_obs[:, 0] = np.clip(merchant_obs[:, 0] / q_cap, 0.0, 1.0)
        merchant_obs[:, 1] = np.clip(merchant_obs[:, 1] / e_cap, 0.0, 1.0)
        merchant_obs[:, 2] = np.clip(merchant_obs[:, 2], 0.0, 1.0)
        merchant_obs[:, 3] = (merchant_obs[:, 3] > 0).astype(np.float32)

        base_obs = np.zeros((self.obs_num_bases, 3), dtype=np.float32)
        for base_id in range(self.obs_num_bases):
            base = self.bases[base_id]
            base_obs[base_id] = self._encode_base(base)

        weather_details = np.zeros(5, dtype=np.float32)
        if self.weather_details:
            temp = self.weather_details.get('temperature', 15)
            weather_details[0] = (temp + 10) / 50
            humidity = self.weather_details.get('humidity', 0.5)
            weather_details[1] = humidity
            wind_speed = self.weather_details.get('wind_speed', 10)
            weather_details[2] = wind_speed / 100
            visibility = self.weather_details.get('visibility', 10)
            weather_details[3] = visibility / 20
            pressure = self.weather_details.get('pressure', 1013)
            weather_details[4] = (pressure - 980) / 60
        weather_details = np.clip(weather_details, 0.0, 1.0).astype(np.float32)

        time_state = self.time_system.get_time_state()
        time_obs = np.array([
            time_state['hour'] / 24.0,
            time_state['minute'] / 60.0,
            time_state['progress'],
            1.0 if time_state['is_peak_hour'] else 0.0,
            1.0 if time_state['is_business_hours'] else 0.0
        ], dtype=np.float32)

        order_pattern = np.asarray(self.order_processor.order_patterns['hourly_pattern'], dtype=np.float32)
        order_pattern = np.clip(order_pattern, 0.0, 1.0)

        pareto_info = np.zeros(self.num_objectives + 2, dtype=np.float32)
        if len(self.pareto_optimizer.pareto_front) > 0:
            pareto_front = self.pareto_optimizer.get_pareto_front()
            pareto_mean = np.mean(pareto_front, axis=0)
            pareto_info[:self.num_objectives] = pareto_mean
            reference_point = np.ones(self.num_objectives, dtype=np.float32) * 0.5
            pareto_info[self.num_objectives] = self.pareto_optimizer.calculate_hypervolume(reference_point)
            pareto_info[self.num_objectives + 1] = self.pareto_optimizer.get_diversity()
        pareto_info = np.nan_to_num(pareto_info, nan=0.0, posinf=1.0, neginf=0.0)
        pareto_info = np.clip(pareto_info, 0.0, 1.0).astype(np.float32)

        order_obs = np.clip(order_obs, 0.0, 1.0).astype(np.float32)
        drone_obs = np.clip(drone_obs, 0.0, 1.0).astype(np.float32)
        base_obs = np.clip(base_obs, 0.0, 1.0).astype(np.float32)
        air_traffic = np.clip(self.air_traffic, 0.0, 1.0).astype(np.float32)

        return {
            'orders': order_obs,
            'drones': drone_obs,
            'merchants': merchant_obs,
            'bases': base_obs,
            'weather': int(self.weather.value),
            'weather_details': weather_details,
            'time': time_obs,
            'day_progress': np.array([time_state['progress']], dtype=np.float32),
            'resource_saturation': np.array([self._calculate_resource_saturation()], dtype=np.float32),
            'air_traffic': air_traffic,
            'order_pattern': order_pattern,
            'pareto_info': pareto_info
        }

    def _encode_order(self, order):
        encoding = np.zeros(10, dtype=np.float32)
        encoding[order['status'].value] = 1.0
        encoding[6] = order['order_type'].value / 2.0
        encoding[7] = (self.time_system.current_step - order['creation_time']) / 50.0

        assigned_drone = order.get('assigned_drone', -1)
        if assigned_drone is None:
            assigned_drone = -1
        encoding[8] = assigned_drone / max(1, self.num_drones)

        encoding[9] = 1.0 if order.get('urgent', False) else 0.0
        return encoding

    def _encode_drone(self, drone):
        encoding = np.zeros(8, dtype=np.float32)

        encoding[0] = drone['status'].value / 7.0
        encoding[1] = drone['location'][0] / self.grid_size
        encoding[2] = drone['location'][1] / self.grid_size

        if 'target_location' in drone:
            encoding[3] = drone['target_location'][0] / self.grid_size
            encoding[4] = drone['target_location'][1] / self.grid_size

        encoding[6] = drone['current_load'] / max(1, drone['max_capacity'])
        encoding[7] = drone['battery_level'] / max(1.0, drone['max_battery'])
        return encoding

    def _encode_merchant(self, merchant):
        encoding = np.zeros(4, dtype=np.float32)
        encoding[0] = len(merchant['queue'])  # 先用原始值，后面统一归一化
        encoding[1] = merchant['efficiency']
        encoding[2] = merchant['cancellation_rate']
        encoding[3] = 1.0 if merchant['landing_zone'] else 0.0
        return encoding

    def _encode_base(self, base):
        encoding = np.zeros(3, dtype=np.float32)
        encoding[0] = len(base['drones_available']) / max(1, base['capacity'])
        encoding[1] = base['charging_stations'] / 5.0
        encoding[2] = len(base['charging_queue']) / 5.0
        return encoding

    # ------------------ info/report ------------------

    def _calculate_resource_saturation(self):
        busy_drones = sum(1 for d in self.drones.values()
                          if d['status'] not in [DroneStatus.IDLE, DroneStatus.CHARGING])
        return busy_drones / max(1, self.num_drones)

    def _monitor_drone_status(self):
        pass

    def _update_system_state(self):
        pass

    def _get_info(self):
        info = {
            'metrics': self.metrics.copy(),
            'daily_stats': self.daily_stats.copy(),
            'resource_saturation': self._calculate_resource_saturation(),
            'weather_impact_stats': self.metrics['weather_impact_stats'].copy(),
            'current_weather': self.weather_details.copy(),
            'pareto_front_size': len(self.pareto_optimizer.pareto_front),
            'pareto_hypervolume': self.pareto_optimizer.calculate_hypervolume(np.ones(self.num_objectives) * 0.5),
            'pareto_diversity': self.pareto_optimizer.get_diversity(),
            'order_history_summary': {
                'total_orders': len(self.order_history),
                'unique_merchants': len(set([o['merchant_id'] for o in self.order_history])),
                'avg_distance': np.mean([o['distance'] for o in self.order_history]) if self.order_history else 0
            },
            'time_state': self.time_system.get_time_state(),
            'backlog_size': len(self.active_orders)
        }

        if self.daily_stats['orders_completed'] > 0:
            info['avg_delivery_time'] = self.metrics['total_delivery_time'] / self.daily_stats['orders_completed']
            info['energy_efficiency'] = self.daily_stats['energy_consumed'] / self.daily_stats['orders_completed']
            info['on_time_rate'] = self.daily_stats['on_time_deliveries'] / self.daily_stats['orders_completed']
            info['distance_efficiency'] = self.daily_stats['optimal_flight_distance'] / max(
                self.daily_stats['total_flight_distance'], 0.1
            )
            info['avg_distance_per_order'] = self.daily_stats['total_flight_distance'] / self.daily_stats[
                'orders_completed']
        else:
            info['avg_delivery_time'] = 0
            info['energy_efficiency'] = 0
            info['on_time_rate'] = 0
            info['distance_efficiency'] = 0
            info['avg_distance_per_order'] = 0

        return info

    def get_daily_report(self):
        time_state = self.time_system.get_time_state()

        report = {
            'day_number': self.time_system.day_number,
            'current_time': f"{time_state['hour']:02d}:{time_state['minute']:02d}",
            'weather': self.weather_details.get('summary', 'Unknown'),
            'order_stats': {
                'generated': self.daily_stats['orders_generated'],
                'completed': self.daily_stats['orders_completed'],
                'cancelled': self.daily_stats['orders_cancelled'],
                'active': len(self.active_orders),
                'completion_rate': self.daily_stats['orders_completed'] / max(1, self.daily_stats['orders_generated'])
            },
            'performance_metrics': {
                'on_time_rate': self.daily_stats['on_time_deliveries'] / max(1, self.daily_stats['orders_completed']),
                'energy_efficiency': self.daily_stats['energy_consumed'] / max(1, self.daily_stats['orders_completed']),
                'resource_utilization': self._calculate_resource_saturation(),
                'distance_efficiency': self.daily_stats['optimal_flight_distance'] / max(
                    self.daily_stats['total_flight_distance'], 0.1
                ),
                'total_flight_distance': self.daily_stats['total_flight_distance'],
                'optimal_flight_distance': self.daily_stats['optimal_flight_distance'],
            },
            'drone_status': self._get_drone_status_summary()
        }

        return report

    def _get_drone_status_summary(self):
        status_count = {
            'idle': 0,
            'assigned': 0,
            'flying_to_merchant': 0,
            'waiting_for_pickup': 0,
            'flying_to_customer': 0,
            'delivering': 0,
            'returning_to_base': 0,
            'charging': 0
        }

        for drone in self.drones.values():
            status = drone['status']
            if status == DroneStatus.IDLE:
                status_count['idle'] += 1
            elif status == DroneStatus.ASSIGNED:
                status_count['assigned'] += 1
            elif status == DroneStatus.FLYING_TO_MERCHANT:
                status_count['flying_to_merchant'] += 1
            elif status == DroneStatus.WAITING_FOR_PICKUP:
                status_count['waiting_for_pickup'] += 1
            elif status == DroneStatus.FLYING_TO_CUSTOMER:
                status_count['flying_to_customer'] += 1
            elif status == DroneStatus.DELIVERING:
                status_count['delivering'] += 1
            elif status == DroneStatus.RETURNING_TO_BASE:
                status_count['returning_to_base'] += 1
            elif status == DroneStatus.CHARGING:
                status_count['charging'] += 1

        return status_count

    # ------------------ Snapshot interfaces for MOPSO ------------------

    def get_ready_orders_snapshot(self, limit: int = 200) -> List[dict]:
        """
        Get snapshot of READY orders for MOPSO scheduling.
        Returns list of order dicts with essential fields.
        """
        ready_orders = []
        for oid in self.active_orders:
            if oid not in self.orders:
                continue
            order = self.orders[oid]
            if order['status'] != OrderStatus.READY:
                continue
            if order.get('assigned_drone', -1) not in (-1, None):
                continue

            # Create snapshot with essential fields
            snapshot = {
                'order_id': oid,
                'merchant_id': order['merchant_id'],
                'merchant_location': order['merchant_location'],
                'customer_location': order['customer_location'],
                'creation_time': order['creation_time'],
                'deadline_step': self._get_delivery_deadline_step(order),
                'urgent': order.get('urgent', False),
                'distance': order.get('distance', 0.0),
            }
            ready_orders.append(snapshot)

            if len(ready_orders) >= limit:
                break

        return ready_orders

    def get_drones_snapshot(self) -> List[dict]:
        """
        Get snapshot of all drones for MOPSO scheduling.
        Returns list of drone dicts with essential fields.
        """
        drones_snapshot = []
        for drone_id, drone in self.drones.items():
            snapshot = {
                'drone_id': drone_id,
                'location': drone['location'],
                'base': drone['base'],
                'status': drone['status'],
                'battery_level': drone['battery_level'],
                'current_load': drone['current_load'],
                'max_capacity': drone['max_capacity'],
                'speed': drone['speed'],
                'battery_consumption_rate': drone['battery_consumption_rate'],
                'has_route': drone.get('route_committed', False),
            }
            drones_snapshot.append(snapshot)

        return drones_snapshot

    def get_merchants_snapshot(self) -> Dict[str, dict]:
        """
        Get snapshot of all merchants for MOPSO scheduling.
        Returns dict mapping merchant_id to merchant info.
        """
        merchants_snapshot = {}
        for merchant_id, merchant in self.merchants.items():
            snapshot = {
                'merchant_id': merchant_id,
                'location': merchant['location'],
                'queue_length': len(merchant.get('queue', [])),
                'cancellation_rate': merchant.get('cancellation_rate', 0.01),
                'landing_zone': merchant.get('landing_zone', True),
            }
            merchants_snapshot[merchant_id] = snapshot

        return merchants_snapshot

    def get_route_plan_constraints(self) -> dict:
        """
        Get constraints for route plan generation.
        Returns dict with constraint parameters.
        """
        constraints = {
            'grid_size': self.grid_size,
            'current_step': self.time_system.current_step,
            'max_capacity_per_drone': self.drone_max_capacity,
            'weather_speed_factor': self._get_weather_speed_factor(),
            'weather_battery_factor': self._get_weather_battery_factor(),
        }
        return constraints
