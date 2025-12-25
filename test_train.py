# ============================================
# train_morl_mpso_ppo.py
# 真多目标（路线1 Policy Set）：
#   - MPSO：订单分配（多目标）
#   - PPO：路径规划 heading
#   - 环境 reward_output_mode="zero"
# ============================================

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from UAV_ENVIRONMENT_5_fixed import ThreeObjectiveDroneDeliveryEnv  # 确保可 import


# -----------------------------
# 工具：距离
# -----------------------------
def euclid(a, b) -> float:
    return float(math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2))


# -----------------------------
# 1) 多目标 MPSO 订单分配（你没有MPSO：这里给出可用实现）
# -----------------------------
@dataclass
class Particle:
    x: np.ndarray  # position: shape (n_orders,) each in [0, n_drones] 0=不分配, k=分配给 drone k-1
    v: np.ndarray  # velocity: same shape
    pbest_x: np.ndarray
    pbest_f: np.ndarray  # vector objective (3,)


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    return np.all(a >= b) and np.any(a > b)


def scalarize(r_vec: np.ndarray, mode: str, w: np.ndarray, z: Optional[np.ndarray] = None) -> float:
    """
    将 3维向量奖励 r_vec 标量化为 PPO 更新所需的标量 reward。
    mode:
      - "ws": weighted sum
      - "tcheby": Tchebycheff (maximize min_i w_i*(r_i - z_i))
    """
    r = r_vec.astype(np.float32)
    w = w.astype(np.float32)

    if mode == "ws":
        return float(np.dot(w, r))

    if mode == "tcheby":
        if z is None:
            # 若未提供参考点，默认用 0 向量（你也可以换成历史 ideal point）
            z = np.zeros_like(r, dtype=np.float32)
        z = z.astype(np.float32)
        return float(np.min(w * (r - z)))

    raise ValueError(f"Unknown scalarization mode: {mode}")


def pareto_filter(items: List[Tuple[np.ndarray, np.ndarray]]) -> List[Tuple[np.ndarray, np.ndarray]]:
    # items: [(x, f_vec)]
    front = []
    for i, (x_i, f_i) in enumerate(items):
        dominated_flag = False
        for j, (x_j, f_j) in enumerate(items):
            if j == i:
                continue
            if dominates(f_j, f_i):
                dominated_flag = True
                break
        if not dominated_flag:
            front.append((x_i, f_i))
    return front


class MPSOScheduler:
    """
    Multi-objective PSO for assignment:
      For the current observation window (READY orders), decide assignment to drones.

    Particle encoding:
      x[j] in [0..num_drones], float; decoded by round+clip to int:
        0 => not assign
        k => assign order_j to drone k-1
    Constraints handled in decoding:
      - order must be READY, unassigned
      - drone battery>10
      - drone capacity not exceeded
      - allow assigning to busy drones (batching) (符合你环境设定)
    """

    def __init__(
            self,
            n_particles: int = 30,
            iters: int = 30,
            inertia: float = 0.6,
            c1: float = 1.4,
            c2: float = 1.4,
            max_assign_per_drone: int = 4,
            seed: int = 0,
    ):
        self.n_particles = int(n_particles)
        self.iters = int(iters)
        self.w = float(inertia)
        self.c1 = float(c1)
        self.c2 = float(c2)
        self.max_assign_per_drone = int(max_assign_per_drone)
        self.rng = np.random.RandomState(seed)

        self.archive: List[Tuple[np.ndarray, np.ndarray]] = []  # [(x, f_vec)]

    def _get_candidate_orders(self, env, max_orders: int) -> List[int]:
        # env.current_obs_order_ids 是观测窗口（按紧急、创建时间排序）
        cand = []
        for oid in env.current_obs_order_ids[:max_orders]:
            o = env.orders.get(oid)
            if o is None:
                continue
            if o["status"] != env.orders[oid]["status"]:  # no-op，防御
                pass
            if o["status"].name == "READY" and (o.get("assigned_drone", -1) in (-1, None)):
                cand.append(oid)
        return cand

    def _get_candidate_drones(self, env) -> List[int]:
        drones = []
        for d_id, d in env.drones.items():
            if d["battery_level"] <= 10:
                continue
            if d["current_load"] >= d["max_capacity"]:
                continue
            drones.append(d_id)
        return drones

    def _decode(self, env, order_ids: List[int], drone_ids: List[int], x: np.ndarray) -> Dict[int, List[int]]:
        """
        将 particle position x 解码成 assignment dict:
          {drone_id: [order_id,...]}
        """
        if len(order_ids) == 0 or len(drone_ids) == 0:
            return {}

        # x -> int in [0..len(drone_ids)]
        a = np.rint(x).astype(int)
        a = np.clip(a, 0, len(drone_ids))

        assign: Dict[int, List[int]] = {d: [] for d in drone_ids}
        # 对每个订单决定给哪个 drone
        for j, oid in enumerate(order_ids):
            k = a[j]
            if k <= 0:
                continue
            d = drone_ids[k - 1]
            assign[d].append(oid)

        # 约束：每个 drone 的容量 / max_assign_per_drone
        final_assign: Dict[int, List[int]] = {}
        for d in drone_ids:
            if not assign[d]:
                continue
            drone = env.drones[d]
            can_take = max(0, drone["max_capacity"] - drone["current_load"])
            cap = min(can_take, self.max_assign_per_drone)
            if cap <= 0:
                continue

            # 优先紧急、等待久的订单
            assign[d].sort(key=lambda oid: (not env.orders[oid].get("urgent", False),
                                            env.orders[oid]["creation_time"]))
            final_assign[d] = assign[d][:cap]

        return final_assign

    def _evaluate(self, env, order_ids: List[int], drone_ids: List[int], x: np.ndarray) -> np.ndarray:
        """
        评价粒子方案的 3 目标（越大越好）：
          f0 吞吐 proxy: 分配的订单数（紧急加权）
          f1 -成本: 负的预计距离/能耗 proxy
          f2 服务质量: 紧急/等待时间奖励 - 超远距离惩罚
        """
        assign = self._decode(env, order_ids, drone_ids, x)
        if not assign:
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)

        f0 = 0.0
        dist_sum = 0.0
        f2 = 0.0

        for d, oids in assign.items():
            drone = env.drones[d]
            dloc = drone["location"]
            for oid in oids:
                o = env.orders[oid]
                mloc = o["merchant_location"]
                cloc = o["customer_location"]
                # 距离 proxy：drone->merchant + merchant->customer
                leg = euclid(dloc, mloc) + euclid(mloc, cloc)
                dist_sum += leg

                # 吞吐：每单1分，紧急+0.5
                f0 += 1.0 + (0.5 if o.get("urgent", False) else 0.0)

                # 服务质量：等待越久越重要（但别爆）
                wait = float(env.time_system.current_step - o["creation_time"])
                f2 += 0.01 * min(wait, 200.0)
                if o.get("urgent", False):
                    f2 += 0.5

                # 超远单惩罚（服务）
                if leg > env.grid_size * 2.5:
                    f2 -= 0.5

        # 成本目标（越大越好）：-distance
        f1 = -0.05 * dist_sum

        return np.array([f0, f1, f2], dtype=np.float32)

    def solve(self, env, max_orders: int = 50) -> Dict[int, List[int]]:
        """
        对当前 env 的 READY 订单进行一次 MPSO 求解，返回 assignment。
        """
        order_ids = self._get_candidate_orders(env, max_orders=max_orders)
        drone_ids = self._get_candidate_drones(env)
        if len(order_ids) == 0 or len(drone_ids) == 0:
            return {}

        dim = len(order_ids)
        # 初始化粒子
        particles: List[Particle] = []
        for _ in range(self.n_particles):
            x = self.rng.uniform(0, len(drone_ids), size=(dim,)).astype(np.float32)
            v = self.rng.normal(0, 0.5, size=(dim,)).astype(np.float32)
            f = self._evaluate(env, order_ids, drone_ids, x)
            particles.append(Particle(x=x, v=v, pbest_x=x.copy(), pbest_f=f.copy()))

        # 初始化 archive
        self.archive = pareto_filter([(p.x.copy(), p.pbest_f.copy()) for p in particles])

        def pick_gbest() -> np.ndarray:
            # 从 archive 随机挑一个作为全局引导（保持多样性）
            if not self.archive:
                return particles[self.rng.randint(len(particles))].x
            return self.archive[self.rng.randint(len(self.archive))][0]

        # 迭代
        for _ in range(self.iters):
            for p in particles:
                gbest_x = pick_gbest()

                r1 = self.rng.rand(dim).astype(np.float32)
                r2 = self.rng.rand(dim).astype(np.float32)

                p.v = self.w * p.v + self.c1 * r1 * (p.pbest_x - p.x) + self.c2 * r2 * (gbest_x - p.x)
                p.x = p.x + p.v
                p.x = np.clip(p.x, 0.0, float(len(drone_ids)))

                f = self._evaluate(env, order_ids, drone_ids, p.x)

                # 更新 pbest：用 Pareto（如果新解支配旧 pbest，就更新；若互不支配，也可按随机/拥挤度更新）
                if dominates(f, p.pbest_f):
                    p.pbest_f = f.copy()
                    p.pbest_x = p.x.copy()

            # 更新 archive（外部档案）
            candidates = self.archive + [(p.x.copy(), p.pbest_f.copy()) for p in particles]
            self.archive = pareto_filter(candidates)

            # 控制 archive 大小（可选）：随机裁剪
            if len(self.archive) > 80:
                self.archive = [self.archive[i] for i in self.rng.choice(len(self.archive), size=80, replace=False)]

        # 从 archive 里选一个“折中解”输出：
        # 这里用简单均衡权重 w=[1,1,1] 的加权和挑一个（仅用于输出，不影响“真多目标训练”）
        best = None
        best_score = -1e9
        for x, f in self.archive:
            score = float(f[0] + f[1] + f[2])
            if score > best_score:
                best_score = score
                best = x
        if best is None:
            best = particles[0].x

        return self._decode(env, order_ids, drone_ids, best)


def apply_mpso(env, scheduler: MPSOScheduler):
    assignments = scheduler.solve(env, max_orders=min(50, env.max_obs_orders))
    for d, oids in assignments.items():
        if oids:
            env._process_batch_assignment(int(d), list(oids))


# -----------------------------
# 2) PPO（heading）最小实现
# -----------------------------
def flatten_obs(obs: Dict[str, np.ndarray]) -> np.ndarray:
    parts = []
    # 为路径规划选取关键观测（避免 merchants 太大）
    for key in ["drones", "orders", "weather_details", "time", "day_progress", "resource_saturation", "air_traffic"]:
        parts.append(obs[key].reshape(-1).astype(np.float32))
    return np.concatenate(parts, axis=0)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh()
        )
        self.mu = nn.Linear(hidden, act_dim)
        self.v = nn.Linear(hidden, 1)
        self.log_std = nn.Parameter(torch.ones(act_dim) * -0.5)

    def forward(self, obs_t: torch.Tensor):
        h = self.net(obs_t)
        mu = self.mu(h)
        v = self.v(h).squeeze(-1)
        std = torch.exp(self.log_std)
        return mu, std, v

    def act(self, obs_t: torch.Tensor):
        mu, std, v = self.forward(obs_t)
        dist = torch.distributions.Normal(mu, std)
        a = dist.sample()
        logp = dist.log_prob(a).sum(-1)
        return a, logp, v

    def logprob_entropy_value(self, obs_t: torch.Tensor, act_t: torch.Tensor):
        mu, std, v = self.forward(obs_t)
        dist = torch.distributions.Normal(mu, std)
        logp = dist.log_prob(act_t).sum(-1)
        ent = dist.entropy().sum(-1)
        return logp, ent, v


@dataclass
class Buffer:
    obs: List[np.ndarray]
    act: List[np.ndarray]
    logp: List[float]
    val: List[float]
    rew: List[float]
    done: List[float]

    def to_tensors(self):
        obs = torch.tensor(np.asarray(self.obs), dtype=torch.float32)
        act = torch.tensor(np.asarray(self.act), dtype=torch.float32)
        logp = torch.tensor(np.asarray(self.logp), dtype=torch.float32)
        val = torch.tensor(np.asarray(self.val), dtype=torch.float32)
        rew = np.asarray(self.rew, dtype=np.float32)
        done = np.asarray(self.done, dtype=np.float32)
        return obs, act, logp, val, rew, done


def compute_gae(rew: np.ndarray, val: np.ndarray, done: np.ndarray, gamma=0.99, lam=0.95):
    T = len(rew)
    adv = np.zeros(T, dtype=np.float32)
    last = 0.0
    for t in reversed(range(T)):
        next_nonterminal = 1.0 - done[t]
        next_val = val[t + 1] if t + 1 < len(val) else 0.0
        delta = rew[t] + gamma * next_val * next_nonterminal - val[t]
        last = delta + gamma * lam * next_nonterminal * last
        adv[t] = last
    ret = adv + val[:T]
    return adv, ret


def ppo_update(model: ActorCritic, opt: optim.Optimizer,
               obs, act, logp_old, val_old, adv, ret,
               clip=0.2, vf_coef=0.5, ent_coef=0.01,
               epochs=5, batch_size=2048):
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    N = obs.shape[0]
    idx = np.arange(N)

    for _ in range(epochs):
        np.random.shuffle(idx)
        for start in range(0, N, batch_size):
            mb = idx[start:start + batch_size]

            mb_obs = obs[mb]
            mb_act = act[mb]
            mb_logp_old = logp_old[mb]
            mb_adv = adv[mb]
            mb_ret = ret[mb]

            logp, ent, v = model.logprob_entropy_value(mb_obs, mb_act)
            ratio = torch.exp(logp - mb_logp_old)

            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * mb_adv
            pi_loss = -torch.min(surr1, surr2).mean()

            v_loss = ((v - mb_ret) ** 2).mean()

            loss = pi_loss + vf_coef * v_loss - ent_coef * ent.mean()

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()


# -----------------------------
# 3) 路线1：Policy Population + Pareto 选择（用 crowding distance 近似 HV）
# -----------------------------
def pareto_front_indices(points: List[np.ndarray]) -> List[int]:
    front = []
    for i, p in enumerate(points):
        dom = False
        for j, q in enumerate(points):
            if j == i:
                continue
            if dominates(q, p):
                dom = True
                break
        if not dom:
            front.append(i)
    return front


def crowding_distance(points: List[np.ndarray], front: List[int]) -> Dict[int, float]:
    if len(front) == 0:
        return {}
    m = len(points[0])
    cd = {i: 0.0 for i in front}
    front_pts = [(i, points[i]) for i in front]
    for k in range(m):
        front_pts.sort(key=lambda t: t[1][k])
        cd[front_pts[0][0]] = float("inf")
        cd[front_pts[-1][0]] = float("inf")
        minv = float(front_pts[0][1][k])
        maxv = float(front_pts[-1][1][k])
        denom = (maxv - minv) if maxv > minv else 1.0
        for t in range(1, len(front_pts) - 1):
            i_prev, p_prev = front_pts[t - 1]
            i_next, p_next = front_pts[t + 1]
            i_mid, _ = front_pts[t]
            cd[i_mid] += float((p_next[k] - p_prev[k]) / denom)
    return cd


def select_population(models, J, keep: int):
    # J: list of np.ndarray(3,)
    remaining = list(range(len(models)))
    selected = []

    while remaining and len(selected) < keep:
        front = pareto_front_indices([J[i] for i in remaining])
        front_global = [remaining[i] for i in front]

        if len(selected) + len(front_global) <= keep:
            selected.extend(front_global)
            remaining = [i for i in remaining if i not in set(front_global)]
        else:
            # need partial selection from this front by crowding distance
            cd = crowding_distance(J, front_global)
            front_sorted = sorted(front_global, key=lambda i: cd.get(i, 0.0), reverse=True)
            need = keep - len(selected)
            selected.extend(front_sorted[:need])
            break

    selected_models = [models[i] for i in selected]
    selected_J = [J[i] for i in selected]
    return selected_models, selected_J


# -----------------------------
# 4) Rollout + 评估（核心：用 info['r_vec']）
# -----------------------------
def sample_scalarization():
    # 给每个策略一次更新时随机一个标量化方式/偏好
    mode = random.choice(["ws", "tcheby"])
    w = np.random.dirichlet(np.ones(3)).astype(np.float32)
    return mode, w


def rollout_episode(env, model: ActorCritic, mpso: MPSOScheduler,
                    mode: str, w: np.ndarray,
                    z_ref: Optional[np.ndarray] = None,
                    max_steps: int = 2000,
                    schedule_every: int = 1):
    """
    单 episode rollout：
    - 每 schedule_every 步调用一次 MPSO 下发分配（READY订单 -> drones）
    - PPO 动作：heading
    - reward：不使用 env reward，而用 info['r_vec'] 标量化得到 scalar reward
    """
    obs, _ = env.reset()
    buf = Buffer(obs=[], act=[], logp=[], val=[], rew=[], done=[])

    ep_r_vec = np.zeros(3, dtype=np.float32)

    for t in range(max_steps):
        # 先让环境构造好 current_obs_order_ids（由 _get_observation 切片决定）
        flat = flatten_obs(obs)
        obs_t = torch.tensor(flat, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            act_t, logp_t, v_t = model.act(obs_t)

        act = act_t.squeeze(0).cpu().numpy().astype(np.float32)
        # reshape 为 (num_drones,2)
        act = act.reshape(env.num_drones, 2)

        # 调用 MPSO 分配（在 env.step 前执行，确保本步就能生效）
        if (t % schedule_every) == 0:
            apply_mpso(env, mpso)

        next_obs, _, terminated, truncated, info = env.step(act)

        r_vec = info["r_vec"].astype(np.float32)
        ep_r_vec += r_vec

        # 标量化 reward（用于 PPO 更新）
        if mode == "tcheby":
            if z_ref is None:
                # 参考点 z：用当前 episode 的累计最好值近似（也可用历史 ideal point）
                z_ref = np.zeros(3, dtype=np.float32)
            rew = scalarize(r_vec, mode=mode, w=w, z=z_ref)
        else:
            rew = scalarize(r_vec, mode=mode, w=w)

        buf.obs.append(flat)
        buf.act.append(act.reshape(-1))  # store flat action
        buf.logp.append(float(logp_t.item()))
        buf.val.append(float(v_t.item()))
        buf.rew.append(float(rew))
        buf.done.append(float(terminated or truncated))

        obs = next_obs
        if terminated or truncated:
            break

    # 转 tensor
    obs_t, act_t, logp_old_t, val_old_t, rew_arr, done_arr = buf.to_tensors()

    adv, ret = compute_gae(rew_arr, val_old_t.numpy(), done_arr, gamma=0.99, lam=0.95)
    adv_t = torch.tensor(adv, dtype=torch.float32)
    ret_t = torch.tensor(ret, dtype=torch.float32)

    return obs_t, act_t, logp_old_t, val_old_t, adv_t, ret_t, ep_r_vec


def evaluate_model(env_ctor, model: ActorCritic, mpso: MPSOScheduler,
                   episodes: int = 3, schedule_every: int = 1, max_steps: int = 2000) -> np.ndarray:
    """
    多次评估：返回 mean episode vector return J ∈ R^3
    评估建议关随机事件：enable_random_events=False（需你环境已加该开关）
    """
    rs = []
    for _ in range(episodes):
        env = env_ctor()
        obs, _ = env.reset()
        ep = np.zeros(3, dtype=np.float32)
        for t in range(max_steps):
            flat = flatten_obs(obs)
            obs_t = torch.tensor(flat, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                act_t, _, _ = model.act(obs_t)
            act = act_t.squeeze(0).cpu().numpy().astype(np.float32).reshape(env.num_drones, 2)

            if (t % schedule_every) == 0:
                apply_mpso(env, mpso)

            obs, _, terminated, truncated, info = env.step(act)
            ep += info["r_vec"].astype(np.float32)
            if terminated or truncated:
                break
        rs.append(ep)
    return np.mean(np.stack(rs, axis=0), axis=0)


# -----------------------------
# 5) 主训练：Population MORL
# -----------------------------
def main():
    device = "cpu"

    # 构造 env 工厂：训练与评估建议分开（评估关随机事件）
    def make_train_env():
        return ThreeObjectiveDroneDeliveryEnv(
            grid_size=16,
            num_drones=6,
            max_orders=100,
            steps_per_hour=24,
            reward_output_mode="zero",
            enable_random_events=True,
            top_k_merchants=100,
        )

    def make_eval_env():
        return ThreeObjectiveDroneDeliveryEnv(
            grid_size=16,
            num_drones=6,
            max_orders=100,
            steps_per_hour=24,
            reward_output_mode="zero",
            enable_random_events=False,  # 评估建议关随机事件
            top_k_merchants=100,
        )

    # 初始化一个 env 获取 obs_dim / act_dim
    env0 = make_train_env()
    obs0, _ = env0.reset()
    obs_dim = flatten_obs(obs0).shape[0]
    act_dim = env0.num_drones * 2

    # policy population
    POP = 8
    KEEP = 8
    models: List[ActorCritic] = []
    opts: List[optim.Optimizer] = []
    for _ in range(POP):
        m = ActorCritic(obs_dim, act_dim, hidden=256).to(device)
        models.append(m)
        opts.append(optim.Adam(m.parameters(), lr=3e-4))

    # MPSO scheduler（订单分配）
    mpso = MPSOScheduler(
        n_particles=30,
        iters=25,
        inertia=0.6,
        c1=1.4,
        c2=1.4,
        max_assign_per_drone=4,
        seed=0
    )

    GENERATIONS = 30
    EPISODES_PER_UPDATE = 4  # 每个 policy 每代用多少 episode 更新（越大越稳，越慢）
    EVAL_EPISODES = 3

    for gen in range(GENERATIONS):
        # ---------- 训练阶段：对每个策略做 PPO 更新 ----------
        for i, (model, opt) in enumerate(zip(models, opts)):
            mode, w = sample_scalarization()

            # 可选：tcheby 的参考点 z，用历史最好（ideal point）更新
            z_ref = None

            # 收集多 episode 叠加成一个大 batch
            all_obs, all_act, all_logp, all_val, all_adv, all_ret = [], [], [], [], [], []
            for _ in range(EPISODES_PER_UPDATE):
                env = make_train_env()
                obs_t, act_t, logp_old_t, val_old_t, adv_t, ret_t, ep_r_vec = rollout_episode(
                    env, model, mpso, mode=mode, w=w, z_ref=z_ref,
                    max_steps=2500, schedule_every=1
                )
                all_obs.append(obs_t)
                all_act.append(act_t)
                all_logp.append(logp_old_t)
                all_val.append(val_old_t)
                all_adv.append(adv_t)
                all_ret.append(ret_t)

            obs_b = torch.cat(all_obs, dim=0)
            act_b = torch.cat(all_act, dim=0)
            logp_b = torch.cat(all_logp, dim=0)
            val_b = torch.cat(all_val, dim=0)
            adv_b = torch.cat(all_adv, dim=0)
            ret_b = torch.cat(all_ret, dim=0)

            ppo_update(
                model, opt,
                obs=obs_b, act=act_b, logp_old=logp_b,
                val_old=val_b, adv=adv_b, ret=ret_b,
                clip=0.2, vf_coef=0.5, ent_coef=0.01,
                epochs=5, batch_size=2048
            )

        # ---------- 评估阶段：得到每个策略的向量回报 ----------
        J = []
        for i, model in enumerate(models):
            J_i = evaluate_model(make_eval_env, model, mpso, episodes=EVAL_EPISODES, schedule_every=1, max_steps=2500)
            J.append(J_i)
        J = [np.asarray(x, dtype=np.float32) for x in J]

        # 输出当前 Pareto front
        front = pareto_front_indices(J)
        print(f"\n[Gen {gen}] Pareto front size={len(front)}")
        for idx in front:
            print(f"  policy#{idx}: J={J[idx]}")

        # ---------- 选择阶段：Pareto + crowding distance ----------
        models, J_sel = select_population(models, J, keep=KEEP)

        # 重新构建 opts（保留对应 optimizer 状态会复杂，这里简单重建）
        opts = [optim.Adam(m.parameters(), lr=3e-4) for m in models]

        # ---------- 复制补齐 population（可选：加入噪声变异） ----------
        while len(models) < POP:
            parent = random.choice(models)
            child = ActorCritic(obs_dim, act_dim, hidden=256)
            child.load_state_dict(parent.state_dict())

            # mutation：给参数加小噪声，促进多样性
            with torch.no_grad():
                for p in child.parameters():
                    p.add_(0.01 * torch.randn_like(p))

            models.append(child)
            opts.append(optim.Adam(child.parameters(), lr=3e-4))

    print("Training done.")


if __name__ == "__main__":
    main()
