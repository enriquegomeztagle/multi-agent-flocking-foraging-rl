"""
Multi-agent flocking and foraging environment (PettingZoo ParallelEnv).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
from gymnasium import spaces
from pettingzoo.utils import ParallelEnv
from . import physics, patches
from metrics.fairness import gini


@dataclass
class EnvConfig:
    """Environment configuration."""

    # World
    width: float = 50.0
    height: float = 50.0
    dt: float = 0.2

    # Agents
    n_agents: int = 10
    k_neighbors: int = 6

    # Dynamics
    v_max: float = 1.2
    a_max: float = 0.15
    turn_max: float = 0.25

    # Foraging
    feed_radius: float = 1.0
    c_max: float = 0.03

    # Flocking
    d_safe: float = 0.6

    # Patches
    S_max: float = 1.0
    S_thr: float = 0.2
    regen_r: float = 0.08
    n_patches: int = 12

    # Episode
    episode_len: int = 1500


class FlockForageParallel(ParallelEnv):
    """
    Multi-agent flocking and foraging environment.

    Observation space (13D):
        - Own velocity (2D, normalized by v_max)
        - Mean neighbor velocity (2D, normalized)
        - Mean neighbor position relative to self (2D, normalized by perception radius)
        - Mean distance to k neighbors (1D, normalized)
        - Vector to nearest patch (2D, normalized)
        - Nearest patch stock (1D, normalized by S_max)
        - Global mean patch stock (1D, normalized)
        - EMA of own intake (1D)
        - Mean intake EMA of neighbors (1D) - for coordination

    Action space (5 discrete):
        0: Turn left
        1: Turn right
        2: Accelerate
        3: Decelerate
        4: No-op
    """

    metadata = {
        "name": "flock_forage_v0",
        "render_modes": ["rgb_array", "human"],
        "render_fps": 20,
    }

    def __init__(self, cfg: EnvConfig, render_mode=None):
        """Initialize environment."""
        self.cfg = cfg
        self.render_mode = render_mode
        self.agents = [f"agent_{i}" for i in range(cfg.n_agents)]
        self.possible_agents = list(self.agents)

        self._rng = np.random.default_rng()
        self._t = 0

        # State
        self._patches = patches.PatchField(cfg)
        self._pos = np.zeros((cfg.n_agents, 2), np.float32)
        self._vel = np.zeros((cfg.n_agents, 2), np.float32)
        self._heading = np.zeros((cfg.n_agents,), np.float32)
        self._intake_total = np.zeros((cfg.n_agents,), np.float32)
        self._intake_ema = np.zeros((cfg.n_agents,), np.float32)

        # Observation and action spaces
        self._obs_dim = 13
        self._obs_space = spaces.Box(
            -10.0, 10.0, shape=(self._obs_dim,), dtype=np.float32
        )
        self._act_space = spaces.Discrete(5)

        # For metrics tracking
        self._neighbors = None
        self._distances = None

        # For reward calculation: track distance to nearest patch
        self._prev_patch_distances = np.zeros((cfg.n_agents,), np.float32)

        # Track patch rotation for optimal foraging
        self._prev_patch_id = np.full((cfg.n_agents,), -1, dtype=np.int32)
        self._time_at_patch = np.zeros((cfg.n_agents,), np.float32)
        self._patches_visited = np.zeros((cfg.n_agents,), np.float32)

    def observation_space(self, agent: str):
        """Get observation space for agent."""
        return self._obs_space

    def action_space(self, agent: str):
        """Get action space for agent."""
        return self._act_space

    def reset(self, seed: Optional[int] = None, options=None):
        """Reset environment."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._t = 0
        self._intake_total[:] = 0.0
        self._intake_ema[:] = 0.0
        self._prev_patch_distances[:] = 0.0
        self._prev_patch_id[:] = -1
        self._time_at_patch[:] = 0.0
        self._patches_visited[:] = 0.0

        # Random initial positions
        self._pos = self._rng.uniform(
            [0, 0], [self.cfg.width, self.cfg.height], (self.cfg.n_agents, 2)
        ).astype(np.float32)

        # Random initial headings and velocities
        ang = self._rng.uniform(-np.pi, np.pi, size=(self.cfg.n_agents,))
        self._heading = ang.astype(np.float32)

        spd = self._rng.uniform(0.2, 0.6, size=(self.cfg.n_agents,))
        self._vel = np.stack([np.cos(ang) * spd, np.sin(ang) * spd], 1).astype(
            np.float32
        )

        # Reset patches with same RNG for determinism
        self._patches.reset(self._rng)

        # Compute initial neighbors
        self._neighbors, self._distances = physics.find_k_nearest(
            self._pos, self.cfg.k_neighbors, self.cfg
        )

        # Initialize previous patch distances and IDs (Euclidean distance with reflective boundaries)
        for i in range(self.cfg.n_agents):
            patch_center, _, _, patch_id = self._patches.get_patch_info(self._pos[i])
            dx = patch_center[0] - self._pos[i, 0]
            dy = patch_center[1] - self._pos[i, 1]
            self._prev_patch_distances[i] = np.sqrt(dx * dx + dy * dy)
            self._prev_patch_id[i] = patch_id

        obs = {a: self._make_obs(i) for i, a in enumerate(self.agents)}
        infos = {a: {} for a in self.agents}

        return obs, infos

    def step(self, actions: Dict[str, int]):
        """Step environment."""
        # Convert discrete actions to continuous control
        a_vec = np.zeros((self.cfg.n_agents, 2), np.float32)
        for i, a in enumerate(self.agents):
            act = actions[a]
            if act == 0:  # Turn left
                a_vec[i, 0] = -self.cfg.turn_max
            elif act == 1:  # Turn right
                a_vec[i, 0] = self.cfg.turn_max
            elif act == 2:  # Accelerate
                a_vec[i, 1] = self.cfg.a_max
            elif act == 3:  # Decelerate
                a_vec[i, 1] = -self.cfg.a_max
            # act == 4 is no-op

        # Integrate physics
        self._pos, self._vel, self._heading = physics.integrate(
            self._pos, self._vel, self._heading, a_vec, self.cfg
        )

        # Update neighbors
        self._neighbors, self._distances = physics.find_k_nearest(
            self._pos, self.cfg.k_neighbors, self.cfg
        )

        # Foraging
        intake = np.zeros((self.cfg.n_agents,), np.float32)
        for i in range(self.cfg.n_agents):
            intake[i] = self._patches.harvest(
                self._pos[i], self.cfg.feed_radius, self.cfg.c_max
            )
            self._intake_total[i] += intake[i]

        # Update EMA of intake (alpha = 0.1)
        self._intake_ema = 0.9 * self._intake_ema + 0.1 * intake

        # Regenerate patches
        self._patches.regenerate()

        # Compute rewards
        rewards = self._compute_rewards(intake)

        # Check termination
        self._t += 1
        done = self._t >= self.cfg.episode_len

        terminations = {a: done for a in self.agents}
        truncations = {a: False for a in self.agents}
        infos = {a: {} for a in self.agents}

        # Compute observations
        obs = {a: self._make_obs(i) for i, a in enumerate(self.agents)}

        # Add fairness bonus at episode end (reward for low inequality)
        if done:
            gini_coef = float(gini(self._intake_total + 1e-8))
            # Convert to positive reward: reward fairness (1 - gini)
            fairness_bonus = 0.5 * (1.0 - gini_coef)  # Range: [0, 0.5]
            for a in rewards:
                rewards[a] += fairness_bonus

        return obs, rewards, terminations, truncations, infos

    def _make_obs(self, i: int) -> np.ndarray:
        """
        Build observation for agent i (13D).

        0-1: Own velocity (normalized by v_max)
        2-3: Mean neighbor velocity (normalized)
        4-5: Mean relative position to neighbors (normalized)
        6: Mean distance to k neighbors (normalized)
        7-8: Vector to nearest patch (normalized)
        9: Nearest patch stock (normalized)
        10: Global mean patch stock (normalized)
        11: EMA of own intake (clipped to [0, 1])
        12: Mean intake EMA of neighbors (for coordination)
        """
        obs = np.zeros(self._obs_dim, dtype=np.float32)

        # Own velocity
        obs[0:2] = self._vel[i] / self.cfg.v_max

        # Neighbor info
        neighbor_indices = self._neighbors[i]
        neighbor_velocities = self._vel[neighbor_indices]
        obs[2:4] = np.mean(neighbor_velocities, axis=0) / self.cfg.v_max

        # Mean relative position to neighbors (Euclidean distance with reflective boundaries)
        relative_positions = []
        for j in neighbor_indices:
            dx = self._pos[j, 0] - self._pos[i, 0]
            dy = self._pos[j, 1] - self._pos[i, 1]
            relative_positions.append([dx, dy])

        relative_positions = np.array(relative_positions)
        obs[4:6] = np.mean(relative_positions, axis=0) / (
            self.cfg.width / 4
        )  # Normalize

        # Mean distance to neighbors
        obs[6] = np.mean(self._distances[i]) / (self.cfg.width / 4)

        # Nearest patch info (direct Euclidean distance with reflective boundaries)
        patch_center, patch_stock, patch_norm, _ = self._patches.get_patch_info(
            self._pos[i]
        )
        dx = patch_center[0] - self._pos[i, 0]
        dy = patch_center[1] - self._pos[i, 1]

        obs[7:9] = np.array([dx, dy]) / (self.cfg.width / 2)
        obs[9] = patch_norm

        # Global mean patch stock
        obs[10] = self._patches.mean_stock_normalized()

        # Own intake EMA (clip to reasonable range)
        obs[11] = np.clip(self._intake_ema[i] / self.cfg.c_max, 0, 1)

        # Mean intake EMA of neighbors (for coordination)
        neighbor_intake = np.mean(self._intake_ema[neighbor_indices])
        obs[12] = np.clip(neighbor_intake / self.cfg.c_max, 0, 1)

        return obs

    def _compute_rewards(self, intake: np.ndarray) -> Dict[str, float]:
        """
        Components:
        1. Food reward: primary objective
        2. Exponential proximity: Strong gradient towards food patches
        3. Approach reward: Encourage moving closer to food
        4. Light overcrowding penalty: Allow group foraging but prevent total clustering
        """
        rewards = np.zeros(self.cfg.n_agents, dtype=np.float32)

        # Track previous distances for approach rewards
        if not hasattr(self, "_prev_patch_distances"):
            self._prev_patch_distances = np.zeros(self.cfg.n_agents, dtype=np.float32)
        if not hasattr(self, "_prev_patch_id"):
            self._prev_patch_id = np.full(self.cfg.n_agents, -1, dtype=np.int32)

        # 1. FOOD REWARD - 200x multiplier
        food_reward = intake / self.cfg.c_max
        rewards += food_reward * 200.0

        for i in range(self.cfg.n_agents):
            # Get current patch info
            patch_center, stock, _, patch_id = self._patches.get_patch_info(
                self._pos[i]
            )

            # Distance to nearest patch
            dx = patch_center[0] - self._pos[i, 0]
            dy = patch_center[1] - self._pos[i, 1]
            current_dist = np.sqrt(dx * dx + dy * dy)

            # 2. PROXIMITY REWARD - Exponential gradient
            if stock > 0.1:
                proximity_reward = 2.0 * np.exp(-current_dist / 5.0)
                rewards[i] += proximity_reward

            # 3. APPROACH REWARD - Reward moving toward food
            if self._prev_patch_distances[i] > 0:
                dist_change = self._prev_patch_distances[i] - current_dist
                if dist_change > 0 and stock > 0.1:
                    rewards[i] += 3.0 * dist_change

            self._prev_patch_distances[i] = current_dist

        # 4. LIGHT OVERCROWDING PENALTY
        for i in range(self.cfg.n_agents):
            _, _, _, my_patch_id = self._patches.get_patch_info(self._pos[i])
            self._prev_patch_id[i] = my_patch_id

            agents_at_my_patch = np.sum(self._prev_patch_id == my_patch_id)
            if agents_at_my_patch > 3:
                rewards[i] -= 0.5 * (agents_at_my_patch - 3)

        return {a: float(rewards[i]) for i, a in enumerate(self.agents)}

    def render(self, mode="human"):
        """Render environment (not implemented)."""
        pass

    def close(self):
        """Close environment."""
        pass
