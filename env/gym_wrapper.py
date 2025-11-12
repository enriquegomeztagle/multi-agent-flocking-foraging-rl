"""
Gymnasium wrapper for PettingZoo ParallelEnv to make it compatible with Stable-Baselines3.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Tuple, Optional

from .flockforage_parallel import FlockForageParallel, EnvConfig


class FlockForageGymWrapper(gym.Env):
    """
    Wrapper to convert PettingZoo ParallelEnv to Gymnasium Env for single-agent RL training.

    This wrapper treats all agents as a single learner with vectorized observations and actions.
    The observation space is a vector of all agents' observations concatenated.
    The action space is a MultiDiscrete space for all agents.
    """

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 20}

    def __init__(self, config: EnvConfig):
        """Initialize wrapper."""
        super().__init__()

        # Create PettingZoo environment
        self.env = FlockForageParallel(config)
        self.config = config

        # Get single agent observation and action spaces
        single_obs_space = self.env._obs_space
        single_act_space = self.env._act_space

        # Vectorized observation: stack all agents' observations
        # Shape: (n_agents * obs_dim,)
        obs_dim = single_obs_space.shape[0]
        total_obs_dim = config.n_agents * obs_dim

        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(total_obs_dim,),
            dtype=np.float32
        )

        # Vectorized action: MultiDiscrete for all agents
        # Shape: (n_agents,) where each element is in [0, 4]
        self.action_space = spaces.MultiDiscrete([single_act_space.n] * config.n_agents)

        # Store agent IDs
        self.agents = self.env.agents
        self.n_agents = len(self.agents)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment."""
        if seed is not None:
            super().reset(seed=seed)

        obs_dict, info_dict = self.env.reset(seed=seed, options=options)

        # Convert dict of observations to single stacked vector
        obs_vector = self._dict_to_vector_obs(obs_dict)

        # Merge info dicts
        info = {"agents": info_dict}

        return obs_vector, info

    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Step environment."""
        # Convert action vector to dict
        action_dict = self._vector_to_dict_action(action)

        # Step environment
        obs_dict, reward_dict, term_dict, trunc_dict, info_dict = self.env.step(action_dict)

        # Convert observations to vector
        obs_vector = self._dict_to_vector_obs(obs_dict)

        # Sum rewards across all agents
        total_reward = sum(reward_dict.values())

        # Check if all agents are done
        terminated = all(term_dict.values())
        truncated = all(trunc_dict.values())

        # Merge info
        info = {
            "agents": info_dict,
            "individual_rewards": reward_dict,
        }

        return obs_vector, total_reward, terminated, truncated, info

    def _dict_to_vector_obs(self, obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """Convert dictionary of observations to single vector."""
        obs_list = [obs_dict[agent] for agent in self.agents]
        return np.concatenate(obs_list, axis=0).astype(np.float32)

    def _vector_to_dict_action(self, action: np.ndarray) -> Dict[str, int]:
        """Convert action vector to dictionary."""
        return {
            agent: int(action[i])
            for i, agent in enumerate(self.agents)
        }

    def render(self):
        """Render environment."""
        return self.env.render()

    def close(self):
        """Close environment."""
        self.env.close()
