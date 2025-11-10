"""
Baseline rules-only (Boids) controller for comparison with RL agents.

Implements classical flocking behaviors:
- Cohesion: Move towards center of mass of neighbors
- Alignment: Match velocity with neighbors
- Separation: Avoid collisions with neighbors
- Foraging: Move towards nearest food patch when hungry
"""

import numpy as np
from typing import Tuple


class BoidsController:
    """
    Classical Boids controller with foraging behavior.

    Uses hand-tuned weights for flocking rules and foraging decisions.
    """

    def __init__(
        self,
        w_cohesion: float = 1.0,
        w_alignment: float = 1.0,
        w_separation: float = 2.0,
        w_foraging: float = 1.5,
        hunger_threshold: float = 0.5,
        separation_radius: float = 0.6,
    ):
        """
        Initialize Boids controller.

        Args:
            w_cohesion: Weight for cohesion force
            w_alignment: Weight for alignment force
            w_separation: Weight for separation force
            w_foraging: Weight for foraging force
            hunger_threshold: Intake threshold below which agent seeks food
            separation_radius: Distance threshold for separation behavior
        """
        self.w_cohesion = w_cohesion
        self.w_alignment = w_alignment
        self.w_separation = w_separation
        self.w_foraging = w_foraging
        self.hunger_threshold = hunger_threshold
        self.separation_radius = separation_radius

    def compute_action(self, obs: np.ndarray, agent_heading: float) -> int:
        """
        Compute action based on observation using Boids rules.

        Observation space (13D):
            0-1: Own velocity (vx, vy)
            2-3: Mean neighbor velocity
            4-5: Mean relative position to neighbors
            6: Mean distance to neighbors
            7-8: Vector to nearest patch
            9: Nearest patch stock
            10: Global mean patch stock
            11: Own intake EMA
            12: Neighbor intake EMA

        Action space (5 discrete):
            0: Turn left
            1: Turn right
            2: Accelerate
            3: Decelerate
            4: No-op

        Args:
            obs: Observation vector (13D)
            agent_heading: Current heading angle in radians

        Returns:
            Action index (0-4)
        """
        # Parse observation
        own_vel = obs[0:2]
        neighbor_vel = obs[2:4]
        neighbor_rel_pos = obs[4:6]
        neighbor_dist = obs[6]
        patch_vec = obs[7:9]
        patch_stock = obs[9]
        own_intake = obs[11]

        # Compute desired direction based on Boids rules
        desired_direction = np.zeros(2)

        # 1. Cohesion: move towards neighbors (opposite of relative position)
        if neighbor_dist > 0.1:  # Have neighbors
            cohesion_force = -neighbor_rel_pos  # Move towards center
            desired_direction += self.w_cohesion * cohesion_force

        # 2. Alignment: match neighbor velocity
        if np.linalg.norm(neighbor_vel) > 0.01:
            alignment_force = neighbor_vel - own_vel
            desired_direction += self.w_alignment * alignment_force

        # 3. Separation: avoid collisions
        if neighbor_dist < self.separation_radius and neighbor_dist > 0.01:
            # Move away from neighbors
            separation_force = neighbor_rel_pos / (neighbor_dist + 1e-6)
            desired_direction += self.w_separation * separation_force

        # 4. Foraging: seek food when hungry
        if own_intake < self.hunger_threshold and patch_stock > 0.1:
            foraging_force = patch_vec
            desired_direction += self.w_foraging * foraging_force

        # Normalize desired direction
        if np.linalg.norm(desired_direction) > 0.01:
            desired_direction = desired_direction / np.linalg.norm(desired_direction)
        else:
            # No clear direction, maintain current heading
            return 4  # No-op

        # Compute desired heading
        desired_heading = np.arctan2(desired_direction[1], desired_direction[0])

        # Compute angle difference (shortest path)
        angle_diff = np.arctan2(
            np.sin(desired_heading - agent_heading),
            np.cos(desired_heading - agent_heading)
        )

        # Decide action based on angle difference and speed
        current_speed = np.linalg.norm(own_vel)

        # Turning actions (prioritize if angle is large)
        if abs(angle_diff) > 0.3:  # ~17 degrees
            if angle_diff > 0:
                return 0  # Turn left
            else:
                return 1  # Turn right

        # Speed control
        desired_speed = 0.8  # Target cruising speed

        if current_speed < desired_speed - 0.2:
            return 2  # Accelerate
        elif current_speed > desired_speed + 0.2:
            return 3  # Decelerate
        else:
            return 4  # No-op (maintain current)


def evaluate_baseline(
    env,
    controller: BoidsController,
    n_episodes: int = 10,
    max_steps: int = 1500,
    verbose: bool = True
) -> dict:
    """
    Evaluate baseline Boids controller on environment.

    Args:
        env: FlockForageParallel environment
        controller: BoidsController instance
        n_episodes: Number of evaluation episodes
        max_steps: Maximum steps per episode
        verbose: Print progress

    Returns:
        Dictionary of aggregated metrics
    """
    from metrics.fairness import gini
    from metrics.flocking import polarization, mean_knn_distance, separation_violations
    from metrics.sustainability import stock_score, min_stock_normalized

    all_metrics = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep * 100)

        # Track episode metrics
        total_rewards = []
        polarizations = []
        knn_distances = []
        separations = []
        stock_history = []

        for step in range(max_steps):
            # Get actions from Boids controller for each agent
            actions = {}
            for agent_id in env.agents:
                agent_obs = obs[agent_id]
                agent_idx = int(agent_id.split("_")[1])
                agent_heading = env._heading[agent_idx]
                actions[agent_id] = controller.compute_action(agent_obs, agent_heading)

            # Step environment
            obs, rewards, terminations, truncations, infos = env.step(actions)

            # Check if episode done
            if any(terminations.values()) or any(truncations.values()):
                break

            # Collect metrics
            total_rewards.append(sum(rewards.values()))
            polarizations.append(polarization(env._vel))
            knn_distances.append(mean_knn_distance(env._distances))
            separations.append(separation_violations(env._distances, env.cfg.d_safe))
            stock_history.append(env._patches.stock.copy())

        # Compute episode metrics
        stock_history = np.array(stock_history)
        intake_per_agent = env._intake_total.copy()

        metrics = {
            "episode_length": len(total_rewards),
            "total_reward": float(np.sum(total_rewards)),
            "mean_reward_per_step": float(np.mean(total_rewards)) if total_rewards else 0.0,
            # Fairness
            "gini": float(gini(intake_per_agent + 1e-8)),
            "intake_mean": float(np.mean(intake_per_agent)),
            "intake_std": float(np.std(intake_per_agent)),
            # Flocking
            "polarization_mean": float(np.mean(polarizations)),
            "knn_distance_mean": float(np.mean(knn_distances)),
            "separation_violations_mean": float(np.mean(separations)),
            # Sustainability
            "stock_final_mean": float(stock_score(stock_history[-1], env.cfg.S_max)),
            "stock_final_min": float(min_stock_normalized(stock_history[-1], env.cfg.S_max)),
            "stock_mean_over_time": float(np.mean([stock_score(s, env.cfg.S_max) for s in stock_history])),
        }

        all_metrics.append(metrics)

        if verbose:
            print(f"Episode {ep+1}/{n_episodes}: Reward={metrics['mean_reward_per_step']:.3f}, "
                  f"Gini={metrics['gini']:.3f}, Polarization={metrics['polarization_mean']:.3f}")

    # Aggregate metrics
    aggregated = {
        key: {
            "mean": float(np.mean([m[key] for m in all_metrics])),
            "std": float(np.std([m[key] for m in all_metrics])),
        }
        for key in all_metrics[0].keys()
    }

    return {
        "individual_episodes": all_metrics,
        "aggregated": aggregated
    }


if __name__ == "__main__":
    """Run baseline evaluation."""
    import yaml
    import json
    import os
    from env.flockforage_parallel import FlockForageParallel, EnvConfig

    # Load configuration
    config_path = "configs/env_curriculum_phase2.yaml"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            env_cfg = yaml.safe_load(f)
    else:
        print(f"Config not found: {config_path}, using defaults")
        env_cfg = {}

    # Create environment
    print("Creating environment...")
    env = FlockForageParallel(EnvConfig(**env_cfg))

    # Create Boids controller
    print("Creating Boids controller...")
    controller = BoidsController(
        w_cohesion=1.0,
        w_alignment=1.0,
        w_separation=2.0,
        w_foraging=1.5,
        hunger_threshold=0.5
    )

    # Evaluate
    print("\nEvaluating baseline Boids controller...")
    print("=" * 60)
    results = evaluate_baseline(env, controller, n_episodes=10, verbose=True)

    # Print summary
    print("\n" + "=" * 60)
    print("BASELINE EVALUATION RESULTS")
    print("=" * 60)
    for key, val in results["aggregated"].items():
        print(f"{key:30s}: {val['mean']:8.4f} ± {val['std']:6.4f}")

    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/baseline_boids_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to results/baseline_boids_metrics.json")

    env.close()
