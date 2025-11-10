"""
Evaluate trained PPO agent and compute metrics.
"""

import yaml
import os
import json
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
import supersuit as ss
from env.flockforage_parallel import FlockForageParallel, EnvConfig
from metrics.fairness import gini
from metrics.flocking import polarization, mean_knn_distance, separation_violations
from metrics.sustainability import stock_score, min_stock_normalized


def load_yaml(path: str) -> dict:
    """Load YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def make_eval_env(env_cfg: dict):
    """Create single evaluation environment."""
    env = FlockForageParallel(EnvConfig(**env_cfg))
    venv = ss.pettingzoo_env_to_vec_env_v1(env)
    venv = ss.concat_vec_envs_v1(venv, num_vec_envs=1, num_cpus=0, base_class="stable_baselines3")
    return venv


def evaluate_episode(vec_env, model, unwrapped_env, max_steps=1500):
    """
    Run single evaluation episode and collect metrics.

    Args:
        vec_env: Vectorized environment for running the model
        model: Trained PPO model
        unwrapped_env: Unwrapped environment for metrics collection
        max_steps: Maximum steps per episode

    Returns:
        Dictionary of episode metrics
    """
    # Reset both environments
    obs = vec_env.reset()
    unwrapped_env.reset(seed=np.random.randint(0, 100000))

    done = False
    step = 0

    # Track metrics over episode
    total_rewards = []
    stock_history = []
    polarizations = []
    knn_distances = []
    separations = []

    while step < max_steps:
        # Get action from trained model
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)

        # Check if episode is done
        if np.any(done):
            break

        # Also step the unwrapped env with same actions (for metrics)
        # Convert vectorized action to dict for parallel env
        action_dict = {f"agent_{i}": int(action[i]) for i in range(len(unwrapped_env.agents))}
        _, _, _, _, _ = unwrapped_env.step(action_dict)

        # Collect metrics from unwrapped env
        total_rewards.append(float(reward.sum()))

        # Flocking metrics
        polarizations.append(polarization(unwrapped_env._vel))
        knn_distances.append(mean_knn_distance(unwrapped_env._distances))
        separations.append(separation_violations(unwrapped_env._distances, unwrapped_env.cfg.d_safe))

        # Sustainability metrics
        stock_history.append(unwrapped_env._patches.stock.copy())

        step += 1

    # Compute episode-level metrics
    stock_history = np.array(stock_history)
    intake_per_agent = unwrapped_env._intake_total.copy()

    metrics = {
        "episode_length": step,
        "total_reward": float(np.sum(total_rewards)),
        "mean_reward_per_step": float(np.mean(total_rewards)),
        # Fairness
        "gini": float(gini(intake_per_agent + 1e-8)),
        "intake_mean": float(np.mean(intake_per_agent)),
        "intake_std": float(np.std(intake_per_agent)),
        # Flocking
        "polarization_mean": float(np.mean(polarizations)),
        "polarization_final": float(polarizations[-1]) if polarizations else 0.0,
        "knn_distance_mean": float(np.mean(knn_distances)),
        "separation_violations_mean": float(np.mean(separations)),
        # Sustainability
        "stock_final_mean": float(stock_score(stock_history[-1], unwrapped_env.cfg.S_max)),
        "stock_final_min": float(min_stock_normalized(stock_history[-1], unwrapped_env.cfg.S_max)),
        "stock_mean_over_time": float(np.mean([stock_score(s, unwrapped_env.cfg.S_max) for s in stock_history])),
    }

    return metrics


def main():
    """Main evaluation loop."""
    # Load configurations
    env_cfg = load_yaml("configs/env.yaml")
    ppo_cfg = load_yaml("configs/ppo_config.yaml")

    # Check if model exists
    if not os.path.exists("models/ppo_policy.zip"):
        print("Error: Trained model not found at models/ppo_policy.zip")
        print("Please train a model first using: python -m train.run_ppo")
        return

    # Create evaluation environment
    print("Creating evaluation environment...")
    venv = make_eval_env(env_cfg)

    # Create unwrapped environment for metrics collection
    unwrapped_env = FlockForageParallel(EnvConfig(**env_cfg))

    # Load VecNormalize if it was used during training
    if ppo_cfg.get("vecnormalize", True) and os.path.exists("models/vecnorm.pkl"):
        print("Loading VecNormalize stats...")
        venv = VecNormalize.load("models/vecnorm.pkl", venv)
        venv.training = False
        venv.norm_reward = False

    # Load trained model
    print("Loading trained model...")
    model = PPO.load("models/ppo_policy", env=venv)

    # Run evaluation episodes
    n_eval_episodes = 10
    print(f"\nRunning {n_eval_episodes} evaluation episodes...")

    all_metrics = []
    for ep in range(n_eval_episodes):
        print(f"  Episode {ep + 1}/{n_eval_episodes}...", end=" ")
        metrics = evaluate_episode(
            venv,
            model,
            unwrapped_env,
            max_steps=env_cfg.get("episode_len", 1500)
        )
        all_metrics.append(metrics)
        print(f"Reward: {metrics['mean_reward_per_step']:.3f}, Gini: {metrics['gini']:.3f}")

    # Aggregate metrics
    print("\n" + "="*60)
    print("EVALUATION RESULTS (mean ± std over episodes)")
    print("="*60)

    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics]
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"{key:30s}: {mean_val:8.4f} ± {std_val:6.4f}")

    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/evaluation_metrics.json", "w") as f:
        json.dump({
            "individual_episodes": all_metrics,
            "aggregated": {
                key: {
                    "mean": float(np.mean([m[key] for m in all_metrics])),
                    "std": float(np.std([m[key] for m in all_metrics])),
                }
                for key in all_metrics[0].keys()
            }
        }, f, indent=2)

    print(f"\nResults saved to results/evaluation_metrics.json")

    venv.close()


if __name__ == "__main__":
    main()
