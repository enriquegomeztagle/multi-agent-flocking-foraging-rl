"""
Evaluación con guardado de trayectorias completas

Guarda las posiciones completas de los mejores episodios
para poder generar videos reproducibles después.
"""

import json
import pickle
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO

from env.flockforage_parallel import FlockForageParallel, EnvConfig
from metrics.fairness import gini


def evaluate_and_save_trajectories(n_episodes: int = 50, save_top_n: int = 5):
    """
    Evalúa el modelo y guarda trayectorias completas de los mejores episodios.
    """
    print("=" * 80)
    print("EVALUACIÓN CON GUARDADO DE TRAYECTORIAS")
    print("=" * 80)
    print()
    print(f"Episodes: {n_episodes}")
    print(f"Saving top: {save_top_n}")
    print("=" * 80)
    print()

    # Load model
    model = PPO.load("models/ppo_final/final/model")

    env_cfg = {
        "n_agents": 10,
        "n_patches": 15,
        "width": 30.0,
        "height": 30.0,
        "episode_len": 1500,
        "feed_radius": 3.0,
        "c_max": 0.06,
        "S_max": 1.0,
        "regen_r": 0.3,
    }

    env = FlockForageParallel(EnvConfig(**env_cfg))

    all_episodes = []

    print(f"Evaluating {n_episodes} episodes...")
    print()

    for ep in range(n_episodes):
        seed = ep * 42
        obs, _ = env.reset(seed=seed)

        # Storage for trajectory
        positions_history = []
        patches_history = []
        intake_history = []
        ep_reward = 0

        # Run episode
        for step in range(1500):
            # Save state BEFORE action
            positions_history.append(env._pos.copy())
            patches_history.append({
                "centers": env._patches.centers.copy(),
                "stock": env._patches.stock.copy(),
            })
            intake_history.append(float(env._intake_total.sum()))

            # Predict and step
            obs_array = np.array([obs[agent] for agent in env.agents])
            action, _ = model.predict(obs_array, deterministic=True)
            action_dict = {agent: int(action[i]) for i, agent in enumerate(env.agents)}

            obs, rew, terms, truncs, _ = env.step(action_dict)
            ep_reward += sum(rew.values())

            if all(terms.values()) or all(truncs.values()):
                break

        # Final metrics
        final_intake = float(env._intake_total.sum())
        gini_val = float(gini(env._intake_total))

        episode_data = {
            "episode": ep + 1,
            "seed": seed,
            "intake": final_intake,
            "reward": float(ep_reward),
            "gini": gini_val,
            "efficiency_percent": (final_intake / 2307) * 100,
            "steps": len(positions_history),
        }

        # Store trajectory for later
        trajectory = {
            "positions": [pos.tolist() for pos in positions_history],
            "patches": [{
                "centers": p["centers"].tolist(),
                "stock": p["stock"].tolist(),
            } for p in patches_history],
            "intake_history": intake_history,
            "episode_data": episode_data,
        }

        all_episodes.append((episode_data, trajectory))

        if (ep + 1) % 10 == 0:
            current_mean = np.mean([e[0]["intake"] for e in all_episodes])
            current_max = np.max([e[0]["intake"] for e in all_episodes])
            print(f"  [{ep+1:3d}/{n_episodes}] Mean: {current_mean:6.1f} | Max: {current_max:6.1f}")

    print()
    print("✅ Evaluation completed")
    print()

    # Sort by intake
    all_episodes.sort(key=lambda x: x[0]["intake"], reverse=True)

    # Get top N
    top_episodes = all_episodes[:save_top_n]

    # Display top
    print("=" * 80)
    print(f"TOP {save_top_n} EPISODES")
    print("=" * 80)
    for i, (data, _) in enumerate(top_episodes, 1):
        marker = "🏆" if i == 1 else "🌟" if i <= 3 else "⭐"
        print(
            f"{marker} #{i}  Episode {data['episode']:3d} (seed={data['seed']:5d}): "
            f"{data['intake']:6.2f} intake ({data['efficiency_percent']:5.2f}%)"
        )
    print("=" * 80)
    print()

    # Save trajectories
    trajectories_dir = Path("results/trajectories")
    trajectories_dir.mkdir(parents=True, exist_ok=True)

    saved_files = []

    for i, (data, trajectory) in enumerate(top_episodes, 1):
        filename = trajectories_dir / f"episode_{data['episode']}_seed_{data['seed']}_intake_{data['intake']:.0f}.pkl"

        with open(filename, 'wb') as f:
            pickle.dump(trajectory, f)

        saved_files.append(str(filename))
        print(f"✅ Saved trajectory {i}: {filename.name}")

    print()

    # Save summary JSON
    summary = {
        "n_episodes_evaluated": n_episodes,
        "top_n_saved": save_top_n,
        "top_episodes": [data for data, _ in top_episodes],
        "saved_trajectory_files": saved_files,
        "env_config": env_cfg,
    }

    summary_file = Path("results/trajectories_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"✅ Summary saved: {summary_file}")
    print()

    # Statistics
    intakes = [e[0]["intake"] for e in all_episodes]
    print("=" * 80)
    print("STATISTICS")
    print("=" * 80)
    print(f"Mean:   {np.mean(intakes):.2f}")
    print(f"Median: {np.median(intakes):.2f}")
    print(f"Max:    {np.max(intakes):.2f}")
    print(f"p95:    {np.percentile(intakes, 95):.2f}")
    print("=" * 80)

    return top_episodes[0][0]["intake"]  # Return best intake


if __name__ == "__main__":
    best_intake = evaluate_and_save_trajectories(n_episodes=50, save_top_n=5)

    print()
    print("=" * 80)
    print("✅ COMPLETED - TRAJECTORIES SAVED")
    print("=" * 80)
    print(f"Best intake: {best_intake:.2f}")
    print("Trajectories saved in: results/trajectories/")
    print("Summary: results/trajectories_summary.json")
    print()
    print("Next step: Use generate_video_from_trajectory.py to create videos")
    print("=" * 80)
