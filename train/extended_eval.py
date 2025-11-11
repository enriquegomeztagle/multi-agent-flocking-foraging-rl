"""
Extended Evaluation: 100 episodios para encontrar los mejores

Objetivo:
- Evaluar modelo final con 100 episodios
- Guardar TODOS los resultados con seeds
- Identificar top 10 mejores episodios
- Calcular percentiles (p50, p75, p90, p95, p99)
- Guardar seeds para reproducir mejores episodios

Tiempo estimado: ~12-15 minutos
"""

import json
import time
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO

from env.flockforage_parallel import FlockForageParallel, EnvConfig
from metrics.fairness import gini


def evaluate_extended(model_path: str, n_episodes: int = 100):
    """
    Evaluación extendida con múltiples episodios.

    Args:
        model_path: Path al modelo entrenado
        n_episodes: Número de episodios a evaluar (default: 100)

    Returns:
        dict con resultados completos
    """
    print("=" * 80)
    print("EVALUACIÓN EXTENDIDA - BÚSQUEDA DE MEJORES EPISODIOS")
    print("=" * 80)
    print()
    print(f"Model: {model_path}")
    print(f"Episodes: {n_episodes}")
    print(f"Estimated time: ~{n_episodes * 8 / 60:.1f} minutes")
    print("=" * 80)
    print()

    # Load model
    print("Loading model...")
    model = PPO.load(model_path)
    print("✅ Model loaded!")
    print()

    # Environment config
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

    # Storage for all episodes
    all_episodes = []

    print(f"Evaluating {n_episodes} episodes...")
    print()

    start_time = time.time()

    for ep in range(n_episodes):
        # Use different seed for each episode
        seed = ep * 42
        obs, _ = env.reset(seed=seed)

        ep_reward = 0
        step_count = 0

        # Run episode
        for step in range(env_cfg["episode_len"]):
            obs_array = np.array([obs[agent] for agent in env.agents])
            action, _ = model.predict(obs_array, deterministic=True)
            action_dict = {agent: int(action[i]) for i, agent in enumerate(env.agents)}

            obs, rew, terms, truncs, _ = env.step(action_dict)
            ep_reward += sum(rew.values())
            step_count += 1

            if all(terms.values()) or all(truncs.values()):
                break

        # Get final metrics
        ep_intake = float(env._intake_total.sum())
        gini_val = float(gini(env._intake_total)) if len(env._intake_total) > 0 else 0.0

        # Store episode data
        episode_data = {
            "episode": ep + 1,
            "seed": seed,
            "intake": ep_intake,
            "reward": float(ep_reward),
            "gini": gini_val,
            "steps": step_count,
            "efficiency_percent": (ep_intake / 2307) * 100,  # vs baseline
        }
        all_episodes.append(episode_data)

        # Progress update every 10 episodes
        if (ep + 1) % 10 == 0:
            elapsed = (time.time() - start_time) / 60
            eta = (elapsed / (ep + 1)) * (n_episodes - ep - 1)
            current_mean = np.mean([e["intake"] for e in all_episodes])
            current_max = np.max([e["intake"] for e in all_episodes])

            print(
                f"  [{ep+1:3d}/{n_episodes}] "
                f"Mean: {current_mean:6.1f} | Max: {current_max:6.1f} | "
                f"Elapsed: {elapsed:4.1f}min | ETA: {eta:4.1f}min"
            )

    total_time = (time.time() - start_time) / 60
    print()
    print(f"✅ Evaluation completed in {total_time:.1f} minutes")
    print()

    # Calculate statistics
    intakes = [e["intake"] for e in all_episodes]
    efficiencies = [e["efficiency_percent"] for e in all_episodes]

    stats = {
        "mean": float(np.mean(intakes)),
        "median": float(np.median(intakes)),
        "std": float(np.std(intakes)),
        "min": float(np.min(intakes)),
        "max": float(np.max(intakes)),
        "p50": float(np.percentile(intakes, 50)),
        "p75": float(np.percentile(intakes, 75)),
        "p90": float(np.percentile(intakes, 90)),
        "p95": float(np.percentile(intakes, 95)),
        "p99": float(np.percentile(intakes, 99)),
        "mean_efficiency": float(np.mean(efficiencies)),
        "median_efficiency": float(np.median(efficiencies)),
    }

    # Find top episodes
    sorted_episodes = sorted(all_episodes, key=lambda x: x["intake"], reverse=True)
    top_10 = sorted_episodes[:10]
    top_5 = sorted_episodes[:5]

    # Count episodes by performance tier
    tiers = {
        "excellent_20plus": len([e for e in all_episodes if e["efficiency_percent"] >= 20]),
        "great_15_20": len([e for e in all_episodes if 15 <= e["efficiency_percent"] < 20]),
        "good_10_15": len([e for e in all_episodes if 10 <= e["efficiency_percent"] < 15]),
        "ok_5_10": len([e for e in all_episodes if 5 <= e["efficiency_percent"] < 10]),
        "poor_below_5": len([e for e in all_episodes if e["efficiency_percent"] < 5]),
    }

    # Display results
    print("=" * 80)
    print("STATISTICS")
    print("=" * 80)
    print(f"Mean intake:      {stats['mean']:.2f} ± {stats['std']:.2f}")
    print(f"Median intake:    {stats['median']:.2f}")
    print(f"Min intake:       {stats['min']:.2f}")
    print(f"Max intake:       {stats['max']:.2f}")
    print()
    print("PERCENTILES:")
    print(f"  p50 (median):   {stats['p50']:.2f}")
    print(f"  p75:            {stats['p75']:.2f}")
    print(f"  p90:            {stats['p90']:.2f}")
    print(f"  p95:            {stats['p95']:.2f}")
    print(f"  p99:            {stats['p99']:.2f}")
    print()
    print("EFFICIENCY (vs Baseline 2,307):")
    print(f"  Mean:           {stats['mean_efficiency']:.2f}%")
    print(f"  Median:         {stats['median_efficiency']:.2f}%")
    print("=" * 80)
    print()

    print("=" * 80)
    print("TOP 10 BEST EPISODES")
    print("=" * 80)
    for i, ep in enumerate(top_10, 1):
        marker = "🏆" if i == 1 else "🌟" if i <= 3 else "⭐" if i <= 5 else "  "
        print(
            f"{marker} #{i:2d}  Episode {ep['episode']:3d} (seed={ep['seed']:5d}): "
            f"{ep['intake']:6.2f} intake ({ep['efficiency_percent']:5.2f}%) | "
            f"Gini: {ep['gini']:.3f}"
        )
    print("=" * 80)
    print()

    print("=" * 80)
    print("PERFORMANCE TIERS")
    print("=" * 80)
    print(f"🏆 Excellent (≥20%):  {tiers['excellent_20plus']:3d} episodes ({tiers['excellent_20plus']/n_episodes*100:5.1f}%)")
    print(f"🌟 Great (15-20%):    {tiers['great_15_20']:3d} episodes ({tiers['great_15_20']/n_episodes*100:5.1f}%)")
    print(f"⭐ Good (10-15%):     {tiers['good_10_15']:3d} episodes ({tiers['good_10_15']/n_episodes*100:5.1f}%)")
    print(f"   OK (5-10%):       {tiers['ok_5_10']:3d} episodes ({tiers['ok_5_10']/n_episodes*100:5.1f}%)")
    print(f"   Poor (<5%):       {tiers['poor_below_5']:3d} episodes ({tiers['poor_below_5']/n_episodes*100:5.1f}%)")
    print("=" * 80)
    print()

    # Compile full results
    results = {
        "model_path": model_path,
        "n_episodes": n_episodes,
        "evaluation_time_min": total_time,
        "baseline_intake": 2307,
        "statistics": stats,
        "performance_tiers": tiers,
        "top_10_episodes": top_10,
        "top_5_episodes": top_5,
        "all_episodes": all_episodes,
    }

    return results


def main():
    """Run extended evaluation."""
    model_path = "models/ppo_final/final/model"
    n_episodes = 100  # Evaluar 100 episodios

    # Check model exists
    if not Path(model_path + ".zip").exists():
        print(f"❌ Model not found: {model_path}")
        return

    # Run evaluation
    results = evaluate_extended(model_path, n_episodes=n_episodes)

    # Save results
    output_file = Path("results/extended_evaluation.json")
    output_file.parent.mkdir(exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print()
    print(f"✅ Results saved to: {output_file}")
    print()

    # Also save just top episodes for easy reference
    top_episodes_file = Path("results/top_episodes.json")
    top_data = {
        "best_episode": results["top_5_episodes"][0],
        "top_5": results["top_5_episodes"],
        "top_10": results["top_10_episodes"],
        "how_to_reproduce": "Use the 'seed' value when calling env.reset(seed=SEED)",
        "statistics": results["statistics"],
    }

    with open(top_episodes_file, "w") as f:
        json.dump(top_data, f, indent=2)

    print(f"✅ Top episodes saved to: {top_episodes_file}")
    print()

    # Summary
    best = results["top_5_episodes"][0]
    print("=" * 80)
    print("🏆 BEST EPISODE FOUND")
    print("=" * 80)
    print(f"Episode:     {best['episode']}")
    print(f"Seed:        {best['seed']}")
    print(f"Intake:      {best['intake']:.2f}")
    print(f"Efficiency:  {best['efficiency_percent']:.2f}%")
    print(f"Reward:      {best['reward']:.2f}")
    print(f"Gini:        {best['gini']:.3f}")
    print()
    print("To reproduce this episode:")
    print(f"  env.reset(seed={best['seed']})")
    print("=" * 80)


if __name__ == "__main__":
    main()
