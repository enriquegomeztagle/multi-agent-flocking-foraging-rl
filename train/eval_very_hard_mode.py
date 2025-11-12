"""
Evaluate models on VERY HARD MODE configuration.

This script:
1. Loads the very hard mode configuration (12 agents, 10 patches, 35×35 world)
2. Evaluates trained model performance
3. Calculates theoretical maximum and efficiency percentages
4. Provides detailed statistics and performance analysis
"""

import json
import time
import argparse
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from env.flockforage_parallel import FlockForageParallel, EnvConfig
from env.gym_wrapper import FlockForageGymWrapper
from metrics.fairness import gini


# Very Hard Mode Configuration V6 (from env_very_hard_mode.yaml)
VERY_HARD_CONFIG = {
    "n_agents": 12,
    "n_patches": 10,        # V6: 10 patches (extreme scarcity)
    "width": 35.0,          # V6: 35×35 world (large exploration challenge)
    "height": 35.0,
    "episode_len": 2500,
    "feed_radius": 2.8,     # V6: 2.8 (very small feed zone)
    "c_max": 0.058,         # V6: 0.058 (very slow intake)
    "S_max": 2.0,           # V6: 2.0 (quick depletion)
    "regen_r": 0.24,        # V6: 0.24 (very slow regen)
    "k_neighbors": 7,
    "v_max": 1.5,
    "a_max": 0.20,
    "turn_max": 0.30,
    "d_safe": 0.8,
    "S_thr": 0.3,
    "dt": 0.2,
}

# Theoretical maximum: 12 agents × 2500 steps × 0.058 c_max = 1740 total
THEORETICAL_MAX = VERY_HARD_CONFIG["n_agents"] * VERY_HARD_CONFIG["episode_len"] * VERY_HARD_CONFIG["c_max"]


def evaluate_model(model_path: str, n_episodes: int = 100, vecnormalize_path: str = None):
    """
    Evaluate trained model on very hard configuration.

    Args:
        model_path: Path to trained PPO model
        n_episodes: Number of episodes to evaluate
        vecnormalize_path: Path to VecNormalize stats (optional)

    Returns:
        dict with results
    """
    print("==" * 40)
    print("EVALUATION: RL MODEL ON VERY HARD MODE V6")
    print("==" * 40)
    print()
    print(f"Model: {model_path}")
    print(f"Configuration: Very Hard Mode V6 (12 agents, 10 patches, 35×35 world)")
    print(f"Episodes: {n_episodes}")
    print(f"Theoretical max: {THEORETICAL_MAX:.0f} intake")
    print(f"Target: 15-25% efficiency (261-435 intake)")
    print("==" * 40)
    print()

    # Load model
    print("Loading model...")
    model = PPO.load(model_path)
    print("✅ Model loaded!")
    print()

    # Create environment with gym wrapper
    env = FlockForageGymWrapper(EnvConfig(**VERY_HARD_CONFIG))

    # Storage
    all_episodes = []

    print(f"Evaluating {n_episodes} episodes...")
    print()

    start_time = time.time()

    for ep in range(n_episodes):
        seed = ep * 42
        obs, _ = env.reset(seed=seed)

        ep_reward = 0
        step_count = 0

        # Run episode
        for step in range(VERY_HARD_CONFIG["episode_len"]):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            step_count += 1

            if terminated or truncated:
                break

        # Get final metrics from underlying PettingZoo environment
        ep_intake = float(env.env._intake_total.sum())
        gini_val = float(gini(env.env._intake_total)) if len(env.env._intake_total) > 0 else 0.0

        # Additional metrics
        agent_intakes = [float(x) for x in env.env._intake_total]
        min_intake = float(np.min(env.env._intake_total))
        max_intake = float(np.max(env.env._intake_total))
        std_intake = float(np.std(env.env._intake_total))

        # Store episode data
        episode_data = {
            "episode": ep + 1,
            "seed": seed,
            "intake": ep_intake,
            "reward": float(ep_reward),
            "gini": gini_val,
            "steps": step_count,
            "efficiency_percent": (ep_intake / THEORETICAL_MAX) * 100,
            "agent_intakes": agent_intakes,
            "min_agent_intake": min_intake,
            "max_agent_intake": max_intake,
            "std_agent_intake": std_intake,
        }
        all_episodes.append(episode_data)

        # Progress update every 10 episodes
        if (ep + 1) % 10 == 0:
            elapsed = (time.time() - start_time) / 60
            eta = (elapsed / (ep + 1)) * (n_episodes - ep - 1)
            current_mean = np.mean([e["intake"] for e in all_episodes])
            current_max = np.max([e["intake"] for e in all_episodes])
            current_eff = (current_mean / THEORETICAL_MAX) * 100

            print(
                f"  [{ep+1:3d}/{n_episodes}] "
                f"Mean: {current_mean:6.1f} ({current_eff:5.1f}%) | "
                f"Max: {current_max:6.1f} | "
                f"Elapsed: {elapsed:4.1f}min | ETA: {eta:4.1f}min"
            )

    total_time = (time.time() - start_time) / 60
    print()
    print(f"✅ Evaluation completed in {total_time:.1f} minutes")
    print()

    # Calculate statistics
    intakes = [e["intake"] for e in all_episodes]
    efficiencies = [e["efficiency_percent"] for e in all_episodes]
    ginis = [e["gini"] for e in all_episodes]

    stats = {
        "mean": float(np.mean(intakes)),
        "median": float(np.median(intakes)),
        "std": float(np.std(intakes)),
        "min": float(np.min(intakes)),
        "max": float(np.max(intakes)),
        "p25": float(np.percentile(intakes, 25)),
        "p50": float(np.percentile(intakes, 50)),
        "p75": float(np.percentile(intakes, 75)),
        "p90": float(np.percentile(intakes, 90)),
        "p95": float(np.percentile(intakes, 95)),
        "p99": float(np.percentile(intakes, 99)),
        "mean_efficiency": float(np.mean(efficiencies)),
        "median_efficiency": float(np.median(efficiencies)),
        "mean_gini": float(np.mean(ginis)),
        "median_gini": float(np.median(ginis)),
    }

    # Find top episodes
    sorted_episodes = sorted(all_episodes, key=lambda x: x["intake"], reverse=True)
    top_10 = sorted_episodes[:10]
    top_5 = sorted_episodes[:5]

    # Count episodes by performance tier (relative to theoretical max)
    tiers = {
        "excellent_30plus": len([e for e in all_episodes if e["efficiency_percent"] >= 30]),
        "great_25_30": len([e for e in all_episodes if 25 <= e["efficiency_percent"] < 30]),
        "good_20_25": len([e for e in all_episodes if 20 <= e["efficiency_percent"] < 25]),
        "ok_15_20": len([e for e in all_episodes if 15 <= e["efficiency_percent"] < 20]),
        "below_15": len([e for e in all_episodes if e["efficiency_percent"] < 15]),
    }

    # Display results
    print("==" * 40)
    print("STATISTICS")
    print("==" * 40)
    print(f"Mean intake:      {stats['mean']:.2f} ± {stats['std']:.2f}")
    print(f"Median intake:    {stats['median']:.2f}")
    print(f"Min intake:       {stats['min']:.2f}")
    print(f"Max intake:       {stats['max']:.2f}")
    print()
    print("PERCENTILES:")
    print(f"  p25:            {stats['p25']:.2f}")
    print(f"  p50 (median):   {stats['p50']:.2f}")
    print(f"  p75:            {stats['p75']:.2f}")
    print(f"  p90:            {stats['p90']:.2f}")
    print(f"  p95:            {stats['p95']:.2f}")
    print(f"  p99:            {stats['p99']:.2f}")
    print()
    print(f"EFFICIENCY (vs Theoretical Max {THEORETICAL_MAX:.0f}):")
    print(f"  Mean:           {stats['mean_efficiency']:.2f}%")
    print(f"  Median:         {stats['median_efficiency']:.2f}%")
    print()
    print(f"FAIRNESS:")
    print(f"  Mean Gini:      {stats['mean_gini']:.3f}")
    print(f"  Median Gini:    {stats['median_gini']:.3f}")
    print("==" * 40)
    print()

    print("==" * 40)
    print("TOP 10 BEST EPISODES")
    print("==" * 40)
    for i, ep in enumerate(top_10, 1):
        marker = "🏆" if i == 1 else "🌟" if i <= 3 else "⭐" if i <= 5 else "  "
        print(
            f"{marker} #{i:2d}  Episode {ep['episode']:3d} (seed={ep['seed']:5d}): "
            f"{ep['intake']:6.2f} intake ({ep['efficiency_percent']:5.2f}%) | "
            f"Gini: {ep['gini']:.3f}"
        )
    print("==" * 40)
    print()

    print("==" * 40)
    print("PERFORMANCE TIERS (vs Theoretical Max)")
    print("==" * 40)
    print(f"🏆 Excellent (≥30%):  {tiers['excellent_30plus']:3d} episodes ({tiers['excellent_30plus']/n_episodes*100:5.1f}%)")
    print(f"🌟 Great (25-30%):    {tiers['great_25_30']:3d} episodes ({tiers['great_25_30']/n_episodes*100:5.1f}%)")
    print(f"⭐ Good (20-25%):     {tiers['good_20_25']:3d} episodes ({tiers['good_20_25']/n_episodes*100:5.1f}%)")
    print(f"   OK (15-20%):       {tiers['ok_15_20']:3d} episodes ({tiers['ok_15_20']/n_episodes*100:5.1f}%)")
    print(f"   Below 15%:         {tiers['below_15']:3d} episodes ({tiers['below_15']/n_episodes*100:5.1f}%)")
    print("==" * 40)
    print()

    # Performance assessment for very hard mode
    if stats['mean_efficiency'] >= 25.0:
        print("🎉 🎉 🎉 EXCEPTIONAL! Exceeded 25% efficiency on VERY HARD MODE! 🎉 🎉 🎉")
    elif stats['mean_efficiency'] >= 20.0:
        print("✅ Excellent performance! 20-25% efficiency achieved - within target!")
    elif stats['mean_efficiency'] >= 15.0:
        print("⭐ Good performance! 15-20% efficiency - at target range lower bound.")
    elif stats['mean_efficiency'] >= 10.0:
        print("📈 Moderate performance. The model is learning but below target.")
    else:
        print("⚠️  Below 10% efficiency. Very challenging - consider longer training.")
    print()

    # Compile full results
    results = {
        "model_path": model_path,
        "configuration": "very_hard_mode",
        "config_params": VERY_HARD_CONFIG,
        "theoretical_max": THEORETICAL_MAX,
        "n_episodes": n_episodes,
        "evaluation_time_min": total_time,
        "statistics": stats,
        "performance_tiers": tiers,
        "top_10_episodes": top_10,
        "top_5_episodes": top_5,
        "all_episodes": all_episodes,
    }

    return results


def main():
    """Run very hard mode evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate model on very hard mode configuration")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained PPO model (e.g., models/ppo_very_hard_mode/best_model/best_model)"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of episodes to evaluate (default: 100)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/very_hard_mode_evaluation.json",
        help="Output file path"
    )
    parser.add_argument(
        "--vecnormalize",
        type=str,
        default=None,
        help="Path to VecNormalize stats (optional)"
    )

    args = parser.parse_args()

    # Check model exists
    if not Path(args.model + ".zip").exists():
        print(f"❌ Model not found: {args.model}.zip")
        print()
        print("Make sure to train the model first:")
        print("  python train/train_ppo.py --config configs/env_very_hard_mode.yaml --output models/ppo_very_hard_mode")
        return

    # Run evaluation
    results = evaluate_model(
        args.model,
        n_episodes=args.episodes,
        vecnormalize_path=args.vecnormalize
    )

    # Save results
    output_file = Path(args.output)
    output_file.parent.mkdir(exist_ok=True, parents=True)

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"✅ Results saved to: {output_file}")
    print()

    # Summary
    best = results["top_5_episodes"][0]
    print("==" * 40)
    print("🏆 BEST EPISODE")
    print("==" * 40)
    print(f"Episode:     {best['episode']}")
    print(f"Seed:        {best['seed']}")
    print(f"Intake:      {best['intake']:.2f}")
    print(f"Efficiency:  {best['efficiency_percent']:.2f}%")
    print(f"Reward:      {best['reward']:.2f}")
    print(f"Gini:        {best['gini']:.3f}")
    print()
    print("To reproduce this episode:")
    print(f"  env.reset(seed={best['seed']})")
    print("==" * 40)


if __name__ == "__main__":
    main()
