"""
Evaluate models on EXPERT MODE configuration.

This script:
1. Loads the expert mode configuration (12 agents, 10 patches, 35×35 world)
2. Evaluates trained model performance
3. Calculates theoretical maximum and efficiency percentages
4. Provides detailed statistics and performance analysis
"""

import json
import time
import argparse
from pathlib import Path
import numpy as np
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from env.flockforage_parallel import FlockForageParallel, EnvConfig
from env.gym_wrapper import FlockForageGymWrapper
from metrics.fairness import gini


# Load Expert Mode Configuration from YAML
CONFIG_PATH = Path(__file__).parent.parent / "configs" / "env_expert.yaml"
with open(CONFIG_PATH, "r") as f:
    EXPERT_CONFIG = yaml.safe_load(f)

# Theoretical maximum: n_agents × episode_len × c_max
THEORETICAL_MAX = (
    EXPERT_CONFIG["n_agents"] * EXPERT_CONFIG["episode_len"] * EXPERT_CONFIG["c_max"]
)


def evaluate_model(
    model_path: str, n_episodes: int = 100, vecnormalize_path: str = None
):
    """
    Evaluate trained model on expert configuration.

    Args:
        model_path: Path to trained PPO model
        n_episodes: Number of episodes to evaluate
        vecnormalize_path: Path to VecNormalize stats (optional)

    Returns:
        dict with results
    """
    print("=" * 80)
    print("EVALUATION: RL MODEL ON EXPERT MODE")
    print("=" * 80)
    print()
    print(f"Model: {model_path}")
    print(f"Configuration: Expert Mode (12 agents, 10 patches, 35×35 world)")
    print(f"Episodes: {n_episodes}")
    print(f"Theoretical max: {THEORETICAL_MAX:.0f} intake")
    print(f"Target: 25-35% efficiency (Expected: 35-40%)")
    print("=" * 80)
    print()

    # Load model
    print("Loading model...")
    model = PPO.load(model_path)
    print("✅ Model loaded!")
    print()

    # Create environment with gym wrapper
    env = FlockForageGymWrapper(EnvConfig(**EXPERT_CONFIG))

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
        for step in range(EXPERT_CONFIG["episode_len"]):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            step_count += 1

            if terminated or truncated:
                break

        # Get final metrics from underlying PettingZoo environment
        ep_intake = float(env.env._intake_total.sum())
        gini_val = (
            float(gini(env.env._intake_total))
            if len(env.env._intake_total) > 0
            else 0.0
        )

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
        "excellent_30plus": len(
            [e for e in all_episodes if e["efficiency_percent"] >= 30]
        ),
        "great_25_30": len(
            [e for e in all_episodes if 25 <= e["efficiency_percent"] < 30]
        ),
        "good_20_25": len(
            [e for e in all_episodes if 20 <= e["efficiency_percent"] < 25]
        ),
        "ok_15_20": len(
            [e for e in all_episodes if 15 <= e["efficiency_percent"] < 20]
        ),
        "below_15": len([e for e in all_episodes if e["efficiency_percent"] < 15]),
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
    print("PERFORMANCE TIERS (vs Theoretical Max)")
    print("=" * 80)
    print(
        f"🏆 Excellent (≥30%):  {tiers['excellent_30plus']:3d} episodes ({tiers['excellent_30plus']/n_episodes*100:5.1f}%)"
    )
    print(
        f"🌟 Great (25-30%):    {tiers['great_25_30']:3d} episodes ({tiers['great_25_30']/n_episodes*100:5.1f}%)"
    )
    print(
        f"⭐ Good (20-25%):     {tiers['good_20_25']:3d} episodes ({tiers['good_20_25']/n_episodes*100:5.1f}%)"
    )
    print(
        f"   OK (15-20%):       {tiers['ok_15_20']:3d} episodes ({tiers['ok_15_20']/n_episodes*100:5.1f}%)"
    )
    print(
        f"   Below 15%:         {tiers['below_15']:3d} episodes ({tiers['below_15']/n_episodes*100:5.1f}%)"
    )
    print("=" * 80)
    print()

    # Performance assessment for expert mode
    if stats["mean_efficiency"] >= 35.0:
        print(
            "🎉 🎉 🎉 EXCELLENT! Achieved 35%+ efficiency on EXPERT MODE! (Expected: 35-40%) 🎉 🎉 🎉"
        )
    elif stats["mean_efficiency"] >= 25.0:
        print("✅ Target achieved! 25-35% efficiency meets target (Expected: 35-40%).")
    elif stats["mean_efficiency"] >= 20.0:
        print(
            "⭐ Good performance! 20-25% efficiency - below target but showing coordination."
        )
    elif stats["mean_efficiency"] >= 15.0:
        print("📈 Moderate performance. 15-20% efficiency - below target range.")
    else:
        print(
            "⚠️  Below 15% efficiency. Very challenging - consider longer training or tuning."
        )
    print()

    # Compile full results
    results = {
        "model_path": model_path,
        "configuration": "expert",
        "config_params": EXPERT_CONFIG,
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
    """Run expert mode evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate model on expert mode configuration"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/ppo_expert/final_model",
        help="Path to trained PPO model (default: models/ppo_expert/final_model)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of episodes to evaluate (default: 100)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/expert_evaluation.json",
        help="Output file path",
    )
    parser.add_argument(
        "--vecnormalize",
        type=str,
        default=None,
        help="Path to VecNormalize stats (optional)",
    )

    args = parser.parse_args()

    # Check model exists
    if not Path(args.model + ".zip").exists():
        print(f"❌ Model not found: {args.model}.zip")
        print()
        print("Train the model first:")
        print(
            "  python -m train.train_ppo --config configs/env_expert.yaml --output models/ppo_expert"
        )
        return

    # Run evaluation
    results = evaluate_model(
        args.model, n_episodes=args.episodes, vecnormalize_path=args.vecnormalize
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
    print("=" * 80)
    print("🏆 BEST EPISODE")
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
