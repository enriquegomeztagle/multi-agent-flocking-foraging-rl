"""
Compare baseline Boids controller vs RL-trained agent.

"""

import yaml
import json
import os
import numpy as np
from pathlib import Path
from sb3_contrib import RecurrentPPO
import supersuit as ss
from env.flockforage_parallel import FlockForageParallel, EnvConfig
from train.baseline_boids import BoidsController, evaluate_baseline
from metrics.fairness import gini
from metrics.flocking import polarization, mean_knn_distance, separation_violations
from metrics.sustainability import stock_score, min_stock_normalized


def evaluate_rl_agent(
    env, model, n_episodes: int = 10, max_steps: int = 1500, verbose: bool = True
) -> dict:
    """
    Evaluate RL agent on environment.

    Args:
        env: FlockForageParallel environment
        model: Trained RecurrentPPO model
        n_episodes: Number of evaluation episodes
        max_steps: Maximum steps per episode
        verbose: Print progress

    Returns:
        Dictionary of aggregated metrics
    """
    all_metrics = []

    # Reset LSTM states
    lstm_states = None

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep * 100)
        episode_starts = np.ones((len(env.agents),), dtype=bool)

        # Track episode metrics
        total_rewards = []
        polarizations = []
        knn_distances = []
        separations = []
        stock_history = []

        for step in range(max_steps):
            # Convert dict obs to array for vectorized prediction
            obs_array = np.array([obs[agent] for agent in env.agents])

            # Get actions from RL model
            actions, lstm_states = model.predict(
                obs_array,
                state=lstm_states,
                episode_start=episode_starts,
                deterministic=True,
            )
            episode_starts = np.zeros((len(env.agents),), dtype=bool)

            # Convert to dict format
            action_dict = {agent: int(actions[i]) for i, agent in enumerate(env.agents)}

            # Step environment
            obs, rewards, terminations, truncations, infos = env.step(action_dict)

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
            "mean_reward_per_step": (
                float(np.mean(total_rewards)) if total_rewards else 0.0
            ),
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
            "stock_final_min": float(
                min_stock_normalized(stock_history[-1], env.cfg.S_max)
            ),
            "stock_mean_over_time": float(
                np.mean([stock_score(s, env.cfg.S_max) for s in stock_history])
            ),
        }

        all_metrics.append(metrics)

        if verbose:
            print(
                f"Episode {ep+1}/{n_episodes}: Reward={metrics['mean_reward_per_step']:.3f}, "
                f"Gini={metrics['gini']:.3f}, Polarization={metrics['polarization_mean']:.3f}"
            )

    # Aggregate metrics
    aggregated = {
        key: {
            "mean": float(np.mean([m[key] for m in all_metrics])),
            "std": float(np.std([m[key] for m in all_metrics])),
        }
        for key in all_metrics[0].keys()
    }

    return {"individual_episodes": all_metrics, "aggregated": aggregated}


def print_comparison_table(baseline_results: dict, rl_results: dict):
    """Print side-by-side comparison table."""
    print("\n" + "=" * 90)
    print("COMPARATIVE ANALYSIS: BASELINE BOIDS vs RL AGENT")
    print("=" * 90)
    print(
        f"{'Metric':<30} {'Baseline (Boids)':<25} {'RL (RecurrentPPO)':<25} {'Winner':<10}"
    )
    print("-" * 90)

    baseline_agg = baseline_results["aggregated"]
    rl_agg = rl_results["aggregated"]

    # Define metrics where higher is better
    higher_is_better = {
        "mean_reward_per_step": True,
        "intake_mean": True,
        "polarization_mean": True,
        "stock_final_mean": True,
        "stock_final_min": True,
        "stock_mean_over_time": True,
    }

    # Define metrics where lower is better
    lower_is_better = {
        "gini": True,
        "intake_std": True,
        "knn_distance_mean": False,  # Depends on context
        "separation_violations_mean": True,
    }

    for key in baseline_agg.keys():
        if key == "episode_length" or key == "total_reward":
            continue  # Skip these

        b_val = baseline_agg[key]["mean"]
        b_std = baseline_agg[key]["std"]
        r_val = rl_agg[key]["mean"]
        r_std = rl_agg[key]["std"]

        baseline_str = f"{b_val:7.4f} ± {b_std:6.4f}"
        rl_str = f"{r_val:7.4f} ± {r_std:6.4f}"

        # Determine winner
        winner = ""
        if key in higher_is_better:
            if r_val > b_val * 1.05:  # 5% margin
                winner = "✅ RL"
            elif b_val > r_val * 1.05:
                winner = "🔵 Baseline"
            else:
                winner = "➖ Tie"
        elif key in lower_is_better:
            if r_val < b_val * 0.95:  # 5% margin
                winner = "✅ RL"
            elif b_val < r_val * 0.95:
                winner = "🔵 Baseline"
            else:
                winner = "➖ Tie"

        print(f"{key:<30} {baseline_str:<25} {rl_str:<25} {winner:<10}")

    print("=" * 90)


def compute_improvement_stats(baseline_results: dict, rl_results: dict) -> dict:
    """Compute percentage improvements from baseline to RL."""
    baseline_agg = baseline_results["aggregated"]
    rl_agg = rl_results["aggregated"]

    improvements = {}
    for key in baseline_agg.keys():
        if key == "episode_length" or key == "total_reward":
            continue

        b_val = baseline_agg[key]["mean"]
        r_val = rl_agg[key]["mean"]

        if abs(b_val) > 1e-6:
            pct_change = ((r_val - b_val) / abs(b_val)) * 100
            improvements[key] = {
                "baseline": b_val,
                "rl": r_val,
                "percent_change": pct_change,
            }

    return improvements


def main():
    """Run full comparison."""
    print("=" * 90)
    print("BASELINE vs RL COMPARISON EVALUATION")
    print("=" * 90)

    # Load configuration
    config_path = "configs/env_curriculum_phase2.yaml"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            env_cfg = yaml.safe_load(f)
        print(f"✅ Loaded config: {config_path}")
    else:
        print(f"⚠️  Config not found: {config_path}, using defaults")
        env_cfg = {}

    # === EVALUATE BASELINE ===
    print("\n" + "=" * 90)
    print("1. EVALUATING BASELINE BOIDS CONTROLLER")
    print("=" * 90)

    env_baseline = FlockForageParallel(EnvConfig(**env_cfg))
    controller = BoidsController(
        w_cohesion=1.0,
        w_alignment=1.0,
        w_separation=2.0,
        w_foraging=1.5,
        hunger_threshold=0.5,
    )

    baseline_results = evaluate_baseline(
        env_baseline, controller, n_episodes=10, verbose=True
    )
    env_baseline.close()

    # === EVALUATE RL AGENT ===
    print("\n" + "=" * 90)
    print("2. EVALUATING RL AGENT (RecurrentPPO)")
    print("=" * 90)

    # Check if model exists
    model_path = "models/advanced_final/recurrent_ppo_final.zip"
    if not os.path.exists(model_path):
        print(f"❌ Trained model not found at: {model_path}")
        print(
            "   Please train the model first using: python -m train.run_advanced_training"
        )
        return

    # Load model
    print(f"✅ Loading model: {model_path}")
    model = RecurrentPPO.load(model_path)

    # Create environment for RL evaluation
    env_rl = FlockForageParallel(EnvConfig(**env_cfg))

    rl_results = evaluate_rl_agent(env_rl, model, n_episodes=10, verbose=True)
    env_rl.close()

    # === COMPARISON ===
    print_comparison_table(baseline_results, rl_results)

    # Compute improvements
    improvements = compute_improvement_stats(baseline_results, rl_results)

    print("\n" + "=" * 90)
    print("KEY IMPROVEMENTS (RL vs Baseline)")
    print("=" * 90)
    for key, vals in improvements.items():
        print(
            f"{key:<30}: {vals['percent_change']:+7.2f}%  "
            f"({vals['baseline']:.4f} → {vals['rl']:.4f})"
        )

    # Save all results
    os.makedirs("results", exist_ok=True)

    comparison_results = {
        "baseline": baseline_results,
        "rl": rl_results,
        "improvements": improvements,
    }

    with open("results/comparison_baseline_vs_rl.json", "w") as f:
        json.dump(comparison_results, f, indent=2)

    print(f"\n✅ Results saved to: results/comparison_baseline_vs_rl.json")

    # Print summary
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print("✅ Objective O2: Classical metrics measured under baseline rules")
    print("✅ Objective O4: Gini index and group stability compared (baseline vs RL)")
    print("\nKey Findings:")

    # Highlight key metrics
    rl_reward = rl_results["aggregated"]["mean_reward_per_step"]["mean"]
    bl_reward = baseline_results["aggregated"]["mean_reward_per_step"]["mean"]
    rl_gini = rl_results["aggregated"]["gini"]["mean"]
    bl_gini = baseline_results["aggregated"]["gini"]["mean"]
    rl_pol = rl_results["aggregated"]["polarization_mean"]["mean"]
    bl_pol = baseline_results["aggregated"]["polarization_mean"]["mean"]

    print(
        f"  • Reward: RL {rl_reward:.3f} vs Baseline {bl_reward:.3f} "
        f"({((rl_reward/bl_reward - 1)*100):+.1f}%)"
    )
    print(
        f"  • Gini (fairness): RL {rl_gini:.3f} vs Baseline {bl_gini:.3f} "
        f"({'better' if rl_gini < bl_gini else 'worse'} fairness)"
    )
    print(
        f"  • Polarization (cohesion): RL {rl_pol:.3f} vs Baseline {bl_pol:.3f} "
        f"({'better' if rl_pol > bl_pol else 'worse'} alignment)"
    )

    print("=" * 90)


if __name__ == "__main__":
    main()
