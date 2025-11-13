"""
Evaluate baseline Boids agent (classical rules, no RL) on environment.

This script:
1. Loads environment configuration
2. Creates ClassicalBoidsAgent
3. Runs episodes and collects metrics
4. Saves results for comparison with RL
"""

import json
import time
import argparse
from pathlib import Path
import numpy as np

from env.flockforage_parallel import FlockForageParallel, EnvConfig
from env.boids_agent import ClassicalBoidsAgent
from metrics.fairness import gini
from metrics.flocking import polarization, mean_knn_distance, separation_violations


def evaluate_baseline_boids(
    config: dict,
    n_episodes: int = 100,
    output_path: str = None
):
    """
    Evaluate classical Boids agent on environment configuration.
    
    Args:
        config: Environment configuration dictionary
        n_episodes: Number of episodes to evaluate
        output_path: Path to save results JSON
        
    Returns:
        dict with results
    """
    print("=" * 80)
    print("EVALUATION: CLASSICAL BOIDS BASELINE (NO RL)")
    print("=" * 80)
    print()
    print(f"Configuration:")
    print(f"  - Agents: {config['n_agents']}")
    print(f"  - Patches: {config['n_patches']}")
    print(f"  - World: {config['width']}x{config['height']}")
    print(f"  - Episode length: {config['episode_len']}")
    print(f"Episodes: {n_episodes}")
    
    # Calculate theoretical maximum
    theoretical_max = config['n_agents'] * config['episode_len'] * config['c_max']
    print(f"Theoretical max: {theoretical_max:.0f} intake")
    print("=" * 80)
    print()
    
    # Create environment
    env = FlockForageParallel(EnvConfig(**config))
    
    # Create Boids agent
    boids_agent = ClassicalBoidsAgent(
        cohesion_weight=1.0,
        alignment_weight=1.0,
        separation_weight=1.5,
        foraging_weight=2.0,
        separation_distance=config.get('d_safe', 0.6) * 2.0,
        max_steering_force=0.3
    )
    
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
        
        # Track metrics during episode
        polarizations = []
        mean_distances = []
        separation_viols = []
        
        # Run episode
        for step in range(config['episode_len']):
            # Compute actions for all agents using Boids rules
            actions = {}
            for i, agent in enumerate(env.agents):
                action = boids_agent.compute_action(i, env)
                actions[agent] = action
            
            # Step environment
            obs, rew, terms, truncs, _ = env.step(actions)
            ep_reward += sum(rew.values())
            step_count += 1
            
            # Compute flocking metrics
            if env._neighbors is not None:
                vel = env._vel
                distances = env._distances
                
                polarizations.append(polarization(vel))
                mean_distances.append(mean_knn_distance(distances))
                separation_viols.append(
                    separation_violations(distances, env.cfg.d_safe)
                )
            
            if all(terms.values()) or all(truncs.values()):
                break
        
        # Get final metrics
        ep_intake = float(env._intake_total.sum())
        gini_val = float(gini(env._intake_total)) if len(env._intake_total) > 0 else 0.0
        
        # Average flocking metrics
        avg_polarization = np.mean(polarizations) if polarizations else 0.0
        avg_mean_distance = np.mean(mean_distances) if mean_distances else 0.0
        avg_separation_viol = np.mean(separation_viols) if separation_viols else 0.0
        
        # Store episode data
        episode_data = {
            "episode": ep + 1,
            "seed": seed,
            "intake": ep_intake,
            "reward": float(ep_reward),
            "gini": gini_val,
            "steps": step_count,
            "efficiency_percent": (ep_intake / theoretical_max) * 100,
            "polarization": float(avg_polarization),
            "mean_neighbor_distance": float(avg_mean_distance),
            "separation_violations": float(avg_separation_viol),
            "agent_intakes": [float(x) for x in env._intake_total],
            "min_agent_intake": float(np.min(env._intake_total)),
            "max_agent_intake": float(np.max(env._intake_total)),
            "std_agent_intake": float(np.std(env._intake_total)),
        }
        all_episodes.append(episode_data)
        
        # Progress update every 10 episodes
        if (ep + 1) % 10 == 0:
            avg_efficiency = np.mean([e["efficiency_percent"] for e in all_episodes[-10:]])
            print(f"Episode {ep + 1}/{n_episodes} - Avg efficiency (last 10): {avg_efficiency:.2f}%")
    
    elapsed_time = time.time() - start_time
    
    # Compute summary statistics
    efficiencies = [e["efficiency_percent"] for e in all_episodes]
    intakes = [e["intake"] for e in all_episodes]
    ginis = [e["gini"] for e in all_episodes]
    polarizations = [e["polarization"] for e in all_episodes]
    
    results = {
        "config": config,
        "theoretical_max": theoretical_max,
        "n_episodes": n_episodes,
        "evaluation_time_seconds": elapsed_time,
        "summary": {
            "mean_efficiency_percent": float(np.mean(efficiencies)),
            "std_efficiency_percent": float(np.std(efficiencies)),
            "min_efficiency_percent": float(np.min(efficiencies)),
            "max_efficiency_percent": float(np.max(efficiencies)),
            "mean_intake": float(np.mean(intakes)),
            "std_intake": float(np.std(intakes)),
            "mean_gini": float(np.mean(ginis)),
            "std_gini": float(np.std(ginis)),
            "mean_polarization": float(np.mean(polarizations)),
            "std_polarization": float(np.std(polarizations)),
        },
        "episodes": all_episodes
    }
    
    # Print summary
    print()
    print("=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Mean efficiency: {results['summary']['mean_efficiency_percent']:.2f}%")
    print(f"Std efficiency:  {results['summary']['std_efficiency_percent']:.2f}%")
    print(f"Min efficiency:  {results['summary']['min_efficiency_percent']:.2f}%")
    print(f"Max efficiency:  {results['summary']['max_efficiency_percent']:.2f}%")
    print()
    print(f"Mean intake:      {results['summary']['mean_intake']:.2f}")
    print(f"Mean Gini:       {results['summary']['mean_gini']:.4f}")
    print(f"Mean polarization: {results['summary']['mean_polarization']:.4f}")
    print()
    print(f"Evaluation time: {elapsed_time:.1f} seconds")
    print("=" * 80)
    print()
    
    # Save results
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✅ Results saved to: {output_file}")
        print()
    
    return results


def main():
    """Main evaluation script."""
    parser = argparse.ArgumentParser(
        description="Evaluate classical Boids baseline agent"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to environment config YAML file (e.g., configs/env_easy_mode.yaml)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for results JSON (e.g., results/baseline_boids_easy_mode.json)"
    )
    
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of episodes to evaluate (default: 100)"
    )
    
    args = parser.parse_args()
    
    # Load config
    import yaml
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Evaluate
    evaluate_baseline_boids(
        config=config,
        n_episodes=args.episodes,
        output_path=args.output
    )


if __name__ == "__main__":
    main()

