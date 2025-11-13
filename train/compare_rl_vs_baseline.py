"""
Compare RL results vs. Baseline Boids results.

This script:
1. Loads RL evaluation results
2. Loads Baseline Boids evaluation results
3. Compares metrics side by side
4. Generates comparison report
"""

import json
import argparse
from pathlib import Path
import numpy as np


def load_results(json_path: str) -> dict:
    """Load evaluation results from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def compare_results(rl_results: dict, baseline_results: dict) -> dict:
    """
    Compare RL and Baseline results.
    
    Args:
        rl_results: Results from RL evaluation
        baseline_results: Results from Baseline Boids evaluation
        
    Returns:
        Comparison dictionary
    """
    rl_summary = rl_results.get("summary", {})
    baseline_summary = baseline_results.get("summary", {})
    
    # Calculate differences
    efficiency_diff = rl_summary.get("mean_efficiency_percent", 0) - baseline_summary.get("mean_efficiency_percent", 0)
    intake_diff = rl_summary.get("mean_intake", 0) - baseline_summary.get("mean_intake", 0)
    gini_diff = rl_summary.get("mean_gini", 0) - baseline_summary.get("mean_gini", 0)
    
    # Calculate improvement percentages
    baseline_eff = baseline_summary.get("mean_efficiency_percent", 1e-6)
    efficiency_improvement = (efficiency_diff / baseline_eff) * 100 if baseline_eff > 0 else 0
    
    comparison = {
        "rl": {
            "mean_efficiency": rl_summary.get("mean_efficiency_percent", 0),
            "std_efficiency": rl_summary.get("std_efficiency_percent", 0),
            "mean_intake": rl_summary.get("mean_intake", 0),
            "mean_gini": rl_summary.get("mean_gini", 0),
        },
        "baseline": {
            "mean_efficiency": baseline_summary.get("mean_efficiency_percent", 0),
            "std_efficiency": baseline_summary.get("std_efficiency_percent", 0),
            "mean_intake": baseline_summary.get("mean_intake", 0),
            "mean_gini": baseline_summary.get("mean_gini", 0),
        },
        "differences": {
            "efficiency_diff": efficiency_diff,
            "efficiency_improvement_percent": efficiency_improvement,
            "intake_diff": intake_diff,
            "gini_diff": gini_diff,
        },
        "better": {
            "efficiency": "RL" if efficiency_diff > 0 else "Baseline",
            "intake": "RL" if intake_diff > 0 else "Baseline",
            "fairness": "RL" if gini_diff < 0 else "Baseline",  # Lower Gini is better
        }
    }
    
    return comparison


def print_comparison(comparison: dict):
    """Print comparison results in a readable format."""
    print("=" * 80)
    print("COMPARISON: RL vs. BASELINE BOIDS")
    print("=" * 80)
    print()
    
    print("EFFICIENCY:")
    print(f"  RL:       {comparison['rl']['mean_efficiency']:.2f}% ± {comparison['rl']['std_efficiency']:.2f}%")
    print(f"  Baseline: {comparison['baseline']['mean_efficiency']:.2f}% ± {comparison['baseline']['std_efficiency']:.2f}%")
    print(f"  Difference: {comparison['differences']['efficiency_diff']:+.2f}%")
    print(f"  Improvement: {comparison['differences']['efficiency_improvement_percent']:+.1f}%")
    print(f"  Winner: {comparison['better']['efficiency']}")
    print()
    
    print("INTAKE:")
    print(f"  RL:       {comparison['rl']['mean_intake']:.2f}")
    print(f"  Baseline: {comparison['baseline']['mean_intake']:.2f}")
    print(f"  Difference: {comparison['differences']['intake_diff']:+.2f}")
    print(f"  Winner: {comparison['better']['intake']}")
    print()
    
    print("FAIRNESS (Gini - lower is better):")
    print(f"  RL:       {comparison['rl']['mean_gini']:.4f}")
    print(f"  Baseline: {comparison['baseline']['mean_gini']:.4f}")
    print(f"  Difference: {comparison['differences']['gini_diff']:+.4f}")
    print(f"  Winner: {comparison['better']['fairness']}")
    print()
    
    print("=" * 80)


def generate_markdown_report(comparison: dict, output_path: str):
    """Generate markdown comparison report."""
    report = f"""# Comparison: RL vs. Baseline Boids

## Summary

| Metric | RL | Baseline | Difference | Winner |
|--------|----|----------|------------|--------|
| **Efficiency** | {comparison['rl']['mean_efficiency']:.2f}% ± {comparison['rl']['std_efficiency']:.2f}% | {comparison['baseline']['mean_efficiency']:.2f}% ± {comparison['baseline']['std_efficiency']:.2f}% | {comparison['differences']['efficiency_diff']:+.2f}% ({comparison['differences']['efficiency_improvement_percent']:+.1f}%) | **{comparison['better']['efficiency']}** |
| **Intake** | {comparison['rl']['mean_intake']:.2f} | {comparison['baseline']['mean_intake']:.2f} | {comparison['differences']['intake_diff']:+.2f} | **{comparison['better']['intake']}** |
| **Fairness (Gini)** | {comparison['rl']['mean_gini']:.4f} | {comparison['baseline']['mean_gini']:.4f} | {comparison['differences']['gini_diff']:+.4f} | **{comparison['better']['fairness']}** |

## Detailed Analysis

### Efficiency
- **RL Performance**: {comparison['rl']['mean_efficiency']:.2f}% ± {comparison['rl']['std_efficiency']:.2f}%
- **Baseline Performance**: {comparison['baseline']['mean_efficiency']:.2f}% ± {comparison['baseline']['std_efficiency']:.2f}%
- **Improvement**: {comparison['differences']['efficiency_improvement_percent']:+.1f}% ({comparison['differences']['efficiency_diff']:+.2f} percentage points)

### Fairness
- **RL Gini**: {comparison['rl']['mean_gini']:.4f} (lower is better)
- **Baseline Gini**: {comparison['baseline']['mean_gini']:.4f}
- **Difference**: {comparison['differences']['gini_diff']:+.4f}

## Conclusion

{'RL outperforms Baseline Boids' if comparison['better']['efficiency'] == 'RL' else 'Baseline Boids outperforms RL'} in terms of efficiency.
{'RL achieves better fairness' if comparison['better']['fairness'] == 'RL' else 'Baseline achieves better fairness'} (lower Gini coefficient).
"""
    
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(f"✅ Markdown report saved to: {output_path}")


def main():
    """Main comparison script."""
    parser = argparse.ArgumentParser(
        description="Compare RL vs. Baseline Boids results"
    )
    
    parser.add_argument(
        "--rl-results",
        type=str,
        required=True,
        help="Path to RL evaluation results JSON (e.g., results/easy_mode_evaluation.json)"
    )
    
    parser.add_argument(
        "--baseline-results",
        type=str,
        required=True,
        help="Path to Baseline Boids evaluation results JSON (e.g., results/baseline_boids_easy_mode.json)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save comparison report (optional, e.g., results/comparison_rl_vs_baseline.md)"
    )
    
    args = parser.parse_args()
    
    # Load results
    print("Loading results...")
    rl_results = load_results(args.rl_results)
    baseline_results = load_results(args.baseline_results)
    print("✅ Results loaded")
    print()
    
    # Compare
    comparison = compare_results(rl_results, baseline_results)
    
    # Print comparison
    print_comparison(comparison)
    
    # Save report if requested
    if args.output:
        generate_markdown_report(comparison, args.output)
        print()
        
        # Also save JSON comparison
        json_output = Path(args.output).with_suffix('.json')
        with open(json_output, 'w') as f:
            json.dump(comparison, f, indent=2)
        print(f"✅ JSON comparison saved to: {json_output}")


if __name__ == "__main__":
    main()

