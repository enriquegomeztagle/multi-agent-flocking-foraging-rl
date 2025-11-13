# Usage Guide: Classical Baseline Boids System

This document explains how to use the classical Baseline Boids system (without RL) for comparison with RL results.

## 📋 Description

The Baseline system implements the classical Boids rules (Reynolds, 1987):
- **Cohesion**: Move towards the center of mass of neighbors
- **Alignment**: Align velocity with the average velocity of neighbors
- **Separation**: Avoid nearby neighbors
- **Foraging**: Move towards the nearest food patch

This allows comparing the behavior learned by RL with a rule-based system.

## 🚀 Quick Start

### 1. Evaluate Baseline Boids

```bash
# Evaluate in Easy mode
PYTHONPATH=. python train/eval_baseline_boids.py \
  --config configs/env_easy_mode.yaml \
  --output results/baseline_boids_easy_mode.json \
  --episodes 100

# Evaluate in Medium mode
PYTHONPATH=. python train/eval_baseline_boids.py \
  --config configs/env_medium_mode.yaml \
  --output results/baseline_boids_medium_mode.json \
  --episodes 100
```

### 2. Compare with RL

```bash
# Compare Easy Mode results
PYTHONPATH=. python train/compare_rl_vs_baseline.py \
  --rl-results results/easy_mode_evaluation.json \
  --baseline-results results/baseline_boids_easy_mode.json \
  --output results/comparison_easy_mode.md
```

## 📊 Boids Agent Parameters

The `ClassicalBoidsAgent` has the following configurable parameters:

```python
ClassicalBoidsAgent(
    cohesion_weight=1.0,      # Weight of the cohesion rule
    alignment_weight=1.0,     # Weight of the alignment rule
    separation_weight=1.5,    # Weight of the separation rule
    foraging_weight=2.0,      # Weight of the foraging behavior
    separation_distance=2.0,  # Threshold distance for separation
    max_steering_force=0.3     # Maximum steering force
)
```

### Parameter Tuning

You can modify these weights in the `eval_baseline_boids.py` script to experiment with different behaviors:

- **Higher `cohesion_weight`**: More grouped agents
- **Higher `alignment_weight`**: More coordinated movement
- **Higher `separation_weight`**: Fewer collisions
- **Higher `foraging_weight`**: More focus on searching for food

## 📈 Calculated Metrics

The evaluation script calculates the same metrics as RL evaluations:

- **Efficiency**: Percentage of theoretical maximum consumption
- **Intake**: Total amount of resources consumed
- **Gini**: Gini coefficient (equity measure)
- **Polarization**: Velocity alignment (0-1)
- **Mean neighbor distance**: Average distance to neighbors
- **Separation violations**: Safe distance violations

## 🔍 Result Interpretation

### RL vs. Baseline Comparison

The `compare_rl_vs_baseline.py` script generates:

1. **Comparative table** with all metrics
2. **Difference analysis** between RL and Baseline
3. **Winner identification** for each metric
4. **Markdown and JSON report**

### Example Output

```
COMPARISON: RL vs. BASELINE BOIDS
================================================================================

EFFICIENCY:
  RL:       87.22% ± 5.34%
  Baseline: 45.30% ± 8.12%
  Difference: +41.92%
  Improvement: +92.5%
  Winner: RL

INTAKE:
  RL:       697.76
  Baseline: 362.40
  Difference: +335.36
  Winner: RL

FAIRNESS (Gini - lower is better):
  RL:       0.2460
  Baseline: 0.3520
  Difference: -0.1060
  Winner: RL
```

## 🎯 Use Cases

### 1. Validate that RL learns better than fixed rules

```bash
# Evaluate both systems
python train/eval_baseline_boids.py --config configs/env_easy_mode.yaml --output results/baseline.json
python train/eval_easy_mode.py --model models/ppo_easy_mode/best_model/best_model --output results/rl.json

# Compare
python train/compare_rl_vs_baseline.py --rl-results results/rl.json --baseline-results results/baseline.json
```

### 2. Analyze behavioral differences

Compare flocking metrics (polarization, mean distance) to understand how behaviors differ.

### 3. Baseline for new experiments

Use the baseline as a reference point when testing new configurations or algorithms.

## 🔧 Customization

### Modify Agent Behavior

Edit `env/boids_agent.py` to:
- Adjust steering rules
- Add new rules (e.g., avoid obstacles)
- Change force to action conversion

### Add New Metrics

Modify `train/eval_baseline_boids.py` to calculate additional metrics during evaluation.

## 📝 Technical Notes

- The Boids agent uses the same discrete actions as RL (0-4)
- Steering forces are converted to actions using heuristics
- The system is deterministic if the same seed is used
- Results are directly comparable with RL since they use the same environment

## 🐛 Troubleshooting

### Error: "ModuleNotFoundError: No module named 'env.boids_agent'"

Make sure to run with `PYTHONPATH=.`:
```bash
PYTHONPATH=. python train/eval_baseline_boids.py ...
```

### Very different results between runs

Verify that you're using the same seeds. The script uses `seed = ep * 42` by default.

### Baseline performs better than RL

This may indicate:
- The Boids agent is well calibrated for the environment
- RL needs more training
- RL rewards are not well balanced

---

**Reference**: Reynolds, C. W. (1987). Flocks, herds and schools: A distributed behavioral model. *ACM SIGGRAPH Computer Graphics*, 21(4), 25-34.

