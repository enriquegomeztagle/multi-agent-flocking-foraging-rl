# Medium Mode - Scaled Coordination Challenge

## 📊 Environment Overview

**Difficulty Level:** ⭐⭐ Intermediate
**Purpose:** Test scalability with doubled agents and moderate resource competition
**Target Efficiency:** 60-70% (Expected: 70-75%)

---

## 🎯 Objectives

1. **Scale Testing**: Double the agents (5 → 10) while maintaining good performance
2. **Moderate Competition**: Introduce controlled resource scarcity (0.56 agents/patch)
3. **Coordination Challenge**: Require more sophisticated multi-agent coordination
4. **Curriculum Step**: Bridge between Easy and Hard difficulties

---

## ⚙️ Environment Configuration

### Core Parameters

```yaml
# Environment file: configs/env_medium.yaml
n_agents: 10             # Doubled from Easy
n_patches: 18            # Slightly reduced from Easy
width: 23.0              # Slightly larger world
height: 23.0
episode_len: 2500        # +25% more time

# Foraging parameters
feed_radius: 3.8         # Slightly smaller than Easy
c_max: 0.077             # Slightly reduced consumption
S_max: 2.8               # Reduced patch capacity
regen_r: 0.37            # Slower regeneration

# Flocking parameters
k_neighbors: 6           # More neighbors to track
v_max: 1.35
a_max: 0.18
turn_max: 0.28
d_safe: 0.8
```

### Difficulty Characteristics

| Metric | Value | vs Easy | Analysis |
|--------|-------|---------|----------|
| **Agent/Patch Ratio** | 0.56 (10/18) | +2.2x | Moderate competition emerging |
| **Patch Density** | 0.034 patches/unit² | -32% | Resources more spread out |
| **World Area** | 529 units² (23×23) | +32% | More exploration needed |
| **Feed Zone Area** | 45.4 units² | -10% | Slightly more precise positioning |
| **Resource Availability** | 50.4 total capacity | -16% | Less abundant resources |
| **Theoretical Max** | 1925 (10×2500×0.077) | +2.4x | Higher max but harder to achieve |

---

## 📈 Evaluation Results

**Model:** [models/ppo_medium/final_model.zip](models/ppo_medium/final_model.zip)
**Evaluation:** 100 episodes
**Date:** 2025-01-12

### Performance Summary

| Metric | Value | vs Target | vs Easy |
|--------|-------|-----------|---------|
| **Mean Efficiency** | **72.55%** | ✅ +2.55% above target | -14.67pp |
| **Median Efficiency** | **71.14%** | ✅ Within target range | -22.31pp |
| **Mean Intake** | 1396.58 ± 288.07 | 73% of theoretical max | +100% total |
| **Median Intake** | 1369.52 | 71% of theoretical max | +83% total |
| **Min Intake** | 770.00 | 40% (worst case) | +148% |
| **Max Intake** | 1924.99 | 100% (near perfect!) | +141% |
| **Mean Gini** | 0.274 | Good fairness | New metric |

### Performance Distribution

**Intake Statistics:**
- p25: 1177.98 (61.20%)
- p50: 1369.52 (71.14%)
- p75: 1609.86 (83.64%)
- p90: 1811.62 (94.11%)
- p95: 1878.61 (97.60%)
- p99: 1921.15 (99.80%)

**Performance Tiers:**
- 🏆 Excellent (≥70%): 52/100 episodes (52%)
- 🌟 Great (60-70%): 28/100 episodes (28%)
- ⭐ Good (50-60%): 11/100 episodes (11%)
- OK (40-50%): 9/100 episodes (9%)
- Below 40%: 0/100 episodes (0%)

### Top 5 Episodes

| Rank | Episode | Seed | Intake | Efficiency | Gini | Notes |
|------|---------|------|--------|------------|------|-------|
| 🥇 | 74 | 3066 | 1924.99 | 100.00% | 0.080 | Near perfect! |
| 🥈 | 100 | 4158 | 1921.67 | 99.83% | 0.060 | Excellent fairness |
| 🥉 | 68 | 2814 | 1900.14 | 98.71% | 0.105 | Great coordination |
| 4 | 76 | 3150 | 1896.62 | 98.52% | 0.119 | Great performance |
| 5 | 25 | 966 | 1889.60 | 98.16% | 0.098 | Consistent behavior |

---

## 🔬 Performance Analysis

### Comparison vs Easy Mode

| Aspect | Easy Mode | Medium Mode | Impact |
|--------|-----------|-------------|--------|
| **Mean Efficiency** | 87.22% | 72.55% | -14.67pp decline |
| **Consistency** | Very high | Good | More variance |
| **Perfect Episodes** | Multiple | 1 near-perfect | Harder to optimize |
| **Worst Case** | 39% | 40% | Similar floor |
| **Resource Fairness** | N/A | Gini 0.274 | Good distribution |

### Why 72% (vs 87% in Easy)?

The efficiency drop is due to:

1. **Agent Scaling** (5 → 10)
   - 2x more agents competing
   - More coordination complexity
   - Higher chance of conflicts

2. **Resource Reduction** (20 → 18 patches)
   - 10% fewer patches
   - Agent/patch ratio: 0.25 → 0.56 (+2.2x)
   - More competition per resource

3. **Larger World** (400 → 529 area)
   - +32% more space to explore
   - Longer travel distances
   - More time spent navigating

4. **Stricter Parameters**
   - Smaller feed radius (4.0 → 3.8)
   - Lower consumption (0.08 → 0.077)
   - Slower regeneration (0.4 → 0.37)

**Key Insight:** Despite doubling agents and reducing resources, the model maintains 72% efficiency - only a 15pp drop. This shows excellent scalability!

---

## 📊 Comparison: Easy → Medium Progression

### Configuration Changes

| Parameter | Easy | Medium | Change | Rationale |
|-----------|------|--------|--------|-----------|
| Agents | 5 | 10 | +100% | Test coordination at scale |
| Patches | 20 | 18 | -10% | Introduce scarcity |
| World | 20×20 | 23×23 | +32% area | Increase exploration challenge |
| Feed Radius | 4.0 | 3.8 | -5% | Require precision |
| Consumption | 0.08 | 0.077 | -4% | Slower resource acquisition |
| Regeneration | 0.4 | 0.37 | -8% | Less forgiving depletion |
| Episode Length | 2000 | 2500 | +25% | More time for coordination |

### Performance Progression

```
Easy Mode:    87.22% ━━━━━━━━━━━━━━━━━━━━━━ 5 agents, 20 patches
                ↓ (-14.67pp)
Medium Mode:  72.55% ━━━━━━━━━━━━━━━━━━━ 10 agents, 18 patches
```

**Efficiency Per Agent:**
- Easy: 87.22% / 5 agents = 17.44% per agent
- Medium: 72.55% / 10 agents = 7.26% per agent

The per-agent efficiency drops by 58%, which is expected with increased competition!

---

## 🎓 Key Learnings

### Successes ✅

1. **Scalability Validated**: Model successfully coordinates 10 agents (2x scaling)
2. **Target Exceeded**: 72.55% beats 60-70% target range
3. **Robust Performance**: 80% of episodes achieve ≥60% efficiency
4. **Fair Resource Distribution**: Gini 0.274 shows equitable foraging
5. **Near-Perfect Episode**: One episode achieved 99.9% efficiency

### Insights 💡

1. **Competition Effects**: Agent/patch ratio of 0.56 creates noticeable but manageable competition
2. **Coordination Emerges**: 10 agents learn to distribute across 18 patches effectively
3. **Variance Increases**: Std deviation grows with more agents (97.56 → 288.07)
4. **Curriculum Works**: Incremental difficulty enables better learning than radical jumps

### Challenges 🔧

1. **Efficiency Drop**: 15pp decline from Easy shows increased difficulty
2. **Consistency**: Fewer "perfect" episodes compared to Easy
3. **Fairness Tracking**: Gini coefficient reveals inequality exists but remains acceptable
4. **Coordination Overhead**: More agents = more potential for conflicts

---

## 🔧 Training Details

**Model:** PPO (Proximal Policy Optimization)
**Policy:** MlpPolicy
**Training Time:** ~90 minutes
**Total Steps:** 2,000,000
**Parallel Envs:** 4

**Hyperparameters:**
- Learning Rate: 3e-4
- Batch Size: 256
- Gamma: 0.99
- N-Steps: 2048
- Clip Range: 0.2
- Entropy Coef: 0.01

---

## 📁 Artifacts

- **Config:** [configs/env_medium.yaml](configs/env_medium.yaml)
- **Model:** [models/ppo_medium/final_model.zip](models/ppo_medium/final_model.zip)
- **Normalization:** [models/ppo_medium/vecnormalize.pkl](models/ppo_medium/vecnormalize.pkl)
- **Results:** [results/medium_evaluation.json](results/medium_evaluation.json)
- **Evaluation Script:** [train/eval_medium.py](train/eval_medium.py)

---

## 🚀 Usage

### Run Evaluation

```bash
python -m train.eval_medium --episodes 100
```

### Reproduce Best Episode

```python
from env.gym_wrapper import FlockForageGymWrapper
from env.flockforage_parallel import EnvConfig
import yaml

# Load config
with open('configs/env_medium.yaml', 'r') as f:
    config = yaml.safe_load(f)

env = FlockForageGymWrapper(EnvConfig(**config))

# Near-perfect episode seed
obs, _ = env.reset(seed=3066)  # Episode 74: 100% efficiency
```

---

## ✅ Conclusions

**Status:** ✅ **SUCCESS**

Medium Mode achieves **72.55% mean efficiency**, exceeding the 60-70% target. This demonstrates:

1. ✅ Model scales well from 5 to 10 agents
2. ✅ Moderate competition (0.56 agents/patch) is well-handled
3. ✅ Incremental difficulty approach is effective
4. ✅ Curriculum learning validated (Easy → Medium progression works)

**Comparison Summary:**
- Easy: 87% with 5 agents
- Medium: 72% with 10 agents
- Performance scales well despite 2x agents and fewer resources

**Next Step:** Progress to Hard Mode (10 agents, 15 patches, larger world) to test with increased scarcity.
