# Hard Mode - High Competition Challenge

## 📊 Environment Overview

**Difficulty Level:** ⭐⭐⭐ Advanced
**Purpose:** Test performance under significant resource scarcity and competition
**Target Efficiency:** 40-50% (Expected: 45-55%)

---

## 🎯 Objectives

1. **Scarcity Challenge**: Reduce patches significantly (18 → 15) while maintaining 10 agents
2. **Spatial Complexity**: Increase world size to test exploration
3. **Competition Stress Test**: Push agent/patch ratio closer to 1:1
4. **Coordination Under Pressure**: Require sophisticated resource sharing

---

## ⚙️ Environment Configuration

### Core Parameters

```yaml
# Environment file: configs/env_hard.yaml
n_agents: 10             # Same as Medium
n_patches: 15            # Reduced from Medium (-17%)
width: 28.0              # Larger world (+22%)
height: 28.0
episode_len: 2500        # Same as Medium

# Foraging parameters
feed_radius: 3.5         # Smaller than Medium
c_max: 0.070             # Further reduced consumption
S_max: 2.5               # Lower patch capacity
regen_r: 0.32            # Slower regeneration

# Flocking parameters
k_neighbors: 6
v_max: 1.35
a_max: 0.18
turn_max: 0.28
d_safe: 0.8
```

### Difficulty Characteristics

| Metric | Value | vs Medium | Analysis |
|--------|-------|-----------|----------|
| **Agent/Patch Ratio** | 0.67 (10/15) | +20% | Approaching 1:1 competition |
| **Patch Density** | 0.019 patches/unit² | -44% | Much more spread out |
| **World Area** | 784 units² (28×28) | +48% | Significantly more exploration |
| **Feed Zone Area** | 38.5 units² | -15% | Tighter positioning required |
| **Resource Availability** | 37.5 total capacity | -26% | Notable scarcity |
| **Theoretical Max** | 1750 (10×2500×0.070) | -9% | Lower ceiling |

---

## 📈 Evaluation Results

**Model:** [models/ppo_hard/final_model.zip](models/ppo_hard/final_model.zip)
**Evaluation:** 100 episodes
**Date:** 2025-01-12

### Performance Summary

| Metric | Value | vs Target | vs Medium |
|--------|-------|-----------|-----------|
| **Mean Efficiency** | **45.90%** | ✅ Within target | -26.65pp |
| **Median Efficiency** | **40.27%** | ✅ At target floor | -30.87pp |
| **Mean Intake** | 803.28 ± 263.55 | 46% of theoretical max | -42% total |
| **Median Intake** | 704.66 | 40% of theoretical max | -49% total |
| **Min Intake** | 308.98 | 18% (worst case) | -60% |
| **Max Intake** | 1400.04 | 80% (best case) | -27% |
| **Mean Gini** | 0.539 | Moderate inequality | +97% worse |

### Performance Distribution

**Intake Statistics:**
- p25: 611.85 (34.96%)
- p50: 704.66 (40.27%)
- p75: 942.68 (53.87%)
- p90: 1180.83 (67.48%)
- p95: 1267.00 (72.40%)
- p99: 1379.52 (78.83%)

**Performance Tiers:**
- 🏆 Excellent (≥70%): 6/100 episodes (6%)
- 🌟 Great (60-70%): 8/100 episodes (8%)
- ⭐ Good (50-60%): 17/100 episodes (17%)
- OK (40-50%): 25/100 episodes (25%)
- Below 40%: 44/100 episodes (44%)

### Top 5 Episodes

| Rank | Episode | Seed | Intake | Efficiency | Gini | Notes |
|------|---------|------|--------|------------|------|-------|
| 🥇 | 92 | 3822 | 1400.04 | 80.00% | 0.437 | Best performance |
| 🥈 | 46 | 1890 | 1398.64 | 79.92% | 0.403 | Excellent coordination |
| 🥉 | 49 | 2016 | 1396.49 | 79.80% | 0.482 | Strong performance |
| 4 | 12 | 462 | 1389.81 | 79.42% | 0.426 | Good distribution |
| 5 | 43 | 1764 | 1322.71 | 75.58% | 0.489 | Above 75% |

---

## 🔬 Performance Analysis

### Comparison vs Medium Mode

| Aspect | Medium Mode | Hard Mode | Impact |
|--------|-------------|-----------|--------|
| **Mean Efficiency** | 72.55% | 45.90% | -26.65pp significant drop |
| **Median Efficiency** | 71.14% | 40.27% | -30.87pp even larger drop |
| **Consistency** | Good | Moderate | More variance |
| **Worst Case** | 40% | 18% | Worse floor |
| **Best Case** | 100% | 80% | Lower ceiling |
| **Fairness (Gini)** | 0.274 | 0.539 | Nearly 2x worse inequality |

### Why 46% (vs 72% in Medium)?

The significant efficiency drop is due to:

1. **Resource Scarcity** (18 → 15 patches)
   - 17% fewer patches
   - Agent/patch ratio: 0.56 → 0.67 (+20%)
   - Much higher competition

2. **Larger Exploration Space** (529 → 784 area)
   - +48% more area to cover
   - Longer travel between patches
   - Harder to locate resources

3. **Stricter Resource Parameters**
   - Smaller feed radius (3.8 → 3.5, -8%)
   - Lower consumption (0.077 → 0.070, -9%)
   - Slower regeneration (0.37 → 0.32, -14%)
   - Lower patch capacity (2.8 → 2.5, -11%)

4. **Coordination Breakdown**
   - Gini 0.539 shows high inequality
   - Some agents get significantly less food
   - Clustering and conflicts increase

5. **Lower Theoretical Max**
   - Despite same agents, lower c_max reduces ceiling
   - 1750 vs 1925 (-9%) theoretical maximum

**Key Insight:** The model struggles with the scarcity-competition-exploration triple challenge, but still maintains 46% efficiency - within the 40-50% target range!

---

## 📊 Progression: Medium → Hard

### Configuration Changes

| Parameter | Medium | Hard | Change | Impact |
|-----------|--------|------|--------|--------|
| Agents | 10 | 10 | Same | No scaling change |
| Patches | 18 | 15 | -17% | Higher competition |
| World | 23×23 | 28×28 | +48% area | More exploration |
| Feed Radius | 3.8 | 3.5 | -8% | Tighter positioning |
| Consumption | 0.077 | 0.070 | -9% | Slower intake |
| Regeneration | 0.37 | 0.32 | -14% | Harder to sustain |
| Patch Capacity | 2.8 | 2.5 | -11% | Quicker depletion |

### Performance Progression

```
Easy Mode:    87.22% ━━━━━━━━━━━━━━━━━━━━━━ 5 agents, 20 patches, 20×20
                ↓ (-14.67pp)
Medium Mode:  72.55% ━━━━━━━━━━━━━━━━━ 10 agents, 18 patches, 23×23
                ↓ (-26.65pp)
Hard Mode:    45.90% ━━━━━━━━━━ 10 agents, 15 patches, 28×28
```

**Efficiency Trend:**
- Easy → Medium: -14.67pp (manageable)
- Medium → Hard: -26.65pp (significant challenge)

The larger drop from Medium→Hard shows the compounding effects of scarcity, space, and reduced parameters.

---

## 🎓 Key Learnings

### Successes ✅

1. **Target Achieved**: 45.90% meets 40-50% target range
2. **Model Adapts**: Successfully handles increased competition
3. **Some Excellence**: 6% of episodes achieve ≥70% efficiency
4. **Scalability**: Model trained for same agents handles harder conditions

### Challenges 🔧

1. **Efficiency Drop**: 27pp decline from Medium shows difficulty jump
2. **High Inequality**: Gini 0.539 indicates some agents struggle to find food
3. **Variance**: Large std deviation (263.55) shows inconsistent performance
4. **Floor Lowered**: Worst case drops to 18% vs 40% in Medium

### Insights 💡

1. **Scarcity Threshold**: Agent/patch ratio of 0.67 approaches tipping point
2. **Exploration Costs**: 48% larger world significantly impacts efficiency
3. **Resource Competition**: With fewer patches, conflicts increase dramatically
4. **Fairness Trade-off**: Higher competition leads to unequal distribution

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

---

## 📁 Artifacts

- **Config:** [configs/env_hard.yaml](configs/env_hard.yaml)
- **Model:** [models/ppo_hard/final_model.zip](models/ppo_hard/final_model.zip)
- **Normalization:** [models/ppo_hard/vecnormalize.pkl](models/ppo_hard/vecnormalize.pkl)
- **Results:** [results/hard_evaluation.json](results/hard_evaluation.json)
- **Evaluation Script:** [train/eval_hard.py](train/eval_hard.py)

---

## 🚀 Usage

### Run Evaluation

```bash
python -m train.eval_hard --episodes 100
```

### Reproduce Best Episode

```python
from env.gym_wrapper import FlockForageGymWrapper
from env.flockforage_parallel import EnvConfig
import yaml

# Load config
with open('configs/env_hard.yaml', 'r') as f:
    config = yaml.safe_load(f)

env = FlockForageGymWrapper(EnvConfig(**config))

# Best episode seed
obs, _ = env.reset(seed=3822)  # Episode 92: 80% efficiency
```

---

## ✅ Conclusions

**Status:** ✅ **SUCCESS** (Within Target Range)

Hard Mode achieves **45.90% mean efficiency**, successfully meeting the 40-50% target. This demonstrates:

1. ✅ Model handles significant resource scarcity (0.67 agents/patch)
2. ✅ Adapts to 48% larger exploration space
3. ✅ Maintains performance despite stricter parameters
4. ⚠️ Shows signs of stress (high Gini, larger variance)

**Performance Summary:**
- Easy: 87% (abundant resources)
- Medium: 72% (moderate competition)
- Hard: 46% (high scarcity) ← Current level

**Key Challenges in Hard Mode:**
- Resource competition approaching 1:1 ratio
- Large world requiring extensive exploration
- Lower consumption and regeneration rates
- Inequality in resource distribution (Gini 0.539)

**Next Step:** Progress to Expert Mode (12 agents, 10 patches, 35×35) to test extreme scarcity where agents significantly outnumber patches.
