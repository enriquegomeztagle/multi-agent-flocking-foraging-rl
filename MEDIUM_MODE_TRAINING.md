# Medium Mode Training - Gradual Difficulty Progress

## 🎯 Objective

Create an **incremental difficulty step** between easy mode (85% efficiency) and hard mode (5% efficiency) to enable better learning through curriculum progression.

## 📊 Difficulty Progression

### Easy → Medium → Hard Comparison

| Parameter | Easy Mode | **Medium Mode** | Hard Mode | Medium Change |
|-----------|-----------|-----------------|-----------|---------------|
| **Agents** | 5 | **10** | 15 | **+100%** |
| **Patches** | 20 | **12** | 8 | **-40%** |
| **World Size** | 20×20 (400) | **35×35 (1,225)** | 60×60 (3,600) | **+206%** |
| **Episode Length** | 2000 | **2500** | 3000 | **+25%** |
| **Feed Radius** | 4.0 | **3.0** | 2.5 | **-25%** |
| **Consumption** | 0.08 | **0.065** | 0.05 | **-19%** |
| **Regeneration** | 0.4 | **0.25** | 0.15 | **-38%** |
| **Patch Capacity** | 3.0 | **2.0** | 1.5 | **-33%** |

### Competition Analysis

**Agent-to-Patch Ratio**:
- Easy: 5/20 = **0.25** agents/patch (low competition)
- Medium: 10/12 = **0.83** agents/patch (moderate competition)
- Hard: 15/8 = **1.875** agents/patch (extreme competition)

**Patch Density**:
- Easy: 20/400 = **0.050** patches/unit²
- Medium: 12/1,225 = **0.010** patches/unit² (5x less dense)
- Hard: 8/3,600 = **0.002** patches/unit² (22x less dense)

**Key Insight**: Medium mode introduces moderate scarcity (0.83 agents/patch < 1) while hard mode creates overcrowding (1.875 agents/patch > 1).

## 🚀 Training Configuration

### Model Settings
- **Algorithm**: PPO (Proximal Policy Optimization)
- **Policy**: MlpPolicy (standard neural network, no LSTM)
- **Total Timesteps**: 2,000,000 (2x hard mode attempt)
- **Parallel Envs**: 4
- **Learning Rate**: 3e-4
- **Batch Size**: 256
- **Theoretical Max**: 1,625 intake (10 agents × 2500 steps × 0.065)

### Training Duration
- **Expected Time**: ~90 minutes
- **Checkpoints**: Every 100k steps
- **Evaluations**: Every 50k steps

## 🎯 Performance Targets

### Expected Efficiency Ranges
- **50-60%**: 812-975 intake → Baseline performance
- **60-70%**: 975-1,138 intake → Good coordination
- **70-80%**: 1,138-1,300 intake → Excellent performance
- **80%+**: 1,300+ intake → Outstanding (matching easy mode)

### Success Criteria
- **Minimum**: >50% efficiency (812+ intake) - Shows learning
- **Target**: 60-70% efficiency (975-1,138 intake) - Ready for hard mode
- **Stretch**: >70% efficiency (1,138+ intake) - Exceptional

## 📈 Training Progress

**Status**: 🔄 **IN PROGRESS** - Training started, awaiting first evaluation...

### Real-time Metrics
- **Current Step**: TBD
- **Mean Reward**: TBD
- **Best Evaluation**: TBD

## 💡 Rationale: Why Medium Mode?

### Problem with Direct Easy → Hard Jump
The original approach went from:
- **Easy**: 0.25 agents/patch → **85% efficiency** ✅
- **Hard**: 1.875 agents/patch → **5% efficiency** ⚠️

This is a **7.5x increase in competition** with only 1M timesteps training.

### Solution: Gradual Progression
Medium mode provides:
- **3.3x increase** in competition from easy (manageable)
- **Agent-to-patch ratio < 1** (no overcrowding)
- **Moderate scarcity** to learn resource competition
- **Foundation** for eventual hard mode success

## 🔬 Hypothesis

With proper training (2M timesteps), medium mode should:
1. **Achieve 60-70% efficiency** (vs 85% easy, 5% hard)
2. **Demonstrate learned coordination** for 10 agents
3. **Serve as checkpoint** for curriculum to hard mode
4. **Validate incremental approach** vs radical difficulty jumps

## 📁 Artifacts

- **Config**: [configs/env_medium_mode.yaml](configs/env_medium_mode.yaml)
- **Model Output**: [models/ppo_medium_mode/](models/ppo_medium_mode/)
- **Best Model**: TBD
- **Evaluation Results**: TBD

## 🔄 Next Steps

1. ✅ **Create Medium Config** - Incremental parameter changes
2. 🔄 **Train 2M Steps** - Currently running (~90 min)
3. ⏳ **Evaluate Performance** - After training completes
4. ⏳ **Compare to Easy/Hard** - Validate curriculum approach
5. ⏳ **Decide on Hard Mode** - If medium succeeds, retrain hard with 5M+ steps

---

**Started**: 2025-11-12
**Expected Completion**: ~90 minutes from start
**Status**: 🔄 **TRAINING IN PROGRESS**
