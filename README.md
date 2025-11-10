# Multi-Agent Flocking and Foraging with Reinforcement Learning

A multi-agent reinforcement learning environment where agents learn to flock (coordinate movement) while foraging for resources in a 2D world with **reflective boundaries** and regenerating patches.

## 🎯 Project Overview

This project models collective animal behavior (birds/fish) combining:
- **Flocking behaviors**: cohesion, alignment, and separation
- **Resource foraging**: patches with logistic regeneration
- **Reinforcement Learning**: RecurrentPPO with LSTM memory + Curriculum Learning

**Key Achievement:** Reached **37-50% efficiency** in resource collection through emergent coordinated behavior.

## ✨ Key Features

- **Reflective Boundaries**: Agents bounce off walls (as per project requirements)
- **RecurrentPPO with LSTM**: Agents remember past locations and decisions
- **Curriculum Learning**: Progressive training from 5 → 10 agents
- **Ultra-Simplified Rewards**: Food (30x) - Overcrowding (3x)
- **Comprehensive Metrics**: Fairness (Gini), flocking quality, sustainability
- **Baseline Comparison**: Classical Boids controller for comparative analysis
- **Visualization Dashboard**: Real-time metrics and agent behavior display
- **Demo Video Generation**: High-quality MP4 videos for presentations

## 📦 Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Quick Start

### Train Agent (RecurrentPPO + Curriculum Learning)

```bash
python -m train.run_advanced_training
```

This will:
- **Phase 1**: Train with 5 agents (easier coordination) for 5M steps
- **Phase 2**: Scale to 10 agents (transfer learning) for 5M steps
- **Total**: 10M timesteps (~20-30 minutes on modern CPU)
- Save model to `models/advanced_final/`
- Save results to `results/advanced_training_results.json`

### Evaluate Trained Agent

```bash
python -m train.evaluate
```

### Compare Baseline vs RL (Objectives O2 & O4)

Compare classical Boids rules with RL-trained agents:

```bash
# Evaluate baseline Boids controller
python -m train.baseline_boids

# Compare both approaches (requires trained model)
python -m train.compare_baseline_vs_rl
```

This generates a comparative analysis table showing:
- Cohesion and alignment metrics
- Gini coefficient (fairness)
- Resource sustainability
- Performance improvements

**Output:** `results/comparison_baseline_vs_rl.json`

### Visualization Dashboard

View real-time simulation with metrics:

```bash
# Baseline Boids controller
python -m visualize.dashboard --mode baseline --steps 500

# RL agent (requires trained model)
python -m visualize.dashboard --mode rl --steps 500

# Save as video instead of showing live
python -m visualize.dashboard --mode rl --steps 500 --save results/dashboard.mp4
```

Shows:
- Agent positions and velocities
- Resource patch levels
- Real-time polarization, Gini, and stock metrics

### Generate Demo Videos

Create presentation-quality videos:

```bash
# Generate both baseline and RL videos
python -m visualize.generate_video --mode both --steps 500

# Only baseline
python -m visualize.generate_video --mode baseline --steps 500

# Only RL (requires trained model)
python -m visualize.generate_video --mode rl --steps 500

# Custom output directory and quality
python -m visualize.generate_video --mode both --steps 800 --fps 30 --dpi 200 --output-dir my_videos
```

**Output:** `results/videos/baseline_boids_demo.mp4` and `results/videos/rl_recurrentppo_demo.mp4`

## 🏗️ Project Structure

```
multi-agent-flocking-foraging-rl/
├── README.md
├── requirements.txt
├── configs/
│   ├── env_curriculum_phase1.yaml     # Phase 1: 5 agents
│   └── env_curriculum_phase2.yaml     # Phase 2: 10 agents
├── env/
│   ├── flockforage_parallel.py        # PettingZoo ParallelEnv
│   ├── physics.py                     # Reflective boundaries
│   └── patches.py                     # Resource patches
├── metrics/
│   ├── fairness.py                    # Gini coefficient
│   ├── flocking.py                    # Polarization, cohesion
│   └── sustainability.py              # Resource metrics
├── train/
│   ├── run_advanced_training.py       # MAIN: RecurrentPPO + Curriculum
│   ├── evaluate.py                    # Evaluation utilities
│   ├── baseline_boids.py              # Classical Boids controller
│   └── compare_baseline_vs_rl.py      # Baseline vs RL comparison
├── visualize/
│   ├── dashboard.py                   # Real-time visualization
│   └── generate_video.py              # Demo video generator
├── docs/
│   └── PropuestaProyectoFinal-v2-UlisesBaez.typ
├── models/                            # Saved models (generated)
└── results/                           # Metrics & results (generated)
    ├── baseline_boids_metrics.json
    ├── comparison_baseline_vs_rl.json
    └── videos/                        # Demo videos
```

## 🔬 Environment Details

### Observation Space (13D)
Each agent observes:
- Own velocity (2D)
- Mean neighbor velocity (2D)
- Mean relative position to neighbors (2D)
- Mean distance to k-nearest neighbors (1D)
- Vector to nearest resource patch (2D)
- Nearest patch stock level (1D)
- Global mean patch stock (1D)
- EMA of own food intake (1D)
- Mean intake EMA of neighbors (1D) - for coordination

### Action Space (5 discrete actions)
- **0**: Turn left
- **1**: Turn right
- **2**: Accelerate
- **3**: Decelerate
- **4**: No-op

### Reward Structure (Ultra-Simplified)
```python
reward = food_intake * 30.0 - overcrowding_penalty * 3.0
```

**Design Philosophy:**
- Food is THE dominant objective (30x multiplier)
- Overcrowding penalty forces distribution across patches
- No complex flocking rewards (they conflict with foraging)

### World Physics
- **Boundaries**: **Reflective** (agents bounce off walls)
- **World Size**: 30×30 (dense environment, minimal travel time)
- **Patches**: 15 patches with logistic regeneration
- **Agents**: 10 agents with local perception

## 📊 Key Metrics

### Flocking Metrics
- **Polarization**: Alignment of agent velocities (0-1, higher is better)
- **k-NN Distance**: Average distance to nearest neighbors
- **Separation Violations**: Fraction of agents too close

### Fairness Metrics
- **Gini Coefficient**: Resource distribution inequality (0-1, lower is better)
- **Intake Statistics**: Mean and standard deviation

### Sustainability Metrics
- **Stock Score**: Mean normalized patch stock
- **Minimum Stock**: Lowest patch stock level
- **Below Threshold**: Fraction depleted patches

## 🎓 Advanced Features

### 1. RecurrentPPO with LSTM Memory
```python
RecurrentPPO(
    "MlpLstmPolicy",
    policy_kwargs=dict(
        lstm_hidden_size=64,
        enable_critic_lstm=True
    )
)
```
Agents **remember** where they've been and can learn temporal patterns.

### 2. Curriculum Learning
- **Phase 1** (5 agents): Easier coordination, higher patch/agent ratio
- **Phase 2** (10 agents): Transfer learning from Phase 1, full difficulty

### 3. Dense Environment
Small world (30×30) with high patch density minimizes travel time and maximizes feeding opportunities.

## 📈 Performance Results

| Metric | Value |
|--------|-------|
| **Best Efficiency** | 50% (episode peak) |
| **Mean Efficiency** | 37% (stable average) |
| **Gini Coefficient** | 0.62 (moderate fairness) |
| **Stock Remaining** | 99% (sustainable) |
| **Patches Used** | 7-9 / 15 (good distribution) |

**Note:** Reaching 70%+ would require:
- Explicit communication between agents
- Centralized coordination
- Further architectural changes (attention mechanisms, graph networks)

## 🔧 Configuration

### Curriculum Phase 1 (Easy)
```yaml
n_agents: 5
n_patches: 12
feed_radius: 3.0
c_max: 0.06
regen_r: 0.3
```

### Curriculum Phase 2 (Full)
```yaml
n_agents: 10
n_patches: 15
feed_radius: 3.0
c_max: 0.06
regen_r: 0.3
```

## 🎯 Project Requirements Met

✅ **All requirements from project proposal fulfilled:**

| Requirement | Status |
|-------------|--------|
| **O1:** Unified flocking + foraging environment | ✅ Implemented |
| **O2:** Measure metrics under classical rules | ✅ Baseline Boids |
| **O3:** RL rewards promoting cooperation | ✅ Food + overcrowding |
| **O4:** Evaluate Gini & stability (baseline vs RL) | ✅ Comparison script |
| PettingZoo ParallelEnv | ✅ Implemented |
| Reflective boundaries | ✅ Implemented |
| 10-14D observation space | ✅ 13D |
| 5 discrete actions | ✅ Complete |
| PPO training | ✅ RecurrentPPO (enhanced) |
| Gini coefficient metrics | ✅ Tracked |
| Sustainability metrics | ✅ Tracked |
| Curriculum learning | ✅ 2 phases |
| 1-2M timesteps | ✅ 10M (exceeded) |
| **Visualization dashboard** | ✅ Real-time metrics |
| **Demo videos** | ✅ Video generator |

## 📚 References

- Reynolds, C. (1987). Flocks, herds, and schools: A distributed behavioral model
- Schulman, J. et al. (2017). Proximal Policy Optimization Algorithms
- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [SB3-Contrib RecurrentPPO](https://sb3-contrib.readthedocs.io/)

## 📄 License

MIT

---

## 🚧 Future Work

- **Explicit Communication**: Message passing between agents
- **Graph Neural Networks**: For better coordination
- **Attention Mechanisms**: Dynamic neighbor weighting
- **3D Environment**: Extend to three-dimensional space
- **Heterogeneous Agents**: Different agent types with specialized roles
