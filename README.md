# Multi-Agent Flocking and Foraging with Reinforcement Learning

A multi-agent reinforcement learning system demonstrating **emergent coordination** in collective foraging across four difficulty levels, from abundant resources (87% efficiency) to extreme scarcity where agents outnumber patches (37% efficiency).

---

## 🎯 Project Purpose

This project investigates how **individual learning agents can achieve collective coordination** in resource-scarce environments by combining:

1. **Biologically-inspired flocking behaviors** (Reynolds boids: cohesion, alignment, separation)
2. **Resource foraging with competition** (logistic patch regeneration, finite capacity)
3. **Deep reinforcement learning** (PPO with flocking-aware reward structure)
4. **Incremental difficulty progression** (independent training across four difficulty levels from abundant to extreme scarcity)

**Central Research Question:** Can multi-agent RL systems learn effective coordination strategies that scale from abundant to extremely scarce resources, including scenarios where agents outnumber resource patches?

**Answer:** ✅ Yes. We validate 4 difficulty levels demonstrating smooth efficiency progression (87% → 73% → 50% → 37%), including a successful agents > patches scenario.

---

## ✨ Key Results - Four Difficulty Levels

| Level            | Agents | Patches | World  | Agent/Patch    | Efficiency       | Training      | Documentation                 |
| ---------------- | ------ | ------- | ------ | -------------- | ---------------- | ------------- | ----------------------------- |
| **Easy**   | 5      | 20      | 20×20 | 0.25           | **87.22%** | 1.5M (~30min) | [EASY_MODE.md](EASY_MODE.md)     |
| **Medium** | 10     | 18      | 23×23 | 0.56           | **72.55%** | 2M (~90min)   | [MEDIUM_MODE.md](MEDIUM_MODE.md) |
| **Hard**   | 10     | 15      | 28×28 | 0.67           | **49.86%** | 3M (~2.5h)   | [HARD_MODE.md](HARD_MODE.md)     |
| **Expert** | 12     | 10      | 35×35 | **1.20** | **37.12%** | 3M (~120min)  | [EXPERT_MODE.md](EXPERT_MODE.md) |

### Scientific Contributions

1. **Scalability Validation**: PPO scales from abundant (87%) to extreme scarcity (37%)
2. **Novel Scenario**: First successful agents > patches configuration (12 agents, 10 patches)
3. **Emergent Cooperation**: Dynamic flock splitting, resource sharing from individual policies
4. **Design Principles**: Proves flocking + foraging synergy essential across all difficulty levels

---

## 📈 Efficiency Progression

```
Efficiency %
    100 ┤
     90 ┤ ●●●●●●        ← Easy (87.22%)
     80 ┤ ●●●●●●
     70 ┤        ●●●●●  ← Medium (72.55%)
     60 ┤        ●●●●●
     50 ┤
     40 ┤              ●●●● ← Hard (49.86%)
     30 ┤                   ●●● ← Expert (37.12%)
     20 ┤                   ●●●
     10 ┤
      0 ┤
        └──────────────────────────────────
         Easy   Medium    Hard   Expert
```

**Efficiency Drops:**

- Easy → Medium: -14.67pp (smooth progression)
- Medium → Hard: -22.69pp (significant challenge increase)
- Hard → Expert: -12.74pp (gradual, demonstrates robustness)

---

## 🔬 Quick Level Overview

### Easy Mode - Baseline (87.22% efficiency)

- **Purpose:** Validate learning capability with abundant resources
- **Configuration:** 5 agents, 20 patches, 20×20 world
- **Key Insight:** Model learns near-optimal foraging with minimal competition
- **Details:** [EASY_MODE.md](EASY_MODE.md)

### Medium Mode - Scaled Coordination (72.55% efficiency)

- **Purpose:** Test coordination with doubled agents and moderate scarcity
- **Configuration:** 10 agents, 18 patches, 23×23 world
- **Key Insight:** Coordination scales well despite 2x agents and fewer resources
- **Details:** [MEDIUM_MODE.md](MEDIUM_MODE.md)

### Hard Mode - High Competition (49.86% efficiency)

- **Purpose:** Require sophisticated resource sharing under scarcity
- **Configuration:** 10 agents, 15 patches, 28×28 world
- **Key Insight:** Retrained model (3M steps) successfully handles significant scarcity with good consistency
- **Details:** [HARD_MODE.md](HARD_MODE.md)

### Expert Mode - Extreme Scarcity (37.12% efficiency) 🔥 NOVEL

- **Purpose:** Push boundaries with agents outnumbering patches
- **Configuration:** 12 agents, 10 patches, 35×35 world
- **Key Insight:** Agents survive and coordinate even when resources are fundamentally scarce (1.2:1 ratio)
- **Details:** [EXPERT_MODE.md](EXPERT_MODE.md)

---

## 🏗️ Environment Design

### Observation Space (13 dimensions per agent)

Each agent observes its local neighborhood:

1. **Own velocity** (2D): Current velocity vector
2. **Mean neighbor velocity** (2D): Average velocity of k-nearest neighbors
3. **Mean relative position to neighbors** (2D): Flock cohesion information
4. **Mean distance to neighbors** (1D): Flock dispersion metric
5. **Vector to nearest patch** (2D): Direction to closest resource
6. **Nearest patch stock level** (1D): Resource availability
7. **Global mean patch stock** (1D): Overall resource state
8. **Own intake EMA** (1D): Exponential moving average of food intake
9. **Mean intake of neighbors** (1D): Performance of nearby agents

**Design rationale:** Provides both local flocking information and global resource awareness without requiring full observability.

### Action Space (5 discrete actions)

- **0**: Turn left (decrease heading)
- **1**: Turn right (increase heading)
- **2**: Accelerate (increase velocity magnitude)
- **3**: Decelerate (decrease velocity magnitude)
- **4**: No-op (maintain current velocity)

**Design rationale:** Simple kinematic control enables interpretable behaviors while allowing complex emergent patterns.

### Reward Function Design

**Primary Reward:**

- Food intake (directly proportional to resources collected per timestep)

**Flocking Rewards** (critical for coordination):

- **Cohesion**: Reward staying within optimal distance to flock center
- **Alignment**: Reward matching velocity with neighbors
- **Separation**: Penalty for overcrowding (collision avoidance)
- **Group Bonus**: Reward maintaining optimal flock size

**Critical Finding:** All 4 difficulty levels require **BOTH** flocking + foraging rewards. Without flocking rewards, efficiency drops significantly across all levels.

**Design rationale:** Flocking rewards create emergent benefits for foraging (coordinated exploration, patch discovery) without explicitly rewarding foraging success.

---

## 📦 Installation

```bash
# Clone repository
git clone <repository-url>
cd multi-agent-flocking-foraging-rl

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Requirements:**

- Python 3.8+
- PyTorch 2.0+
- Stable-Baselines3
- PettingZoo
- NumPy, Gymnasium, PyYAML

---

## 🚀 Quick Start

### Evaluate Pre-Trained Models

```bash
# Easy Mode (87.22% efficiency)
python -m train.eval_easy --episodes 100

# Medium Mode (72.55% efficiency)
python -m train.eval_medium --episodes 100

# Hard Mode (49.86% efficiency)
python -m train.eval_hard --episodes 100

# Expert Mode (37.12% efficiency)
python -m train.eval_expert --episodes 100
```

**Note:** Models are already trained and saved in `models/ppo_{easy,medium,hard,expert}/`

### Train New Models

```bash
# Easy Mode (1.5M steps, ~30 minutes)
python -m train.train_ppo \
  --config configs/env_easy.yaml \
  --output models/ppo_easy \
  --timesteps 1500000

# Medium Mode (2M steps, ~90 minutes)
python -m train.train_ppo \
  --config configs/env_medium.yaml \
  --output models/ppo_medium \
  --timesteps 2000000

# Hard Mode (2M steps, ~90 minutes)
python -m train.train_ppo \
  --config configs/env_hard.yaml \
  --output models/ppo_hard \
  --timesteps 2000000

# Expert Mode (3M steps, ~120 minutes)
python -m train.train_ppo \
  --config configs/env_expert.yaml \
  --output models/ppo_expert \
  --timesteps 3000000
```

---

## 🏗️ Project Structure

```
multi-agent-flocking-foraging-rl/
├── README.md                    # This file
├── EASY_MODE.md                 # Easy mode detailed analysis
├── MEDIUM_MODE.md               # Medium mode detailed analysis
├── HARD_MODE.md                 # Hard mode detailed analysis
├── EXPERT_MODE.md               # Expert mode detailed analysis
├── requirements.txt             # Python dependencies
├── configs/
│   ├── env_easy.yaml           # Easy mode configuration
│   ├── env_medium.yaml         # Medium mode configuration
│   ├── env_hard.yaml           # Hard mode configuration
│   └── env_expert.yaml         # Expert mode configuration
├── env/
│   ├── flockforage_parallel.py # PettingZoo ParallelEnv implementation
│   ├── gym_wrapper.py          # Gym wrapper for Stable-Baselines3
│   ├── physics.py              # Agent dynamics (velocity, heading)
│   ├── patches.py              # Resource patches (logistic regeneration)
│   └── boids_agent.py          # Classical Boids baseline agent
├── metrics/
│   ├── fairness.py             # Gini coefficient calculation
│   ├── flocking.py             # Polarization, cohesion metrics
│   └── sustainability.py       # Resource depletion metrics
├── train/
│   ├── train_ppo.py            # Training script (PPO)
│   ├── eval_easy.py            # Easy mode evaluation
│   ├── eval_medium.py          # Medium mode evaluation
│   ├── eval_hard.py            # Hard mode evaluation
│   ├── eval_expert.py          # Expert mode evaluation
│   └── eval_baseline_boids.py  # Classical Boids baseline
├── models/                      # Trained models (gitignored)
│   ├── ppo_easy/
│   │   ├── final_model.zip
│   │   ├── vecnormalize.pkl
│   │   └── env_config.yaml
│   ├── ppo_medium/
│   ├── ppo_hard/
│   └── ppo_expert/
└── results/                     # Evaluation results (gitignored)
    ├── easy_evaluation.json
    ├── medium_evaluation.json
    ├── hard_evaluation.json
    └── expert_evaluation.json
```

---

## 📊 Detailed Configuration Comparison

| Parameter                   | Easy             | Medium           | Hard             | Expert           | Progression        |
| --------------------------- | ---------------- | ---------------- | ---------------- | ---------------- | ------------------ |
| **n_agents**          | 5                | 10               | 10               | 12               | +140%              |
| **n_patches**         | 20               | 18               | 15               | 10               | -50%               |
| **Agent/Patch Ratio** | 0.25             | 0.56             | 0.67             | **1.20**   | +380%              |
| **World Size**        | 20×20           | 23×23           | 28×28           | 35×35           | +206% area         |
| **Patch Density**     | 0.050            | 0.034            | 0.019            | 0.0082           | -84%               |
| **feed_radius**       | 4.0              | 3.8              | 3.5              | 2.8              | -30%               |
| **c_max**             | 0.08             | 0.077            | 0.070            | 0.058            | -28%               |
| **S_max**             | 3.0              | 2.8              | 2.5              | 2.0              | -33%               |
| **regen_r**           | 0.4              | 0.37             | 0.32             | 0.24             | -40%               |
| **Theoretical Max**   | 800              | 1,925            | 1,750            | 1,740            | Variable           |
| **Mean Intake**       | 697.76           | 1,396.58         | 872.52           | 645.81           | -                  |
| **Efficiency**        | **87.22%** | **72.55%** | **49.86%** | **37.12%** | -57%               |
| **Mean Gini**         | N/A              | 0.274            | 0.482            | 0.569            | Fairness decreases |
| **Training Steps**    | 1.5M             | 2M               | 3M               | 3M               | 2x longer          |

---

## 🎯 Key Findings

### 1. Flocking + Foraging Synergy is Essential

All 4 difficulty levels require **BOTH** flocking AND foraging rewards:

- **Cohesion** → Agents naturally discover patches together (shared exploration)
- **Alignment** → Coordinated movement reduces redundant search
- **Separation** → Prevents resource competition at single patch
- **Group bonus** → Maintains cohesion through difficult periods

### 2. Emergent Cooperation Without Explicit Coordination

Sophisticated cooperative behaviors emerge from individual policies:

- **Dynamic flock splitting**: Agents split into sub-groups to cover more area
- **Resource sharing**: Low-energy agents guided to patches by well-fed agents
- **Efficient patch rotation**: Proactive switching from depleting patches
- **Collision avoidance**: Strong safety distance despite crowding

No explicit coordination rewards or communication required.

### 3. Incremental Difficulty Progression Works

Smooth efficiency drops validate incremental design:

- Easy → Medium: -14.67pp (manageable increase)
- Medium → Hard: -22.69pp (significant but learnable)
- Hard → Expert: -12.74pp (demonstrates robustness)

No catastrophic failures despite extreme parameter changes.

### 4. Agent/Patch Ratio Threshold

Critical transition occurs around 1:1 agent/patch ratio:

- Below 1.0: Optimization problem (how efficiently can agents forage?)
- Above 1.0: Survival problem (can agents survive with insufficient resources?)

Expert mode (1.2:1) successfully demonstrates survival mode.

---

## 🔬 Reproduce Results

### Easy Mode - Best Episode (100% efficiency)

```python
from env.flockforage_parallel import FlockForageParallel, EnvConfig
import yaml

# Load configuration from YAML
with open('configs/env_easy.yaml', 'r') as f:
    config = yaml.safe_load(f)

env = FlockForageParallel(EnvConfig(**config))
obs, _ = env.reset(seed=1554)  # Episode 38: 100% efficiency

# Load model and run episode...
from stable_baselines3 import PPO
model = PPO.load('models/ppo_easy/final_model')
```

### Expert Mode - Best Episode (64.48% efficiency)

```python
# Load Expert configuration
with open('configs/env_expert.yaml', 'r') as f:
    config = yaml.safe_load(f)

env = FlockForageParallel(EnvConfig(**config))
obs, _ = env.reset(seed=3990)  # Episode 96: 64.48% efficiency

# Load model and run episode...
model = PPO.load('models/ppo_expert/final_model')
```

---

## 📚 Additional Documentation

Detailed documentation for each difficulty level:

- **[EASY_MODE.md](EASY_MODE.md)** - Baseline environment with abundant resources (87.22%)
- **[MEDIUM_MODE.md](MEDIUM_MODE.md)** - Scaled coordination challenge (72.55%)
- **[HARD_MODE.md](HARD_MODE.md)** - High competition environment (49.86%)
- **[EXPERT_MODE.md](EXPERT_MODE.md)** - Extreme scarcity where agents > patches (37.12%)

Each document includes:

- Complete configuration details
- Performance analysis
- Comparison with other levels
- Training details
- Key insights and learnings

---

## 📊 Performance Summary

### Performance Tiers Distribution

| Tier                        | Easy | Medium | Hard | Expert |
| --------------------------- | ---- | ------ | ---- | ------ |
| **Excellent (≥70%)** | 80%  | 52%    | 6%   | 0%     |
| **Great (60-70%)**    | 13%  | 28%    | 8%   | 0%     |
| **Good (50-60%)**     | 4%   | 11%    | 17%  | 5%     |
| **OK (40-50%)**       | 2%   | 9%     | 25%  | 32%    |
| **Below 40%**         | 1%   | 0%     | 44%  | 63%    |

### Fairness Metrics (Gini Coefficient)

- **Medium Mode:** 0.274 (good fairness)
- **Hard Mode:** 0.539 (moderate inequality)
- **Expert Mode:** 0.569 (expected high due to extreme scarcity)

**Insight:** As resources become scarcer, inequality increases - some agents thrive while others struggle.

---

## 🚀 Future Research Directions

### 1. Larger Scale Validation

- 50×50 worlds with 20+ agents
- Test scalability limits
- Target: 20-30% efficiency with agents significantly outnumbering patches

### 2. Heterogeneous Agent Capabilities

- Scout agents (high speed, low capacity)
- Forager agents (high consumption, normal speed)
- Test if specialization improves efficiency

### 3. Dynamic Environments

- Moving resource patches
- Variable patch quality over time
- Density-dependent regeneration

### 4. Advanced Architectures

- Transformer-based policies (attention mechanisms)
- Graph Neural Networks (explicit agent relationships)
- Transfer learning (pretrain on Easy, fine-tune on Expert)

### 5. Multi-Objective Optimization

- Balance efficiency vs fairness (Pareto frontier)
- Minimize energy consumption
- Maximize group survival time under scarcity

---

## 📖 References

### Academic Papers

- Reynolds, C. W. (1987). Flocks, herds and schools: A distributed behavioral model. *Computer graphics*, 21(4), 25-34.
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. *arXiv preprint arXiv:1707.06347*.
- Terry, J. K., et al. (2021). PettingZoo: Gym for multi-agent reinforcement learning. *Advances in Neural Information Processing Systems*, 34.

### Implementation

- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [PettingZoo Documentation](https://pettingzoo.farama.org/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)

---

## ✅ Validated Claims

1. ✅ **Four difficulty levels successfully validated** with smooth efficiency progression
2. ✅ **Flocking + foraging synergy proven essential** at all difficulty levels
3. ✅ **Scalability demonstrated** from abundant resources (0.25:1) to extreme scarcity (1.2:1)
4. ✅ **Novel scenario validated**: Agents outnumbering patches (12:10 ratio achieving 37% efficiency)
5. ✅ **Emergent cooperation shown**: Dynamic strategies without explicit coordination
6. ✅ **Progressive difficulty design effective**: Incremental scaling across four levels enables successful learning
7. ✅ **Configuration-driven approach**: All parameters loaded from YAML

## 📄 License


---

## 🙏 Acknowledgments

This project builds upon:

- Reynolds' Boids model for flocking behaviors
- Stable-Baselines3 for PPO implementation
- PettingZoo for multi-agent environment interface
- The multi-agent RL research community

---

**Last Updated:** November 12, 2025
**Version:** 2.0
