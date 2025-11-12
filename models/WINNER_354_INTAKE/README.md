# WINNING MODEL - 354.7 Mean Intake
## Wide MLP (512 Hidden Units) - PRODUCTION READY

**Performance:** 354.7 mean intake (best in class)
**Architecture:** Wide MLP with 512 hidden units
**Training:** 4M steps, 19 minutes
**Status:** PRODUCTION READY ✅

---

## Quick Start

### Load the Model

```python
import torch
from networks.wide_mlp import WideMLP

# Initialize architecture
model = WideMLP(
    obs_dim=13,
    graph_feature_dim=16,
    hidden_dim=512,
    n_actions=5,
    dropout=0.1
)

# Load trained weights
model.load_state_dict(torch.load("models/WINNER_354_INTAKE/model.zip"))
model.eval()
```

---

## Files in This Directory

### Model Files
- **model.zip** (1.6M) - Trained model weights
- **vecnorm.pkl** (4.1K) - Observation normalization statistics

### Results Files
- **ppo_final_results.json** - Training summary and evaluation metrics
- **extended_evaluation.json** - 100-episode extended evaluation results

### Documentation
- **README.md** (this file) - Quick reference
- **../../WINNING_MODEL_354_INTAKE.md** - Complete documentation

---

## Performance Summary

| Metric | Value |
|--------|-------|
| Mean intake | 354.7 |
| Median intake | 316.5 |
| Max intake | 625.2 |
| Std | 121.7 |
| Mean Gini | 0.589 |
| Training time | 19 min |
| Total steps | 4M |

### Distribution (100 episodes)
- **Excellent (20%+):** 15 episodes
- **Great (15-20%):** 31 episodes
- **Good (10-15%):** 32 episodes
- **OK (5-10%):** 18 episodes
- **Poor (<5%):** 4 episodes

**High performers (>10%):** 78/100 episodes

---

## Architecture Details

### Network Structure
```
Input (29D) → 512 hidden units → Residual blocks → Output
```

**Layers:**
1. Input: 29 → 512 (LayerNorm + ReLU + Dropout 0.1)
2. Residual block 1: 512 → 512 (LayerNorm + ReLU + Dropout)
3. Residual block 2: 512 → 512 (LayerNorm + ReLU + Dropout)
4. Actor head: 512 → 256 → 5 actions
5. Critic head: 512 → 256 → 1 value

**Total parameters:** 808,942

### Input Features (29D)
- **Observations (13D):** position, velocity, internal state, patch distances, patch resources
- **Graph features (16D):** structural (4D), spatial (6D), competition (6D)

---

## Why This Model Won

### 1. Superior Performance
- **354.7 mean intake** vs 270 baseline (+31%)
- **625.2 peak** episode (seed 1302)
- **Consistent:** 78% episodes >10% efficiency

### 2. Optimal Architecture
- **Width (512) > Depth:** Captures complex multi-agent interactions
- **Residual connections:** Stable 4-layer training
- **Graph features:** Spatial awareness without expensive GNN

### 3. Efficient Training
- **19 minutes** for 4M steps
- **808k parameters:** Sweet spot for CPU training
- **Curriculum learning:** 5→10 agents, 12→15 patches

---

## Reproducibility

### Best Episode (625.2 intake)
```python
env = ForagingEnvironment(
    n_agents=10,
    n_patches=15,
    width=30.0,
    height=30.0,
    episode_len=1500,
    seed=1302  # Best seed
)
```

### Training Config
```python
PPO_CONFIG = {
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "learning_rate": 3e-4,
    "batch_size": 256,
    "n_epochs": 10,
}
```

---

## Comparison with Other Models

| Model | Mean Intake | Parameters | Time |
|-------|-------------|------------|------|
| **This (Wide MLP)** | **354.7** | **808k** | **19m** |
| GNN Hybrid | 168.5 | 1.5M+ | 62m |
| Baseline MLP | 270.0 | 207k | 80m |
| Random | 30-50 | 0 | - |

**Winner by:** Performance, efficiency, and training speed

---

## Usage Examples

### Inference
```python
# Single agent decision
obs = get_observation(agent_id)  # [13]
graph_feats = compute_graph_features(agents)  # [16]
x = torch.cat([obs, graph_feats])

with torch.no_grad():
    action_logits, value = model(x)
    action = torch.argmax(action_logits)
```

### Evaluation
```python
from eval_and_save_trajectories import evaluate_model

results = evaluate_model(
    model_path="models/WINNER_354_INTAKE/model.zip",
    n_episodes=100,
    save_trajectories=True
)
```

---

## Model History

1. **Architecture search** → Wide MLP (512h) selected
2. **Phase 1 training** (0-2M steps) → 5 agents, 12 patches
3. **Phase 2 training** (2M-4M steps) → 10 agents, 15 patches
4. **Evaluation** → 354.7 mean intake over 100 episodes
5. **Winner declared** → Best model saved to this directory

**Date:** 2025-11-11
**Status:** PRODUCTION READY

---

## Next Steps

### For Deployment
1. Load model weights from `model.zip`
2. Load normalization stats from `vecnorm.pkl`
3. Use with 10 agents, 15 patches environment

### For Further Research
- Fine-tune on specific scenarios
- Extend to 15+ agents
- Add attention mechanisms
- Improve fairness (lower Gini)

---

## Support

- **Full documentation:** See `../../WINNING_MODEL_354_INTAKE.md`
- **Architecture details:** See `../../ARCHITECTURE_WINNER.md`
- **Training code:** See `train/train_ppo.py`
- **Network code:** See `networks/wide_mlp.py`

---

**Model Version:** 1.0
**Last Updated:** 2025-11-11
**Maintainer:** Trained via PPO curriculum learning
**License:** Research use
