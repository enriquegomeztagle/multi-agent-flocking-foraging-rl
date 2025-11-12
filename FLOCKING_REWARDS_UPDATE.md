# Flocking Rewards Update

## Overview
Added explicit flocking rewards to make both **flocking** and **foraging** necessary for success, elevating the complexity of the multi-agent coordination task.

## Motivation
Previously, the reward structure was heavily focused on foraging:
- Food reward: 200x multiplier
- Proximity to patches
- Approach rewards
- Light overcrowding penalty

**Missing**: Explicit incentives for flocking behaviors (cohesion, alignment, separation)

This update makes flocking behaviors essential, not optional, for achieving high performance.

## New Flocking Reward Components

### 5a. Cohesion Reward (Stay Close to Neighbors)
```python
# Reward staying within reasonable distance of neighbors
if mean_neighbor_dist < 15.0:
    cohesion_reward = 1.5 * np.exp(-mean_neighbor_dist / 10.0)
    rewards[i] += cohesion_reward
else:
    # Penalty for being isolated
    rewards[i] -= 1.0
```
**Purpose**: Prevents agents from wandering alone; encourages group formation

### 5b. Alignment Reward (Match Velocities)
```python
# Reward when moving in similar direction as neighbors
vel_similarity = np.dot(self._vel[i], mean_neighbor_vel)
vel_similarity /= (np.linalg.norm(self._vel[i]) * np.linalg.norm(mean_neighbor_vel) + 1e-6)
alignment_reward = 1.0 * max(0, vel_similarity)
rewards[i] += alignment_reward
```
**Purpose**: Encourages coordinated movement; agents move together as a flock

### 5c. Separation Reward (Maintain Safe Distance)
```python
# Penalize being too close to neighbors
too_close = neighbor_distances < self.cfg.d_safe
if np.any(too_close):
    separation_penalty = -2.0 * np.sum(too_close)
    rewards[i] += separation_penalty
```
**Purpose**: Prevents collisions and overcrowding; maintains personal space

### 5d. Group Foraging Bonus (Coordinated Foraging)
```python
# Bonus if neighbors are also feeding or near food
neighbors_feeding = intake[neighbor_indices] > 0
if intake[i] > 0 and np.any(neighbors_feeding):
    group_bonus = 5.0 * (np.sum(neighbors_feeding) / len(neighbor_indices))
    rewards[i] += group_bonus
```
**Purpose**: **Directly rewards flocking + foraging**; agents must coordinate to get bonus

## Reward Weight Balancing

| Component | Weight | Purpose |
|-----------|--------|---------|
| Food Reward | 200x | Primary objective (unchanged) |
| Proximity | 2.0 | Find food patches (unchanged) |
| Approach | 3.0 | Move toward food (unchanged) |
| Overcrowding | -0.5 | Prevent clustering (unchanged) |
| **Cohesion** | **1.5** | **Stay with group (NEW)** |
| **Alignment** | **1.0** | **Move together (NEW)** |
| **Separation** | **-2.0** | **Avoid collisions (NEW)** |
| **Group Foraging** | **5.0** | **Coordinate feeding (NEW)** |

## Expected Behavioral Changes

### Before (Foraging Only)
- Agents could succeed independently
- No incentive to stay together
- Random/scattered movement patterns
- Individual exploitation of patches

### After (Flocking + Foraging Required)
- Agents **must** stay in groups (cohesion reward)
- Agents **must** move together (alignment reward)
- Agents **must** coordinate foraging (group bonus)
- **Emergent flocking behavior** becomes necessary for optimal performance

## Training Configuration

### Easy Mode Training (NEW)
```bash
Configuration: configs/env_easy_mode.yaml
Output: models/ppo_easy_mode_with_flocking
Timesteps: 1,500,000
Agents: 5
Patches: 20
World: 20×20
```

**Goal**: Validate that flocking rewards work in simple environment

### Medium Mode Training (Ongoing)
```bash
Configuration: configs/env_medium_mode.yaml (updated)
Output: models/ppo_medium_mode_v2
Timesteps: 2,000,000
Agents: 10
Patches: 15
World: 28×28
```

**Goal**: Test scalability with easier config and flocking rewards

## Evaluation Metrics

To verify both flocking and foraging are necessary:

1. **Food Efficiency**: % of theoretical max intake
2. **Flock Cohesion**: Mean distance to k-nearest neighbors
3. **Velocity Alignment**: Cosine similarity of velocities
4. **Group Foraging Events**: % of feeding done near other agents
5. **Separation Violations**: Count of collisions/too-close events

## Expected Results

With explicit flocking rewards:
- ✅ More coordinated group movement
- ✅ Better spatial distribution (not scattered)
- ✅ Emergent flock formations
- ✅ Coordinated patch exploitation
- ✅ Both behaviors (flocking + foraging) required for high reward

## Next Steps

1. ✅ Complete easy mode training (~45 min)
2. ✅ Complete medium mode v2 training (~30 min remaining)
3. Evaluate flocking metrics on trained models
4. Compare to previous models (without explicit flocking)
5. Visualize flock formations and coordinated foraging

## Code Changes

**Modified File**: `env/flockforage_parallel.py`
- **Function**: `_compute_rewards()`
- **Lines Added**: ~45 new lines (flocking reward logic)
- **Backward Compatible**: Yes (only additions, no breaking changes)

## References

Classic Boids Flocking Rules (Reynolds, 1987):
- **Separation**: Avoid crowding neighbors
- **Alignment**: Steer towards average heading
- **Cohesion**: Steer towards average position

Our implementation adapts these as **reward signals** rather than hard-coded steering rules, allowing the RL agent to learn the optimal balance between flocking and foraging.
