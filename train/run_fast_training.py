"""
FAST TRAINING: Regular PPO for quick demonstration.

Uses standard PPO (not RecurrentPPO) for faster training and demonstration.
Trains in ~5-10 minutes
"""

import yaml
import os
import json
import time
import sys
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
import supersuit as ss
from env.flockforage_parallel import FlockForageParallel, EnvConfig


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def make_vec_env(env_cfg: dict, n_envs: int):
    def env_fn():
        return FlockForageParallel(EnvConfig(**env_cfg))

    env = env_fn()
    venv = ss.pettingzoo_env_to_vec_env_v1(env)
    venv = ss.concat_vec_envs_v1(
        venv, num_vec_envs=n_envs, num_cpus=0, base_class="stable_baselines3"
    )
    return venv


class ProgressCallback(BaseCallback):
    def __init__(self, total_timesteps: int):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.start_time = time.time()
        self.last_print = 0

    def _on_step(self) -> bool:
        progress = self.num_timesteps / self.total_timesteps
        if progress - self.last_print >= 0.1:  # Print every 10%
            elapsed = time.time() - self.start_time
            eta = (elapsed / progress) * (1 - progress) if progress > 0 else 0
            print(
                f"  Progress: {progress*100:.0f}% | "
                f"Timesteps: {self.num_timesteps:,}/{self.total_timesteps:,} | "
                f"Elapsed: {elapsed/60:.1f}min | "
                f"ETA: {eta/60:.1f}min",
                flush=True,
            )
            self.last_print = progress
        return True


def main():
    print("=" * 80)
    print("FAST TRAINING: Regular PPO for Quick Demonstration")
    print("=" * 80)
    print("\nConfiguration:")
    print("  • Algorithm: Standard PPO (faster than RecurrentPPO)")
    print("  • Agents: 10")
    print("  • Total timesteps: 2,000,000 (~5-10 min)")
    print("  • Reward: 30x food - 3x overcrowding")
    print("=" * 80 + "\n")

    # Load config
    config_path = "configs/env_curriculum_phase2.yaml"
    if os.path.exists(config_path):
        env_cfg = load_yaml(config_path)
        print(f"✅ Loaded config: {config_path}\n")
    else:
        print(f"⚠️  Config not found, using defaults\n")
        env_cfg = {"n_agents": 10, "n_patches": 15}

    # Create environment
    print("Creating vectorized environment...")
    n_envs = 4  # Parallel environments for faster training
    venv = make_vec_env(env_cfg, n_envs)
    venv = VecNormalize(
        venv,
        training=True,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99,
    )
    print(f"✅ Created {n_envs} parallel environments\n")

    # Create PPO model
    print("Creating PPO model...")
    model = PPO(
        "MlpPolicy",
        venv,
        gamma=0.99,
        n_steps=2048,
        batch_size=512,
        n_epochs=10,
        clip_range=0.2,
        ent_coef=0.02,
        learning_rate=5e-4,
        max_grad_norm=0.5,
        verbose=0,
        policy_kwargs=dict(net_arch=[dict(pi=[128, 128], vf=[128, 128])]),
    )
    print("✅ Model created\n")

    # Train
    total_timesteps = 2_000_000
    print(f"Starting training for {total_timesteps:,} timesteps...")
    print("Progress updates every 10%:\n")

    callback = ProgressCallback(total_timesteps)
    start = time.time()

    model.learn(total_timesteps=total_timesteps, progress_bar=False, callback=callback)

    train_time = time.time() - start
    print(f"\n✅ Training completed in {train_time/60:.1f} minutes\n")

    # Save model
    os.makedirs("models/fast_ppo", exist_ok=True)
    model_path = "models/fast_ppo/ppo_model"
    model.save(model_path)
    venv.save("models/fast_ppo/vecnorm.pkl")
    print(f"✅ Model saved to: {model_path}.zip\n")

    # Quick evaluation
    print("=" * 80)
    print("Quick Evaluation (5 episodes)")
    print("=" * 80)

    from metrics.fairness import gini

    venv.training = False
    venv.norm_reward = False

    unwrapped_env = FlockForageParallel(EnvConfig(**env_cfg))

    ginis = []
    rewards = []

    for ep in range(5):
        obs = venv.reset()
        unwrapped_env.reset(seed=ep * 42)

        ep_reward = 0
        for step in range(env_cfg.get("episode_len", 1500)):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = venv.step(action)
            ep_reward += reward[0]

            if np.any(done):
                break

        gini_val = gini(unwrapped_env._intake_total + 1e-8)
        ginis.append(gini_val)
        rewards.append(ep_reward)

        print(f"  Episode {ep+1}: Reward={ep_reward:.2f}, Gini={gini_val:.3f}")

    print(f"\nMean Reward: {np.mean(rewards):.2f}")
    print(f"Mean Gini: {np.mean(ginis):.3f}")

    # Save results
    results = {
        "training_time_min": train_time / 60,
        "total_timesteps": total_timesteps,
        "evaluation": {
            "mean_reward": float(np.mean(rewards)),
            "mean_gini": float(np.mean(ginis)),
        },
    }

    os.makedirs("results", exist_ok=True)
    with open("results/fast_training_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Results saved to: results/fast_training_results.json")
    print("=" * 80)
    print("\nNext steps:")
    print("  • Generate RL video: python -m visualize.generate_video_fast --steps 300")
    print("  • Compare vs baseline: python -m train.compare_baseline_vs_rl_fast")
    print("=" * 80)

    venv.close()
    unwrapped_env.close()


if __name__ == "__main__":
    main()
