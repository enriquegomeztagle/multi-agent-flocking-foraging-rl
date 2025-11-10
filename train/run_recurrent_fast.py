"""
FAST RecurrentPPO: Test with LSTM and Curriculum Learning.

Quick test version:
- Phase 1: 5 agents, 500K steps (~2-3 min)
- Phase 2: 10 agents, 500K steps (~2-3 min)
- Total: 1M steps (~5-6 min)
"""

import yaml
import os
import json
import time
import sys
import numpy as np
from sb3_contrib import RecurrentPPO
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
    venv = ss.concat_vec_envs_v1(venv, num_vec_envs=n_envs, num_cpus=0, base_class="stable_baselines3")
    return venv


class ProgressCallback(BaseCallback):
    def __init__(self, total_timesteps: int, phase_name: str):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.phase_name = phase_name
        self.start_time = time.time()
        self.last_print = 0

    def _on_step(self) -> bool:
        progress = self.num_timesteps / self.total_timesteps
        if progress - self.last_print >= 0.2:  # Print every 20%
            elapsed = time.time() - self.start_time
            eta = (elapsed / progress) * (1 - progress) if progress > 0 else 0
            print(f"  [{self.phase_name}] Progress: {progress*100:.0f}% | "
                  f"Timesteps: {self.num_timesteps:,}/{self.total_timesteps:,} | "
                  f"Elapsed: {elapsed/60:.1f}min | ETA: {eta/60:.1f}min", flush=True)
            self.last_print = progress
        return True


def train_phase(env_cfg, phase_name, total_timesteps, n_envs, prev_model_path=None):
    """Train one curriculum phase."""
    print(f"\n{'='*80}")
    print(f"PHASE: {phase_name}")
    print(f"{'='*80}")
    print(f"Agents: {env_cfg['n_agents']}")
    print(f"Patches: {env_cfg['n_patches']}")
    print(f"Timesteps: {total_timesteps:,}")
    print(f"{'='*80}\n", flush=True)

    venv = make_vec_env(env_cfg, n_envs)
    venv = VecNormalize(venv, training=True, norm_obs=True, norm_reward=True,
                       clip_obs=10.0, clip_reward=10.0, gamma=0.99)

    # Create or load RecurrentPPO model
    if prev_model_path and os.path.exists(f"{prev_model_path}.zip"):
        print(f"Loading previous model from {prev_model_path}...", flush=True)
        model = RecurrentPPO.load(prev_model_path)
        model.set_env(venv)

        # Load previous normalizer
        vecnorm_path = f"{os.path.dirname(prev_model_path)}/vecnorm.pkl"
        if os.path.exists(vecnorm_path):
            venv_prev = VecNormalize.load(vecnorm_path, venv)
            venv.obs_rms = venv_prev.obs_rms
            venv.ret_rms = venv_prev.ret_rms
    else:
        print("Creating RecurrentPPO model with LSTM...", flush=True)
        model = RecurrentPPO(
            "MlpLstmPolicy",
            venv,
            gamma=0.99,
            n_steps=1024,  # Reduced from 2048 for faster training
            batch_size=256,  # Reduced from 512
            n_epochs=5,  # Reduced from 10
            clip_range=0.2,
            ent_coef=0.02,
            learning_rate=5e-4,
            max_grad_norm=0.5,
            verbose=0,
            policy_kwargs=dict(
                n_lstm_layers=1,
                lstm_hidden_size=32,  # Reduced from 64 for faster init
                enable_critic_lstm=True,
                shared_lstm=False
            )
        )
        print("✅ Model created!", flush=True)

    callback = ProgressCallback(total_timesteps, phase_name)
    start = time.time()

    print(f"\nStarting training...\n", flush=True)
    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=False,
        callback=callback,
        reset_num_timesteps=False if prev_model_path else True
    )

    train_time = time.time() - start
    print(f"\n✅ {phase_name} completed in {train_time/60:.1f}min", flush=True)

    return model, venv, train_time


def main():
    print("=" * 80)
    print("FAST RecurrentPPO TEST: LSTM + Curriculum Learning")
    print("=" * 80)
    print("\nConfiguration:")
    print("  • Algorithm: RecurrentPPO with LSTM")
    print("  • Curriculum: Phase 1 (5 agents) → Phase 2 (10 agents)")
    print("  • Total timesteps: 1,000,000 (~5-6 min)")
    print("  • LSTM hidden size: 32 (reduced for speed)")
    print("=" * 80 + "\n", flush=True)

    # Phase 1: Easy (5 agents)
    print("\n" + "=" * 80)
    print("CURRICULUM PHASE 1: EASY (5 agents, 500K steps)")
    print("=" * 80, flush=True)

    phase1_cfg = {
        'n_agents': 5,
        'n_patches': 12,
        'width': 30.0,
        'height': 30.0,
        'episode_len': 1500,
        'feed_radius': 3.0,
        'c_max': 0.06,
        'S_max': 1.0,
        'regen_r': 0.3
    }

    model, venv1, time1 = train_phase(
        phase1_cfg,
        "Phase 1 (5 agents)",
        total_timesteps=500_000,
        n_envs=2  # Reduced from 4 for stability
    )

    # Save Phase 1
    os.makedirs("models/recurrent_fast/phase1", exist_ok=True)
    model.save("models/recurrent_fast/phase1/model")
    venv1.save("models/recurrent_fast/phase1/vecnorm.pkl")
    print(f"✅ Phase 1 model saved\n", flush=True)

    # Phase 2: Full difficulty (10 agents)
    print("\n" + "=" * 80)
    print("CURRICULUM PHASE 2: FULL (10 agents, 500K steps)")
    print("=" * 80, flush=True)

    phase2_cfg = {
        'n_agents': 10,
        'n_patches': 15,
        'width': 30.0,
        'height': 30.0,
        'episode_len': 1500,
        'feed_radius': 3.0,
        'c_max': 0.06,
        'S_max': 1.0,
        'regen_r': 0.3
    }

    model, venv2, time2 = train_phase(
        phase2_cfg,
        "Phase 2 (10 agents)",
        total_timesteps=500_000,
        n_envs=2,
        prev_model_path="models/recurrent_fast/phase1/model"
    )

    # Save final model
    os.makedirs("models/recurrent_fast/final", exist_ok=True)
    model.save("models/recurrent_fast/final/model")
    venv2.save("models/recurrent_fast/final/vecnorm.pkl")
    print(f"✅ Final model saved\n", flush=True)

    # Quick evaluation
    print("=" * 80)
    print("Quick Evaluation (3 episodes)")
    print("=" * 80, flush=True)

    from metrics.fairness import gini

    venv2.training = False
    venv2.norm_reward = False

    unwrapped_env = FlockForageParallel(EnvConfig(**phase2_cfg))

    rewards = []
    intakes = []

    for ep in range(3):
        obs = venv2.reset()
        unwrapped_env.reset(seed=ep * 42)

        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)

        ep_reward = 0
        for step in range(phase2_cfg['episode_len']):
            action, lstm_states = model.predict(obs, state=lstm_states,
                                                episode_start=episode_starts,
                                                deterministic=True)
            episode_starts = np.zeros((1,), dtype=bool)
            obs, reward, done, info = venv2.step(action)
            ep_reward += reward[0]

            if np.any(done):
                break

        intake = np.sum(unwrapped_env._intake_total)
        rewards.append(ep_reward)
        intakes.append(intake)

        print(f"  Episode {ep+1}: Reward={ep_reward:.2f}, Intake={intake:.2f}")

    print(f"\nMean Reward: {np.mean(rewards):.2f}")
    print(f"Mean Intake: {np.mean(intakes):.2f}")

    # Save results
    results = {
        "training_time_min": (time1 + time2) / 60,
        "total_timesteps": 1_000_000,
        "phase1_time_min": time1 / 60,
        "phase2_time_min": time2 / 60,
        "evaluation": {
            "mean_reward": float(np.mean(rewards)),
            "mean_intake": float(np.mean(intakes)),
        }
    }

    os.makedirs("results", exist_ok=True)
    with open("results/recurrent_fast_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Results saved to: results/recurrent_fast_results.json")
    print("=" * 80)
    print("\nTotal training time: {:.1f} minutes".format((time1 + time2) / 60))
    print("=" * 80, flush=True)

    venv1.close()
    venv2.close()
    unwrapped_env.close()


if __name__ == "__main__":
    main()
