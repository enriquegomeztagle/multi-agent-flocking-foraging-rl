"""
ADVANCED TRAINING: RecurrentPPO + Curriculum Learning + Long Training

1. RecurrentPPO with LSTM (agents remember where they've been)
2. Curriculum learning (5 agents → 10 agents)
3. 10M timesteps total (5M phase 1 + 5M phase 2)
4. Ultra-simple rewards (30x food - 3x overcrowding)
"""

import yaml
import os
import json
import time
import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
import supersuit as ss
from env.flockforage_parallel import FlockForageParallel, EnvConfig
from metrics.fairness import gini


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
    def __init__(self, total_timesteps: int, phase_name: str):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.phase_name = phase_name
        self.start_time = time.time()
        self.last_print = 0

    def _on_step(self) -> bool:
        progress = self.num_timesteps / self.total_timesteps
        if progress - self.last_print >= 0.1:
            elapsed = time.time() - self.start_time
            eta = (elapsed / progress) * (1 - progress) if progress > 0 else 0
            print(
                f"  [{self.phase_name}] {progress*100:.0f}% | {self.num_timesteps:,} steps | ETA: {eta/60:.1f}min"
            )
            self.last_print = progress
        return True


def train_phase(
    phase_name: str,
    env_cfg: dict,
    total_timesteps: int,
    n_envs: int,
    prev_model_path: str = None,
):
    """Train a single curriculum phase."""
    print(f"\n{'='*80}")
    print(f"PHASE: {phase_name}")
    print(f"{'='*80}")
    print(f"Agents: {env_cfg['n_agents']}")
    print(f"Patches: {env_cfg['n_patches']}")
    print(f"Timesteps: {total_timesteps:,}")
    print(f"{'='*80}\n")

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

    # Create or load RecurrentPPO model
    if prev_model_path and os.path.exists(f"{prev_model_path}.zip"):
        print(f"Loading previous model from {prev_model_path}...")
        # Load previous model and update env
        model = RecurrentPPO.load(prev_model_path)
        model.set_env(venv)
        # Load previous normalizer
        if os.path.exists(f"{prev_model_path.replace('/model', '')}/vecnorm.pkl"):
            venv_prev = VecNormalize.load(
                f"{prev_model_path.replace('/model', '')}/vecnorm.pkl", venv
            )
            venv.obs_rms = venv_prev.obs_rms
            venv.ret_rms = venv_prev.ret_rms
    else:
        print("Creating new RecurrentPPO model with LSTM...")
        model = RecurrentPPO(
            "MlpLstmPolicy",  # LSTM policy for memory
            venv,
            gamma=0.99,
            n_steps=2048,
            batch_size=512,  # Smaller batch for LSTM
            n_epochs=10,
            clip_range=0.2,
            ent_coef=0.02,
            learning_rate=5e-4,
            max_grad_norm=0.5,
            verbose=0,
            policy_kwargs=dict(
                n_lstm_layers=1,
                lstm_hidden_size=64,  # LSTM hidden size
                enable_critic_lstm=True,
                shared_lstm=False,
            ),
        )

    callback = ProgressCallback(total_timesteps, phase_name)
    start = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=False,
        callback=callback,
        reset_num_timesteps=False if prev_model_path else True,
    )
    train_time = time.time() - start

    print(f"\n✓ {phase_name} completed in {train_time/60:.1f}min")

    return model, venv, train_time


def evaluate_model(model, env_cfg: dict, venv, n_episodes: int = 20):
    """Evaluate model performance."""
    print(f"\nEvaluating {n_episodes} episodes...")

    venv.training = False
    venv.norm_reward = False

    unwrapped_env = FlockForageParallel(EnvConfig(**env_cfg))

    all_intakes = []
    all_ginis = []

    for ep in range(n_episodes):
        obs = venv.reset()
        unwrapped_env.reset(seed=ep * 42)

        # Reset LSTM states
        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)

        for step in range(env_cfg["episode_len"]):
            action, lstm_states = model.predict(
                obs, state=lstm_states, episode_start=episode_starts, deterministic=True
            )
            obs, reward, done, info = venv.step(action)
            episode_starts = done

            action_dict = {
                f"agent_{i}": int(action[i]) for i in range(len(unwrapped_env.agents))
            }
            unwrapped_env.step(action_dict)

            if np.any(done):
                break

        intake = float(unwrapped_env._intake_total.sum())
        all_intakes.append(intake)
        all_ginis.append(float(gini(unwrapped_env._intake_total + 1e-8)))

        if (ep + 1) % 5 == 0:
            eff = (
                intake
                / (env_cfg["c_max"] * env_cfg["episode_len"] * env_cfg["n_agents"])
            ) * 100
            print(f"  Episode {ep+1}: {intake:.1f} ({eff:.1f}%)")

    return all_intakes, all_ginis


def main():
    print("\n" + "=" * 80)
    print("ADVANCED TRAINING: RecurrentPPO + Curriculum + Long Training")
    print("=" * 80)
    print("\nImprovements:")
    print("  • RecurrentPPO with LSTM (agents remember past)")
    print("  • Curriculum: Phase 1 (5 agents) → Phase 2 (10 agents)")
    print("  • 10M total timesteps (5M per phase)")
    print("  • Ultra-simple rewards: 30x food - 3x overcrowding")
    print("=" * 80)

    n_envs = 16

    # === PHASE 1: Easy (5 agents) ===
    print("\n" + "=" * 80)
    print("CURRICULUM PHASE 1: EASY (5 agents, 5M timesteps)")
    print("=" * 80)

    phase1_cfg = load_yaml("configs/env_curriculum_phase1.yaml")
    phase1_timesteps = 5_000_000

    model_phase1, venv_phase1, time_phase1 = train_phase(
        "Phase 1 (5 agents)", phase1_cfg, phase1_timesteps, n_envs
    )

    # Save Phase 1
    os.makedirs("models/curriculum_phase1", exist_ok=True)
    model_phase1.save("models/curriculum_phase1/model")
    venv_phase1.save("models/curriculum_phase1/vecnorm.pkl")

    # Quick eval Phase 1
    print("\n" + "=" * 80)
    print("PHASE 1 EVALUATION")
    print("=" * 80)
    intakes_p1, ginis_p1 = evaluate_model(
        model_phase1, phase1_cfg, venv_phase1, n_episodes=10
    )
    efficiency_p1 = (
        np.mean(intakes_p1)
        / (phase1_cfg["c_max"] * phase1_cfg["episode_len"] * phase1_cfg["n_agents"])
    ) * 100
    print(
        f"\nPhase 1 Results: {np.mean(intakes_p1):.1f} intake ({efficiency_p1:.1f}% efficiency)"
    )

    venv_phase1.close()

    # === PHASE 2: Full difficulty (10 agents) ===
    print("\n" + "=" * 80)
    print("CURRICULUM PHASE 2: FULL (10 agents, 5M timesteps)")
    print("=" * 80)
    print("Transferring knowledge from Phase 1...")

    phase2_cfg = load_yaml("configs/env_curriculum_phase2.yaml")
    phase2_timesteps = 5_000_000

    model_phase2, venv_phase2, time_phase2 = train_phase(
        "Phase 2 (10 agents)",
        phase2_cfg,
        phase2_timesteps,
        n_envs,
        prev_model_path="models/curriculum_phase1/model",  # Transfer learning
    )

    # Save Phase 2 (final model)
    os.makedirs("models/advanced_final", exist_ok=True)
    model_phase2.save("models/advanced_final/model")
    venv_phase2.save("models/advanced_final/vecnorm.pkl")

    # === FINAL EVALUATION ===
    print("\n" + "=" * 80)
    print("FINAL EVALUATION (20 episodes)")
    print("=" * 80)

    intakes_final, ginis_final = evaluate_model(
        model_phase2, phase2_cfg, venv_phase2, n_episodes=20
    )

    venv_phase2.close()

    # Results
    mean_intake = np.mean(intakes_final)
    theoretical_max = (
        phase2_cfg["c_max"] * phase2_cfg["episode_len"] * phase2_cfg["n_agents"]
    )
    efficiency_final = (mean_intake / theoretical_max) * 100

    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"Training time:    {(time_phase1 + time_phase2)/60:.1f} minutes")
    print(f"\nPhase 1 (5 agents):   {np.mean(intakes_p1):.1f} ({efficiency_p1:.1f}%)")
    print(f"Phase 2 (10 agents):  {mean_intake:.1f} ({efficiency_final:.1f}%)")
    print(f"\nFinal Performance:")
    print(f"  Mean Intake:    {mean_intake:.1f} ± {np.std(intakes_final):.1f}")
    print(f"  Best:           {np.max(intakes_final):.1f}")
    print(f"  Worst:          {np.min(intakes_final):.1f}")
    print(f"  Gini (mean):    {np.mean(ginis_final):.3f}")
    print(f"\n🎯 Theoretical:   {theoretical_max:.0f}")
    print(f"📊 Efficiency:    {efficiency_final:.1f}%")
    print("=" * 80)

    if efficiency_final >= 70:
        print("\n🎉 EXCELLENT! Achieved 70%+ target!")
    elif efficiency_final >= 60:
        print("\n✅ VERY GOOD! Close to 70% target!")
    elif efficiency_final >= 50:
        print("\n👍 GOOD! Significant improvement!")
    elif efficiency_final >= 40:
        print("\n⚠️  IMPROVED but below target")
    else:
        print("\n❌ Still below target")

    # Save results
    results = {
        "efficiency_pct": float(efficiency_final),
        "mean_intake": float(mean_intake),
        "phase1_efficiency": float(efficiency_p1),
        "phase2_efficiency": float(efficiency_final),
        "all_intakes": [float(x) for x in intakes_final],
        "all_ginis": [float(x) for x in ginis_final],
        "train_time_min": (time_phase1 + time_phase2) / 60,
        "config": phase2_cfg,
    }

    os.makedirs("results", exist_ok=True)
    with open("results/advanced_training_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Saved to: results/advanced_training_results.json\n")


if __name__ == "__main__":
    main()
