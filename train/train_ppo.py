import os
import json
import time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
import supersuit as ss
from env.flockforage_parallel import FlockForageParallel, EnvConfig
from metrics.fairness import gini


# ============================================================================
# CONFIGURACIÓN - REPLICA OF SUCCESS (270 INTAKE, 11.7%)
# ============================================================================
CONFIG = {
    # Timesteps por fase - EXACT CONFIG THAT WORKED
    "phase1_steps": 2_000_000,  # 2M para fase fácil (~9 min)
    "phase2_steps": 2_000_000,  # 2M para fase completa (~8 min)
    # Total: 4M steps, ~17 min
    # Environment configs
    "phase1": {
        "n_agents": 5,
        "n_patches": 12,
        "width": 30.0,
        "height": 30.0,
        "episode_len": 1500,
        "feed_radius": 3.0,
        "c_max": 0.06,
        "S_max": 1.0,
        "regen_r": 0.3,
    },
    "phase2": {
        "n_agents": 10,
        "n_patches": 15,
        "width": 30.0,
        "height": 30.0,
        "episode_len": 1500,
        "feed_radius": 3.0,
        "c_max": 0.06,
        "S_max": 1.0,
        "regen_r": 0.3,
    },
    # PPO Hyperparameters
    "ppo": {
        "gamma": 0.99,
        "n_steps": 2048,  # Steps antes de actualizar
        "batch_size": 512,  # Batch size para entrenamiento
        "n_epochs": 10,  # Epochs por actualización
        "clip_range": 0.2,
        "ent_coef": 0.02,  # Entropy (exploración)
        "learning_rate": 3e-4,  # Learning rate estándar
        "max_grad_norm": 0.5,
        "n_envs": 4,  # 4 parallel environments
    },
}
# ============================================================================


def make_vec_env(env_cfg: dict, n_envs: int):
    """Create vectorized environment."""

    def env_fn():
        return FlockForageParallel(EnvConfig(**env_cfg))

    env = env_fn()
    venv = ss.pettingzoo_env_to_vec_env_v1(env)
    venv = ss.concat_vec_envs_v1(
        venv, num_vec_envs=n_envs, num_cpus=0, base_class="stable_baselines3"
    )
    return venv


class ProgressCallback(BaseCallback):
    """Callback para mostrar progreso cada 10%."""

    def __init__(self, total_timesteps: int, phase_name: str):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.phase_name = phase_name
        self.start_time = time.time()
        self.last_print = 0

    def _on_step(self) -> bool:
        progress = self.num_timesteps / self.total_timesteps
        if progress - self.last_print >= 0.1:  # Print every 10%
            elapsed = time.time() - self.start_time
            eta = (elapsed / progress) * (1 - progress) if progress > 0 else 0
            print(
                f"  [{self.phase_name}] {progress*100:.0f}% | "
                f"{self.num_timesteps:,}/{self.total_timesteps:,} steps | "
                f"Elapsed: {elapsed/60:.1f}min | ETA: {eta/60:.1f}min",
                flush=True,
            )
            self.last_print = progress
        return True


def train_phase(env_cfg, phase_name, total_timesteps, prev_model_path=None):
    """Train one curriculum phase."""
    print(f"\n{'='*80}")
    print(f"PHASE: {phase_name}")
    print(f"{'='*80}")
    print(f"Agents: {env_cfg['n_agents']}")
    print(f"Patches: {env_cfg['n_patches']}")
    print(f"Timesteps: {total_timesteps:,}")
    print(f"{'='*80}\n", flush=True)

    # Create vectorized environment
    venv = make_vec_env(env_cfg, CONFIG["ppo"]["n_envs"])
    venv = VecNormalize(
        venv,
        training=True,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99,
    )

    # Create or load PPO model
    if prev_model_path and os.path.exists(f"{prev_model_path}.zip"):
        print(f"Loading previous model from {prev_model_path}...", flush=True)
        model = PPO.load(prev_model_path, env=venv)

        # Load previous normalizer
        vecnorm_path = f"{os.path.dirname(prev_model_path)}/vecnorm.pkl"
        if os.path.exists(vecnorm_path):
            venv_prev = VecNormalize.load(vecnorm_path, venv)
            venv.obs_rms = venv_prev.obs_rms
            venv.ret_rms = venv_prev.ret_rms
        print("✅ Model loaded with transfer learning!", flush=True)
    else:
        print("Creating new PPO model...", flush=True)
        model = PPO(
            "MlpPolicy",
            venv,
            gamma=CONFIG["ppo"]["gamma"],
            n_steps=CONFIG["ppo"]["n_steps"],
            batch_size=CONFIG["ppo"]["batch_size"],
            n_epochs=CONFIG["ppo"]["n_epochs"],
            clip_range=CONFIG["ppo"]["clip_range"],
            ent_coef=CONFIG["ppo"]["ent_coef"],
            learning_rate=CONFIG["ppo"]["learning_rate"],
            max_grad_norm=CONFIG["ppo"]["max_grad_norm"],
            verbose=0,
            policy_kwargs=dict(
                net_arch=[dict(pi=[256, 256], vf=[256, 256])]  # Redes más grandes
            ),
        )
        print("✅ Model created!", flush=True)

    # Train
    callback = ProgressCallback(total_timesteps, phase_name)
    start = time.time()

    print(f"\nStarting training...\n", flush=True)
    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=False,
        callback=callback,
        reset_num_timesteps=False if prev_model_path else True,
    )

    train_time = time.time() - start
    print(f"\n✅ {phase_name} completed in {train_time/60:.1f}min\n", flush=True)

    return model, venv, train_time


def evaluate_model(model, env_cfg: dict, n_episodes: int = 5):
    """Evaluate model performance."""
    print(f"\nEvaluating {n_episodes} episodes...")

    env = FlockForageParallel(EnvConfig(**env_cfg))

    intakes = []
    rewards_list = []
    ginis = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep * 42)

        ep_reward = 0
        for step in range(env_cfg["episode_len"]):
            # Convert dict obs to array
            obs_array = np.array([obs[agent] for agent in env.agents])

            # Predict
            action, _ = model.predict(obs_array, deterministic=True)

            # Convert to dict
            action_dict = {agent: int(action[i]) for i, agent in enumerate(env.agents)}

            # Step
            obs, rew, terms, truncs, _ = env.step(action_dict)
            ep_reward += sum(rew.values())

            if any(terms.values()) or any(truncs.values()):
                break

        intake = np.sum(env._intake_total)
        gini_val = gini(env._intake_total + 1e-8)

        intakes.append(intake)
        rewards_list.append(ep_reward)
        ginis.append(gini_val)

        print(
            f"  Episode {ep+1}: Intake={intake:.2f}, Reward={ep_reward:.2f}, Gini={gini_val:.3f}"
        )

    env.close()

    return {
        "mean_intake": float(np.mean(intakes)),
        "mean_reward": float(np.mean(rewards_list)),
        "mean_gini": float(np.mean(ginis)),
    }


def main():
    print("=" * 80)
    print("TRAINING FINAL: PPO Simple + Rewards v3 + Long Training")
    print("=" * 80)
    print("\nConfiguration:")
    print(f"  • Algorithm: PPO (Simple, sin LSTM)")
    print(f"  • Rewards: v3 Optimized (200x food, proximity, approach)")
    print(f"  • Curriculum: Phase 1 (5 agents) → Phase 2 (10 agents)")
    print(f"  • Phase 1: {CONFIG['phase1_steps']:,} steps")
    print(f"  • Phase 2: {CONFIG['phase2_steps']:,} steps")
    print(f"  • Total: {CONFIG['phase1_steps'] + CONFIG['phase2_steps']:,} steps")
    print(f"  • Estimated time: ~45-50 minutes")
    print("=" * 80 + "\n", flush=True)

    # PHASE 1: Easy (5 agents)
    print("\n" + "=" * 80)
    print("CURRICULUM PHASE 1: EASY (5 agents)")
    print("=" * 80, flush=True)

    model, venv1, time1 = train_phase(
        CONFIG["phase1"], "Phase 1 (5 agents)", CONFIG["phase1_steps"]
    )

    # Save Phase 1
    os.makedirs("models/ppo_final/phase1", exist_ok=True)
    model.save("models/ppo_final/phase1/model")
    venv1.save("models/ppo_final/phase1/vecnorm.pkl")
    print("✅ Phase 1 model saved\n", flush=True)

    # PHASE 2: Full (10 agents)
    print("\n" + "=" * 80)
    print("CURRICULUM PHASE 2: FULL (10 agents)")
    print("=" * 80, flush=True)

    model, venv2, time2 = train_phase(
        CONFIG["phase2"],
        "Phase 2 (10 agents)",
        CONFIG["phase2_steps"],
        prev_model_path="models/ppo_final/phase1/model",
    )

    # Save final model
    os.makedirs("models/ppo_final/final", exist_ok=True)
    model.save("models/ppo_final/final/model")
    venv2.save("models/ppo_final/final/vecnorm.pkl")
    print("✅ Final model saved\n", flush=True)

    # Evaluate
    print("=" * 80)
    print("FINAL EVALUATION")
    print("=" * 80, flush=True)

    results = evaluate_model(model, CONFIG["phase2"], n_episodes=10)

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Mean Intake:  {results['mean_intake']:.2f}")
    print(f"Mean Reward:  {results['mean_reward']:.2f}")
    print(f"Mean Gini:    {results['mean_gini']:.3f}")
    print()
    print("COMPARISON WITH BASELINE:")
    print(f"  Baseline (Boids):  2,307 intake (100%)")
    print(
        f"  PPO Final:         {results['mean_intake']:.0f} intake ({results['mean_intake']/2307*100:.1f}%)"
    )
    print("=" * 80)

    # Save results
    final_results = {
        "training_time_min": (time1 + time2) / 60,
        "total_timesteps": CONFIG["phase1_steps"] + CONFIG["phase2_steps"],
        "phase1_time_min": time1 / 60,
        "phase2_time_min": time2 / 60,
        "evaluation": results,
    }

    os.makedirs("results", exist_ok=True)
    with open("results/ppo_final_results.json", "w") as f:
        json.dump(final_results, f, indent=2)

    print(f"\n✅ Results saved to: results/ppo_final_results.json")
    print(f"\nTotal training time: {(time1 + time2)/60:.1f} minutes")
    print("=" * 80, flush=True)

    venv1.close()
    venv2.close()


if __name__ == "__main__":
    main()
