"""
Train PPO agent on multi-agent flocking and foraging environment.

This script:
1. Loads environment configuration from YAML
2. Creates vectorized parallel environments
3. Trains PPO with proper hyperparameters
4. Saves model checkpoints periodically
5. Supports resuming from checkpoints
"""

import argparse
import yaml
from pathlib import Path
import time
from typing import Dict, Any
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from sb3_contrib import RecurrentPPO

from env.flockforage_parallel import FlockForageParallel, EnvConfig
from env.gym_wrapper import FlockForageGymWrapper


def load_config(config_path: str) -> Dict[str, Any]:
    """Load environment configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def make_env(config: Dict[str, Any]):
    """Factory function to create environment instances."""
    def _init():
        env = FlockForageGymWrapper(EnvConfig(**config))
        return env
    return _init


def create_vectorized_env(config: Dict[str, Any], n_envs: int = 4):
    """
    Create vectorized environment with normalization.

    Args:
        config: Environment configuration dictionary
        n_envs: Number of parallel environments

    Returns:
        VecNormalize wrapped environment
    """
    # Create parallel environments
    env = DummyVecEnv([make_env(config) for _ in range(n_envs)])

    # Add monitoring
    env = VecMonitor(env)

    # Add normalization (important for RL)
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99,
    )

    return env


def train(
    config_path: str,
    output_dir: str,
    total_timesteps: int = 5_000_000,
    n_envs: int = 4,
    save_freq: int = 100_000,
    eval_freq: int = 50_000,
    resume_from: str = None,
    learning_rate: float = 3e-4,
    batch_size: int = 256,
    n_epochs: int = 10,
    use_lstm: bool = True,
):
    """
    Train PPO agent on the environment.

    Args:
        config_path: Path to environment config YAML
        output_dir: Directory to save models and logs
        total_timesteps: Total training timesteps
        n_envs: Number of parallel environments
        save_freq: Save checkpoint every N timesteps
        eval_freq: Evaluate every N timesteps
        resume_from: Path to model checkpoint to resume from
        learning_rate: Learning rate for PPO
        batch_size: Batch size for training
        n_epochs: Number of epochs per update
        use_lstm: Use LSTM policy for memory
    """
    print("=" * 80)
    print("TRAINING PPO ON MULTI-AGENT FLOCKING AND FORAGING")
    print("=" * 80)
    print()

    # Load configuration
    print(f"Loading config from: {config_path}")
    config = load_config(config_path)
    print(f"Configuration loaded:")
    print(f"  - Agents: {config['n_agents']}")
    print(f"  - Patches: {config['n_patches']}")
    print(f"  - World: {config['width']}x{config['height']}")
    print(f"  - Episode length: {config['episode_len']}")
    print()

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save config to output directory
    config_save_path = output_path / "env_config.yaml"
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f)
    print(f"Config saved to: {config_save_path}")
    print()

    # Create training environment
    print(f"Creating {n_envs} parallel environments...")
    train_env = create_vectorized_env(config, n_envs=n_envs)
    print(f"✅ Training environment created")
    print()

    # Create evaluation environment
    print("Creating evaluation environment...")
    eval_env = create_vectorized_env(config, n_envs=2)
    print("✅ Evaluation environment created")
    print()

    # Calculate theoretical maximum for efficiency tracking
    theoretical_max = config['n_agents'] * config['episode_len'] * config['c_max']
    print(f"Theoretical maximum intake: {theoretical_max:.0f}")
    print()

    # Create callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq // n_envs,  # Divide by n_envs since it's per environment
        save_path=str(output_path / "checkpoints"),
        name_prefix="ppo_model",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(output_path / "best_model"),
        log_path=str(output_path / "eval_logs"),
        eval_freq=eval_freq // n_envs,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )

    callbacks = [checkpoint_callback, eval_callback]

    # Create or load model
    if resume_from:
        print(f"Resuming training from: {resume_from}")
        if use_lstm:
            model = RecurrentPPO.load(
                resume_from,
                env=train_env,
                device='auto',
            )
        else:
            model = PPO.load(
                resume_from,
                env=train_env,
                device='auto',
            )
        # Load normalization stats if available
        vecnorm_path = Path(resume_from).parent / "vecnormalize.pkl"
        if vecnorm_path.exists():
            train_env = VecNormalize.load(str(vecnorm_path), train_env)
            print(f"Loaded VecNormalize stats from: {vecnorm_path}")
        print("✅ Model loaded, resuming training...")
    else:
        print("Creating new PPO model...")

        # Choose policy type and algorithm
        if use_lstm:
            # Use RecurrentPPO with LSTM
            policy_type = "MlpLstmPolicy"
            model = RecurrentPPO(
                policy_type,
                train_env,
                learning_rate=learning_rate,
                n_steps=2048,  # Longer rollouts for complex tasks
                batch_size=batch_size,
                n_epochs=n_epochs,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                clip_range_vf=None,
                ent_coef=0.01,  # Encourage exploration
                vf_coef=0.5,
                max_grad_norm=0.5,
                target_kl=0.01,
                tensorboard_log=str(output_path / "tensorboard"),
                verbose=1,
                device='auto',
            )
        else:
            # Use standard PPO with MLP
            policy_type = "MlpPolicy"
            model = PPO(
                policy_type,
                train_env,
                learning_rate=learning_rate,
                n_steps=2048,  # Longer rollouts for complex tasks
                batch_size=batch_size,
                n_epochs=n_epochs,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                clip_range_vf=None,
                ent_coef=0.01,  # Encourage exploration
                vf_coef=0.5,
                max_grad_norm=0.5,
                use_sde=False,
                sde_sample_freq=-1,
                target_kl=0.01,
                tensorboard_log=str(output_path / "tensorboard"),
                verbose=1,
                device='auto',
            )

        print(f"✅ Model created with {policy_type}")
        print(f"  - Learning rate: {learning_rate}")
        print(f"  - Batch size: {batch_size}")
        print(f"  - LSTM: {use_lstm}")
    print()

    # Training info
    print("=" * 80)
    print("TRAINING CONFIGURATION")
    print("=" * 80)
    print(f"Total timesteps:      {total_timesteps:,}")
    print(f"Parallel envs:        {n_envs}")
    print(f"Save frequency:       every {save_freq:,} steps")
    print(f"Eval frequency:       every {eval_freq:,} steps")
    print(f"Output directory:     {output_path}")
    print("=" * 80)
    print()

    # Train
    print("🚀 Starting training...")
    print()
    start_time = time.time()

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print()
        print("⚠️  Training interrupted by user")

    training_time = (time.time() - start_time) / 60
    print()
    print(f"✅ Training completed in {training_time:.1f} minutes")
    print()

    # Save final model
    final_model_path = output_path / "final_model"
    model.save(str(final_model_path))
    train_env.save(str(output_path / "vecnormalize.pkl"))

    print(f"✅ Final model saved to: {final_model_path}")
    print(f"✅ Normalization stats saved to: {output_path / 'vecnormalize.pkl'}")
    print()

    print("=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print(f"Configuration:        {config_path}")
    print(f"Total timesteps:      {total_timesteps:,}")
    print(f"Training time:        {training_time:.1f} minutes")
    print(f"Output directory:     {output_path}")
    print(f"Final model:          {final_model_path}")
    print("=" * 80)
    print()

    # Close environments
    train_env.close()
    eval_env.close()

    return model


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(
        description="Train PPO on multi-agent flocking and foraging environment"
    )

    # Required arguments
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to environment config YAML file (e.g., configs/env_hard_mode.yaml)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for models and logs (e.g., models/ppo_hard_mode)"
    )

    # Training hyperparameters
    parser.add_argument(
        "--timesteps",
        type=int,
        default=5_000_000,
        help="Total training timesteps (default: 5M)"
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=4,
        help="Number of parallel environments (default: 4)"
    )
    parser.add_argument(
        "--save-freq",
        type=int,
        default=100_000,
        help="Save checkpoint every N timesteps (default: 100k)"
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=50_000,
        help="Evaluate every N timesteps (default: 50k)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Learning rate (default: 3e-4)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size (default: 256)"
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=10,
        help="Number of epochs per update (default: 10)"
    )
    parser.add_argument(
        "--no-lstm",
        action="store_true",
        help="Disable LSTM policy (use MLP instead)"
    )

    # Resume training
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )

    args = parser.parse_args()

    # Validate config file exists
    if not Path(args.config).exists():
        print(f"❌ Config file not found: {args.config}")
        print()
        print("Available configs:")
        configs_dir = Path("configs")
        if configs_dir.exists():
            for config_file in configs_dir.glob("*.yaml"):
                print(f"  - {config_file}")
        return

    # Train
    train(
        config_path=args.config,
        output_dir=args.output,
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        save_freq=args.save_freq,
        eval_freq=args.eval_freq,
        resume_from=args.resume,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        use_lstm=not args.no_lstm,
    )


if __name__ == "__main__":
    main()
