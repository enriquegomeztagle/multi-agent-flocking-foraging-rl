"""
Generate videos of agent behavior from trained models or baseline.

This script can generate videos from:
1. Trained RL models
2. Baseline Boids agent
3. Random agent (for comparison)
"""

import argparse
import numpy as np
import imageio
from pathlib import Path
import json
import yaml

from env.flockforage_parallel import FlockForageParallel, EnvConfig
from env.boids_agent import ClassicalBoidsAgent
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize


def generate_video_from_rl(
    model_path: str,
    config: dict,
    output_path: str,
    n_episodes: int = 1,
    episode_length: int = None,
    fps: int = 10,
    seed: int = None
):
    """
    Generate video from trained RL model.
    
    Args:
        model_path: Path to trained PPO model
        config: Environment configuration dictionary
        output_path: Output video path
        n_episodes: Number of episodes to record
        episode_length: Maximum steps per episode (None = use config default)
        fps: Frames per second for video
        seed: Random seed (None = random)
    """
    print(f"Loading RL model from: {model_path}")
    model = PPO.load(model_path)
    print("✅ Model loaded")
    
    env = FlockForageParallel(EnvConfig(**config))
    
    episode_length = episode_length or config['episode_len']
    
    all_frames = []
    
    for ep in range(n_episodes):
        print(f"Recording episode {ep + 1}/{n_episodes}...")
        episode_seed = seed if seed is not None else ep * 42
        obs, _ = env.reset(seed=episode_seed)
        
        episode_frames = []
        
        for step in range(episode_length):
            # Get action from model
            obs_array = np.array([obs[agent] for agent in env.agents])
            obs_flat = obs_array.flatten()
            action, _ = model.predict(obs_flat, deterministic=True)
            action_dict = {agent: int(action[i]) for i, agent in enumerate(env.agents)}
            
            # Render frame
            frame = env.render(mode="rgb_array")
            episode_frames.append(frame)
            
            # Step environment
            obs, rew, terms, truncs, _ = env.step(action_dict)
            
            if all(terms.values()) or all(truncs.values()):
                break
        
        all_frames.extend(episode_frames)
        print(f"  Recorded {len(episode_frames)} frames")
    
    # Save video
    print(f"\nSaving video to: {output_path}")
    imageio.mimsave(output_path, all_frames, fps=fps)
    print(f"✅ Video saved: {len(all_frames)} frames at {fps} fps")
    
    env.close()


def generate_video_from_baseline(
    config: dict,
    output_path: str,
    n_episodes: int = 1,
    episode_length: int = None,
    fps: int = 10,
    seed: int = None
):
    """
    Generate video from baseline Boids agent.
    
    Args:
        config: Environment configuration dictionary
        output_path: Output video path
        n_episodes: Number of episodes to record
        episode_length: Maximum steps per episode (None = use config default)
        fps: Frames per second for video
        seed: Random seed (None = random)
    """
    print("Using Baseline Boids agent")
    
    env = FlockForageParallel(EnvConfig(**config))
    boids_agent = ClassicalBoidsAgent()
    
    episode_length = episode_length or config['episode_len']
    
    all_frames = []
    
    for ep in range(n_episodes):
        print(f"Recording episode {ep + 1}/{n_episodes}...")
        episode_seed = seed if seed is not None else ep * 42
        obs, _ = env.reset(seed=episode_seed)
        
        episode_frames = []
        
        for step in range(episode_length):
            # Get action from Boids agent
            actions = {}
            for i, agent in enumerate(env.agents):
                action = boids_agent.compute_action(i, env)
                actions[agent] = action
            
            # Render frame
            frame = env.render(mode="rgb_array")
            episode_frames.append(frame)
            
            # Step environment
            obs, rew, terms, truncs, _ = env.step(actions)
            
            if all(terms.values()) or all(truncs.values()):
                break
        
        all_frames.extend(episode_frames)
        print(f"  Recorded {len(episode_frames)} frames")
    
    # Save video
    print(f"\nSaving video to: {output_path}")
    imageio.mimsave(output_path, all_frames, fps=fps)
    print(f"✅ Video saved: {len(all_frames)} frames at {fps} fps")
    
    env.close()


def main():
    """Main script."""
    parser = argparse.ArgumentParser(
        description="Generate videos of agent behavior"
    )
    
    parser.add_argument(
        "--type",
        type=str,
        choices=["rl", "baseline"],
        required=True,
        help="Type of agent: 'rl' for trained model, 'baseline' for Boids"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to environment config YAML"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output video path (e.g., videos/easy_mode_rl.mp4)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to RL model (required if type=rl)"
    )
    
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of episodes to record (default: 1)"
    )
    
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Maximum steps per episode (default: use config)"
    )
    
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Frames per second for video (default: 10)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (default: random)"
    )
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate video
    if args.type == "rl":
        if args.model is None:
            print("❌ Error: --model is required when type=rl")
            return
        generate_video_from_rl(
            model_path=args.model,
            config=config,
            output_path=str(output_path),
            n_episodes=args.episodes,
            episode_length=args.steps,
            fps=args.fps,
            seed=args.seed
        )
    elif args.type == "baseline":
        generate_video_from_baseline(
            config=config,
            output_path=str(output_path),
            n_episodes=args.episodes,
            episode_length=args.steps,
            fps=args.fps,
            seed=args.seed
        )


if __name__ == "__main__":
    main()

