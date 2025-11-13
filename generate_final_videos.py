#!/usr/bin/env python3
"""Generate final MP4 videos with proper VecNormalize for all difficulty levels."""

import numpy as np
import imageio
import yaml
import pickle
from pathlib import Path
from env.flockforage_parallel import FlockForageParallel, EnvConfig
from stable_baselines3 import PPO

def generate_video(name, model_dir, config_path, output_path):
    """Generate video with proper VecNormalize."""
    print(f"🎬 Generating {name} video...")
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load model and VecNormalize
    model_path = Path(model_dir) / "final_model.zip"
    vecnorm_path = Path(model_dir) / "vecnormalize.pkl"
    
    model = PPO.load(model_path)
    
    if vecnorm_path.exists():
        with open(vecnorm_path, 'rb') as f:
            vec_normalize = pickle.load(f)
    else:
        vec_normalize = None
    
    # Create environment
    env = FlockForageParallel(EnvConfig(**config))
    obs, _ = env.reset(seed=42)
    
    frames = []
    
    for step in range(200):
        # Prepare observations
        obs_array = np.array([obs[agent] for agent in env.agents])
        obs_flat = obs_array.flatten()
        
        # Normalize observations if VecNormalize exists
        if vec_normalize is not None:
            obs_flat = vec_normalize.normalize_obs(obs_flat.reshape(1, -1)).flatten()
        
        # Get action from model (stochastic for variety)
        action, _ = model.predict(obs_flat, deterministic=False)
        action_dict = {agent: int(action[i]) for i, agent in enumerate(env.agents)}
        
        # Render and step
        frame = env.render(mode="rgb_array")
        frames.append(frame)
        
        obs, rew, terms, truncs, _ = env.step(action_dict)
        
        if all(terms.values()) or all(truncs.values()):
            break
    
    # Save video
    imageio.mimsave(output_path, frames, fps=20)
    print(f"✅ {name} completed: {output_path}")
    
    env.close()

def main():
    """Generate final videos for all difficulty levels."""
    videos = [
        ("Easy", "models/ppo_easy", "configs/env_easy.yaml", "videos/easy_final.mp4"),
        ("Medium", "models/ppo_medium", "configs/env_medium.yaml", "videos/medium_final.mp4"),
        ("Hard", "models/ppo_hard", "configs/env_hard.yaml", "videos/hard_final.mp4"),
        ("Expert", "models/ppo_expert", "configs/env_expert.yaml", "videos/expert_final.mp4")
    ]
    
    for name, model_dir, config, output in videos:
        generate_video(name, model_dir, config, output)
    
    print("\n🎉 All final videos generated!")

if __name__ == "__main__":
    main()
