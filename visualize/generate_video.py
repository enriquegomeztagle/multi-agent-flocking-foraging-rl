"""
Generate demo videos of agent behavior for presentations.

Creates high-quality MP4 videos showing:
- Agent movements and interactions
- Resource patch consumption
- Emergent flocking behaviors
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import Circle
import yaml
import os
from pathlib import Path
from env.flockforage_parallel import FlockForageParallel, EnvConfig
from train.baseline_boids import BoidsController


class VideoGenerator:
    """Generate videos of simulation runs."""

    def __init__(self, env, controller=None, model=None):
        """
        Initialize video generator.

        Args:
            env: FlockForageParallel environment
            controller: Optional BoidsController for baseline
            model: Optional trained RL model
        """
        self.env = env
        self.controller = controller
        self.model = model

        # Tracking
        self.frames = []
        self.obs = None
        self.lstm_states = None
        self.episode_starts = None

    def generate_episode(self, max_steps=500, seed=None):
        """
        Generate a full episode and record frames.

        Args:
            max_steps: Maximum steps to simulate
            seed: Random seed for reproducibility

        Returns:
            List of frame data dictionaries
        """
        self.obs, _ = self.env.reset(seed=seed)
        self.frames = []

        if self.model is not None:
            self.lstm_states = None
            self.episode_starts = np.ones((len(self.env.agents),), dtype=bool)

        for step in range(max_steps):
            # Get actions
            if self.controller is not None:
                actions = {}
                for agent_id in self.env.agents:
                    agent_obs = self.obs[agent_id]
                    agent_idx = int(agent_id.split("_")[1])
                    agent_heading = self.env._heading[agent_idx]
                    actions[agent_id] = self.controller.compute_action(agent_obs, agent_heading)

            elif self.model is not None:
                obs_array = np.array([self.obs[agent] for agent in self.env.agents])
                action_array, self.lstm_states = self.model.predict(
                    obs_array,
                    state=self.lstm_states,
                    episode_start=self.episode_starts,
                    deterministic=True
                )
                self.episode_starts = np.zeros((len(self.env.agents),), dtype=bool)
                actions = {agent: int(action_array[i]) for i, agent in enumerate(self.env.agents)}

            else:
                actions = {agent: self.env.action_space(agent).sample() for agent in self.env.agents}

            # Step environment
            self.obs, rewards, terminations, truncations, _ = self.env.step(actions)

            # Record frame data
            frame_data = {
                "step": step,
                "positions": self.env._pos.copy(),
                "velocities": self.env._vel.copy(),
                "intake": self.env._intake_total.copy(),
                "patch_positions": self.env._patches.centers.copy(),
                "patch_stocks": self.env._patches.stock.copy(),
                "total_reward": sum(rewards.values()),
            }
            self.frames.append(frame_data)

            # Check if done
            if any(terminations.values()) or any(truncations.values()):
                break

        return self.frames

    def render_video(self, output_path, fps=20, dpi=150, title=None):
        """
        Render recorded frames to video file.

        Args:
            output_path: Path to save video (e.g., 'demo.mp4')
            fps: Frames per second
            dpi: Resolution (higher = better quality, larger file)
            title: Optional title for video
        """
        if not self.frames:
            print("❌ No frames to render. Run generate_episode() first.")
            return

        print(f"🎬 Rendering video: {output_path}")
        print(f"   Frames: {len(self.frames)}, FPS: {fps}, DPI: {dpi}")

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.set_xlim(0, self.env.cfg.width)
        ax.set_ylim(0, self.env.cfg.height)
        ax.set_aspect('equal')
        ax.set_xlabel('X Position', fontsize=12)
        ax.set_ylabel('Y Position', fontsize=12)
        ax.grid(True, alpha=0.3)

        # Title
        if title is None:
            title = "Baseline (Boids)" if self.controller else "RL (RecurrentPPO)"
        fig_title = ax.set_title(
            f'{title} - Step 0',
            fontsize=14, fontweight='bold'
        )

        # Initialize plot elements
        patch_circles = []
        agent_scatter = None
        agent_arrows = None

        def update_frame(frame_idx):
            """Update function for animation."""
            nonlocal agent_scatter, agent_arrows

            frame = self.frames[frame_idx]

            # Clear previous patches
            for item in patch_circles:
                item.remove()
            patch_circles.clear()

            # Draw patches
            for pos, stock in zip(frame["patch_positions"], frame["patch_stocks"]):
                stock_ratio = stock / self.env.cfg.S_max
                color = plt.cm.Greens(0.3 + 0.7 * stock_ratio)
                radius = self.env.cfg.feed_radius

                circle = Circle(pos, radius, color=color, alpha=0.5, zorder=1)
                ax.add_patch(circle)
                patch_circles.append(circle)

                text = ax.text(
                    pos[0], pos[1], f'{stock:.2f}',
                    ha='center', va='center', fontsize=8,
                    color='darkgreen', zorder=2
                )
                patch_circles.append(text)

            # Draw agents
            if agent_scatter is not None:
                agent_scatter.remove()
            if agent_arrows is not None:
                agent_arrows.remove()

            positions = frame["positions"]
            velocities = frame["velocities"]
            intake = frame["intake"]

            # Color by intake
            intake_norm = intake / (np.max(intake) + 1e-6)
            colors = plt.cm.plasma(intake_norm)

            agent_scatter = ax.scatter(
                positions[:, 0], positions[:, 1],
                c=colors, s=150, edgecolors='black',
                linewidths=1.5, zorder=3
            )

            agent_arrows = ax.quiver(
                positions[:, 0], positions[:, 1],
                velocities[:, 0], velocities[:, 1],
                color='black', alpha=0.7, scale=10,
                width=0.004, zorder=4
            )

            # Update title
            fig_title.set_text(
                f'{title} - Step {frame["step"]} | '
                f'Total Reward: {frame["total_reward"]:.2f} | '
                f'Mean Intake: {np.mean(intake):.2f}'
            )

        # Create animation
        anim = FuncAnimation(
            fig, update_frame,
            frames=len(self.frames),
            interval=1000 // fps,
            repeat=False,
            blit=False
        )

        # Save video
        writer = FFMpegWriter(fps=fps, bitrate=2000)
        anim.save(output_path, writer=writer, dpi=dpi)

        plt.close(fig)
        print(f"✅ Video saved: {output_path}")


def main():
    """Generate demo videos for both baseline and RL agents."""
    import argparse

    parser = argparse.ArgumentParser(description='Generate demo videos')
    parser.add_argument(
        '--mode', type=str, default='both',
        choices=['baseline', 'rl', 'both'],
        help='Which controller to generate video for'
    )
    parser.add_argument(
        '--config', type=str, default='configs/env_curriculum_phase2.yaml',
        help='Environment config file'
    )
    parser.add_argument(
        '--model', type=str, default='models/advanced_final/recurrent_ppo_final.zip',
        help='Path to trained RL model'
    )
    parser.add_argument(
        '--steps', type=int, default=500,
        help='Number of simulation steps'
    )
    parser.add_argument(
        '--output-dir', type=str, default='results/videos',
        help='Output directory for videos'
    )
    parser.add_argument(
        '--fps', type=int, default=20,
        help='Frames per second'
    )
    parser.add_argument(
        '--dpi', type=int, default=150,
        help='Video resolution (DPI)'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility'
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load config
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            env_cfg = yaml.safe_load(f)
        print(f"✅ Loaded config: {args.config}")
    else:
        print(f"⚠️  Config not found, using defaults")
        env_cfg = {}

    # Generate baseline video
    if args.mode in ['baseline', 'both']:
        print("\n" + "=" * 60)
        print("GENERATING BASELINE (BOIDS) VIDEO")
        print("=" * 60)

        env = FlockForageParallel(EnvConfig(**env_cfg))
        controller = BoidsController(
            w_cohesion=1.0,
            w_alignment=1.0,
            w_separation=2.0,
            w_foraging=1.5
        )

        gen = VideoGenerator(env, controller=controller)
        gen.generate_episode(max_steps=args.steps, seed=args.seed)

        output_path = os.path.join(args.output_dir, 'baseline_boids_demo.mp4')
        gen.render_video(
            output_path,
            fps=args.fps,
            dpi=args.dpi,
            title="Baseline Boids Controller"
        )

        env.close()

    # Generate RL video
    if args.mode in ['rl', 'both']:
        print("\n" + "=" * 60)
        print("GENERATING RL (RecurrentPPO) VIDEO")
        print("=" * 60)

        if not os.path.exists(args.model):
            print(f"❌ Model not found: {args.model}")
            print("   Train first using: python -m train.run_advanced_training")
        else:
            from sb3_contrib import RecurrentPPO

            env = FlockForageParallel(EnvConfig(**env_cfg))
            model = RecurrentPPO.load(args.model)

            gen = VideoGenerator(env, model=model)
            gen.generate_episode(max_steps=args.steps, seed=args.seed)

            output_path = os.path.join(args.output_dir, 'rl_recurrentppo_demo.mp4')
            gen.render_video(
                output_path,
                fps=args.fps,
                dpi=args.dpi,
                title="RL RecurrentPPO Agent"
            )

            env.close()

    print("\n" + "=" * 60)
    print("✅ ALL VIDEOS GENERATED")
    print(f"   Output directory: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
