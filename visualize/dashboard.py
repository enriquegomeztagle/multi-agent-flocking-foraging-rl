"""
Real-time visualization dashboard for flocking and foraging simulation.

Displays:
- Agent positions and velocities in 2D world
- Resource patches
- Metrics: cohesion, alignment, Gini index, resource usage
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import yaml
import os
from env.flockforage_parallel import FlockForageParallel, EnvConfig
from train.baseline_boids import BoidsController
from metrics.fairness import gini
from metrics.flocking import polarization, mean_knn_distance, separation_violations
from metrics.sustainability import stock_score, min_stock_normalized


class Dashboard:
    """Interactive dashboard for visualizing agent behavior and metrics."""

    def __init__(self, env, controller=None, model=None, max_steps=500):
        """
        Initialize dashboard.

        Args:
            env: FlockForageParallel environment
            controller: Optional BoidsController for baseline
            model: Optional trained RL model
            max_steps: Maximum simulation steps
        """
        self.env = env
        self.controller = controller
        self.model = model
        self.max_steps = max_steps
        self.step_count = 0

        # Metrics history
        self.history = {
            "polarization": [],
            "gini": [],
            "mean_stock": [],
            "min_stock": [],
            "mean_reward": [],
            "knn_distance": [],
        }

        # Setup figure with subplots
        self.fig = plt.figure(figsize=(16, 9))
        gs = self.fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Main world view (larger)
        self.ax_world = self.fig.add_subplot(gs[:, 0:2])
        self.ax_world.set_xlim(0, env.cfg.width)
        self.ax_world.set_ylim(0, env.cfg.height)
        self.ax_world.set_aspect('equal')
        self.ax_world.set_title('Agent Positions & Resource Patches', fontsize=14, fontweight='bold')
        self.ax_world.set_xlabel('X Position')
        self.ax_world.set_ylabel('Y Position')
        self.ax_world.grid(True, alpha=0.3)

        # Metrics subplots
        self.ax_pol = self.fig.add_subplot(gs[0, 2])
        self.ax_gini = self.fig.add_subplot(gs[1, 2])
        self.ax_stock = self.fig.add_subplot(gs[2, 2])

        # Configure metrics plots
        for ax, title, ylabel in [
            (self.ax_pol, 'Polarization (Alignment)', 'Polarization'),
            (self.ax_gini, 'Gini Coefficient (Fairness)', 'Gini'),
            (self.ax_stock, 'Resource Stock', 'Stock Level'),
        ]:
            ax.set_title(title, fontsize=10)
            ax.set_xlabel('Step', fontsize=8)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, max_steps)

        self.ax_pol.set_ylim(0, 1)
        self.ax_gini.set_ylim(0, 1)
        self.ax_stock.set_ylim(0, 1)

        # Initialize plot elements
        self.agent_scatter = None
        self.agent_arrows = None
        self.patch_circles = []
        self.lines = {
            "polarization": self.ax_pol.plot([], [], 'b-', linewidth=2)[0],
            "gini": self.ax_gini.plot([], [], 'r-', linewidth=2)[0],
            "mean_stock": self.ax_stock.plot([], [], 'g-', label='Mean', linewidth=2)[0],
            "min_stock": self.ax_stock.plot([], [], 'orange', label='Min', linewidth=2)[0],
        }
        self.ax_stock.legend(loc='upper right', fontsize=8)

        # Reset environment
        self.obs, _ = self.env.reset()
        self.lstm_states = None
        self.episode_starts = np.ones((len(self.env.agents),), dtype=bool)

    def update(self, frame):
        """Update animation frame."""
        if self.step_count >= self.max_steps:
            return

        # Get actions
        if self.controller is not None:
            # Use Boids controller
            actions = {}
            for agent_id in self.env.agents:
                agent_obs = self.obs[agent_id]
                agent_idx = int(agent_id.split("_")[1])
                agent_heading = self.env._heading[agent_idx]
                actions[agent_id] = self.controller.compute_action(agent_obs, agent_heading)

        elif self.model is not None:
            # Use RL model
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
            # Random actions (fallback)
            actions = {agent: self.env.action_space(agent).sample() for agent in self.env.agents}

        # Step environment
        self.obs, rewards, terminations, truncations, _ = self.env.step(actions)

        # Check if done
        if any(terminations.values()) or any(truncations.values()):
            self.step_count = self.max_steps
            return

        # Update world visualization
        self._update_world_view()

        # Compute and update metrics
        self._update_metrics()

        self.step_count += 1

    def _update_world_view(self):
        """Update the main world view with agents and patches."""
        # Clear previous patches
        for circle in self.patch_circles:
            circle.remove()
        self.patch_circles = []

        # Draw patches
        for i, (pos, stock) in enumerate(zip(self.env._patches.centers, self.env._patches.stock)):
            # Color based on stock level
            stock_ratio = stock / self.env.cfg.S_max
            color = plt.cm.Greens(0.3 + 0.7 * stock_ratio)
            radius = self.env.cfg.feed_radius

            circle = Circle(pos, radius, color=color, alpha=0.5, zorder=1)
            self.ax_world.add_patch(circle)
            self.patch_circles.append(circle)

            # Add stock text
            text = self.ax_world.text(
                pos[0], pos[1], f'{stock:.2f}',
                ha='center', va='center', fontsize=7, color='darkgreen', zorder=2
            )
            self.patch_circles.append(text)

        # Draw agents
        if self.agent_scatter is not None:
            self.agent_scatter.remove()
        if self.agent_arrows is not None:
            self.agent_arrows.remove()

        positions = self.env._pos
        velocities = self.env._vel

        # Color agents by intake
        intake_norm = self.env._intake_total / (np.max(self.env._intake_total) + 1e-6)
        colors = plt.cm.plasma(intake_norm)

        self.agent_scatter = self.ax_world.scatter(
            positions[:, 0], positions[:, 1],
            c=colors, s=100, edgecolors='black', linewidths=1, zorder=3
        )

        # Draw velocity arrows
        self.agent_arrows = self.ax_world.quiver(
            positions[:, 0], positions[:, 1],
            velocities[:, 0], velocities[:, 1],
            color='black', alpha=0.7, scale=10, width=0.003, zorder=4
        )

        # Update title with step info
        controller_type = "Baseline (Boids)" if self.controller else "RL (RecurrentPPO)"
        self.ax_world.set_title(
            f'Agent Positions & Resource Patches - Step {self.step_count} ({controller_type})',
            fontsize=14, fontweight='bold'
        )

    def _update_metrics(self):
        """Compute and update metric plots."""
        # Compute metrics
        pol = polarization(self.env._vel)
        gini_val = gini(self.env._intake_total + 1e-8)
        mean_stock = stock_score(self.env._patches.stock, self.env.cfg.S_max)
        min_stock = min_stock_normalized(self.env._patches.stock, self.env.cfg.S_max)

        # Append to history
        self.history["polarization"].append(pol)
        self.history["gini"].append(gini_val)
        self.history["mean_stock"].append(mean_stock)
        self.history["min_stock"].append(min_stock)

        # Update line plots
        steps = np.arange(len(self.history["polarization"]))
        self.lines["polarization"].set_data(steps, self.history["polarization"])
        self.lines["gini"].set_data(steps, self.history["gini"])
        self.lines["mean_stock"].set_data(steps, self.history["mean_stock"])
        self.lines["min_stock"].set_data(steps, self.history["min_stock"])

    def run(self, save_path=None):
        """
        Run the dashboard animation.

        Args:
            save_path: Optional path to save animation (e.g., 'output.mp4')
        """
        anim = FuncAnimation(
            self.fig, self.update,
            frames=self.max_steps,
            interval=50,  # 50ms between frames
            repeat=False,
            blit=False
        )

        if save_path:
            print(f"Saving animation to {save_path}...")
            anim.save(save_path, writer='ffmpeg', fps=20, dpi=100)
            print(f"✅ Animation saved!")
        else:
            plt.show()

        return anim


def main():
    """Run dashboard with specified controller."""
    import argparse

    parser = argparse.ArgumentParser(description='Run visualization dashboard')
    parser.add_argument(
        '--mode', type=str, default='baseline',
        choices=['baseline', 'rl'],
        help='Controller mode: baseline (Boids) or rl (trained agent)'
    )
    parser.add_argument(
        '--config', type=str, default='configs/env_curriculum_phase2.yaml',
        help='Environment config file'
    )
    parser.add_argument(
        '--model', type=str, default='models/advanced_final/recurrent_ppo_final.zip',
        help='Path to trained RL model (for rl mode)'
    )
    parser.add_argument(
        '--steps', type=int, default=500,
        help='Number of simulation steps'
    )
    parser.add_argument(
        '--save', type=str, default=None,
        help='Optional path to save video (e.g., output.mp4)'
    )

    args = parser.parse_args()

    # Load config
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            env_cfg = yaml.safe_load(f)
        print(f"✅ Loaded config: {args.config}")
    else:
        print(f"⚠️  Config not found, using defaults")
        env_cfg = {}

    # Create environment
    env = FlockForageParallel(EnvConfig(**env_cfg))

    # Create controller
    controller = None
    model = None

    if args.mode == 'baseline':
        print("🔵 Using Baseline Boids Controller")
        controller = BoidsController(
            w_cohesion=1.0,
            w_alignment=1.0,
            w_separation=2.0,
            w_foraging=1.5
        )
    elif args.mode == 'rl':
        if not os.path.exists(args.model):
            print(f"❌ Model not found: {args.model}")
            print("   Train first using: python -m train.run_advanced_training")
            return

        print(f"✅ Loading RL model: {args.model}")
        from sb3_contrib import RecurrentPPO
        model = RecurrentPPO.load(args.model)

    # Create and run dashboard
    print(f"\n🚀 Starting dashboard (mode: {args.mode}, steps: {args.steps})")
    dashboard = Dashboard(env, controller=controller, model=model, max_steps=args.steps)
    dashboard.run(save_path=args.save)

    env.close()


if __name__ == "__main__":
    main()
