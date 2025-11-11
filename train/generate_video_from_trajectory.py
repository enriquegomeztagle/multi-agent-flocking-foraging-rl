"""
Generar video desde trayectoria guardada (reproducible)
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from pathlib import Path


def generate_video_from_trajectory(trajectory_file: str, output_file: str = None):
    """Generate video from saved trajectory."""
    print("=" * 80)
    print("GENERATING VIDEO FROM SAVED TRAJECTORY")
    print("=" * 80)
    print()

    # Load trajectory
    print(f"Loading: {trajectory_file}")
    with open(trajectory_file, 'rb') as f:
        trajectory = pickle.load(f)

    positions = [np.array(p) for p in trajectory["positions"]]
    patches = trajectory["patches"]
    intake_history = trajectory["intake_history"]
    episode_data = trajectory["episode_data"]

    print(f"✅ Loaded trajectory")
    print(f"   Episode: {episode_data['episode']}")
    print(f"   Seed: {episode_data['seed']}")
    print(f"   Intake: {episode_data['intake']:.2f} ({episode_data['efficiency_percent']:.1f}%)")
    print(f"   Frames: {len(positions)}")
    print()

    # Default output file
    if output_file is None:
        output_file = f"results/videos/episode_{episode_data['episode']}_intake_{episode_data['intake']:.0f}.mp4"

    # Generate video
    print("Generating video...")

    fig, ax = plt.subplots(figsize=(14, 12))

    def animate(frame):
        ax.clear()
        ax.set_xlim(0, 30)
        ax.set_ylim(0, 30)
        ax.set_aspect('equal')

        current_intake = intake_history[frame] if frame < len(intake_history) else episode_data['intake']
        efficiency = (current_intake / 2307) * 100

        ax.set_title(
            f"RL Multi-Agent System (PPO + Rewards v3 + Curriculum)\n"
            f"Episode {episode_data['episode']} (Seed {episode_data['seed']}) | "
            f"Step {frame}/{len(positions)} | Intake: {current_intake:.1f} ({efficiency:.1f}%)",
            fontsize=18, fontweight='bold', pad=25
        )

        ax.set_xlabel("X Position", fontsize=14)
        ax.set_ylabel("Y Position", fontsize=14)

        # Resource patches
        if frame < len(patches):
            for i, center in enumerate(patches[frame]["centers"]):
                stock = patches[frame]["stock"][i]
                if stock > 0.05:
                    color = plt.cm.Greens(min(stock, 1.0))
                    circle = Circle(center, radius=2.5, color=color, alpha=0.75, zorder=1)
                    ax.add_patch(circle)

                    if stock > 0.2:
                        ax.text(center[0], center[1], f'{stock:.2f}',
                               ha='center', va='center', fontsize=9,
                               color='darkgreen', fontweight='bold', zorder=2)

        # Agents
        if frame < len(positions):
            pos = positions[frame]
            ax.scatter(pos[:, 0], pos[:, 1],
                      c='#FF4444', s=250, marker='o',
                      edgecolors='darkred', linewidths=3.5,
                      label='RL Agents (10)', zorder=10, alpha=0.95)

        ax.legend(loc='upper right', fontsize=13, framealpha=0.95)
        ax.grid(True, alpha=0.25, linestyle='--')

        # Info panel
        info_text = (
            f"BEST EPISODE RECORDED\n"
            f"\n"
            f"Model: PPO Simple\n"
            f"Training: 4M steps (19 min)\n"
            f"Curriculum: 5→10 agents\n"
            f"\n"
            f"Final Intake: {episode_data['intake']:.2f}\n"
            f"Efficiency: {episode_data['efficiency_percent']:.1f}%\n"
            f"Baseline: 2,307 (100%)\n"
            f"Fairness (Gini): {episode_data['gini']:.3f}\n"
            f"\n"
            f"Episode: {episode_data['episode']}\n"
            f"Seed: {episode_data['seed']}"
        )
        ax.text(0.02, 0.98, info_text,
               transform=ax.transAxes,
               fontsize=11,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95, pad=0.8),
               family='monospace')

    print("  Rendering... (2-3 min)")

    anim = animation.FuncAnimation(
        fig, animate,
        frames=len(positions),
        interval=50,
        repeat=False
    )

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    anim.save(output_file, writer='ffmpeg', fps=20, dpi=120, bitrate=2500)
    plt.close()

    file_size = Path(output_file).stat().st_size / (1024 * 1024)
    print(f"✅ Video saved: {output_file} ({file_size:.2f} MB)")
    print()

    return output_file


if __name__ == "__main__":
    # Generate video from BEST episode (625 intake!)
    trajectory_file = "results/trajectories/episode_32_seed_1302_intake_625.pkl"

    print("=" * 80)
    print("VIDEO GENERATION - BEST EPISODE (625 INTAKE, 27.1%)")
    print("=" * 80)
    print()

    if not Path(trajectory_file).exists():
        print(f"❌ Trajectory file not found: {trajectory_file}")
        print("Run eval_and_save_trajectories.py first!")
    else:
        video_file = generate_video_from_trajectory(trajectory_file)

        print("=" * 80)
        print("✅ COMPLETED - VIDEO IS REPRODUCIBLE")
        print("=" * 80)
        print()
        print(f"Video file: {video_file}")
        print("This video shows the ACTUAL best episode (625 intake, 27.1%)")
        print("The trajectory was captured during evaluation and is fully reproducible.")
        print("=" * 80)
