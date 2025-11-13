"""
Rendering utilities for the flocking and foraging environment.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
from typing import Optional, Tuple
import io


def render_frame(
    pos: np.ndarray,
    vel: np.ndarray,
    patches_centers: np.ndarray,
    patches_stock: np.ndarray,
    patches_max_stock: float,
    width: float,
    height: float,
    feed_radius: float,
    ax: Optional[plt.Axes] = None,
    show_velocities: bool = True,
    show_patch_levels: bool = True,
    show_trajectories: bool = False,
    trajectories: Optional[list] = None
) -> plt.Axes:
    """
    Render a single frame of the environment.
    
    Args:
        pos: (N, 2) agent positions
        vel: (N, 2) agent velocities
        patches_centers: (M, 2) patch center positions
        patches_stock: (M,) patch stock levels
        patches_max_stock: Maximum patch stock capacity
        width: World width
        height: World height
        feed_radius: Feeding radius for agents
        ax: Matplotlib axes (if None, creates new figure)
        show_velocities: Whether to show velocity vectors
        show_patch_levels: Whether to show patch stock levels
        show_trajectories: Whether to show agent trajectories
        trajectories: List of position histories for each agent
        
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    else:
        ax.clear()
    
    # Set limits
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect('equal')
    ax.set_facecolor('#1a1a2e')
    
    # Draw patches
    for i, (center, stock) in enumerate(zip(patches_centers, patches_stock)):
        # Patch circle with color based on stock level
        stock_normalized = stock / patches_max_stock
        color_intensity = max(0.3, stock_normalized)  # Minimum visibility
        
        # Color: green when full, red when empty
        color = plt.cm.RdYlGn(stock_normalized)
        
        # Draw patch circle
        circle = plt.Circle(
            center,
            feed_radius,
            color=color,
            alpha=0.4,
            edgecolor='white',
            linewidth=1.5
        )
        ax.add_patch(circle)
        
        # Draw patch center
        ax.plot(center[0], center[1], 'o', color='white', markersize=4, alpha=0.8)
        
        # Show stock level as text
        if show_patch_levels:
            ax.text(
                center[0], center[1] + feed_radius + 0.5,
                f'{stock:.2f}',
                color='white',
                fontsize=8,
                ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.5)
            )
    
    # Draw trajectories if requested
    if show_trajectories and trajectories is not None:
        for traj in trajectories:
            if len(traj) > 1:
                traj_array = np.array(traj)
                ax.plot(traj_array[:, 0], traj_array[:, 1], '--', color='white', alpha=0.2, linewidth=0.5)
    
    # Draw agents
    for i, (p, v) in enumerate(zip(pos, vel)):
        # Agent position as circle
        agent_circle = plt.Circle(
            p,
            0.3,
            color='cyan',
            alpha=0.9,
            edgecolor='white',
            linewidth=1.5
        )
        ax.add_patch(agent_circle)
        
        # Agent direction indicator (arrow)
        if show_velocities and np.linalg.norm(v) > 0.01:
            v_normalized = v / (np.linalg.norm(v) + 1e-6)
            arrow_length = 0.5
            ax.arrow(
                p[0], p[1],
                v_normalized[0] * arrow_length,
                v_normalized[1] * arrow_length,
                head_width=0.2,
                head_length=0.15,
                fc='yellow',
                ec='yellow',
                alpha=0.8
            )
    
    # Title
    ax.set_title('Multi-Agent Flocking and Foraging', color='white', fontsize=14, pad=10)
    
    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor='green', alpha=0.4, label='Patch (high stock)'),
        mpatches.Patch(facecolor='red', alpha=0.4, label='Patch (low stock)'),
        mpatches.Patch(facecolor='cyan', alpha=0.9, label='Agent'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', facecolor='black', edgecolor='white', labelcolor='white')
    
    return ax


def render_to_rgb_array(
    pos: np.ndarray,
    vel: np.ndarray,
    patches_centers: np.ndarray,
    patches_stock: np.ndarray,
    patches_max_stock: float,
    width: float,
    height: float,
    feed_radius: float,
    dpi: int = 100
) -> np.ndarray:
    """
    Render frame to RGB array for video generation.
    
    Args:
        pos: (N, 2) agent positions
        vel: (N, 2) agent velocities
        patches_centers: (M, 2) patch center positions
        patches_stock: (M,) patch stock levels
        patches_max_stock: Maximum patch stock capacity
        width: World width
        height: World height
        feed_radius: Feeding radius for agents
        dpi: Resolution for rendering
        
    Returns:
        RGB array (H, W, 3)
    """
    fig, ax = plt.subplots(figsize=(10, 10), dpi=dpi)
    ax = render_frame(
        pos, vel, patches_centers, patches_stock, patches_max_stock,
        width, height, feed_radius, ax=ax
    )
    
    # Convert to RGB array
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', facecolor='#1a1a2e')
    buf.seek(0)
    img = plt.imread(buf)
    plt.close(fig)
    
    return img

