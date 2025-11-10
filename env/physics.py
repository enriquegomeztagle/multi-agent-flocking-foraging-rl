"""
Physics utilities for reflective boundaries and agent dynamics.
"""

import numpy as np
from typing import Tuple


def apply_reflective_boundaries(
    pos: np.ndarray,
    vel: np.ndarray,
    W: float,
    H: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply reflective boundary conditions.

    When agents hit a boundary, they bounce back (velocity is reflected).

    Args:
        pos: (N, 2) positions
        vel: (N, 2) velocities
        W: world width
        H: world height

    Returns:
        Updated (pos, vel) with reflections applied
    """
    # Check x boundaries
    hit_left = pos[:, 0] < 0
    hit_right = pos[:, 0] > W

    # Reflect x position and velocity
    pos[hit_left, 0] = -pos[hit_left, 0]
    pos[hit_right, 0] = 2 * W - pos[hit_right, 0]
    vel[hit_left | hit_right, 0] = -vel[hit_left | hit_right, 0]

    # Check y boundaries
    hit_bottom = pos[:, 1] < 0
    hit_top = pos[:, 1] > H

    # Reflect y position and velocity
    pos[hit_bottom, 1] = -pos[hit_bottom, 1]
    pos[hit_top, 1] = 2 * H - pos[hit_top, 1]
    vel[hit_bottom | hit_top, 1] = -vel[hit_bottom | hit_top, 1]

    # Clamp positions to ensure they stay within bounds
    pos[:, 0] = np.clip(pos[:, 0], 0, W)
    pos[:, 1] = np.clip(pos[:, 1], 0, H)

    return pos, vel


def integrate(
    pos: np.ndarray,
    vel: np.ndarray,
    heading: np.ndarray,
    action_vec: np.ndarray,
    cfg
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Integrate agent dynamics with discrete actions and reflective boundaries.

    Args:
        pos: (N, 2) positions
        vel: (N, 2) velocities
        heading: (N,) heading angles in radians
        action_vec: (N, 2) [turn, acceleration]
        cfg: EnvConfig with dt, turn_max, a_max, v_max, width, height

    Returns:
        Updated (pos, vel, heading)
    """
    # Apply turning with limits
    heading = heading + np.clip(action_vec[:, 0], -cfg.turn_max, cfg.turn_max)

    # Apply acceleration with limits
    speed = np.linalg.norm(vel, axis=1)
    speed = np.clip(speed + action_vec[:, 1], 0.0, cfg.v_max)

    # Update velocity from heading and speed
    vel = np.stack([np.cos(heading) * speed, np.sin(heading) * speed], axis=1)

    # Integrate position
    pos = pos + vel * cfg.dt

    # Apply reflective boundaries
    pos, vel = apply_reflective_boundaries(pos, vel, cfg.width, cfg.height)

    # Update heading to match reflected velocity
    heading = np.arctan2(vel[:, 1], vel[:, 0])

    return pos, vel, heading


def euclidean_distance(xy1: np.ndarray, xy2: np.ndarray) -> np.ndarray:
    """
    Compute Euclidean distance between points (for reflective boundaries).

    Args:
        xy1: (N, 2) or (2,) positions
        xy2: (M, 2) or (2,) positions

    Returns:
        (N, M) or scalar distance matrix
    """
    if xy1.ndim == 1:
        xy1 = xy1[None, :]
    if xy2.ndim == 1:
        xy2 = xy2[None, :]

    dx = xy1[:, None, 0] - xy2[None, :, 0]
    dy = xy1[:, None, 1] - xy2[None, :, 1]

    return np.sqrt(dx**2 + dy**2).squeeze()


def find_k_nearest(pos: np.ndarray, k: int, cfg) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find k nearest neighbors for each agent (excluding self).

    Args:
        pos: (N, 2) agent positions
        k: number of neighbors
        cfg: EnvConfig (not used with reflective boundaries, kept for compatibility)

    Returns:
        neighbors: (N, k) indices of nearest neighbors
        distances: (N, k) distances to nearest neighbors
    """
    N = pos.shape[0]
    k = min(k, N - 1)  # Can't have more neighbors than agents - 1

    # Compute all pairwise Euclidean distances
    dist_matrix = euclidean_distance(pos, pos)

    # Set diagonal to infinity to exclude self
    np.fill_diagonal(dist_matrix, np.inf)

    # Get k nearest for each agent
    neighbors = np.argpartition(dist_matrix, k-1, axis=1)[:, :k]
    distances = np.take_along_axis(dist_matrix, neighbors, axis=1)

    return neighbors, distances
