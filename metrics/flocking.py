"""
Flocking metrics: polarization, nearest-neighbor distance, separation violations.
"""

import numpy as np


def polarization(vel: np.ndarray) -> float:
    """
    Compute group polarization (alignment).

    Args:
        vel: (N, 2) velocity vectors

    Returns:
        Polarization in [0, 1], where 1 is perfect alignment
    """
    if vel.shape[0] == 0:
        return 0.0

    # Normalize velocities
    speeds = np.linalg.norm(vel, axis=1, keepdims=True)
    speeds = np.maximum(speeds, 1e-8)  # Avoid division by zero
    vel_norm = vel / speeds

    # Average direction
    avg_direction = np.mean(vel_norm, axis=0)
    return float(np.linalg.norm(avg_direction))


def mean_knn_distance(distances: np.ndarray) -> float:
    """
    Compute mean distance to k-nearest neighbors.

    Args:
        distances: (N, k) distances to k nearest neighbors

    Returns:
        Mean distance
    """
    return float(np.mean(distances))


def separation_violations(distances: np.ndarray, d_safe: float) -> float:
    """
    Compute fraction of agent pairs violating safe distance.

    Args:
        distances: (N, k) distances to k nearest neighbors
        d_safe: minimum safe distance

    Returns:
        Fraction of violations in [0, 1]
    """
    if distances.size == 0:
        return 0.0

    violations = distances < d_safe
    return float(np.mean(violations))


def cohesion_score(distances: np.ndarray, target_distance: float = 3.0) -> float:
    """
    Compute cohesion score based on deviation from target distance.

    Args:
        distances: (N, k) distances to k nearest neighbors
        target_distance: desired average distance to neighbors

    Returns:
        Cohesion score (higher is better)
    """
    mean_dist = np.mean(distances)
    # Return negative squared error
    return -float((mean_dist - target_distance) ** 2)
