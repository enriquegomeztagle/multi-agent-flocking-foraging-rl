"""
Sustainability metrics for resource management.
"""

import numpy as np


def stock_score(stock: np.ndarray, S_max: float) -> float:
    """
    Compute normalized stock score.

    Args:
        stock: Array of patch stocks
        S_max: Maximum stock capacity

    Returns:
        Mean normalized stock in [0, 1]
    """
    return float(np.mean(stock / S_max))


def below_threshold_time(stock_history: np.ndarray, S_thr: float) -> float:
    """
    Compute fraction of time any patch was below threshold.

    Args:
        stock_history: (T, n_patches) stock levels over time
        S_thr: sustainability threshold

    Returns:
        Fraction of timesteps with at least one patch below threshold
    """
    if stock_history.shape[0] == 0:
        return 0.0

    below = np.any(stock_history < S_thr, axis=1)
    return float(np.mean(below))


def min_stock_normalized(stock: np.ndarray, S_max: float) -> float:
    """
    Compute minimum normalized stock across patches.

    Args:
        stock: Array of patch stocks
        S_max: Maximum stock capacity

    Returns:
        Minimum normalized stock in [0, 1]
    """
    return float(np.min(stock) / S_max)


def sustainability_score(
    stock: np.ndarray,
    S_max: float,
    S_thr: float,
    penalty_weight: float = 1.0
) -> float:
    """
    Combined sustainability score with penalty for depleted patches.

    Args:
        stock: Array of patch stocks
        S_max: Maximum stock capacity
        S_thr: Sustainability threshold
        penalty_weight: Weight for below-threshold penalty

    Returns:
        Sustainability score (higher is better)
    """
    mean_stock = stock_score(stock, S_max)
    below_thr = np.mean(stock < S_thr)
    return mean_stock - penalty_weight * below_thr
