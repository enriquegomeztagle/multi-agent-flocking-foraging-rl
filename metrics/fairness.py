"""
Fairness metrics: Gini coefficient and Theil index.
"""

import numpy as np


def gini(x: np.ndarray) -> float:
    """
    Compute Gini coefficient of inequality.

    Args:
        x: Array of values (e.g., food intake per agent)

    Returns:
        Gini coefficient in [0, 1], where 0 is perfect equality
    """
    x = np.asarray(x, dtype=float).ravel()

    if np.all(x == 0):
        return 0.0

    x_sorted = np.sort(x)
    n = x_sorted.size
    cum = np.cumsum(x_sorted)

    return (n + 1 - 2 * np.sum(cum) / cum[-1]) / n


def theil(x: np.ndarray) -> float:
    """
    Compute Theil index of inequality.

    Args:
        x: Array of values (e.g., food intake per agent)

    Returns:
        Theil index >= 0, where 0 is perfect equality
    """
    x = np.asarray(x, dtype=float).ravel()
    x = x[x > 0]  # Filter out zeros

    if len(x) == 0:
        return 0.0

    mean_x = np.mean(x)
    return float(np.mean(x / mean_x * np.log(x / mean_x)))
