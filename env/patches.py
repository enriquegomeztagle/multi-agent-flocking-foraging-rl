"""
Resource patches with logistic regeneration.
"""

import numpy as np


class PatchField:
    """Manages resource patches with harvesting and regeneration."""

    def __init__(self, cfg):
        """
        Initialize patch field.

        Args:
            cfg: EnvConfig with n_patches, S_max, S_thr, regen_r, width, height
        """
        self.cfg = cfg
        self._rng = np.random.default_rng()
        self.centers = None
        self.stock = None

    def reset(self, rng=None):
        """Reset patches with random centers and initial stock.

        Args:
            rng: Optional numpy random generator for deterministic initialization
        """
        if rng is not None:
            self._rng = rng

        self.centers = self._rng.uniform(
            [0, 0],
            [self.cfg.width, self.cfg.height],
            size=(self.cfg.n_patches, 2)
        ).astype(np.float32)
        # Initialize at 60% of max capacity
        self.stock = np.full(
            (self.cfg.n_patches,),
            0.6 * self.cfg.S_max,
            dtype=np.float32
        )

    def nearest(self, xy: np.ndarray) -> int:
        """
        Find nearest patch to position.

        Args:
            xy: (2,) position

        Returns:
            Index of nearest patch
        """
        d = self.centers - xy[None, :]
        return int(np.argmin(np.sum(d * d, axis=1)))

    def harvest(self, xy: np.ndarray, radius: float, c_max: float) -> float:
        """
        Harvest from nearest patch if within radius.

        Args:
            xy: (2,) agent position
            radius: feeding radius
            c_max: maximum consumption per step

        Returns:
            Amount harvested
        """
        j = self.nearest(xy)

        # Check if agent is within feeding radius
        dist = np.linalg.norm(self.centers[j] - xy)
        if dist <= radius and self.stock[j] > 0.0:
            take = float(min(c_max, self.stock[j]))
            self.stock[j] -= take
            return take

        return 0.0

    def regenerate(self):
        """Apply logistic regeneration to all patches."""
        S = self.stock
        r = self.cfg.regen_r
        S_max = self.cfg.S_max

        # Logistic growth: dS/dt = r*S*(1 - S/S_max)
        self.stock = np.clip(
            S + r * S * (1.0 - S / S_max),
            0.0,
            S_max
        )

    def get_patch_info(self, xy: np.ndarray) -> tuple:
        """
        Get information about nearest patch.

        Args:
            xy: (2,) agent position

        Returns:
            (patch_center, stock_level, normalized_stock, patch_id)
        """
        j = self.nearest(xy)
        return (
            self.centers[j],
            self.stock[j],
            self.stock[j] / self.cfg.S_max,
            j
        )

    def mean_stock_normalized(self) -> float:
        """Get mean stock across all patches, normalized."""
        return float(np.mean(self.stock) / self.cfg.S_max)

    def fraction_below_threshold(self) -> float:
        """Get fraction of patches below sustainability threshold."""
        return float(np.mean(self.stock < self.cfg.S_thr))
