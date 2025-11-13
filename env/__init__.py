"""
Multi-agent flocking and foraging environment.
"""

from .flockforage_parallel import FlockForageParallel, EnvConfig
from .boids_agent import ClassicalBoidsAgent

__all__ = ["FlockForageParallel", "EnvConfig", "ClassicalBoidsAgent"]
