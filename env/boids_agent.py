"""
Classical Boids agent implementation (without RL).

Implements the three classic Boids rules (Reynolds, 1987):
1. Cohesion: Move towards the average position of neighbors
2. Alignment: Match velocity with average velocity of neighbors
3. Separation: Avoid crowding neighbors

Additionally includes foraging behavior: move towards nearest food patch.
"""

import numpy as np
from typing import Tuple, Dict, Any
from .flockforage_parallel import FlockForageParallel, EnvConfig


class ClassicalBoidsAgent:
    """
    Classical Boids agent that uses steering rules instead of RL.
    
    Combines:
    - Boids rules (cohesion, alignment, separation)
    - Foraging behavior (move towards nearest food patch)
    """
    
    def __init__(
        self,
        cohesion_weight: float = 1.0,
        alignment_weight: float = 1.0,
        separation_weight: float = 1.5,
        foraging_weight: float = 2.0,
        separation_distance: float = 2.0,
        max_steering_force: float = 0.3
    ):
        """
        Initialize Boids agent with rule weights.
        
        Args:
            cohesion_weight: Weight for cohesion rule
            alignment_weight: Weight for alignment rule
            separation_weight: Weight for separation rule
            foraging_weight: Weight for foraging behavior
            separation_distance: Distance threshold for separation
            max_steering_force: Maximum steering force magnitude
        """
        self.cohesion_weight = cohesion_weight
        self.alignment_weight = alignment_weight
        self.separation_weight = separation_weight
        self.foraging_weight = foraging_weight
        self.separation_distance = separation_distance
        self.max_steering_force = max_steering_force
    
    def compute_action(
        self,
        agent_idx: int,
        env: FlockForageParallel
    ) -> int:
        """
        Compute action for agent using classical Boids rules.
        
        Args:
            agent_idx: Index of the agent
            env: Environment instance with current state
            
        Returns:
            Discrete action (0-4) matching the environment's action space
        """
        # Get agent state
        pos = env._pos[agent_idx]
        vel = env._vel[agent_idx]
        heading = env._heading[agent_idx]
        
        # Get neighbors
        neighbor_indices = env._neighbors[agent_idx]
        neighbor_positions = env._pos[neighbor_indices]
        neighbor_velocities = env._vel[neighbor_indices]
        neighbor_distances = env._distances[agent_idx]
        
        # Get nearest patch info
        patch_center, patch_stock, _, _ = env._patches.get_patch_info(pos)
        
        # Compute steering forces
        cohesion_force = self._compute_cohesion(pos, neighbor_positions, neighbor_distances)
        alignment_force = self._compute_alignment(vel, neighbor_velocities, neighbor_distances)
        separation_force = self._compute_separation(pos, neighbor_positions, neighbor_distances)
        foraging_force = self._compute_foraging(pos, patch_center, patch_stock)
        
        # Combine forces
        total_force = (
            self.cohesion_weight * cohesion_force +
            self.alignment_weight * alignment_force +
            self.separation_weight * separation_force +
            self.foraging_weight * foraging_force
        )
        
        # Limit steering force
        force_magnitude = np.linalg.norm(total_force)
        if force_magnitude > self.max_steering_force:
            total_force = total_force / force_magnitude * self.max_steering_force
        
        # Convert force to discrete action
        action = self._force_to_action(total_force, vel, heading, env.cfg)
        
        return action
    
    def _compute_cohesion(
        self,
        pos: np.ndarray,
        neighbor_positions: np.ndarray,
        neighbor_distances: np.ndarray
    ) -> np.ndarray:
        """
        Compute cohesion force: move towards average position of neighbors.
        
        Args:
            pos: Current position (2D)
            neighbor_positions: Positions of neighbors (N, 2)
            neighbor_distances: Distances to neighbors (N,)
            
        Returns:
            Cohesion force vector (2D)
        """
        if len(neighbor_positions) == 0:
            return np.zeros(2, dtype=np.float32)
        
        # Calculate center of mass of neighbors
        center_of_mass = np.mean(neighbor_positions, axis=0)
        
        # Steering force towards center
        desired_velocity = center_of_mass - pos
        desired_magnitude = np.linalg.norm(desired_velocity)
        
        if desired_magnitude > 0:
            # Normalize and scale by desired speed
            desired_velocity = desired_velocity / desired_magnitude * 0.8
        else:
            desired_velocity = np.zeros(2, dtype=np.float32)
        
        return desired_velocity
    
    def _compute_alignment(
        self,
        vel: np.ndarray,
        neighbor_velocities: np.ndarray,
        neighbor_distances: np.ndarray
    ) -> np.ndarray:
        """
        Compute alignment force: match velocity with average velocity of neighbors.
        
        Args:
            vel: Current velocity (2D)
            neighbor_velocities: Velocities of neighbors (N, 2)
            neighbor_distances: Distances to neighbors (N,)
            
        Returns:
            Alignment force vector (2D)
        """
        if len(neighbor_velocities) == 0:
            return np.zeros(2, dtype=np.float32)
        
        # Average velocity of neighbors
        avg_neighbor_vel = np.mean(neighbor_velocities, axis=0)
        avg_magnitude = np.linalg.norm(avg_neighbor_vel)
        
        if avg_magnitude > 0:
            # Normalize and scale
            desired_velocity = avg_neighbor_vel / avg_magnitude * 0.8
        else:
            desired_velocity = np.zeros(2, dtype=np.float32)
        
        return desired_velocity
    
    def _compute_separation(
        self,
        pos: np.ndarray,
        neighbor_positions: np.ndarray,
        neighbor_distances: np.ndarray
    ) -> np.ndarray:
        """
        Compute separation force: avoid crowding neighbors.
        
        Args:
            pos: Current position (2D)
            neighbor_positions: Positions of neighbors (N, 2)
            neighbor_distances: Distances to neighbors (N,)
            
        Returns:
            Separation force vector (2D)
        """
        if len(neighbor_positions) == 0:
            return np.zeros(2, dtype=np.float32)
        
        separation_force = np.zeros(2, dtype=np.float32)
        
        for i, (neighbor_pos, dist) in enumerate(zip(neighbor_positions, neighbor_distances)):
            if dist < self.separation_distance and dist > 0:
                # Repulsive force inversely proportional to distance
                diff = pos - neighbor_pos
                diff_magnitude = np.linalg.norm(diff)
                if diff_magnitude > 0:
                    # Stronger force for closer neighbors
                    strength = 1.0 / (diff_magnitude + 0.1)
                    separation_force += (diff / diff_magnitude) * strength
        
        # Normalize separation force
        separation_magnitude = np.linalg.norm(separation_force)
        if separation_magnitude > 0:
            separation_force = separation_force / separation_magnitude * 1.0
        
        return separation_force
    
    def _compute_foraging(
        self,
        pos: np.ndarray,
        patch_center: np.ndarray,
        patch_stock: float
    ) -> np.ndarray:
        """
        Compute foraging force: move towards nearest food patch.
        
        Args:
            pos: Current position (2D)
            patch_center: Position of nearest patch (2D)
            patch_stock: Stock level of nearest patch (0-1)
            
        Returns:
            Foraging force vector (2D)
        """
        # Only forage if patch has food
        if patch_stock < 0.1:
            return np.zeros(2, dtype=np.float32)
        
        # Direction to patch
        direction = patch_center - pos
        distance = np.linalg.norm(direction)
        
        if distance > 0:
            # Normalize direction
            direction = direction / distance
            
            # Scale by patch stock (more food = stronger attraction)
            foraging_force = direction * patch_stock * 1.0
        else:
            foraging_force = np.zeros(2, dtype=np.float32)
        
        return foraging_force
    
    def _force_to_action(
        self,
        force: np.ndarray,
        current_vel: np.ndarray,
        current_heading: float,
        cfg: EnvConfig
    ) -> int:
        """
        Convert steering force to discrete action.
        
        Args:
            force: Desired steering force (2D)
            current_vel: Current velocity (2D)
            current_heading: Current heading angle (radians)
            cfg: Environment configuration
            
        Returns:
            Discrete action (0-4):
            0: Turn left
            1: Turn right
            2: Accelerate
            3: Decelerate
            4: No-op
        """
        # Current velocity direction
        current_speed = np.linalg.norm(current_vel)
        current_direction = np.array([np.cos(current_heading), np.sin(current_heading)])
        
        # Desired direction from force
        force_magnitude = np.linalg.norm(force)
        if force_magnitude < 0.01:
            return 4  # No-op if force is too small
        
        desired_direction = force / force_magnitude
        
        # Calculate angle difference
        current_angle = np.arctan2(current_direction[1], current_direction[0])
        desired_angle = np.arctan2(desired_direction[1], desired_direction[0])
        
        angle_diff = desired_angle - current_angle
        
        # Normalize angle difference to [-pi, pi]
        while angle_diff > np.pi:
            angle_diff -= 2 * np.pi
        while angle_diff < -np.pi:
            angle_diff += 2 * np.pi
        
        # Decide action based on angle difference and speed
        turn_threshold = 0.1  # radians (~6 degrees)
        speed_threshold = 0.3
        
        # Priority 1: Turn if misaligned
        if abs(angle_diff) > turn_threshold:
            if angle_diff > 0:
                return 1  # Turn right
            else:
                return 0  # Turn left
        
        # Priority 2: Adjust speed
        if force_magnitude > 0.5 and current_speed < speed_threshold:
            return 2  # Accelerate
        elif force_magnitude < 0.3 and current_speed > speed_threshold * 1.5:
            return 3  # Decelerate
        
        # Default: maintain current state
        return 4  # No-op

