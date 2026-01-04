# world_map.py - Spatial World Map Module

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

class WorldMapModule:
    """
    Manages spatial awareness, environmental mapping, and location-based cognition.
    Provides world representation for navigation and context.
    """
    
    def __init__(self, map_size: Tuple[int, int] = (50, 50), feature_dim: int = 32):
        self.map_size = map_size
        self.feature_dim = feature_dim
        # Spatial map (x, y, features)
        self.spatial_map = np.zeros((map_size[0], map_size[1], feature_dim), dtype=np.float32)
        # Current position
        self.current_position = (map_size[0] // 2, map_size[1] // 2)
        # Exploration history
        self.explored_positions: set = set()
        # Landmark database
        self.landmarks: Dict[str, Tuple[int, int]] = {}
        
    def update_position(self, new_position: Tuple[int, int], sensory_features: np.ndarray):
        """
        Update current position and map with sensory features.
        """
        x, y = new_position
        if 0 <= x < self.map_size[0] and 0 <= y < self.map_size[1]:
            self.current_position = new_position
            self.explored_positions.add(new_position)
            
            # Update map with sensory features
            self.spatial_map[x, y] = sensory_features[:self.feature_dim]
            
    def get_local_context(self, radius: int = 3) -> np.ndarray:
        """
        Get local spatial context around current position.
        """
        x, y = self.current_position
        x_min, x_max = max(0, x - radius), min(self.map_size[0], x + radius + 1)
        y_min, y_max = max(0, y - radius), min(self.map_size[1], y + radius + 1)
        
        local_map = self.spatial_map[x_min:x_max, y_min:y_max]
        return local_map.flatten()
    
    def add_landmark(self, name: str, position: Tuple[int, int], features: np.ndarray):
        """
        Add a landmark to the map.
        """
        self.landmarks[name] = position
        self.update_position(position, features)
        
    def find_path_to_landmark(self, landmark_name: str) -> List[Tuple[int, int]]:
        """
        Find simple path to a landmark (basic A* placeholder).
        """
        if landmark_name not in self.landmarks:
            return []
            
        # Simple direct path (placeholder for full pathfinding)
        target = self.landmarks[landmark_name]
        current = self.current_position
        
        path = []
        dx = target[0] - current[0]
        dy = target[1] - current[1]
        
        steps = max(abs(dx), abs(dy))
        if steps == 0:
            return [current]
            
        for i in range(steps + 1):
            x = current[0] + int(dx * i / steps)
            y = current[1] + int(dy * i / steps)
            path.append((x, y))
            
        return path
    
    def get_exploration_coverage(self) -> float:
        """
        Get fraction of map explored.
        """
        total_cells = self.map_size[0] * self.map_size[1]
        return len(self.explored_positions) / total_cells