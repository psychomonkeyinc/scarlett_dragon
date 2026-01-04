import numpy as np
import logging
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PhysicalSensation:
    """
    Handles physical sensations including:
    - Temperature (hot/cold)
    - Pressure (touch/pain)
    - Internal states (hunger/satiation, energy/fatigue)
    - Proprioception (body position/movement)
    """
    def __init__(self, sensation_dim: int = 1024):
        self.sensation_dim = sensation_dim
        
        # Core sensation vectors
        self.temperature = np.zeros(64, dtype=np.float32)  # Hot/cold sensations
        self.pressure = np.zeros(128, dtype=np.float32)    # Touch/pressure/pain
        self.internal = np.zeros(256, dtype=np.float32)    # Hunger, energy, etc
        self.proprioception = np.zeros(576, dtype=np.float32)  # Body awareness
        
        # Sensation thresholds
        self.pain_threshold = 0.7
        self.discomfort_threshold = 0.5
        self.pleasure_threshold = 0.8
        
        # Current state trackers
        self.energy_level = 1.0
        self.comfort_level = 1.0
        self.last_movement_time = 0.0
        
        # Homeostatic targets
        self.optimal_temperature = 0.5  # Normalized comfort zone
        self.optimal_energy = 0.8       # Preferred energy level
        
    def update_temperature(self, ambient_temp: float, duration: float):
        """Update temperature sensation based on environmental input"""
        delta = ambient_temp - self.optimal_temperature
        self.temperature *= 0.9  # Decay
        sensation = np.clip(delta * duration, -1, 1)
        self.temperature += sensation * np.random.normal(1, 0.1, self.temperature.shape)
        
    def update_internal_state(self, dt: float):
        """Update internal sensations like energy and comfort"""
        # Energy naturally depletes over time
        self.energy_level *= (1.0 - 0.01 * dt)
        self.internal[0:64] = self.energy_level * np.random.normal(1, 0.05, 64)
        
        # Comfort changes based on movement and position
        stillness_penalty = dt * 0.02 if self.last_movement_time > 10 else 0
        self.comfort_level = np.clip(self.comfort_level - stillness_penalty, 0, 1)
        self.internal[64:128] = self.comfort_level
        
    def process_movement(self, movement_vector: np.ndarray):
        """Update proprioception based on movement"""
        self.proprioception *= 0.8  # Decay old signals
        self.proprioception += movement_vector * np.random.normal(1, 0.1, self.proprioception.shape)
        self.last_movement_time = 0.0  # Reset stillness timer
        
    def get_unified_sensation(self) -> np.ndarray:
        """Combine all sensation vectors into unified sensation state"""
        return np.concatenate([
            self.temperature,
            self.pressure,
            self.internal,
            self.proprioception
        ]).astype(np.float32)
        
    def is_in_distress(self) -> bool:
        """Check if any sensation is beyond discomfort threshold"""
        return (np.max(np.abs(self.temperature)) > self.pain_threshold or
                np.max(np.abs(self.pressure)) > self.pain_threshold or
                self.energy_level < 0.2 or
                self.comfort_level < 0.3)