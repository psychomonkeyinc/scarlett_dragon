import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
import time
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EnergyState:
    cognitive_energy: float = 1.0  # Mental processing power
    physical_energy: float = 1.0   # Physical action capability
    recovery_rate: float = 1.0     # How fast energy replenishes
    stress_level: float = 0.0      # Affects energy consumption

@dataclass
class ResourceState:
    memory_allocation: float = 0.0  # Neural resource usage
    processing_load: float = 0.0    # Computational load
    attention_resources: float = 1.0 # Available attention
    learning_capacity: float = 1.0   # Current ability to learn

class Metabolism:
    """
    Manages energy distribution,this is a more advanced fatigue management than original.need 
     to add decresaseexhaustion or maturity scale
       resource allocation, and system maintenance.
    Simulates biological metabolism for cognitive and physical resources.
    """
    def __init__(self):
        # Core energy pools
        self.base_energy = 1.0
        self.max_energy = 1.0
        self.current_energy = EnergyState()
        
        # Resource tracking
        self.resources = ResourceState()
        
        # Activity costs (energy per second)
        self.energy_costs = {
            'thinking': 0.02,
            'learning': 0.05,
            'vocal_output': 0.03,
            'emotional_processing': 0.02,
            'memory_access': 0.01,
            'dream_state': 0.01
        }
        
        # Performance curves (how performance scales with energy)
        self.performance_curves = {
            'cognitive': lambda e: 1.0 - 0.8 * np.exp(-2.0 * e),
            'physical': lambda e: 1.0 - 0.9 * np.exp(-2.5 * e),
            'emotional': lambda e: 1.0 - 0.7 * np.exp(-1.8 * e)
        }
        
        # System state
        self.last_update = time.time()
        self.total_uptime = 0.0
        self.maintenance_needed = False
        self.dream_cycles = 0
        
        # Adaptive parameters
        self.adaptation_rate = 0.1
        self.stress_tolerance = 0.7
        self.recovery_bonus = 1.0
        
        # Resource pools (for different types of processing)
        self.resource_pools = {
            'short_term_memory': 1.0,
            'working_memory': 1.0,
            'attention_focus': 1.0,
            'emotional_capacity': 1.0,
            'learning_buffer': 1.0,
            'prediction_capacity': 1.0,
        }
        
        # Track burned calories (as metaphor for processing cost)
        self.calories_burned = 0.0
        self.calorie_history = []

    def update(self, dt: float):
        """Update metabolic state"""
        self.total_uptime += dt
        
        # Natural energy recovery
        self._process_energy_recovery(dt)
        
        # Resource reallocation
        self._manage_resources(dt)
        
        # Check for maintenance needs
        self._check_maintenance_status()
        
        # Update adaptation parameters
        self._adapt_parameters(dt)
        
        # Track calorie burn
        self._update_calorie_tracking(dt)

    def _process_energy_recovery(self, dt: float):
        """Handle natural energy recovery and limits"""
        # Base recovery rate affected by stress
        effective_recovery = self.current_energy.recovery_rate * \
                           (1.0 - 0.5 * self.current_energy.stress_level)
        
        # Apply recovery
        self.current_energy.cognitive_energy = min(
            self.max_energy,
            self.current_energy.cognitive_energy + effective_recovery * dt * 0.1
        )
        self.current_energy.physical_energy = min(
            self.max_energy,
            self.current_energy.physical_energy + effective_recovery * dt * 0.08
        )

    def _manage_resources(self, dt: float):
        """Manage and reallocate resources based on needs"""
        # Decay resource usage
        self.resources.memory_allocation *= 0.95
        self.resources.processing_load *= 0.90
        
        # Recover attention resources
        self.resources.attention_resources = min(
            1.0,
            self.resources.attention_resources + dt * 0.1
        )
        
        # Update learning capacity based on energy and stress
        self.resources.learning_capacity = self.performance_curves['cognitive'](
            self.current_energy.cognitive_energy
        ) * (1.0 - 0.5 * self.current_energy.stress_level)

    def _check_maintenance_status(self):
        """Check if system needs maintenance (like sleep)"""
        # Trigger maintenance need if:
        # - Energy too low
        # - Too much uptime
        # - Resource depletion
        energy_low = min(self.current_energy.cognitive_energy,
                        self.current_energy.physical_energy) < 0.3
        long_uptime = self.total_uptime > 16 * 3600  # 16 hours
        resources_low = min(self.resource_pools.values()) < 0.3
        
        self.maintenance_needed = energy_low or long_uptime or resources_low

    def _adapt_parameters(self, dt: float):
        """Adapt metabolic parameters based on usage patterns"""
        # Increase stress tolerance with exposure
        if self.current_energy.stress_level > self.stress_tolerance:
            self.stress_tolerance = min(
                0.9,
                self.stress_tolerance + dt * self.adaptation_rate
            )
        
        # Adjust recovery bonus based on maintenance cycles
        if self.dream_cycles > 0:
            self.recovery_bonus = min(
                2.0,
                self.recovery_bonus + 0.1 * self.dream_cycles
            )

    def _update_calorie_tracking(self, dt: float):
        """Track calorie burn as metaphor for processing cost"""
        # Base metabolic rate
        base_burn = 0.1 * dt  # 0.1 calories per second base rate
        
        # Additional costs based on activity
        cognitive_cost = (1.0 - self.current_energy.cognitive_energy) * 0.2 * dt
        physical_cost = (1.0 - self.current_energy.physical_energy) * 0.3 * dt
        stress_cost = self.current_energy.stress_level * 0.1 * dt
        
        total_burn = base_burn + cognitive_cost + physical_cost + stress_cost
        self.calories_burned += total_burn
        
        # Keep history for analysis
        self.calorie_history.append((time.time(), total_burn))
        if len(self.calorie_history) > 1000:
            self.calorie_history.pop(0)

    def consume_energy(self, activity: str, intensity: float = 1.0) -> bool:
        """Attempt to consume energy for an activity"""
        if activity not in self.energy_costs:
            return False
            
        cost = self.energy_costs[activity] * intensity
        
        # Check if we have enough energy
        if activity in ['thinking', 'learning', 'memory_access']:
            if self.current_energy.cognitive_energy < cost:
                return False
            self.current_energy.cognitive_energy -= cost
        else:
            if self.current_energy.physical_energy < cost:
                return False
            self.current_energy.physical_energy -= cost
            
        return True

    def allocate_resources(self, resource_type: str, amount: float) -> bool:
        """Attempt to allocate resources of a specific type"""
        if resource_type not in self.resource_pools:
            return False
            
        if self.resource_pools[resource_type] < amount:
            return False
            
        self.resource_pools[resource_type] -= amount
        return True

    def dream_cycle_completed(self):
        """Called when a dream cycle completes"""
        self.dream_cycles += 1
        self.maintenance_needed = False
        self.total_uptime = 0.0  # Reset uptime
        
        # Boost recovery and clear stress
        self.current_energy.recovery_rate = self.recovery_bonus
        self.current_energy.stress_level *= 0.5
        
        # Replenish resource pools
        for pool in self.resource_pools:
            self.resource_pools[pool] = min(1.0, self.resource_pools[pool] + 0.3)

    def get_performance_multiplier(self, activity_type: str) -> float:
        """Get current performance level for an activity type"""
        if activity_type in self.performance_curves:
            energy = (self.current_energy.cognitive_energy 
                     if activity_type == 'cognitive'
                     else self.current_energy.physical_energy)
            return self.performance_curves[activity_type](energy)
        return 1.0

    def get_status_report(self) -> Dict[str, float]:
        """Get current metabolic status"""
        return {
            'cognitive_energy': self.current_energy.cognitive_energy,
            'physical_energy': self.current_energy.physical_energy,
            'stress_level': self.current_energy.stress_level,
            'recovery_rate': self.current_energy.recovery_rate,
            'maintenance_needed': float(self.maintenance_needed),
            'calories_burned': self.calories_burned,
            'average_burn_rate': np.mean([b for _, b in self.calorie_history[-100:]])
        }
