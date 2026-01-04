import numpy as np
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class HormoneProfile:
    dopamine: float = 0.0    # Reward/pleasure
    serotonin: float = 0.5   # Mood/wellbeing
    oxytocin: float = 0.0    # Bonding/trust
    cortisol: float = 0.2    # Stress/alertness
    adrenaline: float = 0.0  # Excitement/fear
    melatonin: float = 0.0   # Sleep/wake
    endorphins: float = 0.0  # Pain relief/euphoria
    vasopressin: float = 0.0 # Social bonding/memory

class EndocrineSystem:
    """
    Simulates a complex hormone system that influences emotional, physical,
    and cognitive states through chemical messenger analogs.
    """
    def __init__(self, base_dimension: int = 1024):
        self.base_dimension = base_dimension
        
        # Hormone state vectors (each hormone has multiple subtypes/effects)
        self.hormone_states = {
            'dopamine': np.zeros(128, dtype=np.float32),
            'serotonin': np.full(128, 0.5, dtype=np.float32),
            'oxytocin': np.zeros(128, dtype=np.float32),
            'cortisol': np.zeros(128, dtype=np.float32),
            'adrenaline': np.zeros(128, dtype=np.float32),
            'melatonin': np.zeros(128, dtype=np.float32),
            'endorphins': np.zeros(128, dtype=np.float32),
            'vasopressin': np.zeros(128, dtype=np.float32)
        }
        
        # Hormone interaction matrix (how hormones affect each other)
        self.interaction_matrix = self._initialize_interaction_matrix()
        
        # Circadian rhythm tracking
        self.internal_clock = 0.0  # 0-24 hour cycle
        self.last_update = time.time()
        
        # Homeostatic targets (ideal hormone levels)
        self.homeostatic_targets = HormoneProfile(
            dopamine=0.3,
            serotonin=0.6,
            oxytocin=0.2,
            cortisol=0.2,
            adrenaline=0.1,
            melatonin=0.1,
            endorphins=0.1,
            vasopressin=0.2
        )
        
        # Recovery and decay rates
        self.decay_rates = {
            'dopamine': 0.1,
            'serotonin': 0.05,
            'oxytocin': 0.15,
            'cortisol': 0.08,
            'adrenaline': 0.2,
            'melatonin': 0.1,
            'endorphins': 0.12,
            'vasopressin': 0.07
        }

    def _initialize_interaction_matrix(self) -> np.ndarray:
        """Create matrix defining how hormones influence each other"""
        hormones = list(self.hormone_states.keys())
        n_hormones = len(hormones)
        matrix = np.zeros((n_hormones, n_hormones), dtype=np.float32)
        
        # Define hormone interactions (based on biological systems)
        interactions = {
            ('dopamine', 'serotonin'): 0.2,    # Dopamine boost affects serotonin
            ('cortisol', 'serotonin'): -0.3,   # Stress reduces serotonin
            ('adrenaline', 'cortisol'): 0.4,   # Adrenaline increases cortisol
            ('melatonin', 'cortisol'): -0.2,   # Sleep hormone reduces stress
            ('oxytocin', 'cortisol'): -0.3,    # Social bonding reduces stress
            ('endorphins', 'cortisol'): -0.2,  # Natural painkillers reduce stress
            ('adrenaline', 'dopamine'): 0.2,   # Excitement can be pleasurable
            ('oxytocin', 'vasopressin'): 0.3,  # Social hormones work together
        }
        
        for (h1, h2), strength in interactions.items():
            i1, i2 = hormones.index(h1), hormones.index(h2)
            matrix[i1, i2] = strength
            
        return matrix

    def update(self, dt: float):
        """Update hormone levels based on time passing and interactions"""
        # Update internal clock (24-hour cycle)
        self.internal_clock = (self.internal_clock + dt/3600) % 24
        
        # Apply circadian rhythm effects
        self._apply_circadian_effects()
        
        # Process hormone interactions
        self._process_hormone_interactions(dt)
        
        # Apply natural decay and homeostatic regulation
        self._apply_homeostasis(dt)

    def _apply_circadian_effects(self):
        """Apply time-of-day effects on hormones"""
        hour = self.internal_clock
        
        # Melatonin peaks at night
        night_factor = np.sin(((hour - 20) % 24) * np.pi / 12)
        self.hormone_states['melatonin'] += max(0, night_factor) * 0.1
        
        # Cortisol peaks in morning
        morning_factor = np.sin(((hour - 8) % 24) * np.pi / 12)
        self.hormone_states['cortisol'] += max(0, morning_factor) * 0.1
        
        # Serotonin affected by daylight
        day_factor = np.sin(((hour - 12) % 24) * np.pi / 12)
        self.hormone_states['serotonin'] += max(0, day_factor) * 0.05

    def _process_hormone_interactions(self, dt: float):
        """Process how hormones affect each other"""
        hormones = list(self.hormone_states.keys())
        
        # Get current levels for interaction calculation
        current_levels = np.array([
            np.mean(self.hormone_states[h]) for h in hormones
        ])
        
        # Calculate interaction effects
        interaction_effects = self.interaction_matrix @ current_levels
        
        # Apply effects with time scaling
        for i, hormone in enumerate(hormones):
            effect = interaction_effects[i] * dt
            self.hormone_states[hormone] += effect
            
        # Ensure bounds
        for hormone in self.hormone_states:
            self.hormone_states[hormone] = np.clip(
                self.hormone_states[hormone], 0, 1)

    def _apply_homeostasis(self, dt: float):
        """Apply natural decay and homeostatic regulation"""
        for hormone, target in vars(self.homeostatic_targets).items():
            if hormone in self.hormone_states:
                current = self.hormone_states[hormone]
                decay = self.decay_rates[hormone] * dt
                
                # Calculate pull toward homeostatic target
                delta = (target - np.mean(current)) * decay
                
                # Apply with some randomness
                self.hormone_states[hormone] += delta * np.random.normal(1, 0.1, current.shape)
                self.hormone_states[hormone] = np.clip(self.hormone_states[hormone], 0, 1)

    def trigger_response(self, event_type: str, intensity: float):
        """Trigger hormone response to various events"""
        responses = {
            'pleasure': {'dopamine': 0.5, 'serotonin': 0.2, 'endorphins': 0.3},
            'stress': {'cortisol': 0.6, 'adrenaline': 0.4, 'serotonin': -0.2},
            'social': {'oxytocin': 0.4, 'vasopressin': 0.3, 'dopamine': 0.2},
            'pain': {'endorphins': 0.5, 'cortisol': 0.3, 'adrenaline': 0.4},
            'fear': {'adrenaline': 0.7, 'cortisol': 0.5, 'oxytocin': -0.2},
            'achievement': {'dopamine': 0.6, 'serotonin': 0.3, 'endorphins': 0.2}
        }
        
        if event_type in responses:
            for hormone, change in responses[event_type].items():
                # Apply change with intensity scaling and randomness
                delta = change * intensity * np.random.normal(1, 0.1, 128)
                self.hormone_states[hormone] += delta
                self.hormone_states[hormone] = np.clip(
                    self.hormone_states[hormone], 0, 1)

    def get_hormone_profile(self) -> HormoneProfile:
        """Get current hormone levels as a profile"""
        return HormoneProfile(
            **{h: float(np.mean(self.hormone_states[h])) 
               for h in self.hormone_states}
        )

    def get_hormone_vector(self) -> np.ndarray:
        """Get complete hormone state as a single vector"""
        return np.concatenate(list(self.hormone_states.values())).astype(np.float32)

    def get_emotional_influence(self) -> Dict[str, float]:
        """Calculate hormone influence on emotional states"""
        profile = self.get_hormone_profile()
        
        return {
            'happiness': (profile.dopamine * 0.4 + 
                        profile.serotonin * 0.4 +
                        profile.endorphins * 0.2),
            'anxiety': (profile.cortisol * 0.5 +
                       profile.adrenaline * 0.3 -
                       profile.serotonin * 0.2),
            'trust': (profile.oxytocin * 0.6 +
                     profile.vasopressin * 0.4),
            'alertness': (profile.adrenaline * 0.4 +
                         profile.cortisol * 0.3 -
                         profile.melatonin * 0.3),
            'calmness': (profile.serotonin * 0.4 +
                        profile.melatonin * 0.3 -
                        profile.cortisol * 0.3)
        }
