# physical.py - Physical Constraint Module

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

class PhysicalConstraints:
    """
    Manages physical constraints for biological realism.
    Applies limits to vocal parameters, movement, etc.
    """
    
    def __init__(self):
        # Vocal range limits (Hz)
        self.f0_min = 85.0  # Lowest fundamental frequency
        self.f0_max = 1100.0  # Highest fundamental frequency
        
        # Formant limits
        self.formant_limits = {
            'F1': (200, 1200),
            'F2': (500, 3000),
            'F3': (1500, 4000)
        }
        
        # Amplitude limits
        self.amp_min = 0.0
        self.amp_max = 1.0
        
        # Fatigue factors
        self.fatigue_factors = {
            'vocal': 0.1,
            'motor': 0.05
        }
    
    def apply_vocal_constraints(self, vocal_params: Dict[str, float]) -> Dict[str, float]:
        """
        Apply physical constraints to vocal parameters.
        """
        constrained = vocal_params.copy()
        
        # Constrain fundamental frequency
        if 'f0' in constrained:
            constrained['f0'] = np.clip(constrained['f0'], self.f0_min, self.f0_max)
        
        # Constrain formants
        for i in range(1, 4):
            key = f'F{i}'
            if key in constrained:
                min_val, max_val = self.formant_limits[key]
                constrained[key] = np.clip(constrained[key], min_val, max_val)
        
        # Constrain amplitude
        if 'amplitude' in constrained:
            constrained['amplitude'] = np.clip(constrained['amplitude'], self.amp_min, self.amp_max)
        
        return constrained

    def constrain_vocal_parameter(self, param_name: str, param_value: float) -> float:
        """
        Apply physical constraints to a single vocal parameter.
        """
        if param_name == 'F0_hz':
            return np.clip(param_value, self.f0_min, self.f0_max)
        elif param_name.startswith('F') and param_name.endswith('_hz'):
            # Formant frequencies
            formant_num = param_name[1]  # Extract number from F1_hz, F2_hz, etc.
            if formant_num in ['1', '2', '3']:
                min_val, max_val = self.formant_limits[f'F{formant_num}']
                return np.clip(param_value, min_val, max_val)
        elif param_name == 'Amplitude':
            return np.clip(param_value, self.amp_min, self.amp_max)
        
        # For other parameters, return as-is for now
        return param_value

    def get_system_health(self) -> float:
        """
        Returns the current system health score (0.0 to 1.0).
        Used by output.py bionet module to assess physical constraints.
        
        Returns:
            float: Health score between 0.0 (critical) and 1.0 (optimal)
        """
        # Default implementation returns a reasonable middle-ground health score
        # Subclasses or calling code can modify this based on actual system metrics
        return 0.7  # 70% health by default

class Physical:
    """
    Main physical simulation module.
    Manages body state, constraints, and physical interactions.
    """
    
    def __init__(self):
        self.constraints = PhysicalConstraints()
        self.body_state = {
            'energy': 100.0,
            'fatigue': 0.0,
            'health': 100.0
        }
        
        logger.info("Physical module initialized")
    
    def update_body_state(self, activity_level: float):
        """
        Update body state based on activity.
        """
        # Simple energy consumption
        energy_cost = activity_level * 0.1
        self.body_state['energy'] = max(0, self.body_state['energy'] - energy_cost)
        
        # Fatigue accumulation
        fatigue_gain = activity_level * 0.05
        self.body_state['fatigue'] = min(100, self.body_state['fatigue'] + fatigue_gain)
    
    def get_constraints(self):
        """
        Get physical constraints.
        """
        return self.constraints
