# --- START OF FILE health.py ---
# health.py

import os
import numpy as np
import logging
import pickle
import time
from typing import Dict, List, Optional, Self, Tuple, Any

# Assume nn.py is available for any internal health models
from nn import Sequential, Linear, Sigmoid, ReLU, Tanh
from config import (MEMORY_EMOTIONAL_STATE_DIM, HEALTH_SOM_MAP_SIZE, HEALTH_UNIFIED_COGNITIVE_STATE_DIM,
                   HEALTH_MORAL_TENSION_DIM, HEALTH_MAX_SOM_FAILURE_LOGS, HEALTH_MAX_UNRESTED_CYCLES,
                   HEALTH_CRITICAL_HEALTH_THRESHOLD, HEALTH_DREAM_TRIGGER_THRESHOLD, HEALTH_MAX_EMOTIONAL_DISTRESS_NORM)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Health:
    """
    Lillith's System Health / Self-Preservation System.
    Monitors internal metrics to assess well-being and trigger self-regulatory actions
    like entering a Dream state or initiating a recovery protocol.
    """
    def __init__(self,
                 som_map_size: Tuple[int, int] = HEALTH_SOM_MAP_SIZE,
                 unified_cognitive_state_dim: int = HEALTH_UNIFIED_COGNITIVE_STATE_DIM, # From Mind.py
                 emotional_state_dim: int = MEMORY_EMOTIONAL_STATE_DIM,         # From Emotion.py
                 moral_tension_dim: int = HEALTH_MORAL_TENSION_DIM,              # Scalar from Conscience.py
                 max_som_failure_logs: int = HEALTH_MAX_SOM_FAILURE_LOGS,         # Max entries in SOM's failure log to consider
                 max_unrested_cycles: int = HEALTH_MAX_UNRESTED_CYCLES,       # ~20 hours at 1Hz cycle for tiredness (max time awake before critical)
                 critical_health_threshold: float = HEALTH_CRITICAL_HEALTH_THRESHOLD, # Below this, immediate action needed
                 dream_trigger_threshold: float = HEALTH_DREAM_TRIGGER_THRESHOLD,   # Below this, recommend/trigger dream
                 max_emotional_distress_norm: float = HEALTH_MAX_EMOTIONAL_DISTRESS_NORM): # Norm of emotional state (108D) before triggering distress
        
        self.som_map_size = som_map_size
        self.unified_cognitive_state_dim = unified_cognitive_state_dim
        self.emotional_state_dim = emotional_state_dim
        self.moral_tension_dim = moral_tension_dim
        self.max_som_failure_logs = max_som_failure_logs

        self.max_unrested_cycles = max_unrested_cycles # For tiredness tracking
        self._current_unrested_cycles = 0 # Increments with each cycle Lillith is awake

        self.critical_health_threshold = np.float32(critical_health_threshold)
        self.dream_trigger_threshold = np.float32(dream_trigger_threshold)
        self.max_emotional_distress_norm = np.float32(max_emotional_distress_norm)
        
        self._current_health_score = np.float32(1.0) # 1.0 is optimal health
        self._tiredness_factor = np.float32(0.0) # 0.0 is rested, 1.0 is critically tired
        
        # Flags for external orchestrator
        self.trigger_nap_needed = False
        self.trigger_sleep_needed = False
        self.trigger_emergency_dreamstate_needed = False
        self.trigger_recovery_protocol_needed = False
        
        # Internal NN for holistic health assessment
        self.health_assessment_network = Sequential(
            Linear(7, 64),  # 7 inputs -> 64 hidden
            ReLU(),
            Linear(64, 32),  # Add another layer
            ReLU(),
            Linear(32, 1),   # Output 1D health score
            Sigmoid()        # Scale 0-1
        )
        
        logger.info("Health System initialized.")

    def update_health_metrics(self,
                              som_fatigue_map: np.ndarray,          # From SOM.py
                              som_failure_log_count: int,           # len(SOM.failure_log)
                              predict_error_norm: float,            # Norm of prediction error from Predict.py
                              moral_tension_level: float,           # From Conscience.py
                              emotional_state_norm: float,          # Norm of EmotionCore's state
                              cognitive_load_norm: float,           # Norm of Mind.py's unified state
                              is_dreaming: bool                     # From Dream.py
                              ):
        """
        Updates internal health metrics and assesses overall system health.
        Returns the current health score.
        """
        try:
            # Validate inputs
            if som_fatigue_map is None or som_fatigue_map.size == 0:
                som_fatigue_map = np.zeros(self.som_map_size, dtype=np.float32)
            if not isinstance(som_failure_log_count, (int, np.integer)):
                som_failure_log_count = 0
            predict_error_norm = float(predict_error_norm) if predict_error_norm is not None else 0.0
            moral_tension_level = float(moral_tension_level) if moral_tension_level is not None else 0.0
            emotional_state_norm = float(emotional_state_norm) if emotional_state_norm is not None else 0.0
            cognitive_load_norm = float(cognitive_load_norm) if cognitive_load_norm is not None else 0.0

            # --- Update Tiredness Factor ---
            if not is_dreaming:
                self._current_unrested_cycles = min(self._current_unrested_cycles + 1, self.max_unrested_cycles)
            else:
                # Lillith is resting, so tiredness decreases. This needs to be managed by Dream.py more finely.
                self._current_unrested_cycles = max(self._current_unrested_cycles - 100, 0) # Reduce faster than accumulate
            
            self._tiredness_factor = np.float32(self._current_unrested_cycles / self.max_unrested_cycles)
            self._tiredness_factor = np.clip(self._tiredness_factor, 0.0, 1.0) # Ensure 0-1 range

            # --- Aggregate Input Features for Health Assessment ---
            # Normalize inputs for the health assessment network/heuristics
            
            # 1. SOM Fatigue (Overall Health)
            avg_som_fatigue = np.mean(som_fatigue_map).astype(np.float32)
            
            # 2. SOM Failures (Learning/Robustness Issue)
            normalized_som_failures = np.clip(som_failure_log_count / self.max_som_failure_logs, 0.0, 1.0)
            
            # 3. Prediction Error (Understanding/Anticipation Issue)
            # predict_error_norm already normalized to its scale by Predict.py. Normalize further if needed.
            normalized_predict_error = np.clip(predict_error_norm / 10.0, 0.0, 1.0)  # Assuming max error ~10

            # 4. Moral Tension (Internal Conflict/Ethics Issue)
            normalized_moral_tension = np.clip(moral_tension_level, 0.0, 1.0) # Assuming 0-1 from Conscience.py

            # 5. Emotional Distress (Affective Well-being)
            # emotional_state_norm is norm of 108D vector. Higher means more intense emotion.
            # We want to detect distress, so this might be high negative emotions OR just high overall emotional load.
            # For simplicity, if current overall emotional intensity is too high (stress), or if negative emotions are dominant.
            
            # This will need access to EmotionCore's POSITIVE_SET and NEGATIVE_SET or its derived score.
            # For now, let's use the provided emotional_state_norm, higher means higher distress.
            normalized_emotional_distress = np.clip(emotional_state_norm / self.max_emotional_distress_norm, 0.0, 1.0)

            # 6. Cognitive Load (Mental Strain)
            # Already norm of Mind's state. Higher could mean more active processing, too high means overload.
            normalized_cognitive_load = np.clip(cognitive_load_norm / 10.0, 0.0, 1.0)  # Assuming max norm ~10

            # --- Use Health Assessment Network to calculate overall health ---
            # Network input order: (avg_som_fatigue, normalized_som_failures, normalized_predict_error, normalized_moral_tension, normalized_emotional_distress)
            health_input_vector = np.array([
                1.0 - avg_som_fatigue, # Higher fatigue = lower health. So 1-fatigue.
                1.0 - normalized_som_failures, # More failures = lower health. So 1-failures.
                1.0 - normalized_predict_error, # Higher error = lower health. So 1-error.
                1.0 - normalized_moral_tension, # Higher tension = lower health. So 1-tension.
                1.0 - normalized_emotional_distress, # Higher distress = lower health. So 1-distress.
                1.0 - normalized_cognitive_load, # Higher load = lower health. So 1-load.
                1.0 - self._tiredness_factor # Higher tiredness = lower health. So 1-fatigue.
            ], dtype=np.float32).reshape(1, -1)
            
            self._current_health_score = self.health_assessment_network.forward(health_input_vector)[0, 0]
            self._current_health_score = np.clip(self._current_health_score, 0.0, 1.0)

            # --- Check for Triggers (based on calculated health score and tiredness) ---
            prev_health = getattr(self, '_previous_health_score', 1.0)
            health_drop = prev_health - self._current_health_score
            
            self.trigger_nap_needed = False
            self.trigger_sleep_needed = False
            self.trigger_emergency_dreamstate_needed = False
            self.trigger_recovery_protocol_needed = False
            
            # Low health score or high tiredness can trigger dream state
            if self._current_health_score < self.dream_trigger_threshold and not is_dreaming:
                if self._tiredness_factor > 0.8 or health_drop > 0.2:
                    self.trigger_sleep_needed = True # Suggest full sleep
                elif self._tiredness_factor > 0.4 or health_drop > 0.1: # Moderately tired AND low health
                    self.trigger_nap_needed = True # Suggest nap
                else: # Low health, but not primarily due to tiredness
                    self.trigger_recovery_protocol_needed = True # Suggest other recovery

            # Specific critical flags from SOM override other triggers (like a forced shutdown)
            # This would be SOM.critical_threshold_exceeded from som.py
            # For integration:
            if som_failure_log_count > 0 and self._current_health_score < self.critical_health_threshold:
                # If there are active SOM failures and health is critically low
                # This could signal an emergency dreamstate or full recovery protocol
                self.trigger_emergency_dreamstate_needed = True
                logger.warning("Health: Critical SOM failures/low health detected. Suggesting emergency dreamstate.")

            self._previous_health_score = self._current_health_score
            
            logger.debug(f"Health updated. Score: {self._current_health_score:.4f}, Tiredness: {self._tiredness_factor:.4f}")
            return self._current_health_score
            
        except Exception as e:
            logger.error(f"Error updating health metrics: {e}")
            return self._current_health_score

    def get_current_health_score(self) -> float:
        """Returns the current overall health score (0-1)."""
        return self._current_health_score.copy()
    
    def get_tiredness_factor(self) -> float:
        """Returns the current tiredness factor (0-1)."""
        return self._tiredness_factor.copy()

    def get_networks(self) -> List[Any]:
        """Returns a list of all internal neural networks for optimization."""
        return [self.health_assessment_network]
   
    
    # Persistence methods (save/load state)
    def save_state(self, save_path: str):
        """Saves the Health module's state to a file."""
        try:
            state = {
                '_current_health_score': float(self._current_health_score),
                '_current_unrested_cycles': self._current_unrested_cycles,
                '_tiredness_factor': float(self._tiredness_factor),
                '_previous_health_score': float(getattr(self, '_previous_health_score', 1.0)),
                'health_assessment_network_weights': [(p[0].tolist(), p[1]) for p in self.health_assessment_network.get_trainable_params()]
            }
            with open(save_path, 'wb') as f:
                pickle.dump(state, f)
            logger.info(f"Health state saved to {save_path}")
        except Exception as e:
            logger.error(f"Error saving Health state: {e}")

    def load_state(self, load_path: str):
        """Loads the Health module's state from a file."""
        try:
            with open(load_path, 'rb') as f:
                state = pickle.load(f)
            
            self._current_health_score = np.float32(state.get('_current_health_score', 1.0))
            self._current_unrested_cycles = state.get('_current_unrested_cycles', 0)
            self._tiredness_factor = np.float32(state.get('_tiredness_factor', 0.0))
            self._previous_health_score = state.get('_previous_health_score', 1.0)
            
            # Load network weights
            loaded_params = state.get('health_assessment_network_weights', [])
            current_params = self.health_assessment_network.get_trainable_params()
            
            if len(loaded_params) == len(current_params):
                for i, (param_val_list, grad_name_str) in enumerate(loaded_params):
                    param_array, _, _ = current_params[i]
                    param_array[:] = np.array(param_val_list, dtype=np.float32)
                logger.info(f"Health network weights loaded from {load_path}")
            else:
                logger.warning("Health network weights mismatch. Using current weights.")
                
        except FileNotFoundError:
            logger.warning(f"Health state file not found at {load_path}. Using defaults.")
        except Exception as e:
            logger.error(f"Error loading Health state: {e}. Using defaults.")

    def update(self, som_bmu_activity=None, unified_state_vector=None, emotional_state=None):
        """
        Simplified update method for the orchestrator.
        """
        # Calculate norms or use defaults
        som_failure_count = 0
        predict_error = 0.0
        moral_tension = 0.0
        
        emo_norm = np.linalg.norm(emotional_state) if emotional_state is not None else 0.0
        cog_norm = np.linalg.norm(unified_state_vector) if unified_state_vector is not None else 0.0
        
        return self.update_health_metrics(
            som_fatigue_map=som_bmu_activity,
            som_failure_log_count=som_failure_count,
            predict_error_norm=predict_error,
            moral_tension_level=moral_tension,
            emotional_state_norm=emo_norm,
            cognitive_load_norm=cog_norm,
            is_dreaming=False
        )
