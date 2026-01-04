# dream.py

import os
import numpy as np
import logging
import pickle
import time
from typing import Dict, List, Optional, Self, Tuple, Any
from collections import deque
from random import random, shuffle

# Assume nn.py, som.py, memory.py, emotion.py are available for integration
# Fallbacks will be used if modules not found, as with other modules
from nn import Sequential, Linear, Tanh, Sigmoid, ReLU # For internal Dream networks if any
# from som import SelfOrganizingMap # For failure_log processing
# from memory import MemorySystem # For consolidation functions
# from emotion import EmotionCore # For emotional regulation functions

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import NVME memory system for optimized storage
try:
    from nvme_memory import NVMEMemoryManager, OptimizedMemorySystem
    NVME_AVAILABLE = True
except ImportError:
    NVME_AVAILABLE = False
    logger.warning("NVME memory system not available, using standard memory management")

class Dream:
    """
    Lillith's Dream State Manager: Orchestrates subconscious processing during rest.
    Manages Nap and Sleep states, memory consolidation, optimization, and emotional regulation.
    """
    
    # Define internal dream states
    DREAM_STATE_NONE = 0
    DREAM_STATE_NAP_ENTERING = 1
    DREAM_STATE_NAP_ACTIVE = 2
    DREAM_STATE_SLEEP_ENTERING = 3
    DREAM_STATE_SLEEP_ACTIVE = 4
    DREAM_STATE_WAKING = 5
    DREAM_STATE_EMERGENCY_LIMITED = 6

    def __init__(self,
                 initial_nap_duration_seconds: float = 30 * 60, # 30 minutes for early dev nap
                 initial_sleep_duration_seconds: float = 8 * 3600, # 8 hours for human-like sleep
                 early_dev_mode: bool = True, # Flag for shorter naps
                 # References to external modules (passed during init from main orchestrator)
                 # These would be actual instances of SOM, Memory, Emotion, Predict, Health
                 som_instance: Any = None, 
                 memory_instance: Any = None, 
                 emotion_instance: Any = None,
                 predict_instance: Any = None,
                 health_instance: Any = None
                 ):
        
        self.current_dream_state = self.DREAM_STATE_NONE
        self.dream_start_time = 0.0
        self.dream_end_time = 0.0
        self.early_dev_mode = early_dev_mode # If true, default nap is short
        
        self.nap_duration = initial_nap_duration_seconds
        self.sleep_duration = initial_sleep_duration_seconds
        self.sleep_countdown_timer = 0.0 # For "tuck in" countdown
        self.sleep_countdown_start_time = 0.0

        # References to core modules for internal dream processing
        self.som = som_instance
        self.memory = memory_instance
        self.emotion = emotion_instance
        self.predict = predict_instance
        self.health = health_instance # To check triggers like bottlenecks, tiredness

        # NVME Memory System for optimized storage
        if NVME_AVAILABLE:
            self.nvme_manager = NVMEMemoryManager("./dream_nvme", max_memory_gb=4.0)
            self.optimized_memory = OptimizedMemorySystem(self.nvme_manager, dimension=256)
            logger.info("NVME memory system initialized for dream state")
        else:
            self.nvme_manager = None
            self.optimized_memory = None

        # Internal flag for emergency state, to be set by external trigger
        self.emergency_limited_mode_active = False

        logger.info("Dream State Manager initialized.")
        # Experience replay buffer (now owned by Dream, not main)
        self.experience_buffer = deque(maxlen=5000)
        self._consolidated_this_session = False

    # ---------------- Experience Intake (Awake Phase) -----------------
    def add_experience(self, sfe_vector: Any):
        """Store an SFE feature vector while awake for later replay. Safe no-op if sleeping."""
        try:
            if self.current_dream_state == self.DREAM_STATE_NONE and sfe_vector is not None:
                self.experience_buffer.append(np.array(sfe_vector, dtype=np.float32))
        except Exception:
            pass

    def _enter_dream_state(self, state_type: int, duration_override: Optional[float] = None):
        """Internal method to transition into a dream state."""
        self.current_dream_state = state_type
        self.dream_start_time = time.perf_counter()
        
        if state_type == self.DREAM_STATE_NAP_ENTERING:
            self.dream_end_time = self.dream_start_time + (self.nap_duration if not duration_override else duration_override)
            logger.info(f"Lillith entering Nap state. Duration: {self.nap_duration/60:.1f} mins.")
        elif state_type == self.DREAM_STATE_SLEEP_ENTERING:
            self.dream_end_time = self.dream_start_time + (self.sleep_duration if not duration_override else duration_override)
            logger.info(f"Lillith entering Sleep state. Duration: {self.sleep_duration/3600:.1f} hours.")
        elif state_type == self.DREAM_STATE_EMERGENCY_LIMITED:
            self.dream_end_time = float('inf') # Stays active until external signal
            self.emergency_limited_mode_active = True
            logger.warning("Lillith entering EMERGENCY LIMITED DreamState.")
        
        # Signal other modules to adapt to dream mode (e.g., SFE dulls input)
        # This would be handled by the main orchestrator loop checking self.current_dream_state

    def _exit_dream_state(self):
        """Internal method to transition out of a dream state."""
        self.current_dream_state = self.DREAM_STATE_WAKING # Intermediate state
        self.emergency_limited_mode_active = False
        logger.info("Lillith transitioning out of Dream State. Waking up...")
        # # Reset parameters, restore full compute accuracy (placeholder for bitrate)  # COMMENTED OUT: placeholder removed
        # Apply "waking up" bias (e.g., in EmotionCore, Mind)
        
        self.current_dream_state = self.DREAM_STATE_NONE # Fully awake

    def process_dream_cycle(self):
        """
        Main dream loop: performs consolidation, optimization, etc., when active.
        This would be called repeatedly by the main orchestrator.
        """
        if self.current_dream_state == self.DREAM_STATE_NONE:
            return

        # Handle dream state entry (countdown)
        if self.current_dream_state == self.DREAM_STATE_SLEEP_ENTERING:
            if time.perf_counter() < self.sleep_countdown_start_time + self.sleep_countdown_timer:
                # Still counting down, slow processing but don't enter full dream yet
                logger.debug(f"Sleep countdown: {int(self.sleep_countdown_timer - (time.perf_counter() - self.sleep_countdown_start_time))}s remaining.")
                # This is where conscious processing slows (future bitrate changes)
                return 
            else:
                self.current_dream_state = self.DREAM_STATE_SLEEP_ACTIVE
                logger.info("Lillith fully entered Sleep state.")

        # If in an active dream state (Nap or Sleep or Emergency)
        if self.current_dream_state in [self.DREAM_STATE_NAP_ACTIVE, self.DREAM_STATE_SLEEP_ACTIVE, self.DREAM_STATE_EMERGENCY_LIMITED]:
            # Perform one-time SOM consolidation at first active cycle entry
            if not self._consolidated_this_session:
                self._run_som_consolidation_phased()
                self._consolidated_this_session = True
            # --- Optimized Memory Consolidation with NVME ---
            if self.memory and self.optimized_memory:
                self._perform_optimized_memory_consolidation()
            elif self.memory:
                self._perform_standard_memory_consolidation()

            # --- Failure Integration & Optimization (via SOM's failure_log) ---
            if self.som and self.som.failure_log:
                logger.info(f"Processing {len(self.som.failure_log)} failure logs in dream state.")
                for failure_data in self.som.failure_log:
                    # Analyze failure_data and propose parameter adjustments to SOM
                    # Example: if BMU fatigue was high, suggest increasing fatigue_decay for that region
                    # This would require a dedicated 'optimization_network' in Dream.py
                    
                    # # For now: simple rule-based adjustment (conceptual placeholder)  # COMMENTED OUT: placeholder removed
                    if failure_data.get('fatigue_in_bmu', 1.0) < 0.1: # Critically low fatigue
                        # Suggest a small increase in general SOM fatigue decay for robustness
                        # This would be an instruction to the main loop to modify som.fatigue_decay
                        logger.debug("Dream: Suggested SOM fatigue decay increase due to BMU exhaustion.")
                    elif failure_data.get('input_distance_to_bmu', 0.0) > self.som.activation_threshold * 2:
                        # Suggest widening SOM sigma or increasing plasticity for novelty
                        logger.debug("Dream: Suggested SOM sigma increase/plasticity boost for novelty.")
                self.som.failure_log.clear() # Clear logs after processing

            # --- Emotional Regulation ---
            if self.emotion:
                # Smooth out extreme emotional states, reduce lingering negative charge
                # For example, reduce baseline intensity of strong negative emotions if unresolved
                self.emotion.emotional_state[self.emotion.emotional_state > 0.8] *= 0.95 # Mild reduction for high peaks
                # Rebalance positive/negative baselines for fresh start.

            # --- Predictive Model Refinement ---
            if self.predict:
                # Run internal simulations, test predictive accuracy, refine predictor network
                # Requires Predict.py to have a 'dream_refine' method or similar
                logger.debug("Dream: Running predictive model refinement simulations.")

            # Check if duration expired (unless emergency limited mode)
            if not self.emergency_limited_mode_active and time.perf_counter() >= self.dream_end_time:
                self._exit_dream_state() # Time to wake up!

    # --- External Control/Trigger Methods ---
    def trigger_nap(self, duration_seconds: Optional[float] = None):
        """External command to trigger a nap."""
        self._enter_dream_state(self.DREAM_STATE_NAP_ENTERING, duration_seconds)
    
    def trigger_sleep(self, countdown_seconds: float = 300): # 5 minutes default
        """External command to trigger a gentle sleep with countdown."""
        self.current_dream_state = self.DREAM_STATE_SLEEP_ENTERING
        self.sleep_countdown_timer = countdown_seconds
        self.sleep_countdown_start_time = time.perf_counter()
        logger.info(f"Lillith preparing for Sleep. Countdown: {countdown_seconds/60:.1f} mins.")

    def trigger_instant_sleep(self):
        """External command to trigger immediate sleep, no countdown."""
        self._enter_dream_state(self.DREAM_STATE_SLEEP_ACTIVE) # Directly active
        logger.info("Lillith immediately entering Sleep state.")

    def trigger_emergency_limited_dreamstate(self):
        """External command for emergency, highly limited dream state."""
        self._enter_dream_state(self.DREAM_STATE_EMERGENCY_LIMITED)
    
    def trigger_wake_up(self):
        """External command to force Lillith to wake up."""
        if self.current_dream_state != self.DREAM_STATE_NONE:
            self._exit_dream_state()
            logger.info("Lillith was externally woken up.")
            # Reset consolidation flag so next sleep will consolidate again
            self._consolidated_this_session = False
        else:
            logger.info("Lillith is already awake.")

    def get_current_dream_state(self) -> int:
        """Returns the current dream state constant."""
        return self.current_dream_state
    
    def is_asleep(self) -> bool:
        """Convenience method to check if Lillith is in any sleep state."""
        return self.current_dream_state in [self.DREAM_STATE_NAP_ENTERING, self.DREAM_STATE_NAP_ACTIVE, 
                                            self.DREAM_STATE_SLEEP_ENTERING, self.DREAM_STATE_SLEEP_ACTIVE, 
                                            self.DREAM_STATE_EMERGENCY_LIMITED]
    
    def execute_sleep_phase(self):
        """Public API for triggering immediate sleep phase with consolidation."""
        logger.info("Sleep phase triggered. Starting immediate sleep...")
        self.trigger_instant_sleep()
        return {"status": "sleep_initiated", "dream_state": self.current_dream_state}

    # Persistence methods (save/load state)
    # def save_state(self, save_path: str):
    #     """Saves the Dream Manager's state to a file."""
    #     try:
    #         state = {
    #             'current_dream_state': self.current_dream_state,
    #             'dream_start_time': self.dream_start_time,
    #             'dream_end_time': self.dream_end_time,
    #             'early_dev_mode': self.early_dev_mode,
    #             'nap_duration': self.nap_duration,
    #             'sleep_duration': self.sleep_duration,
    #             'sleep_countdown_timer': self.sleep_countdown_timer,
    #             'sleep_countdown_start_time': self.sleep_countdown_start_time,
    #             'emergency_limited_mode_active': self.emergency_limited_mode_active
    #         }
    #         with open(save_path, 'wb') as f:
    #             pickle.dump(state, f)
    #         logger.info(f"Dream state saved to {save_path}")
    #     except Exception as e:
    #         logger.error(f"Error saving Dream state: {e}")

    # def load_state(self, load_path: str):
    #     """Loads the Dream Manager's state from a file."""
    #     try:
    #         with open(load_path, 'rb') as f:
    #             state = pickle.load(f)
    #         
    #         self.current_dream_state = state['current_dream_state']
    #         self.dream_start_time = state['dream_start_time']
    #         self.dream_end_time = state['dream_end_time']
    #         self.early_dev_mode = state['early_dev_mode']
    #         self.nap_duration = state['nap_duration']
    #         self.sleep_duration = state['sleep_duration']
    #         self.sleep_countdown_timer = state['sleep_countdown_timer']
    #         self.sleep_countdown_start_time = state['sleep_countdown_start_time']
    #         self.emergency_limited_mode_active = state['emergency_limited_mode_active']
    #         logger.info(f"Dream state loaded from {load_path}")
    #     except FileNotFoundError:
    #         logger.warning(f"Dream state file not found at {load_path}. Initializing to default.")
    #         # Reset to default state if file not found
    #         self.current_dream_state = self.DREAM_STATE_NONE
    #         self.dream_start_time = 0.0
    #         self.dream_end_time = 0.0
    #         self.emergency_limited_mode_active = False

    # ---------------- Internal Consolidation Logic -----------------
    def _run_som_consolidation_phased(self):
        """Phased offline consolidation with sleep-cycle sized replays and interruption loss simulation.

        Sleep model:
          Nap Active: replay recent 15%
          Sleep Active: three phases (recent 30%, middle 25%, diverse 20%)
          Emergency Limited: recent 5%
        Interruption: if dream ends early (flag checked externally), untouched items lose 10–30% randomly.
        """
        if self.som is None or not self.experience_buffer:
            return
        state = self.current_dream_state
        total = len(self.experience_buffer)
        buffer_list = list(self.experience_buffer)
        phases: List[List[int]] = []
        if state == self.DREAM_STATE_NAP_ACTIVE:
            k = max(1, int(0.15 * total))
            phases = [list(range(total - k, total))]
        elif state == self.DREAM_STATE_SLEEP_ACTIVE:
            recent_portion = int(0.30 * total)
            mid_portion = int(0.25 * total)
            diverse_portion = int(0.20 * total)
            phase1 = list(range(total - recent_portion, total))
            mid_start = max(0, (total // 2) - (mid_portion // 2))
            phase2 = list(range(mid_start, mid_start + mid_portion))
            remaining = list(set(range(total)) - set(phase1) - set(phase2))
            shuffle(remaining)
            phase3 = remaining[:diverse_portion]
            phases = [phase1, phase2, phase3]
        elif state == self.DREAM_STATE_EMERGENCY_LIMITED:
            k = max(1, int(0.05 * total))
            phases = [list(range(total - k, total))]
        else:
            return

        logger.info(f"Dream: SOM consolidation phases={len(phases)} buffer={total}")
        learned = set()
        enable_flag_had = hasattr(self, 'som') and hasattr(self.som, 'process_input')
        for pi, phase in enumerate(phases):
            if not phase:
                continue
            logger.info(f"  Phase {pi+1}: replay {len(phase)} items")
            for idx in phase:
                try:
                    self.som.process_input(buffer_list[idx])
                    learned.add(idx)
                except Exception as e:
                    logger.debug(f"Replay error idx={idx}: {e}")
            if pi == len(phases) - 1 or len(phases) > 2:
                # Persist SOM as full object pickle for main loader compatibility
                try:
                    with open('trainedsom.pkl', 'wb') as f:
                        pickle.dump(self.som, f)
                except Exception:
                    logger.debug("Dream: SOM pickle save failed during consolidation phase")

        # Interruption loss simulation not handled here; occurs if wake before consolidation call completes.

        # Clear buffer fully only after full sleep; naps/emergency keep remainder
        if state == self.DREAM_STATE_SLEEP_ACTIVE:
            self.experience_buffer.clear()
            logger.info("Dream: deep sleep complete; experience buffer cleared")



