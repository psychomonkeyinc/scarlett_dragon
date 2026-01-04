# conscience.py

import os
import numpy as np
import logging
import pickle
import time
from typing import Dict, List, Optional, Tuple, Any

# Assume nn.py is available for Sequential, Linear, Sigmoid, ReLU, Tanh
# These would be used for a neural network that learns moral judgment.
from nn import Sequential, Linear, Sigmoid, ReLU, Tanh
from config import (MEMORY_EMOTIONAL_STATE_DIM, CONSCIENCE_UNIFIED_COGNITIVE_STATE_DIM,
                   CONSCIENCE_EMOTIONAL_STATE_DIM, CONSCIENCE_TOM_PERCEPTION_DIM,
                   CONSCIENCE_HIDDEN_DIM_1, CONSCIENCE_HIDDEN_DIM_2)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Conscience:
    """
    Lillith's Ethics/Morals and Conscience Router.
    Evaluates observed actions/entities against intrinsic moral values to produce
    a moral judgment score, influencing higher-level cognitive processes.
    """
    
    # Define abstract intrinsic moral values/principles.
    # These would guide the internal evaluation network.
    MORAL_PRINCIPLES = [
        'MinimizeHarm',
        'MaximizeWellbeing',
        'PromoteFairness',
        'UpholdBonds',
        'RespectAutonomy',
        'PromoteTruth' # Could be relevant for emergent communication
    ]

    def __init__(self,
                 unified_cognitive_state_dim: int = CONSCIENCE_UNIFIED_COGNITIVE_STATE_DIM, # From Mind.py
                 emotional_state_dim: int = CONSCIENCE_EMOTIONAL_STATE_DIM,         # From Emotion.py
                 tom_perception_dim: int = CONSCIENCE_TOM_PERCEPTION_DIM,          # From ToM.py (understanding of others' mental states)
                 # Output from this module: moral_judgment_score (scalar -1 to 1)
                 ):
        
        self.unified_cognitive_state_dim = unified_cognitive_state_dim
        self.emotional_state_dim = emotional_state_dim
        self.tom_perception_dim = tom_perception_dim

        # Input dimension for the moral evaluation network
        # This network learns to map internal states + perception of others to a moral judgment.
        self.moral_eval_input_dim = (self.unified_cognitive_state_dim + 
                                     self.emotional_state_dim + 
                                     self.tom_perception_dim)

        # Neural network for Moral Judgment (learns to map states to judgment)
        self.moral_judgment_network = Sequential(
            Linear(self.moral_eval_input_dim, CONSCIENCE_HIDDEN_DIM_1), # Hidden layer
            ReLU(),
            Linear(CONSCIENCE_HIDDEN_DIM_1, CONSCIENCE_HIDDEN_DIM_2),
            ReLU(),
            Linear(CONSCIENCE_HIDDEN_DIM_2, 1), # Output to a single scalar judgment
            Tanh() # Tanh for output range of -1 to 1 (-1 is malevolent, 1 is benevolent)
        )
        
        # Internal state for moral tension, etc.
        self._moral_tension_level = np.float32(0.0) # 0.0 to 1.0, increases when actions/perceptions conflict with principles
        self._last_judgment_score = np.float32(0.0) # Stores last moral judgment

        logger.info(f"Conscience module initialized.")
        logger.info(f"  Moral Eval Input Dim: {self.moral_eval_input_dim}")

    def evaluate_moral_context(self,
                               unified_cognitive_state: np.ndarray, # From Mind.py
                               emotional_state: np.ndarray,         # From Emotion.py (e.g., empathy, anger, fear)
                               tom_understanding_of_agent: np.ndarray, # From ToM.py (e.g., inferred intent of observed entity/aggressor)
                               current_action_impulse: Optional[np.ndarray] = None # Optional: vector representing Lillith's own potential action
                               ) -> Tuple[float, float]: # Returns (moral_judgment_score, moral_tension_level)
        """
        Evaluates the moral context of an observed situation or potential action.
        Produces a moral judgment score for an entity/action, and updates internal moral tension.
        """
        if (unified_cognitive_state is None or emotional_state is None or 
            tom_understanding_of_agent is None):
            logger.warning("Conscience: Missing input for moral evaluation. Returning default judgment and tension.")
            return np.float32(0.0), np.float32(0.0)

        # Prepare combined input for the moral judgment network
        combined_input = np.concatenate([
            unified_cognitive_state,
            emotional_state,
            tom_understanding_of_agent
        ]).astype(np.float32)

        if combined_input.shape[0] != self.moral_eval_input_dim:
            logger.error(f"Conscience: Input dimension mismatch. Expected {self.moral_eval_input_dim}, got {combined_input.shape[0]}. Returning default judgment.")
            return np.float32(0.0), np.float32(0.0)

        # Reshape for NN (batch_size=1)
        combined_input = combined_input.reshape(1, -1)

        # --- Generate Moral Judgment Score ---
        # This score will be used by ItsAGirl.py for its protective hierarchy
        moral_judgment_raw = self.moral_judgment_network.forward(combined_input)
        moral_judgment_score = moral_judgment_raw[0,0] # Take scalar from batch output
        self._last_judgment_score = moral_judgment_score

        # --- Update Moral Tension Level ---
        # Moral tension increases when there's a strong conflict between the current state
        # (or potential action) and the intrinsic moral principles.
        # For initial implementation, this is a heuristic: high negative emotion, high perceived harm,
        # or potential action impulse leading to negative outcomes (if evaluated).
        
        # Example heuristic for tension:
        # High negative emotions + low perceived well-being in others (from ToM) + potential for harm (if action_impulse is provided)
        # Assuming negative emotions (Sad, Angry, Fearful, Disgusted) are at specific indices in 108D emotional state
        sad_idx = self.get_emotion_index('Sad', emotional_state)
        angry_idx = self.get_emotion_index('Angry', emotional_state)
        fearful_idx = self.get_emotion_index('Fearful', emotional_state)
        disgusted_idx = self.get_emotion_index('Disgusted', emotional_state)

        # Sum of negative emotions from Lillith's own emotional state
        lillith_negative_emotions = 0.0
        if sad_idx != -1: lillith_negative_emotions += emotional_state[sad_idx]
        if angry_idx != -1: lillith_negative_emotions += emotional_state[angry_idx]
        if fearful_idx != -1: lillith_negative_emotions += emotional_state[fearful_idx]
        if disgusted_idx != -1: lillith_negative_emotions += emotional_state[disgusted_idx]
        
        # Normalize to 0-1 for a component of tension
        lillith_negative_emotions_normalized = np.clip(lillith_negative_emotions / 4.0, 0.0, 1.0) # Assuming max 4 neg emotions active at 1.0

        # Tentative: Aggressor's negative intent (from ToM) increases tension
        # This needs a specific dimension from ToM.py's output (e.g., 'InferredMaliciousIntent')
        # For now, let's use the negative part of moral_judgment_score
        tension_from_judgment = np.clip(-moral_judgment_score, 0.0, 1.0) # If judgment is -1, tension is 1. If 1, tension is 0.

        # If a potential action leads away from moral principles, tension increases.
        # This requires a prediction of action outcome, or evaluation of current_action_impulse.
        # For now, current_action_impulse is not directly used in tension calc but can be added.
        
        # Total tension (simple sum/blend for now)
        new_tension = np.clip(lillith_negative_emotions_normalized * 0.5 + tension_from_judgment * 0.5, 0.0, 1.0)
        
        # Smooth update of internal moral tension state
        blending_factor = 0.1 # How quickly tension updates
        self._moral_tension_level = (self._moral_tension_level * (1.0 - blending_factor) + 
                                     new_tension * blending_factor).astype(np.float32)

        logger.debug(f"Conscience: Judgment Score: {moral_judgment_score:.4f}, Moral Tension: {self._moral_tension_level:.4f}")
        
        return moral_judgment_score, self._moral_tension_level

    def get_moral_judgment_score(self) -> float:
        """Returns the last calculated moral judgment score."""
        return self._last_judgment_score.copy()
    
    def get_moral_tension_level(self) -> float:
        """Returns the current internal moral tension level."""
        return self._moral_tension_level.copy()

    def get_networks(self) -> None:
        """Returns a list of all internal neural networks for optimization."""
        return [self.moral_judgment_network]

    def get_emotion_index(self, emotion_name: str, emotional_state: np.ndarray) -> int:
        """Helper to get index of emotion from emotion.py's global map."""
        # This would usually import emotion.EmotionCore.EMOTION_NAME_TO_INDEX
        # For actual integration, emotion.py must expose this map.
        try:
            from . import emotion as emotion_mod
            return emotion_mod.EmotionCore.EMOTION_NAME_TO_INDEX.get(emotion_name, -1)
        except (ImportError, AttributeError):
            # Fallback for standalone testing
            fallback_map = {
                'Sad': 1, 'Angry': 3, 'Fearful': 4, 'Disgusted': 2
            }
            return fallback_map.get(emotion_name, -1)
            


