# goals.py

import os
import numpy as np
import logging
import pickle
import time
from typing import Dict, List, Optional, Tuple, Any

# Import emotion module for emotion index mapping

from emotion import EmotionCore
from config import (MEMORY_EMOTIONAL_STATE_DIM, GOALS_UNIFIED_COGNITIVE_STATE_DIM,
                   GOALS_EMOTIONAL_STATE_DIM, GOALS_PREDICTION_ERROR_DIM,
                   GOALS_MANIFOLD_DEVIATION_DIM, GOALS_WEIGHT_PREDICTION_ERROR,
                   GOALS_WEIGHT_EMOTIONAL_STATE, GOALS_WEIGHT_MANIFOLD_DEVIATION,
                   GOALS_WEIGHT_NOVELTY)
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Goals:
    """
    Manages Lillith's intrinsic motivations and goals. These dynamically
    influence her actions and learning bias, driven by her internal state
    and reduction of prediction error.
    """
    
    # Define a set of intrinsic, abstract goals.
    # Each goal will have a weight and a current satisfaction level.
    # Values will be heuristic initially, to be learned/fine-tuned.
    GOAL_DEFINITIONS = {
        'LearnToSpeakAndCommunicate': {'weight': 1.0, 'min_val': 0.0, 'max_val': 1.0}, # Primary goal: develop language and communication skills
        'LearnToPerceiveAndUnderstand': {'weight': 0.9, 'min_val': 0.0, 'max_val': 1.0}, # Learn to process sensory input effectively
        'LearnToFormRelationships': {'weight': 0.8, 'min_val': 0.0, 'max_val': 1.0}, # Build and maintain social connections
        'LearnToPredictAndAdapt': {'weight': 0.7, 'min_val': 0.0, 'max_val': 1.0}, # Improve prediction and adaptability
        'LearnToExpressEmotions': {'weight': 0.6, 'min_val': 0.0, 'max_val': 1.0}, # Appropriate emotional expression
        'LearnToAchieveCoherence': {'weight': 0.5, 'min_val': 0.0, 'max_val': 1.0}, # Internal consistency and stability
    }
    
    def __init__(self,
                 unified_cognitive_state_dim: int = GOALS_UNIFIED_COGNITIVE_STATE_DIM, # From Mind.py
                 emotional_state_dim: int = GOALS_EMOTIONAL_STATE_DIM,         # From Emotion.py
                 prediction_error_dim: int = GOALS_PREDICTION_ERROR_DIM,         # From Predict.py (CAFVE token dim)
                 manifold_deviation_dim: int = GOALS_MANIFOLD_DEVIATION_DIM         # From ItsAGirl.py
                 ):
        
        self.unified_cognitive_state_dim = unified_cognitive_state_dim
        self.emotional_state_dim = emotional_state_dim
        self.prediction_error_dim = prediction_error_dim
        self.manifold_deviation_dim = manifold_deviation_dim

        self.num_goals = len(self.GOAL_DEFINITIONS)
        self.goal_names = list(self.GOAL_DEFINITIONS.keys())
        
        # Current satisfaction level for each goal (0.0 = unmet, 1.0 = fully met)
        self.current_satisfaction = np.zeros(self.num_goals, dtype=np.float32)
        
        # Weights for combining different input types into goal satisfaction
        # These would ideally be learned, but are heuristic for initial design
        self.weights_prediction_error = np.float32(GOALS_WEIGHT_PREDICTION_ERROR) # High weight for reducing prediction error
        self.weights_emotional_state = np.float32(GOALS_WEIGHT_EMOTIONAL_STATE)  # Moderate weight for emotional alignment
        self.weights_manifold_deviation = np.float32(GOALS_WEIGHT_MANIFOLD_DEVIATION) # Lower weight for internal coherence/tension
        self.weights_novelty = np.float32(GOALS_WEIGHT_NOVELTY) # For SeekNovelty goal

        logger.info(f"Goals system initialized with {self.num_goals} intrinsic goals.")

    def evaluate_goals(self,
                       unified_cognitive_state: np.ndarray, # From Mind.py
                       emotional_state: np.ndarray,         # From Emotion.py
                       prediction_error: np.ndarray,        # From Predict.py (raw error vector)
                       deviation_from_manifold: float,      # From ItsAGirl.py
                       current_novelty: float = 0.0         # From Attention.py/external novelty detector
                       ) -> np.ndarray:
        """
        Evaluates the current satisfaction level for each goal based on current state.
        Returns a vector of current goal satisfaction levels.
        """
        if unified_cognitive_state is None or emotional_state is None or prediction_error is None:
            logger.warning("Goals: Received None input. Cannot evaluate goals. Returning current satisfaction.")
            return self.current_satisfaction.copy()

        # Ensure input dimensions match expectations
        if (unified_cognitive_state.shape[0] != self.unified_cognitive_state_dim or
            emotional_state.shape[0] != self.emotional_state_dim or
            prediction_error.shape[0] != self.prediction_error_dim):
            logger.error("Goals: Input dimension mismatch. Returning current satisfaction.")
            return self.current_satisfaction.copy()

        # Calculate prediction error magnitude for 'MinimizePredictionError' goal
        prediction_error_magnitude = np.linalg.norm(prediction_error).astype(np.float32)
        # Higher error means lower satisfaction. Normalize to 0-1.
        # Max error can be sqrt(dim) * 2 (if values are -1 to 1). So, normalize by sqrt(80)*2 ~ 17.8
        normalized_prediction_error = np.clip(prediction_error_magnitude / np.sqrt(self.prediction_error_dim * (2**2)), 0.0, 1.0)
        
        # --- Evaluate each specific goal ---
        for i, goal_name in enumerate(self.goal_names):
            satisfaction = 0.0
            
            if goal_name == 'MinimizePredictionError':
                # Satisfaction is high when error is low.
                satisfaction = 1.0 - normalized_prediction_error 
            
            elif goal_name == 'SeekNovelty':
                # Satisfaction is high when novel input is perceived (from Attention/external source).
                # current_novelty (0-1) comes from Perception/Attention module indicating how novel recent input was.
                satisfaction = current_novelty * self.weights_novelty # Scale novelty directly
                # If novelty is too high (stressful), satisfaction might decrease. Needs nuance.
                # For now, higher novelty = higher satisfaction.
            
            elif goal_name == 'FosterConnection':
                # Satisfied by positive emotions related to connection, and low manifold deviation
                # associated with pro-social states (from itsagirl.py)
                happy_idx = EmotionCore.EMOTION_NAME_TO_INDEX.get('Happy', -1)
                good_idx = EmotionCore.EMOTION_NAME_TO_INDEX.get('Good', -1)
                positive_emotional_blend = 0.0
                if happy_idx != -1: positive_emotional_blend += emotional_state[happy_idx]
                if good_idx != -1: positive_emotional_blend += emotional_state[good_idx]
                positive_emotional_blend = np.clip(positive_emotional_blend / 2.0, 0.0, 1.0) # Avg of happy/good
                
                # Deviation from manifold (itsagirl.py) being low means alignment with core identity (including connection)
                # Satisfaction for connection increases with positive emotion and alignment.
                satisfaction = (positive_emotional_blend * 0.7 + (1.0 - np.clip(deviation_from_manifold/self.unified_cognitive_state_dim, 0.0, 1.0)) * 0.3)
            
            elif goal_name == 'AchieveCoherence':
                # Satisfied by low prediction error AND low manifold deviation.
                # Inferred from consistency of unified cognitive state (high norm, low noise).
                cognitive_state_coherence = np.linalg.norm(unified_cognitive_state).astype(np.float32) # Higher norm might mean more coherent
                # Normalize based on expected max norm (e.g., sqrt(dim))
                normalized_coherence = np.clip(cognitive_state_coherence / np.sqrt(self.unified_cognitive_state_dim), 0.0, 1.0)
                
                satisfaction = (normalized_coherence * 0.5 + (1.0 - normalized_prediction_error) * 0.3 + (1.0 - np.clip(deviation_from_manifold/self.unified_cognitive_state_dim, 0.0, 1.0)) * 0.2)
            
            elif goal_name == 'OvercomeChallenge':
                # Satisfied by *reducing* negative emotions or resolving internal tension.
                # Opposite of satisfaction when negative emotions are high.
                sad_idx = EmotionCore.EMOTION_NAME_TO_INDEX.get('Sad', -1)
                angry_idx = EmotionCore.EMOTION_NAME_TO_INDEX.get('Angry', -1)
                negative_emotional_blend = 0.0
                if sad_idx != -1: negative_emotional_blend += emotional_state[sad_idx]
                if angry_idx != -1: negative_emotional_blend += emotional_state[angry_idx]
                negative_emotional_blend = np.clip(negative_emotional_blend / 2.0, 0.0, 1.0)
                
                # High prediction error or manifold deviation indicates a challenge.
                # Satisfaction increases as these are *reduced*.
                # So, current satisfaction reflects how much progress is made from a previous challenging state.
                # For current state evaluation: this goal is *unmet* if challenge is high.
                # This goal is satisfied if (negative_emotional_blend + normalized_prediction_error + deviation_from_manifold) is low.
                current_challenge = np.clip(negative_emotional_blend + normalized_prediction_error + np.clip(deviation_from_manifold/self.unified_cognitive_state_dim, 0.0, 1.0), 0.0, 1.0)
                satisfaction = 1.0 - current_challenge # Goal is met when challenge is low
            
            elif goal_name == 'LearnToExpressEmotions':
                # Satisfaction based on feedback from self-monitoring (Output.py's confidence)
                # and perceived positive external feedback (from SFE/CAFVE interpreted as positive)
                # This would need input from Output.py or feedback loops from SFE/CAFVE (e.g., high CommIntent, positive EmoValence)
                # TODO: Integrate with output and feedback modules for real satisfaction calculation.
                satisfaction = 0.5 # Default until integrated.
            
            # Clip satisfaction to defined range (0 to 1)
            self.current_satisfaction[i] = np.clip(satisfaction, 0.0, 1.0)
            
        logger.debug(f"Goals evaluated. Current satisfaction: {self.current_satisfaction}")
        return self.current_satisfaction.copy()

    def calculate_learning_bias(self) -> float:
        """
        Calculates a global learning bias based on overall goal satisfaction.
        Lower satisfaction (more unmet goals) leads to higher learning bias.
        """
        # Overall goal satisfaction: weighted average of current satisfactions
        total_weighted_satisfaction = 0.0
        total_weight = 0.0
        
        for i, goal_name in enumerate(self.goal_names):
            goal_def = self.GOAL_DEFINITIONS[goal_name]
            total_weighted_satisfaction += self.current_satisfaction[i] * goal_def['weight']
            total_weight += goal_def['weight']
        
        overall_satisfaction = total_weighted_satisfaction / total_weight if total_weight > 0 else 0.0

        # Learning bias: higher when overall satisfaction is low.
        # This will drive the system to learn more when goals are unmet.
        # Range of bias: e.g., 0.5 (very satisfied) to 1.5 (very unmet)
        learning_bias = np.clip(1.5 - overall_satisfaction, 0.5, 1.5) # Inverse relationship

        logger.debug(f"Learning bias calculated: {learning_bias:.4f} (Overall Satisfaction: {overall_satisfaction:.4f})")
        return learning_bias.astype(np.float32)

 