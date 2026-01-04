
# --- START OF FILE attention.py ---
# attention.py

import os
import numpy as np
import logging
import pickle
import time
from typing import Dict, List, Optional, Tuple, Any
from config import FLOATX, ATTENTION_SFE_FEATURE_DIM, ATTENTION_CAFVE_TOKEN_DIM, ATTENTION_UNIFIED_COGNITIVE_STATE_DIM, ATTENTION_EMOTIONAL_STATE_DIM, ATTENTION_GOALS_SATISFACTION_DIM, ATTENTION_PREDICT_ERROR_DIM, ATTENTION_PREDICT_NEXT_STATE_DIM, ATTENTION_SALIENCE_DIM, ATTENTION_FOCUS_VECTOR_DIM, ATTENTION_INTERNAL_DIM, ATTENTION_NOVELTY_THRESHOLD

# Import neural network layers from nn.py
from nn import Sequential, Linear, Sigmoid, ReLU, Tanh

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Attention:
    """
    Lillith's Attention / State Selector: Manages cognitive resources,
    selectively focuses on salient information, and allocates processing
    power based on goals, emotional state, and predictions.
    """
    def __init__(self,
                 sfe_feature_dim: int = ATTENTION_SFE_FEATURE_DIM,
                 cafve_token_dim: int = ATTENTION_CAFVE_TOKEN_DIM,
                 unified_cognitive_state_dim: int = ATTENTION_UNIFIED_COGNITIVE_STATE_DIM,
                 emotional_state_dim: int = ATTENTION_EMOTIONAL_STATE_DIM,
                 goals_satisfaction_dim: int = ATTENTION_GOALS_SATISFACTION_DIM,
                 predict_error_dim: int = ATTENTION_PREDICT_ERROR_DIM,
                 predict_next_state_dim: int = ATTENTION_PREDICT_NEXT_STATE_DIM,
                 salience_dim: int = ATTENTION_SALIENCE_DIM,
                 focus_vector_dim: int = ATTENTION_FOCUS_VECTOR_DIM,
                 internal_dim: int = ATTENTION_INTERNAL_DIM,
                 novelty_threshold: float = ATTENTION_NOVELTY_THRESHOLD):
        
        self.sfe_feature_dim = sfe_feature_dim
        self.cafve_token_dim = cafve_token_dim
        self.unified_cognitive_state_dim = unified_cognitive_state_dim
        self.emotional_state_dim = emotional_state_dim
        self.goals_satisfaction_dim = goals_satisfaction_dim
        self.predict_error_dim = predict_error_dim
        self.predict_next_state_dim = predict_next_state_dim
        self.salience_dim = salience_dim
        self.focus_vector_dim = focus_vector_dim
        self.novelty_threshold = FLOATX(novelty_threshold)

        # Network to generate saliency map from sensory and internal cues
        # Input: SFE features, CAFVE token, Prediction Error, Emotional state (as a driver of attention)
        self.saliency_network_input_dim = (
            self.sfe_feature_dim + 
            self.cafve_token_dim + 
            self.predict_error_dim +
            self.emotional_state_dim)
        
        self.saliency_network = Sequential(
            Linear(self.saliency_network_input_dim, internal_dim),
            ReLU(),
            Linear(internal_dim, self.salience_dim),
            Sigmoid()
        )

        # Network to determine the current focus vector
        # Input: Unified Cognitive State, Goals Satisfaction, Saliency Map, Predicted Next State, Moral Tension
        self.focus_network_input_dim = (self.unified_cognitive_state_dim +
                                        self.goals_satisfaction_dim + 
                                        self.salience_dim +
                                        self.predict_next_state_dim +
                                        1) # 1 for moral tension from Conscience.py
        
        self.focus_network = Sequential(
            Linear(self.focus_network_input_dim, internal_dim),
            ReLU(),
            Linear(internal_dim, self.focus_vector_dim),
            Tanh()
        )
        
        self._current_focus_vector = np.zeros(self.focus_vector_dim, dtype=FLOATX)
        self._current_novelty_score = FLOATX(0.0) # 0-1
        logger.info(f"Attention module initialized. Focus Vector Dim: {self.focus_vector_dim}")

    def generate_focus(self,
                       sfe_features: np.ndarray,            # ~55D SFE features
                       cafve_token: np.ndarray,             # ~80D CAFVE token
                       unified_cognitive_state: np.ndarray, # ~256D Mind state
                       emotional_state: np.ndarray,         # ~108D Emotion state
                       goals_satisfaction: np.ndarray,      # ~6D Goals satisfaction
                       predict_error: np.ndarray,           # ~80D Predict error vector
                       predict_next_state: np.ndarray,      # ~256D Predict next Mind state
                       moral_tension_level: float           # Scalar from Conscience.py
                       ) -> Tuple[np.ndarray, float]: # Returns (focus_vector, novelty_score)
        """
        Generates Lillith's current focus vector and novelty score.
        """
        # --- 1. Saliency Map Generation ---
        saliency_input = np.concatenate([
            sfe_features,
            cafve_token,
            predict_error,
            emotional_state
        ]).astype(np.float32).reshape(1, -1)
        
        if saliency_input.shape[1] != self.saliency_network_input_dim:
            logger.error(f"Attention: Saliency input dim mismatch. Expected {self.saliency_network_input_dim}, got {saliency_input.shape[1]}. Returning zeros.")
            return np.zeros(self.focus_vector_dim, dtype=np.float32), 0.0

        current_saliency_map = self.saliency_network.forward(saliency_input)[0,:]

        # --- 2. Novelty Detection ---
        # Novelty is high when prediction error is high AND saliency is high
        # OR when the current CAFVE token is very far from any known SOM prototypes.
        # For this lean implementation, use predict_error_norm as primary novelty signal.
        current_novelty_score = np.linalg.norm(predict_error).astype(np.float32) / np.sqrt(self.predict_error_dim) # Normalize
        current_novelty_score = np.clip(current_novelty_score, 0.0, 1.0) # Ensure 0-1 range
        self._current_novelty_score = current_novelty_score

        # --- 3. Focus Allocation ---
        focus_input = np.concatenate([
            unified_cognitive_state,
            goals_satisfaction,
            current_saliency_map,
            predict_next_state,
            np.array([moral_tension_level], dtype=np.float32)
        ]).astype(np.float32).reshape(1, -1)

        if focus_input.shape[1] != self.focus_network_input_dim:
            logger.error(f"Attention: Focus input dim mismatch. Expected {self.focus_network_input_dim}, got {focus_input.shape[1]}. Returning zeros.")
            return np.zeros(self.focus_vector_dim, dtype=np.float32), self._current_novelty_score

        self._current_focus_vector = self.focus_network.forward(focus_input)[0,:]
        
        logger.debug(f"Attention: Focus vector updated. Norm: {np.linalg.norm(self._current_focus_vector):.4f}, Novelty: {self._current_novelty_score:.4f}")
        
        return self._current_focus_vector.copy(), self._current_novelty_score

    def get_current_focus_vector(self) -> np.ndarray:
        """Returns the current focus vector."""
        return self._current_focus_vector.copy()

    def get_current_novelty_score(self) -> float:
        """Returns the current novelty score (0-1)."""
        return self._current_novelty_score.copy()

    def get_networks(self) -> None:
        """Returns a list of all internal neural networks for optimization."""
        return [self.saliency_network, self.focus_network]

    # --- Backward Compatibility Layer ---
        """
        def compute_attention(self, cognitive_state: np.ndarray, emotional_state: np.ndarray) -> np.ndarray:
        """#Legacy interface used by main.py.
        """
        Provides a reduced attention focus computation when the full multi-signal
        generate_focus() pathway isn't supplied with all inputs yet.

        This is NOT a mock: it derives a focus vector directly from current
        cognitive + emotional states using a lightweight projection and the
        existing focus_vector_dim. It keeps internal state updated so downstream
        modules receive a coherent focus vector until the full pipeline wiring
        (predict error, goals satisfaction, etc.) is integrated.
        """
        try:
            # Lazy init projection weights (kept deterministic across calls)
            if not hasattr(self, '_legacy_proj_weights'):
                rng = np.random.default_rng(seed=42)
                self._legacy_proj_weights = rng.standard_normal(
                    (self.unified_cognitive_state_dim + self.emotional_state_dim, self.focus_vector_dim)
                ).astype(np.float32) * (1.0 / np.sqrt(self.unified_cognitive_state_dim + self.emotional_state_dim))
                self._legacy_proj_bias = np.zeros(self.focus_vector_dim, dtype=np.float32)
            # Pad/truncate incoming pieces to expected sizes
            def _fit(vec, dim):
                v = np.array(vec, dtype=np.float32).reshape(-1)
                if v.shape[0] == dim:
                    return v
                if v.shape[0] > dim:
                    return v[:dim]
                out = np.zeros(dim, dtype=np.float32)
                out[:v.shape[0]] = v
                return out
            cognitive_state = _fit(cognitive_state, self.unified_cognitive_state_dim)
            emotional_state = _fit(emotional_state, self.emotional_state_dim)
            combined = np.concatenate([cognitive_state, emotional_state])

            raw_focus = combined @ self._legacy_proj_weights + self._legacy_proj_bias
            focus_vec = np.tanh(raw_focus)  # keep within -1..1 like main focus network
            self._current_focus_vector = focus_vec.astype(np.float32)
            return self._current_focus_vector.copy()
        except Exception as e:
            logger.error(f"Attention.compute_attention error: {e}")
            return self._current_focus_vector.copy()



