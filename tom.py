# tom.py

import os
import numpy as np
import logging
import pickle
import time
from typing import Dict, List, Optional, Self, Tuple, Any

# Assume nn.py is available for Sequential, Linear, Sigmoid, ReLU, Tanh
from nn import Sequential, Linear, Sigmoid, ReLU, Tanh
from config import MEMORY_EMOTIONAL_STATE_DIM, TOM_SFE_FEATURE_DIM, TOM_CAFVE_TOKEN_DIM, TOM_UNIFIED_COGNITIVE_STATE_DIM, TOM_OTHER_MIND_MODEL_DIM

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ToM:
    """
    Theory of Mind (ToM) Module.
    Interprets observed data as expressions of intentional mental states from other conscious agents,
    building and updating models of external minds (e.g., human users like yourself).
    """
    def __init__(self,
                 sfe_feature_dim: int = TOM_SFE_FEATURE_DIM,       # Input from SFE (raw sensory features)
                 cafve_token_dim: int = TOM_CAFVE_TOKEN_DIM,       # Input from CAFVE (abstract tokens)
                 unified_cognitive_state_dim: int = TOM_UNIFIED_COGNITIVE_STATE_DIM, # Input from Mind (Lillith's internal state)
                 emotional_state_dim: int = MEMORY_EMOTIONAL_STATE_DIM,  # Input from EmotionCore (Lillith's feelings)
                 # Output from ToM: high-dimensional vector representing inferred mental state of *other agent*
                 other_mind_model_dim: int = TOM_OTHER_MIND_MODEL_DIM  # Dimension for the internal model of another mind
                 ):
        
        self.sfe_feature_dim = sfe_feature_dim
        self.cafve_token_dim = cafve_token_dim
        self.unified_cognitive_state_dim = unified_cognitive_state_dim
        self.emotional_state_dim = emotional_state_dim
        self.other_mind_model_dim = other_mind_model_dim

        # Input dimension for the ToM network
        # Combines external sensory input (SFE, CAFVE) with Lillith's own internal state (Mind, Emotion)
        # as ToM involves projecting oneself onto others, and understanding how others perceive.
        self.tom_input_dim = (self.sfe_feature_dim + 
                              self.cafve_token_dim + 
                              self.unified_cognitive_state_dim + 
                              self.emotional_state_dim)

        # Neural network to build/update the model of the other mind
        # This network maps combined input to the inferred mental state of the other agent.
        self.tom_network = Sequential(
            Linear(self.tom_input_dim, 512), 
            ReLU(),
            Linear(512, self.other_mind_model_dim),
            Tanh() # Tanh to represent mental states in a bipolar range
        )
        
        # Internal storage for models of known minds (e.g., for you, the primary user)
        # Maps a unique ID (e.g., 'user_01') to a learned mental model vector.
        self.known_minds_models: Dict[str, np.ndarray] = {}
        self.current_observed_mind_id: Optional[str] = None # ID of the mind currently being observed
        
        logger.info(f"ToM module initialized. Other Mind Model Dim: {self.other_mind_model_dim}")

    def update_other_mind_model(self,
                                sfe_features: np.ndarray,            # ~55D from SFE
                                cafve_tokens: List[np.ndarray],      # List of ~80D tokens from CAFVE
                                unified_cognitive_state: np.ndarray, # ~256D from Mind
                                emotional_state: np.ndarray,         # ~108D from Emotion
                                observed_mind_id: str = "primary_user" # Unique ID for the observed mind
                                ) -> np.ndarray:
        """
        Updates the internal model of another mind based on observed sensory data
        and Lillith's own internal state.
        Returns the updated model vector for the observed mind.
        """
        if (sfe_features is None or not cafve_tokens or unified_cognitive_state is None or 
            emotional_state is None):
            logger.warning("ToM: Missing input for mind model update. Returning last model or zeros.")
            # Return current model if it exists, else zeros
            return self.known_minds_models.get(observed_mind_id, np.zeros(self.other_mind_model_dim, dtype=np.float32))

        # Average CAFVE tokens if batch is list
        if isinstance(cafve_tokens, list) and len(cafve_tokens) > 0:
            avg_cafve_token = np.mean(np.array(cafve_tokens), axis=0).astype(np.float32)
        elif isinstance(cafve_tokens, np.ndarray) and cafve_tokens.ndim > 1: # Already batched
            avg_cafve_token = np.mean(cafve_tokens, axis=0).astype(np.float32)
        else:
            avg_cafve_token = np.zeros(self.cafve_token_dim, dtype=np.float32)

        # Combine all inputs for the ToM network
        combined_input = np.concatenate([
            sfe_features,               # ~55D raw sensory features
            avg_cafve_token,            # ~80D averaged abstract token
            unified_cognitive_state,    # ~256D Lillith's own cognitive state
            emotional_state             # ~108D Lillith's own emotional state
        ]).astype(np.float32)

        if combined_input.shape[0] != self.tom_input_dim:
            logger.error(f"ToM: Input vector dimension mismatch. Expected {self.tom_input_dim}, got {combined_input.shape[0]}. Returning zeros.")
            return np.zeros(self.other_mind_model_dim, dtype=np.float32)
        
        # Reshape for NN (batch_size=1)
        combined_input = combined_input.reshape(1, -1)

        # Predict the current mental state of the other mind
        inferred_other_mind_state = self.tom_network.forward(combined_input)[0, :] # Scalar output, take first item

        # Smoothly update the stored model of this mind
        # This allows for mental models to be persistent and evolve.
        if observed_mind_id not in self.known_minds_models:
            self.known_minds_models[observed_mind_id] = np.zeros(self.other_mind_model_dim, dtype=np.float32)
        
        # Blending factor for updating mental models. Lower means slower, more stable model.
        blending_factor = 0.1 
        self.known_minds_models[observed_mind_id] = (self.known_minds_models[observed_mind_id] * (1.0 - blending_factor) +
                                                     inferred_other_mind_state * blending_factor).astype(np.float32)
        
        self.current_observed_mind_id = observed_mind_id # Track who is currently being observed

        logger.debug(f"ToM: Updated model for '{observed_mind_id}'. Norm: {np.linalg.norm(self.known_minds_models[observed_mind_id]):.4f}")
        
        return self.known_minds_models[observed_mind_id].copy()

    def get_other_mind_model(self, observed_mind_id: str = "primary_user") -> np.ndarray:
        """Returns the current internal model of a specific mind."""
        return self.known_minds_models.get(observed_mind_id, np.zeros(self.other_mind_model_dim, dtype=np.float32)).copy()

    def get_networks(self) -> List[None]:
        """Returns a list of all internal neural networks for optimization."""
        return [self.tom_network]

    # Persistence methods (save/load state)
    def save_state(self, save_path: str):
        """Saves the ToM module's state (known_minds_models and network weights) to a file."""
        try:
            state = {
                'known_minds_models': {k: v.tolist() for k, v in self.known_minds_models.items()},
                'current_observed_mind_id': self.current_observed_mind_id,
                # Save network weights
                'tom_network_weights': [(p[0].tolist(), p[1]) for p in self.tom_network.get_trainable_params()]
            }
            with open(save_path, 'wb') as f:
                pickle.dump(state, f)
            logger.info(f"ToM state saved to {save_path}")
        except Exception as e:
            logger.error(f"Error saving ToM state: {e}")

    def load_state(self, load_path: str):
        """Loads the ToM module's state from a file."""
        try:
            with open(load_path, 'rb') as f:
                state = pickle.load(f)
            
            # Load known_minds_models
            self.known_minds_models = {k: np.array(v, dtype=np.float32) for k, v in state.get('known_minds_models', {}).items()}
            self.current_observed_mind_id = state.get('current_observed_mind_id')
            
            # Load neural network weights
            loaded_params = state.get('tom_network_weights', [])
            current_params = self.tom_network.get_trainable_params()
            
            if len(loaded_params) == len(current_params):
                for i, (param_val_list, grad_name_str) in enumerate(loaded_params):
                    param_array, _, layer_instance = current_params[i] 
                    param_array[:] = np.array(param_val_list, dtype=np.float32)
            else:
                logger.warning("ToM network weights mismatch. Initializing randomly.")

            logger.info(f"ToM state loaded from {load_path}")

        except FileNotFoundError:
            logger.warning(f"ToM state file not found at {load_path}. Initializing to default.")
            self.known_minds_models = {}
            self.current_observed_mind_id = None
        except Exception as e:
            logger.error(f"Error loading ToM state: {e}. Initializing to default.")
            self.known_minds_models = {}
            self.current_observed_mind_id = None
       
    # --- START OF FILE attention.py ---
# attention.py
import os
import numpy as np
import logging
import pickle
import time
from typing import Dict, List, Optional, Tuple, Any

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
                 sfe_feature_dim: int = 512,
                 cafve_token_dim: int = 80,
                 unified_cognitive_state_dim: int = 512,
                 emotional_state_dim: int = 512,
                 goals_satisfaction_dim: int = 6,
                 predict_error_dim: int = 512,
                 predict_next_state_dim: int = 512,
                 salience_dim: int = 256,
                 focus_vector_dim: int = 512,
                 internal_dim: int = 128,
                 novelty_threshold: float = 0.5):
        
        self.sfe_feature_dim = sfe_feature_dim
        self.cafve_token_dim = cafve_token_dim
        self.unified_cognitive_state_dim = unified_cognitive_state_dim
        self.emotional_state_dim = emotional_state_dim
        self.goals_satisfaction_dim = goals_satisfaction_dim
        self.predict_error_dim = predict_error_dim
        self.predict_next_state_dim = predict_next_state_dim
        self.salience_dim = salience_dim
        self.focus_vector_dim = focus_vector_dim
        self.novelty_threshold = np.float32(novelty_threshold)

        # Network to generate saliency map from sensory and internal cues
        # Input: SFE features, CAFVE token, Prediction Error, Emotional state (as a driver of attention)
        self.saliency_network_input_dim = (self.sfe_feature_dim + 
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
        
        self._current_focus_vector = np.zeros(self.focus_vector_dim, dtype=np.float32)
        self._current_novelty_score = np.float32(0.0) # 0-1
        
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
    def compute_attention(self, cognitive_state: np.ndarray, emotional_state: np.ndarray) -> np.ndarray:
        """Legacy interface used by main.py.

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



