# emotion.py

import numpy as np
from config import FLOATX
import time
import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

from nn import Sequential, Linear, Sigmoid, ReLU, Tanh
from config import (EMOTION_INPUT_DIM, EMOTION_OUTPUT_DIM, EMOTION_INTERNAL_DIM,
                   EMOTION_DECAY_RATE_GOOD, EMOTION_DECAY_RATE_BAD, EMOTION_DECAY_RATE_NEUTRAL,
                   EMOTION_PERSONA_OUTLIER_THRESHOLD, EMOTION_BLENDING_FACTOR, EMOTION_TEMPORAL_CONTEXT_SIZE)

# Bio system integration for emotional processing
try:
    from neurotransmitter import NeurotransmitterSystem
    from endocrine import EndocrineSystem
    BIO_SYSTEMS_AVAILABLE = True
except ImportError:
    # Bio systems not available - will use fallback processing
    BIO_SYSTEMS_AVAILABLE = False

# Single consolidated state file for all modules

class EmotionalState:
    """Represents a complex emotional state with multiple dimensions"""
    def __init__(self, base_dimension: int):
        self.dimension = base_dimension
        self.valence = np.zeros(base_dimension)  # Positive/negative
        self.arousal = np.zeros(base_dimension)  # Energy level
        self.dominance = np.zeros(base_dimension)  # Control/influence
        self.intensity = np.zeros(base_dimension)  # Strength of emotion
        self.temporal_context = []  # Historical emotional context
        
    def update(self, new_valence: np.ndarray, new_arousal: np.ndarray, 
              new_dominance: np.ndarray, new_intensity: np.ndarray):
        self.valence = new_valence
        self.arousal = new_arousal
        self.dominance = new_dominance
        self.intensity = new_intensity
        self.temporal_context.append((time.time(), self.get_unified_state()))
        if len(self.temporal_context) > EMOTION_TEMPORAL_CONTEXT_SIZE:  # Keep last N states
            self.temporal_context.pop(0)
            
    def get_unified_state(self) -> np.ndarray:
        """Combine all emotional dimensions into a unified representation"""
        base_state = np.stack([
            self.valence * self.intensity,
            self.arousal * self.intensity
        ]).mean(axis=0)
        
        # Bio integration - neurotransmitter modulation of emotional state
        if BIO_SYSTEMS_AVAILABLE and hasattr(self, '_neurotransmitters'):
            return self._neurotransmitters.modulate_emotional_state(base_state)
        return base_state
        
    def set_bio_systems(self, neurotransmitters=None, endocrine=None):
        """Link bio systems for emotional modulation"""
        if neurotransmitters:
            self._neurotransmitters = neurotransmitters
        if endocrine:
            self._endocrine = endocrine

class EmotionalModulation:
    """Handles how emotions modulate cognitive processes"""
    def __init__(self, emotion_dim: int, cognitive_dim: int):
        self.emotion_dimension = emotion_dim
        self.cognitive_dimension = cognitive_dim
        self.modulation_network = Sequential([
            Linear(emotion_dim, (emotion_dim + cognitive_dim) // 2),
            Tanh(),
            Linear((emotion_dim + cognitive_dim) // 2, cognitive_dim),
            Sigmoid()])

    def compute_modulation(self, emotional_state: np.ndarray) -> np.ndarray:
        """Compute cognitive modulation factors based on emotional state"""
        return self.modulation_network.forward(emotional_state)

    def get_emotional_influence(self, emotional_state: np.ndarray) -> float:
        """Calculate overall emotional influence strength"""
        return np.mean(np.abs(emotional_state))


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EmotionCore:
    """
    The Heart: Manages Lillith's 34,000-dimensional emotional state space.
    Allows for complex emotional combinations (10-12 simultaneous emotions like humans).
    Input: CAFVE tokens. Output: 34k dimensional emotional state vector.
    """
    
    # Define the base  emotions 
    # Ordered for consistency in indexing
    EMOTION_DIMENSIONS = [
        # Core Primary (7)
        #'Happy', 'Sad', 'Disgusted', 'Angry', 'Fearful', 'Surprised', 'Good',
        # Pleasure-Based / Good Spinoffs (13)'Euphoria', '
        'Warmth', 'Satiation', 'Contentment_Deep', 'Thrill', 'Admiration', 'Harmony', 'Prideful',
        # Other Wheel Spin-offs (88 unique from the exhaustive list provided previously)
        # These represent base emotions that can combine into 34k nuances in 32768D space.
        'Bad', 'Startled', 'Confused', 'Amazed', 'Excited', 'Playful', 'Content', 
        'Interested', 'Proud', 'Accepted', 'Powerful', 'Peaceful', 'Trusting', 'Optimistic', 
        'Loving', 'Thankful', 'Sensitive', 'Intimate', 'Hopeful', 'Inspired', 'Joyful', 
        'Curious', 'Inquisitive', 'Successful', 'Confident', 'Respected', 'Valued', 
        'Courageous', 'Creative', 'Energetic', 'Aroused', 'Cheeky', 'Free', 'Eager', 'Awe', 
        'Astonished', 'Perplexed', 'Overwhelmed', 'Out_of_control', 'Unfocused', 'Sleepy', 
        'Rushed', 'Pressured', 'Apathetic', 'Indifferent', 'Helpless', 'Frightened', 'Worried', 
        'Inadequate', 'Inferior', 'Worthless', 'Insignificant', 'Excluded', 'Persecuted', 
        'Nervous', 'Exposed', 'Betrayed', 'Resentful', 'Disrespected', 'Ridiculed', 
        'Indignant', 'Violated', 'Furious', 'Jealous', 'Provoked', 'Hostile', 'Infuriated', 
        'Annoyed', 'Withdrawn', 'Numb', 'Sceptical', 'Dismissive', 'Judgmental', 'Embarrassed', 
        'Appalled', 'Revolted', 'Nauseated', 'Detestable', 'Horrified', 'Hesitant', 
        'Lonely', 'Vulnerable', 'Despair', 'Guilty', 'Depressed', 'Hurt', 'Repelled', 'Awful'
    ]
    
    # Global mapping of emotion name to its index in the vector
    EMOTION_NAME_TO_INDEX = {name: i for i, name in enumerate(EMOTION_DIMENSIONS)}

    # Categorize emotions for differential decay rates
    # These sets are based on the base emotion list with combinatorial expansion
    POSITIVE_SET = {
        'Happy', 'Good', 'Euphoria', 'Eureka', 'Sexual_Climax', 'Ecstasy', 'Bliss', 
        'Arousal_Positive', 'Warmth', 'Satiation', 'Contentment_Deep', 'Thrill', 
        'Admiration', 'Harmony', 'Prideful', 'Playful', 'Content', 'Interested', 
        'Proud', 'Accepted', 'Powerful', 'Peaceful', 'Trusting', 'Optimistic', 
        'Loving', 'Thankful', 'Sensitive', 'Intimate', 'Hopeful', 'Inspired', 
        'Joyful', 'Curious', 'Inquisitive', 'Successful', 'Confident', 'Respected', 
        'Valued', 'Courageous', 'Creative', 'Energetic', 'Awe', 'Eager' # 42 emotions
    } 
    NEGATIVE_SET = {
        'Sad', 'Disgusted', 'Angry', 'Fearful', 'Lonely', 'Vulnerable', 'Despair', 
        'Guilty', 'Depressed', 'Hurt', 'Repelled', 'Awful', 'Disappointed', 'Disapproving', 
        'Critical', 'Distant', 'Frustrated', 'Aggressive', 'Mad', 'Bitter', 'Humiliated', 
        'Let down', 'Threatened', 'Rejected', 'Insecure', 'Weak', 'Anxious', 'Scared', 
        'Bored', 'Busy', 'Stressed', 'Tired', 'Overwhelmed', 'Out_of_control', 'Unfocused', 
        'Sleepy', 'Rushed', 'Pressured', 'Apathetic', 'Indifferent', 'Helpless', 
        'Frightened', 'Worried', 'Inadequate', 'Inferior', 'Worthless', 'Insignificant', 
       
        'Disrespected', 'Ridiculed', 'Indignant', 'Violated', 'Furious', 'Jealous', 
        'Provoked', 'Hostile', 'Infuriated', 'Annoyed', 'Withdrawn', 'Numb', 'Sceptical', 
        'Dismissive', 'Judgmental', 'Embarrassed', 'Appalled', 'Revolted', 'Nauseated', 
        'Detestable', 'Horrified', 'Hesitant' # 60 emotions
    } 
    # The remaining emotions are Neutral/Complex:
    # 'Bad', busy"Good",'Startled', 'Confused', 'Amazed', 'Excited', 'Perplexed'

    def __init__(self, input_dim: int = EMOTION_INPUT_DIM,
                 output_dim: int = EMOTION_OUTPUT_DIM,
                 internal_dim: int = EMOTION_INTERNAL_DIM,
                 decay_rate_good: float = EMOTION_DECAY_RATE_GOOD, # Slower decay for positive
                 decay_rate_bad: float = EMOTION_DECAY_RATE_BAD,  # Faster decay for negative
                 decay_rate_neutral: float = EMOTION_DECAY_RATE_NEUTRAL, # Moderate decay for neutral/complex
                 persona_outlier_threshold: float = EMOTION_PERSONA_OUTLIER_THRESHOLD, # Below this intensity, emotions don't impact persona
                 blending_factor: float = EMOTION_BLENDING_FACTOR): # How much new input influences current state
        
        self.output_dim = output_dim  # Use dynamic dimension instead of hardcoded 108
        self.input_dim = input_dim # CAFVE token input
        
        # Current emotional state (intensities for each dimension, values 0-1)
        self.emotional_state = np.zeros(self.output_dim, dtype=FLOATX)
        # Timestamp of last update for each emotion, for individual decay calculation
        self.last_update_times = np.full(self.output_dim, time.perf_counter(), dtype=FLOATX)

        # Map each emotion index to its decay type ('POSITIVE', 'NEGATIVE', 'NEUTRAL_COMPLEX')
        self._emotion_types = {}
        for i, emotion_name in enumerate(self.EMOTION_DIMENSIONS):
            if emotion_name in self.POSITIVE_SET:
                self._emotion_types[i] = 'POSITIVE'
            elif emotion_name in self.NEGATIVE_SET:
                self._emotion_types[i] = 'NEGATIVE'
            else: # Anything not explicitly positive or negative from the sets
                self._emotion_types[i] = 'NEUTRAL_COMPLEX'
        # For any additional emergent dimensions beyond the named list, default to NEUTRAL_COMPLEX
        if self.output_dim > len(self.EMOTION_DIMENSIONS):
            for i in range(len(self.EMOTION_DIMENSIONS), self.output_dim):
                self._emotion_types[i] = 'NEUTRAL_COMPLEX'
        
        self.decay_rates = {
            'POSITIVE': FLOATX(decay_rate_good),
            'NEGATIVE': FLOATX(decay_rate_bad),
            'NEUTRAL_COMPLEX': FLOATX(decay_rate_neutral)
        }
        self.persona_outlier_threshold = FLOATX(persona_outlier_threshold)
        self.blending_factor = FLOATX(blending_factor)

        # Neural network to map input (CAFVE tokens) to 34k dimensional emotional state
        # Uses Sequential, Linear, ReLU, Sigmoid from nn.py
        
        self.emotional_mapper = Sequential([
            Linear(self.input_dim, internal_dim),
            ReLU(),
            Linear(internal_dim, internal_dim),
            ReLU(),
            Linear(internal_dim, self.output_dim),
            Sigmoid()])
        
        # COMPRESSION LAYER: Reduce 34000D emotional space → 512D for Mind integration
        # This preserves emotional richness while fitting downstream architecture
        from config import MIND_EMOTIONAL_STATE_DIM
        self.emotion_compression_dim = MIND_EMOTIONAL_STATE_DIM  # 512D
        self.emotion_compressor = Sequential([
            Linear(self.output_dim, 2048),
            ReLU(),
            Linear(2048, 1024),
            ReLU(),
            Linear(1024, self.emotion_compression_dim),
            Tanh()
        ]) if self.output_dim != self.emotion_compression_dim else None
        
        logger.info(f"EmotionCore initialized with {self.output_dim} dimensions. Input_dim: {self.input_dim}")

        # Track a combined dimension attribute for downstream dimension audits
        self.combined_dim = self.output_dim

    def _apply_temporal_decay(self):
        """Applies temporal decay to each emotional dimension based on its category."""
        current_time = time.perf_counter()
        
        for i in range(self.output_dim):
            # Calculate time elapsed since last update for this specific emotion
            time_elapsed_for_this_emotion = current_time - self.last_update_times[i]
            
            decay_rate = self.decay_rates[self._emotion_types[i]]
            
            # Apply decay only if intensity is positive
            if self.emotional_state[i] > 0.0:
                # decay_factor = exp(-rate * time_elapsed)
                decay_factor = np.exp(-decay_rate * time_elapsed_for_this_emotion)
                self.emotional_state[i] = self.emotional_state[i] * decay_factor
                # Ensure intensity doesn't go below zero due to float precision
                self.emotional_state[i] = np.clip(self.emotional_state[i], 0.0, 1.0)
            
            # Update last_update_time for this specific emotion
            self.last_update_times[i] = current_time

    def process_input(self, cafve_token_batch: List[np.ndarray]) -> np.ndarray:
        """
        Processes a batch of CAFVE tokens, updates emotional state, and applies decay.
        Returns the current 108-dimensional emotional state vector.
        """
        # Apply decay based on time elapsed since the last full processing cycle
        self._apply_temporal_decay() 

        if not cafve_token_batch:
            # If no new input, just return the decayed state
            return self.emotional_state.copy()
        # Concatenate tokens for batch processing by the emotional_mapper NN
        # Ensure batch_size x input_dim
        batch_input = np.array(cafve_token_batch, dtype=FLOATX)

        # Ensure shape is (batch, input_dim)
        if batch_input.ndim == 1:
            batch_input = batch_input.reshape(1, -1)

        # Fix mismatched input dim by padding/truncating columns
        if batch_input.shape[1] != self.input_dim:
            tmp = np.zeros((batch_input.shape[0], self.input_dim), dtype=FLOATX)
            n = min(batch_input.shape[1], self.input_dim)
            tmp[:, :n] = batch_input[:, :n]
            batch_input = tmp

        # Predict new emotional intensities based on input
        predicted_intensities_batch = self.emotional_mapper.forward(batch_input)

        # Aggregate predictions from the batch: average is a common strategy
        new_emotion_input_impulse = np.mean(predicted_intensities_batch, axis=0)

        # Update current emotional state by blending new input with existing state
        self.emotional_state = (self.emotional_state * (1 - self.blending_factor) +
                    new_emotion_input_impulse * self.blending_factor)

        # Ensure state is within valid bounds (0 to 1 intensity) after blending
        self.emotional_state = np.clip(self.emotional_state, 0.0, 1.0)

        return self.emotional_state.copy()

    # --- Backward Compatibility Wrapper ---
    def process_emotions(self, cognitive_state: np.ndarray) -> np.ndarray:
        """Legacy interface expected by main.py.

        Since the new design maps CAFVE tokens to emotion, if only a cognitive_state
        is provided we derive a lightweight pseudo-token via deterministic projection.
        This avoids mock randomness while preserving structural mapping.
        """
        try:
            if not hasattr(self, '_legacy_proj'):
                rng = np.random.default_rng(seed=123)
                self._legacy_proj = rng.standard_normal((cognitive_state.shape[0], self.input_dim)).astype(FLOATX) * (FLOATX(1.0) / np.sqrt(cognitive_state.shape[0]))
            pseudo_token = (cognitive_state.astype(FLOATX) @ self._legacy_proj).astype(FLOATX)
            # Clamp to reasonable range then sigmoid-like squashing
            pseudo_token = np.tanh(pseudo_token)
            # Use existing pipeline expecting a batch
            return self.process_input([pseudo_token])
        except Exception as e:
            logger.error(f"EmotionCore.process_emotions error: {e}")
            return self.emotional_state.copy()

    def get_emotional_state(self) -> np.ndarray:
        """Returns the current, up-to-date emotional state vector."""
        # Always apply decay to ensure the returned state is current
        self._apply_temporal_decay() 
        return self.emotional_state.copy()

    def get_emotion_vector(self) -> np.ndarray:
        """Returns the current emotional state vector for Mind integration (compressed to 512D)."""
        full_emotional_state = self.get_emotional_state()
        # Compress 34000D → 512D for downstream integration
        if self.emotion_compressor is not None:
            return self.emotion_compressor.forward(full_emotional_state.reshape(1, -1)).flatten()
        return full_emotional_state

    def get_full_emotional_state(self) -> np.ndarray:
        """Returns the full 34000-dimensional emotional state (uncompressed)."""
        self._apply_temporal_decay()
        return self.emotional_state.copy()

    def get_persona_emotional_output(self) -> np.ndarray:
        """
        Returns a filtered emotional state for persona impact, ignoring low-value outliers.
        Higher-level modules like the Manifold/Mind will use this filtered output.
        """
        current_state = self.get_emotional_state() # Get up-to-date state
        
        # Create a copy and zero out emotions below the outlier threshold
        filtered_state = current_state.copy()
        filtered_state[filtered_state < self.persona_outlier_threshold] = 0.0
        
        return filtered_state

# >>> EMBEDDED_EMOTION_WEIGHTS
EMBEDDED_EMOTION_WEIGHTS = {
    'initialized': False,
    'mapper_weights': [],
    'compressor_weights': []
}
# <<< EMBEDDED_EMOTION_WEIGHTS

def save_embedded_emotion_weights(emotion_core: EmotionCore) -> bool:
    """Save emotion mapper and compressor weights to embedded dict."""
    global EMBEDDED_EMOTION_WEIGHTS
    try:
        mapper_weights = []
        for layer in emotion_core.emotional_mapper.layers:
            if hasattr(layer, 'weights') and hasattr(layer, 'biases'):
                mapper_weights.append({
                    'weights': layer.weights.tolist(),
                    'biases': layer.biases.tolist()
                })
        
        compressor_weights = []
        if emotion_core.emotion_compressor is not None:
            for layer in emotion_core.emotion_compressor.layers:
                if hasattr(layer, 'weights') and hasattr(layer, 'biases'):
                    compressor_weights.append({
                        'weights': layer.weights.tolist(),
                        'biases': layer.biases.tolist()
                    })
        
        EMBEDDED_EMOTION_WEIGHTS = {
            'initialized': True,
            'mapper_weights': mapper_weights,
            'compressor_weights': compressor_weights
        }
        return True
    except Exception as e:
        logger.error(f"Error saving embedded emotion weights: {e}")
        return False

def load_embedded_emotion_weights(emotion_core: EmotionCore) -> bool:
    """Load emotion mapper and compressor weights from embedded dict."""
    global EMBEDDED_EMOTION_WEIGHTS
    try:
        if not EMBEDDED_EMOTION_WEIGHTS.get('initialized', False):
            return False
        
        # Restore mapper weights
        for i, layer_weights in enumerate(EMBEDDED_EMOTION_WEIGHTS['mapper_weights']):
            if i < len(emotion_core.emotional_mapper.layers):
                layer = emotion_core.emotional_mapper.layers[i]
                if hasattr(layer, 'weights'):
                    layer.weights = np.array(layer_weights['weights'], dtype=FLOATX)
                if hasattr(layer, 'biases'):
                    layer.biases = np.array(layer_weights['biases'], dtype=FLOATX)
        
        # Restore compressor weights
        if emotion_core.emotion_compressor is not None:
            for i, layer_weights in enumerate(EMBEDDED_EMOTION_WEIGHTS['compressor_weights']):
                if i < len(emotion_core.emotion_compressor.layers):
                    layer = emotion_core.emotion_compressor.layers[i]
                    if hasattr(layer, 'weights'):
                        layer.weights = np.array(layer_weights['weights'], dtype=FLOATX)
                    if hasattr(layer, 'biases'):
                        layer.biases = np.array(layer_weights['biases'], dtype=FLOATX)
        
        return True
    except Exception as e:
        logger.error(f"Error loading embedded emotion weights: {e}")
        return False
    def get_emotion_names(self) -> List[str]:
        """Returns the list of base emotion names (expanded to 34k dimensional space)."""
        return self.EMOTION_DIMENSIONS

    def get_emotion_index(self, emotion_name: str) -> Optional[int]:
        """Returns the index of an emotion name."""
        return self.EMOTION_NAME_TO_INDEX.get(emotion_name)

