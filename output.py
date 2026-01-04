# --- START OF FILE output.py ---
# output.py

import os
import sys
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True

import numpy as np
import logging
import pickle
from typing import Dict, List, Optional, Tuple, Any
from config import FLOATX, SOM_MAP_ROWS, SOM_MAP_COLS

# Directly import neural network layers
from nn import Sequential, Linear, Sigmoid, ReLU, Tanh
from config import (OUTPUT_INTERNAL_LANG_DIM, OUTPUT_VOCAL_SYNTH_DIM, 
                    OUTPUT_ATTENTION_FOCUS_DIM, OUTPUT_INTERNAL_DIM)

# Bio system integration for realistic vocal synthesis
try:
    from physical import PhysicalConstraints
    from fatigue import FatigueSystem
    BIO_VOCAL_AVAILABLE = True
except ImportError:
    BIO_VOCAL_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Output:
    """
    Lillith's Vocal Expression System.
    Translates Lillith's unified cognitive state, emotional state, and internal language
    into precise control parameters for VocalSynth.
    """
    # Define the precise list of VocalSynth parameters (from vocalsynth.py)
    VOCAL_SYNTH_PARAMS = [
        'F0_hz', 'Vibrato_Rate', 'Vibrato_Extent',           # Pitch & Timing (3)
        'Amplitude', 'Voicing_Mix', 'Spectral_Tilt', 'Breathiness_Amount', # Loudness & Voice Quality (4)
        'F1_hz', 'F2_hz', 'F3_hz', 'F4_hz',                  # Formant Frequencies (4)
        'BW1_hz', 'BW2_hz', 'BW3_hz', 'BW4_hz',              # Formant Bandwidths (4)
        'Tension_Param', 'Roughness_Param'                   # Dynamic Expression (2)
    ] # Total 17 parameters

    # Vocal synthesis only - motor outputs removed

    def __init__(self,
                 unified_cognitive_state_dim: int,
                 emotional_state_dim: int,
                 internal_lang_dim: int = OUTPUT_INTERNAL_LANG_DIM,
                 vocal_synth_params_dim: int = OUTPUT_VOCAL_SYNTH_DIM,
                 attention_focus_dim: int = OUTPUT_ATTENTION_FOCUS_DIM,
                 internal_dim: int = OUTPUT_INTERNAL_DIM,
                 **kwargs):
        # Required / external interface dimensions
        self.unified_dim = int(unified_cognitive_state_dim)
        self.emotion_dim = int(emotional_state_dim)
        # Standardized large external feature spaces
        self.internal_lang_dim = int(internal_lang_dim)
        self.vocal_synth_params_dim = int(vocal_synth_params_dim)
        self.focus_vector_dim = int(attention_focus_dim)
        # Compact internal processing width
        self.internal_hidden_dim = int(internal_dim)
        
        # Call update_dimensions to set up derived attributes
        self.update_dimensions(
            unified_cognitive_state_dim=unified_cognitive_state_dim,
            emotional_state_dim=emotional_state_dim,
            internal_lang_dim=internal_lang_dim,
            vocal_synth_params_dim=vocal_synth_params_dim,
            attention_focus_dim=attention_focus_dim,
            internal_hidden_dim=self.internal_hidden_dim,
            **kwargs
        )

        # Bio system integration for vocal realism
        if BIO_VOCAL_AVAILABLE:
            self.physical_constraints = PhysicalConstraints()
            self.fatigue_system = FatigueSystem(shape=(SOM_MAP_ROWS, SOM_MAP_COLS))
            logger.info("Bio vocal systems initialized")
        else:
            self.physical_constraints = None
            self.fatigue_system = None

        # Initialize last params and confidence
        self._last_generated_params = {param: FLOATX(0.0) for param in self.VOCAL_SYNTH_PARAMS}
        self._output_confidence = FLOATX(0.0) # 0-1 confidence score

        logger.info(f"Output module ready. Total Control Params Dim: {self.total_output_params_dim}")

    def update_dimensions(self,
                         unified_cognitive_state_dim: int = None,
                         emotional_state_dim: int = None,
                         internal_lang_dim: int = None,
                         vocal_synth_params_dim: int = None,
                         attention_focus_dim: int = None,
                         internal_dim: int = None,
                         goals_dim: int = None,
                         **kwargs):
        """Dynamically update output dimensions."""
        if unified_cognitive_state_dim is not None:
            self.unified_dim = int(unified_cognitive_state_dim)
        if emotional_state_dim is not None:
            self.emotion_dim = int(emotional_state_dim)
        if internal_lang_dim is not None:
            self.internal_lang_dim = int(internal_lang_dim)
        if vocal_synth_params_dim is not None:
            self.vocal_synth_params_dim = int(vocal_synth_params_dim)
        if attention_focus_dim is not None:
            self.focus_vector_dim = int(attention_focus_dim)
        if internal_dim is not None:
            self.internal_hidden_dim = int(internal_dim)
        # Determine goals_dim early so subsequent total_input_dim calculations can use it
        if goals_dim is not None:
            self.goals_dim = int(goals_dim)
        else:
            # fallback to kwargs or existing value or default 6
            self.goals_dim = int(kwargs.get('goals_dim', getattr(self, 'goals_dim', 6)))
        # Recompute total input/output dims
        self.total_output_params_dim = len(self.VOCAL_SYNTH_PARAMS)
        self.total_input_dim = (
            self.unified_dim +
            self.emotion_dim +
            self.focus_vector_dim +
            self.internal_lang_dim +
            self.vocal_synth_params_dim +
            self.goals_dim
        )

    # (goals_dim already set above)

        # Track internal language activity dim separately for clarity proxy calc
        self.internal_lang_activity_dim = self.internal_lang_dim

        # Output control param dimensionality (vocal only)
        self.total_output_params_dim = len(self.VOCAL_SYNTH_PARAMS)

        # Aggregate input dimension to mapper network
        # Compute total input as sum of externally provided vectors actually used in generate_output path
        self.total_input_dim = (
            self.unified_dim +          # cognitive state
            self.emotion_dim +          # emotional state
            self.focus_vector_dim +     # attention focus
            self.internal_lang_dim +    # internal language activity
            self.vocal_synth_params_dim + # reserved future coupling (zeros now)
            self.goals_dim              # goals satisfaction
        )

        # (optional) log for diagnostics
        # logger.info(f"Output dims: unified={self.unified_dim}, emotion={self.emotion_dim}, "
        #             f"focus={self.focus_vector_dim}, lang={self.internal_lang_dim}, synth={self.vocal_synth_params_dim}, "
        #             f"total_in={self.total_input_dim}")

        # Expressive Mapping Network
        # This network maps Lillith's complex internal state to VocalSynth and UI control parameters
        try:
            self.expressive_mapper = Sequential(
                Linear(self.total_input_dim, self.internal_hidden_dim),
                ReLU(),
                Linear(self.internal_hidden_dim, self.internal_hidden_dim),
                ReLU(),
                Linear(self.internal_hidden_dim, self.total_output_params_dim),
                Tanh()
            )
        except Exception as e:
            logger.error(f"Output: Failed to init expressive mapper with total_input_dim={self.total_input_dim}: {e}")
            # Fallback minimal network
            self.expressive_mapper = Sequential(
                Linear(max(1,self.total_input_dim), self.total_output_params_dim),
                Tanh()
            )

        # Initialize last params and confidence
        self._last_generated_params = {param: np.float16(0.0) for param in self.VOCAL_SYNTH_PARAMS}
        self._output_confidence = np.float16(0.0) # 0-1 confidence score

        logger.info(f"Output module ready. Total Control Params Dim: {self.total_output_params_dim}")

    def generate_controls(self,
                          unified_cognitive_state: np.ndarray,      # 1024D float64 from Mind.py
                          emotional_state_filtered: np.ndarray,     # 1024D from Emotion.py
                          internal_lang_activity: np.ndarray,       # 512D from Language.py
                          focus_vector: np.ndarray,                 # 256D from Attention.py
                          goals_satisfaction: np.ndarray,           # 6D from Goals.py
                          vocal_synth_embedding: Optional[np.ndarray] = None, # Optional vocal embedding
                          ) -> Dict[str, Any]:
        """
        Translates Lillith's internal state into VocalSynth and UI control parameters.
        Returns a dictionary of all control parameters.
        """
        # Prepare / validate each component to expected dims
        def _fit(vec, dim):
            v = np.array(vec, dtype=np.float16).reshape(-1)
            if v.shape[0] == dim:
                return v
            if v.shape[0] > dim:
                return v[:dim]
            out = np.zeros(dim, dtype=np.float16)
            out[:v.shape[0]] = v
            return out

        unified_cognitive_state = _fit(unified_cognitive_state, self.unified_dim)
        emotional_state_filtered = _fit(emotional_state_filtered, self.emotion_dim)
        focus_vector = _fit(focus_vector, self.focus_vector_dim)
        internal_lang_activity = _fit(internal_lang_activity, self.internal_lang_dim)
        if vocal_synth_embedding is None:
            vocal_synth_embedding = np.zeros(self.vocal_synth_params_dim, dtype=FLOATX)
        else:
            vocal_synth_embedding = _fit(vocal_synth_embedding, self.vocal_synth_params_dim)
        goals_satisfaction = _fit(goals_satisfaction, self.goals_dim)

        # Order MUST match construction of self.total_input_dim
        combined_input = np.concatenate([
            unified_cognitive_state,
            emotional_state_filtered,
            focus_vector,
            internal_lang_activity,
            vocal_synth_embedding,
            goals_satisfaction
    ]).astype(np.float16)

        if combined_input.shape[0] != self.total_input_dim:
            logger.error(f"Output: Input dimension mismatch. Expected {self.total_input_dim}, got {combined_input.shape[0]}. Returning default controls.")
            return {param: np.float16(0.0) for param in self.VOCAL_SYNTH_PARAMS}
        
        # Reshape for NN (batch_size=1)
        combined_input = combined_input.reshape(1, -1)

        # Generate raw output parameters from the network
        # Bio integration: Apply physical constraints and fatigue effects to vocal synthesis
        if self.physical_constraints and self.fatigue_system:
            # Apply biological constraints to vocal parameters before generation
            bio_modulated_input = self.fatigue_system.apply_vocal_fatigue(combined_input)
            raw_output_params = self.expressive_mapper.forward(bio_modulated_input)[0,:]
        else:
            # Fallback to standard processing if bio systems not available
            raw_output_params = self.expressive_mapper.forward(combined_input)[0,:]

        # --- Map raw network output (-1 to 1) to specific control ranges ---
        control_dict = {}
        
        # VocalSynth Parameters
        for i, param_name in enumerate(self.VOCAL_SYNTH_PARAMS):
            param_value = np.float16(raw_output_params[i])

            if param_name == 'F0_hz': param_value = (param_value + 1.0) * 125.0 + 75.0 # Map -1 to 1 to 75-325 Hz
            elif param_name == 'Amplitude': param_value = (param_value + 1.0) / 2.0 # Map -1 to 1 to 0-1
            elif param_name == 'Voicing_Mix': param_value = (param_value + 1.0) / 2.0
            elif param_name == 'Spectral_Tilt': param_value = param_value
            elif param_name == 'Breathiness_Amount': param_value = (param_value + 1.0) / 2.0
            elif param_name.startswith('F') and param_name.endswith('_hz'): # Formants
                param_value = (param_value + 1.0) / 2.0 * 2000.0 + 200.0
            elif param_name.startswith('BW') and param_name.endswith('_hz'): # Bandwidths
                param_value = (param_value + 1.0) / 2.0 * 200.0 + 30.0
            elif param_name.endswith('Param'): # Tension, Roughness
                param_value = param_value
            elif param_name == 'Vibrato_Rate': param_value = (param_value + 1.0) / 2.0 * 10.0
            elif param_name == 'Vibrato_Extent': param_value = (param_value + 1.0) / 2.0 * 10.0
            
            # Apply physical constraints to individual vocal parameters
            if self.physical_constraints:
                param_value = self.physical_constraints.constrain_vocal_parameter(param_name, param_value)
            
            control_dict[param_name] = param_value

        self._last_generated_params = control_dict.copy()

        # Motor outputs removed - vocal synthesis only

        # --- Calculate Output Confidence with Bio Integration ---
        language_clarity_proxy = np.linalg.norm(internal_lang_activity).astype(np.float16) / np.sqrt(self.internal_lang_activity_dim)
        goals_satisfaction_avg = np.mean(goals_satisfaction).astype(np.float16)
        
        # Bio integration: Factor in fatigue and physical constraints
        bio_confidence_factor = 1.0
        if self.fatigue_system:
            bio_confidence_factor *= self.fatigue_system.get_vocal_efficiency()
        if self.physical_constraints:
            bio_confidence_factor *= self.physical_constraints.get_system_health()

        self._output_confidence = np.clip((language_clarity_proxy * 0.7 + goals_satisfaction_avg * 0.3) * bio_confidence_factor, 0.0, 1.0)

        logger.debug(f"Output generated controls. Conf: {self._output_confidence:.4f}")
        return control_dict

    # --- Backward Compatibility ---
    def generate_output(self, unified_cognitive_state: np.ndarray, emotional_state: np.ndarray, focus_vector: np.ndarray) -> Dict[str, Any]:
        """Legacy interface expected by main.py. Routes to generate_controls with safe defaults.
        #HOW MANY TIMES I GOT TO TELL YOU THERE IS NOT A LEGACY SYSTEM IMPLEMENTED   MAIN IS GONE FOR EVER ITS RUN NOW
        Since main.py isn't yet passing internal language activity or goals satisfaction
        into this call, we approximate:
            - internal_lang_activity: zeros if unavailable
            - goals_satisfaction: zeros vector of length goals_dim
        This preserves deterministic behavior without mock semantics.
        """
        internal_lang_activity = np.zeros(self.internal_lang_activity_dim, dtype=np.float16)
        goals_satisfaction = np.zeros(self.goals_dim, dtype=np.float16)
        # Ensure input pieces have correct dimensions; if not, pad/truncate.
        def _fit(vec, dim):
            v = np.array(vec, dtype=np.float16).reshape(-1)
            if v.shape[0] == dim:
                return v
            if v.shape[0] > dim:
                return v[:dim]
            return np.pad(v, (0, dim - v.shape[0]))

        unified_cognitive_state = _fit(unified_cognitive_state, self.unified_dim)
        emotional_state = _fit(emotional_state, self.emotion_dim)
        focus_vector = _fit(focus_vector, self.focus_vector_dim)
        internal_lang_activity = _fit(internal_lang_activity, self.internal_lang_activity_dim)
        # vocal_synth_params_dim: reserved future coupling; supply zeros
        vocal_embedding = np.zeros(self.vocal_synth_params_dim, dtype=FLOATX)
        goals_satisfaction = _fit(goals_satisfaction, self.goals_dim)

        combined_input = np.concatenate([
            unified_cognitive_state,
            emotional_state,
            focus_vector,
            internal_lang_activity,
            vocal_embedding,
            goals_satisfaction
        ])

        if combined_input.shape[0] != self.total_input_dim:
            expected_breakdown = [
                ('unified', self.unified_dim),
                ('emotion', self.emotion_dim),
                ('focus', self.focus_vector_dim),
                ('internal_lang', self.internal_lang_dim),
                ('vocal_embedding', self.vocal_synth_params_dim),
                ('goals', self.goals_dim)
            ]
            actual_breakdown = [
                ('unified', unified_cognitive_state.shape[0]),
                ('emotion', emotional_state.shape[0]),
                ('focus', focus_vector.shape[0]),
                ('internal_lang', internal_lang_activity.shape[0]),
                ('vocal_embedding', vocal_embedding.shape[0]),
                ('goals', goals_satisfaction.shape[0])
            ]
            logger.warning(
                f"Output.generate_output: dim mismatch total {combined_input.shape[0]} expected {self.total_input_dim} "
                f"expected_breakdown={expected_breakdown} actual_breakdown={actual_breakdown}")
            if combined_input.shape[0] < self.total_input_dim:
                pad_len = self.total_input_dim - combined_input.shape[0]
                combined_input = np.concatenate([combined_input, np.zeros(pad_len, dtype=np.float16)])
            else:
                combined_input = combined_input[:self.total_input_dim]
        # Reuse generate_controls path with bio-integrated processing
        try:
            return self.generate_controls(
                unified_cognitive_state,
                emotional_state,
                internal_lang_activity,
                focus_vector,
                goals_satisfaction,
                vocal_synth_embedding=None  # Real audio embeddings come from 4D conv perception pipeline
            )
        except Exception as e:
            logger.error(f"Output.generate_output: Bio-integrated generate_controls failed: {e}")
            return {param: np.float16(0.0) for param in self.VOCAL_SYNTH_PARAMS}

    def get_output_confidence(self) -> float:
        """Returns the current output confidence score (0-1)."""
        return self._output_confidence

    def get_networks(self) -> List[None]:
        """Returns a list of all internal neural networks for optimization."""
        return [self.expressive_mapper]
    
    # Persistence methods (save/load state)
    # def save_state(self, save_path: str):
    #     """Saves the Output module's state to a file."""
    #     try:
    #         state = {
    #             'last_generated_params': {k: float(v) for k, v in self._last_generated_params.items()},
    #             'output_confidence': float(self._output_confidence),
    #             'expressive_mapper_weights': [(p[0].tolist(), p[1]) for p in self.expressive_mapper.get_trainable_params()]
    #         }
    #         with open(save_path, 'wb') as f:
    #             pickle.dump(state, f)
    #         logger.info(f"Output state saved to {save_path}")
    #     except Exception as e:
    #         logger.error(f"Error saving Output state: {e}")

    # def load_state(self, load_path: str):
    #     """Loads the Output module's state from a file."""
    #     try:
    #         with open(load_path, 'rb') as f:
    #             state = pickle.load(f)
    #         
    #         for k, v in state['last_generated_params'].items():
    #             if k in self._last_generated_params:
    #                 self._last_generated_params[k] = np.float64(v)
    #         self._output_confidence = np.float32(state['output_confidence'])
    #         
    #         loaded_params = state.get('expressive_mapper_weights', [])
    #         current_params = self.expressive_mapper.get_trainable_params()
    #         
    #         if len(loaded_params) == len(current_params):
    #             for i, (param_val_list, grad_name_str) in enumerate(loaded_params):
    #                 param_array, _, layer_instance = current_params[i] 
    #                 param_array[:] = np.array(param_val_list, dtype=np.float32)
    #         else: logger.warning("Expressive mapper weights mismatch. Initializing randomly.")

    #         logger.info(f"Output state loaded from {load_path}")
    #     except FileNotFoundError:
    #         logger.warning(f"Output state file not found at {load_path}. Initializing to default.")
    #     except Exception as e:
    #         logger.error(f"Error loading Output state: {e}. Initializing to default.")


# Bio-integrated vocal synthesis module ready for real audio/video processing pipeline
# Features: Physical constraints, fatigue effects, real sensory input processing
