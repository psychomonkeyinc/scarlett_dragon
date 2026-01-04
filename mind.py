#mind.py
import os
import sys
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True

import queue
import threading
import time
import os
import signal

from typing import Optional, List
import numpy as np
import logging
import pickle
from typing import Dict, List, Optional, Tuple, Any, Callable
from collections import deque
from pathlib import Path

import config as cfg
from globalstate import save_global_state

# Configuration bindings (use sensible defaults when missing)
FLOATX = getattr(cfg, 'FLOATX', __import__('numpy').float32)
SOM_INPUT_DIM = getattr(cfg, 'SOM_INPUT_DIM', getattr(cfg, 'TOTAL_FEATURE_DIM', 128))
SOM_MAP_ROWS = getattr(cfg, 'SOM_MAP_ROWS', 53)
SOM_MAP_COLS = getattr(cfg, 'SOM_MAP_COLS', 53)
MIND_SOM_ACTIVATION_DIM = getattr(cfg, 'MIND_SOM_ACTIVATION_DIM', getattr(cfg, 'SOM_OUTPUT_DIM', SOM_MAP_ROWS * SOM_MAP_COLS))
MIND_SOM_BMU_COORDS_DIM = getattr(cfg, 'MIND_SOM_BMU_COORDS_DIM', 2)
MIND_EMOTIONAL_STATE_DIM = getattr(cfg, 'MIND_EMOTIONAL_STATE_DIM', 512)
MIND_MEMORY_RECALL_DIM = getattr(cfg, 'MIND_MEMORY_RECALL_DIM', 512)
MIND_PREDICTIVE_ERROR_DIM = getattr(cfg, 'MIND_PREDICTIVE_ERROR_DIM', 80)
MIND_UNIFIED_COGNITIVE_STATE_DIM = getattr(cfg, 'MIND_UNIFIED_COGNITIVE_STATE_DIM', 512)
MEMORY_EMOTIONAL_STATE_DIM = getattr(cfg, 'MEMORY_EMOTIONAL_STATE_DIM', 128)
TOTAL_FEATURE_DIM = getattr(cfg, 'TOTAL_FEATURE_DIM', getattr(cfg, 'AUDIO_FEATURE_DIM', 64) + getattr(cfg, 'VIDEO_FEATURE_DIM', 64))
CAFVE_TOKEN_DIM = getattr(cfg, 'CAFVE_TOKEN_DIM', 128)
AUDIO_FEATURE_DIM = getattr(cfg, 'AUDIO_FEATURE_DIM', 64)
VIDEO_FEATURE_DIM = getattr(cfg, 'VIDEO_FEATURE_DIM', 64)
AUDIO_DEVICE_INDEX = getattr(cfg, 'AUDIO_DEVICE_INDEX', None)
AUDIO_SAMPLE_RATE = getattr(cfg, 'AUDIO_SAMPLE_RATE', getattr(cfg, 'AUDIO_FRAME_SIZE', 44100))
# AUDIO_CHUNK_SIZE: fall back to AUDIO_FRAME_SIZE if available
AUDIO_CHUNK_SIZE = getattr(cfg, 'AUDIO_CHUNK_SIZE', getattr(cfg, 'AUDIO_FRAME_SIZE', 1024))
VIDEO_DEVICE_INDEX = getattr(cfg, 'VIDEO_DEVICE_INDEX', 0)
PREDICT_VECTOR_DIM = getattr(cfg, 'PREDICT_VECTOR_DIM', 128)
EMOTION_INPUT_DIM = getattr(cfg, 'EMOTION_INPUT_DIM', TOTAL_FEATURE_DIM)
QA_DATASET_DIR = getattr(cfg, 'QA_DATASET_DIR', None)
QA_SEGMENT_GLOB = getattr(cfg, 'QA_SEGMENT_GLOB', None)
QA_SEGMENT_DURATION_SECONDS = getattr(cfg, 'QA_SEGMENT_DURATION_SECONDS', None)
QA_STATS_PATH = getattr(cfg, 'QA_STATS_PATH', None)

from nn import Sequential, Linear, Sigmoid, ReLU, Tanh
from som import SelfOrganizingMap
from emotion import EmotionCore
from memory import MemorySystem
from predict import Predict
from health import Health
from data import DataCollection
from input import AudioIn, VideoIn
from output import Output
from vocalsynth import VocalSynth
from audioout import AudioOut
from sfe import SensoryFeatureExtractor
from cafve import ConsciousnessAwareFeatureVectorEncoder
logger = logging.getLogger(__name__)

STATE_DIR = Path(__file__).resolve().parent / "states"
MASTER_STATE_DIR = Path(__file__).resolve().parent / "master"
SOM_STATE_PATH = STATE_DIR / "som_state.npz"
NN_STATE_PATH = STATE_DIR / "nn_state.npz"
PREDICT_STATE_PATH = STATE_DIR / "predictor_state.npz"
MASTER_STATE_PATH = MASTER_STATE_DIR / "master_model_state.pkl"

# from cafve import ACEConsciousnessTokenizer

# Dimension constants for import compatibility
SOM_ACTIVATION_DIM = MIND_SOM_ACTIVATION_DIM
SOM_BMU_COORD_DIM = MIND_SOM_BMU_COORDS_DIM

class CognitiveScaling:
    """Manages dynamic dimensionality scaling across cognitive modules"""

    DIMENSION_STAGES = getattr(cfg, 'COGNITIVE_SCALING_DIMENSION_STAGES', [
        (512, 1024, 2048),    # Initial stage (base, mid, high)
        (1024, 2048, 4096),   # First expansion
        (2048, 4096, 8192),   # Second expansion
        (4096, 8192, 16384)   # Final expansion tier
    ])

    def __init__(self, initial_stage: int = getattr(cfg, 'COGNITIVE_SCALING_INITIAL_STAGE', 0)):
        self.current_stage = initial_stage
        self.base_dim, self.mid_dim, self.high_dim = self.DIMENSION_STAGES[initial_stage]
        self.growth_metrics = {
            'complexity_score': 0.0,
            'integration_score': 0.0,
            'stability_score': 0.0,
            'utilization_score': 0.0
        }
        self.cycles_since_growth = 0
        self.cooldown = getattr(cfg, 'SCALING_COOLDOWN_CYCLES', 10000)

    def should_grow(self) -> bool:
        """Determine if cognitive dimensions should expand based on strict rules"""
        if self.current_stage >= len(self.DIMENSION_STAGES) - 1:
            return False

        if self.cycles_since_growth < self.cooldown:
            return False

        # Rule 1: Resource Headroom Check
        try:
            import psutil
            cpu_avail = (100 - psutil.cpu_percent()) / 100.0
            mem_avail = (100 - psutil.virtual_memory().percent) / 100.0
            if min(cpu_avail, mem_avail) < getattr(cfg, 'SCALING_RESOURCE_HEADROOM', 0.3):
                return False
        except ImportError:
            pass

        # Rule 2: Stability Check
        if self.growth_metrics['stability_score'] < getattr(cfg, 'SCALING_STABILITY_THRESHOLD', 0.9):
            return False

        # Rule 3: Weighted scoring for growth decision
        growth_weights = getattr(cfg, 'COGNITIVE_SCALING_GROWTH_WEIGHTS', {
            'complexity': 0.3,
            'integration': 0.3,
            'stability': 0.2,
            'utilization': 0.2
        })
        growth_score = (
            growth_weights['complexity'] * self.growth_metrics['complexity_score'] +
            growth_weights['integration'] * self.growth_metrics['integration_score'] +
            growth_weights['stability'] * self.growth_metrics['stability_score'] +
            growth_weights['utilization'] * self.growth_metrics['utilization_score']
        )

        growth_threshold = getattr(cfg, 'COGNITIVE_SCALING_GROWTH_THRESHOLD', 0.85)
        return growth_score > growth_threshold

    def grow(self) -> bool:
        """Attempt to grow to next dimension stage"""
        if not self.should_grow():
            self.cycles_since_growth += 1
            return False

        if self.current_stage < len(self.DIMENSION_STAGES) - 1:
            self.current_stage += 1
            self.base_dim, self.mid_dim, self.high_dim = self.DIMENSION_STAGES[self.current_stage]
            self.cycles_since_growth = 0
            return True
        return False

    def update_metrics(self, complexity: float, integration: float,
                      stability: float, utilization: float):
        """Update growth metrics based on system performance"""
        self.growth_metrics['complexity_score'] = complexity
        self.growth_metrics['integration_score'] = integration
        self.growth_metrics['stability_score'] = stability
        self.growth_metrics['utilization_score'] = utilization

class CognitiveState:
    def __init__(self, dim: int):
        self.dimension = int(dim)
        self.state_vector = np.zeros(self.dimension, dtype=FLOATX)
        self.emotional_influence = np.zeros(self.dimension, dtype=FLOATX)
        self.attention_mask = np.ones(self.dimension, dtype=FLOATX)
        self.uncertainty = np.zeros(self.dimension, dtype=FLOATX)

    @staticmethod
    def _fit(vector: np.ndarray, dim: int) -> np.ndarray:
        arr = np.asarray(vector, dtype=FLOATX).reshape(-1)
        if arr.shape[0] < dim:
            arr = np.pad(arr, (0, dim - arr.shape[0]))
        elif arr.shape[0] > dim:
            arr = arr[:dim]
        return arr

    def update(self, new_state: np.ndarray, emotional_context: np.ndarray,
              attention: np.ndarray, certainty: np.ndarray):
        self.state_vector = self._fit(new_state, self.dimension)
        self.emotional_influence = self._fit(emotional_context, self.dimension)
        self.attention_mask = self._fit(attention, self.dimension)
        certainty = np.clip(self._fit(certainty, self.dimension), 0.0, 1.0)
        self.uncertainty = 1 - certainty

    def get_weighted_state(self) -> np.ndarray:
        return self.state_vector * self.attention_mask * (1 - self.uncertainty)

    def get_confidence(self) -> float:
        """Calculate overall confidence in current cognitive state"""
        return 1.0 - np.mean(self.uncertainty)

    def get_attention_focus(self) -> np.ndarray:
        """Get the current attention distribution"""
        epsilon = getattr(cfg, 'ATTENTION_FOCUS_EPSILON', 1e-6)
        return self.attention_mask / (np.sum(self.attention_mask) + epsilon)

class MetaCognition:
    """Handles self-reflection and cognitive monitoring"""
    def __init__(self, state_dim: int):
        self.state_dimension = state_dim
        reflection_multiplier = getattr(cfg, 'METACOGNITION_REFLECTION_MULTIPLIER', 2)
        self.reflection_network = Sequential([
            Linear(state_dim, state_dim * reflection_multiplier),
            Tanh(),
            Linear(state_dim * reflection_multiplier, state_dim),
            Sigmoid()
        ])
        self.metacognitive_history = []

    def reflect(self, cognitive_state: CognitiveState) -> Tuple[np.ndarray, float]:
        """Analyze current cognitive state and generate insights"""
        state_vector = cognitive_state.get_weighted_state().reshape(1, -1)
        reflection = self.reflection_network.forward(state_vector).reshape(-1)
        coherence = cognitive_state.get_confidence() * np.mean(reflection)
        self.metacognitive_history.append((reflection, coherence))
        return reflection, coherence

# Module logger only: do not configure root logging handlers here.
# Logging configuration (handlers/format) should be centralized in `run.py` or the application entrypoint.

class Mind:
    """
    Core cognitive architecture coordinating SOM, Memory, Emotion, and more.
    Supports dynamic dimensional scaling for enhanced cognitive capacity.
    """
    def __init__(self,
                 initial_dim_stage: int = 0,
                 som_activation_dim: int = MIND_SOM_ACTIVATION_DIM,
                 som_bmu_coords_dim: int = MIND_SOM_BMU_COORDS_DIM,
                 emotional_state_dim: int = MIND_EMOTIONAL_STATE_DIM,
                 memory_recall_dim: int = MIND_MEMORY_RECALL_DIM,
                 predictive_error_dim: int = MIND_PREDICTIVE_ERROR_DIM,
                 unified_cognitive_state_dim: int = MIND_UNIFIED_COGNITIVE_STATE_DIM,
                 som_instance: Optional[SelfOrganizingMap] = None,
                 emotion_instance: Optional[EmotionCore] = None,
                 memory_instance: Optional[MemorySystem] = None,
                 predict_instance: Optional[Predict] = None,
                 data_collector: Optional[DataCollection] = None):
        """
        Initializes the Mind module, which is the central cognitive processor.
        """
        
        # Create scaling manager
        self.scaling = CognitiveScaling(initial_stage=initial_dim_stage)
        if self.scaling.base_dim != unified_cognitive_state_dim:
            self.scaling.base_dim = unified_cognitive_state_dim
            self.scaling.mid_dim = max(self.scaling.mid_dim, unified_cognitive_state_dim * 2)
            self.scaling.high_dim = max(self.scaling.high_dim, unified_cognitive_state_dim * 4)

        self.is_dreaming = False
        self.sensory_processing_enabled = True
        self.memory_replay_active = False

        # Live sensory capture pipeline
        self.sensory_extractor = SensoryFeatureExtractor()
        self.sensory_extractor.start()

        # CAFVE tokenization pipeline maintains recent sensory embeddings for prediction.
        # Use the provided CAFVE implementation (ensures FP32 token vectors).
        # Use conservative defaults if config lacks CAFVE hyperparameters
        cafve_vocab = getattr(cfg, 'CAFVE_VOCAB_SIZE', 50000)
        cafve_weight = getattr(cfg, 'CAFVE_CONSCIOUS_WEIGHT', 0.3)
        self.cafve = ConsciousnessAwareFeatureVectorEncoder(
            vocab_size=int(cafve_vocab),
            sfe_feature_dim=TOTAL_FEATURE_DIM,
            token_output_dim=CAFVE_TOKEN_DIM,
            consciousness_weight=float(cafve_weight),
        )
        cafve_window_size = getattr(cfg, 'CAFVE_WINDOW_SIZE', 6)
        self._cafve_window: deque[np.ndarray] = deque(maxlen=cafve_window_size)
        # Ensure last CAFVE vector is FP32 to match CAFVE outputs
        self._last_cafve_vector = np.zeros(CAFVE_TOKEN_DIM, dtype=np.float32)
        self._last_cafve_ids: List[int] = []
        self._last_cabpe_pairs: List[Tuple[Tuple[int, int], float]] = []

        # Store dimensions
        self.som_activation_dim = som_activation_dim
        self.som_bmu_coords_dim = som_bmu_coords_dim
        self.emotional_state_dim = emotional_state_dim
        self.memory_recall_dim = memory_recall_dim
        self.predictive_error_dim = predictive_error_dim
        self.unified_cognitive_state_dim = unified_cognitive_state_dim

        # Initialize core modules with scalable dimensions
        # Dependency injection to avoid duplicate module instantiation during stabilization.
        # If external instances are provided we reuse them; otherwise we create new ones (legacy behavior).
        
        # Add NN, Health, and DataCollection modules for full cognition orchestration
        perception_hidden = max(self.scaling.mid_dim, SOM_INPUT_DIM)
        self.nn = Sequential(
            Linear(TOTAL_FEATURE_DIM, perception_hidden),
            ReLU(),
            Linear(perception_hidden, SOM_INPUT_DIM),
            Sigmoid()
        )
        nn_learning_rate = getattr(cfg, 'NN_LEARNING_RATE', 1e-4)
        self._base_nn_learning_rate = float(getattr(self.nn, "learning_rate", FLOATX(nn_learning_rate)))
        self.health = Health(
            som_map_size=(SOM_MAP_ROWS, SOM_MAP_COLS),
            unified_cognitive_state_dim=self.scaling.base_dim,
            emotional_state_dim=self.emotional_state_dim
        )
        self.data_collector = data_collector or DataCollection()
        self.output_controller = Output(
            unified_cognitive_state_dim=self.scaling.base_dim,
            emotional_state_dim=self.emotional_state_dim,
            internal_lang_dim=self.scaling.mid_dim,
            vocal_synth_params_dim=self.scaling.mid_dim,
            attention_focus_dim=self.scaling.mid_dim
        )
        self.vocal_synth = VocalSynth(sample_rate=getattr(cfg, 'AUDIO_SAMPLE_RATE_DEFAULT', 44100), buffer_size_samples=getattr(cfg, 'AUDIO_CHUNK_SIZE_DEFAULT', 4410))
        self.audio_out = AudioOut(sample_rate=getattr(cfg, 'AUDIO_SAMPLE_RATE_DEFAULT', 44100), chunk_duration=getattr(cfg, 'AUDIO_CHUNK_DURATION', 0.1))
        
        # Start audio output
        try:
            self.audio_out.start()
        except Exception as e:
            logger.warning(f"Audio output failed to start: {e}")

        # Offline SOM training capture setup (plain-text log)
        self._som_capture_cache: List[np.ndarray] = []
        self._som_capture_flush_size = getattr(cfg, 'SOM_CAPTURE_FLUSH_SIZE', 128)

        # State for full cognition processing
        self.cycle_count = 0
        self.running = True
        self._previous_input = None

        if emotion_instance is not None:
            self.emotion = emotion_instance
        else:
            self.emotion = EmotionCore(
                input_dim=EMOTION_INPUT_DIM,
                output_dim=emotional_state_dim
            )

        if memory_instance is not None:
            self.memory = memory_instance
        else:
            self.memory = MemorySystem(
                cognitive_state_dim=memory_recall_dim,
                emotional_state_dim=MEMORY_EMOTIONAL_STATE_DIM
            )

        if som_instance is not None:
            self.som = som_instance
        else:
            self.som = SelfOrganizingMap(
                input_dim=SOM_INPUT_DIM,
                map_size=self._calculate_som_size()
            )
        self.som_activation_dim = self.som.map_size[0] * self.som.map_size[1]

        # Recompute integrator input dimension with finalized components
        self.total_input_dim = (
            self.som_activation_dim
            + self.som_bmu_coords_dim
            + self.emotional_state_dim
            + self.memory_recall_dim
            + self.predictive_error_dim
            + self.unified_cognitive_state_dim
        )

        if predict_instance is not None:
            self.predict = predict_instance
        else:
            # Create Predict instance with appropriate dimensions
            self.predict = Predict(
                sfe_feature_dim=SOM_INPUT_DIM,
                cafve_token_dim=CAFVE_TOKEN_DIM,
                unified_cognitive_state_dim=self.scaling.base_dim,
                emotional_state_dim=self.emotional_state_dim,
                other_mind_model_dim=PREDICT_VECTOR_DIM,
                focus_vector_dim=PREDICT_VECTOR_DIM,
                predicted_output_dim=self.scaling.base_dim,
                prediction_error_dim=self.predictive_error_dim
            )
        predictor_lr = getattr(self.predict.predictor_network, "learning_rate", FLOATX(getattr(cfg, 'PREDICTOR_LEARNING_RATE', 1e-4)))
        self._base_predictor_learning_rate = float(predictor_lr)

        # Initialize Dream module for sleep consolidation
        try:
            from dream import Dream
            self.dream = Dream(
                som=self.som,
                emotion_core=self.emotion,
                memory=self.memory
            )
            logger.info("Dream module initialized successfully")
        except ImportError:
            logger.warning("Dream module import failed, sleep consolidation will be disabled")
            self.dream = None
        except Exception as e:
            logger.warning(f"Dream module initialization failed: {e}")
            self.dream = None

        # Module persistence paths
        self._state_dir = STATE_DIR
        self._master_state_dir = MASTER_STATE_DIR
        self._master_state_path = MASTER_STATE_PATH
        self._som_state_path = SOM_STATE_PATH
        self._nn_state_path = NN_STATE_PATH
        self._predict_state_path = PREDICT_STATE_PATH
        
        # State rotation tracking
        self._state_slot = 0

        # Integration networks
        internal_hidden_dim = self.scaling.mid_dim
        self.integration_network = Sequential(
            Linear(self.total_input_dim, internal_hidden_dim),
            Tanh(),
            Linear(internal_hidden_dim, self.scaling.base_dim),
            Sigmoid()
        )

        # Internal state
        self._unified_cognitive_state = np.zeros(self.scaling.base_dim, dtype=FLOATX)

        # Restore persisted module parameters if available
        try:
            self._load_module_states()
        except Exception:
            logger.warning("Module state restoration failed", exc_info=True)

        
    def apply_training_weight(self, weight: float) -> None:
        """Scale learnable module rates so MetaMind can modulate plasticity."""
        try:
            weight_min = getattr(cfg, 'TRAINING_WEIGHT_MIN', 0.05)
            weight_max = getattr(cfg, 'TRAINING_WEIGHT_MAX', 4.0)
            multiplier = float(np.clip(weight, weight_min, weight_max))
        except Exception:
            multiplier = 1.0

        if hasattr(self.nn, "learning_rate"):
            self.nn.learning_rate = FLOATX(self._base_nn_learning_rate * multiplier)

        predictor_net = getattr(self.predict, "predictor_network", None)
        if predictor_net is not None and hasattr(predictor_net, "learning_rate"):
            predictor_net.learning_rate = FLOATX(self._base_predictor_learning_rate * multiplier)

    @staticmethod
    def _save_som_state(self, path: Path) -> None:
        """Save SOM state to compressed numpy file."""
        som_data = {
            'weights': self.som.weights,
            'plasticity': self.som.plasticity,
            'fatigue': self.som.fatigue,
            'map_size': self.som.map_size,
            'input_dim': self.som.input_dim,
            'current_growth_stage': self.som.current_growth_stage,
            'cycles_since_growth': self.som.cycles_since_growth
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(path), **som_data)

    def _load_som_state(self, path: Path) -> bool:
        """Load SOM state from compressed numpy file."""
        if not path.exists():
            return False
        try:
            with np.load(str(path), allow_pickle=False) as data:
                if 'weights' in data and data['weights'].shape == self.som.weights.shape:
                    self.som.weights = data['weights']
                if 'plasticity' in data:
                    self.som.plasticity = data['plasticity']
                if 'fatigue' in data:
                    self.som.fatigue = data['fatigue']
                if 'current_growth_stage' in data:
                    self.som.current_growth_stage = int(data['current_growth_stage'])
                if 'cycles_since_growth' in data:
                    self.som.cycles_since_growth = int(data['cycles_since_growth'])
            return True
        except Exception as e:
            logger.warning(f"Failed to load SOM state from {path}: {e}")
            return False

    def _get_network_state(self, network: Optional[Sequential]) -> Optional[Dict[str, np.ndarray]]:
        """Extract state from a Sequential network."""
        if network is None or not hasattr(network, "layers"):
            return None
        
        state = {}
        layer_count = 0
        for layer in network.layers:
            if isinstance(layer, Linear):
                state[f"layer_{layer_count}_weights"] = layer.weights
                state[f"layer_{layer_count}_biases"] = layer.biases
                layer_count += 1
        
        if hasattr(network, "learning_rate"):
            state["learning_rate"] = np.array([float(network.learning_rate)])
        
        state["layer_count"] = np.array([layer_count])
        return state

    def _set_network_state(self, network: Optional[Sequential], state: Optional[Dict[str, np.ndarray]]) -> bool:
        """Restore state to a Sequential network."""
        if network is None or state is None or not hasattr(network, "layers"):
            return False
        
        try:
            layer_count = int(state.get("layer_count", [0])[0])
            layer_idx = 0
            for layer in network.layers:
                if isinstance(layer, Linear) and layer_idx < layer_count:
                    weight_key = f"layer_{layer_idx}_weights"
                    bias_key = f"layer_{layer_idx}_biases"
                    if weight_key in state and bias_key in state:
                        layer.weights = state[weight_key]
                        layer.biases = state[bias_key]
                    layer_idx += 1
            
            if "learning_rate" in state and hasattr(network, "learning_rate"):
                network.learning_rate = FLOATX(float(state["learning_rate"][0]))
            
            return True
        except Exception as e:
            logger.warning(f"Failed to restore network state: {e}")
            return False

    def _get_som_state(self) -> Dict[str, np.ndarray]:
        """Extract current SOM state."""
        return {
            'weights': self.som.weights,
            'plasticity': self.som.plasticity,
            'fatigue': self.som.fatigue,
            'current_growth_stage': np.array([self.som.current_growth_stage]),
            'cycles_since_growth': np.array([self.som.cycles_since_growth])
        }

    def _set_som_state(self, state: Dict[str, np.ndarray]) -> bool:
        """Restore SOM state."""
        try:
            if 'weights' in state:
                self.som.weights = state['weights']
            if 'plasticity' in state:
                self.som.plasticity = state['plasticity']
            if 'fatigue' in state:
                self.som.fatigue = state['fatigue']
            if 'current_growth_stage' in state:
                self.som.current_growth_stage = int(state['current_growth_stage'][0])
            if 'cycles_since_growth' in state:
                self.som.cycles_since_growth = int(state['cycles_since_growth'][0])
            return True
        except Exception as e:
            logger.warning(f"Failed to restore SOM state: {e}")
            return False

    @staticmethod
    def _load_sequential_state(network: Sequential, path: Path) -> bool:
        """Restore Linear layer weights/biases for a Sequential network."""
        if network is None or not hasattr(network, "layers"):
            return False
        if not path.exists():
            return False

        try:
            with np.load(str(path), allow_pickle=False) as data:
                layer_idx = 0
                for layer in network.layers:
                    if isinstance(layer, Linear):
                        weight_key = f"layer_{layer_idx}_weights"
                        bias_key = f"layer_{layer_idx}_biases"
                        if weight_key not in data or bias_key not in data:
                            logger.warning(
                                "Sequential state at %s missing keys for layer %d",
                                path,
                                layer_idx,
                            )
                            return False
                        layer.weights = np.asarray(data[weight_key], dtype=layer.weights.dtype)
                        layer.biases = np.asarray(data[bias_key], dtype=layer.biases.dtype)
                        layer_idx += 1

                if "learning_rate" in data and hasattr(network, "learning_rate"):
                    network.learning_rate = FLOATX(float(np.asarray(data["learning_rate"]).reshape(-1)[0]))

            logger.info("Sequential network weights restored from %s", path)
            return True
        except Exception as exc:
            logger.warning("Failed to load sequential network state from %s: %s", path, exc)
            return False

    def _save_module_states(self) -> None:
        """Persist all neural weights to single rotating state file (3-slot rotation)."""
        try:
            self._state_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            logger.warning("Unable to create state directory %s: %s", self._state_dir, exc)
            return

        # Rotate state files (0, 1, 2) - single consolidated file per slot
        current_slot = getattr(self, '_state_slot', 0)
        state_path = self._state_dir / f"state_{current_slot}.pkl"
        
        state_data = {
            'timestamp': time.time(),
            'cycle_count': getattr(self, 'cycle_count', 0),
            'slot': current_slot,
            'nn_state': self._get_network_state(self.nn),
            'som_state': self._get_som_state(),
            'predictor_state': self._get_network_state(getattr(self.predict, "predictor_network", None)),
            'metadata': {
                'som_size': (self.som.map_size[0], self.som.map_size[1]),
                'input_dim': self.som.input_dim,
                'unified_cognitive_dim': self.unified_cognitive_state_dim
            }
        }
        
        try:
            with open(state_path, 'wb') as f:
                pickle.dump(state_data, f)
            logger.info(f"State saved to slot {current_slot}: {state_path}")
        except Exception as e:
            logger.error(f"Failed to save state to slot {current_slot}: {e}")
            return
        
        # Update slot for next save
        self._state_slot = (current_slot + 1) % 3

    def _load_module_states(self) -> bool:
        """Load all neural weights from most recent state slot."""
        restored_any = False
        
        # Try to load from most recent slot (check all 3, prefer newest)
        for slot in [2, 1, 0]:  # Check newest first
            state_path = self._state_dir / f"state_{slot}.pkl"
            
            if not state_path.exists():
                continue
                
            try:
                with open(state_path, 'rb') as f:
                    state_data = pickle.load(f)
                
                # Restore networks
                if 'nn_state' in state_data and state_data['nn_state']:
                    self._set_network_state(self.nn, state_data['nn_state'])
                    restored_any = True
                    
                if 'som_state' in state_data and state_data['som_state']:
                    self._set_som_state(state_data['som_state'])
                    restored_any = True
                    
                if 'predictor_state' in state_data and state_data['predictor_state']:
                    predictor_net = getattr(self.predict, "predictor_network", None)
                    if predictor_net:
                        self._set_network_state(predictor_net, state_data['predictor_state'])
                        restored_any = True
                        
                if restored_any:
                    self._state_slot = slot  # Set current slot to loaded slot
                    logger.info(f"States restored from slot {slot} (saved at {time.ctime(state_data.get('timestamp', 0))})")
                    break
                    
            except Exception as e:
                logger.warning(f"Failed to load state from slot {slot}: {e}")
                continue

        return restored_any

    def save_master_state(self) -> None:
        """Save current state as the master working model (untouched backup)."""
        try:
            self._master_state_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            logger.warning("Unable to create master state directory %s: %s", self._master_state_dir, exc)
            return

        master_data = {
            'timestamp': time.time(),
            'cycle_count': getattr(self, 'cycle_count', 0),
            'nn_state': self._get_network_state(self.nn),
            'som_state': self._get_som_state(),
            'predictor_state': self._get_network_state(getattr(self.predict, "predictor_network", None)),
            'metadata': {
                'som_size': (self.som.map_size[0], self.som.map_size[1]),
                'input_dim': self.som.input_dim,
                'unified_cognitive_dim': self.unified_cognitive_state_dim
            }
        }
        
        try:
            with open(self._master_state_path, 'wb') as f:
                pickle.dump(master_data, f)
            logger.info(f"Master state saved to {self._master_state_path}")
        except Exception as e:
            logger.error(f"Failed to save master state: {e}")

    def load_master_state(self) -> bool:
        """Load state from master working model."""
        if not self._master_state_path.exists():
            logger.warning("No master state file found")
            return False
            
        try:
            with open(self._master_state_path, 'rb') as f:
                master_data = pickle.load(f)
                
            # Restore networks
            if 'nn_state' in master_data:
                self._set_network_state(self.nn, master_data['nn_state'])
                
            if 'som_state' in master_data:
                self._set_som_state(master_data['som_state'])
                
            if 'predictor_state' in master_data:
                predictor_net = getattr(self.predict, "predictor_network", None)
                if predictor_net:
                    self._set_network_state(predictor_net, master_data['predictor_state'])
                    
            logger.info(f"Master state loaded from {self._master_state_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load master state: {e}")
            return False


    def integrate_cognition(self,
                            som_activation_map: np.ndarray, # SOM map sized per configuration
                            som_bmu_coords: Tuple[int, int], # (row, col) of BMU
                            learning_bias_from_desire: float = 1.0, # From Desire module
                            attention_weights_from_attention: Optional[np.ndarray] = None # From Attention module
                            ) -> np.ndarray:
        """
        Processes and integrates various cognitive, emotional, and memory inputs
        into a unified cognitive state.
        """
        try:
            # 1. Flatten SOM activation map
            flat_som_activation = som_activation_map.reshape(-1).astype(FLOATX)

            # 2. Convert BMU coords to normalized vector (e.g., 0-1 range)
            # Use configured SOM map size for normalization
            norm_row = som_bmu_coords[0] / max(SOM_MAP_ROWS - 1, 1)
            norm_col = som_bmu_coords[1] / max(SOM_MAP_COLS - 1, 1)
            normalized_bmu_coords = np.array([norm_row, norm_col], dtype=FLOATX)

            # 3. Get emotional state from emotion module
            emotional_state = self.emotion.get_emotion_vector()

            # 4. Get memory recall from memory module
            memory_recall_vector = self.memory.recall(self._unified_cognitive_state)
            # Ensure memory recall is 1D for concatenation
            if memory_recall_vector.ndim > 1:
                memory_recall_vector = memory_recall_vector.flatten()
            memory_recall_vector = memory_recall_vector.astype(FLOATX)

            # 5. Prediction error provides uncertainty context
            try:
                prediction_error = self.predict.get_prediction_error()
            except AttributeError:
                prediction_error = np.zeros(self.predictive_error_dim, dtype=FLOATX)
            prediction_error = np.asarray(prediction_error, dtype=FLOATX).reshape(-1)

            # --- Combine all inputs into a single vector ---
            parts = [
                flat_som_activation,
                normalized_bmu_coords,
                emotional_state,
                memory_recall_vector,
                prediction_error,
                self._unified_cognitive_state.astype(FLOATX)
            ]
            combined_input = np.concatenate(parts).astype(FLOATX)

            # Handle dimensional mismatches dynamically
            incoming_len = combined_input.shape[0]
            if incoming_len > self.total_input_dim:
                self._expand_integration_input(incoming_len)
            elif incoming_len < self.total_input_dim:
                pad_len = self.total_input_dim - incoming_len
                combined_input = np.concatenate([combined_input, np.zeros(pad_len, dtype=FLOATX)])

            # Ensure no errors are thrown for mismatches
            if combined_input.shape[0] != self.total_input_dim:
                logger.warning(f"Post-adjustment length mismatch: {combined_input.shape[0]} vs {self.total_input_dim}. Using fallback state.")
                return self._unified_cognitive_state
            
            # --- Apply attentional modulation (if Attention module provides weights) ---
            # This would apply attention_weights_from_attention as a gate or scalar multiplier
            # to parts of the combined_input or specific layers within the integrator_network.
            # For lean implementation, we'll assume it's an external influence on learning or processing.

            # --- Process through the Cognitive Integration Network ---
            # Add batch dimension for neural network
            combined_input_batched = combined_input.reshape(1, -1)
            new_unified_state = self.integration_network.forward(combined_input_batched)

            # Update internal unified cognitive state
            self._unified_cognitive_state = new_unified_state[0, :].astype(FLOATX)
            
            return self._unified_cognitive_state
        except Exception as e:
            logger.error(f"Error in integrate_cognition: {e}")
            return self._unified_cognitive_state.copy()

    def process_full_cycle(
        self,
        manual_features: Optional[np.ndarray] = None,
        learning_override: Optional[bool] = None
    ) -> Dict[str, Any]:
        """Process one cognition cycle, optionally with injected sensory data and forced learning."""
        cycle_start = time.time()
        training_losses: Dict[str, float] = {}
        learning_enabled = bool(learning_override)

        try:
            if manual_features is not None:
                sensory_input = np.asarray(manual_features, dtype=FLOATX).reshape(-1)
            else:
                # Extract features using sensory extractor (real-time capture)
                sensory_input = self.sensory_extractor.get_features(None, None)

            if sensory_input.size == 0:
                sensory_input = self.sensory_extractor.get_last_features()

            if sensory_input.shape[0] != TOTAL_FEATURE_DIM:
                padded = np.zeros(TOTAL_FEATURE_DIM, dtype=FLOATX)
                limit = min(sensory_input.shape[0], TOTAL_FEATURE_DIM)
                padded[:limit] = sensory_input[:limit]
                sensory_input = padded

            audio_features = sensory_input[:AUDIO_FEATURE_DIM]
            visual_features = sensory_input[AUDIO_FEATURE_DIM:AUDIO_FEATURE_DIM + VIDEO_FEATURE_DIM]

            if self._previous_input is not None:
                sensory_input = 0.7 * sensory_input + 0.3 * self._previous_input
            self._previous_input = sensory_input.copy()

            nn_input = sensory_input.reshape(1, -1)
            nn_output = self.nn.forward(nn_input)
            nn_output_flat = nn_output.reshape(-1).astype(FLOATX)

            self._record_som_training_sample(nn_output_flat)

            cafve_vector = self._update_cafve_context(nn_output_flat, learning_enabled)

            som_activation = self.som.process_input(nn_output_flat)
            som_activation_2d = som_activation.reshape(self.som.map_size)
            som_bmu = np.unravel_index(np.argmax(som_activation_2d), som_activation_2d.shape)

            emotional_state = self.emotion.process_emotions(nn_output_flat)
            memory_recall = self.memory.recall(self._unified_cognitive_state)

            cognitive_state = self.integrate_cognition(
                som_activation_map=som_activation_2d,
                som_bmu_coords=som_bmu
            )

            try:
                self.predict.observe_outcome(cognitive_state)
            except Exception as predict_record_exc:
                logger.debug("Predict observe_outcome error: %s", predict_record_exc)

            focus_vector = cognitive_state.astype(FLOATX)
            vocal_params = self.output_controller.generate_output(
                unified_cognitive_state=cognitive_state,
                emotional_state=emotional_state,
                focus_vector=focus_vector
            )

            audio_buffer = self.vocal_synth.generate_audio_buffer(vocal_params)

            # EXPERIMENT: Gate output - only play audio if confidence is high enough (prevent forced output)
            output_confidence = getattr(self.output_controller, '_output_confidence', 0.5)
            speech_gate_threshold = 0.6  # Only speak when 60%+ confident
            
            if output_confidence >= speech_gate_threshold:
                try:
                    self.audio_out.play_chunk(audio_buffer.astype(np.float64))
                except Exception as audio_err:
                    logger.warning("Audio output failed: %s", audio_err)
            # else: System is thinking, not speaking

            try:
                prediction_error_vector = np.asarray(self.predict.get_prediction_error(), dtype=FLOATX).reshape(-1)
            except AttributeError:
                prediction_error_vector = np.zeros(self.predictive_error_dim, dtype=FLOATX)

            prediction_error_norm = float(np.linalg.norm(prediction_error_vector))
            moral_tension = float(np.clip(np.mean(np.abs(emotional_state)), 0.0, 1.0))

            health_score = self.health.update_health_metrics(
                som_fatigue_map=self.som.fatigue_map if hasattr(self.som, 'fatigue_map') else np.zeros(self.som.map_size, dtype=FLOATX),
                som_failure_log_count=len(getattr(self.som, 'failure_log', [])),
                predict_error_norm=prediction_error_norm,
                moral_tension_level=moral_tension,
                emotional_state_norm=np.linalg.norm(emotional_state),
                cognitive_load_norm=np.linalg.norm(cognitive_state),
                is_dreaming=False
            )

            cycle_timestamp = time.time()

            self.memory.store_experience(
                cognitive_state=cognitive_state,
                sensory_snapshot=sensory_input,
                emotion_state=emotional_state,
                reward_signal=float(health_score),
                timestamp=cycle_timestamp
            )

            if learning_enabled:
                if hasattr(self.som, 'weights'):
                    bmu_row, bmu_col = int(som_bmu[0]), int(som_bmu[1])
                    som_target = self.som.weights[bmu_row, bmu_col].reshape(1, -1)
                    perception_loss = self.nn.train_step(nn_input, som_target)
                    training_losses['perception_loss'] = float(perception_loss)

                predict_loss = self._train_predictor(
                    sensory_features=nn_output_flat,
                    unified_state=cognitive_state,
                    emotional_state=emotional_state,
                    cafve_vector=cafve_vector
                )
                training_losses['predict_loss'] = float(predict_loss)

            current_time = cycle_timestamp
            snapshot = {
                "cycle": self.cycle_count,
                "timestamp": current_time,
                "cycle_duration_ms": (time.time() - cycle_start) * 1000,
                "visual_features_norm": float(np.linalg.norm(visual_features)),
                "audio_features_norm": float(np.linalg.norm(audio_features)),
                "sensory_input_norm": float(np.linalg.norm(sensory_input)),
                "nn_output_norm": float(np.linalg.norm(nn_output_flat)),
                "cafve_token_norm": float(np.linalg.norm(self._last_cafve_vector)),
                "cafve_recent_ids": self._last_cafve_ids[-4:],
                "som_activation_max": float(np.max(som_activation)),
                "som_activation_mean": float(np.mean(som_activation)),
                "som_bmu_coords": [int(som_bmu[0]), int(som_bmu[1])],
                "emotional_state_norm": float(np.linalg.norm(emotional_state)),
                "emotional_dominant": "positive" if np.mean(emotional_state) > 0 else "negative",
                "memory_recall_norm": float(np.linalg.norm(memory_recall)),
                "cognitive_state_norm": float(np.linalg.norm(cognitive_state)),
                "cognitive_confidence": float(1.0 / (1.0 + np.var(cognitive_state))),
                "vocal_f0_hz": float(vocal_params.get('F0_hz', 0)),
                "vocal_amplitude": float(vocal_params.get('Amplitude', 0)),
                "vocal_voicing_mix": float(vocal_params.get('Voicing_Mix', 0)),
                "audio_buffer_norm": float(np.linalg.norm(audio_buffer)),
                "audio_buffer_peak": float(np.max(np.abs(audio_buffer))),
                "health_score": float(health_score),
                "tiredness_factor": float(self.health.get_tiredness_factor()),
                "system_stability": float(1.0 - np.std([health_score, np.linalg.norm(cognitive_state), np.linalg.norm(emotional_state)])),
                "modules_active": ["nn", "som", "emotion", "memory", "mind", "health", "output", "vocalsynth", "audioout"],
                "processing_complete": True,
                "learning_active": learning_enabled,
                "vocal_output_active": bool(np.any(audio_buffer))
            }

            if training_losses:
                snapshot['training_losses'] = training_losses

            self.cycle_count += 1
            return snapshot

        except Exception as exc:
            logger.exception("Cognition cycle %s failed", self.cycle_count)
            return {
                "cycle": self.cycle_count,
                "timestamp": time.time(),
                "error": str(exc),
                "processing_complete": False
            }

    @staticmethod
    def _fit_dimension(vector: np.ndarray, target_dim: int) -> np.ndarray:
        arr = np.asarray(vector, dtype=FLOATX).reshape(-1)
        if arr.size < target_dim:
            padded = np.zeros(target_dim, dtype=FLOATX)
            padded[:arr.size] = arr
            return padded
        if arr.size > target_dim:
            return arr[:target_dim]
        return arr

    def _update_cafve_context(
        self,
        sensory_vector: np.ndarray,
        learning_enabled: bool,
    ) -> np.ndarray:
        if not hasattr(self, "cafve") or self.predict is None:
            return np.zeros(CAFVE_TOKEN_DIM, dtype=FLOATX)

        fitted = self._fit_dimension(sensory_vector, self.predict.sfe_feature_dim).astype(FLOATX, copy=False)
        self._cafve_window.append(fitted.copy())
        sequence = list(self._cafve_window)

        result = self.cafve.tokenize(sequence, return_metadata=True)
        self._last_cafve_ids = [int(token_id) for token_id in result.token_ids]

        token_vectors = [
            np.asarray(vec, dtype=FLOATX).reshape(-1)
            for vec in result.token_vectors
        ]

        if learning_enabled and sequence:
            self.cafve.train(sequence)

        if token_vectors:
            stacked = np.stack(token_vectors, axis=0)
            cafve_vector = stacked.mean(axis=0).astype(FLOATX)
        else:
            cafve_vector = np.zeros(self.predict.cafve_token_dim, dtype=FLOATX)

        pair_stats: List[Tuple[Tuple[int, int], float]] = []
        for entry in result.extras.get("pair_stats", []):
            try:
                pair, score = entry
                left, right = pair
                pair_stats.append(((int(left), int(right)), float(score)))
            except Exception:
                continue

        self._last_cafve_vector = cafve_vector
        self._last_cabpe_pairs = pair_stats
        return cafve_vector

    def _compose_predict_input(
        self,
        sensory_features: np.ndarray,
        unified_state: np.ndarray,
        emotional_state: np.ndarray,
        cafve_vector: Optional[np.ndarray] = None
    ) -> np.ndarray:
        cafve_vector_fitted = self._fit_dimension(
            cafve_vector if cafve_vector is not None else np.zeros(self.predict.cafve_token_dim, dtype=FLOATX),
            self.predict.cafve_token_dim,
        )
        # other_mind_stub = np.zeros(self.predict.other_mind_model_dim, dtype=FLOATX)  # COMMENTED OUT: stub removed
        focus_vector = self._fit_dimension(unified_state, self.predict.focus_vector_dim)
        emotional_vector = self._fit_dimension(emotional_state, self.predict.emotional_state_dim)

        parts = [
            self._fit_dimension(sensory_features, self.predict.sfe_feature_dim),
            cafve_vector_fitted,
            self._fit_dimension(unified_state, self.predict.unified_cognitive_state_dim),
            # other_mind_stub,  # COMMENTED OUT: stub removed
            focus_vector,
            emotional_vector
        ]
        combined = np.concatenate(parts).astype(FLOATX).reshape(1, -1)
        return combined

    def _train_predictor(
        self,
        sensory_features: np.ndarray,
        unified_state: np.ndarray,
        emotional_state: np.ndarray,
        cafve_vector: Optional[np.ndarray] = None
    ) -> float:
        predict_input = self._compose_predict_input(
            sensory_features,
            unified_state,
            emotional_state,
            cafve_vector=cafve_vector,
        )
        target = self._fit_dimension(unified_state, self.predict.predicted_output_dim).reshape(1, -1)
        loss = self.predict.predictor_network.train_step(predict_input, target)
        return float(loss if loss is not None else 0.0)

    def run_qa_training(
        self,
        dataset_root: Optional[Path] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> Dict[str, Any]:
        """Execute QA training schedule and embed trained SOM state."""
        root = Path(dataset_root) if dataset_root else QA_DATASET_DIR
        root = root.expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"QA dataset not found at {root}")

        self.persist_state()  # Ensure baseline snapshot

        batches = self._discover_qa_batches(root)
        if not batches:
            raise FileNotFoundError(f"No QA batches matching {QA_SEGMENT_GLOB} in {root}")

        batch_seconds = max(QA_SEGMENT_DURATION_SECONDS, 1)
        warm_batches = min(len(batches), max(1, int((2 * 60) / batch_seconds)))
        cooldown_batches = min(len(batches) - warm_batches, max(1, int((4 * 60) / batch_seconds)))
        active_batches = min(
            len(batches) - warm_batches - cooldown_batches,
            max(1, int((24 * 60) / batch_seconds))
        )

        schedule = (
            [(batches[i], False, False, "warm") for i in range(warm_batches)] +
            [(batches[i + warm_batches], True, True, "active") for i in range(active_batches)] +
            [
                (
                    batches[i + warm_batches + active_batches],
                    False,
                    True,
                    "cooldown"
                )
                for i in range(cooldown_batches)
            ]
        )

        stats_accumulator: List[Dict[str, Any]] = []

        def emit(update: Dict[str, Any]) -> None:
            if progress_callback is None:
                return
            try:
                progress_callback(update)
            except Exception as exc:  # pragma: no cover - diagnostics only
                logger.warning("Progress callback failed: %s", exc)

        emit({
            "event": "training_started",
            "dataset_root": str(root),
            "batches_total": len(schedule)
        })

        for index, (batch_path, enable_learning, persist_after, stage) in enumerate(schedule, start=1):
            features = self._load_qa_batch(batch_path)
            batch_stats = self._run_qa_batch(
                batch_path,
                features,
                learning_enabled=enable_learning,
                persist_after=persist_after
            )
            batch_stats.update({
                "stage": stage,
                "batch_index": index,
                "batches_total": len(schedule)
            })
            stats_accumulator.append(batch_stats)
            emit({"event": "training_batch", **batch_stats})

        embedded_state = self.som.export_state()
        self.persist_state()
        self._embed_trained_som(embedded_state)

        summary = self._summarize_training(stats_accumulator)
        summary.update({
            "event": "training_complete",
            "dataset_root": str(root),
            "batches_total": len(schedule),
            "timestamp": time.time()
        })

        self._write_training_stats(summary)
        emit(summary)
        try:
            self.data_collector.flush_buffer()
        except Exception as exc:  # pragma: no cover - IO failure path
            logger.warning("Failed to flush data collector after training: %s", exc)
        return summary

    def run_cfd_training(
        self,
        dataset_root: Optional[Path] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> Dict[str, Any]:
        logger.warning("run_cfd_training is deprecated. Use run_qa_training instead.")
        return self.run_qa_training(dataset_root=dataset_root, progress_callback=progress_callback)

    def _discover_qa_batches(self, dataset_root: Path) -> List[Path]:
        batches = sorted(dataset_root.glob(QA_SEGMENT_GLOB))
        if batches:
            return batches
        # Fallbacks for common formats
        for pattern in ("segment_*.npy", "segment_*.npz", "segment_*.jsonl"):
            fallback = sorted(dataset_root.glob(pattern))
            if fallback:
                return fallback
        return []

    def _load_qa_batch(self, batch_path: Path) -> np.ndarray:
        suffix = batch_path.suffix.lower()
        if suffix == ".npz":
            with np.load(batch_path, allow_pickle=True) as data:
                if "features" in data.files:
                    loaded = data["features"]
                else:
                    loaded = data[data.files[0]]
        elif suffix == ".npy":
            loaded = np.load(batch_path)
        else:
            raise ValueError(f"Unsupported QA batch format: {batch_path.suffix}")

        features = np.asarray(loaded, dtype=FLOATX)
        if features.ndim == 1:
            features = features.reshape(1, -1)
        if features.shape[1] != TOTAL_FEATURE_DIM:
            padded = np.zeros((features.shape[0], TOTAL_FEATURE_DIM), dtype=FLOATX)
            limit = min(features.shape[1], TOTAL_FEATURE_DIM)
            padded[:, :limit] = features[:, :limit]
            features = padded
        return features

    def _run_qa_batch(
        self,
        batch_path: Path,
        features: np.ndarray,
        learning_enabled: bool,
        persist_after: bool
    ) -> Dict[str, Any]:
        health_scores: List[float] = []
        predict_losses: List[float] = []
        completed_cycles = 0
        failed_cycles = 0

        for feature in features:
            snapshot = self.process_full_cycle(manual_features=feature, learning_override=learning_enabled)
            try:
                self.data_collector.capture_snapshot(snapshot, snapshot.get("timestamp", time.time()))
            except Exception as log_err:
                logger.warning("Data collection during training failed: %s", log_err)
            if snapshot.get("processing_complete"):
                completed_cycles += 1
                health_scores.append(snapshot.get("health_score", 0.0))
                losses = snapshot.get("training_losses") or {}
                if "predict_loss" in losses:
                    predict_losses.append(losses["predict_loss"])
            else:
                failed_cycles += 1

        if persist_after:
            self.persist_state()

        return {
            "batch_path": str(batch_path),
            "cycles": completed_cycles,
            "failed_cycles": failed_cycles,
            "health_mean": float(np.mean(health_scores)) if health_scores else 0.0,
            "predict_loss_mean": float(np.mean(predict_losses)) if predict_losses else 0.0,
            "learning_enabled": learning_enabled,
        }

    def _summarize_training(self, stats: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not stats:
            return {"health_mean": 0.0, "predict_loss_mean": 0.0, "cycles": 0}

        health_values = [record.get("health_mean", 0.0) for record in stats]
        predict_values = [record.get("predict_loss_mean", 0.0) for record in stats if record.get("learning_enabled")]
        total_cycles = sum(record.get("cycles", 0) for record in stats)

        return {
            "health_mean": float(np.mean(health_values)) if health_values else 0.0,
            "predict_loss_mean": float(np.mean(predict_values)) if predict_values else 0.0,
            "cycles": total_cycles
        }

    def _write_training_stats(self, summary: Dict[str, Any]) -> None:
        try:
            stats_path = QA_STATS_PATH
            stats_path.parent.mkdir(parents=True, exist_ok=True)
            with open(stats_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(summary) + "\n")
        except Exception as exc:  # pragma: no cover - IO failure path
            logger.warning("Unable to record QA stats: %s", exc)

    def _embed_trained_som(self, som_state: Dict[str, np.ndarray]) -> None:
        try:
            som_file = Path(__file__).resolve().parent / "som.py"
            content = som_file.read_text(encoding="utf-8")
            start_marker = "# >>> EMBEDDED_SOM_STATE"
            end_marker = "# <<< EMBEDDED_SOM_STATE"
            start_idx = content.find(start_marker)
            end_idx = content.find(end_marker)
            if start_idx == -1 or end_idx == -1:
                raise ValueError("Embedded state markers missing in som.py")

            payload = {
                "som_weights": np.round(som_state.get("som_weights"), 6).tolist(),
                "som_plasticity": np.round(som_state.get("som_plasticity"), 6).tolist(),
                "som_fatigue": np.round(som_state.get("som_fatigue"), 6).tolist(),
                "som_map_size": np.asarray(som_state.get("som_map_size"), dtype=np.int32).tolist(),
                "som_input_dim": np.asarray(som_state.get("som_input_dim"), dtype=np.int32).tolist(),
                "som_growth_stage": np.asarray(som_state.get("som_growth_stage", [0]), dtype=np.int32).tolist(),
                "som_cycles_since_growth": np.asarray(som_state.get("som_cycles_since_growth", [0]), dtype=np.int32).tolist()
            }

            replacement = (
                f"{start_marker}\n"
                f"EMBEDDED_SOM_STATE = {repr(payload)}\n"
                f"{end_marker}"
            )
            updated_content = content[:start_idx] + replacement + content[end_idx:]
            som_file.write_text(updated_content, encoding="utf-8")
        except Exception as exc:
            logger.warning("Failed to embed trained SOM state: %s", exc)

    def _record_som_training_sample(self, vector: np.ndarray) -> None:
        """Cache SOM input vectors for offline training batches."""
        try:
            arr = np.asarray(vector, dtype=FLOATX).reshape(-1)
            if arr.size != SOM_INPUT_DIM:
                padded = np.zeros(SOM_INPUT_DIM, dtype=FLOATX)
                limit = min(arr.size, SOM_INPUT_DIM)
                padded[:limit] = arr[:limit]
                arr = padded
            self._som_capture_cache.append(arr)
            if len(self._som_capture_cache) >= self._som_capture_flush_size:
                self._flush_som_training_cache()
        except Exception as exc:
            logger.warning("Unable to record SOM training vector: %s", exc)

    def _flush_som_training_cache(self) -> None:
        if not self._som_capture_cache:
            return
        try:
            batch = np.stack(self._som_capture_cache).astype(FLOATX)
            timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
            archive_dir = Path("updates")
            archive_dir.mkdir(parents=True, exist_ok=True)
            record = {
                "timestamp": timestamp,
                "cycle": int(self.cycle_count),
                "samples": batch.tolist()
            }
            som_log = archive_dir / "somlog.txt"
            with open(som_log, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            
        except Exception as exc:
            logger.error("Failed to flush SOM training cache: %s", exc)
        finally:
            self._som_capture_cache.clear()

    def shutdown(self) -> None:
        """Stop sensory and audio subsystems and halt further processing."""
        self.running = False
        try:
            self.audio_out.stop()
        except Exception as exc:
            logger.warning("Audio output shutdown error: %s", exc)

        try:
            self.sensory_extractor.stop()
        except Exception as exc:
            logger.warning("Sensory extractor shutdown error: %s", exc)

        try:
            self._flush_som_training_cache()
        except Exception:
            logger.warning("Failed to flush SOM capture cache during shutdown", exc_info=True)

        try:
            self.persist_state()
        except Exception:
            logger.warning("Mind state persistence failed during shutdown", exc_info=True)

        try:
            if self.data_collector is not None:
                self.data_collector.flush_buffer()
        except Exception:
            logger.warning("Data collection flush failed during shutdown", exc_info=True)

    def persist_state(self) -> None:
        """Persist module weights alongside the core identity snapshot."""
        self._save_module_states()

        archive_update: Dict[str, np.ndarray] = {
            "mind_unified_state": self._unified_cognitive_state.astype(np.float32)
        }

        try:
            # clear_global_state_keys(["mind_cycle_count"])  # self_image module missing
            pass
        except Exception:
            logger.debug("Unable to clear legacy mind_cycle_count key", exc_info=True)

        try:
            save_global_state(archive_update, cycle_count=self.cycle_count if hasattr(self, 'cycle_count') else 0)
        except Exception:
            logger.warning("Mind identity persistence failed", exc_info=True)
    
    def load_state(self, state_path: str | Path | None = None) -> bool:
        """Load the unified identity snapshot from the plain-text archive."""
        target = str(state_path) if state_path is not None else None

        try:
            # payload = load_global_state(target) if target else load_global_state()  # self_image module missing
            payload = None
        except Exception as exc:
            logger.error("Failed to read unified state archive: %s", exc)
            return False

        if not payload:
            return False

        restored = False

        if "mind_unified_state" in payload:
            try:
                unified_state = np.asarray(payload["mind_unified_state"], dtype=np.float32).reshape(-1)
                if unified_state.size == self._unified_cognitive_state.size:
                    self._unified_cognitive_state = unified_state.astype(FLOATX)
                    restored = True
                else:
                    logger.warning(
                        "Unified state size mismatch (stored=%d, current=%d)",
                        unified_state.size,
                        self._unified_cognitive_state.size,
                    )
            except Exception as exc:
                logger.warning("Failed to restore unified cognitive state: %s", exc)

        legacy_keys = [key for key in payload if key.startswith("som_")]
        if legacy_keys:
            logger.info("Ignoring legacy SOM keys in unified state archive")

        return restored

    def get_current_dimensions(self) -> Dict[str, int]:
        return {
            'base_dim': self.scaling.base_dim,
            'mid_dim': self.scaling.mid_dim,
            'high_dim': self.scaling.high_dim,
            'integrator_input_dim': self.total_input_dim
        }

    def _expand_integration_input(self, new_total_input_dim: int):
        """Expand the first layer of the integration network to accept larger input.
        Copies existing weights; new columns are Xavier-initialized. Adjusts total_input_dim.
        """
        if new_total_input_dim <= self.total_input_dim:
            return
        try:
            first_layer = self.integration_network.layers[0]
            if not isinstance(first_layer, Linear):
                logger.warning("First layer of integration_network not Linear; skipping expansion.")
                return
            old_W = first_layer.weights  # shape (in_dim, out_dim)
            old_in = old_W.shape[0]
            out_dim = old_W.shape[1]
            # Xavier init for new rows (additional input features)
            import math
            new_rows = new_total_input_dim - old_in
            limit = math.sqrt(6.0 / (new_total_input_dim + out_dim))
            extra_W = np.random.uniform(-limit, limit, (new_rows, out_dim)).astype(old_W.dtype)
            new_W = np.concatenate([old_W, extra_W], axis=0)
            first_layer.weights = new_W
            # Biases unchanged (shape (1, out_dim))
            self.total_input_dim = new_total_input_dim
            
        except Exception as e:
            logger.warning(f"Failed to expand integration input dimension: {e}")

    def get_unified_cognitive_state(self) -> np.ndarray:
        """Returns the current unified cognitive state vector."""
        return self._unified_cognitive_state.copy()

    # --- Backward Compatibility Adapter ---
    def process_cognitive_state(self, sensory_input: np.ndarray) -> np.ndarray:
        """Legacy method expected by main.py

        The newer architecture uses integrate_cognition with richer inputs. This
        adapter reconstructs minimal placeholders so existing call sites work.
        sensory_input: Raw sensory input (e.g., flattened image, audio features)
        Returns current unified cognitive state (updated).
        """
        try:
            # Validate input
            logger.debug(f"Received sensory_input: {sensory_input}")
            if sensory_input is None:
                logger.error("Mind.process_cognitive_state received None for sensory_input. Investigating upstream data flow.")
                return self._unified_cognitive_state.copy()

            # --- SOM Processing ---
            sensory_array = np.asarray(sensory_input, dtype=FLOATX).reshape(-1)
            som_activation = self.som.process_input(sensory_array)
            som_map = som_activation.reshape(self.som.map_size)

            # 2. Derive BMU coordinates
            bmu_index = int(np.argmax(som_map))
            bmu_coords = (bmu_index // self.som.map_size[1], bmu_index % self.som.map_size[1])
            logger.debug(f"SOM BMU coordinates: {bmu_coords}")

            # --- Cognitive Integration ---
            unified = self.integrate_cognition(som_map, bmu_coords)
            logger.debug(f"Unified cognitive state: {unified}")
            return unified
        except Exception as e:
            logger.exception(f"Exception in Mind.process_cognitive_state: {e}")
            return self._unified_cognitive_state.copy()

    def get_current_dimensions(self) -> Dict[str, int]:
        """Get current dimension configuration"""
        return {
            'base_dim': self.scaling.base_dim,
            'mid_dim': self.scaling.mid_dim,
            'high_dim': self.scaling.high_dim,
            'stage': self.scaling.current_stage
        }

    def attempt_growth(self) -> bool:
        """Attempt to grow cognitive dimensions based on current metrics"""
        # Calculate current metrics
        complexity = self._calculate_complexity_score()
        integration = self._calculate_integration_score()
        stability = self._calculate_stability_score()
        utilization = self._calculate_utilization_score()

        # Update scaling metrics
        self.scaling.update_metrics(complexity, integration, stability, utilization)

        # Attempt growth
        if self.scaling.grow():
            self._grow_cognitive_dimensions()
            return True
        return False

    def get_networks(self) -> List[Any]:
        """Returns a list of all internal neural networks for optimization."""
        networks = [self.nn, self.integration_network, self.metacognition.reflection_network]
        networks.extend(self.health.get_networks())
        return networks

    def _calculate_som_size(self) -> Tuple[int, int]:
        """Calculate SOM map size using prime numbers based on current base dimension"""
        # Use prime numbers for SOM map size (user requirement)
        primes = [17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]

        # Check if config explicitly sets SOM dimensions
        if hasattr(cfg, 'SOM_MAP_ROWS') and hasattr(cfg, 'SOM_MAP_COLS'):
            config_rows = getattr(cfg, 'SOM_MAP_ROWS', 53)
            config_cols = getattr(cfg, 'SOM_MAP_COLS', 53)
            if config_rows == config_cols:
                return (config_rows, config_cols)
        
        # Find appropriate prime based on current dimension stage
        stage = self.scaling.current_stage
        if stage < len(primes):
            size = primes[stage]
        else:
            # For stages beyond our prime list, use the largest prime
            size = primes[-1]

        return (size, size)

    def _calculate_complexity_score(self) -> float:
        """Calculate system complexity based on state variance and network activity"""
        if not hasattr(self, '_complexity_history'):
            self._complexity_history = []

        # Measure state complexity using variance and entropy
        state_variance = np.var(self._unified_cognitive_state)
        state_entropy = -np.sum(self._unified_cognitive_state * np.log(self._unified_cognitive_state + 1e-10))
        state_entropy = np.clip(state_entropy, 0, 100)  # Prevent extreme values

        # Network activity based on parameter magnitudes
        total_params = 0
        param_magnitude = 0
        for param, _, _ in self.integration_network.get_trainable_params():
            total_params += param.size
            param_magnitude += np.sum(np.abs(param))

        network_activity = param_magnitude / (total_params + 1e-6)

        complexity = (0.4 * state_variance + 0.3 * state_entropy + 0.3 * network_activity)
        self._complexity_history.append(complexity)

        if len(self._complexity_history) > 50:
            self._complexity_history.pop(0)

        return np.mean(self._complexity_history)

    def _calculate_stability_score(self) -> float:
        """Calculate system stability based on recent state changes"""
        if not hasattr(self, '_state_history'):
            self._state_history = []
            return 1.0

        self._state_history.append(self.cognitive_state.state_vector.copy())
        if len(self._state_history) > 100:
            self._state_history.pop(0)

        if len(self._state_history) < 2:
            return 1.0

        # Calculate average state change
        changes = [np.mean(np.abs(self._state_history[i] - self._state_history[i-1]))
                  for i in range(1, len(self._state_history))]

        # Convert to stability score (1 = very stable, 0 = very unstable)
        avg_change = np.mean(changes)
        stability = 1.0 / (1.0 + avg_change)
        return stability

    def _calculate_integration_score(self) -> float:
        """Calculate how well different cognitive components are integrated"""
        # Get current states
        cognitive = self.cognitive_state.state_vector
        emotional = self.cognitive_state.emotional_influence
        attention = self.cognitive_state.attention_mask

        # Calculate correlations between components
        corr_cog_emo = np.corrcoef(cognitive, emotional)[0,1]
        corr_cog_att = np.corrcoef(cognitive, attention)[0,1]
        corr_emo_att = np.corrcoef(emotional, attention)[0,1]

        # Average the absolute correlations
        integration = np.mean([np.abs(corr_cog_emo),
                             np.abs(corr_cog_att),
                             np.abs(corr_emo_att)])

        return float(integration)

    def _calculate_utilization_score(self) -> float:
        """Calculate how effectively the cognitive capacity is being used"""
        # Measure non-zero activation across the state vector
        utilization = np.mean(np.abs(self._unified_cognitive_state) > 0.01)
        return float(utilization)

    def _grow_cognitive_dimensions(self) -> None:
        """Handle growth of all cognitive dimensions"""
        old_base_dim = self.cognitive_state.dimension
        previous_mid_dim = (
            self.integration_network.layers[0].weights.shape[1]
            if hasattr(self.integration_network.layers[0], 'weights')
            else self.scaling.mid_dim
        )
        new_base_dim = self.scaling.base_dim
        new_mid_dim = self.scaling.mid_dim
        new_high_dim = self.scaling.high_dim

        if new_base_dim <= old_base_dim:
            
            return

        # Scale up cognitive state
        new_cognitive_state = CognitiveState(new_base_dim)
        new_cognitive_state.state_vector[:old_base_dim] = self.cognitive_state.state_vector
        new_cognitive_state.emotional_influence[:old_base_dim] = self.cognitive_state.emotional_influence
        new_cognitive_state.attention_mask[:old_base_dim] = self.cognitive_state.attention_mask
        new_cognitive_state.uncertainty[:old_base_dim] = self.cognitive_state.uncertainty
        self.cognitive_state = new_cognitive_state

        # Update unified cognitive state array
        new_unified_state = np.zeros(new_base_dim, dtype=FLOATX)
        new_unified_state[:old_base_dim] = self._unified_cognitive_state
        self._unified_cognitive_state = new_unified_state

        # Scale up emotion system
        self.emotion = EmotionCore(
            input_dim=EMOTION_INPUT_DIM,
            output_dim=self.emotional_state_dim
        )

        # Scale up memory system
        new_memory = MemorySystem(
            cognitive_state_dim=max(MEMORY_COGNITIVE_STATE_DIM, new_base_dim),
            emotional_state_dim=MEMORY_EMOTIONAL_STATE_DIM
        )
        new_memory.transfer_memories(self.memory)
        self.memory = new_memory

        # Update SOM
        new_som_size = self._calculate_som_size()
        self.som = SelfOrganizingMap(
            input_dim=SOM_INPUT_DIM,
            map_size=new_som_size
        )

        # Update integration network
        self.integration_network = Sequential(
            Linear(self.total_input_dim, new_mid_dim),
            Tanh(),
            Linear(new_mid_dim, new_base_dim),
            Sigmoid()
        )

        # Update predict module to new unified dimension
        self.predict = Predict(
            sfe_feature_dim=SOM_INPUT_DIM,
            cafve_token_dim=CAFVE_TOKEN_DIM,
            unified_cognitive_state_dim=new_base_dim,
            emotional_state_dim=self.emotional_state_dim,
            other_mind_model_dim=PREDICT_VECTOR_DIM,
            focus_vector_dim=PREDICT_VECTOR_DIM,
            predicted_output_dim=new_base_dim,
            prediction_error_dim=self.predictive_error_dim
        )

        # Update metacognition
        self.metacognition = MetaCognition(new_base_dim)
        self.unified_cognitive_state_dim = new_base_dim

        

    def _embed_trained_som(self):
        """
        Self-Modifying DNA: Writes the current SOM state directly into som.py.
        This ensures Lillith is 'born' with her learned instincts even if state files are lost.
        """
        try:
            import som as som_module
            som_path = som_module.__file__
            
            # Prepare the state dictionary
            state = {
                'som_weights': self.som.weights.tolist(),
                'som_plasticity': self.som.plasticity.tolist(),
                'som_fatigue': self.som.fatigue.tolist(),
                'som_map_size': self.som.map_size,
                'som_input_dim': self.som.input_dim
            }
            
            state_str = "EMBEDDED_SOM_STATE = " + repr(state)
            
            with open(som_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            start_marker = "# >>> EMBEDDED_SOM_STATE"
            end_marker = "# <<< EMBEDDED_SOM_STATE"
            
            start_idx = content.find(start_marker)
            end_idx = content.find(end_marker)
            
            if start_idx != -1 and end_idx != -1:
                new_content = (content[:start_idx + len(start_marker)] + 
                               "\n" + state_str + "\n" + 
                               content[end_idx:])
                
                with open(som_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                logger.info("Successfully embedded SOM state into som.py 'DNA'.")
            else:
                logger.warning("Could not find EMBEDDED_SOM_STATE markers in som.py.")
                
        except Exception as e:
            logger.error(f"Failed to embed SOM state: {e}")

    def save_state(self, save_path: str):
        """Saves the Mind's state and triggers SOM DNA embedding."""
        try:
            state = {
                'unified_cognitive_state': self._unified_cognitive_state.tolist(),
                'scaling_stage': self.scaling.current_stage,
                'total_input_dim': self.total_input_dim,
                'integration_network_params': self.integration_network.get_trainable_params()
            }
            with open(save_path, 'wb') as f:
                pickle.dump(state, f)
            
            # Trigger DNA embedding for SOM
            self._embed_trained_som()
            
            logger.info(f"Mind state saved to {save_path}")
        except Exception as e:
            logger.error(f"Error saving Mind state: {e}")

    # def load_state(self, load_path: str):
    # """Loads the Mind's state from a file."""
    # try:
    #     with open(load_path, 'rb') as f:
    #         state = pickle.load(f)
    #
    #     # Load scaling stage
    #     # if 'scaling_stage' in state:
    #         self.scaling.current_stage = state['scaling_stage']
    #         self.scaling.base_dim, self.scaling.mid_dim, self.scaling.high_dim = self.scaling.DIMENSION_STAGES[self.scaling.current_stage]
    #
    #     # Load unified cognitive state
    #     # if np.array(state['unified_cognitive_state']).shape == self._unified_cognitive_state.shape:
    #         self._unified_cognitive_state = np.array(state['unified_cognitive_state'], dtype=np.float32)
    #
    #         # Load cognitive state
    #         if 'cognitive_state' in state:
    #             cog_state = state['cognitive_state']
    #             self.cognitive_state.state_vector = np.array(cog_state['state_vector'], dtype=np.float32)
    #             self.cognitive_state.emotional_influence = np.array(cog_state['emotional_influence'], dtype=np.float32)
    #             self.cognitive_state.attention_mask = np.array(cog_state['attention_mask'], dtype=np.float32)
    #             self.cognitive_state.uncertainty = np.array(cog_state['uncertainty'], dtype=np.float32)
    #
    #         # Load neural network weights
    #         # loaded_params = state.get('integrator_network_weights', [])
    #         # current_params = self.integration_network.get_trainable_params()
    #
    #
    #         # Load metacognition weights
    #         # meta_params = state.get('metacognition_weights', [])
    #         # meta_current = self.metacognition.reflection_network.get_trainable_params()
    #
    #
    ##                logger.info(f"Mind state loaded from {load_path}")
    # #           else:
    #         logger.warning("Loaded Mind state dimensions mismatch. Initializing to default.")
    #         self._unified_cognitive_state = np.zeros(self.unified_cognitive_state_dim, dtype=np.float32)
    #
    # except FileNotFoundError:
    #     logger.warning(f"Mind state file not found at {load_path}. Initializing to default.")
    #     self._unified_cognitive_state = np.zeros(self.unified_cognitive_state_dim, dtype=np.float32)
    # except Exception as e:
    #     logger.error(f"Error loading Mind state: {e}. Initializing to default.")
    #     self._unified_cognitive_state = np.zeros(self.unified_cognitive_state_dim, dtype=np.float32)


 