# --- START OF FILE memory.py ---
# memory.py

import numpy as np
from config import FLOATX
import logging
from collections import deque
import time
from typing import List, Dict, Tuple, Any, Optional
import pickle
from pathlib import Path
from config import MEMORY_SFE_FEATURE_DIM, MEMORY_COGNITIVE_STATE_DIM, MEMORY_EMOTIONAL_STATE_DIM, MEMORY_SENSORY_BUFFER_SIZE, MEMORY_TIME_DECAY_HOURS, MEMORY_EPSILON, MEMORY_EMOTIONAL_WEIGHT, MEMORY_ACCESS_WEIGHT, MEMORY_CONSOLIDATION_DIM

from nn import mse_loss_prime

from nn import Sequential, Linear, Tanh, mse_loss_prime

class MemoryFragment:
    def __init__(self, content: np.ndarray, emotional_context: np.ndarray, timestamp: float):
        self.content = content
        self.emotional_context = emotional_context
        self.creation_time = timestamp
        self.last_access_time = timestamp
        self.access_count = 0
        self.emotional_intensity = np.mean(np.abs(emotional_context))
        self.connections: List[Tuple[int, float]] = []  # (fragment_id, connection_strength)
        
    def update_access(self, current_time: float):
        self.last_access_time = current_time
        self.access_count += 1
        
    def get_relevance(self, current_time: float, query_emotion: np.ndarray) -> float:
        time_factor = np.exp(-(current_time - self.last_access_time) / MEMORY_TIME_DECAY_HOURS)  # Decay over hours
        emotional_similarity = np.dot(self.emotional_context, query_emotion) / (np.linalg.norm(self.emotional_context) * np.linalg.norm(query_emotion) + MEMORY_EPSILON)
        access_factor = np.log1p(self.access_count)
        return time_factor * (MEMORY_EMOTIONAL_WEIGHT * emotional_similarity + MEMORY_ACCESS_WEIGHT * access_factor)

class MemoryConsolidation:
    def __init__(self, memory_dimension: int, internal_dim: int = MEMORY_CONSOLIDATION_DIM):
        self.dimension = memory_dimension
        self.consolidation_network = Sequential(
            Linear(memory_dimension, internal_dim),
            Tanh(),
            Linear(internal_dim, memory_dimension)
        )
        self.emotional_weighting = np.ones(memory_dimension, dtype=FLOATX)
        
    def update_dimensions(self, memory_dimension: int, internal_dim: int = MEMORY_CONSOLIDATION_DIM):
        """Dynamically update memory dimensions."""
        self.dimension = memory_dimension
        self.consolidation_network = Sequential(
            Linear(memory_dimension, internal_dim),
            Tanh(),
            Linear(internal_dim, memory_dimension)
        )
        self.emotional_weighting = np.ones(memory_dimension, dtype=FLOATX)
    def consolidate_memory(self, memory_fragment: MemoryFragment, 
                         emotional_context: np.ndarray,
                         related_fragments: List[MemoryFragment]) -> np.ndarray:
        """Consolidate memory by integrating emotional context and related memories"""
        # Combine memory content with emotional context
        consolidated = np.concatenate([
            memory_fragment.content,
            emotional_context,
            np.mean([f.content for f in related_fragments], axis=0)
        ])
        
        # Process through consolidation network
        consolidated = self.consolidation_network.forward(consolidated)
        
        # Apply emotional weighting
        consolidated *= self.emotional_weighting
        return consolidated
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SensoryBuffer:
    """Holds a high-fidelity, short-term log of raw SFE feature vectors."""
    def __init__(self, size: int = MEMORY_SENSORY_BUFFER_SIZE, sfe_feature_dim: int = MEMORY_SFE_FEATURE_DIM):
        self.size = size
        self.sfe_feature_dim = sfe_feature_dim
        self.buffer = deque(maxlen=size)
        logger.debug(f"SensoryBuffer initialized (size={size}, dim={sfe_feature_dim}).")

    def add(self, sfe_feature_vector: np.ndarray):
        if sfe_feature_vector is None or sfe_feature_vector.size == 0: return
        if sfe_feature_vector.shape[0] != self.sfe_feature_dim:
            logger.error(f"SFE vector dim mismatch. Expected {self.sfe_feature_dim}, got {sfe_feature_vector.shape[0]}.")
            return
        self.buffer.append(sfe_feature_vector.astype(FLOATX))
    def update_dimensions(self, sfe_feature_dim: int):
        """Dynamically update SFE feature dimension."""
        self.sfe_feature_dim = sfe_feature_dim

    def get_snapshot(self) -> List[np.ndarray]:
        return list(self.buffer)
    
    def clear(self):
        self.buffer.clear()

from config import WORKING_MEMORY_SIZE, WORKING_MEMORY_COGNITIVE_STATE_DIM

class WorkingMemory:
    """The mind's 'scratchpad.' Holds recent unified cognitive states."""
    def __init__(self, size: int = WORKING_MEMORY_SIZE, cognitive_state_dim: int = WORKING_MEMORY_COGNITIVE_STATE_DIM):
        self.size = size
        self.cognitive_state_dim = cognitive_state_dim
        self.canvas = np.zeros((size, cognitive_state_dim), dtype=FLOATX) # V4 uses float32
        logger.debug(f"WorkingMemory initialized (size={size}, dim={cognitive_state_dim}, precision=float32).")

    def update(self, cognitive_state_vector: np.ndarray):
        if cognitive_state_vector is None or cognitive_state_vector.size == 0: return
        if cognitive_state_vector.ndim == 1:
            cognitive_state_vector = cognitive_state_vector.reshape(1, -1)
        if cognitive_state_vector.shape[1] != self.cognitive_state_dim:
            logger.error(f"Cognitive state dim mismatch. Expected {self.cognitive_state_dim}, got {cognitive_state_vector.shape[1]}.")
            return
        self.canvas = np.roll(self.canvas, -1, axis=0)
        self.canvas[-1] = cognitive_state_vector[0, :self.cognitive_state_dim].astype(FLOATX)

    def get_state(self) -> np.ndarray:
        return self.canvas.flatten().astype(FLOATX)

    def clear(self):
        self.canvas = np.zeros((self.size, self.cognitive_state_dim), dtype=FLOATX)

class AffectiveLedger:
    """A permanent, chronological log of emotionally significant events."""
    def __init__(self, max_memories: int = 2000, emotional_state_dim: int = MEMORY_EMOTIONAL_STATE_DIM, cognitive_state_dim: int = 256): # V4 dims
        self.max_memories = max_memories
        self.emotional_state_dim = emotional_state_dim
        self.cognitive_state_dim = cognitive_state_dim
        self.memories = deque(maxlen=max_memories)
        self.emotional_threshold = FLOATX(0.6)
        logger.debug(f"AffectiveLedger initialized (max_memories={max_memories}).")

    def record_event(self, cognitive_state: np.ndarray, emotional_state: np.ndarray, timestamp: float):
        if cognitive_state is None or emotional_state is None: return
        if emotional_state.shape[0] != self.emotional_state_dim:
            logger.error(f"Emotional state dim mismatch. Expected {self.emotional_state_dim}, got {emotional_state.shape[0]}.")
            return
        emotional_charge = np.linalg.norm(emotional_state)
        if emotional_charge > self.emotional_threshold:
            memory_entry = {
                "cognitive_state": cognitive_state.copy().astype(FLOATX),
                "emotional_state": emotional_state.copy().astype(FLOATX),
                "charge": FLOATX(emotional_charge),
                "timestamp": FLOATX(timestamp)
            }
            self.memories.append(memory_entry)

    def store_emotional_memory(self, cognitive_state: np.ndarray, emotional_state: np.ndarray, reward_signal: float, timestamp: float):
        """Store an emotional memory with reward context."""
        self.record_event(cognitive_state, emotional_state, timestamp)

    def get_recent_memories(self, num: int = 10) -> List[Dict[str, Any]]:
        return list(self.memories)[-num:]
    
    def clear(self):
        self.memories.clear()

class AssociativeMemory:
    """Long-term memory network learning connections between cognitive states."""
    def __init__(self, cognitive_state_dim: int = 256, num_neurons: int = 512): # V4 cognitive state dim
        self.cognitive_state_dim = cognitive_state_dim
        self.num_neurons = num_neurons
        self.weights = np.random.randn(cognitive_state_dim, num_neurons).astype(FLOATX) * FLOATX(0.01)
        logger.debug(f"AssociativeMemory initialized (dim={cognitive_state_dim}, neurons={num_neurons}).")

    def learn(self, cognitive_state: np.ndarray, learning_rate: float):
        if cognitive_state is None or cognitive_state.size == 0:
            return
        input_pattern = cognitive_state.reshape(1, -1).astype(FLOATX)
        if input_pattern.shape[1] != self.cognitive_state_dim:
            logger.error(f"AssociativeMemory learn dim mismatch. Expected {self.cognitive_state_dim}, got {input_pattern.shape[1]}.")
            return
        activation = np.dot(input_pattern, self.weights)
        self.weights += learning_rate * np.dot(input_pattern.T, activation)

    def vectorized_learn(self, cognitive_states: np.ndarray, learning_rate: float):
        """
        Vectorized learning for batch processing of multiple cognitive states.
        Significantly faster than processing individual states sequentially.
        """
        if cognitive_states is None or cognitive_states.size == 0:
            return

        # Ensure proper shape and type
        if cognitive_states.ndim == 1:
            cognitive_states = cognitive_states.reshape(1, -1)

        if cognitive_states.shape[1] != self.cognitive_state_dim:
            logger.error(f"AssociativeMemory vectorized_learn dim mismatch. Expected {self.cognitive_state_dim}, got {cognitive_states.shape[1]}.")
            return

        cognitive_states = cognitive_states.astype(FLOATX)

        # Vectorized activation computation
        activations = np.dot(cognitive_states, self.weights)

        # Vectorized weight update
        # Using outer product sum for batch learning
        weight_updates = np.dot(cognitive_states.T, activations)
        self.weights += learning_rate * weight_updates

        logger.debug(f"Vectorized learning processed {len(cognitive_states)} cognitive states")


    def recall(self, cognitive_state: np.ndarray) -> np.ndarray:
        if cognitive_state is None or cognitive_state.size == 0:
            return np.zeros((1, self.num_neurons), dtype=FLOATX)
        query_pattern = cognitive_state.reshape(1, -1).astype(FLOATX)
        if query_pattern.shape[1] != self.cognitive_state_dim:
            logger.error(f"AssociativeMemory recall dim mismatch. Expected {self.cognitive_state_dim}, got {query_pattern.shape[1]}.")
            return np.zeros((1, self.num_neurons), dtype=FLOATX)
        return np.dot(query_pattern, self.weights).astype(FLOATX)
    
    def update(self, cognitive_state: np.ndarray, reward_signal: float):
        """Update associative memory with a cognitive state and reward signal."""
        learning_rate = 0.01 * (1.0 + reward_signal)  # Scale learning rate by reward
        self.learn(cognitive_state, learning_rate)
    
    def clear(self):
        # Training needed - weights should be learned, not random. See config.py MEMORY_TRAINING section
        self.weights = np.zeros((self.cognitive_state_dim, self.num_neurons), dtype=FLOATX)

class PredictiveEngine:
    """Predicts the next cognitive state based on the current working memory."""
    def __init__(self, input_state_dim: int, output_state_dim: int, hidden_dim: int = 128):
        self.input_state_dim = input_state_dim
        self.output_state_dim = output_state_dim
        self.predictor_network = Sequential(
            Linear(input_state_dim, hidden_dim),
            Tanh(),
            Linear(hidden_dim, output_state_dim)
        )
        logger.debug(f"PredictiveEngine initialized (input_dim={input_state_dim}, output_dim={output_state_dim}).")

    def predict(self, current_state: np.ndarray) -> np.ndarray:
        if current_state is None or current_state.size == 0:
            return np.zeros((1, self.output_state_dim), dtype=FLOATX)
        input_for_predictor = current_state.reshape(1, -1).astype(FLOATX)
        if input_for_predictor.shape[1] != self.input_state_dim:
            logger.error(f"PredictiveEngine predict dim mismatch. Expected {self.input_state_dim}, got {input_for_predictor.shape[1]}.")
            return np.zeros((1, self.output_state_dim), dtype=FLOATX)
        return self.predictor_network.forward(input_for_predictor).astype(FLOATX)

    def get_network(self) -> None: 
        return self.predictor_network

class MemorySystem:
    """The complete, 5-part memory architecture for Lillith."""
    def __init__(self, 
                 sfe_feature_dim: int = MEMORY_SFE_FEATURE_DIM,       # V4
                 cognitive_state_dim: int = MEMORY_COGNITIVE_STATE_DIM,  # V4
                 emotional_state_dim: int = MEMORY_EMOTIONAL_STATE_DIM   # V4
                 ):
        
        self.sfe_feature_dim = sfe_feature_dim
        self.cognitive_state_dim = cognitive_state_dim
        self.emotional_state_dim = emotional_state_dim

        self.sensory_buffer = SensoryBuffer(sfe_feature_dim=self.sfe_feature_dim)
        self.working_memory = WorkingMemory(cognitive_state_dim=self.cognitive_state_dim)
        self.affective_ledger = AffectiveLedger(emotional_state_dim=self.emotional_state_dim, cognitive_state_dim=self.cognitive_state_dim)
        self.associative_memory = AssociativeMemory(cognitive_state_dim=self.cognitive_state_dim)
        self.predictive_engine = PredictiveEngine(
            input_state_dim=self.working_memory.size * self.working_memory.cognitive_state_dim,
            output_state_dim=self.cognitive_state_dim # Predicts the next unified cognitive state
        )
        
        logger.info("MemorySystem (5-Part Holistic, V4) initialized.")

    def get_networks(self) -> List[None]:
        return [self.predictive_engine.get_network()]

    def recall(self, cognitive_state: np.ndarray) -> np.ndarray:
        """Retrieves relevant memories based on current cognitive state."""
        return self.associative_memory.recall(cognitive_state)

    def store_experience(self, cognitive_state: np.ndarray, sensory_snapshot: np.ndarray, 
                        emotion_state: np.ndarray, reward_signal: float, timestamp: float):
        """Store a new experience in the memory system."""
        # Add to sensory buffer
        self.sensory_buffer.add(sensory_snapshot)
        
        # Update working memory
        self.working_memory.update(cognitive_state)
        
        # Store in affective ledger
        self.affective_ledger.store_emotional_memory(cognitive_state, emotion_state, reward_signal, timestamp)
        
        # Update associative memory
        self.associative_memory.update(cognitive_state, reward_signal)

    # def save_state(self, save_dir: str):
    #     path = Path(save_dir)
    #     path.mkdir(parents=True, exist_ok=True)
    #     try:
    #         # Saving PredictiveEngine's network weights would require state_dict in nn.py
    #         np.savez_compressed(path / "associative_mem.npz", weights=self.associative_memory.weights)
    #         with open(path / "affective_ledger.pkl", 'wb') as f:
    #             pickle.dump(list(self.affective_ledger.memories), f)
    #         logger.info(f"MemorySystem state saved to {path}.")
    #     except Exception as e:
    #         logger.error(f"Error saving MemorySystem state: {e}")

    # def load_state(self, load_dir: str):
    #     path = Path(load_dir)
    #     try:
    #         assoc_mem_path = path / "associative_mem.npz"
    #         if assoc_mem_path.exists():
    #             with np.load(assoc_mem_path, allow_pickle=True) as data:
    #                 if 'weights' in data and data['weights'].shape == self.associative_memory.weights.shape:
    #                     self.associative_memory.weights = data['weights'].astype(np.float32)
    #         ledger_path = path / "affective_ledger.pkl"
    #         if ledger_path.exists():
    #             with open(ledger_path, 'rb') as f:
    #                 loaded_memories = pickle.load(f)
    #                 self.affective_ledger.memories = deque(loaded_memories, maxlen=self.affective_ledger.max_memories)
    #         logger.info(f"MemorySystem state loaded from {path}.")
    #     except Exception as e:
    #         logger.error(f"Error loading MemorySystem state: {e}. Re-initializing components.")
    #         self.sensory_buffer.clear()
    #         self.working_memory.clear()
    #         self.affective_ledger.clear()
    #         self.associative_memory.clear()


